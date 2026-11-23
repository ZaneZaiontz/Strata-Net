"""Join a PC to a pool with one invite, without copying keys by hand (docs/REMOTE_STAGE.md, "Join with an invite").

    python tools/stage_join.py invite [--address HOST[:PORT]] [NODE_JSON]    the admin's PC: prints an invite, waits

NODE_JSON (default ~/strata-node/node.json) names the key folder ("tls_dir"; STRATA_STAGE_TLS wins) and the engine
("exe") that makes the key when there is none; without the file, the folder is stage-tls/ next to it and the engine
this install's engine/.

`invite` makes this PC's key if it has none, and overseers/ if it has none (from then on only this PC drives its
agent), then prints an invite and serves one join on PORT (default 7839, every IPv4 interface, at most
MAX_CONNECTIONS at once) until a PC joins, the invite expires (INVITE_HOURS) or Ctrl+C.  HOST:PORT (default: this
PC's address on the default route) is what goes in the invite: the address the joining PC reaches, a forwarded port
or a tailnet address too.

The join is HTTPS, TLS 1.3, with this PC's key.  The joining PC checks this PC's certificate against the invite's
fingerprint and trusts nothing else; this PC asks for no client certificate.  A request shows the invite's token as
"Authorization: Bearer TOKEN" and sends {"cert": PEM, "name": NAME, "agent": "host:port"}.  The token is checked
first (in constant time), the body only after; every refusal gets the same answer.  The certificate goes into
peers/ under a name made of NAME's plain characters and its fingerprint, nothing else from the request.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import hmac
import json
import os
import re
import secrets
import socket
import ssl
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import stage_node  # noqa: E402

INVITE_PREFIX = "strata-net:"
INVITE_HOURS = 1
JOIN_PORT = 7839
MAX_BODY = 16 << 10
MAX_CONNECTIONS = 16
DEFAULT_ROUTE_PROBE = ("192.0.2.1", 9)
ADDRESS = re.compile(r"(\[[0-9A-Fa-f:.]{2,45}\]|[A-Za-z0-9.\-]{1,253})(?::(\d{1,5}))?")
FINGERPRINT = re.compile(r"[0-9a-f]{64}")


def split_address(addr: str, port: int = JOIN_PORT) -> tuple[str, int]:
    """HOST[:PORT] or [V6][:PORT] -> (host, port); ValueError for anything else."""
    m = ADDRESS.fullmatch(addr)
    if not m or not 0 < int(m[2] or port) < 65536:
        raise ValueError(f"not a HOST:PORT address: {addr!r}")
    return m[1].strip("[]"), int(m[2] or port)


def make_invite(addr: str, fp: str, token: str) -> str:
    """The invite as one string."""
    raw = json.dumps({"v": 1, "addr": addr, "fp": fp, "token": token}, separators=(",", ":")).encode()
    return INVITE_PREFIX + base64.urlsafe_b64encode(raw).decode().rstrip("=")


def parse_invite(text: str) -> dict:
    """An invite string -> {"addr", "fp", "token"}; ValueError when it is not one."""
    text = text.strip()
    if not text.startswith(INVITE_PREFIX):
        raise ValueError(f"an invite starts with {INVITE_PREFIX}")
    try:
        b = text[len(INVITE_PREFIX):]
        d = json.loads(base64.urlsafe_b64decode(b + "=" * (-len(b) % 4)))
        split_address(d["addr"])
        if d["v"] != 1 or not FINGERPRINT.fullmatch(d["fp"]) or not isinstance(d["token"], str) \
                or not 20 <= len(d["token"]) <= 100:
            raise ValueError("a field is wrong")
    except (ValueError, KeyError, TypeError) as e:
        raise ValueError(f"not a valid invite ({e})") from None
    return {k: d[k] for k in ("addr", "fp", "token")}


def default_exe() -> Path:
    """This install's engine (engine/ next to terminal/, as setup.py lays it out)."""
    return Path(__file__).resolve().parents[2] / "engine" / ("strata.exe" if os.name == "nt" else "strata")


def node_paths(node_json: Path) -> tuple[dict | None, Path, Path]:
    """NODE_JSON's contents (None without the file), its key folder and its engine."""
    cfg = json.loads(node_json.read_text(encoding="utf-8")) if node_json.exists() else None
    d = os.environ.get("STRATA_STAGE_TLS") or (cfg or {}).get("tls_dir") or node_json.parent / "stage-tls"
    return cfg, Path(d).resolve(), Path((cfg or {}).get("exe") or default_exe())


def ensure_key(d: Path, exe: Path) -> None:
    """This PC's key in d, made with the engine's `--stage-keygen` when it has none (Python cannot make one)."""
    if (d / "cert.pem").exists() and (d / "key.pem").exists():
        return
    if not exe.exists():
        sys.exit(f"stage_join: no key in {d}, and no engine at {exe} to make one")
    d.parent.mkdir(parents=True, exist_ok=True)
    var = "PATH" if os.name == "nt" else "LD_LIBRARY_PATH"
    env = dict(os.environ, **{var: os.pathsep.join(filter(None, [str(exe.parent), os.environ.get(var)]))})
    r = subprocess.run([str(exe), "--stage-keygen", str(d)], capture_output=True, text=True, env=env)
    if r.returncode != 0:
        sys.exit(f"stage_join: {exe} --stage-keygen {d}: {(r.stderr or r.stdout).strip()}")
    print(f"stage_join: made this PC's key in {d}")


def fingerprint(pem: str) -> str:
    return hashlib.sha256(ssl.PEM_cert_to_DER_cert(pem)).hexdigest()


def tls_base(server: bool) -> ssl.SSLContext:
    """TLS 1.3 only, at the engine's security level, as stage_node.tls_context, with no peer certificate checked."""
    ctx = (ssl.SSLContext if server else stage_node.PinnedContext)(
        ssl.PROTOCOL_TLS_SERVER if server else ssl.PROTOCOL_TLS_CLIENT)
    ctx.minimum_version = ssl.TLSVersion.TLSv1_3
    ctx.set_ciphers("DEFAULT:@SECLEVEL=2")
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx


def server_context(d: Path) -> ssl.SSLContext:
    """The join listener's: this PC's key, no client certificate asked for, no resumption."""
    ctx = tls_base(True)
    ctx.load_cert_chain(d / "cert.pem", d / "key.pem")
    ctx.num_tickets = 0
    return ctx


def client_context(fp: str) -> ssl.SSLContext:
    """A joining PC's: the listener's certificate must have the invite's fingerprint, and nothing else is trusted."""
    ctx = tls_base(False)
    ctx.pinned = {fp}
    return ctx


class Invite:
    """One invite: its token (kept as a SHA-256), its deadline, and whether a PC joined with it."""

    def __init__(self, hours: float = INVITE_HOURS):
        self.token = secrets.token_urlsafe(32)
        self.digest = hashlib.sha256(self.token.encode()).digest()
        self.deadline = time.time() + hours * 3600
        self.lock = threading.Lock()
        self.done = threading.Event()

    def admits(self, header: str) -> bool:
        """An Authorization header with this invite's token, before it is used or expired."""
        given = hashlib.sha256(header.removeprefix("Bearer ").encode()).digest()
        return hmac.compare_digest(given, self.digest) and not self.done.is_set() and time.time() < self.deadline


def write_peer(d: Path, pem: str, name: str) -> Path:
    """A joined PC's certificate in d/peers/, as NAME's plain characters (at most 32) and its fingerprint."""
    safe = re.sub(r"[^A-Za-z0-9_-]", "", name)[:32] or "pc"
    p = d / "peers" / f"{safe}-{fingerprint(pem)[:16]}.pem"
    p.parent.mkdir(exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(pem, encoding="ascii")
    os.replace(tmp, p)
    return p


class JoinServer(stage_node.TlsServer):
    """The join listener: at most MAX_CONNECTIONS at once, one more is closed as it comes, so a flood blocks the
    invite but does not slow the PC."""

    def __init__(self, addr, handler, ctx: ssl.SSLContext):
        self.slots = threading.BoundedSemaphore(MAX_CONNECTIONS)
        super().__init__(addr, handler, ctx)

    def get_request(self):
        sock, addr = super().get_request()
        if not self.slots.acquire(blocking=False):
            sock.close()
            raise OSError("too many connections")
        return sock, addr

    def shutdown_request(self, request):
        super().shutdown_request(request)
        self.slots.release()


def make_join_handler(d: Path, invite: Invite):
    refused: set[str] = set()

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.0"
        timeout = stage_node.HANDSHAKE_TIMEOUT_S

        def handle(self):
            try:
                self.connection.settimeout(self.timeout)
                self.connection.do_handshake()
            except (ssl.SSLError, OSError):
                return
            super().handle()

        def log_message(self, fmt, *args):
            pass

        def answer(self, code: int, obj: dict) -> None:
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            with invite.lock:
                why = self.admit()
            if why:
                if self.client_address[0] not in refused:
                    refused.add(self.client_address[0])
                    sys.stderr.write(f"stage_join: {self.client_address[0]} refused: {why} (its next refusals are "
                                     "not shown)\n")
                self.drain()
                self.answer(403, {"ok": False, "error": "refused"})

        do_GET = do_POST
        unread = True

        def drain(self) -> None:
            """Reads up to 4 * MAX_BODY of a refused request's body when it was not read: closed with bytes unread,
            the socket would be reset under the sender (on Windows) and the answer lost."""
            n = self.headers.get("Content-Length", "")
            if self.unread and n.isdigit():
                try:
                    self.rfile.read(min(int(n), 4 * MAX_BODY))
                except OSError:
                    pass

        def admit(self) -> str | None:
            """Adds the request's certificate to peers/ and answers; else why it was refused."""
            if self.path != "/join" or not invite.admits(self.headers.get("Authorization", "")):
                return "no valid invite (wrong, used or expired)"
            n = self.headers.get("Content-Length", "")
            if not n.isdigit() or not 0 < int(n) <= MAX_BODY:
                return "no body, or one too large"
            try:
                self.unread = False
                req = json.loads(self.rfile.read(int(n)))
                certs, name, agent = stage_node.PEM_CERT.findall(req["cert"]), str(req["name"]), str(req["agent"])
                if len(certs) != 1:
                    raise ValueError("not one certificate")
                pem = certs[0] + "\n"
                ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT).load_verify_locations(cadata=pem)
                split_address(agent)
            except (ValueError, KeyError, TypeError, OSError) as e:
                return f"a bad request ({e})"
            p = write_peer(d, pem, name)
            invite.done.set()
            self.answer(200, {"ok": True})
            print(f"stage_join: {p.stem} joined (SHA256:{fingerprint(pem)[:16]}, its certificate is in peers/); "
                  f"its agent: https://{agent}", flush=True)
            return None

    return Handler


def default_address() -> str:
    """This PC's address on the default route (no packet is sent)."""
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        try:
            s.connect(DEFAULT_ROUTE_PROBE)
            return s.getsockname()[0]
        except OSError:
            return "127.0.0.1"


def invite_command(address: str | None, node_json: Path) -> None:
    host, port = split_address(address or default_address())
    if ":" in host:
        raise ValueError(f"{address}: the join listener is IPv4 only, give an IPv4 address or a host name")
    _, d, exe = node_paths(node_json)
    ensure_key(d, exe)
    if not (d / "overseers").is_dir():
        (d / "overseers").mkdir()
        print("stage_join: made overseers/: this PC's agent now takes orders from this PC only")
    addr = f"{host}:{port}"
    inv = Invite()
    srv = JoinServer(("", port), make_join_handler(d, inv), server_context(d))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    print(make_invite(addr, fingerprint((d / "cert.pem").read_text(encoding="ascii")), inv.token))
    print(f"stage_join: waiting on port {port} for one PC to join, until "
          f"{time.strftime('%Y-%m-%d %H:%M', time.localtime(inv.deadline))} (Ctrl+C cancels the invite)", flush=True)
    try:
        while not inv.done.wait(1) and time.time() < inv.deadline:
            pass
        if not inv.done.is_set():
            print("stage_join: the invite expired")
    except KeyboardInterrupt:
        print("stage_join: the invite is cancelled")
    finally:
        srv.shutdown()
        srv.server_close()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    inv = sub.add_parser("invite", help="print an invite and wait for one PC to join")
    inv.add_argument("--address", help="HOST[:PORT] the joining PC reaches (default: this PC's LAN address)")
    inv.add_argument("node_json", nargs="?", type=Path, default=Path.home() / "strata-node" / "node.json")
    a = ap.parse_args()
    try:
        invite_command(a.address, a.node_json)
    except ValueError as e:
        sys.exit(f"stage_join: {e}")


if __name__ == "__main__":
    main()
