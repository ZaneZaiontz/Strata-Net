"""Tests for tools/stage_join.py: the invite string and the join listener, on 127.0.0.1 only.  The keys are made with
the openssl command, as tools/test_stage_node.py makes them (the tests that need them are skipped without it).

    python -m unittest tools.test_stage_join
"""
from __future__ import annotations

import base64
import contextlib
import http.client
import io
import json
import os
import re
import shutil
import socket
import ssl
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
import stage_join as sj  # noqa: E402
import stage_node as sn  # noqa: E402
from test_stage_node import OPENSSL, make_keys  # noqa: E402

FP, TOKEN = "ab" * 32, "t" * 43


class InviteStringTest(unittest.TestCase):
    def test_round_trip(self):
        s = sj.make_invite("admin.lan:7839", FP, TOKEN)
        self.assertNotIn(" ", s)
        self.assertEqual(sj.parse_invite(f" {s}\n"), {"addr": "admin.lan:7839", "fp": FP, "token": TOKEN})

    def test_not_invites(self):
        v2 = base64.urlsafe_b64encode(json.dumps({"v": 2, "addr": "a:1", "fp": FP, "token": TOKEN}).encode()).decode()
        for bad in ("", sj.INVITE_PREFIX, "other:" + sj.make_invite("a:1", FP, TOKEN)[len(sj.INVITE_PREFIX):],
                    sj.INVITE_PREFIX + "!!!", sj.INVITE_PREFIX + v2, sj.make_invite("a b:1", FP, TOKEN),
                    sj.make_invite("a:1", FP.upper(), TOKEN), sj.make_invite("a:1", FP, "short")):
            with self.assertRaises(ValueError, msg=bad):
                sj.parse_invite(bad)

    def test_an_ipv6_invite_is_refused(self):
        with tempfile.TemporaryDirectory() as t:
            with self.assertRaisesRegex(ValueError, "IPv4 only"):
                sj.invite_command("[::1]:7839", Path(t) / "node" / "node.json")
            self.assertFalse((Path(t) / "node").exists())

    def test_addresses(self):
        self.assertEqual(sj.split_address("pc.lan"), ("pc.lan", sj.JOIN_PORT))
        self.assertEqual(sj.split_address("[::1]:8"), ("::1", 8))
        for bad in ("a:0", "a:99999", "a b", "a:1:2", "a/b:1", ""):
            with self.assertRaises(ValueError, msg=bad):
                sj.split_address(bad)


class ListenerCase(unittest.TestCase):
    """An admin's key folder of its own, an invite and its listener on 127.0.0.1."""

    @classmethod
    def setUpClass(cls):
        cls.keydir = tempfile.TemporaryDirectory()
        k = Path(cls.keydir.name)
        cls.admin_keys, cls.member = make_keys(k / "admin", "admin"), make_keys(k / "member", "member")
        cls.member_pem = (cls.member / "cert.pem").read_text()

    @classmethod
    def tearDownClass(cls):
        cls.keydir.cleanup()

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.admin = Path(shutil.copytree(self.admin_keys, Path(self.tmp.name) / "admin"))
        self.fp = sj.fingerprint((self.admin / "cert.pem").read_text())
        self.inv = sj.Invite()
        self.err = io.StringIO()
        quiet = mock.patch.object(sys, "stderr", self.err)
        quiet.start()
        self.addCleanup(quiet.stop)
        srv = sj.JoinServer(("127.0.0.1", 0), sj.make_join_handler(self.admin, self.inv), sj.server_context(self.admin))
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        self.addCleanup(srv.server_close)
        self.addCleanup(srv.shutdown)
        self.port = srv.server_address[1]

    def post(self, token: str, body=None, fp: str | None = None) -> tuple[int, dict]:
        body = body if body is not None else {"cert": self.member_pem, "name": "member", "agent": "127.0.0.1:7840"}
        c = http.client.HTTPSConnection("127.0.0.1", self.port, timeout=10, context=sj.client_context(fp or self.fp))
        try:
            c.request("POST", "/join", body if isinstance(body, bytes) else json.dumps(body),
                      {"Authorization": f"Bearer {token}"})
            r = c.getresponse()
            return r.status, json.loads(r.read())
        finally:
            c.close()

    def peers(self, folder: Path | None = None) -> list[str]:
        return sorted(p.name for p in (folder or self.admin / "peers").iterdir())


@unittest.skipIf(OPENSSL is None, "no openssl command to make the keys")
class ListenerTest(ListenerCase):
    """The join listener: one join per invite, refusals all alike, and nothing from the request in a path."""

    def test_joins_once(self):
        with contextlib.redirect_stdout(io.StringIO()) as out:
            self.assertEqual(self.post(self.inv.token), (200, {"ok": True}))
        name = f"member-{sj.fingerprint(self.member_pem)[:16]}.pem"
        self.assertEqual(self.peers(), [name])
        self.assertEqual((self.admin / "peers" / name).read_text(), self.member_pem)
        self.assertIn("member-", out.getvalue())
        self.assertTrue(self.inv.done.is_set())
        self.assertEqual(self.post(self.inv.token), (403, {"ok": False, "error": "refused"}))

    def test_wrong_or_expired_refused_alike(self):
        answers = [self.post("x" * 43), self.post("x" * 43, b"x" * (sj.MAX_BODY + 1))]
        self.inv.deadline = time.time() - 1
        answers.append(self.post(self.inv.token))
        self.assertEqual(answers, [(403, {"ok": False, "error": "refused"})] * 3)
        self.assertEqual(self.err.getvalue().count(" refused: "), 1)
        self.assertEqual(self.peers(), [])

    def test_a_bad_body_does_not_use_the_invite(self):
        for body in (b"x" * (sj.MAX_BODY + 1), b"not json", {"cert": "no certificate", "name": "m", "agent": "a:1"},
                     {"cert": self.member_pem * 2, "name": "m", "agent": "a:1"},
                     {"cert": self.member_pem, "name": "m", "agent": "a b"}):
            self.assertEqual(self.post(self.inv.token, body), (403, {"ok": False, "error": "refused"}))
        self.assertEqual(self.peers(), [])
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(self.post(self.inv.token)[0], 200)

    def test_a_name_never_shapes_a_path(self):
        with contextlib.redirect_stdout(io.StringIO()):
            self.post(self.inv.token, {"cert": "junk\n" + self.member_pem + "more", "name": "../..\\CON.x/" + "n" * 99,
                                       "agent": "127.0.0.1:7840"})
        [name] = self.peers()
        self.assertRegex(name, r"^CONxn{28}-[0-9a-f]{16}\.pem$")
        self.assertEqual((self.admin / "peers" / name).read_text(), self.member_pem)

    def test_connections_beyond_the_cap_are_closed(self):
        with mock.patch.object(sj, "MAX_CONNECTIONS", 2):
            srv = sj.JoinServer(("127.0.0.1", 0), sj.make_join_handler(self.admin, self.inv),
                                sj.server_context(self.admin))
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        self.addCleanup(srv.server_close)
        self.addCleanup(srv.shutdown)
        self.port = srv.server_address[1]
        idle = [socket.create_connection(("127.0.0.1", self.port)) for _ in range(2)]
        time.sleep(0.5)
        with socket.create_connection(("127.0.0.1", self.port), timeout=5) as third:
            with contextlib.suppress(ConnectionResetError):
                self.assertEqual(third.recv(1), b"")
        for s in idle:
            s.close()
        time.sleep(0.5)
        self.assertEqual(self.post("x" * 43)[0], 403)

    def test_a_wrong_fingerprint_is_refused(self):
        with self.assertRaisesRegex(ssl.SSLError, "is not this PC's"):
            self.post(self.inv.token, fp="0" * 64)
        self.assertFalse(self.inv.done.is_set())


@unittest.skipIf(OPENSSL is None, "no openssl command to make the keys")
class JoinCommandTest(ListenerCase):
    """`join` against the listener: what each side has after, and nothing written when it fails."""

    def setUp(self):
        super().setUp()
        self.node_json = Path(self.tmp.name) / "node" / "node.json"
        self.keys = Path(shutil.copytree(self.member, self.node_json.parent / "stage-tls"))
        env = mock.patch.dict("os.environ", {"STRATA_STAGE_TLS": ""})
        env.start()
        self.addCleanup(env.stop)

    def join(self, fp: str | None = None) -> str:
        with contextlib.redirect_stdout(io.StringIO()) as out:
            sj.join_command(sj.make_invite(f"127.0.0.1:{self.port}", fp or self.fp, self.inv.token), self.node_json)
        return out.getvalue()

    def test_joins_and_the_agent_takes_the_admin_without_a_restart(self):
        cfg = {"_tls": self.keys, "bind": "127.0.0.1", "_stamp": sn.trust_stamp(self.keys),
               "_server_ctx": sn.tls_context(self.keys, True), "_client_ctx": sn.tls_context(self.keys, False)}
        agent = sn.TlsServer(("127.0.0.1", 0), sn.make_handler(cfg, sn.Workers(cfg), sn.Store(cfg)),
                             cfg["_server_ctx"], cfg)
        threading.Thread(target=agent.serve_forever, daemon=True).start()
        self.addCleanup(agent.server_close)
        self.addCleanup(agent.shutdown)
        url = f"https://127.0.0.1:{agent.server_address[1]}/info"
        with self.assertRaises(OSError):
            sn.https_opener(sn.tls_context(self.admin, False)).open(url, timeout=10)
        self.addCleanup(os.chdir, os.getcwd())
        os.chdir(self.tmp.name)
        self.node_json = Path("node", "node.json")
        self.assertIn("made overseers/", self.join())
        admin_pem = (self.admin / "cert.pem").read_text()
        for folder in ("peers", "overseers"):
            [name] = self.peers(self.keys / folder)
            self.assertEqual((self.keys / folder / name).read_text(), admin_pem)
        [name] = self.peers()
        self.assertEqual((self.admin / "peers" / name).read_text(), self.member_pem)
        node = json.loads(self.node_json.read_text())
        self.assertEqual((node["tls_dir"], node["bind"], node["agent_port"]), (str(self.keys), "127.0.0.1", 7840))
        self.assertTrue(all(Path(node[k]).is_absolute() for k in ("exe", "log_dir", "tls_dir", "data_dir")))
        with sn.https_opener(sn.tls_context(self.admin, False)).open(url, timeout=10) as r:
            self.assertEqual(r.status, 200)

    def test_a_wrong_fingerprint_changes_nothing(self):
        with self.assertRaisesRegex(SystemExit, "not the PC this invite is from"):
            self.join(fp="0" * 64)
        self.assertEqual(self.peers(self.keys / "peers"), [])
        self.assertFalse((self.keys / "overseers").exists() or self.node_json.exists())
        self.assertFalse(self.inv.done.is_set())

    def test_a_pc_with_an_overseer_does_not_join_another_pool(self):
        (self.keys / "overseers").mkdir()
        shutil.copyfile(self.admin_keys / "cert.pem", self.keys / "overseers" / "other.pem")
        with self.assertRaisesRegex(SystemExit, "already takes orders from 1 PC"):
            self.join()
        self.assertEqual(self.peers(self.keys / "peers"), [])
        self.assertFalse(self.inv.done.is_set())

    def test_a_lost_answer_says_so(self):
        with mock.patch.object(http.client.HTTPSConnection, "getresponse", side_effect=ConnectionResetError("reset")), \
                self.assertRaisesRegex(SystemExit, "answer was lost"):
            self.join()
        self.assertTrue(self.inv.done.wait(5))

    def test_a_used_invite_changes_nothing(self):
        self.inv.done.set()
        with self.assertRaisesRegex(SystemExit, "refused"):
            self.join()
        self.assertEqual(self.peers(self.keys / "peers"), [])
        self.assertFalse(self.node_json.exists())

    def test_an_existing_key_and_node_json_are_kept(self):
        cert = (self.keys / "cert.pem").read_bytes()
        self.node_json.write_text(json.dumps({"exe": "none", "tls_dir": str(self.keys), "agent_port": 17840}))
        before = self.node_json.read_text()
        self.join()
        self.assertEqual(((self.keys / "cert.pem").read_bytes(), self.node_json.read_text()), (cert, before))


@unittest.skipIf(OPENSSL is None, "no openssl command to make the keys")
class InviteCommandTest(unittest.TestCase):
    def test_prints_an_invite_and_makes_overseers_once(self):
        with tempfile.TemporaryDirectory() as t:
            keys = make_keys(Path(t) / "stage-tls", "admin")
            with socket.socket() as s:
                s.bind(("127.0.0.1", 0))
                port = s.getsockname()[1]
            outs, real = [], sj.Invite
            for _ in range(2):
                with mock.patch.object(sj, "Invite", lambda: real(hours=0)), \
                        mock.patch.dict("os.environ", {"STRATA_STAGE_TLS": ""}), \
                        contextlib.redirect_stdout(io.StringIO()) as out:
                    sj.invite_command(f"127.0.0.1:{port}", Path(t) / "node.json")
                outs.append(out.getvalue())
            self.assertIn("made overseers/", outs[0])
            self.assertNotIn("made overseers/", outs[1])
            self.assertTrue((keys / "overseers").is_dir())
            line = next(x for x in outs[0].splitlines() if x.startswith(sj.INVITE_PREFIX))
            got = sj.parse_invite(line)
            self.assertEqual((got["addr"], got["fp"]), (f"127.0.0.1:{port}",
                                                        sj.fingerprint((keys / "cert.pem").read_text())))
            self.assertIn("expired", outs[0])
            self.assertTrue(re.search(r"waiting on port \d+", outs[0]))


if __name__ == "__main__":
    unittest.main()
