"""Download a model's GGUF files from Hugging Face into a node's data_dir, each checked against the SHA-256 that
Hugging Face's API gives for it.

    python tools/stage_fetch.py get REPO@COMMIT FILE... --data-dir DIR [--jsonl]
    python tools/stage_fetch.py start --agent URL [--agent URL ...] [--tls-dir DIR] REPO[@COMMIT] FILE...
    python tools/stage_fetch.py status --agent URL [--agent URL ...] [--tls-dir DIR] [--watch]
    python tools/stage_fetch.py cancel --agent URL [--agent URL ...] [--tls-dir DIR]

REPO is owner/name and COMMIT a full commit hash, never a branch, so every PC gets the same bytes.  FILE is a .gguf
path in the repository.  The files land in DIR/models/<owner>--<name>/ under their base names, as setup keeps them
(the pack's fingerprint holds the shard names).

The sizes and SHA-256 come from huggingface.co's API at that commit (tools/hf_pin_fetch.py), not from the caller or
a mirror.  The bytes come from HF_ENDPOINT or huggingface.co through setup's download() (resumable, several
connections, a damaged piece fetched again) and are checked with its verify_sha256().  So the check protects the
transfer: the files are the ones the repository holds at that commit.  It says nothing about what they are: the
repository's owner chose both the bytes and the hash, so whoever picks the repository picks the model's content.

A file whose mark names its SHA-256 is skipped without the network; one this tool checked for another commit (its
mark names another SHA-256) is deleted first.  The names are checked before anything reaches the network, and
nothing else from the caller goes into a URL or a path.  Less than FREE_MARGIN left on the disk after the download
is refused before anything is written.  A second `get` into the same folder while one runs (one an agent left
running when it was killed) stops at once.

`start`, `status` and `cancel` run on the admin's PC and drive the agents' /fetch (tools/stage_node.py, which runs
`get`) with this PC's key folder (--tls-dir, else STRATA_STAGE_TLS).  In `start`, FILE may be a pattern
('UD-IQ4_XS/*.gguf') matched against the repository's files here; each agent gets the exact list and checks it
again.  Without @COMMIT the commit is setup's pin for the repository (what setup installs on this PC), else the
repository's current one.  `start` then follows the agents until none is downloading; Ctrl+C stops following, not
the downloads.
--jsonl writes setup's JSON events on stdout, plus a "file" event each time a file moves on (downloading, checking,
ok).
"""
from __future__ import annotations

import argparse
import fnmatch
import http.client
import json
import os
import re
import shutil
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(1, str(Path(__file__).resolve().parents[1]))
import stage_node  # noqa: E402

FREE_MARGIN = 10 << 30
WATCH_S = 5


def model_dir(data_dir: Path, repo: str) -> Path:
    """Where a repository's files go under data_dir (the folder name setup uses for a repository)."""
    return Path(data_dir) / "models" / repo.replace("/", "--")


def on_disk(p: Path) -> int:
    """The bytes `p` takes on the disk (a .part or a shipped file is sparse where nothing was written), 0 without it."""
    try:
        st = p.stat()
    except OSError:
        return 0
    return st.st_blocks * 512 if hasattr(st, "st_blocks") else st.st_size


def marked_sha256(text: str | None) -> str | None:
    """The SHA-256 a setup finish mark names, None when it names none."""
    m = re.search(r"sha256 ([0-9a-f]{64})", text or "")
    return m.group(1) if m else None


def hold_lock(folder: Path):
    """An exclusive lock on folder/.fetch.lock while the returned file is open; ValueError when another process
    holds it."""
    f = open(folder / ".fetch.lock", "a+b")
    try:
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        f.close()
        raise ValueError(f"another download into {folder} is still running") from None
    return f


def fetch(repo, revision, files, data_dir: Path) -> list[Path]:
    """Download `files` of `repo` at `revision` and check each against Hugging Face's SHA-256; their paths.
    ValueError, before any file is downloaded, for a name this tool does not take, a file the commit does not hold as
    a stored (LFS) file, or too little disk; setup's stop (SystemExit) for a download or a check that fails."""
    names = stage_node.check_request(repo, revision, files)
    import hf_pin_fetch
    import setup

    try:
        tree = hf_pin_fetch.tree(repo, revision)
    except (OSError, ValueError, KeyError, TypeError, AttributeError, http.client.HTTPException) as e:
        raise ValueError(f"cannot read the files of {repo} at {revision} from Hugging Face: {e}") from e
    absent = [f for f in files if f not in tree]
    if absent:
        raise ValueError(f"{repo} at {revision} has no stored file {', '.join(absent)}")
    dest = model_dir(data_dir, repo)
    paths = [dest / n for n in names]
    need = sum(0 if p.is_file() and marked_sha256(setup.mark_text(p)) == tree[f][1] else
               max(0, tree[f][0] - on_disk(p.with_name(p.name + ".part"))) for f, p in zip(files, paths))
    base = dest
    while not base.exists() and base != base.parent:
        base = base.parent
    free = shutil.disk_usage(base).free
    if free - need < FREE_MARGIN:
        raise ValueError(f"not enough disk space in {dest}: {need / 1e9:.1f} GB to download, {free / 1e9:.1f} GB "
                         f"free, and {FREE_MARGIN >> 30} GiB must stay free")
    dest.mkdir(parents=True, exist_ok=True)
    lock = hold_lock(dest)
    key = (repo, revision)
    kept = setup.REV_PINS.get(key)
    setup.REV_PINS[key] = {**(kept or {}), **{n: tree[f] for f, n in zip(files, names)}}
    try:
        for f, p in zip(files, paths):
            size, sha = tree[f]
            if marked_sha256(setup.mark_text(p)) not in (None, sha):
                p.unlink(missing_ok=True)
                p.with_name(p.name + ".done").unlink(missing_ok=True)
            setup.event("file", name=p.name, state="downloading", size=size)
            setup.download(f"{setup.hf_endpoint()}/{repo}/resolve/{revision}/{f}", p, p.name)
            setup.event("file", name=p.name, state="checking", size=size)
            setup.verify_sha256(p, size, sha)
            setup.event("file", name=p.name, state="ok", size=size)
    finally:
        if kept is None:
            setup.REV_PINS.pop(key, None)
        else:
            setup.REV_PINS[key] = kept
        lock.close()
    return paths


def resolve(source: str, patterns: list[str]) -> tuple[str, str, list[str]]:
    """REPO[@COMMIT] and file patterns: the repository, the commit (setup's pin for the repository, else its current
    one) and the paths in it that match, in order; ValueError when one matches nothing or the names are not ones
    `get` takes."""
    import hf_pin_fetch
    import setup

    repo, _, rev = source.partition("@")
    if not stage_node.REPO.fullmatch(repo) or rev and not stage_node.COMMIT.fullmatch(rev):
        raise ValueError(f"REPO or REPO@COMMIT (a full commit hash) expected, not {source!r}")
    rev = rev or setup.HF_REVISIONS.get(repo)
    if not rev:
        with urllib.request.urlopen(f"https://huggingface.co/api/models/{repo}", timeout=60) as r:
            rev = json.load(r)["sha"]
    tree = hf_pin_fetch.tree(repo, rev)
    files = []
    for pattern in patterns:
        hit = [f for f in sorted(tree) if fnmatch.fnmatchcase(f, pattern)]
        if not hit:
            raise ValueError(f"nothing in {repo} at {rev} matches {pattern}")
        files += [f for f in hit if f not in files]
    stage_node.check_request(repo, rev, files)
    return repo, rev, files


def call(url: str, ctx, body: dict | None = None) -> dict:
    """GET an agent's /fetch, or POST `body` to it; its JSON answer (a refusal's too)."""
    data = None if body is None else json.dumps(body).encode()
    try:
        with stage_node.https_opener(ctx).open(stage_node.agent_url(url) + "/fetch", data=data, timeout=60) as r:
            return json.load(r)
    except urllib.error.HTTPError as e:
        with e:
            try:
                return json.load(e)
            except ValueError:
                return {"ok": False, "error": f"HTTP {e.code}"}
    except (OSError, ValueError, http.client.HTTPException) as e:
        return {"ok": False, "error": str(e)}


def printable(text) -> str:
    """`text` with every control character escaped: what an agent sends must not drive the admin's terminal."""
    return "".join(c if c.isprintable() else c.encode("unicode_escape").decode("ascii") for c in str(text))


def running(answer) -> bool:
    """Whether an agent's answer shows a download running."""
    job = answer.get("fetch") if isinstance(answer, dict) else None
    return isinstance(job, dict) and job.get("state") == "running"


def summary(answer) -> str:
    """An agent's answer in one line: its job's state and bytes, else the error; escaped for the terminal."""
    try:
        job = answer.get("fetch")
        if not answer.get("ok") or job is None:
            return printable(answer.get("error") or "no download since the agent started")
        files = job["files"].values()
        done, total = sum(f["done_bytes"] or 0 for f in files), sum(f["size"] or 0 for f in files)
        ok = sum(f["state"] == "ok" for f in files)
        line = (f"{job['state']}: {ok}/{len(files)} files, {done / 1e9:.2f} / {total / 1e9:.2f} GB "
                f"({job['repo']} at {job['revision'][:12]})")
        return printable(line + (f" - {job['error']}" if job.get("error") else ""))
    except (AttributeError, TypeError, KeyError, ValueError):
        return "a malformed answer"


def watch(agents: list[str], ctx) -> None:
    """Print each agent's job every WATCH_S seconds until none is downloading."""
    while True:
        answers = {a: call(a, ctx) for a in agents}
        for a, answer in answers.items():
            print(f"{a}: {summary(answer)}", flush=True)
        if not any(map(running, answers.values())):
            return
        time.sleep(WATCH_S)


def get_command(a) -> None:
    repo, _, revision = a.source.partition("@")
    import setup

    if a.jsonl:
        setup.jsonl_start()
    try:
        paths = fetch(repo, revision, a.files, a.data_dir)
    except (ValueError, setup.acquisition.AcquisitionRefused) as e:
        setup.event("error", message=str(e), exit_code=1)
        sys.exit(f"stage_fetch: {e}")
    setup.ok(f"{len(paths)} files checked in {paths[0].parent}")
    setup.event("done", dir=str(paths[0].parent))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    get = sub.add_parser("get", help="download and check files of one repository at one commit")
    get.add_argument("source", help="REPO@COMMIT")
    get.add_argument("files", nargs="+", help="a .gguf path in the repository")
    get.add_argument("--data-dir", required=True, type=Path)
    get.add_argument("--jsonl", action="store_true", help="setup's JSON events on stdout")
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--agent", action="append", required=True, help="https://HOST:PORT of a node agent")
    common.add_argument("--tls-dir", help="this PC's key folder (default: STRATA_STAGE_TLS)")
    start = sub.add_parser("start", parents=[common], help="have the agents download files of one repository")
    start.add_argument("source", help="REPO or REPO@COMMIT")
    start.add_argument("files", nargs="+", help="a .gguf path or pattern in the repository")
    status = sub.add_parser("status", parents=[common], help="show the agents' downloads")
    status.add_argument("--watch", action="store_true", help="until none is downloading")
    sub.add_parser("cancel", parents=[common], help="stop the agents' downloads")
    a = ap.parse_args()
    if a.cmd == "get":
        get_command(a)
        return
    ctx = stage_node.client_context(stage_node.tls_dir(a.tls_dir, "stage_fetch"), "stage_fetch")
    body = {"cancel": True} if a.cmd == "cancel" else None
    if a.cmd == "start":
        try:
            repo, rev, files = resolve(a.source, a.files)
        except (OSError, ValueError, KeyError, TypeError, AttributeError, http.client.HTTPException) as e:
            sys.exit(f"stage_fetch: {printable(e)}")
        print(f"{repo} at {rev}: {', '.join(files)}")
        body = {"repo": repo, "revision": rev, "files": files}
    if a.cmd == "start" or getattr(a, "watch", False):
        for u in a.agent if body else []:
            print(f"{u}: {summary(call(u, ctx, body))}")
        try:
            watch(a.agent, ctx)
        except KeyboardInterrupt:
            print("stopped following; the downloads go on (stage_fetch.py status --watch)")
        return
    for u in a.agent:
        print(f"{u}: {summary(call(u, ctx, body))}")


if __name__ == "__main__":
    main()
