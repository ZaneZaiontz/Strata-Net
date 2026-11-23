"""Tests for tools/stage_fetch.py against a local server that plays Hugging Face: the tree API with each file's size
and SHA-256, and resolve/ answering with a redirect to the bytes, as Hugging Face's CDN does.  No internet.

    python -m unittest tools.test_stage_fetch
"""
from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os
import random
import shutil
import sys
import tempfile
import threading
import unittest
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import setup  # noqa: E402
import stage_fetch as sf  # noqa: E402

REPO, REV = "owner/Model-GGUF", "c0ffee" + "0" * 34
DATA = {"Q4/Model-Q4-00001-of-00002.gguf": random.Random(1).randbytes(70000),
        "Q4/Model-Q4-00002-of-00002.gguf": random.Random(2).randbytes(50000)}
FILES = list(DATA)


class Hub(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_HEAD(self):
        self.answer(False)

    def do_GET(self):
        self.answer(True)

    def answer(self, body: bool) -> None:
        hub = self.server
        hub.requests.append(self.path)
        if self.path.startswith(f"/api/models/{REPO}/tree/{REV}?"):
            data = b'{"not": "a list"}' if hub.junk else json.dumps([{"type": "file", "path": p, "size": len(b),
                                "lfs": {"oid": hashlib.sha256(b).hexdigest(), "size": len(b)}}
                               for p, b in DATA.items()] + [{"type": "file", "path": "README.md", "size": 5}]).encode()
        elif self.path.startswith(f"/{REPO}/resolve/{REV}/"):
            self.send_response(302)
            self.send_header("Location", "/cdn/" + self.path.split(f"/resolve/{REV}/", 1)[1])
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        elif self.path[5:] in DATA:
            data = bytearray(DATA[self.path[5:]])
            if self.path[5:] in hub.corrupt:
                data[100] ^= 0xFF
        else:
            self.send_response(404)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        self.send_response(200)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        if body:
            self.wfile.write(bytes(data))


class FetchTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="fetch"))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.hub = ThreadingHTTPServer(("127.0.0.1", 0), Hub)
        self.hub.requests, self.hub.corrupt, self.hub.junk = [], set(), False
        threading.Thread(target=self.hub.serve_forever, daemon=True).start()
        self.addCleanup(self.hub.server_close)
        self.addCleanup(self.hub.shutdown)
        base = f"http://127.0.0.1:{self.hub.server_address[1]}"
        real = urllib.request.urlopen

        def to_hub(req, *a, **k):
            url = req.full_url if isinstance(req, urllib.request.Request) else req
            if url.startswith("https://huggingface.co/"):
                req = urllib.request.Request(base + url[len("https://huggingface.co"):])
            return real(req, *a, **k)

        env = {"HF_ENDPOINT": base, "STRATA_DL_CONNECTIONS": "1", "STRATA_SOURCE": "huggingface"}
        for p in (mock.patch.object(urllib.request, "urlopen", to_hub),
                  mock.patch.object(setup.time, "sleep", lambda s: None), mock.patch.dict(os.environ, env)):
            p.start()
            self.addCleanup(p.stop)
        self.pins = {k: dict(v) for k, v in setup.REV_PINS.items()}
        self.dest = self.tmp / "models" / "owner--Model-GGUF"

    def fetch(self, files=FILES):
        with contextlib.redirect_stdout(io.StringIO()):
            return sf.fetch(REPO, REV, files, self.tmp)

    def downloads(self) -> list[str]:
        return [r for r in self.hub.requests if "/resolve/" in r]

    def test_downloads_and_checks_each_file(self):
        paths = self.fetch()
        self.assertEqual([p.name for p in paths], [f.split("/")[-1] for f in FILES])
        for f, p in zip(FILES, paths):
            self.assertEqual(p, self.dest / p.name)
            self.assertEqual(p.read_bytes(), DATA[f])
            self.assertIn(f"sha256 {hashlib.sha256(DATA[f]).hexdigest()}", setup.mark_text(p))
        self.assertTrue(any(r.startswith("/cdn/") for r in self.hub.requests))
        self.assertEqual(setup.REV_PINS, self.pins)

    def test_files_already_checked_are_not_downloaded_again(self):
        self.fetch()
        self.hub.requests.clear()
        self.fetch()
        self.assertEqual(self.downloads(), [])

    def test_a_copy_without_a_mark_is_hashed_not_trusted(self):
        self.dest.mkdir(parents=True)
        good, bad = (self.dest / f.split("/")[-1] for f in FILES)
        good.write_bytes(DATA[FILES[0]])
        bad.write_bytes(b"\0" * len(DATA[FILES[1]]))
        with self.assertRaises(SystemExit):
            self.fetch()
        self.assertEqual([r for r in self.downloads() if r.endswith(good.name)], [])
        self.assertTrue(setup.done(good))
        self.assertFalse(bad.exists())
        self.assertTrue(bad.with_name(bad.name + ".bad").is_file())

    def test_a_file_checked_for_another_commit_is_replaced_at_once(self):
        self.fetch()
        p = self.dest / FILES[1].split("/")[-1]
        p.write_bytes(b"older" * 30000)
        p.with_name(p.name + ".done").write_text("sha256 " + "e" * 64)
        self.fetch()
        self.assertEqual(p.read_bytes(), DATA[FILES[1]])

    def test_wrong_bytes_stop_and_are_kept_as_bad(self):
        self.hub.corrupt.add(FILES[1])
        with self.assertRaises(SystemExit):
            self.fetch()
        second = self.dest / FILES[1].split("/")[-1]
        self.assertTrue((self.dest / FILES[0].split("/")[-1]).is_file())
        self.assertFalse(second.exists())
        self.assertTrue(second.with_name(second.name + ".bad").is_file())
        self.assertEqual(setup.REV_PINS, self.pins)
        self.hub.corrupt.clear()
        self.fetch()
        self.assertEqual(second.read_bytes(), DATA[FILES[1]])

    def test_a_file_the_commit_lacks_is_refused_before_downloading(self):
        for f in ("Q4/Other.gguf", "README.gguf"):
            with self.assertRaisesRegex(ValueError, "has no stored file"):
                self.fetch([FILES[0], f])
        self.assertEqual(self.downloads(), [])
        self.assertFalse(any(self.dest.glob("*.gguf")))

    def test_too_little_disk_is_refused_before_downloading(self):
        with mock.patch.object(sf.shutil, "disk_usage", lambda p: shutil._ntuple_diskusage(1 << 40, 0, sf.FREE_MARGIN)):
            with self.assertRaisesRegex(ValueError, "not enough disk space"):
                self.fetch()
        self.assertEqual(self.downloads(), [])
        self.assertEqual(setup.REV_PINS, self.pins)
        self.assertFalse((self.tmp / "models").exists())

    def test_a_file_checked_for_another_commit_counts_as_missing(self):
        self.dest.mkdir(parents=True)
        for f in FILES:
            p = self.dest / f.split("/")[-1]
            p.write_bytes(b"x" * 300000)
            p.with_name(p.name + ".done").write_text("sha256 " + "e" * 64)
        free = sf.FREE_MARGIN + sum(map(len, DATA.values())) - 1
        with mock.patch.object(sf.shutil, "disk_usage", lambda p: shutil._ntuple_diskusage(1 << 40, 0, free)):
            with self.assertRaisesRegex(ValueError, "not enough disk space"):
                self.fetch()

    def test_a_malformed_file_list_is_refused(self):
        self.hub.junk = True
        with self.assertRaisesRegex(ValueError, "cannot read the files"):
            self.fetch()

    def test_bad_names_are_refused_before_the_network(self):
        cases = [("owner", REV, FILES), ("../x", REV, FILES), ("a/..", REV, FILES), ("a/b/c", REV, FILES),
                 (REPO, "main", FILES), (REPO, REV.upper(), FILES), (REPO, REV[:12], FILES), (REPO, REV, []),
                 (REPO, REV, "x.gguf"), (REPO, REV, [3]), (REPO, REV, ["x.bin"]), (REPO, REV, ["../x.gguf"]),
                 (REPO, REV, ["a/../x.gguf"]), (REPO, REV, ["a\\x.gguf"]), (REPO, REV, ["/x.gguf"]),
                 (REPO, REV, ["CON.gguf"]), (REPO, REV, ["a//x.gguf"]), (REPO, REV, ["a/x.gguf", "b/x.gguf"]),
                 (REPO, REV, [f"{i}.gguf" for i in range(sf.MAX_FILES + 1)])]
        for repo, rev, files in cases:
            with self.subTest(repo=repo, rev=rev, files=str(files)[:40]), self.assertRaises(ValueError):
                sf.fetch(repo, rev, files, self.tmp)
        self.assertEqual(self.hub.requests, [])
        self.assertFalse((self.tmp / "models").exists())


if __name__ == "__main__":
    unittest.main()
