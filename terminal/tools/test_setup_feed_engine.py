"""#3469 #3437: the 0.2 releases carry no flat engine zip, so a source checkout (UPDATE.bat, setup) fetches the engine package
for its card from the signed update feed and installs its files in engine/.

    python -m unittest tools.test_setup_feed_engine
"""
from __future__ import annotations

import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
import setup  # noqa: E402
from strata_control import engine_component as ec  # noqa: E402


class FeedEngine(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.eng = self.root / "engine"
        self.eng.mkdir()
        (self.eng / "BUILD.json").write_text(json.dumps({"version": "0.1.45", "source": "prebuilt", "archs": [89]}), encoding="utf-8")
        (self.eng / setup.EXE).write_bytes(b"old")
        self.package = self.root / "components" / "engine" / "0.1.46-test"
        self.package.mkdir(parents=True)
        (self.package / "BUILD.json").write_text(json.dumps({"version": "0.1.46", "archs": [89]}), encoding="utf-8")
        (self.package / setup.EXE).write_bytes(b"new")
        for name, value in (("home", lambda: self.root), ("say", lambda *a, **k: None), ("ok", lambda *a, **k: None),
                            ("warn", lambda *a, **k: None)):
            patch = mock.patch.object(setup, name, value)
            patch.start()
            self.addCleanup(patch.stop)
        patch = mock.patch.object(setup.acquisition, "require", lambda what: None)
        patch.start()
        self.addCleanup(patch.stop)

    def run_update(self, code=0):
        return mock.patch.object(setup.subprocess, "run", return_value=types.SimpleNamespace(returncode=code, stdout="", stderr="no feed"))

    def test_the_package_becomes_the_installed_engine(self):
        with self.run_update() as run, mock.patch.object(ec, "resolve", return_value=types.SimpleNamespace(root=self.package)):
            got = setup.engine_from_feed(self.eng, updating=True)
        self.assertEqual(got, self.eng)
        cmd = run.call_args.args[0]
        self.assertIn("update", cmd)
        self.assertIn("--no-control", cmd)
        self.assertNotIn("STRATA_NO_ACQUIRE", run.call_args.kwargs["env"])        # this one step may fetch
        self.assertEqual((self.eng / setup.EXE).read_bytes(), b"new")
        meta = json.loads((self.eng / "BUILD.json").read_text(encoding="utf-8"))
        self.assertEqual((meta["version"], meta["source"]), ("0.1.46", "prebuilt"))
        self.assertEqual((self.eng / ".previous" / setup.EXE).read_bytes(), b"old")      # one generation to roll back to

    def test_no_feed_keeps_the_installed_engine(self):
        with self.run_update(code=6):
            self.assertIsNone(setup.engine_from_feed(self.eng, updating=True))
        self.assertEqual((self.eng / setup.EXE).read_bytes(), b"old")

    def test_a_release_without_the_zip_does_not_fall_back_to_the_feed(self):
        # Strata Link: upstream installs its feed package here; the fork's releases carry the zips, and upstream's
        # package would be an engine without Link's changes (the remote stage), so it compiles instead
        gpu = {"arch": 89, "archs": [89], "vram_gb": 8.0, "index": 0}
        (self.eng / "BUILD.json").write_text(json.dumps({"version": "0.1.45", "source": "prebuilt", "archs": [89]}), encoding="utf-8")
        with mock.patch.object(setup, "engine_dir", lambda toolkit=13: self.eng),                 mock.patch.object(setup, "_urlopen", side_effect=OSError("HTTP Error 404")),                 mock.patch.object(setup, "engine_from_feed", return_value=self.eng) as feed:
            for toolkit in (13, 12):
                self.assertIsNone(setup.get_prebuilt(setup.PREBUILT_URL, gpu, "none", updating=True, toolkit=toolkit))
            feed.assert_not_called()

    def test_arm64_linux_compiles_instead_of_taking_the_feed_package(self):
        # Strata Link: upstream 0.2.7.1 installs the linux-aarch64 feed package on a DGX Spark; Link compiles it there, as before
        gpu = {"arch": 121, "archs": [121], "vram_gb": 100.0, "index": 0, "uma": True}
        (self.eng / "BUILD.json").unlink()                                    # a first install
        with mock.patch.object(setup, "engine_dir", lambda toolkit=13: self.eng), mock.patch.object(setup, "is_arm64", return_value=True), \
                mock.patch.object(setup.sys, "platform", "linux"), \
                mock.patch.object(setup, "_urlopen", side_effect=AssertionError("no network")) as urlopen, \
                mock.patch.object(setup, "download", side_effect=AssertionError("no download")), \
                mock.patch.object(setup, "engine_from_feed", return_value=self.eng) as feed:
            for updating in (False, True):
                self.assertIsNone(setup.get_prebuilt(setup.PREBUILT_URL, gpu, "none", updating=updating))
            feed.assert_not_called()
            urlopen.assert_not_called()

    def test_a_feed_package_without_the_cards_code_is_not_used(self):
        for archs, ptx, want, covered in (([60, 70, 89], False, [60], True), ([75, 86, 89], False, [60], False),
                                          ([75, 86, 89], False, [60, 89], False), ([60, 70], True, [60, 89], True), ([], False, [60], True)):
            (self.eng / "BUILD.json").write_text(json.dumps({"version": "0.1.62", "archs": archs, "ptx": ptx}), encoding="utf-8")
            with self.subTest(archs=archs, ptx=ptx, want=want):
                self.assertIs(setup.feed_engine_covers(self.eng, {"arch": want[0], "archs": want}), covered)
        (self.eng / "BUILD.json").unlink()
        self.assertFalse(setup.feed_engine_covers(self.eng, {"arch": 60, "archs": [60]}))

    def test_the_minimum_is_the_0_2_engine(self):
        self.assertLess((0, 1, 45), setup.MIN_ENGINE)
        self.assertGreaterEqual((0, 1, 46), setup.WIN_HIP_MIN_ENGINE)

    def test_the_amd_engine_fetch_tells_the_update_which_family_this_install_is(self):
        """#4861: get_prebuilt_hip fetches the ready-made engine by running `strata update` as a child process. The
        child detects the GPU itself, and on a PC that has an NVIDIA card beside the AMD one that detection takes
        the NVIDIA card first (the documented order), so the AMD setup was handed the CUDA 13 package: it cannot
        run on the card and the update failed its own test start, ending in "no ready-made AMD engine"."""
        gpu = {"arch": "gfx1201", "archs": ["gfx1201"], "vram_gb": 16.0, "index": 0}
        with self.run_update(code=6) as run, \
                mock.patch.object(setup, "ROOT", self.root), \
                mock.patch.object(setup, "_urlopen", side_effect=OSError("HTTP Error 404")):
            self.assertIsNone(setup.get_prebuilt_hip(setup.PREBUILT_URL, gpu))
        self.assertEqual(run.call_args.kwargs["env"].get("STRATA_ENGINE_BACKEND"), "hip")

    def test_the_fetched_package_is_checked_as_the_family_setup_was_told_to_use(self):
        """The second half of #4861, found on real mixed hardware by its reporter: after the fetch, setup read the
        package back with `ec.resolve(store, full=True)` and no family. `resolve()` then runs `detect_backend()`,
        which on a PC with an AMD card and an NVIDIA card takes the NVIDIA card first, so the hip package the update
        had just fetched was refused with "This Strata is for AMD RX 9000 cards. Your PC has an NVIDIA card ..."
        (an `EngineMissing`, which is not a `ComponentProblem`, so it went to setup's top-level handler and stopped
        the install). The check must be asked for the family setup chose, exactly like the fetch is."""
        asked = {}

        def resolve(root, *, backend=None, **kwargs):
            asked["root"], asked["backend"] = root, backend
            raise ec.ComponentProblem("this test stops at the call it checks")

        gpu = {"arch": "gfx1201", "archs": ["gfx1201"], "vram_gb": 16.0, "index": 0}
        with self.run_update(code=0), mock.patch.object(setup, "ROOT", self.root), \
                mock.patch.object(ec, "resolve", side_effect=resolve):
            self.assertIsNone(setup.engine_from_feed(self.eng, backend="hip"))
        self.assertEqual(asked.get("backend"), "hip",
                         "the package read back is checked without the family setup was told to use")


if __name__ == "__main__":
    unittest.main()
