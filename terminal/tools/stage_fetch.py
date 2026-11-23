"""Download a model's GGUF files from Hugging Face into a node's data_dir, each checked against the SHA-256 that
Hugging Face's API gives for it.

    python tools/stage_fetch.py get REPO@COMMIT FILE... --data-dir DIR [--jsonl]

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
is refused before anything is written.
--jsonl writes setup's JSON events on stdout, plus a "file" event each time a file moves on (downloading, checking,
ok).
"""
from __future__ import annotations

import argparse
import http.client
import re
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(1, str(Path(__file__).resolve().parents[1]))
import stage_node  # noqa: E402

REPO = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*")
COMMIT = re.compile(r"[0-9a-f]{40}")
MAX_FILES = 64
FREE_MARGIN = 10 << 30


def check_request(repo, revision, files) -> list[str]:
    """The base names of `files` when the repository, the commit and every file are names this tool takes: .gguf
    paths whose every part is a plain name (stage_node's rule for /start), no two with the same base name.
    ValueError otherwise."""
    if not isinstance(repo, str) or not REPO.fullmatch(repo):
        raise ValueError(f"repository: owner/name expected, not {repo!r}")
    if not isinstance(revision, str) or not COMMIT.fullmatch(revision):
        raise ValueError(f"commit: a full 40-character commit hash expected, not {revision!r}")
    if not isinstance(files, list) or not 0 < len(files) <= MAX_FILES:
        raise ValueError(f"files: 1 to {MAX_FILES} paths expected")
    names = []
    for f in files:
        parts = f.split("/") if isinstance(f, str) and f.endswith(".gguf") else [""]
        if not all(stage_node.PLAIN_NAME.fullmatch(p) and p.strip(".") and not stage_node.DEVICE_NAME.fullmatch(p)
                   for p in parts):
            raise ValueError(f"file: a .gguf path of plain names expected, not {f!r}")
        names.append(parts[-1])
    if len(set(names)) != len(names):
        raise ValueError("files: two of them have the same name")
    return names


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


def fetch(repo, revision, files, data_dir: Path) -> list[Path]:
    """Download `files` of `repo` at `revision` and check each against Hugging Face's SHA-256; their paths.
    ValueError, before any file is downloaded, for a name this tool does not take, a file the commit does not hold as
    a stored (LFS) file, or too little disk; setup's stop (SystemExit) for a download or a check that fails."""
    names = check_request(repo, revision, files)
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
    return paths


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    get = sub.add_parser("get", help="download and check files of one repository at one commit")
    get.add_argument("source", help="REPO@COMMIT")
    get.add_argument("files", nargs="+", help="a .gguf path in the repository")
    get.add_argument("--data-dir", required=True, type=Path)
    get.add_argument("--jsonl", action="store_true", help="setup's JSON events on stdout")
    a = ap.parse_args()
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


if __name__ == "__main__":
    main()
