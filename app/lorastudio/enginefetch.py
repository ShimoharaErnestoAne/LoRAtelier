"""The engines (sd-scripts, ComfyUI, custom nodes ...) without git: which commit a folder holds is read from the
folder, and the files of a pinned commit are fetched from GitHub as a zip.

LoRAtelier is downloaded as a zip by people who have no git.  Asking git for a folder's commit failed there after a
restart ("[WinError 2] 指定されたファイルが見つかりません。": ComfyUI could not start), so nothing here starts a
program.  Standard library only: setup_env.py uses it before any environment exists."""
from __future__ import annotations

import os
import shutil
import stat
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

MARK = ".lorastudio_commit"     # in a folder put in place without git: the commit its files are from
FILES = ".lorastudio_files"     # the files the zip put there (one per line): the next version removes those it no longer has


def head_commit(d: Path) -> str:
    """The commit the folder's files are from ("" when it cannot be told), read without running git: the mark of a
    copy / a zip, else what .git says (HEAD holds the commit itself after a checkout of a commit, or names the
    branch whose ref holds it)."""
    d = Path(d)
    try:
        if (d / MARK).is_file():
            return (d / MARK).read_text(encoding="utf-8").strip()
        git = d / ".git"
        if not git.is_dir():
            return ""
        head = (git / "HEAD").read_text(encoding="utf-8").strip()
        if not head.startswith("ref:"):
            return head
        ref = head[4:].strip()
        if (git / ref).is_file():
            return (git / ref).read_text(encoding="utf-8").strip()
        for line in (git / "packed-refs").read_text(encoding="utf-8").splitlines():
            if line.endswith(" " + ref):
                return line.split(" ", 1)[0]
    except (OSError, UnicodeDecodeError):
        pass
    return ""


def forget_mark(d: Path) -> None:
    """After git changed the folder: .git says which commit it is, the mark of an earlier zip no longer does."""
    for name in (MARK, FILES):
        try:
            (Path(d) / name).unlink()
        except OSError:
            pass


def archive_url(url: str, commit: str) -> str:
    """GitHub's zip of a repository at one commit."""
    base = url[:-4] if url.endswith(".git") else url
    if not base.startswith("https://github.com/"):
        raise RuntimeError(f"{url} は GitHub のリポジトリではないため、zip では取得できません。")
    return f"{base.rstrip('/')}/archive/{commit}.zip"


def _download(url: str, dest: Path, timeout: float = 60) -> None:
    with urllib.request.urlopen(url, timeout=timeout) as r, open(dest, "wb") as f:
        shutil.copyfileobj(r, f, 1 << 20)


def _rmtree(d: Path) -> None:
    """shutil.rmtree that also removes read-only files (git's pack files are read-only on Windows)."""
    def writable(fn, path, _exc):
        os.chmod(path, stat.S_IWRITE)
        fn(path)
    shutil.rmtree(d, onerror=writable)


def _unpack(z: Path, work: Path) -> list[str]:
    """The zip's files into work, without the one top folder GitHub wraps them in (<repo>-<commit>/: left out so the
    paths stay as short as in the final place).  Returns the files' paths, relative to work, with "/"."""
    files = []
    with zipfile.ZipFile(z) as f:
        tops = {i.filename.split("/", 1)[0] for i in f.infolist()}
        if len(tops) != 1:
            raise zipfile.BadZipFile("one top folder expected")
        root = work.resolve()
        for i in f.infolist():
            rel = i.filename.split("/", 1)[1] if "/" in i.filename else ""
            if not rel or i.is_dir():
                continue
            out = (work / rel).resolve()
            if root not in out.parents:   # a name that points outside the folder
                raise zipfile.BadZipFile(f"unexpected name {i.filename!r}")
            out.parent.mkdir(parents=True, exist_ok=True)
            with f.open(i) as src, open(out, "wb") as dst:
                shutil.copyfileobj(src, dst, 1 << 20)
            files.append(rel)
    if not files:
        raise zipfile.BadZipFile("no files")
    return sorted(files)


def fetch_archive(url: str, commit: str, dest: Path, log=None) -> None:
    """Put the repository's files at `commit` in dest, from GitHub's zip (no git needed).

    dest is new: it appears only when everything is unpacked.  dest is there (an older version, from a zip or from
    git): the new files are written over it, the files only the older zip had are removed, and everything else in
    the folder stays (the custom nodes inside ComfyUI, files made while running).  A .git folder is removed: it would
    describe the older files."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    src = archive_url(url, commit)
    work = dest.with_name(dest.name + ".part.disabled")   # ".disabled": ComfyUI skips such a folder in custom_nodes
    z = dest.with_name(dest.name + ".zip.part")
    if work.exists():
        _rmtree(work)
    (log or (lambda m: print(m, flush=True)))(f"{dest.name} を取得しています（{src}）")
    try:
        try:
            _download(src, z)
        except (urllib.error.URLError, OSError) as e:
            raise RuntimeError(f"{dest.name} を GitHub から取得できませんでした（{e}）。インターネットに接続できるか確認してください。") from None
        work.mkdir()
        try:
            files = _unpack(z, work)
        except (zipfile.BadZipFile, OSError) as e:
            raise RuntimeError(f"取得した {dest.name} の zip を展開できませんでした（{e}）。通信の途中で壊れた可能性があります。"
                               "もう一度実行してください。") from None
        (work / FILES).write_text("\n".join(files) + "\n", encoding="utf-8")
        if not dest.exists():
            (work / MARK).write_text(commit + "\n", encoding="utf-8")
            os.replace(work, dest)
            return
        # over an older version.  Its mark is removed first and the new one written last: files replaced only
        # halfway (the PC turned off) must not look like either version
        try:
            old = {n for n in (dest / FILES).read_text(encoding="utf-8").splitlines() if n}
        except OSError:
            old = set()
        (dest / MARK).unlink(missing_ok=True)
        if (dest / ".git").is_dir():
            _rmtree(dest / ".git")
        root = dest.resolve()
        for n in old - set(files):
            p = (dest / n).resolve()
            if root in p.parents:
                p.unlink(missing_ok=True)
        shutil.copytree(work, dest, dirs_exist_ok=True)
        (dest / MARK).write_text(commit + "\n", encoding="utf-8")
    finally:
        z.unlink(missing_ok=True)
        shutil.rmtree(work, ignore_errors=True)
