"""LoRAtelier works on a PC without git (the zip download, unpacked).

Reported from another PC, 2026-11-11: a comparison failed with only "[WinError 2] 指定されたファイルが見つかりません。".
rgthree-comfy was fetched with git and, after a restart, asked for its commit by starting git - which is not on PATH
there - so ComfyUI could not start.  Now: which commit a folder holds is read from the folder, the engines come from
GitHub as zips of their pinned commits, and a program that really is missing is named in the message.

The zips here are made in the test (the real ones were fetched once by hand and compared with git checkouts)."""
from __future__ import annotations

import importlib.util
import os
import stat
import sys
import urllib.error
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lorastudio import enginefetch  # noqa: E402
from lorastudio.enginefetch import FILES, MARK, fetch_archive, head_commit  # noqa: E402

NODE = "rgthree-comfy"
URL = "https://github.com/someone/some-repo.git"
OLD, NEW = "a" * 40, "b" * 40


def _zip(path: Path, commit: str, files: dict) -> Path:
    with zipfile.ZipFile(path, "w") as z:
        z.writestr(f"some-repo-{commit}/", "")
        for name, text in files.items():
            z.writestr(f"some-repo-{commit}/{name}", text)
    return path


@pytest.fixture
def github(tmp_path, monkeypatch):
    """GitHub's zips: {commit: files}.  Returns the dict and the list of URLs asked for."""
    zips, asked = {}, []

    def download(url, dest, timeout=60):
        asked.append(url)
        commit = url.rsplit("/", 1)[1][:-4]
        if commit not in zips:
            raise urllib.error.URLError("no network")
        if zips[commit] == "broken":
            Path(dest).write_bytes(b"PK\x03\x04 cut off")
        else:
            _zip(Path(dest), commit, zips[commit])
    # other tests load lorastudio again: the module this file imported, the one in use now, and the one addons
    # took its fetch_archive from may be three different objects
    from lorastudio import addons
    for g in (enginefetch.__dict__, sys.modules["lorastudio.enginefetch"].__dict__, addons.fetch_archive.__globals__):
        monkeypatch.setitem(g, "_download", download)
    return zips, asked


@pytest.fixture
def no_programs(monkeypatch):
    """A PC without git: nothing named git can be started, and any attempt to start a program is an error."""
    import subprocess
    monkeypatch.setenv("PATH", os.pathsep.join([r"C:\Windows\System32", r"C:\Windows"]))

    def refuse(*a, **k):
        raise FileNotFoundError(2, "指定されたファイルが見つかりません。")
    monkeypatch.setattr(subprocess, "run", refuse)
    monkeypatch.setattr(subprocess, "Popen", refuse)


@pytest.fixture
def install(tmp_path, monkeypatch):
    """An install folder with one custom node in engines.lock.json."""
    from lorastudio import addons
    monkeypatch.setattr(addons.config, "ROOT", tmp_path)
    lock = {"engines": {NODE: {"url": URL, "commit": NEW, "path": f"engines/ComfyUI/custom_nodes/{NODE}"}}}
    monkeypatch.setattr(addons, "_lockfile", lambda: lock)
    monkeypatch.setattr(addons, "_tried", set())
    monkeypatch.setattr(addons, "_ready", {})
    return addons, lock["engines"][NODE], tmp_path / "engines" / "ComfyUI" / "custom_nodes" / NODE


def _git_folder(d: Path, commit: str, files: dict) -> None:
    """A folder as git left it: the files, and .git with a read-only pack file."""
    (d / ".git" / "objects").mkdir(parents=True)
    (d / ".git" / "HEAD").write_text(commit + "\n", encoding="utf-8")
    pack = d / ".git" / "objects" / "pack-1.pack"
    pack.write_bytes(b"x")
    os.chmod(pack, stat.S_IREAD)
    for name, text in files.items():
        (d / name).parent.mkdir(parents=True, exist_ok=True)
        (d / name).write_text(text, encoding="utf-8")


# ------------------------------------------------------------------ which commit a folder holds, without git
def test_the_commit_is_read_from_the_folder(tmp_path):
    d = tmp_path / "engine"
    assert head_commit(d) == "" and head_commit(tmp_path) == ""          # not there / nothing to tell by
    (d / ".git").mkdir(parents=True)
    (d / ".git" / "HEAD").write_text(OLD + "\n", encoding="utf-8")       # after `git checkout <commit>`
    assert head_commit(d) == OLD
    (d / ".git" / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")   # a clone left on its branch
    assert head_commit(d) == ""                                          # the branch has no commit yet
    (d / ".git" / "packed-refs").write_text(f"# pack-refs with: peeled\n{'c' * 40} refs/heads/dev\n{NEW} refs/heads/main\n",
                                            encoding="utf-8")
    assert head_commit(d) == NEW
    (d / ".git" / "refs" / "heads").mkdir(parents=True)
    (d / ".git" / "refs" / "heads" / "main").write_text(OLD + "\n", encoding="utf-8")   # the loose ref is the newer one
    assert head_commit(d) == OLD
    (d / MARK).write_text(NEW + "\n", encoding="utf-8")                  # put there from a zip
    assert head_commit(d) == NEW


def test_a_node_git_fetched_earlier_is_ready_on_a_pc_without_git(install, no_programs):
    """The reported failure: this raised [WinError 2] and ComfyUI did not start."""
    addons, e, d = install
    _git_folder(d, NEW, {"__init__.py": ""})
    assert addons.engine_ready(NODE) and addons.ensure_node(NODE)


# ------------------------------------------------------------------ the zip from GitHub
def test_a_new_engine_is_fetched_as_a_zip(tmp_path, github):
    zips, asked = github
    zips[NEW] = {"__init__.py": "new", "web/js/a.js": "a"}
    dest = tmp_path / "custom_nodes" / NODE
    fetch_archive(URL, NEW, dest, log=lambda m: None)
    assert asked == [f"https://github.com/someone/some-repo/archive/{NEW}.zip"]
    assert (dest / "__init__.py").read_text() == "new" and (dest / "web" / "js" / "a.js").is_file()   # no top folder
    assert head_commit(dest) == NEW and (dest / FILES).read_text().split() == ["__init__.py", "web/js/a.js"]
    assert sorted(p.name for p in dest.parent.iterdir()) == [NODE]       # nothing left beside it


def test_a_newer_zip_replaces_an_older_one_and_keeps_what_else_is_in_the_folder(tmp_path, github):
    zips, _asked = github
    zips[OLD] = {"main.py": "old", "gone.py": "old", "custom_nodes/example.py": "x"}
    zips[NEW] = {"main.py": "new", "added.py": "new", "custom_nodes/example.py": "x"}
    dest = tmp_path / "ComfyUI"
    fetch_archive(URL, OLD, dest, log=lambda m: None)
    (dest / "custom_nodes" / "ComfyUI-ppm").mkdir()
    (dest / "custom_nodes" / "ComfyUI-ppm" / "__init__.py").write_text("node", encoding="utf-8")   # put there later
    fetch_archive(URL, NEW, dest, log=lambda m: None)
    assert (dest / "main.py").read_text() == "new" and (dest / "added.py").is_file()
    assert not (dest / "gone.py").exists()                               # the new version no longer has it
    assert (dest / "custom_nodes" / "ComfyUI-ppm" / "__init__.py").read_text() == "node"
    assert head_commit(dest) == NEW


def test_a_zip_over_a_folder_git_made(tmp_path, github):
    zips, _asked = github
    zips[NEW] = {"main.py": "new"}
    dest = tmp_path / "sd-scripts"
    _git_folder(dest, OLD, {"main.py": "old", "venv_note.txt": "mine"})
    fetch_archive(URL, NEW, dest, log=lambda m: None)
    assert not (dest / ".git").exists()                                  # it described the older files
    assert (dest / "main.py").read_text() == "new" and (dest / "venv_note.txt").read_text() == "mine"
    assert head_commit(dest) == NEW


@pytest.mark.parametrize("state", ["no network", "broken"])
def test_a_failed_fetch_leaves_what_was_there(tmp_path, github, state):
    zips, _asked = github
    zips[OLD] = {"main.py": "old"}
    if state == "broken":
        zips[NEW] = "broken"
    dest, new = tmp_path / "engine", tmp_path / "other"
    fetch_archive(URL, OLD, dest, log=lambda m: None)
    for target in (dest, new):
        with pytest.raises(RuntimeError) as e:
            fetch_archive(URL, NEW, target, log=lambda m: None)
        assert target.name in str(e.value) and "WinError" not in str(e.value)
    assert head_commit(dest) == OLD and (dest / "main.py").read_text() == "old"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["engine"]      # no half-made folder, no zip left


def test_a_zip_cannot_write_outside_its_folder(tmp_path, monkeypatch):
    def download(url, dest, timeout=60):
        with zipfile.ZipFile(dest, "w") as z:
            z.writestr(f"some-repo-{NEW}/ok.py", "x")
            z.writestr(f"some-repo-{NEW}/../../evil.py", "x")
    monkeypatch.setitem(fetch_archive.__globals__, "_download", download)
    with pytest.raises(RuntimeError):
        fetch_archive(URL, NEW, tmp_path / "a" / "engine", log=lambda m: None)
    assert not list(tmp_path.rglob("evil.py")) and not (tmp_path / "a" / "engine").exists()


def test_only_github_repositories():
    with pytest.raises(RuntimeError):
        enginefetch.archive_url("https://example.com/x/y.git", NEW)


# ------------------------------------------------------------------ the app: custom nodes and optional engines
def test_a_node_is_fetched_without_git(install, github, no_programs):
    addons, e, d = install
    zips, asked = github
    zips[NEW] = {"__init__.py": "node"}
    assert not addons.engine_ready(NODE)
    assert addons.ensure_node(NODE) and (d / "__init__.py").is_file() and len(asked) == 1
    addons._ready.clear()
    assert addons.ensure_node(NODE) and len(asked) == 1                  # the next start: nothing fetched again


def test_a_folder_git_made_is_updated_without_git(install, github, no_programs):
    addons, e, d = install
    zips, _asked = github
    zips[NEW] = {"__init__.py": "new"}
    _git_folder(d, OLD, {"__init__.py": "old"})                          # engines.lock.json moved on since
    assert not addons.engine_ready(NODE)
    addons.install_engine(NODE)
    assert addons.engine_ready(NODE) and (d / "__init__.py").read_text() == "new"


def test_an_engine_with_a_patch_and_packages_without_git(install, github, monkeypatch, tmp_path):
    addons, e, d = install
    zips, _asked = github
    zips[NEW] = {"ops.py": "plain"}
    (tmp_path / "patch.py").write_text(
        "def applied(d):\n    return (d / 'ops.py').read_text() == 'patched'\n"
        "def apply(d):\n    (d / 'ops.py').write_text('patched')\n", encoding="utf-8")
    e.update(patch="patch.py", pip=["gguf==0.17.1"], env="comfy")
    ran = []
    monkeypatch.setattr(addons, "_run", lambda argv, cwd=None: ran.append([str(a) for a in argv]))
    monkeypatch.setattr(addons.shutil, "which", lambda name: None)       # no git, no uv on PATH
    addons.install_engine(NODE)
    assert addons.engine_ready(NODE) and (d / "ops.py").read_text() == "patched"
    assert len(ran) == 1 and ran[0][-1] == "gguf==0.17.1" and "git" not in ran[0]


def test_git_is_used_for_a_folder_git_made_and_the_zip_when_git_fails(install, github, monkeypatch):
    addons, e, d = install
    zips, asked = github
    zips[NEW] = {"__init__.py": "from zip"}
    _git_folder(d, OLD, {"__init__.py": "old"})
    monkeypatch.setattr(addons.shutil, "which", lambda name: "git.exe" if name == "git" else None)
    monkeypatch.setattr(addons.subprocess, "run", lambda *a, **k: type("R", (), {"returncode": 0})())
    ran = []

    def run(argv, cwd=None):
        ran.append(argv[5:])
        if "checkout" in argv:
            (d / ".git" / "HEAD").write_text(argv[-1] + "\n", encoding="utf-8")
    monkeypatch.setattr(addons, "_run", run)
    addons.install_engine(NODE)
    assert addons.engine_ready(NODE) and not asked and ["checkout", "--force", NEW] in ran   # git removes dropped files itself

    (d / ".git" / "HEAD").write_text(OLD + "\n", encoding="utf-8")
    addons._ready.clear()

    def broken(argv, cwd=None):
        raise RuntimeError("git が失敗しました: fatal")
    monkeypatch.setattr(addons, "_run", broken)
    addons.install_engine(NODE)
    assert addons.engine_ready(NODE) and len(asked) == 1 and (d / "__init__.py").read_text() == "from zip"


def test_a_run_records_the_commit_of_an_engine_fetched_as_a_zip(tmp_path, github, monkeypatch):
    """It was read from .git only: "unknown" for every version of a zip-fetched engine, also in the training cache's key."""
    from lorastudio import training
    zips, _asked = github
    zips[NEW] = {"sdxl_train_network.py": "x"}
    monkeypatch.setattr(training.config, "SD_SCRIPTS_DIR", tmp_path / "engines" / "sd-scripts")
    monkeypatch.setattr(training.config, "MUSUBI_DIR", tmp_path / "engines" / "musubi-tuner")
    assert training.engine_commit("sd-scripts") == "unknown"            # not there
    sys.modules[training.__name__.rsplit(".", 1)[0] + ".enginefetch"].fetch_archive(
        URL, NEW, tmp_path / "engines" / "sd-scripts", log=lambda m: None)
    assert training.engine_commit("sd-scripts") == NEW[:12]
    _git_folder(tmp_path / "engines" / "musubi-tuner", OLD, {"a.py": "x"})   # a folder git made: as before
    assert training.engine_commit("musubi-tuner") == OLD[:12]


# ------------------------------------------------------------------ setup
@pytest.fixture
def setup_env(tmp_path, monkeypatch):
    for k in ("UV_PYTHON_INSTALL_DIR", "UV_CACHE_DIR"):   # set at import with setdefault: put back after the test
        monkeypatch.setenv(k, os.environ.get(k, str(tmp_path / k)))
    spec = importlib.util.spec_from_file_location("setup_env_under_test", ROOT.parent / "setup_env.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    monkeypatch.setattr(mod, "ROOT", tmp_path)
    monkeypatch.setattr(mod, "GIT_DIR", tmp_path / "tools" / "git")
    monkeypatch.setattr(mod, "LOCK", {"engines": {"sd-scripts": {"url": URL, "commit": NEW}}})
    return mod


def test_setup_fetches_the_engines_without_git(setup_env, github, no_programs, tmp_path):
    zips, asked = github
    zips[NEW] = {"sdxl_train_network.py": "x"}
    assert not hasattr(setup_env, "ensure_git")                          # no git is downloaded any more
    setup_env.ensure_engine("sd-scripts")
    d = tmp_path / "engines" / "sd-scripts"
    assert (d / "sdxl_train_network.py").is_file() and head_commit(d) == NEW
    setup_env.ensure_engine("sd-scripts")                                # run again: nothing to do
    assert len(asked) == 1


def test_setup_keeps_an_engine_git_checked_out_at_the_pinned_commit(setup_env, github, no_programs, tmp_path):
    _zips, asked = github
    _git_folder(tmp_path / "engines" / "sd-scripts", NEW, {"sdxl_train_network.py": "x"})
    setup_env.ensure_engine("sd-scripts")
    assert not asked and (tmp_path / "engines" / "sd-scripts" / ".git").is_dir()


def test_setup_updates_a_folder_git_made_without_git(setup_env, github, no_programs, tmp_path):
    zips, _asked = github
    zips[NEW] = {"sdxl_train_network.py": "new"}
    d = tmp_path / "engines" / "sd-scripts"
    _git_folder(d, OLD, {"sdxl_train_network.py": "old"})
    setup_env.ensure_engine("sd-scripts")
    assert (d / "sdxl_train_network.py").read_text() == "new" and head_commit(d) == NEW


# ------------------------------------------------------------------ a program that really is missing
def test_a_missing_program_is_named(tmp_path):
    from lorastudio.evaluate import start_program
    exe = tmp_path / "envs" / "comfy" / "Scripts" / "python.exe"
    for run in (False, True):
        with pytest.raises(RuntimeError) as e:
            start_program("ComfyUI ", [str(exe), "main.py"], run=run)
        assert str(exe) in str(e.value) and "ComfyUI を起動できません" in str(e.value) and "setup.bat" in str(e.value)
        assert "WinError" not in str(e.value)
