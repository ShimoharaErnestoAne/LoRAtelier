"""One-time environment setup (run by setup.bat).  Idempotent: re-running only does what is
missing.  Dependencies are installed from the lock files recorded after verification, and
engines are checked out at the pinned commits - never "latest on every start".

  envs/app    : UI server (no torch)
  envs/train  : sd-scripts + musubi-tuner + analysis worker (torch cu128)
  envs/comfy  : ComfyUI for evaluation (separate process)

When everything is installed, envs/.setup_ok records the folder and the lock files it was made for;
start.bat runs `setup_env.py --check` and runs the setup again when that is missing or out of date
(interrupted setup, moved folder, updated lock files).
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import sysconfig
import tempfile
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LOCK = json.loads((ROOT / "engines.lock.json").read_text(encoding="utf-8"))
# Envs are built on a uv-managed Python kept in tools/python.  A Python found on PATH (e.g. miniconda)
# can ship an old VC++ runtime (msvcp140.dll) next to python.exe that is loaded first, and then
# onnxruntime-gpu / torch fail to initialise their DLLs inside the env.
os.environ.setdefault("UV_PYTHON_INSTALL_DIR", str(ROOT / "tools" / "python"))
MANAGED = ["--python-preference", "only-managed"]
# the cache on the same drive as envs/, so packages are hard-linked instead of copied into each env
os.environ.setdefault("UV_CACHE_DIR", str(ROOT / "tools" / "uv-cache"))
# the pinned uv that setup.bat fetched first: an older uv on PATH may lack the options used here
UV = str(ROOT / "tools" / "uv" / "uv.exe") if (ROOT / "tools" / "uv" / "uv.exe").exists() else shutil.which("uv")
MARKER = ROOT / "envs" / ".setup_ok"


def run(cmd, cwd=None):
    print("$", " ".join(str(c) for c in cmd), flush=True)
    subprocess.run([str(c) for c in cmd], cwd=cwd, check=True)


def ensure_uv():
    global UV
    if UV:
        return
    run([sys.executable, "-m", "pip", "install", "--user", "uv==0.11.14"])
    # --user puts uv.exe in the user scripts folder, which is usually not on PATH
    user = Path(sysconfig.get_path("scripts", f"{os.name}_user")) / "uv.exe"
    UV = str(user) if user.exists() else shutil.which("uv") or str(Path(sys.executable).parent / "Scripts" / "uv.exe")


# portable git (MinGit) fetched into tools/git when the PC has no git; the app puts it on PATH too (addons.py)
MINGIT_URL = "https://github.com/git-for-windows/git/releases/download/v2.47.1.windows.1/MinGit-2.47.1-64-bit.zip"
MINGIT_SHA256 = "50b04b55425b5c465d076cdb184f63a0cd0f86f6ec8bb4d5860114a713d2c29a"
GIT_DIR = ROOT / "tools" / "git"


def use_local_git():
    exe = GIT_DIR / "cmd" / "git.exe"
    if exe.exists() and str(exe.parent) not in os.environ["PATH"].split(os.pathsep):
        os.environ["PATH"] = str(exe.parent) + os.pathsep + os.environ["PATH"]


def ensure_git():
    use_local_git()
    if shutil.which("git"):
        return
    print("git が PC に入っていないため、このフォルダーの中に専用の git（約 45 MB）を用意します。", flush=True)
    tmp = GIT_DIR.with_name("git.partial")
    shutil.rmtree(tmp, ignore_errors=True)
    shutil.rmtree(GIT_DIR, ignore_errors=True)   # a tools/git whose git.exe does not run
    with tempfile.TemporaryDirectory() as td:
        z = Path(td) / "mingit.zip"
        urllib.request.urlretrieve(MINGIT_URL, z)
        if hashlib.sha256(z.read_bytes()).hexdigest() != MINGIT_SHA256:
            sys.exit("ダウンロードした git の内容が想定と違います（通信の途中で壊れた可能性があります）。"
                     "もう一度 setup.bat を実行してください。")
        with zipfile.ZipFile(z) as f:
            f.extractall(tmp)
    tmp.rename(GIT_DIR)   # only a fully extracted git becomes tools/git
    use_local_git()
    if not shutil.which("git") or subprocess.run(["git", "--version"], capture_output=True).returncode:
        sys.exit("用意した git を起動できませんでした。ウイルス対策ソフトが tools\\git をブロックしていないか確認してください。")


def git(d, *args):
    # safe.directory: git refuses folders it thinks belong to another user (exFAT drives, copied folders)
    return ["git", "-c", f"safe.directory={Path(d).as_posix()}", "-C", str(d), *args]


def ensure_engine(name):
    e = LOCK["engines"][name]
    d = ROOT / "engines" / name
    if not (d / ".git").exists():
        run(["git", "clone", e["url"], str(d)])
    cur = subprocess.run(git(d, "rev-parse", "HEAD"), capture_output=True, text=True).stdout.strip()
    if cur != e["commit"]:
        run(git(d, "fetch", "--depth", "200", "origin"))
        if subprocess.run(git(d, "cat-file", "-e", e["commit"] + "^{commit}"), capture_output=True).returncode:
            run(git(d, "fetch", "origin", e["commit"]))
        run(git(d, "checkout", e["commit"]))


def venv(name):
    py = ROOT / "envs" / name / "Scripts" / "python.exe"
    if py.exists() and subprocess.run([str(py), "-c", "pass"], capture_output=True).returncode:
        # e.g. the folder was moved: the env points at the Python of the old place ("No Python at ...")
        print(f"envs\\{name} が動かないため作り直します（フォルダーを移動した場合などに起こります）。", flush=True)
        shutil.rmtree(ROOT / "envs" / name)
    if not py.exists():
        run([UV, "venv", str(ROOT / "envs" / name), "--python", LOCK["python"], *MANAGED, "--seed"])
    return py


def is_managed(name):
    cfg = ROOT / "envs" / name / "pyvenv.cfg"
    home = next((ln.split("=", 1)[1].strip() for ln in cfg.read_text(encoding="utf-8").splitlines()
                 if ln.split("=", 1)[0].strip() == "home"), "")
    found = subprocess.run([UV, "python", "find", LOCK["python"], *MANAGED], capture_output=True, text=True).stdout.strip()
    return bool(found) and Path(found).parent.resolve() == Path(home).resolve()


def imports_ok(py, module):
    r = subprocess.run([str(py), "-c", f"import {module}"], capture_output=True, text=True)
    if r.returncode:
        print(((r.stderr or r.stdout).strip().splitlines() or [""])[-1], flush=True)
    return r.returncode == 0


def env_with_lock(name, lockfile, check):
    """Create/update the env; an env built on a non-managed Python whose key module fails to import
    (DLL conflict) is rebuilt once on the uv-managed Python."""
    py = venv(name)
    install_lock(py, lockfile)
    if imports_ok(py, check):
        return py
    if is_managed(name):
        sys.exit(f"envs\\{name} で {check} を読み込めません。上のエラーを確認してください（GPU ドライバーが古い場合などに起こります）。")
    print(f"envs\\{name} は PC に入っている Python を土台にしていて {check} を読み込めないため、専用の Python で作り直します。", flush=True)
    shutil.rmtree(ROOT / "envs" / name)
    py = venv(name)
    install_lock(py, lockfile)
    if not imports_ok(py, check):
        sys.exit(f"作り直した envs\\{name} でも {check} を読み込めません。上のエラーを確認してください。")
    return py


def install_lock(py, lockfile, extra_index=None):
    cmd = [UV, "pip", "install", "--python", str(py), "-r", str(ROOT / "envs" / lockfile),
           "--extra-index-url", "https://download.pytorch.org/whl/cu128", "--index-strategy", "unsafe-best-match"]
    run(cmd, cwd=ROOT)  # lock files reference engines as ./engines/... (editable installs)


def warn_cuda(py):
    """torch imports even when the NVIDIA driver is too old for its CUDA build; say so now, not at the
    first training.  Only a warning: material editing works without a GPU."""
    r = subprocess.run([str(py), "-c", "import torch; print(torch.cuda.is_available())"], capture_output=True, text=True)
    if r.stdout.strip() == "True":
        return
    drv = ""
    smi = shutil.which("nvidia-smi")
    if smi:
        drv = subprocess.run([smi, "--query-gpu=driver_version", "--format=csv,noheader"], capture_output=True,
                             text=True).stdout.strip().splitlines()[:1]
        drv = f"（今のドライバー: {drv[0]}）" if drv else ""
    print(f"\n注意: GPU（CUDA）を使えません{drv}。NVIDIA GPU とドライバー 570 以降が必要です。"
          "素材の編集はできますが、学習と画像の生成はできません。ドライバーを更新してから setup.bat を実行し直してください。\n",
          flush=True)


def lock_hash():
    h = hashlib.sha256()
    for f in [ROOT / "engines.lock.json", *sorted((ROOT / "envs").glob("*.lock.txt"))]:
        h.update(f.name.encode() + b"\0" + f.read_bytes())
    return h.hexdigest()


def marker():
    return {"root": str(ROOT), "locks": lock_hash()}


def check():
    """--check: 0 when the setup finished for this folder and these lock files (stdlib only)."""
    try:
        return 0 if json.loads(MARKER.read_text(encoding="utf-8")) == marker() else 1
    except (OSError, ValueError):
        return 1


def app_running():
    sys.path.insert(0, str(ROOT / "app"))
    from lorastudio.procjob import instance_lock_held
    data = Path(os.environ.get("LORASTUDIO_DATA", ROOT / "data"))
    return instance_lock_held(data / "app.lock")


def main():
    if app_running():
        sys.exit("LoRAtelier が起動しています。起動中は使用中のファイルを入れ替えられないため、"
                 "LoRAtelier のウィンドウを閉じてから setup.bat を実行してください。")
    # two setups at once (e.g. start.bat opened twice) would install into the same envs together
    from lorastudio.procjob import hold_instance_lock
    if not hold_instance_lock(ROOT / "envs" / ".setup.lock"):
        sys.exit("別のウィンドウでセットアップが実行中です。そちらが終わるまで待ってから、もう一度起動してください。")
    MARKER.unlink(missing_ok=True)
    for d in ("checkpoints", "loras", "diffusion_models", "text_encoders", "vae"):
        (ROOT / "models" / d).mkdir(parents=True, exist_ok=True)   # also done by setup.bat; kept for direct runs
    ensure_git()
    ensure_uv()
    for name, e in LOCK["engines"].items():
        if not e.get("optional"):   # optional engines are fetched when the model that needs them is added
            ensure_engine(name)
    app = venv("app")
    run([UV, "pip", "install", "--python", str(app), "-r", str(ROOT / "envs" / "app.lock.txt")])
    train = env_with_lock("train", "train.lock.txt", "torch")
    warn_cuda(train)
    env_with_lock("comfy", "comfy.lock.txt", "torch")
    env_with_lock("analysis", "analysis.lock.txt", "onnxruntime")
    # SDXL tokenizer files the engines fetch on first use; fetched now so that training runs offline afterwards.
    # Tokenizers of other model families are fetched when such a model is added in the app.
    run([train, "-c", "from huggingface_hub import snapshot_download as d\n"
         "for r in ['openai/clip-vit-large-patch14','laion/CLIP-ViT-bigG-14-laion2B-39B-b160k']:\n"
         "    d(r, allow_patterns=['*.json','*.txt','merges*','vocab*','*.model','*.jinja'])"])
    MARKER.write_text(json.dumps(marker(), ensure_ascii=False), encoding="utf-8")
    print("\nセットアップが完了しました。start.bat で起動できます。")
    print("ベースモデル（SDXL 系 3 種）は、起動後の最初の画面からダウンロードできます。")
    print("手元のモデルファイルは models\\checkpoints に置けば、そのまま選べます。")


if __name__ == "__main__":
    if "--check" in sys.argv:
        sys.exit(check())
    try:
        main()
    except subprocess.CalledProcessError as e:
        sys.exit(f"\nセットアップの途中で失敗しました（{Path(str(e.cmd[0])).name} が終了コード {e.returncode}）。上の表示を確認してください。\n"
                 "よくある原因: インターネットに接続できない（プロキシを含む）、ディスクの空き不足、"
                 "ファイルが使用中（LoRAtelier やウイルス対策ソフト）。直してから setup.bat をもう一度実行すると、続きから進みます。")
    except OSError as e:
        sys.exit(f"\nセットアップの途中で失敗しました: {e}\n"
                 "ディスクの空きや、ファイルが使用中でないか（LoRAtelier やウイルス対策ソフト）を確認して、setup.bat をもう一度実行してください。")
