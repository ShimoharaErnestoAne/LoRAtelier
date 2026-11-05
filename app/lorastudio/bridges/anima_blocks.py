"""Runs an sd-scripts Anima script with the DiT built at the block count of the checkpoint (launched by accelerate from
the engine checkout as the working directory: anima_blocks.py [--resume-fix] <script> <script args...>).

sd-scripts builds every Anima DiT with 28 blocks (library/anima_utils.py "We currently support fixed DiT config"), so
layer-expanded models such as Anima 2.9B (Gazingstars123/Anima-2.9B, 40 blocks) stop with "Unexpected keys in
checkpoint".  The wrapper below counts blocks.N in the checkpoint header and builds the DiT with that many; everything
else in the config is the engine's.  A 28-block file loads exactly as before.  The engine checkout is never modified.
--resume-fix also applies bridges/sd_resume.py (resumed runs)."""
from __future__ import annotations

import importlib.util
import json
import os
import re
import runpy
import struct
import sys
from pathlib import Path

ENGINE_BLOCKS = 28
_BLOCK = re.compile(r"(?:net\.|model\.diffusion_model\.)?blocks\.(\d+)\.")


def count_blocks(path: str | os.PathLike) -> int | None:
    """Number of DiT blocks in an Anima safetensors file (blocks.0 .. blocks.N-1; the LLM adapter's own blocks do not
    count).  None when the header cannot be read or has no blocks."""
    try:
        with open(path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            if not 0 < n < 512 << 20:
                return None
            header = json.loads(f.read(n))
    except (OSError, ValueError, struct.error):
        return None
    found = {int(m.group(1)) for k in header for m in [_BLOCK.match(k)] if m}
    return max(found) + 1 if found else None


def install() -> None:
    """Patch the engine's loader (library.anima_utils) in memory.  Call after the engine folder is on sys.path."""
    from library import anima_models, anima_utils
    if getattr(anima_utils, "_loratelier_blocks", False):
        return
    load, init = anima_utils.load_anima_model, anima_models.Anima.__init__
    wanted: dict[str, int] = {}

    def patched_init(self, *args, **kwargs):
        n = wanted.get("n")
        if n and kwargs.get("num_blocks") == ENGINE_BLOCKS:
            kwargs["num_blocks"] = n
        init(self, *args, **kwargs)

    def patched_load(*args, **kwargs):
        path = kwargs.get("dit_path", args[1] if len(args) > 1 else None)
        n = count_blocks(path) if path else None
        if n and n != ENGINE_BLOCKS:
            print(f"LoRAtelier: Anima DiT with {n} blocks (anima_blocks.py)", flush=True)
            wanted["n"] = n
        try:
            return load(*args, **kwargs)
        finally:
            wanted.pop("n", None)

    anima_models.Anima.__init__ = patched_init
    anima_utils.load_anima_model = patched_load
    anima_utils._loratelier_blocks = True


def main():
    engine = Path(os.getcwd()).resolve()
    here = Path(__file__).resolve().parent
    resume_fix = len(sys.argv) > 1 and sys.argv[1] == "--resume-fix"
    if resume_fix:
        sys.argv.pop(1)
    script = sys.argv.pop(1)
    sd_resume = None
    if resume_fix:
        spec = importlib.util.spec_from_file_location("loratelier_sd_resume", here / "sd_resume.py")
        sd_resume = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(sd_resume)
    # this file's folder must not shadow the engine's modules
    sys.path[:] = [p for p in sys.path if Path(p or ".").resolve() != here]
    sys.path.insert(0, str(engine))
    install()
    if sd_resume:
        sd_resume.fix_train_network(engine)
    sys.argv[0] = str(engine / script)
    runpy.run_path(str(engine / script), run_name="__main__")


if __name__ == "__main__":
    main()
