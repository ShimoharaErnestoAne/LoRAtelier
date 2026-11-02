"""Runs a Qwen-Image-2.1 script of musubi-tuner so that it also reads ComfyUI's ConvRot INT8 files (launched from the
engine checkout as the working directory: qwen21_int8.py <script> <script args...>).

The engine reads only bf16 files for Qwen-Image-2.1, so training needed the bf16 DiT (14.2GB) and text encoder
(17.5GB) next to the INT8 ones ComfyUI generates with (7.3GB / 9.4GB).  ComfyUI's *_int8_convrot files hold the same
tensors as the bf16 ones (same names and shapes, checked on Comfy-Org/Qwen-Image-2.1), each quantized layer as an int8
`.weight` in the rotated basis + an fp32 [N, 1] `.weight_scale` + a `.comfy_quant` JSON spec.  Here every such layer is
turned back into bf16 while it is read (the engine's own dequantize_int8_convrot_weight), so the rest of the run is
unchanged: the DiT is then quantized to fp8 as with the bf16 file, the text encoder to fp8 by --fp8_vl.  The weights
are the INT8 ones, not the original bf16: close, not identical.
Only the loaders are replaced in memory; the engine checkout is never modified.  A bf16 file goes through untouched."""
from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path

QUANT = ".comfy_quant"
SCALE = ".weight_scale"


def quant_specs(f) -> dict[str, int]:
    """Module path -> ConvRot group size of every pre-quantized layer of an open safetensors file."""
    from musubi_tuner.modules.convrot_int8_utils import parse_comfy_quant_spec
    return {k[:-len(QUANT)]: parse_comfy_quant_spec(k, f.get_tensor(k))["convrot_groupsize"]
            for k in f.keys() if k.endswith(QUANT)}


def dequant(q, scale, groupsize: int):
    """INT8 ConvRot weight -> bf16 in the original basis (on the GPU when there is one: the rotation is a matmul)."""
    import torch
    from musubi_tuner.modules.convrot_int8_kernels import dequantize_int8_convrot_weight
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    return dequantize_int8_convrot_weight(q.to(dev), scale.to(dev), groupsize).to(torch.bfloat16).cpu()


def patch_dit():
    """qwen_image_21_model.load_model streams the file through canonical_weight_hook (a module global, looked up
    per call): for an INT8 file the hook drops the scale/spec tensors and hands the engine bf16 weights."""
    from safetensors import safe_open

    from musubi_tuner.qwen_image_21 import qwen_image_21_model as m
    load, hook = m.load_model, m.canonical_weight_hook

    def load_model(dit_path, *a, **kw):
        path = Path(dit_path)
        if not path.is_file():
            return load(dit_path, *a, **kw)
        with safe_open(str(path), framework="pt", device="cpu") as f:
            specs = quant_specs(f)
            scales = {k[:-len(SCALE)]: f.get_tensor(k) for k in f.keys() if k.endswith(SCALE)} if specs else {}
        if not specs:
            return load(dit_path, *a, **kw)
        print(f"LoRAtelier: reading the INT8 DiT as bf16 ({len(specs)} layers, qwen21_int8.py)", flush=True)

        def int8_hook(key, tensor):
            if key.endswith((QUANT, SCALE)):
                return [], None if tensor is None else []
            mod = key[:-len(".weight")] if key.endswith(".weight") else None
            if tensor is not None and mod in specs:
                tensor = dequant(tensor, scales[mod], specs[mod])
            return hook(key, tensor)

        m.canonical_weight_hook = int8_hook
        try:
            return load(dit_path, *a, **kw)
        finally:
            m.canonical_weight_hook = hook

    m.load_model = load_model


def patch_text_encoder():
    """_load_single_file_text_encoder reads the whole file with qwen_image_21_utils.load_file: for an INT8 file it
    gets bf16 weights instead, made one layer at a time (the int8 file is never held whole next to the bf16 copy)."""
    from safetensors import safe_open

    from musubi_tuner.qwen_image_21 import qwen_image_21_utils as u
    load = u.load_file

    def load_file(path, device="cpu"):
        with safe_open(str(path), framework="pt", device="cpu") as f:
            specs = quant_specs(f)
            if not specs:
                return load(path, device=device)
            print(f"LoRAtelier: reading the INT8 text encoder as bf16 ({len(specs)} layers, qwen21_int8.py)", flush=True)
            sd = {}
            for k in f.keys():
                if k.endswith((QUANT, SCALE)):
                    continue
                mod = k[:-len(".weight")] if k.endswith(".weight") else None
                t = f.get_tensor(k)
                sd[k] = dequant(t, f.get_tensor(mod + SCALE), specs[mod]) if mod in specs else t
        return {k: v.to(device) for k, v in sd.items()} if str(device) != "cpu" else sd

    u.load_file = load_file


def main():
    engine = Path(os.getcwd()).resolve()
    script = sys.argv.pop(1)
    # this file's folder must not shadow the engine's modules
    sys.path[:] = [p for p in sys.path if Path(p or ".").resolve() != Path(__file__).resolve().parent]
    sys.path.insert(0, str(engine / "src"))
    sys.path.insert(0, str(engine))
    patch_dit()
    patch_text_encoder()
    sys.argv[0] = str(engine / script)
    runpy.run_path(str(engine / script), run_name="__main__")


if __name__ == "__main__":
    main()
