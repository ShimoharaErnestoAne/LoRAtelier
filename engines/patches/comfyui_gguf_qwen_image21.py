"""Local fix for ComfyUI-GGUF (leejet fork, pinned edd981b: city96's with Qwen-Image-2.1 / Krea 2 model detection) on the pinned ComfyUI: Qwen-Image-2.1 calls Linear layers
with a fused input activation (input_act="swiglu"), which GGMLOps.Linear drops, so fused GGUF files fail
with "mat1 and mat2 shapes cannot be multiplied".  Upstream: city96/ComfyUI-GGUF#487 (open, unmerged).
Applied by LoRAtelier after checking out the pinned commit; idempotent."""
from pathlib import Path

OLD = '''        def forward_ggml_cast_weights(self, input):
            weight, bias = self.cast_bias_weight(input)
            return torch.nn.functional.linear(input, weight, bias)
'''
NEW = '''        def forward_ggml_cast_weights(self, input, input_act=None, act_weight=None, act_eps=0.0,
                                      residual=None, residual_scale=None):
            # LoRAtelier patch (city96/ComfyUI-GGUF#487): newer ComfyUI passes fused input activations
            # (Qwen-Image-2.1 uses input_act="swiglu") and residuals to Linear layers.
            input = comfy.ops._eager_input_act(input, input_act, act_weight, act_eps)
            weight, bias = self.cast_bias_weight(input)
            out = torch.nn.functional.linear(input, weight, bias)
            return comfy.ops._linear_residual(out, residual, 1.0 if residual_scale is None else residual_scale)
'''


def applied(repo: Path) -> bool:
    p = Path(repo) / "ops.py"
    return p.exists() and "LoRAtelier patch" in p.read_bytes().decode("utf-8", "replace")


def apply(repo: Path) -> None:
    p = Path(repo) / "ops.py"
    s = p.read_bytes().decode("utf-8")
    if "LoRAtelier patch" in s:
        return
    crlf = "\r\n" in s
    s = s.replace("\r\n", "\n")
    if OLD not in s:
        raise RuntimeError("ComfyUI-GGUF の修正箇所が見つかりません（版が想定と違います）")
    s = s.replace(OLD, NEW, 1)
    p.write_bytes((s.replace("\n", "\r\n") if crlf else s).encode("utf-8"))
