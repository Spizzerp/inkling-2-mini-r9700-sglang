"""Build-time patch for the R9700 (gfx1201) image.

In this PyTorch ROCm build, torch.nn.functional.scaled_dot_product_attention fails on gfx1201 with
"CUDA error: invalid argument" for every dtype, while the same maths written out (matmul, softmax,
matmul) works. Inkling2's vision tower is the only place the Inkling code calls it, so swap that call.
"""
import pathlib

path = pathlib.Path("/sgl-workspace/sglang/python/sglang/srt/models/inkling2_vision.py")
src = path.read_text()
old = "                    F.scaled_dot_product_attention("
new = "                    _sdpa_manual("
assert src.count(old) == 1, "unexpected source layout"
helper = '''

def _sdpa_manual(q, k, v, scale):
    """scaled_dot_product_attention written out; softmax in float32 for accuracy.
    Done in slices of 512 query rows: the full (heads, L, L) float32 weights of a large screenshot
    (L about 5,000 patches) took about 4 GB of transient GPU memory, each row's result is the same."""
    k_t = k.float().transpose(-1, -2)
    v_f = v.float()
    out = torch.empty(q.shape[:-1] + (v.shape[-1],), dtype=v.dtype, device=v.device)
    for start in range(0, q.shape[-2], 512):
        rows = slice(start, start + 512)
        weights = torch.softmax((q[..., rows, :].float() @ k_t) * scale, dim=-1)
        out[..., rows, :] = (weights @ v_f).to(v.dtype)
    return out


class Inkling2VisionAttention(nn.Module):'''
anchor = "\n\nclass Inkling2VisionAttention(nn.Module):"
assert src.count(anchor) == 1
path.write_text(src.replace(old, new).replace(anchor, helper))
print("patched", path)
