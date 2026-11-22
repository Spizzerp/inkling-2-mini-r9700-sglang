"""Build-time patch: zero the attention output buffers on ROCm (found on the R9700, 21 Nov 2026).

A full prefill graph replays at a padded token bucket (a 57-token prompt runs in the 64-token graph). The
attention kernels write only the real tokens' rows, and the output buffer is torch.empty: the padded rows keep
whatever another recorded graph left in the shared graph memory pool. When those leftovers contain NaN or Inf
bit patterns, the real rows of the same batch come out NaN as well (seen: all 57 hidden-state rows of a
57-token prefill), and the request then generates garbage until it hits its token limit.

Which leftovers are there depends on what else was recorded, so the symptom came and went with unrelated
changes: with extra expert-GEMM tile configurations enabled, GSM8K fell from 97% to 88% in bursts; with the
shipped tiles it was rare (1 runaway answer in 400).

SGLang zeroes the padded tail in its eager-attention wrapper (radix_attention.py, _zero_padded_tokens) and
requests it on ROCm (prefill runner, _zero_padded_tail), but the full-graph ROCm path used for Inkling 2 never
does it. Fix: allocate these buffers zeroed. Cost: one memset per attention call.
"""
import pathlib

root = pathlib.Path("/sgl-workspace/sglang/python/sglang/srt/layers/attention")

path = root / "triton_backend.py"
src = path.read_text()
a = src.index("    def forward_extend(")
b = src.index("\n    def ", a + 10)
body = src[a:b]
n = body.count("q.new_empty(") + body.count("torch.empty_like(")
assert n == 3, f"unexpected source layout in forward_extend ({n})"
body = body.replace("q.new_empty(", "q.new_zeros(").replace("torch.empty_like(", "torch.zeros_like(")
path.write_text(src[:a] + body + src[b:])
print("patched", path, n)

path = root / "triton_backend_hip.py"
src = path.read_text()
n = src.count("q.new_empty(")
assert n == 3, f"unexpected source layout in triton_backend_hip ({n})"
path.write_text(src.replace("q.new_empty(", "q.new_zeros("))
print("patched", path, n)
