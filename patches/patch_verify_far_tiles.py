"""Build-time patch: fix an out-of-bounds read in SGLang's ROCm verify attention (found on the R9700, 21 Nov 2026).

kernels/ops/attention/inkling2_attn_hip.py, _rel_verify_split: prefix key tiles "farther than the extent
from every query" are read without a mask, up to

    far_end = max(prefix - EXTENT + 1, 0) // BN * BN

EXTENT is the width of the relative-bias table. The DSpark draft model has no such table (EXTENT == 0), so
far_end can equal prefix + 1 rounded down to a tile: when prefix % BN == BN - 1 the last unmasked tile covers
index position `prefix`, one past the valid key list. The kernel then follows whatever that index entry holds:
a wrong key in the draft's attention at best, a GPU memory fault ("kernel: _rel_verify_split") at worst.
Only the draft model is affected (the target model has EXTENT > 0), so verified output was never wrong.

Fix: never let the unmasked range pass the prefix. A no-op when EXTENT >= 1.
"""
import pathlib

path = pathlib.Path("/sgl-workspace/sglang/python/sglang/kernels/ops/attention/inkling2_attn_hip.py")
src = path.read_text()
old = "        far_end = tl.maximum(prefix - EXTENT + 1, 0) // BN * BN\n"
new = "        far_end = tl.minimum(tl.maximum(prefix - EXTENT + 1, 0), prefix) // BN * BN\n"
n = src.count(old)
assert n >= 1, "unexpected source layout"
path.write_text(src.replace(old, new))
print(f"patched {n} occurrence(s) in", path)
