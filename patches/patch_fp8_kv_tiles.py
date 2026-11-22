"""Build-time patch: let the ROCm attention kernels read an 8-bit KV cache (--kv-cache-dtype fp8_e4m3).

With an FP8 cache the server stores K and V in one byte per value and the pool holds twice the tokens
(235,776 against 117,888 on the R9700). It failed to start: _prefix_tile in
kernels/ops/attention/inkling2_attn_hip.py passes the loaded K tile straight to tl.dot, which accepts no fp8
operand here ("Unsupported rhs dtype fp8e4nv"). Widening the tile to the query's dtype first is enough; every
read of the KV buffers in that file goes through _prefix_tile. A no-op for the default BF16 cache.

Measured 21 Nov 2026 with FP8: GSM8K 390/400 and 238/240 (padded), 12-fact sums 10/10 at 60K tokens and 4/4 at
181K, a 229K-token prompt answered correctly. Reads are slower: 29K tokens 9.3 s against 8.3 s, 108K tokens
49 s against 40 s.
"""
import pathlib

path = pathlib.Path("/sgl-workspace/sglang/python/sglang/kernels/ops/attention/inkling2_attn_hip.py")
src = path.read_text()
old = "    return tl.dot(q, k), v\n"
assert src.count(old) == 1, "unexpected source layout"
path.write_text(src.replace(old, "    k = k.to(q.dtype)  # fp8 KV cache: tl.dot takes no fp8 operand\n    v = v.to(q.dtype)\n    return tl.dot(q, k), v\n"))
print("patched", path)
