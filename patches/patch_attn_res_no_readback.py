"""Build-time patch: the block attention residual must not read its FP32 scratch buffer back on gfx1201.

kernels/ops/layernorm/inkling2_norm_hip.py, _attn_res_kernel (the one-workgroup kernel behind every layer's
attention input and MLP input): with a fused RMSNorm it stores each FP32 tile to a scratch buffer
(torch.empty) and loads it back later in the same kernel to normalise it. On the Radeon AI PRO R9700 that
load sometimes returns what the buffer held before the store, for part of a row.

Found 22 Nov 2026. Symptoms and evidence:
- Any change in memory layout (one extra unused allocation in the expert layer was enough) took GSM8K from
  97% to about 77% with bursts of answers running to the token limit. With the usual layout the stale data is
  the previous call's result for similar rows, so the error is small and silent.
- Zero-filling only that scratch buffer restored accuracy (79/80); NaN-filling it gave 0/80.
- With CUDA graphs off and a NaN-filled scratch buffer, attn_res returned NaN in 5 of 24 rows (columns
  512..1791) from finite inputs: tokens 24, 3 sources, 1 query, one tile of 8192; five immediate repeats of the
  same call were clean. A synthetic loop of 75,000 calls never reproduced it.
- Two server starts used to give different temperature-0 text for the same prompt; with this patch they give
  identical text, also with every torch.empty buffer in the process NaN-filled.

The patch: when the whole row fits one tile, normalise the tile in registers and store only the result (no
scratch store, no load). attn_res then always picks one tile per row when it applies a norm on this chip
(4096 wide, 16 warps, where the shipped choice was three 1024 tiles). Results equal the shipped path to one
BF16 rounding step (different summation order of the squared norm). R9700_ATTN_RES_READBACK=1 restores the
shipped path for comparison. Other chips are unchanged.
"""
import pathlib
import sys

path = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else
                    "/sgl-workspace/sglang/python/sglang/kernels/ops/layernorm/inkling2_norm_hip.py")
src = path.read_text()


def once(old, new):
    global src
    assert src.count(old) == 1, f"unexpected source layout near: {old[:60]!r}"
    src = src.replace(old, new)


once(
    "@triton.jit\ndef _attn_res_kernel(\n",
    '''# --- Radeon AI PRO R9700 (gfx1201), see research/sglang-r9700/patch_attn_res_no_readback.py ---
import os as _os_r9700

try:
    _R9700_NO_READBACK = bool(
        torch.version.hip
        and torch.cuda.is_available()
        and torch.cuda.get_device_properties().gcnArchName.startswith("gfx120")
    )
except Exception:
    _R9700_NO_READBACK = False
if _os_r9700.environ.get("R9700_ATTN_RES_READBACK") == "1":
    _R9700_NO_READBACK = False
_NO_READBACK = tl.constexpr(_R9700_NO_READBACK)


@triton.jit
def _attn_res_kernel(
''',
)

# The part of the kernel after the softmax becomes the else branch of the new one-tile path.
head = "    probs = probs / tl.sum(probs, 1)[:, None]\n"
marker = head + "    norm_ss = tl.zeros((QP,), tl.float32)\n"
tail = "\n\n# One row counter per 256 bytes"
assert src.count(marker) == 1 and src.count(tail) == 1, "unexpected kernel layout"
i = src.index(marker) + len(head)
j = src.index(tail)
shipped = "".join(("    " + line if line.strip() else line) for line in src[i:j].splitlines(True))
one_tile = '''    if NORM and _NO_READBACK and H <= BLOCK:
        # The row is one tile: RMSNorm it in registers. The shipped path below stores the FP32 tile to `out`
        # and loads it back, and on gfx1201 that load can return what the buffer held before the store.
        mask = cols < H
        acc = tl.zeros((QP, BLOCK), tl.float32)
        for n in tl.static_range(N):
            v = _source_chunk(sources, addend, n, N, row * H + cols, mask, HAS_ADDEND)
            if NORMALIZE:
                v = v * tl.sum(tl.where(ns == n, invs, 0.0), 0)
            p = tl.sum(tl.where(ns[None, :] == n, probs, 0.0), 1)
            acc += p[:, None] * v[None, :]
        u = acc
        if HAS_PW:
            u = u * _norm_tile(norm_pw, qs, cols, mask, Q, QP, BLOCK)
        norm_inv = tl.math.rsqrt(tl.sum(u * u, 1) / H + norm_eps)
        y = u * norm_inv[:, None]
        if HAS_W:
            y = y * _norm_tile(norm_w, qs, cols, mask, Q, QP, BLOCK)
        m2 = q_mask[:, None] & mask[None, :]
        offs = (qs[:, None] * tokens + row) * H + cols[None, :]
        tl.store(normed + offs, y.to(normed.dtype.element_ty), m2)
    else:
'''
src = src[:i] + one_tile + shipped + src[j:]

once(
    "    if tokens:\n        _attn_res_kernel[(tokens,)](\n",
    '''    if _R9700_NO_READBACK and norms is not None and block < hidden:
        # one tile per row, so the kernel takes its no-read-back path
        block, num_warps = triton.next_power_of_2(hidden), 16
    if tokens:
        _attn_res_kernel[(tokens,)](
''',
)
path.write_text(src)
print("patched", path)
