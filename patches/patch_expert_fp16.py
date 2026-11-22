"""Build-time patch: run the NVFP4 expert GEMM on FP16 instead of BF16 on gfx1201.

kernels/ops/moe/inkling2_experts_hip.py: _nvfp4_weight_tile builds each weight tile in FP16 (e2m1 value times
e4m3 block scale, exact) and then converts it to BF16, because off gfx950 the fake-quantized activations
arrive in BF16. gfx1201 has no cheap FP16 -> BF16 conversion, and the tile is (BN, BK) = 64 x 256 values per
step of every workgroup. SGLang already has the FP16 route for MI350X: activation_dtype() returns float16
there and every producer of the activations (route_topk_metadata, fake_quantize, swiglu_fake_quantize) takes
that dtype. NVFP4 values are exact in both formats, and the product accumulates in FP32 either way.

This patch returns the FP16 tile as it is and makes activation_dtype() answer float16 for NVFP4 experts on
gfx120x. Measured 22 Nov 2026 on the R9700 (tune/expert_variants in the report):
- the kernel at draft-verify size (8 tokens, 64 rows): w13 0.171 -> 0.127 ms, w2 0.095 -> 0.071 ms (weights
  read at about 500 GB/s against 365);
- output bit-identical to the BF16 route at 18 batch sizes from 1 to 4096 tokens, both GEMMs;
- the served model gives the same temperature-0 text as before.
It needs patch_attn_res_no_readback.py: before that fix, any change of memory layout such as this one
exposed stale reads in the attention-residual kernel (which looked like this patch breaking draft mode).
R9700_EXPERT_FP16=0 restores BF16. MXFP8 experts (the last four layers) are unchanged.
"""
import pathlib
import sys

path = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else
                    "/sgl-workspace/sglang/python/sglang/kernels/ops/moe/inkling2_experts_hip.py")
src = path.read_text()


def once(old, new):
    global src
    assert src.count(old) == 1, f"unexpected source layout near: {old[:60]!r}"
    src = src.replace(old, new)


once(
    "@triton.jit\ndef _nvfp4_weight_tile(\n",
    '''# --- Radeon AI PRO R9700 (gfx1201), see research/sglang-r9700/patch_expert_fp16.py ---
import os as _os_r9700_fp16

try:
    _R9700_FP16 = bool(
        IS_HIP
        and torch.cuda.is_available()
        and torch.cuda.get_device_properties().gcnArchName.startswith("gfx120")
        and _os_r9700_fp16.environ.get("R9700_EXPERT_FP16", "1") != "0"
    )
except Exception:
    _R9700_FP16 = False
_NVFP4_TILE_FP16 = tl.constexpr(_R9700_FP16)


@triton.jit
def _nvfp4_weight_tile(
''',
)
old_tail = '''    w = tl.reshape(values, (BN, BK // 16, 16)) * scales[:, :, None]
    return tl.reshape(w, (BN, BK)).to(tl.bfloat16)
'''
assert src.count(old_tail) >= 1
k = src.index(old_tail)  # the first one is _nvfp4_weight_tile
assert src.rfind("def _nvfp4_weight_tile(", 0, k) > src.rfind("\ndef ", 0, src.rfind("def _nvfp4_weight_tile(", 0, k))
src = src[:k] + '''    w = tl.reshape(tl.reshape(values, (BN, BK // 16, 16)) * scales[:, :, None], (BN, BK))
    if not _NVFP4_TILE_FP16:
        w = w.to(tl.bfloat16)
    return w
''' + src[k + len(old_tail):]
once(
    "        torch.float16 if weight_dtype == torch.uint8 and NATIVE_CVT else torch.bfloat16\n",
    '''        torch.float16
        if weight_dtype == torch.uint8 and (NATIVE_CVT or _R9700_FP16)
        else torch.bfloat16
''',
)
path.write_text(src)
print("patched", path)
