"""Build-time patch: tile sizes for the expert GEMM on the Radeon AI PRO R9700 (gfx1201).

kernels/ops/moe/inkling2_experts_hip.py picks the tiles of _grouped_dequant_gemm in default_config(), from
sweeps on MI350X. On gfx1201 the large-batch choice (128, 128, 128, 1, 4, 1) runs at 9-11 TFLOP/s where
other tiles reach 40-55, and reading a long prompt spends 76% of its GPU time in this kernel.

The table below comes from tune/expert_sweep.py on this card (21 Nov 2026, results in tune/results/).
Only batches above 1024 routed rows change (prefill of more than 128 tokens); decode and draft-verify
tiles stay as shipped, where the sweep found nothing better. Every entry keeps split_k = 1, like the shipped tiles for these sizes, and
gave output identical to the shipped tiles in the sweep (the block sizes only regroup the same sums).
The two GEMMs of a layer (w13: 2304 wide, w2: 1152 wide) share the block height, as the routing requires.
"""
import pathlib

path = pathlib.Path("/sgl-workspace/sglang/python/sglang/kernels/ops/moe/inkling2_experts_hip.py")
src = path.read_text()

old_def = "def default_config(rows, num_experts, nvfp4=False, mfma_order=False):\n"
assert src.count(old_def) == 1
table = '''# --- Radeon AI PRO R9700 (gfx1201) tiles, see research/sglang-r9700/patch_expert_tiles.py ---
try:
    _R9700_TILES = bool(
        IS_HIP
        and torch.cuda.is_available()
        and torch.cuda.get_device_properties().gcnArchName.startswith("gfx120")
    )
except Exception:
    _R9700_TILES = False


import os as _os_r9700
_R9700_ENABLED = _os_r9700.environ.get("R9700_EXPERT_TILES", "all")


def _r9700_config(rows, num_experts, nvfp4, width):
    """(bm, bn, bk, split_k, warps, stages) for prefill-sized batches, or None to keep the
    shipped choice. width: K of the GEMM (None or >= 2048: w13, below: w2)."""
    if rows <= 1024:
        return None
    narrow = width is not None and width < 2048
    rows_per_expert = triton.cdiv(rows, num_experts)
    # R9700_EXPERT_TILES: "all" (default), "off", or a comma list of buckets: s16, s32, large.
    # s16/s32 (129-512 token batches) were once disabled: with them GSM8K fell to 88% in bursts. The
    # cause was not these tiles but uninitialised padded rows (patch_zero_padded_attention.py and
    # patch_mask_padded_rows.py); with those fixes all buckets pass the gate.
    bucket = "s16" if rows_per_expert <= 16 else "s32" if rows_per_expert <= 32 else "large"
    if _R9700_ENABLED != "all" and bucket not in _R9700_ENABLED.split(","):
        return None
    if nvfp4:
        if rows_per_expert <= 16:
            return GemmConfig(32, 256, 64, 1, 8, 1) if narrow else GemmConfig(32, 64, 64, 1, 4, 2)
        if rows_per_expert <= 32:
            return GemmConfig(64, 64, 64, 1, 4, 2) if narrow else GemmConfig(64, 256, 64, 1, 8, 1)
        if rows_per_expert <= 64:
            return None  # shipped (64, 128, 64, 1, 8, 2) is already the fastest here
        return GemmConfig(128, 256, 32, 1, 8, 2) if narrow else GemmConfig(128, 64, 64, 1, 4, 2)
    if rows_per_expert <= 16:
        return GemmConfig(32, 64, 64, 1, 4, 2)
    if rows_per_expert <= 32:
        return GemmConfig(64, 128, 32, 1, 4, 2)
    if rows_per_expert <= 64:
        return None
    return GemmConfig(64, 128, 32, 1, 4, 2)


'''
new_def = "def default_config(rows, num_experts, nvfp4=False, mfma_order=False, width=None):\n"
src = src.replace(old_def, table + new_def)

old_body = "    if is_batch_invariant_mode_enabled():\n        # One tile and K order for every batch"
assert src.count(old_body) == 1
src = src.replace(old_body, "    if _R9700_TILES and not mfma_order and not is_batch_invariant_mode_enabled():\n        tuned = _r9700_config(rows, num_experts, nvfp4, width)\n        if tuned is not None:\n            return tuned\n" + old_body)

old_call = "    cfg = config or default_config(rows, num_experts, nvfp4, mfma_order)\n"
assert src.count(old_call) == 1
src = src.replace(old_call, "    cfg = config or default_config(rows, num_experts, nvfp4, mfma_order, width)\n")
path.write_text(src)
print("patched", path)
