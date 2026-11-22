# Inkling 2 Mini on a Radeon AI PRO R9700 (gfx1201) through SGLang.
# SGLang's AMD image for this model is built for MI350X/MI355X (gfx950) only. This adds AMD's device packages
# for gfx1201 (same ROCm and PyTorch versions) and ten small source patches; docs/PATCHES.md explains each.
# The patches assert on the exact source text they change, so the base image is pinned by digest.
ARG BASE=lmsysorg/sglang@sha256:f53f969460964a88a203b41c8ac52efcd1cb99954f7d49c57489a3b4d761228a
FROM ${BASE}
RUN pip install --no-cache-dir --index-url https://stable.repo.amd.com/rocm/whl-next/ \
      "rocm-sdk-device-gfx1201==10.0.0" \
      "amd-torch-device-gfx1201==2.11.0+rocm10.0.0" \
      "amd-torchvision-device-gfx1201==0.26.0+rocm10.0.0"
COPY patches/ /tmp/r9700/
# 1. Vision tower: PyTorch's scaled_dot_product_attention fails on gfx1201 in this build.
RUN python3 /tmp/r9700/patch_vision.py
# 2. Draft mode: two kernels of the prebuilt sgl_kernel package are gfx950-only.
RUN python3 /tmp/r9700/patch_prebuilt_kernels.py
# 3. Draft mode: off-by-one out-of-bounds read in the verify attention kernel (GPU memory fault).
RUN python3 /tmp/r9700/patch_verify_far_tiles.py
# 4. Expert GEMM tiles tuned for gfx1201 (reading long prompts 2.5x faster).
RUN python3 /tmp/r9700/patch_expert_tiles.py
# 5, 6. Padded rows of prefill graphs were uninitialised and could turn a whole prompt into NaN.
RUN python3 /tmp/r9700/patch_zero_padded_attention.py
RUN python3 /tmp/r9700/patch_mask_padded_rows.py
# 7. 8-bit KV cache in the ROCm attention kernels (twice the context).
RUN python3 /tmp/r9700/patch_fp8_kv_tiles.py
# 8. The attention-residual kernel read its scratch buffer back and sometimes got stale data on gfx1201.
RUN python3 /tmp/r9700/patch_attn_res_no_readback.py
# 9. NVFP4 expert GEMM on FP16 instead of BF16 (about 1.3x on that kernel); needs patch 8.
RUN python3 /tmp/r9700/patch_expert_fp16.py
# 10. Adaptive verify length in draft mode. Off unless R9700_VERIFY_ADAPT=1 (run.sh sets it).
RUN python3 /tmp/r9700/patch_adaptive_verify.py
