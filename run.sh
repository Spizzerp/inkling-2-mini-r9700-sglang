#!/usr/bin/env bash
# Start Inkling 2 Mini on SGLang on a Radeon AI PRO R9700. Build the image first:  docker build -t sglang-inkling-r9700 .
# usage: run.sh [extra sglang.launch_server options]
# Settings (environment):
#   MODEL_DIR   directory with the original checkpoint (default ~/models/inkling-2-mini-hf)
#   PORT        default 18991          NAME   container name, default sglang-inkling
#   KV=fp8      8-bit KV cache: about 235K tokens of context instead of 118K, reads 10-20% slower
#   DRAFT=0     turn draft mode (DSpark) off: about 61 tok/s for everything
#   ADAPT=0     turn the adaptive verify length off
#   WARMUPS=2   two warm-up passes before each recorded GPU graph, as in stock SGLang (default 1; start 10 s longer)
#   IMAGE       default sglang-inkling-r9700
# The server takes about 47 s to start (the first start after a build takes longer: it fills the kernel cache
# in CACHE_DIR, default ~/.cache/sglang-r9700). It is ready when GET /health answers 200; /v1/models answers
# about 5 s earlier, while the server still runs its own warm-up request.
# Stop it with: docker stop $NAME
set -euo pipefail
name="${NAME:-sglang-inkling}"; port="${PORT:-18991}"; cache="${CACHE_DIR:-$HOME/.cache/sglang-r9700}"
mkdir -p "$cache/triton" "$cache/cache"
kv=(--context-length 131072)
if [[ "${KV:-bf16}" == fp8 ]]; then kv=(--context-length 262144 --kv-cache-dtype fp8_e4m3); fi
draft=(--speculative-algorithm DSPARK)
if [[ "${DRAFT:-1}" == 0 ]]; then draft=(); fi
docker rm -f "$name" >/dev/null 2>&1 || true
# -v /sys: rootless Docker gives containers an empty /sys and ROCm then finds no GPU (harmless otherwise).
# SGLANG_INKLING2_ALLOW_UNVALIDATED_ROCM: SGLang's own switch for AMD GPUs other than gfx950.
# --mem-fraction-static 0.85: a cache pool of 117,888 tokens with draft mode on a 32 GB card that drives no display.
#   If the card also drives your monitor, lower it (0.78 gave 80,512 tokens) or the desktop may run out of VRAM.
exec docker run -d --name "$name" --network host \
  -v /sys:/sys:ro --device /dev/kfd --device /dev/dri --group-add video --ipc host \
  -v "${MODEL_DIR:-$HOME/models/inkling-2-mini-hf}:/model:ro" \
  -v "$cache/triton:/root/.triton" -v "$cache/cache:/root/.cache" \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  -e SGLANG_INKLING2_ALLOW_UNVALIDATED_ROCM=1 -e ROCM_QUICK_REDUCE_QUANTIZATION=NONE \
  -e R9700_VERIFY_ADAPT="${ADAPT:-1}" -e R9700_GRAPH_WARMUPS="${WARMUPS:-1}" \
  "${IMAGE:-sglang-inkling-r9700}" \
  python3 -m sglang.launch_server --model-path /model --served-model-name inkling-2-mini \
    --host 127.0.0.1 --port "$port" --tp 1 --ep 1 --attention-backend triton --dtype bfloat16 --page-size 128 \
    --enable-multimodal "${draft[@]}" "${kv[@]}" --mem-fraction-static 0.85 \
    --max-running-requests 4 "$@"
