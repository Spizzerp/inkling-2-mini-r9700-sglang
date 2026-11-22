# Inkling 2 Mini on a Radeon AI PRO R9700, through SGLang

Runs Thinking Machines' [Inkling 2 Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini) (26B
mixture-of-experts, text and image input, Apache-2.0) from its original weights on one AMD Radeon AI PRO
R9700 (32 GB, gfx1201), using SGLang with draft mode (DSpark).

SGLang supports this model on AMD only for MI350X/MI355X. This repository is a Dockerfile that adds AMD's
device packages for gfx1201 to SGLang's AMD image, plus ten small source patches: four make it run, three
fix bugs that showed up on this card, three make it faster. Nothing is re-quantised: the weights are the
published ones.

**Status: unofficial and experimental.** Tested on one machine. Not affiliated with Thinking Machines,
the SGLang project or AMD. Version 0.1.0, 22 November 2026.

Source and issues: https://github.com/Spizzerp/inkling-2-mini-r9700-sglang ·
mirror: https://huggingface.co/spizzerp/inkling-2-mini-r9700-sglang

## What you get

Measured on one R9700 that drives no display (the monitor is on the CPU's integrated graphics).

| | |
|---|---|
| Writing code | 112-125 tok/s |
| Writing prose | 63-70 tok/s |
| Tool-call turns | about 200 tok/s |
| Reading a 29K-token prompt | 8 s (3,620 tok/s); a repeated prompt comes from the cache |
| Context | 117,888 tokens; 235,776 with the 8-bit KV cache (`KV=fp8`) |
| Requests at once | 4 |
| Images | yes |
| Start-up | about 100 s |
| Peak VRAM | about 30.6 GB |
| GSM8K (first 400 test questions, temperature 0) | 388/400, against 389/400 with draft mode off |

Speeds are at temperature 1 with the lowest reasoning setting and vary by about 10% with what the model
writes. How everything was measured, and more numbers: [docs/RESULTS.md](docs/RESULTS.md).

## Requirements

- Radeon AI PRO R9700 (gfx1201). Other cards are untested; several patches switch on only for `gfx120x`.
- Linux with the `amdgpu` driver (tested on kernel 7.2), Docker (rootless works) able to pass `/dev/kfd`
  and `/dev/dri` to a container.
- About 80 GB of disk: the image is about 60 GB, the weights 19 GB.
- The card should not drive your display: the server fills it to about 30.6 GB. If it does, lower
  `--mem-fraction-static` in `run.sh` (smaller context) or the desktop can run out of video memory.

## Install and run

```bash
hf download thinkingmachines/Inkling-2-Mini --local-dir ~/models/inkling-2-mini-hf
docker build -t sglang-inkling-r9700 .
./run.sh                      # OpenAI-compatible server on http://127.0.0.1:18991, model name inkling-2-mini
curl -s http://127.0.0.1:18991/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "inkling-2-mini", "messages": [{"role": "user", "content": "Hello"}],
  "chat_template_kwargs": {"reasoning_effort": "low"}}'
docker stop sglang-inkling
```

`run.sh` settings (environment variables): `MODEL_DIR`, `PORT`, `NAME`, `KV=fp8` (8-bit KV cache, twice the
context, reads 10-20% slower, lossy), `DRAFT=0` (no draft mode, about 61 tok/s for everything), `ADAPT=0`
(adaptive verify length off). Extra arguments go to `sglang.launch_server`.

The model reasons before every answer, even at `reasoning_effort: none`; `low` keeps that short.

## Checking your build

```bash
./tools/fetch_gsm8k.sh                                                    # GSM8K test set (MIT licence)
python3 tools/gsm8k.py http://127.0.0.1:18991 mine --n 400 --parallel 4   # expect about 388/400, no answer at the token limit
python3 tools/gsm8k.py http://127.0.0.1:18991 mine-padded --n 240 --parallel 4 --pad 250 --waves
python3 tools/accept_ab.py http://127.0.0.1:18991 mine 0                  # tokens per pass, ms per pass, tok/s
```

Run the gate before trusting any change to the patches: kernel-level equality checks and single-request
tests passed here while the served model was losing 20 points.

## The patches

| # | File | What it does |
|---|---|---|
| 1 | `patch_vision.py` | The vision tower avoids `scaled_dot_product_attention`, which fails on gfx1201 in this PyTorch build |
| 2 | `patch_prebuilt_kernels.py` | Draft mode: two prebuilt kernels exist for gfx950 only; use their PyTorch versions |
| 3 | `patch_verify_far_tiles.py` | Draft mode: off-by-one unmasked read in the verify attention (GPU memory fault) |
| 4 | `patch_expert_tiles.py` | Expert GEMM tile sizes for this card: reading a 29K prompt 21 s to 8 s |
| 5, 6 | `patch_zero_padded_attention.py`, `patch_mask_padded_rows.py` | Padded rows of prefill graphs were uninitialised and could turn a whole prompt into NaN |
| 7 | `patch_fp8_kv_tiles.py` | Lets the ROCm attention kernels read an 8-bit KV cache |
| 8 | `patch_attn_res_no_readback.py` | A kernel that stored a tile and read it back in the same launch sometimes got stale data on this card |
| 9 | `patch_expert_fp16.py` | 4-bit expert layers computed in FP16 instead of BF16: that kernel 1.3x faster, bit-identical |
| 10 | `patch_adaptive_verify.py`, `r9700_adaptive_verify.py` | Draft mode checks fewer drafted tokens per pass while few are being accepted: prose 11-14% faster, code unchanged |

Details, evidence and how each was found: [docs/PATCHES.md](docs/PATCHES.md). Patches 3, 5, 6 and 8 look
like bugs in SGLang or the ROCm stack rather than anything specific to this setup; they have not been
reported upstream yet.

## Known limits

- One card, one machine. The base image is pinned by digest because the patches assert on exact source
  text; a newer SGLang image will need the patches redone.
- The adaptive verify length applies to one request at a time; with several at once every draft is checked,
  as in stock SGLang.
- The 8-bit KV cache is lossy and uncalibrated. It passed the same tests, including recall in a 229K-token
  prompt, but has not been used in long sessions.
- One answer in 1,320 gate questions ran to the 3,000-token limit after patch 10 (an ambiguous question the
  model deliberates on for 600 to 2,850 tokens with or without the patch). See docs/RESULTS.md.
- Other kernels with the store-then-read-back pattern of patch 8 exist (in the expert routing) and are
  unchanged; they passed a test that fills every such buffer with NaN.
- Start-up takes about 100 s every time.

## Licence and credits

The patches and tools here are under the Apache License 2.0 (see `LICENSE`), the licence of SGLang, whose
source they modify. SGLang: github.com/sgl-project/sglang. Inkling 2 Mini: Thinking Machines Lab, Apache-2.0
with its own acceptable-use policy. ROCm and the gfx1201 device packages: AMD. GSM8K: OpenAI, MIT licence.
Written with Claude Code.
