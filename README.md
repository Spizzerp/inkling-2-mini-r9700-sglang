# Inkling 2 Mini on a Radeon AI PRO R9700, through SGLang

Runs Thinking Machines' [Inkling 2 Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini) (26B
mixture-of-experts, text and image input, Apache-2.0) from its original weights on one AMD Radeon AI PRO
R9700 (32 GB, gfx1201), using SGLang with draft mode (DSpark).

SGLang supports this model on AMD only for MI350X/MI355X. This repository is a Dockerfile that adds AMD's
device packages for gfx1201 to SGLang's AMD image, plus twelve small source patches and two build steps:
four patches make it run, three fix bugs that showed up on this card, four make it faster, and one patch
with the two build steps cuts the start from about 100 s to about 47 s. Nothing is re-quantised: the
weights are the published ones.

**Original precision, nothing cut down.** This serves the checkpoint exactly as Thinking Machines publishes
it: attention, router and output-layer weights in 16-bit (BF16), the expert layers in the 4-bit and 8-bit
formats they ship in (NVFP4, MXFP8), and a 16-bit KV cache by default. Nothing is converted to GGUF and no
16-bit layer is reduced to 8 bits for speed. The one lossy setting is the optional 8-bit KV cache (`KV=fp8`),
which is off unless you ask for it.

**Status: unofficial and experimental.** Tested on one machine. Not affiliated with Thinking Machines,
the SGLang project or AMD. Version 0.1.2, 23 November 2026.

Source and issues: https://github.com/Spizzerp/inkling-2-mini-r9700-sglang ·
mirror: https://huggingface.co/spizzerp/inkling-2-mini-r9700-sglang

## What you get

Measured on one R9700 that drives no display (the monitor is on the CPU's integrated graphics).

| | |
|---|---|
| Writing code | about 118 tok/s on four fixed code prompts (104 in 0.1.1) |
| Writing prose | about 75 tok/s on two fixed prose prompts (64 in 0.1.1) |
| Tool-call turns | about 200 tok/s (measured on 0.1.0) |
| Reading a 29K-token prompt | 8 s (3,620 tok/s); a repeated prompt comes from the cache |
| At 100K tokens of context | cold read 33 s, then about 88 tok/s on a code task and 59 on prose (table below) |
| Context | 117,888 tokens; 235,776 with the 8-bit KV cache (`KV=fp8`) |
| Requests at once | 4 |
| Weights | the published checkpoint, unmodified (16-bit dense layers, vendor's 4-bit / 8-bit expert layers) |
| Images | yes |
| Start-up | about 47 s (about 100 s in 0.1.0); the first start after a build about 3 minutes |
| Peak VRAM | about 30.7 GB |
| GSM8K (first 400 test questions, temperature 0) | 390/400, against 389/400 with draft mode off |

Writing speeds are at temperature 0 with the lowest reasoning setting (`tools/accept_ab.py`) and depend on
what the model writes: over 91 mixed prompts 0.1.2 averages 99 tok/s on code and 77 on prose, 7% more than
0.1.1 overall. At temperature 1 the fixed prompts gave 100 to 113 tok/s for code and 68 to 70 for prose in
three runs. How everything was measured, and more numbers: [docs/RESULTS.md](docs/RESULTS.md).

## Speed with long context

Temperature 0, 400 tokens written, the context is Python source followed by a task (`tools/ctx_bench.py`).
"Cold read" is the time to read the whole prompt with nothing cached; a follow-up on the same context
normally starts in under a second.

Default cache (up to 118K tokens):

| Context | Cold read | Code task | Prose task | Time per pass, code / prose |
|---|---|---|---|---|
| 1K | 0.3 s | 89 tok/s | 70 tok/s | 29.6 / 27.0 ms |
| 9K | 2.4 s | 102 | 68 | 30.9 / 27.4 |
| 32K | 8.8 s | 115 | 67 | 34.2 / 29.5 |
| 64K | 19.5 s | 87 | 60 | 34.0 / 30.6 |
| 100K | 33.1 s | 88 | 59 | 37.1 / 33.6 |

8-bit cache (`KV=fp8`, up to 235K tokens):

| Context | Cold read | Code task | Prose task | Time per pass, code / prose |
|---|---|---|---|---|
| 1K | 0.3 s | 88 tok/s | 66 tok/s | 30.6 / 27.5 ms |
| 36K | 11.3 s | 103 | 67 | 33.7 / 29.9 |
| 90K | 35.0 s | 107 | 58 | 37.4 / 32.6 |
| 133K | 59.8 s | 75 | 53 | 37.8 / 34.4 |
| 181K | 94.0 s | 69 | 52 | 40.5 / 37.5 |

Measured on 0.1.2; a pass is 1 to 4 ms shorter than on 0.1.0, depending on the answer. A pass takes about 25% longer
at 100K than at 1K and about 32% longer at 181K. The time per pass is the steady measure: tokens per second
on the code task also moves with how many drafted tokens an answer happens to accept. Twice in these runs a
follow-up request was read again in full instead of coming from the cache: at 100K on the default cache (85%
of its capacity, as in 0.1.0) and at 133K on the 8-bit cache.

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
(adaptive verify length off), `SHORTLIST=0` (draft shortlist off: the server then behaves exactly as 0.1.1;
use it for Chinese or Japanese text), `WARMUPS=2` (two warm-up passes per recorded GPU graph as in stock
SGLang; the start takes 10 s longer), `CACHE_DIR`. Extra arguments go to `sglang.launch_server`.

The server is ready when `GET /health` answers 200. `/v1/models` answers about 5 s earlier, while the server
is still running its own warm-up request; a request sent in that gap is slow.

The model reasons before every answer, even at `reasoning_effort: none`; `low` keeps that short.

## Checking your build

```bash
./tools/fetch_gsm8k.sh                                                    # GSM8K test set (MIT licence)
python3 tools/gsm8k.py http://127.0.0.1:18991 mine --n 400 --parallel 4   # expect about 390/400, no answer at the token limit
python3 tools/gsm8k.py http://127.0.0.1:18991 mine-padded --n 240 --parallel 4 --pad 250 --waves
python3 tools/accept_ab.py http://127.0.0.1:18991 mine 0                  # tokens per pass, ms per pass, tok/s
python3 tools/ctx_bench.py http://127.0.0.1:18991                          # reading and writing speed at 1K to 100K context
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
| 11 | `prebuild_aiter_gfx1201.py` (build step) | Three kernel modules that were recompiled for this card at every start are compiled once into the image: start 100 s to 64 s |
| 12 | Dockerfile only (build step) | The Rust toolchain SGLang asks for at every start is installed in the image instead of downloaded each time: 64 s to 58 s, and a start no longer needs internet |
| 13 | `patch_graph_warmups.py` | One warm-up pass before each recorded GPU graph instead of two (`WARMUPS=2` restores two): 58 s to 47 s |
| 14 | `patch_draft_shortlist.py`, `r9700_draft_shortlist.py`, `draft_shortlist_32k.bin` | The draft model proposes from a list of 32,000 tokens instead of the whole vocabulary of 201,024 and reads only those rows: a pass 2.4 ms shorter, 1% more passes, about 7% faster overall (`SHORTLIST=0` turns it off) |

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
- The first start after a build takes about 3 minutes (177 s here) while the kernel cache in `CACHE_DIR`
  fills; later starts take about 47 s.
- Patch 13 runs each GPU graph once instead of twice before recording it. The accuracy gates, long-context
  answers and speeds were the same with one pass and with two; `WARMUPS=2` is there if you see otherwise.
- The draft shortlist (patch 14) was built from English text and code. Chinese and Japanese prose got slower
  with it (11% and 4%, one prompt each); Spanish, German, French, Russian and Hebrew got faster. `SHORTLIST=0`
  turns it off. It needs 156 MiB of VRAM; the context length is unchanged.
- The main model still picks every token with the shortlist, but which draft tokens are accepted changes
  where passes end, and at temperature 0 that changes the exact wording of most answers compared with 0.1.1
  (78 of 91 test prompts; draft mode on against off does the same). A particular prompt can therefore come
  out slower: 8 of 40 code prompts did, while code as a whole gained 8%.

## Licence and credits

The patches and tools here are under the Apache License 2.0 (see `LICENSE`), the licence of SGLang, whose
source they modify. SGLang: github.com/sgl-project/sglang. Inkling 2 Mini: Thinking Machines Lab, Apache-2.0
with its own acceptable-use policy. ROCm and the gfx1201 device packages: AMD. GSM8K: OpenAI, MIT licence.
Written with Claude Code.
