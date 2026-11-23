# The patches

Each patch is a Python script that edits one or more files of the SGLang source inside the image and asserts
that the text it replaces is there exactly once. The scripts' own docstrings hold the full notes from the
investigation (paths such as `research/sglang-r9700/` in them refer to this repository's `patches/`).
Model facts used below: 24 layers of which the middle 12 run three times (48 layer visits per token), 128
experts with 8 per token, expert weights in NVFP4 (layers 0-19) and MXFP8 (20-23), a draft head (DSpark) that
proposes 7 tokens which the model then checks in one pass of 8 rows.

## Making it run

**1. `patch_vision.py`.** `torch.nn.functional.scaled_dot_product_attention` fails on gfx1201 in this PyTorch
ROCm build ("CUDA error: invalid argument", every dtype); the same maths written out works. The vision tower
is the only caller in the Inkling code. The patch also slices that attention so an image request peaks about
1 GB above the resting level instead of 4 GB.

**2. `patch_prebuilt_kernels.py`.** The prebuilt `sgl_kernel` package has GPU code for gfx950 only. Its
launches do not check their result, so on gfx1201 the failure ("invalid device function") surfaces later at
an unrelated PyTorch call. The draft model uses two of these kernels (SiluAndMul, rotary embedding); both
have a PyTorch `forward_native`, selected here on gfx120x. Tip for this class of error on ROCm: a one-element
PyTorch operation after each step localises the launch that failed.

**7. `patch_fp8_kv_tiles.py`.** With `--kv-cache-dtype fp8_e4m3` the server failed to start: `_prefix_tile`
passed the 8-bit K tile straight to `tl.dot` ("Unsupported rhs dtype fp8e4nv"). Widening the tile first is
enough. No effect with the default BF16 cache.

## Bugs found on this card

**3. `patch_verify_far_tiles.py`: out-of-bounds read in `_rel_verify_split`.** Key tiles "farther than the
extent from every query" are read unmasked up to `max(prefix - EXTENT + 1, 0) // BN * BN`. The draft model
has no relative-bias table (`EXTENT == 0`), so when `prefix % BN == BN - 1` the last unmasked tile covers one
index past the valid keys. Result: a wrong key in the draft's attention at best, a GPU memory fault that kills
the server at worst (here: 4 of 4 long requests sent first after a start). Only the draft model is affected,
so verified output was never wrong.

**5, 6. `patch_zero_padded_attention.py`, `patch_mask_padded_rows.py`: uninitialised padded rows.** A prefill
graph replays at a padded token count (a 57-token prompt runs in the 64-token graph). Several kernels write
only the real rows of `torch.empty` outputs, so the padded rows keep whatever another graph left in the
shared pool; when that is NaN or Inf, the real rows of the batch come out NaN too and the request writes
garbage until its token limit. It came and went with unrelated changes (GSM8K 97% to 88% in bursts).
SGLang passes every full prefill graph the live token count for this and other models mask with it; the
Inkling 2 path did not. Part of this symptom was probably the stale read of patch 8; the masks are kept.

**8. `patch_attn_res_no_readback.py`: stale read in `_attn_res_kernel`.** With a fused RMSNorm this
one-workgroup kernel stores each FP32 tile to a `torch.empty` scratch buffer and loads it back later in the
same launch. On gfx1201 the load sometimes returns what the buffer held before the store, for part of a row.
- With the usual memory layout the stale data is the previous call's result for similar rows: a small, silent
  error. Any layout change (one extra unused allocation elsewhere was enough) took GSM8K from 97% to about 77%.
- Zero-filling only that buffer restored accuracy; NaN-filling it gave 0 correct.
- With graphs off and the buffer NaN-filled, the kernel returned NaN in 5 of 24 rows (columns 512..1791) from
  finite inputs. The same call repeated five times in isolation was clean, and 75,000 synthetic calls never
  reproduced it; it needed the real server.
- The patch normalises in registers instead (one tile across the hidden size). After it, two server starts
  give identical temperature-0 output, which was not the case before.
How it was found: a control experiment (old arithmetic plus the same extra allocation), zero-filling
`torch.empty` buffers by file and then by line, NaN-filling the suspect, then graphs off with finite checks
on every module output. Whether the cause is in Triton's AMD backend, LLVM or the driver is not known.

## Speed

**4. `patch_expert_tiles.py`.** The shipped large-batch tile of the expert GEMM (tuned on MI350X) runs at
9-11 TFLOP/s on this card where other tiles reach 40-55, and reading a long prompt spent 76% of its GPU time
there. New tiles for batches above 1,024 routed rows, same output: 29K tokens in 8.3 s instead of 21 s.

**9. `patch_expert_fp16.py`.** The NVFP4 weight tile is built in FP16 and then converted to BF16, which is
expensive on gfx1201. SGLang's MI350X route keeps FP16 throughout; this enables it on gfx120x. Kernel at
draft-verify size: 0.171 to 0.127 ms and 0.095 to 0.071 ms, output bit-identical at 18 batch sizes. Needs
patch 8 (it changes the memory layout).

**10. `patch_adaptive_verify.py`, `r9700_adaptive_verify.py`.** Every draft pass checks 8 rows and every row
reads its own 8 experts in each of the 48 layer visits, about 1.5 ms per row of a 34 ms pass; on prose about
half of all passes accept no draft. Per request, an exponentially weighted estimate of the chance that the
next draft is accepted picks how many rows are worth checking (3-4 on prose and reasoning, 8 on code). Rows
past that length reuse the first row's expert routing (one pointer argument on the routing kernel, no extra
launch) and SGLang's own cutoff caps the accept, so nothing from them is committed. All on the GPU, two tiny
launches per pass. Every request starts at full length; batches of several requests check everything.
With the length pinned at 8 (`R9700_VERIFY_FORCE_LEN=8`) output is identical to the unpatched mode and a pass
costs 0.1 ms more. SGLang's own per-request mode (`SGLANG_RAGGED_VERIFY_MODE=cap-accept`) works on this card
too but its host-side planning idles the GPU about 1.1 ms per pass, and its `compact` variant crashes.

## Start-up (0.1.1)

Where the 100 s of a start went, from the timestamps of the server log and a stack sampler: 34 s compiling
three kernel modules, 6 s downloading a Rust toolchain, 28 s recording GPU graphs, 5 s for the server's own
warm-up request, the rest imports, tokenizers, weights and cache pools.

**11. `prebuild_aiter_gfx1201.py` (build step).** The base image ships aiter's kernel modules built for
gfx950. On another chip aiter sees that a module has no code for the running GPU and rebuilds it into the
container's file system, which is discarded with the container: `module_custom_all_reduce`,
`module_rmsnorm_quant` and `module_sample`, 42 s of compiling at every start. The build step compiles the
same three with `GPU_ARCHS=gfx1201`, which needs no GPU, and checks that each result carries gfx1201 code.
If a later base image rebuilds other modules, the server log names them ("rebuilding for gfx1201").

**12. Rust toolchain (Dockerfile only).** SGLang's prefix-cache extension runs `cargo --version` in its Rust
workspace at every start. The workspace pins Rust 1.92 and the image holds only `stable`, so rustup
downloaded 600 MB each time, and a start without internet waited on it. Running the same two commands
during the build installs the toolchain.

**13. `patch_graph_warmups.py`.** `capture_one` in `full_cuda_graph_backend.py` runs every graph twice
unrecorded and a third time while recording it; with 50 prompt sizes up to 4,096 tokens that was 27 s.
`R9700_GRAPH_WARMUPS=1` (set by `run.sh`) runs one unrecorded pass. The recorded graph is the same sequence
of kernel launches either way. Checked with one pass against two on the same image: GSM8K 388/400 and
236/240 padded, 234/240 padded on the 8-bit cache, identical long-context answers, and the same speeds to
within 0.3% (docs/RESULTS.md). Unset or `2` leaves SGLang's behaviour unchanged.

Left alone: recording fewer prompt sizes (`--cuda-graph-bs-prefill`) would save another 5 to 7 s, but a
prompt whose last chunk falls between two recorded sizes is padded to the next one, up to about 70 ms more
per request.

## Draft shortlist (0.1.2)

A profile of one draft pass (SGLang's `/start_profile`, every launch averaged by its position in the pass)
showed that the draft model reads the whole vocabulary twice: one projection of its 7 hidden rows through
the main model's output layer for its base logits (201,024 x 2,304 BF16, 926 MB, 1.5 ms) and, for each of the
7 proposed tokens, its folded low-rank head (201,024 x 256, 103 MB, 0.19 ms each). Both run at the card's
memory read speed already, so the only way to make them cheaper is to read less.

**14. `patch_draft_shortlist.py`, `r9700_draft_shortlist.py`, `draft_shortlist_32k.bin`.** With
`R9700_DRAFT_SHORTLIST=FILE` (set by `run.sh`) the draft model keeps compact copies of the listed rows of the
output layer and of its folded head (156 MiB for 32,000 tokens) and computes its base logits and step biases
over those. Each step's add, softcap and argmax or sample is SGLang's own `MarkovSampleStep` over the list;
the winning position is mapped back to a token id (the list is in ascending id order, so ties break as in
the full walk). For the sampling accept the step logits are written into the full-width buffer it reads,
with -inf for every token off the list: that is exactly the distribution the draft sampled from, so
speculative sampling stays exact at any temperature. The main model, the verify pass and the accept rule are
untouched; a token off the list is never proposed and the main model's own token is committed there, as
after any rejected draft. Five hooks in two files; with the variable unset nothing else runs.

- The list: tokens ranked by frequency in 240,000 tokens of the model's own answers to 388 prompts (code in
  14 languages, file rewrites, tool calls, config files, prose) and in 85 million tokens of public code and
  documentation, plus the 58 control tokens. Tested on 91 other prompts.
- Before building it the effect was measured by adding an offset to the draft's base logits inside the
  recorded graph, so that it could only propose listed tokens: 32,000 tokens cost 1.0% +/- 0.7 more passes,
  16,000 cost 4.5%. A 32,000 list misses 0.4% of the confirmed draft tokens on code and 2.8% on prose.
  The real build then reproduced that emulated run exactly (91 of 91 answers, every pass).
- The compact copies are made after SGLang has sized the KV cache pool (in `ensure_propose_staging`); made
  when the list is read they took 2% of the context length.
- The adaptive verify length (patch 10) needs the time of a pass at each check length; `run.sh` passes the
  re-measured table (`R9700_VERIFY_COST_MS`, 2.4 ms lower at every length).
- `tools/test_draft_shortlist.py` compares the path with SGLang's own on a synthetic head: identical tokens
  and step logits with a list of the whole vocabulary at temperature 0 and 1, identical to the masked
  full path with a sublist, and at temperature 1 the drafted token follows the softmax of the listed logits.
  Run it with no server up:
  `docker run --rm -v /sys:/sys:ro --device /dev/kfd --device /dev/dri --group-add video --ipc host -e SGLANG_INKLING2_ALLOW_UNVALIDATED_ROCM=1 -v $PWD/tools/test_draft_shortlist.py:/tmp/t.py:ro --entrypoint python3 sglang-inkling-r9700 /tmp/t.py`
- Your own list: `SHORTLIST=/path/to/ids.bin`, raw little-endian int32 token ids (ids past the vocabulary
  limit and duplicates are dropped; at least 1,024).

Limits. The list is English and code. The vocabulary has 19,700 Han and 7,900 Hangul tokens and the list
holds 485 and 279 of them: Chinese and Japanese prose lost 11% and 4% (one prompt each), while Spanish,
German, French, Russian and Hebrew gained 4 to 16%. A longer fixed list does not repair that (16,000 more
tokens taken from translated software messages in 24 languages bring Chinese from 33% to 19% of confirmed
tokens off the list); adding each conversation's own tokens to the list would, and is not built.

## Tried without gain

Draft length other than 8 as a fixed setting (prose +14%, code -17%); `GPU_MAX_HW_QUEUES` and 24 other ROCm
runtime switches; hipBLASLt tuning of the dense layers (under 2%); merging small kernel launches (a launch
costs 2.2 us in a replayed graph; about 3% reachable).
