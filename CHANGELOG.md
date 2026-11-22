# Changelog

## 0.1.1 (23 November 2026)

Start-up from about 100 s to about 47 s. Same base image, same ten patches as 0.1.0; added:

- Build step 11 (`patches/prebuild_aiter_gfx1201.py`): three kernel modules that aiter recompiled for gfx1201
  at every start are compiled once into the image.
- Build step 12: the Rust toolchain SGLang's prefix-cache extension asks for is installed in the image. It
  was downloaded at every start, so a start also needed internet.
- Patch 13 (`patches/patch_graph_warmups.py`): one warm-up pass before each recorded GPU graph instead of
  two. `run.sh` turns it on; `WARMUPS=2` restores the old behaviour.
- `run.sh`: `WARMUPS` setting; documents that `/health`, not `/v1/models`, says when the server is ready.
- Same accuracy gate results and speeds as 0.1.0 (docs/RESULTS.md). The image is about 1 GB larger.

## After 0.1.0

- Documentation: speed at 1K to 199K tokens of context for both cache settings; the weight precision stated
  prominently. No change to the patches or the image.
- `tools/ctx_bench.py`: measures reading and writing speed by context depth.

## 0.1.0 (22 November 2026)

First release. Base: `lmsysorg/sglang:inkling-2-mini-mi35x` as of 20 November 2026
(digest `sha256:f53f9694...`, SGLang `0.0.0.dev1+inkling2`, PyTorch 2.11.0 + ROCm 10.0.0, Triton 3.8.0).

- Runs Inkling 2 Mini on gfx1201 with draft mode, image input, 118K context (235K with the 8-bit KV cache).
- Ten patches (see README). Validated with the GSM8K gate on both cache settings.
