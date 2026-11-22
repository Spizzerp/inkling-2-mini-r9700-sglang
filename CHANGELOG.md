# Changelog

## 0.1.0 (22 November 2026)

First release. Base: `lmsysorg/sglang:inkling-2-mini-mi35x` as of 20 November 2026
(digest `sha256:f53f9694...`, SGLang `0.0.0.dev1+inkling2`, PyTorch 2.11.0 + ROCm 10.0.0, Triton 3.8.0).

- Runs Inkling 2 Mini on gfx1201 with draft mode, image input, 118K context (235K with the 8-bit KV cache).
- Ten patches (see README). Validated with the GSM8K gate on both cache settings.
