"""Build-time patch for the R9700 (gfx1201) image: make draft mode (DSpark) work.

The image's prebuilt kernel package sgl_kernel (common_ops.*.so) contains GPU code for gfx950 only. Calling
one of its kernels on gfx1201 fails with "CUDA error: invalid device function", and because these launches
do not check their own result, the error surfaces later at an unrelated PyTorch call. Inkling 2's main model
does not use them; its DSpark draft model uses two:

  1. SiluAndMul (the draft MLP's activation)            -> srt/layers/activation.py
  2. RotaryEmbedding via apply_k_rope (context K rope)  -> srt/layers/rotary_embedding/base.py

Both classes already have a plain-PyTorch forward_native; select it on gfx120x.
Found on 21 Nov 2026 by probing the draft model step by step (see the trial report).
"""
import pathlib

ROOT = pathlib.Path("/sgl-workspace/sglang/python/sglang")
CHECK = (
    "            _is_hip\n"
    "            and torch.cuda.is_available()\n"
    "            and torch.cuda.get_device_properties(0).gcnArchName.startswith(\"gfx120\")\n"
)


def patch(rel, old, new):
    path = ROOT / rel
    src = path.read_text()
    assert src.count(old) == 1, f"unexpected source layout in {rel}"
    path.write_text(src.replace(old, new))
    print("patched", path)


old = (
    "        elif _use_aiter and envs.SGLANG_OPT_USE_AITER_SILU_MUL.get():\n"
    "            self._forward_method = self.forward_aiter\n"
    "\n"
    "    def forward_native(self, x: torch.Tensor) -> torch.Tensor:\n"
    "        d = x.shape[-1] // 2\n"
    "        return F.silu(x[..., :d]) * x[..., d:]\n"
)
new = (
    "        elif _use_aiter and envs.SGLANG_OPT_USE_AITER_SILU_MUL.get():\n"
    "            self._forward_method = self.forward_aiter\n"
    "        # gfx120x: the prebuilt silu_and_mul kernel is gfx950-only\n"
    "        if (\n" + CHECK + "        ):\n"
    "            self._forward_method = self.forward_native\n"
    "\n"
    "    def forward_native(self, x: torch.Tensor) -> torch.Tensor:\n"
    "        d = x.shape[-1] // 2\n"
    "        return F.silu(x[..., :d]) * x[..., d:]\n"
)
patch("srt/layers/activation.py", old, new)

old = "        self.position_cos, self.position_sin = None, None\n"
new = (
    "        # gfx120x: sgl_kernel's prebuilt rotary_embedding is gfx950-only\n"
    "        if (\n" + CHECK + "        ):\n"
    "            self._forward_method = self.forward_native\n" + old
)
patch("srt/layers/rotary_embedding/base.py", old, new)
