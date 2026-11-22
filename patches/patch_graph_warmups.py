"""Build-time patch: make the number of warm-up passes before each GPU graph capture a setting.

srt/model_executor/runner_backend/full_cuda_graph_backend.py, capture_one: every graph is run twice
unrecorded ("so kernels are loaded and one-time setup is paid before capture") and then a third time
while it is recorded. With 50 prefill sizes up to 4,096 tokens that is 27 s of every start on the R9700
(measured 22 Nov 2026, the largest single part of a 64 s start).

R9700_GRAPH_WARMUPS=1 runs one warm-up pass instead of two. The recorded graph is the same sequence of
kernel launches either way; what changes is how much runs before recording starts. Unset or 2: unchanged.
Whether 1 is safe on this card is a question for the accuracy gates, not for this file: see the launcher
for what is switched on and the report for the gate results.
"""
import pathlib
import sys

path = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else
                    "/sgl-workspace/sglang/python/sglang/srt/model_executor/runner_backend/full_cuda_graph_backend.py")
src = path.read_text()


def once(old, new):
    global src
    assert src.count(old) == 1, f"unexpected source layout near: {old[:60]!r}"
    src = src.replace(old, new)


once(
    "        for warmup_step in range(2):\n",
    "        for warmup_step in range(_R9700_GRAPH_WARMUPS):\n",
)
once(
    "            if self._reuse_output_buffer and warmup_step == 1:\n",
    "            if self._reuse_output_buffer and warmup_step == _R9700_GRAPH_WARMUPS - 1:\n",
)
src += '''

# --- Radeon AI PRO R9700, see research/sglang-r9700/patch_graph_warmups.py ---
import os as _os_r9700_warmups

try:
    _R9700_GRAPH_WARMUPS = int(_os_r9700_warmups.environ.get("R9700_GRAPH_WARMUPS", "2"))
except ValueError:
    _R9700_GRAPH_WARMUPS = 2
if _R9700_GRAPH_WARMUPS not in (1, 2):
    _R9700_GRAPH_WARMUPS = 2
'''
path.write_text(src)
print("patched", path)
