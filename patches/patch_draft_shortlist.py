"""Installs the token shortlist for the DSpark draft model (see r9700_draft_shortlist.py for what it does).

usage: patch_draft_shortlist.py [SGLANG_ROOT]   (default: /sgl-workspace/sglang/python/sglang)
Copies r9700_draft_shortlist.py (next to this script) into SGLANG_ROOT/srt and hooks two files.
Off unless R9700_DRAFT_SHORTLIST=FILE is set when the server starts; with it off the hooks are five
attribute checks and nothing else changes.
"""
import os, shutil, sys
root = sys.argv[1] if len(sys.argv) > 1 else "/sgl-workspace/sglang/python/sglang"
here = os.path.dirname(os.path.abspath(__file__))
FILES = ["srt/models/inkling2_mtp.py", "srt/speculative/dspark_components/dspark_draft.py"]
if len(sys.argv) > 2 and sys.argv[2] == "--list":
    print(" ".join(FILES)); sys.exit(0)
shutil.copy(os.path.join(here, "r9700_draft_shortlist.py"), os.path.join(root, "srt", "r9700_draft_shortlist.py"))


def edit(rel, pairs):
    path = os.path.join(root, rel)
    s = open(path).read()
    for old, new in pairs:
        assert s.count(old) == 1, (rel, s.count(old), old[:70])
        s = s.replace(old, new)
    open(path, "w").write(s)
    print("patched", rel)


# 1. The draft model: read the list once the main model's output layer is attached, build the compact
#    copies after the cache pools are sized, keep the folded head's copy in step with it, and project the
#    base logits through the listed rows only.
edit(FILES[0], [
    ("            self._fold_weight[rows].copy_(folded_weight)\n            self._fold_bias[rows].copy_(folded_bias)\n",
     "            self._fold_weight[rows].copy_(folded_weight)\n            self._fold_bias[rows].copy_(folded_bias)\n"
     "        from sglang.srt import r9700_draft_shortlist\n\n"
     "        r9700_draft_shortlist.refresh_fold(self)\n"),
    ("        self.markov_head._get_staged_f = lambda: self._f_rnn_rows\n",
     "        self.markov_head._get_staged_f = lambda: self._f_rnn_rows\n"
     "        from sglang.srt import r9700_draft_shortlist\n\n"
     "        r9700_draft_shortlist.attach(self)\n"),
    ("    def ensure_propose_staging(self, req_pool_size):\n",
     "    def ensure_propose_staging(self, req_pool_size):\n"
     "        from sglang.srt import r9700_draft_shortlist\n\n"
     "        r9700_draft_shortlist.build(self)\n"),
    ("            hidden = hidden / self.logits_mup_width_multiplier\n"
     "        logits = project_through_lm_head(hidden, self.lm_head)\n",
     "            hidden = hidden / self.logits_mup_width_multiplier\n"
     "        if getattr(self, \"_r9700_sl\", None) is not None:\n"
     "            from sglang.srt import r9700_draft_shortlist\n\n"
     "            return r9700_draft_shortlist.base_logits(self, hidden), None\n"
     "        logits = project_through_lm_head(hidden, self.lm_head)\n"),
])

# 2. The seven draft steps: over the list when the head has one.
edit(FILES[1], [
    ("    shard with ``sharded_accept``. A greedy draft keeps no logits.\"\"\"\n    bs, gamma = base_logits.shape[:2]\n",
     "    shard with ``sharded_accept``. A greedy draft keeps no logits.\"\"\"\n"
     "    if getattr(markov_head, \"_r9700_sl\", None) is not None:\n"
     "        from sglang.srt import r9700_draft_shortlist\n\n"
     "        return r9700_draft_shortlist.markov_block(\n"
     "            markov_head=markov_head,\n"
     "            base_logits=base_logits,\n"
     "            anchor_tokens=anchor_tokens,\n"
     "            draft_hidden=draft_hidden,\n"
     "            step_temperatures=step_temperatures,\n"
     "            noise=noise,\n"
     "            tp_sync=tp_sync,\n"
     "            logits_out=logits_out,\n"
     "            greedy_draft=greedy_draft,\n"
     "            sharded_accept=sharded_accept,\n"
     "        )\n"
     "    bs, gamma = base_logits.shape[:2]\n"),
])
