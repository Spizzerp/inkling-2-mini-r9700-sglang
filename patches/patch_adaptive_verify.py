"""Installs adaptive verify length for DSpark draft mode (see r9700_adaptive_verify.py for what it does).

usage: patch_adaptive_verify.py [SGLANG_ROOT]   (default: /sgl-workspace/sglang/python/sglang)
Copies r9700_adaptive_verify.py (next to this script) into SGLANG_ROOT/srt and hooks six files.
Off unless R9700_VERIFY_ADAPT=1 is set when the server starts; with it off the only change is one
unused argument on the top-k routing kernel.
"""
import os, shutil, sys
root = sys.argv[1] if len(sys.argv) > 1 else "/sgl-workspace/sglang/python/sglang"
here = os.path.dirname(os.path.abspath(__file__))
shutil.copy(os.path.join(here, "r9700_adaptive_verify.py"), os.path.join(root, "srt", "r9700_adaptive_verify.py"))
FILES = ["srt/models/inkling2.py", "kernels/ops/moe/inkling2_route.py", "kernels/ops/moe/inkling2_route_hip.py",
         "srt/speculative/dspark_components/dspark_worker_v2.py", "srt/speculative/dspark_components/dspark_verify.py",
         "srt/speculative/inkling2_mtp_worker.py"]
if len(sys.argv) > 2 and sys.argv[2] == "--list":
    print(" ".join(FILES)); sys.exit(0)


def edit(rel, pairs):
    path = os.path.join(root, rel)
    s = open(path).read()
    for old, new in pairs:
        assert s.count(old) == 1, (rel, s.count(old), old[:70])
        s = s.replace(old, new)
    open(path, "w").write(s)
    print("patched", rel)


# 1. The model: create the row-source buffer before any graph is captured, and tell the routing
#    kernel when it is being built into a target-verify graph.
edit(FILES[0], [
    ("        super().__init__()\n        self.config = config\n        self.layer_id = layer_id\n",
     "        super().__init__()\n        self.config = config\n        self.layer_id = layer_id\n"
     "        from sglang.srt import r9700_adaptive_verify\n\n        r9700_adaptive_verify.verify_src()\n"),
    ("        mlp = self.mlp(mlp_input, forward_batch, combine=not routed)\n",
     "        from sglang.srt import r9700_adaptive_verify\n\n"
     "        r9700_adaptive_verify.routing_verify(forward_batch.forward_mode.is_target_verify())\n"
     "        try:\n"
     "            mlp = self.mlp(mlp_input, forward_batch, combine=not routed)\n"
     "        finally:\n"
     "            r9700_adaptive_verify.routing_verify(False)\n"),
])

# 2. The top-k routing kernel: a row may take its router logits from another row (its request's anchor).
edit(FILES[1], [
    ("""    route_scale,
    M,
    N: tl.constexpr,
    GROUPS: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    TOPK: tl.constexpr,
    TOPK_POW2: tl.constexpr,
    BLOCK_N: tl.constexpr,
    RENORM: tl.constexpr,
):
    gdc_wait()
    gdc_launch_dependents()
    _topk_row(
        P, Bias, Logits, Ids, Weights, GlobalScale, route_scale, M,
        tl.program_id(0), N, GROUPS, HAS_BIAS, TOPK, TOPK_POW2, BLOCK_N, RENORM,
    )  # fmt: skip
""",
     """    route_scale,
    M,
    Src,
    N: tl.constexpr,
    GROUPS: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    TOPK: tl.constexpr,
    TOPK_POW2: tl.constexpr,
    BLOCK_N: tl.constexpr,
    RENORM: tl.constexpr,
    HAS_SRC: tl.constexpr,
):
    gdc_wait()
    gdc_launch_dependents()
    row = tl.program_id(0)
    prow = row
    if HAS_SRC:
        prow = tl.load(Src + row).to(tl.int32)
    _topk_row(
        P, Bias, Logits, Ids, Weights, GlobalScale, route_scale, M,
        row, prow, N, GROUPS, HAS_BIAS, TOPK, TOPK_POW2, BLOCK_N, RENORM,
    )  # fmt: skip
"""),
    ("    M,\n    row,\n    N: tl.constexpr,\n", "    M,\n    row,\n    prow,\n    N: tl.constexpr,\n"),
    ("""    logits = tl.load(P + row * N + cols, mask, other=0)
    for group in tl.static_range(1, GROUPS):
        logits = logits + tl.load(P + (group * M + row) * N + cols, mask, other=0)
""",
     """    logits = tl.load(P + prow * N + cols, mask, other=0)
    for group in tl.static_range(1, GROUPS):
        logits = logits + tl.load(P + (group * M + prow) * N + cols, mask, other=0)
"""),
    ("    if m:\n        _route_topk[(m,)](\n",
     "    from sglang.srt import r9700_adaptive_verify\n\n"
     "    src = r9700_adaptive_verify.route_src(m)\n"
     "    if m:\n        _route_topk[(m,)](\n"),
    ("            route_scale if fuse_renorm else 1.0,\n            m,\n            n,\n",
     "            route_scale if fuse_renorm else 1.0,\n            m,\n            src,\n            n,\n"),
    ("            fuse_renorm,\n            enable_fp_fusion=False,\n",
     "            fuse_renorm,\n            src is not None,\n            enable_fp_fusion=False,\n"),
])
edit(FILES[2], [
    ("                P, Bias, Logits, Ids, Weights, GlobalScale, route_scale, 1, 0, N,\n",
     "                P, Bias, Logits, Ids, Weights, GlobalScale, route_scale, 1, 0, 0, N,\n"),
])

# 3. The worker: choose the lengths before the static verify, feed the result back after the accept.
edit(FILES[3], [
    ("""                    self._verify_epilogue.begin_static_step(bs, fold_eligible)
                target_verify = self._verify_executor.run_non_compact(
""",
     """                    self._verify_epilogue.begin_static_step(bs, fold_eligible)
                from sglang.srt import r9700_adaptive_verify

                if r9700_adaptive_verify.ENABLED and layout is None:
                    r9700_adaptive_verify.begin_pass(
                        batch.req_pool_indices,
                        bs,
                        self.verify_num_draft_tokens,
                        self.model_runner.req_to_token_pool.req_to_token.shape[0] + 1,
                        (
                            None
                            if self._verify_epilogue is None
                            else self._verify_epilogue.verify_lens_buf
                        ),
                    )
                target_verify = self._verify_executor.run_non_compact(
"""),
    ("""        if proposal.acceptance_features is not None and self._simulate_acc_len <= 0:
            self._observe_acceptance(
""",
     """        if not run_compact:
            from sglang.srt import r9700_adaptive_verify

            if r9700_adaptive_verify.ENABLED:
                r9700_adaptive_verify.observe(accept.correct_len)
        if proposal.acceptance_features is not None and self._simulate_acc_len <= 0:
            self._observe_acceptance(
"""),
])

# 4. The accept step: cap it at the scheduled lengths, in the in-graph (greedy) and the eager path.
edit(FILES[4], [
    ("""        commit_lens = self._accept(
            candidates=io.candidates,
            logits=io.logits,
            draft_tokens=io.candidates[:, 1:].contiguous(),
            seq_lens=io.seq_lens,
        )
""",
     """        from sglang.srt import r9700_adaptive_verify

        commit_lens = self._accept(
            candidates=io.candidates,
            logits=io.logits,
            draft_tokens=io.candidates[:, 1:].contiguous(),
            seq_lens=io.seq_lens,
            cutoff_verify_lens=(
                verify_lens.to(torch.int32) if r9700_adaptive_verify.ENABLED else None
            ),
        )
"""),
    ("""        correct_len, bonus, cap_trim_lens = accept_draft_tokens(
            candidates=verify_ids_2d,
""",
     """        if layout is None:
            from sglang.srt import r9700_adaptive_verify

            if r9700_adaptive_verify.ENABLED:
                layout = r9700_adaptive_verify.cutoff_layout()
        correct_len, bonus, cap_trim_lens = accept_draft_tokens(
            candidates=verify_ids_2d,
"""),
])

# 5. Inkling's draft worker: a new request forgets the previous user of its slot.
edit(FILES[5], [
    ("""        last_rows = target_hidden.index_select(0, last_row_ids - 1)
        self.draft_model.store_f_rnn(batch.req_pool_indices, last_rows)
""",
     """        last_rows = target_hidden.index_select(0, last_row_ids - 1)
        self.draft_model.store_f_rnn(batch.req_pool_indices, last_rows)
        from sglang.srt import r9700_adaptive_verify

        if r9700_adaptive_verify.ENABLED:
            r9700_adaptive_verify.reset(batch.req_pool_indices)
"""),
])
