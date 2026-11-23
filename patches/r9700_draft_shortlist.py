"""Token shortlist for the DSpark draft model on the Radeon R9700 (installed by patch_draft_shortlist.py).

Every draft pass the draft model reads the whole vocabulary twice: once through the main model's output
layer for its base logits (201,024 x 2,304 BF16, 926 MB) and once per proposed token through its folded
head (201,024 x 256, 103 MB, seven times). Both already run at the card's memory read speed, 2.8 ms of a
32 ms pass. Almost all tokens the main model then confirms come from a small part of the vocabulary, so
with a shortlist the draft keeps compact copies of the listed rows and reads only those:

  * base logits = hidden @ W[list].T, step bias = fold_bias[list] + latent @ fold_weight[list].T;
  * each step's add, softcap and argmax or sample run over the list (SGLang's own MarkovSampleStep) and
    the winning position is mapped back to its token id (the list is kept in ascending id order, so ties
    break as in the full walk);
  * for the sampling accept the step logits are written into the full-width buffer it reads, with -inf
    for every token off the list: the draft's distribution there is exactly the one it sampled from,
    so speculative sampling stays exact at any temperature.

The main model, the verify pass and the accept rule are untouched: a token off the list is simply never
proposed, the main model's own token is committed there as after any rejected draft. Measured on 91 test
prompts: a list of 32,000 tokens costs about 1% more passes and a pass is 2.4 ms shorter.

Environment: R9700_DRAFT_SHORTLIST=FILE enables (raw little-endian int32 token ids; unset, empty or 0:
off, and then nothing here runs); R9700_DRAFT_SHORTLIST_FOLD=1 computes the step bias inside the sampling
kernel instead of a separate GEMM (experiment).
"""
import logging
import os
from types import SimpleNamespace

import torch

logger = logging.getLogger(__name__)

_PATH = os.environ.get("R9700_DRAFT_SHORTLIST", "")
ENABLED = _PATH not in ("", "0")
_FOLD = os.environ.get("R9700_DRAFT_SHORTLIST_FOLD") == "1"


def attach(model):
    """After Inkling2ForConditionalGenerationMTP.attach_shared_modules: read the list. The compact copies are
    made later (build), after SGLang has sized the KV cache pool from the memory that is free at this
    point; made here they would cost 2% of the context length."""
    head = model.markov_head
    model._r9700_sl = head._r9700_sl = None
    model._r9700_sl_ids = None
    if not ENABLED:
        return
    if head.vocab_shard is not None:
        logger.warning("R9700 draft shortlist off: the vocabulary is sharded over several GPUs.")
        return
    import numpy as np

    full_width = int(model.lm_head.org_vocab_size)
    limit = min(int(head.step_vocab_limit or full_width), full_width, head._fold_weight.shape[0])
    ids = np.fromfile(_PATH, dtype="<i4")
    ids = np.unique(ids[(ids >= 0) & (ids < limit)])  # ascending, no duplicates
    if len(ids) < 1024:
        raise ValueError(f"R9700_DRAFT_SHORTLIST={_PATH}: only {len(ids)} usable token ids")
    model._r9700_sl_ids = (torch.from_numpy(ids.astype("int64")), limit, full_width)


def build(model):
    """From Inkling2ForConditionalGenerationMTP.ensure_propose_staging (the pools exist, no graph is
    recorded yet): the compact copies of the listed rows."""
    if getattr(model, "_r9700_sl_ids", None) is None or model._r9700_sl is not None:
        return
    head = model.markov_head
    weight = model.lm_head.weight
    ids, limit, full_width = model._r9700_sl_ids
    ids_t = ids.to(weight.device)
    sl = SimpleNamespace(
        ids=ids_t,
        n=int(ids_t.numel()),
        full_width=full_width,
        weight=weight.index_select(0, ids_t).contiguous(),
        fold_weight=None,
        fold_bias=None,
    )
    model._r9700_sl = head._r9700_sl = sl
    refresh_fold(head)
    mb = (sl.weight.numel() * sl.weight.element_size() + sl.fold_weight.numel() * sl.fold_weight.element_size()) / 2**20
    logger.info("R9700 draft shortlist on: %d of %d tokens from %s, compact copies %.0f MiB%s.",
                sl.n, limit, _PATH, mb, ", step bias in the sampling kernel" if _FOLD else "")


def refresh_fold(head):
    """After Inkling2MTPHead.refresh_fold: the compact copies of the folded head follow it."""
    sl = getattr(head, "_r9700_sl", None)
    if sl is None:
        return
    sl.fold_weight = head._fold_weight.index_select(0, sl.ids).contiguous()
    sl.fold_bias = head._fold_bias.index_select(0, sl.ids).contiguous()


def base_logits(model, hidden):
    """compute_base_logits over the list: [rows, n]."""
    weight = model._r9700_sl.weight
    return torch.matmul(hidden.to(weight.dtype), weight.T)


def markov_block(*, markov_head, base_logits, anchor_tokens, draft_hidden, step_temperatures, noise,
                 tp_sync, logits_out=None, greedy_draft=False, sharded_accept=False):
    """draft_markov_block over the list. Returns the drafted token ids [bs, gamma] and, unless the draft is
    greedy, the full-width step logits [bs, gamma, vocab] with -inf off the list."""
    from sglang.kernels.ops.speculative.dspark.markov_sample import MarkovSampleStep
    from sglang.srt.speculative.spec_tp_sync import SpecTpSyncSite

    sl = markov_head._r9700_sl
    bs, gamma, n = base_logits.shape
    assert n == sl.n and not sharded_accept
    step_dtype = markov_head.step_logits_dtype(base_logits.dtype)
    compact = None
    if not greedy_draft:
        compact = torch.empty((bs, gamma, n), dtype=step_dtype, device=base_logits.device)
    state = markov_head.init_state(draft_hidden)
    prev_tokens = anchor_tokens.long()
    draft_tokens = []
    for step in range(gamma):
        latent, state = markov_head.step_latent(
            prev_tokens,
            hidden_states=None if draft_hidden is None else draft_hidden[:, step],
            state=state,
        )
        latent = latent.to(sl.fold_weight.dtype)
        if _FOLD:
            bias, fold = latent, {"fold": (sl.fold_weight, sl.fold_bias)}
        else:
            bias, fold = torch.addmm(sl.fold_bias, latent, sl.fold_weight.T), {}
        position = MarkovSampleStep.execute(
            base_logits=base_logits[:, step],
            bias=bias,
            temperatures=step_temperatures,
            key=noise.key,
            offset=noise.offset,
            step=step,
            step_dtype=step_dtype,
            logits_out=None if greedy_draft else compact[:, step],
            has_sampled_rows=False if greedy_draft else None,
            softcap=markov_head.step_softcap,
            vocab_limit=None,  # every listed id is below the limit (attach)
            **fold,
        )
        prev_tokens = sl.ids.index_select(0, position)
        prev_tokens = tp_sync.sync(SpecTpSyncSite.DSPARK_DRAFT, prev_tokens)
        draft_tokens.append(prev_tokens)
    draft_tokens = torch.stack(draft_tokens, dim=1)
    if greedy_draft:
        return draft_tokens, None
    if logits_out is None:
        logits_out = torch.empty((bs, gamma, sl.full_width), dtype=step_dtype, device=base_logits.device)
    assert logits_out.shape == (bs, gamma, sl.full_width) and logits_out.dtype == step_dtype
    logits_out.fill_(float("-inf"))
    logits_out.index_copy_(2, sl.ids, compact)
    noise.offset.add_(1)
    return draft_tokens, logits_out
