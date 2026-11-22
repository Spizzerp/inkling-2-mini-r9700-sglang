"""Adaptive verify length for DSpark draft mode on the Radeon R9700 (installed by patch_adaptive_verify.py).

Every draft pass verifies 8 rows (anchor + 7 drafts) and every row reads its own 8 experts in each of the
48 layer visits, so a verified row costs about 1.5 ms of a 34 ms pass. On prose about half of all passes
accept no draft. This module decides, per request and per pass, how many rows are worth verifying:

  * rows past that length are routed with the request's anchor row's router logits, so they use experts
    that are read anyway (a row-source buffer read by the top-k routing kernel inside the captured
    verify graph: no extra launch);
  * the accept step is capped at that length (SGLang's own cutoff_verify_lens), so nothing from a masked
    row is ever committed: it is treated exactly like a row after a rejected draft;
  * the length comes from an exponentially weighted estimate of the chance h that the next draft is
    accepted given the previous one was, which is roughly constant along a block (0.45-0.5 on prose,
    0.7-0.85 on code): expected tokens with length L are (1 - h^L) / (1 - h), the pass costs COST_MS[L-1],
    and the length with the best ratio is looked up in a table. Only drafts that were actually checked
    count, so capping does not bias the estimate.

Everything stays on the GPU (two tiny launches per pass) and nothing waits for a result, unlike SGLANG_RAGGED_VERIFY_MODE=cap-accept,
whose host-side planning idles the GPU for about 1.1 ms per pass on this machine.
Batches of more than one request verify everything (the cost table is for one request).

Environment: R9700_VERIFY_ADAPT=1 enables; R9700_VERIFY_FORCE_LEN=N pins the length (calibration);
R9700_VERIFY_COST_MS="a,b,...": ms per pass at lengths 1..8; R9700_VERIFY_DECAY (default 0.93);
R9700_VERIFY_LOG=N prints the estimate every N passes (forces a sync: debugging only).
"""
import os
from types import SimpleNamespace

import torch

ENABLED = os.environ.get("R9700_VERIFY_ADAPT") == "1"
_FORCE = int(os.environ.get("R9700_VERIFY_FORCE_LEN", "0"))
_DECAY = float(os.environ.get("R9700_VERIFY_DECAY", "0.93"))
_LOG = int(os.environ.get("R9700_VERIFY_LOG", "0"))
# ms per pass with 1..8 rows verified, one request, measured 22 Nov 2026 (see tune/fuse/README.md)
_COST_MS = [float(x) for x in os.environ.get("R9700_VERIFY_COST_MS", "22.8,25.1,26.9,28.6,30.0,31.3,32.4,33.6").split(",")]
_BUCKETS = 64
# Every request starts at h = 0.8, which means full length: short code and tool-call answers must not
# pay for a warm-up; a prose request comes down to 3-4 rows within about ten passes.
_PRIOR_ACCEPTED, _PRIOR_TESTED = 4.0, 5.0
_MAX_ROWS = 4096
_STRIDE = 8  # anchor + 7 drafts; the kernels below are written for it

_src = None
_state = None
_routing_verify = False

if ENABLED:
    import triton
    import triton.language as tl

    @triton.jit
    def _begin_kernel(Idx, LensSlot, LensCur, LensCur32, EpiLens, Src, force, STRIDE: tl.constexpr, HAS_EPI: tl.constexpr):
        r = tl.program_id(0)
        slot = tl.load(Idx + r).to(tl.int64)
        length = tl.load(LensSlot + slot)
        length = tl.where(force > 0, force, length).to(tl.int64)
        tl.store(LensCur + r, length)
        tl.store(LensCur32 + r, length.to(tl.int32))
        if HAS_EPI:
            tl.store(EpiLens + r, length)
        pos = tl.arange(0, STRIDE)
        base = (r * STRIDE).to(tl.int64)
        tl.store(Src + base + pos, tl.where(pos < length, base + pos, base))

    @triton.jit
    def _observe_kernel(Idx, Correct, LensCur, Accepted, Tested, Table, LensSlot, decay, NB: tl.constexpr):
        r = tl.program_id(0)
        slot = tl.load(Idx + r).to(tl.int64)
        a = tl.maximum(tl.load(Correct + r).to(tl.float32), 0.0)
        length = tl.load(LensCur + r).to(tl.float32)
        t = tl.minimum(a + 1.0, length - 1.0)
        accepted = tl.load(Accepted + slot) * decay + a
        tested = tl.load(Tested + slot) * decay + t
        bucket = (accepted / tl.maximum(tested, 0.001) * NB).to(tl.int64)
        bucket = tl.minimum(tl.maximum(bucket, 0), NB)
        nxt = tl.load(Table + bucket)
        tl.store(Accepted + slot, accepted)
        tl.store(Tested + slot, tested)
        tl.store(LensSlot + slot, nxt)


def verify_src():
    """The row-source buffer; must exist before the verify graphs are captured."""
    global _src
    if ENABLED and _src is None:
        _src = torch.arange(_MAX_ROWS, dtype=torch.int64, device="cuda")
    return _src


def routing_verify(on):
    """Set by the model around its MoE call while a target-verify forward is being built."""
    global _routing_verify
    _routing_verify = bool(on)


def route_src(rows):
    """For the top-k routing kernel: the row-source buffer, or None outside a verify forward."""
    if not _routing_verify or _src is None or rows > _MAX_ROWS or rows % _STRIDE:
        return None
    return _src


def best_len_table(stride):
    table = []
    for b in range(_BUCKETS + 1):
        h = min(b / _BUCKETS, 0.999)
        rates = {L: (1 - h**L) / (1 - h) / _COST_MS[L - 1] for L in range(2, stride + 1)}
        top = max(rates.values())
        # code speed first: on a near tie take the longer length
        table.append(max(L for L, r in rates.items() if r >= 0.99 * top))
    return table


class _State:
    def __init__(self, slots, device):
        self.slots = slots
        self.table = torch.tensor(best_len_table(_STRIDE), dtype=torch.int64, device=device)
        self.start_len = int(self.table[int(_PRIOR_ACCEPTED / _PRIOR_TESTED * _BUCKETS)])
        self.accepted = torch.full((slots,), _PRIOR_ACCEPTED, dtype=torch.float32, device=device)
        self.tested = torch.full((slots,), _PRIOR_TESTED, dtype=torch.float32, device=device)
        self.lens_slot = torch.full((slots,), self.start_len, dtype=torch.int64, device=device)
        self.lens_cur = torch.full((_MAX_ROWS // _STRIDE,), _STRIDE, dtype=torch.int64, device=device)
        self.lens_cur32 = torch.full((_MAX_ROWS // _STRIDE,), _STRIDE, dtype=torch.int32, device=device)
        self.identity = torch.arange(_MAX_ROWS, dtype=torch.int64, device=device)
        self.identity_bs = None
        self.cur = None  # (req_pool_indices, bs) of the pass being verified
        self.passes = 0


def begin_pass(req_pool_indices, bs, stride, slots, epilogue_lens=None):
    """Before the verify graph (and after the epilogue's own per-step setup): choose this pass's verify
    lengths and point the row-source buffer accordingly. Batches of several requests verify everything."""
    global _state
    if _src is None or stride != _STRIDE or bs * stride > _MAX_ROWS:
        return
    if _state is None or _state.slots < slots:
        _state = _State(slots, _src.device)
    st = _state
    if bs != 1 and not _FORCE:
        if st.identity_bs != bs:
            _src[: bs * stride].copy_(st.identity[: bs * stride])
            st.identity_bs = bs
        st.cur = None
        return
    st.identity_bs = None
    st.cur = (req_pool_indices, bs)
    _begin_kernel[(bs,)](
        req_pool_indices, st.lens_slot, st.lens_cur, st.lens_cur32, epilogue_lens, _src,
        max(0, min(_FORCE, _STRIDE)), STRIDE=_STRIDE, HAS_EPI=epilogue_lens is not None,
    )


def cutoff_layout():
    """For SGLang's eager accept: an object carrying this pass's cutoff lengths, or None."""
    st = _state
    if st is None or st.cur is None:
        return None
    return SimpleNamespace(verify_lens=st.lens_cur32[: st.cur[1]])


def observe(correct_len):
    """After the accept: correct_len is the number of accepted drafts per request (already capped)."""
    st = _state
    if st is None or st.cur is None:
        return
    idx, bs = st.cur
    st.cur = None
    if _FORCE:
        return
    _observe_kernel[(bs,)](
        idx, correct_len, st.lens_cur, st.accepted, st.tested, st.table, st.lens_slot, _DECAY, NB=_BUCKETS
    )
    if _LOG:
        st.passes += 1
        if st.passes % _LOG == 0:
            i = idx[:bs].to(torch.int64)
            print(f"R9700 adaptive verify: len {st.lens_cur[:bs].tolist()} accepted {correct_len[:bs].tolist()} "
                  f"h {(st.accepted[i] / st.tested[i]).tolist()} next {st.lens_slot[i].tolist()}", flush=True)


def reset(req_pool_indices):
    """A request starts (prefill): forget the previous user of its slot."""
    st = _state
    if st is None:
        return
    idx = req_pool_indices.view(-1).to(torch.int64)
    st.accepted.index_fill_(0, idx, _PRIOR_ACCEPTED)
    st.tested.index_fill_(0, idx, _PRIOR_TESTED)
    st.lens_slot.index_fill_(0, idx, st.start_len)
