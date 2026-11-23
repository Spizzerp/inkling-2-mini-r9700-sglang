"""Checks r9700_draft_shortlist.markov_block against SGLang's own draft_markov_block on a synthetic head.
Runs inside the image on the GPU, with no server running; no model weights needed.

A. List = every token below the vocabulary limit, in order. Then list position = token id, the sampling
   noise is the same, and tokens and the full-width step logits must be equal at temperature 0 and at
   temperature 1 (greedy and sampling draft).
B. A real sublist (a quarter of the vocabulary). Temperature 0: against SGLang's path with the unlisted base
   logits pushed to -1e4; tokens equal, listed step logits equal, unlisted ones -inf.
C. The sublist at temperature 1: 64 rows x 400 draws of the first draft token against the softmax of the
   listed step logits (chi-square over the 30 likeliest tokens and the rest), and no token off the list.
"""
import sys
from types import SimpleNamespace

import torch

from sglang.srt import r9700_draft_shortlist as sl
from sglang.srt.speculative.dspark_components import dspark_draft as dd

dev = "cuda"
V, LIMIT, R, H, GAMMA = 6000, 5958, 64, 96, 7
torch.manual_seed(0)


class Head:
    """Duck-typed stand-in for Inkling2MTPHead: folded low-rank step bias from a small recurrent state."""

    step_softcap = 30.0
    step_vocab_limit = LIMIT
    vocab_shard = None

    def __init__(self):
        self._fold_weight = (torch.randn(V, R) * 0.4).to(torch.bfloat16).to(dev)
        self._fold_bias = torch.randn(V).to(torch.bfloat16).to(dev)
        self.emb = torch.randn(V, R).to(torch.bfloat16).to(dev)
        self.proj = (torch.randn(H, R) * 0.2).to(torch.bfloat16).to(dev)
        self._r9700_sl = None

    def init_state(self, hidden):
        return torch.zeros(hidden.shape[0], R, dtype=torch.bfloat16, device=dev)

    def step_latent(self, prev, *, hidden_states, state):
        state = torch.tanh(0.5 * state + self.emb[prev] + hidden_states @ self.proj)
        return state, state

    def step_bias(self, prev, *, hidden_states, state):
        latent, state = self.step_latent(prev, hidden_states=hidden_states, state=state)
        return torch.addmm(self._fold_bias, latent.to(self._fold_weight.dtype), self._fold_weight.T), state

    def step_logits_dtype(self, base_dtype):
        return torch.promote_types(base_dtype, self._fold_weight.dtype)

    def greedy_w2_weight(self):
        return None


def use_list(head, ids):
    ids = ids.to(dev)
    head._r9700_sl = SimpleNamespace(ids=ids, n=int(ids.numel()), full_width=V, weight=None,
                                     fold_weight=head._fold_weight.index_select(0, ids).contiguous(),
                                     fold_bias=head._fold_bias.index_select(0, ids).contiguous())


def run(head, base, anchor, hidden, temps, greedy, offset=0, out=None):
    noise = dd.DraftNoise(key=0x1234567, offset=torch.full((1,), offset, dtype=torch.int64, device=dev))
    sync = SimpleNamespace(sync=lambda site, t: t)
    return dd.draft_markov_block(markov_head=head, base_logits=base, anchor_tokens=anchor, draft_hidden=hidden,
                                 step_temperatures=temps, noise=noise, tp_sync=sync, logits_out=out, greedy_draft=greedy)


head = Head()
ok = True


def report(name, good, detail):
    global ok
    ok &= bool(good)
    print("%-4s %s  %s" % ("ok" if good else "FAIL", name, detail), flush=True)


for bs in (1, 4):
    base = (torch.randn(bs, GAMMA, V) * 3).to(torch.bfloat16).to(dev)
    hidden = torch.randn(bs, GAMMA, H).to(torch.bfloat16).to(dev)
    anchor = torch.randint(0, LIMIT, (bs,), device=dev)
    # A
    all_ids = torch.arange(LIMIT)
    for temp, greedy in ((0.0, True), (0.0, False), (1.0, False)):
        temps = torch.full((bs,), temp, dtype=torch.float32, device=dev)
        head._r9700_sl = None
        t_ref, l_ref = run(head, base, anchor, hidden, temps, greedy)
        use_list(head, all_ids)
        t_new, l_new = run(head, base[..., :LIMIT].contiguous(), anchor, hidden, temps, greedy)
        same_l = True if greedy else bool(torch.equal(l_ref, l_new))
        report("A full list, bs %d, T %.0f, %s" % (bs, temp, "greedy draft" if greedy else "sampling draft"),
               torch.equal(t_ref, t_new) and same_l and (greedy or (l_new is not None and l_new.shape == (bs, GAMMA, V))),
               "tokens equal %s, step logits equal %s" % (bool(torch.equal(t_ref, t_new)), same_l))
    # B
    ids = torch.sort(torch.randperm(LIMIT)[: LIMIT // 4]).values
    listed = torch.zeros(V, dtype=torch.bool)
    listed[ids] = True
    listed = listed.to(dev)
    temps = torch.zeros(bs, dtype=torch.float32, device=dev)
    masked = base.clone()
    masked[..., ~listed] = -1e4
    head._r9700_sl = None
    t_ref, l_ref = run(head, masked, anchor, hidden, temps, False)
    use_list(head, ids)
    stale = torch.full((bs, GAMMA, V), 7.0, dtype=torch.bfloat16, device=dev)  # a used buffer, as in the server
    t_new, l_new = run(head, base.index_select(2, ids.to(dev)).contiguous(), anchor, hidden, temps, False, out=stale)
    on = torch.equal(l_ref[..., listed], l_new[..., listed])
    off = bool(torch.isneginf(l_new[..., ~listed]).all())
    report("B sublist of %d, bs %d, T 0" % (len(ids), bs), torch.equal(t_ref, t_new) and on and off and bool(listed[t_new].all()),
           "tokens equal %s, listed logits equal %s, unlisted -inf %s" % (bool(torch.equal(t_ref, t_new)), on, off))

# C
bs, draws = 64, 400
ids = torch.sort(torch.randperm(LIMIT)[: LIMIT // 4]).values
use_list(head, ids)
row = (torch.randn(1, GAMMA, V) * 3).to(torch.bfloat16)
base = row.expand(bs, GAMMA, V).contiguous().to(dev)
hid = torch.randn(1, GAMMA, H).to(torch.bfloat16).expand(bs, GAMMA, H).contiguous().to(dev)
anchor = torch.full((bs,), 17, device=dev)
temps = torch.ones(bs, dtype=torch.float32, device=dev)
compact = base.index_select(2, ids.to(dev)).contiguous()
counts = torch.zeros(V, dtype=torch.float64, device=dev)
for d in range(draws):
    t, l = run(head, compact, anchor, hid, temps, False, offset=d)
    counts.index_add_(0, t[:, 0], torch.ones(bs, dtype=torch.float64, device=dev))
p = torch.softmax(l[0, 0].double(), 0)
n = bs * draws
top = torch.topk(p, 30).indices
exp = torch.cat([p[top], (1 - p[top].sum()).view(1)]) * n
obs = torch.cat([counts[top], (n - counts[top].sum()).view(1)])
chi2 = float(((obs - exp) ** 2 / exp).sum())
listed = torch.zeros(V, dtype=torch.bool, device=dev)
listed[ids.to(dev)] = True
off_list = float(counts[~listed].sum())
# 30 degrees of freedom: mean 30, 99.9th percentile 59.7
report("C sublist, T 1, %d draws of the first token" % n, chi2 < 59.7 and off_list == 0,
       "chi-square %.1f on 30 degrees of freedom (limit 59.7), draws off the list %d, likeliest token p %.3f seen %.3f" % (
           chi2, off_list, float(p[top[0]]), float(counts[top[0]] / n)))
print("ALL OK" if ok else "FAILURES")
sys.exit(0 if ok else 1)
