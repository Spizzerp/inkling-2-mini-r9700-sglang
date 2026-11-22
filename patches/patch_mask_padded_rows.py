"""Build-time patch: keep padded rows at zero through every Inkling 2 layer during prefill (R9700, 21 Nov 2026).

Second half of the padded-row fix (the first is patch_zero_padded_attention.py). A full prefill graph runs at a
padded token bucket. Several kernels write only the real tokens' rows of their torch.empty outputs, so padded
rows hold leftovers of other graphs; when a leftover is NaN or Inf, the real rows of the batch turn NaN too and
the request generates garbage until its token limit. Zeroing the attention outputs removed most cases but not
all (2 runaway answers in 400, both on prompts one or two tokens short of a bucket).

SGLang hands every full prefill graph the live token count (forward_batch.num_token_non_padded) for exactly
this, and other models mask their padded rows with it; the Inkling 2 code does not. This masks the layer's
three intermediate results (attention after its short convolution, the MLP input, the MLP output) with
torch.where, which also clears NaN. Prefill only: decode and draft-verify batches are not token-padded.
"""
import pathlib

path = pathlib.Path("/sgl-workspace/sglang/python/sglang/srt/models/inkling2.py")
src = path.read_text()

helper = '''

def _zero_padded_rows(x, forward_batch):
    """x with the rows past the batch's real tokens set to zero, in padded prefill graphs."""
    if not isinstance(x, torch.Tensor) or x.dim() != 2:
        return x
    mode = forward_batch.forward_mode
    if mode.is_decode() or mode.is_target_verify():
        return x
    count = getattr(forward_batch, "num_token_non_padded", None)
    if count is None:
        return x
    keep = torch.arange(x.shape[0], device=x.device, dtype=torch.int32) < count
    return torch.where(keep.unsqueeze(-1), x, torch.zeros((), dtype=x.dtype, device=x.device))


class Inkling2DecoderLayer(nn.Module):'''
anchor = "\n\nclass Inkling2DecoderLayer(nn.Module):"
assert src.count(anchor) == 1
src = src.replace(anchor, helper)

old = "        attention = self.attn_sconv(attention, positions, forward_batch)\n"
assert src.count(old) == 1
src = src.replace(old, old + "        attention = _zero_padded_rows(attention, forward_batch)\n")

old = "            mlp_input = self.mlp_norm(mlp_input)\n        routed = (\n"
assert src.count(old) == 1
src = src.replace(old, "            mlp_input = self.mlp_norm(mlp_input)\n        mlp_input = _zero_padded_rows(mlp_input, forward_batch)\n        routed = (\n")

old = "        mlp = self.mlp_sconv(mlp, positions, forward_batch)\n        state.finalize_layer(self.layer_id, attention, mlp)\n"
assert src.count(old) == 1
src = src.replace(old, "        mlp = self.mlp_sconv(mlp, positions, forward_batch)\n        mlp = _zero_padded_rows(mlp, forward_batch)\n        state.finalize_layer(self.layer_id, attention, mlp)\n")
path.write_text(src)
print("patched", path)
