"""Attention masks for KV-Streams rollouts (per-token ``kv_death``).

A KV-Streams engine evicts tokens from a sequence's KV cache mid-rollout, so
later tokens were computed without attending to them. The trainer replays the
rollout as one chronological stream and reproduces that with an attention mask:
query ``i`` attends to key ``j`` iff both are in the same packed sequence,
``j <= i``, and ``j`` was still resident when ``i`` was computed, i.e.
``kv_death[j] == -1`` (never evicted) or ``pos(i) < kv_death[j]``, where
positions count from the start of the sequence.

Only full-attention layers take the mask; linear-attention (DeltaNet) layers
keep their recurrent state over the whole stream, as the engine does.
"""

import torch
from torch import Tensor
from torch.nn.attention.flex_attention import BlockMask, create_block_mask, flex_attention

# Flex attention is only efficient compiled; eager flex materializes the score matrix.
_compiled_flex_attention = torch.compile(flex_attention, dynamic=True)


def _sequence_ids_and_positions(seq_lens: Tensor, total: int) -> tuple[Tensor, Tensor]:
    seq_lens = seq_lens.to(torch.int64)
    if int(seq_lens.sum().item()) != total:
        raise ValueError(f"seq_lens sum {int(seq_lens.sum().item())} != {total} tokens")
    seq_ids = torch.repeat_interleave(torch.arange(seq_lens.numel(), device=seq_lens.device), seq_lens)
    starts = torch.cumsum(seq_lens, 0) - seq_lens
    positions = torch.arange(total, device=seq_lens.device) - starts[seq_ids]
    return seq_ids, positions


def kv_death_mask_mod(kv_death: Tensor, seq_lens: Tensor):
    """The flex ``mask_mod`` of a packed micro batch. ``kv_death``: [tokens] (positions
    within each sequence, -1 = never evicted); ``seq_lens``: [sequences]."""
    death = kv_death.to(device=seq_lens.device, dtype=torch.int64).reshape(-1)
    seq_ids, positions = _sequence_ids_and_positions(seq_lens, death.numel())

    def mask_mod(b, h, q_idx, kv_idx):
        same_sequence = seq_ids[q_idx] == seq_ids[kv_idx]
        resident = (death[kv_idx] < 0) | (positions[q_idx] < death[kv_idx])
        return same_sequence & (kv_idx <= q_idx) & resident

    return mask_mod


def build_kv_death_block_mask(kv_death: Tensor, seq_lens: Tensor) -> BlockMask:
    """The flex attention block mask of a packed micro batch (see ``kv_death_mask_mod``)."""
    total = kv_death.numel()
    mask_mod = kv_death_mask_mod(kv_death, seq_lens)
    return create_block_mask(mask_mod, None, None, total, total, device=seq_lens.device, _compile=seq_lens.is_cuda)


def evicted_context(kv_death: Tensor, seq_lens: Tensor) -> Tensor:
    """[tokens] bool: whether some earlier token of the same sequence was already
    evicted at this position (the token's computation differs from full context)."""
    total = kv_death.numel()
    death = kv_death.to(device=seq_lens.device, dtype=torch.int64).reshape(-1)
    seq_ids, positions = _sequence_ids_and_positions(seq_lens, total)
    big = torch.iinfo(torch.int64).max
    first_death = torch.full((seq_lens.numel(),), big, dtype=torch.int64, device=seq_lens.device)
    first_death.scatter_reduce_(0, seq_ids, torch.where(death >= 0, death, big), reduce="amin")
    return positions >= first_death[seq_ids]


def masked_attention(q: Tensor, k: Tensor, v: Tensor, block_mask: BlockMask, scale: float) -> Tensor:
    """Flex attention over packed [tokens, heads, dim] q/k/v (k/v may have fewer heads: GQA)."""
    out = _compiled_flex_attention(
        q.transpose(0, 1).unsqueeze(0),
        k.transpose(0, 1).unsqueeze(0),
        v.transpose(0, 1).unsqueeze(0),
        block_mask=block_mask,
        scale=scale,
        enable_gqa=q.shape[1] != k.shape[1],
    )
    return out.squeeze(0).transpose(0, 1)
