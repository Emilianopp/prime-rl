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

# Flex attention is only efficient compiled; eager flex materializes the score matrix,
# and an uncompiled create_block_mask materializes the full [tokens, tokens] mask.
_compiled_flex_attention = torch.compile(flex_attention, dynamic=True)
_compiled_create_block_mask = torch.compile(create_block_mask, dynamic=True)


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
    build = _compiled_create_block_mask if seq_lens.is_cuda else create_block_mask
    return build(mask_mod, None, None, total, total, device=seq_lens.device)


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


# Flex's default tiles are a poor fit for some (GPU, head_dim) pairs, and inside
# prime-rl's compiled transformer blocks a mode="max-autotune" on the flex call is
# ignored (the outer compile's settings win), so tuned tiles are passed explicitly.
# Keys: (compute capability major, head_dim). Measured with max-autotune at
# Qwen3.5-4B shapes (16 q / 4 kv heads, ~33k packed tokens, bf16) on A100: forward +
# backward 733 ms with default tiles, 53 ms with these, flash varlen 42 ms.
_KERNEL_OPTIONS: dict[tuple[int, int], dict] = {
    (8, 256): {
        "fwd_BLOCK_M": 128,
        "fwd_BLOCK_N": 128,
        "fwd_num_stages": 2,
        "fwd_num_warps": 8,
        "bwd_BLOCK_M1": 32,
        "bwd_BLOCK_N1": 32,
        "bwd_BLOCK_M2": 32,
        "bwd_BLOCK_N2": 32,
        "bwd_num_stages": 1,
        "bwd_num_warps": 4,
    },
}


def _kernel_options(q: Tensor) -> dict | None:
    if not q.is_cuda:
        return None
    return _KERNEL_OPTIONS.get((torch.cuda.get_device_capability(q.device)[0], q.shape[-1]))


def masked_attention(q: Tensor, k: Tensor, v: Tensor, block_mask: BlockMask, scale: float) -> Tensor:
    """Flex attention over packed [tokens, heads, dim] q/k/v (k/v may have fewer heads: GQA)."""
    out = _compiled_flex_attention(
        q.transpose(0, 1).unsqueeze(0),
        k.transpose(0, 1).unsqueeze(0),
        v.transpose(0, 1).unsqueeze(0),
        block_mask=block_mask,
        scale=scale,
        enable_gqa=q.shape[1] != k.shape[1],
        kernel_options=_kernel_options(q),
    )
    return out.squeeze(0).transpose(0, 1)
