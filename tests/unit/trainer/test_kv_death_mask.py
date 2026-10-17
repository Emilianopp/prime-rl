"""The trainer's KV-Streams attention mask (kv_death) and its flex attention path."""

import pytest
import torch
from torch.nn.attention.flex_attention import create_mask

from prime_rl.trainer.models.layers.kv_death import (
    build_kv_death_block_mask,
    evicted_context,
    kv_death_mask_mod,
    masked_attention,
)


def reference_mask(kv_death: list[int], seq_lens: list[int]) -> torch.Tensor:
    """Brute force: query i sees key j iff same sequence, j <= i, and j resident at i."""
    total = len(kv_death)
    allowed = torch.zeros(total, total, dtype=torch.bool)
    start = 0
    for length in seq_lens:
        for i in range(start, start + length):
            for j in range(start, i + 1):
                death = kv_death[j]
                allowed[i, j] = death < 0 or (i - start) < death
        start += length
    return allowed


# Two packed sequences and a padding segment. Sequence 0 evicts positions 2..4 from
# position 7 on (the turn-window record [2, 5) at 7); sequence 1 evicts its position 1
# from position 3 on.
KV_DEATH = [-1, -1, 7, 7, 7, -1, -1, -1, -1, -1] + [-1, 3, -1, -1, -1] + [-1, -1]
SEQ_LENS = [10, 5, 2]


def test_mask_mod_matches_the_reference():
    kv_death, seq_lens = torch.tensor(KV_DEATH), torch.tensor(SEQ_LENS)
    total = len(KV_DEATH)
    dense = create_mask(kv_death_mask_mod(kv_death, seq_lens), None, None, total, total, device="cpu")
    assert torch.equal(dense[0, 0], reference_mask(KV_DEATH, SEQ_LENS))


def test_all_alive_mask_is_causal_per_sequence():
    kv_death = [-1] * 9
    dense = create_mask(kv_death_mask_mod(torch.tensor(kv_death), torch.tensor([4, 5])), None, None, 9, 9, "cpu")
    expected = torch.block_diag(torch.ones(4, 4).tril(), torch.ones(5, 5).tril()).bool()
    assert torch.equal(dense[0, 0], expected)


def test_evicted_context_marks_positions_after_the_first_eviction():
    marked = evicted_context(torch.tensor(KV_DEATH), torch.tensor(SEQ_LENS))
    expected = [False] * 7 + [True] * 3 + [False] * 3 + [True] * 2 + [False] * 2
    assert marked.tolist() == expected


def test_seq_lens_must_cover_every_token():
    with pytest.raises(ValueError, match="seq_lens sum"):
        kv_death_mask_mod(torch.tensor([-1] * 5), torch.tensor([3]))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="flex attention kernels need a GPU")
def test_flex_attention_matches_masked_sdpa():
    # Qwen3.5 full-attention shapes: 16 query heads, 4 KV heads (GQA), head_dim 256.
    torch.manual_seed(0)
    seq_lens = [300, 180, 32]
    total = sum(seq_lens)
    kv_death = [-1] * total
    for j in range(40, 120):  # sequence 0 evicts [40, 120) from position 200 on
        kv_death[j] = 200
    for j in range(310, 330):  # sequence 1 evicts its [10, 30) from its position 90 on
        kv_death[j] = 90
    q = torch.randn(total, 16, 256, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(total, 4, 256, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(total, 4, 256, device="cuda", dtype=torch.bfloat16)
    death, lens = torch.tensor(kv_death, device="cuda"), torch.tensor(seq_lens, device="cuda")

    out = masked_attention(q, k, v, build_kv_death_block_mask(death, lens), scale=256**-0.5)

    allowed = reference_mask(kv_death, seq_lens).cuda()
    ref = torch.nn.functional.scaled_dot_product_attention(
        q.transpose(0, 1).float(),
        k.transpose(0, 1).repeat_interleave(4, dim=0).float(),
        v.transpose(0, 1).repeat_interleave(4, dim=0).float(),
        attn_mask=allowed,
        scale=256**-0.5,
    ).transpose(0, 1)
    torch.testing.assert_close(out.float(), ref, atol=2e-2, rtol=2e-2)
