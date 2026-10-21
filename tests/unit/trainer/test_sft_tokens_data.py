import importlib.util
from pathlib import Path

from datasets import Dataset

from prime_rl.trainer.sft.data import CatDataset, TokensDataset, cat_collate


def _rows():
    return [
        {"input_ids": [1, 2, 3, 4, 5], "loss_mask": [False, False, True, True, True], "kv_death": [-1, 4, 4, -1, -1]},
        {"input_ids": [6, 7, 8], "loss_mask": [False, True, True], "kv_death": [-1, -1, -1]},
    ]


def test_tokens_samples_shift_and_keep_eviction_positions():
    dataset = TokensDataset(Dataset.from_list(_rows()), shuffle=False, seq_len=16)
    first = dataset._process(_rows()[0])
    assert first["input_ids"] == [1, 2, 3, 4] and first["target_ids"] == [2, 3, 4, 5]
    assert first["loss_mask"] == [False, True, True, True]
    assert first["kv_death"] == [-1, 4, 4, -1]  # each input token keeps its own eviction position
    assert first["position_ids"] == [0, 1, 2, 3] and first["seq_lens"] == [4]


def test_packing_carries_kv_death_per_sequence():
    dataset = TokensDataset(Dataset.from_list(_rows()), shuffle=False, seq_len=16, max_epochs=1)
    (pack,) = list(CatDataset(dataset, seq_len=16, pad_to_multiple_of=1))
    assert pack["seq_lens"] == [4, 2]
    assert pack["kv_death"] == [-1, 4, 4, -1, -1, -1]  # positions stay per sequence: no offsets
    batch = cat_collate([pack])
    assert batch["kv_death"].tolist() == [[-1, 4, 4, -1, -1, -1]]


def test_packs_without_evictions_carry_none():
    rows = [{"input_ids": [1, 2, 3], "loss_mask": [False, True, True]}]
    dataset = TokensDataset(Dataset.from_list(rows), shuffle=False, seq_len=8, max_epochs=1)
    (pack,) = list(CatDataset(dataset, seq_len=8, pad_to_multiple_of=1))
    assert pack["kv_death"] is None and cat_collate([pack])["kv_death"] is None


def _converter():
    path = Path(__file__).parents[3] / "scripts" / "kv_streams" / "traces_to_sft.py"
    spec = importlib.util.spec_from_file_location("traces_to_sft", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_converter_replays_the_engines_eviction_schedule():
    # The r2-1 example of test_sglang_sessions: system, then turns of
    # user + generation prompt + a reply ending in the message end id (7).
    E = 7
    system, user, gen, bridge = [1, 2, E], [3, E], [4], [5]
    nodes = [{"token_ids": system + user, "sampled": False}]
    for turn in range(4):
        if turn:
            nodes.append({"token_ids": bridge + user, "sampled": False})
        nodes.append({"token_ids": gen + [90, E], "sampled": True, "mask": [False, True, True]})
    converter = _converter()
    stream, loss, calls, dropped = converter.trace_calls({"nodes": nodes})
    assert sum(loss) == 8 and len(calls) == 4 and dropped == 0
    death = converter.evict(stream, calls, window=2, stride=1, message_end_id=E)
    # Turn 1 is chronological [3, 8), turn 2 [8, 14); each dies from the previous call's
    # last sampled token on (the engine's eviction records: at 13, then 19).
    assert death[:3] == [-1] * 3
    assert death[3:8] == [13] * 5 and death[8:14] == [19] * 6
    assert all(d == -1 for d in death[14:])



def test_converter_leaves_malformed_replies_out_of_the_loss():
    nodes = [
        {"token_ids": [1, 2, 7], "sampled": False},
        {"token_ids": [4, 90, 7], "sampled": True, "mask": [False, True, True], "message": {"content": "<action>do</action>"}},
        {"token_ids": [5, 3, 7], "sampled": False},
        {"token_ids": [4, 27, 7], "sampled": True, "mask": [False, True, True], "message": {"content": "<"}},
    ]
    converter = _converter()
    stream, loss, calls, dropped = converter.trace_calls({"nodes": nodes}, lambda content: "<action>" in content)
    assert dropped == 1 and len(calls) == 2  # still replayed and kept in the context
    assert loss == [False, False, False, False, True, True, False, False, False, False, False, False]
