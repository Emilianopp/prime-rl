import importlib.util
import json
import sys
from pathlib import Path

from datasets import Dataset

from prime_rl.inference.sglang.model_evictions import qwen35
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


def _r2_nodes(turns: int = 4) -> list[dict]:
    # The r2-1 example of test_sglang_sessions: system, then turns of
    # user + generation prompt + a reply ending in the message end id (7).
    E = 7
    system, user, gen, bridge = [1, 2, E], [3, E], [4], [5]
    nodes = [{"token_ids": system + user, "sampled": False}]
    for turn in range(turns):
        if turn:
            nodes.append({"token_ids": bridge + user, "sampled": False})
        nodes.append({"token_ids": gen + [90, E], "sampled": True, "mask": [False, True, True]})
    return nodes


def test_converter_replays_the_engines_eviction_schedule():
    E = 7
    nodes = _r2_nodes()
    converter = _converter()
    stream, loss, calls, dropped = converter.trace_calls({"nodes": nodes})
    assert sum(loss) == 8 and len(calls) == 4 and dropped == 0
    death = converter.evict(stream, calls, window=2, stride=1, message_end_id=E)
    # Turn 1 is chronological [3, 8), turn 2 [8, 14); each dies from the previous call's
    # last sampled token on (the engine's eviction records: at 13, then 19).
    assert death[:3] == [-1] * 3
    assert death[3:8] == [13] * 5 and death[8:14] == [19] * 6
    assert all(d == -1 for d in death[14:])


def test_converter_writes_a_row_per_segment(tmp_path, monkeypatch):
    # A continuous episode streams one trace per segment, each its own engine session;
    # every segment must become a row, in play order, with its own eviction schedule.
    later, first = _r2_nodes(turns=3), _r2_nodes(turns=4)
    episode = {"traces": [{"id": "b", "info": {"segment": 1}, "nodes": later}, {"id": "a", "info": {"segment": 0}, "nodes": first}]}
    stream, out = tmp_path / "stream.jsonl", tmp_path / "rows.jsonl"
    stream.write_text(json.dumps(episode) + "\n")
    converter = _converter()
    monkeypatch.setattr(converter, "model_plans", lambda model: [])
    argv = ["traces_to_sft.py", str(stream), str(out), "--model", "m", "--windows", "2", "--message-end-id", "7"]
    monkeypatch.setattr(sys, "argv", argv)
    converter.main()
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert [(row["segment"], row["trace"]) for row in rows] == [(0, "a"), (1, "b")]
    for row, nodes in zip(rows, [first, later]):
        stream_ids, _, calls, _ = converter.trace_calls({"nodes": nodes})
        assert row["input_ids"] == stream_ids
        # Every sampled token is a target, although no reply holds an <action>.
        assert sum(row["loss_mask"]) == 2 * sum(bool(node.get("sampled")) for node in nodes)
        assert row["kv_death"] == converter.evict(stream_ids, calls, window=2, stride=1, message_end_id=7)
    assert any(d >= 0 for d in rows[1]["kv_death"])  # three turns past an r2 window: the later segment evicts too



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


def test_converter_applies_the_models_eviction_plan():
    # As the server does for Qwen3.5: a reply's empty think block dies once a later
    # user query arrives, long before the turn window would drop the turn.
    E, think = 7, [60, 62, 61, 62]
    nodes = [{"token_ids": [1, 2, E, 3, E], "sampled": False}]
    for turn in range(3):
        if turn:
            nodes.append({"token_ids": [5, 3, E], "sampled": False})
        nodes.append({"token_ids": [4, *think, 90, E], "sampled": True, "mask": [False] + [True] * 6})
    converter = _converter()
    stream, _, calls, _ = converter.trace_calls({"nodes": nodes})
    assert converter.evict(stream, calls, window=100, stride=1, message_end_id=E) == [-1] * len(stream)
    plan = qwen35.Plan(empty_think_ids=think, user_ids=[3], tool_response_ids=[66])
    death = converter.evict(stream, calls, window=100, stride=1, message_end_id=E, plans=[plan])
    # Reply 1's block is [6, 10), reply 2's [16, 20); calls 2 and 3 start at 15 and 25.
    assert len(set(death[6:10])) == 1 and 10 <= death[6] < 15
    assert len(set(death[16:20])) == 1 and 20 <= death[16] < 25
    assert all(d == -1 for j, d in enumerate(death) if not (6 <= j < 10 or 16 <= j < 20))  # reply 3's block: no later query
