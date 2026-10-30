"""Turn token-exact rollout traces into SFT rows that train under KV-Streams eviction.

Each trace of an eval/rollout trace stream becomes one row: an episode has one trace,
or one per segment when it is a continuous episode (each segment is its own engine
session with fresh positions, so its own row). A row holds the chronological token ids
exactly as sampled (no re-rendering), a loss mask on every token the model sampled,
and `kv_death` - the position from which each token's KV is evicted - for a turn window
r{window}-{stride} drawn per row. The eviction
schedule comes from replaying the episode's calls through the server's own
`SessionTable` / `TurnWindow` plus the served model's eviction plan (`--model`; e.g.
Qwen3.5's empty think blocks), with an engine that returns the recorded replies, so
the trainer sees exactly the context state the engine would build, stale KV
included. Feed the output to `sft` with `[data] type = "tokens"`.

    python traces_to_sft.py stream.jsonl out.jsonl --model Qwen/Qwen3.5-4B --windows 8,16,32,50 --copies 1
"""

import argparse
import asyncio
import json
import random
import re
from collections.abc import Callable

from prime_rl.inference.sglang.launcher import hf_model_type
from prime_rl.inference.sglang.model_evictions import MODEL_TYPES, PLANS, build
from prime_rl.inference.sglang.server import SessionTable, TurnWindow


class ReplayEngine:
    """Answers each call with the reply the trace recorded."""

    def __init__(self, outputs: list[list[int]]):
        self.outputs, self.calls = outputs, 0

    async def generate(self, input_ids, sampling_params, *, kv_eviction=None, **kwargs):
        out = self.outputs[self.calls]
        self.calls += 1
        state = {"cache_state": {"state_id": f"s{self.calls}"}} if kv_eviction is not None else None
        return {
            "output_ids": list(out),
            "meta_info": {"id": "replay", "finish_reason": {"type": "stop"}, "kv_eviction": state},
        }

    async def open_session(self, session_id, timeout):
        return True

    async def close_session(self, session_id):
        pass

    def abort_request(self, rid):
        pass


def _list(value):
    return json.loads(value) if isinstance(value, str) else list(value or [])


def trace_calls(
    trace: dict, keep_reply: Callable[[str], bool] = lambda content: True
) -> tuple[list[int], list[bool], list[tuple[int, list[int]]], int]:
    """(chronological tokens, loss mask, [(prompt end, reply tokens)], replies left out
    of the loss) of one trace. Every reply stays in the context; `keep_reply` decides
    which replies are targets."""
    stream, loss, calls, dropped = [], [], [], 0
    for node in trace["nodes"]:
        ids, mask = _list(node["token_ids"]), _list(node.get("mask"))
        sampled = bool(node.get("sampled")) and True in mask
        if sampled:
            first = mask.index(True)
            if not all(mask[first:]):
                # The replay treats the rest of the node as the engine's output: tokens the
                # engine did not sample there would shift every eviction point after it.
                raise ValueError("a sampled node has unsampled tokens after its reply")
            calls.append((len(stream) + first, ids[first:]))
            target = keep_reply(str((node.get("message") or {}).get("content") or ""))
            dropped += not target
            loss += [target and bool(m) for m in mask]
        else:
            loss += [False] * len(ids)
        stream += ids
    return stream, loss, calls, dropped


def segments(episode: dict) -> list[dict]:
    """The episode's traces in play order: one per segment of a continuous episode."""
    return sorted(episode["traces"], key=lambda trace: (trace.get("info") or {}).get("segment", 0))


def model_plans(model: str) -> list:
    """The eviction plan the server adds for this model (by its ``model_type``), if any."""
    plan = MODEL_TYPES.get(hf_model_type(model))
    if plan is None:
        return []
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model)
    ids = {f"{name}_ids": tokenizer.encode(text, add_special_tokens=False) for name, text in PLANS[plan].MARKERS.items()}
    return [build(plan, ids)]


def evict(
    stream: list[int],
    calls: list[tuple[int, list[int]]],
    window: int,
    stride: int,
    message_end_id: int,
    plans: list = (),
) -> list[int]:
    """kv_death for the stream under a turn window and model plans, as the engine would evict."""
    table = SessionTable(
        ReplayEngine([reply for _, reply in calls]),
        timeout=60,
        eviction=TurnWindow(window, stride, message_end_id),
        overlap_schedule=False,
        plans=plans,
    )
    death = [-1] * len(stream)

    async def replay():
        for prompt_end, _ in calls:
            ret = await table.generate("trace", stream[:prompt_end], {})
            if table.stats["reopened"] or table.stats["failed_calls"]:
                raise ValueError("the trace does not extend its own session (rewritten history)")
            for start, end in (ret.get("eviction") or {}).get("spans", []):
                for j in range(start, end):
                    if death[j] < 0:  # first death wins
                        death[j] = ret["eviction"]["at"]

    asyncio.run(replay())
    return death


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stream", help="trace stream JSONL (one episode per line)")
    ap.add_argument("out")
    ap.add_argument("--model", required=True, help="the served model: its eviction plan applies as on the server")
    ap.add_argument("--windows", default="8,16,32,50", help="turn windows to draw from, per row")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--copies", type=int, default=1, help="rows per trace, each with its own window")
    ap.add_argument("--message-end-id", type=int, default=248046, help="<|im_end|> for Qwen")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pack-seq-len", type=int, default=65536, help="the SFT data.seq_len, to count packed rows")
    ap.add_argument(
        "--only-action-replies",
        action="store_true",
        help="a data-curation choice, not the default: learn only replies with a complete <action>...</action> "
        "(Crafter/Craftax format); the others (summaries, malformed replies) stay in the context, out of the loss",
    )
    args = ap.parse_args()
    rng = random.Random(args.seed)
    windows = [int(w) for w in args.windows.split(",")]
    plans = model_plans(args.model)
    rows = loss_tokens = evicted_loss_tokens = skipped = dropped = 0
    # The env's own action pattern: a reply it would play (not repeat the last action for).
    action = re.compile(r"<action>\s*(.*?)\s*</action>", re.DOTALL)
    keep_reply = (lambda content: bool(action.search(content))) if args.only_action_replies else (lambda content: True)
    packs, pack_fill = 0, 0  # greedy packing as the SFT dataloader does (CatDataset)
    with open(args.stream) as f, open(args.out, "w") as out:
        for line in f:
            for trace in segments(json.loads(line)):
                stream, loss, calls, left_out = trace_calls(trace, keep_reply)
                dropped += left_out * args.copies
                segment = (trace.get("info") or {}).get("segment", 0)
                for _ in range(args.copies):
                    window = rng.choice(windows)
                    try:
                        death = evict(stream, calls, window, args.stride, args.message_end_id, plans)
                    except ValueError:
                        skipped += 1
                        continue
                    first = min((d for d in death if d >= 0), default=None)
                    loss_tokens += sum(loss)
                    evicted_loss_tokens += sum(1 for i, m in enumerate(loss) if m and first is not None and i >= first)
                    row = {"input_ids": stream, "loss_mask": loss, "kv_death": death, "window": window}
                    out.write(json.dumps(row | {"trace": trace.get("id"), "segment": segment}) + "\n")
                    rows += 1
                    length = min(len(stream) - 1, args.pack_seq_len)
                    if pack_fill and pack_fill + length > args.pack_seq_len:
                        packs, pack_fill = packs + 1, 0
                    pack_fill += length
                    if pack_fill >= args.pack_seq_len:
                        packs, pack_fill = packs + 1, 0
    packs += pack_fill > 0
    print(f"eviction plans: turn window r{args.windows}-{args.stride}" + "".join(f", {type(p).__module__}" for p in plans))
    print(f"{packs} packed rows of up to {args.pack_seq_len} tokens: one epoch = ceil({packs} / batch_size) steps")
    print(
        f"{rows} rows ({skipped} skipped); {dropped} replies left out of the loss; {loss_tokens} loss tokens, "
        f"{100 * evicted_loss_tokens / max(1, loss_tokens):.0f}% of them after the first eviction"
    )


if __name__ == "__main__":
    main()
