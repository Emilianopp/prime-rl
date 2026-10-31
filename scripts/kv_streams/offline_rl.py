"""Off-policy RL on stored rollouts: stand in for the orchestrator and feed an RL trainer
batches built from trace streams, with per-turn credit from a scorer.

The traces keep everything the trainer needs: the chronological token stream, the
sampling logprobs (the importance ratio's denominator) and, for KV-Streams runs, the
engine's `kv_death` marks, so the trainer recomputes each token's logprob under the
exact context it was sampled in. A scorer turns a trace into per-reply rewards; each
reply's sampled tokens get its reward as their advantage, and zero-advantage tokens
leave the RL loss (only scored replies move). Step 1 is on-policy when the traces were
sampled by the trainer's starting weights: its mismatch KL checks the replay.

    python offline_rl.py 'run/*/monitors/file/traces/stream/*.zst' --output-dir <trainer output_dir> \
        --scorer craftax_env.turn_rewards:turn_rewards --model Qwen/Qwen3.5-4B --workers 4 \
        --batch-size 32 --steps 10 --seq-len 131072

Run it beside `torchrun ... prime_rl.trainer.rl.train @ trainer.toml` with
`rollout_transport.type = "filesystem"` and the same output_dir. The feeder starts at
whatever step the trainer first asks for; it writes `offline_rl.jsonl` (per-step reward
counts) into the output dir.
"""

import argparse
import asyncio
import glob
import importlib
import json
import random
import subprocess
import time
from collections import Counter
from pathlib import Path

import msgspec
import verifiers.v1 as vf
from transformers import AutoConfig

from prime_rl.orchestrator.batch import build_bin_cost, prepare_batch
from prime_rl.orchestrator.train_sink import _prune_zero_advantages
from prime_rl.orchestrator.trajectories import trace_to_samples
from prime_rl.transports.batch import FileSystemBatchSender, TrainingSample
from prime_rl.transports.weights.base import FINISHED_MARKER, RECEIVER_READY_MARKER, SENDER_READY_MARKER
from prime_rl.utils.pathing import get_batch_dir, get_broadcast_dir


def load_scorer(spec: str):
    module, name = spec.split(":")
    return getattr(importlib.import_module(module), name)


def episodes(pattern: str):
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise SystemExit(f"no trace streams match {pattern}")
    for path in paths:
        with subprocess.Popen(["zstdcat", path], stdout=subprocess.PIPE, text=True) as proc:
            yield from (json.loads(line) for line in proc.stdout)


def build(args, scorer) -> tuple[list[bytes], list[dict]]:
    """Encoded samples (scored segments with any nonzero credit) and their stats."""
    encoder = msgspec.msgpack.Encoder()
    samples, stats = [], []
    seen = skipped = full_context = 0
    for episode in episodes(args.streams):
        if not episode.get("ok", True):
            continue
        for raw in episode["traces"]:
            seen += 1
            trace = vf.Trace.model_validate(raw)
            if trace.has_error:
                continue
            scores = scorer(trace, **args.scorer_args)
            rewards = {score.node: score.reward for score in scores}
            for i, node in enumerate(trace.nodes):
                if node.sampled and any(node.mask):
                    node.advantages = [float(rewards.get(i, 0.0))] * sum(node.mask)
            for sample in trace_to_samples(trace, env_name=args.env_name):
                full_context += sample.kv_death is None  # nothing evicted (a short segment) or not KV-Streams
                if len(sample.token_ids) > args.seq_len:
                    skipped += 1
                    continue
                sample.temperatures = [args.temperature] * len(sample.token_ids)
                if not _prune_zero_advantages(sample):
                    continue
                reasons = Counter(r for score in scores for r in score.reasons)
                stats.append(
                    {
                        "trace": trace.id,
                        "segment": (trace.info or {}).get("segment", 0),
                        "tokens": len(sample.token_ids),
                        "replies": len(scores),
                        "scored": sum(1 for score in scores if score.reward),
                        "reward": sum(score.reward for score in scores),
                        "reasons": dict(reasons),
                    }
                )
                samples.append(encoder.encode(sample))
    print(
        f"{seen} traces read, {len(samples)} with credit ({full_context} without kv_death), "
        f"{skipped} longer than seq_len {args.seq_len} skipped"
    )
    return samples, stats


def first_requested_step(batch_dir: Path, timeout: float) -> int:
    """The step the trainer asks for first (its resume point)."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        steps = [int(p.parent.name.split("_")[1]) for p in batch_dir.glob("step_*/rank_*.request")]
        if steps:
            return min(steps)
        time.sleep(2)
    raise SystemExit(f"no trainer asked for a batch in {batch_dir} within {timeout:.0f} s")


async def acknowledge_broadcasts(output_dir: Path) -> None:
    """Stand in for the weight consumer: the trainer offers every policy version and waits
    for an acknowledgement before writing it. The weights stay on disk (the last two)."""
    broadcast_dir = get_broadcast_dir(output_dir)
    while True:
        for offered in broadcast_dir.glob(f"step_*/{SENDER_READY_MARKER}"):
            (offered.parent / RECEIVER_READY_MARKER).touch()
        await asyncio.sleep(1)


async def feed(args, samples: list[bytes], stats: list[dict]) -> None:
    decoder = msgspec.msgpack.Decoder(type=TrainingSample)
    bin_cost = build_bin_cost(AutoConfig.from_pretrained(args.model, trust_remote_code=True))
    output_dir = Path(args.output_dir)
    acker = asyncio.create_task(acknowledge_broadcasts(output_dir))
    step = await asyncio.to_thread(first_requested_step, get_batch_dir(output_dir), args.wait_seconds)
    sender = FileSystemBatchSender(output_dir, args.workers, step)
    rng = random.Random(args.seed)
    order: list[int] = []
    log = open(output_dir / "offline_rl.jsonl", "a")
    for n in range(args.steps):
        picked = []
        while len(picked) < args.batch_size:
            if not order:  # a new epoch
                order = list(range(len(samples)))
                rng.shuffle(order)
            picked.append(order.pop())
        batch = [decoder.decode(samples[i]) for i in picked]
        grid = await asyncio.to_thread(prepare_batch, batch, args.seq_len, args.workers, bin_cost, args.pad_to_multiple_of)
        await sender.send(grid)
        reasons = Counter()
        for i in picked:
            reasons.update(stats[i]["reasons"])
        record = {
            "step": step + n,
            "segments": len(picked),
            "tokens": sum(stats[i]["tokens"] for i in picked),
            "replies": sum(stats[i]["replies"] for i in picked),
            "scored_replies": sum(stats[i]["scored"] for i in picked),
            "reward": sum(stats[i]["reward"] for i in picked),
            "reasons": dict(reasons),
        }
        log.write(json.dumps(record) + "\n")
        log.flush()
        print(json.dumps(record), flush=True)
    # Keep acknowledging until the trainer's final broadcast is through.
    final = get_broadcast_dir(output_dir) / f"step_{step + args.steps - 1}" / FINISHED_MARKER
    while not final.exists() and not acker.done():
        await asyncio.sleep(5)
    acker.cancel()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("streams", help="glob of trace stream .jsonl.zst files")
    ap.add_argument("--output-dir", required=True, help="the trainer's output_dir")
    ap.add_argument("--scorer", required=True, help="module:function, trace -> [score with .node, .reward, .reasons]")
    ap.add_argument("--scorer-args", type=json.loads, default={}, help="JSON keyword arguments for the scorer")
    ap.add_argument("--model", required=True, help="for the packer's cost model")
    ap.add_argument("--workers", type=int, required=True, help="trainer data-parallel ranks")
    ap.add_argument("--batch-size", type=int, default=32, help="segments per step")
    ap.add_argument("--steps", type=int, required=True)
    ap.add_argument("--seq-len", type=int, required=True, help="the trainer's micro batch length")
    ap.add_argument("--temperature", type=float, default=1.0, help="the sampling temperature of the traces")
    ap.add_argument("--pad-to-multiple-of", type=int, default=1)
    ap.add_argument("--env-name", default="craftax")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--wait-seconds", type=float, default=3600)
    ap.add_argument("--dry-run", action="store_true", help="score and report, send nothing")
    args = ap.parse_args()
    samples, stats = build(args, load_scorer(args.scorer))
    totals = Counter()
    for s in stats:
        totals.update(s["reasons"])
    print(f"credit over {sum(s['replies'] for s in stats)} replies: {dict(totals)}", flush=True)
    if not samples:
        raise SystemExit("no sample carries credit")
    if args.dry_run:
        return
    asyncio.run(feed(args, samples, stats))


if __name__ == "__main__":
    main()
