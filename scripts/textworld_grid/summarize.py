"""Summarize one TextWorld grid cell: episode milestones and scores from the
env's episode log, engine peaks from the sampled /metrics gauges.

    python summarize.py CELL_DIR --cell vllm_c24 --cpr 24 --arm vllm

Milestone N = seconds from the first episode's start to the N-th finished
episode (the kv-eviction grid's ``sampling_seconds``). Scores are over the
first N finished episodes, so cells are compared on the same count.
"""

import argparse
import json
import re
from pathlib import Path

MILESTONES = (128, 256, 512)


def read_gauges(path: Path) -> dict[str, float]:
    """Per sample, gauges and counters are summed over engines; returns each gauge's
    peak and each counter's last value."""
    peaks: dict[str, float] = {}
    sample: dict[str, float] = {}
    counters: dict[str, float] = {}
    last: dict[str, float] = {}

    def flush():
        for name, value in sample.items():
            peaks[name] = max(peaks.get(name, 0.0), value)
        last.update(counters)
        sample.clear()
        counters.clear()

    if not path.exists():
        return {}
    for line in path.read_text().splitlines():
        if line.startswith("# time"):
            flush()
            continue
        m = re.match(r"^((?:vllm|kvs):[a-z_]+)(\{[^}]*\})?\s+([0-9.eE+-]+)$", line)
        if not m:
            continue
        name, labels, value = m.group(1), m.group(2) or "", float(m.group(3))
        if name == "vllm:cache_config_info":
            tokens = re.search(r'kv_cache_size_tokens="(\d+)"', labels)
            if tokens:
                last["kv_cache_size_tokens"] = float(tokens.group(1))
        elif name.endswith("_total"):
            counters[name] = counters.get(name, 0.0) + value
        else:
            sample[name] = sample.get(name, 0.0) + value
    flush()
    out = {f"peak_{k.split(':')[1]}": v for k, v in peaks.items()}
    out.update({k.split(":")[-1]: v for k, v in last.items()})
    if last.get("vllm:prefix_cache_queries_total"):
        out["prefix_cache_hit_rate"] = last["vllm:prefix_cache_hits_total"] / last["vllm:prefix_cache_queries_total"]
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("cell_dir", type=Path)
    parser.add_argument("--cell", required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--cpr", type=int, required=True)
    parser.add_argument("--episodes", type=int, default=512)
    args = parser.parse_args()

    log = args.cell_dir / "episodes.jsonl"
    episodes = [json.loads(line) for line in log.read_text().splitlines() if line.strip()] if log.exists() else []
    episodes.sort(key=lambda e: e["end"])
    t0 = min((e["start"] for e in episodes), default=0.0)
    summary: dict = {"cell": args.cell, "arm": args.arm, "cpr": args.cpr, "finished": len(episodes), "milestones": {}}
    for n in (*MILESTONES, args.episodes):
        if len(episodes) >= n and str(n) not in summary["milestones"]:
            first = episodes[:n]
            seconds = first[-1]["end"] - t0
            summary["milestones"][str(n)] = {
                "sampling_seconds": round(seconds, 1),
                "rollouts_per_second": round(n / seconds, 4),
                "mean_reward": round(sum(e["reward"] for e in first) / n, 4),
                "win_rate": round(sum(bool(e["won"]) for e in first) / n, 4),
                "mean_steps": round(sum(e["steps"] for e in first) / n, 1),
            }
    if episodes:
        summary["all_finished"] = {
            "mean_reward": round(sum(e["reward"] for e in episodes) / len(episodes), 4),
            "win_rate": round(sum(bool(e["won"]) for e in episodes) / len(episodes), 4),
            "by_task": {},
        }
        for task in sorted({e["task"] for e in episodes}):
            group = [e for e in episodes if e["task"] == task]
            summary["all_finished"]["by_task"][task] = {
                "n": len(group),
                "mean_reward": round(sum(e["reward"] for e in group) / len(group), 4),
            }
    summary["engine"] = read_gauges(args.cell_dir / "metrics.prom")
    (args.cell_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
