"""Compare Crafter eval arms on the seeds they share: achievements, Crafter score,
survival, per-achievement success and memory-tool use.

    python scripts/crafter/compare_arms.py plain=a/episodes.jsonl notebook=b/episodes.jsonl ...
"""

import json
import math
import statistics
import sys

from summarize_episodes import ACHIEVEMENTS


def load(path: str) -> dict[int, dict]:
    return {e["seed"]: e for e in (json.loads(line) for line in open(path) if line.strip())}


def score(episodes: list[dict]) -> float:
    rates = [100.0 * sum(a in e.get("unlocked_at", {}) for e in episodes) / len(episodes) for a in ACHIEVEMENTS]
    return math.exp(sum(math.log(1 + r) for r in rates) / len(rates)) - 1


def mean_se(values: list[float]) -> str:
    m = statistics.mean(values)
    se = statistics.stdev(values) / math.sqrt(len(values)) if len(values) > 1 else 0.0
    return f"{m:.2f}±{se:.2f}"


def tool_calls(e: dict) -> float:
    if e.get("workspace"):
        c = e["workspace"]["counts"]
        return sum(c.get(k, 0) for k in ("search", "read", "write", "append"))
    if e.get("notebook"):
        return e["notebook"]["writes"] + e["notebook"]["forgets"] + e["notebook"]["scratchpad_updates"]
    return 0.0


def main(args: list[str]) -> None:
    arms = {name: load(path) for name, path in (a.split("=", 1) for a in args)}
    seeds = sorted(set.intersection(*(set(v) for v in arms.values())))
    print(f"{len(seeds)} shared seeds\n")
    rows = {name: [arm[s] for s in seeds] for name, arm in arms.items()}
    # Memory text costs tokens, so under a token cap memory arms play fewer steps:
    # also compare what every arm had unlocked by the same game step.
    k = min(sorted(e["steps"] for e in eps)[len(eps) // 10] for eps in rows.values())
    print(f"(ach@{k}: achievements unlocked by game step {k}, the shortest arm's 10th-percentile length)\n")
    print(f"{'arm':14s} {'achievements':>12s} {f'ach@{k}':>11s} {'score%':>7s} {'deaths':>11s} {'steps':>7s} {'tools/ep':>9s} {'used':>6s}")
    for name, eps in rows.items():
        deaths = [len(e.get("deaths", [])) for e in eps]
        used = sum(tool_calls(e) > 0 for e in eps)
        print(
            f"{name:14s} {mean_se([e['achievements'] for e in eps]):>12s} "
            f"{mean_se([sum(t <= k for t in e.get('unlocked_at', {}).values()) for e in eps]):>11s} {score(eps):7.2f} {mean_se(deaths):>11s} "
            f"{statistics.mean(e['steps'] for e in eps):7.0f} {statistics.mean(tool_calls(e) for e in eps):9.1f} {used:>3d}/{len(eps)}"
        )
    print(f"\n{'achievement':20s}" + "".join(f"{n[:10]:>11s}" for n in rows))
    for a in ACHIEVEMENTS:
        rates = [100.0 * sum(a in e.get("unlocked_at", {}) for e in eps) / len(eps) for eps in rows.values()]
        if any(rates):
            print(f"{a:20s}" + "".join(f"{r:10.1f}%" for r in rates))


if __name__ == "__main__":
    sys.path.insert(0, __import__("os").path.dirname(__file__))
    main(sys.argv[1:])
