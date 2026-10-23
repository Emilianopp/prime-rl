"""Per-task summary of an eval's episode log (an env's ``episode_log``): episodes,
mean reward, games won, mean steps of won games, and replies without an action
when the env counts them.

    python summarize_eval.py episodes.jsonl
"""

import json
import sys
from collections import defaultdict

episodes = [json.loads(line) for line in open(sys.argv[1])]
groups = defaultdict(list)
for ep in episodes:
    groups[ep.get("task_type") or ep.get("task") or "?"].append(ep)
print(f"{'task':<32} {'n':>5} {'reward':>7} {'won':>7} {'steps (won)':>12} {'malformed':>10}")
for name, eps in sorted(groups.items()) + [("all", episodes)]:
    won = [ep for ep in eps if ep.get("won")]
    reward = sum(ep["reward"] for ep in eps) / len(eps)
    steps = sum(ep["steps"] for ep in won) / len(won) if won else float("nan")
    malformed = (
        f"{sum(ep['malformed_replies'] for ep in eps) / max(1, sum(ep['steps'] for ep in eps)):.1%}"
        if all("malformed_replies" in ep for ep in eps)
        else "-"
    )
    print(f"{name:<32} {len(eps):>5} {reward:>7.3f} {len(won) / len(eps):>7.1%} {steps:>12.1f} {malformed:>10}")
