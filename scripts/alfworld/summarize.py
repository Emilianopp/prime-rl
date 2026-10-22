"""Success rate per ALFWorld task type from an eval's episode log (env.episode_log).

    python summarize.py episodes.jsonl
"""

import json
import sys
from collections import defaultdict

episodes = [json.loads(line) for line in open(sys.argv[1])]
by_type = defaultdict(list)
for ep in episodes:
    by_type[ep["task_type"]].append(ep)
print(f"{'task type':<28} {'n':>4} {'success':>8} {'steps (won)':>12} {'malformed':>10}")
for name, eps in sorted(by_type.items()) + [("all", episodes)]:
    won = [ep for ep in eps if ep["won"]]
    steps = sum(ep["steps"] for ep in won) / len(won) if won else float("nan")
    malformed = sum(ep["malformed_replies"] for ep in eps) / max(1, sum(ep["steps"] for ep in eps))
    print(f"{name:<28} {len(eps):>4} {len(won) / len(eps):>8.1%} {steps:>12.1f} {malformed:>10.1%}")
