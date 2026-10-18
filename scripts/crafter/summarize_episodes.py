"""Summarize a Crafter episode log (one JSON line per episode, from the env's
`episode_log`): per-achievement success rates, the Crafter score, how episodes
ended, when achievements unlock and what the policy does.

    python scripts/crafter/summarize_episodes.py episodes.jsonl
"""

import json
import math
import statistics
import sys
from collections import Counter

ACHIEVEMENTS = [
    "collect_coal", "collect_diamond", "collect_drink", "collect_iron", "collect_sapling",
    "collect_stone", "collect_wood", "defeat_skeleton", "defeat_zombie", "eat_cow", "eat_plant",
    "make_iron_pickaxe", "make_iron_sword", "make_stone_pickaxe", "make_stone_sword",
    "make_wood_pickaxe", "make_wood_sword", "place_furnace", "place_plant", "place_stone",
    "place_table", "wake_up",
]


def quantiles(values: list[float]) -> str:
    if not values:
        return "-"
    values = sorted(values)
    pick = lambda q: values[min(len(values) - 1, int(q * len(values)))]
    return f"p10 {pick(0.1):.0f} / p50 {pick(0.5):.0f} / p90 {pick(0.9):.0f} / max {values[-1]:.0f}"


def main(path: str) -> None:
    episodes = [json.loads(line) for line in open(path) if line.strip()]
    n = len(episodes)
    print(f"{n} episodes")
    rates = {a: 100.0 * sum(a in e.get("unlocked_at", {}) for e in episodes) / n for a in ACHIEVEMENTS}
    score = math.exp(sum(math.log(1 + r) for r in rates.values()) / len(rates)) - 1
    print(f"Crafter score {score:.2f}%  (geometric mean of per-achievement success, Hafner 2021)")
    print(f"achievements per episode: mean {statistics.mean(e['achievements'] for e in episodes):.2f}, "
          f"{quantiles([e['achievements'] for e in episodes])}")
    print(f"game steps: {quantiles([e['steps'] for e in episodes])}")
    print(f"trace tokens: {quantiles([e.get('trace_tokens', 0) for e in episodes])}")
    print("ended:", dict(Counter(e.get("ended_by", "?") for e in episodes).most_common()))
    print(f"invalid actions per step: {sum(e['invalid_actions'] for e in episodes) / max(1, sum(e['steps'] for e in episodes)):.3f}")
    print("\nachievement            success   first unlock step (p50 / p90)")
    for a in sorted(ACHIEVEMENTS, key=lambda a: -rates[a]):
        steps = sorted(e["unlocked_at"][a] for e in episodes if a in e.get("unlocked_at", {}))
        when = f"{steps[len(steps) // 2]} / {steps[min(len(steps) - 1, int(0.9 * len(steps)))]}" if steps else "-"
        print(f"  {a:20s} {rates[a]:6.1f}%   {when}")
    actions = Counter()
    for e in episodes:
        actions.update(e.get("actions", {}))
    total = sum(actions.values()) or 1
    print("\nactions:", ", ".join(f"{a} {100 * c / total:.1f}%" for a, c in actions.most_common()))
    inventory = Counter()
    for e in episodes:
        inventory.update({k: v for k, v in e.get("inventory", {}).items() if k not in {"health", "food", "drink", "energy"}})
    print("final inventory (mean):", ", ".join(f"{k} {v / n:.2f}" for k, v in inventory.most_common()))


if __name__ == "__main__":
    main(sys.argv[1])
