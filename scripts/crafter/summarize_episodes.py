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
    died = [e for e in episodes if e.get("ended_by") == "died"]
    if died:
        causes = Counter()
        for e in died:
            starved = [k for k in ("food", "drink", "energy") if e.get("inventory", {}).get(k, 1) == 0]
            causes[" + ".join(starved) if starved else "damage"] += 1
        print(f"died: {len(died)}, at step {quantiles([e['steps'] for e in died])}; causes (vital at 0, else damage):", dict(causes.most_common()))
    deaths = [len(e.get("deaths", [])) for e in episodes]
    if any(deaths):
        print(f"deaths per episode: mean {statistics.mean(deaths):.2f}, {quantiles(deaths)}")
        first = [e["deaths"][0]["step"] for e in episodes if e.get("deaths")]
        print(f"first death step: {quantiles(first)}")
        causes = Counter()
        for e in episodes:
            for d in e.get("deaths", []):
                starved = [k for k in ("food", "drink", "energy") if d["vitals"].get(k, 1) == 0]
                causes[" + ".join(starved) if starved else "damage"] += 1
        print("death causes (vital at 0, else damage):", dict(causes.most_common()))
    books = [e["notebook"] for e in episodes if e.get("notebook")]
    if books:
        print(f"notebook: memory writes per episode {statistics.mean(b['writes'] for b in books):.1f}, "
              f"scratchpad updates {statistics.mean(b['scratchpad_updates'] for b in books):.1f}, "
              f"episodes that wrote anything {sum(1 for b in books if b['writes'] or b['scratchpad_updates'])}/{len(books)}")
        for b in books[:3]:
            print("  e.g. memories:", b["memories"][:5], "| scratchpad:", b["scratchpad"][:160])
    spaces = [e["workspace"] for e in episodes if e.get("workspace")]
    if spaces:
        tools = Counter()
        for w in spaces:
            tools.update(w["counts"])
        used = sum(1 for w in spaces if any(w["counts"].get(k) for k in ("search", "read", "write", "append")))
        print(f"workspace: tool calls per episode {', '.join(f'{k} {v / len(spaces):.1f}' for k, v in tools.most_common())}; "
              f"episodes that used it {used}/{len(spaces)}; ledger entries {statistics.mean(w['ledger_entries'] for w in spaces):.0f}")
        queries = Counter(s["query"].lower() for w in spaces for s in w["searches"])
        hit = [s["hits"] > 0 for w in spaces for s in w["searches"]]
        if hit:
            print(f"  searches with hits {100 * sum(hit) / len(hit):.0f}%; top queries: {queries.most_common(8)}")
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
