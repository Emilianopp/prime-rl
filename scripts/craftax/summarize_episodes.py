"""Summarize a craftax-env episode log. usage: summarize_episodes.py <episodes.jsonl>

Return as GlyphBench reports it (% of max reward = 100 * return / 226), the unlocked
achievements' reward alone (continuous runs' reward), respawned deaths, the
Craftax score (Hafner's geometric mean of per-achievement success rates over the
67 achievements), achievements per episode, floors reached, how episodes ended and
forfeited turns.
"""

import json
import math
import statistics as st
import sys
from collections import Counter

FLOORS = ("Overworld", "Dungeon", "Gnomish Mines", "Sewers", "Vaults", "Troll Mines", "Fire Realm", "Ice Realm", "Graveyard")
N_ACHIEVEMENTS = 67

eps = [json.loads(line) for line in open(sys.argv[1]) if line.strip()]
if not eps:
    sys.exit("no episodes")
n = len(eps)


def q(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p * len(xs)))]


pct = [100 * e["return"] / 226 for e in eps]
ach = [len(e["achievements"]) for e in eps]
turns = [e["turns"] for e in eps]
tokens = [e.get("trace_tokens", 0) for e in eps]
ach_pct = [100 * e.get("achievement_score", 0) / 226 for e in eps]
deaths = [len(e.get("deaths") or []) for e in eps]
print(f"{n} episodes")
print(f"% of max reward (100 * return / 226): mean {st.mean(pct):.2f}  p10 {q(pct, .1):.1f} / p50 {q(pct, .5):.1f} / p90 {q(pct, .9):.1f} / max {max(pct):.1f}")
print(f"achievements' reward % of 226 (no health term): mean {st.mean(ach_pct):.2f}  p50 {q(ach_pct, .5):.1f} / max {max(ach_pct):.1f}")
if any(deaths):
    print(f"respawned deaths per episode: mean {st.mean(deaths):.1f}, p10 {q(deaths, .1)} / p50 {q(deaths, .5)} / p90 {q(deaths, .9)} / max {max(deaths)}")
counts = Counter(a for e in eps for a in e["achievements"])
rates = [counts.get(a, 0) / n for a in counts] + [0.0] * (N_ACHIEVEMENTS - len(counts))
score = math.exp(sum(math.log(1 + 100 * r) for r in rates) / N_ACHIEVEMENTS) - 1
print(f"Craftax score {score:.2f}%  (geometric mean of per-achievement success over {N_ACHIEVEMENTS}, Hafner 2021)")
print(f"achievements per episode: mean {st.mean(ach):.2f}, p10 {q(ach, .1)} / p50 {q(ach, .5)} / p90 {q(ach, .9)} / max {max(ach)}")
print(f"turns: p10 {q(turns, .1)} / p50 {q(turns, .5)} / p90 {q(turns, .9)} / max {max(turns)}")
print(f"trace tokens: p10 {q(tokens, .1)} / p50 {q(tokens, .5)} / p90 {q(tokens, .9)} / max {max(tokens)}")
print("ended:", dict(Counter(e["ended_by"] for e in eps)))
print("deepest floor:", {FLOORS[f] if f < len(FLOORS) else f: c for f, c in sorted(Counter(e["max_floor"] for e in eps).items())})
forfeits = Counter()
for e in eps:
    forfeits.update(e["forfeits"])
print(f"forfeited turns per turn: {sum(forfeits.values()) / max(sum(turns), 1):.3f}  {dict(forfeits)}")
print()
print("achievement            success   first unlock turn (p50)")
for a, c in counts.most_common():
    first = [e["unlocked_at"][a] for e in eps if a in e["unlocked_at"]]
    print(f"  {a:22s} {100 * c / n:6.1f}%   {q(first, .5) if first else '-'}")
actions = Counter()
for e in eps:
    actions.update(e["actions"])
total = sum(actions.values()) or 1
print()
print("actions: " + ", ".join(f"{a} {100 * c / total:.1f}%" for a, c in actions.most_common(15)))
