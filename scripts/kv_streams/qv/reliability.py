"""How repeatable is the judge? Grade the same states several times and measure the spread.

Takes whole games from replay.py's output (every reset, so segment rewards can be
recomputed per repeat), sends each state to the judge `repeats` times with identical
prompts, and reports per quantity (P parts, V, G*, D, Q, and the segment reward R from
repeat i at both ends): the mean within-state standard deviation, the spread across
states, and ICC(1) = the share of the variance that is between states rather than
repeat noise (1: perfectly repeatable; 0: pure noise). Flaws: the mean count's spread
and the Jaccard overlap of flaw types between two repeats.

    GEMINI_API_KEY=... python reliability.py <replay.jsonl> <out.json> [games] [repeats]
"""

from __future__ import annotations

import asyncio
import itertools
import json
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import judge  # noqa: E402


def icc1(groups: list[list[float]]) -> float | None:
    """One-way random-effects ICC(1,1) over groups of repeated measurements."""
    groups = [g for g in groups if len(g) > 1]
    if len(groups) < 2:
        return None
    k = min(len(g) for g in groups)
    groups = [g[:k] for g in groups]
    grand = st.mean(x for g in groups for x in g)
    msb = k * sum((st.mean(g) - grand) ** 2 for g in groups) / (len(groups) - 1)
    msw = sum((x - st.mean(g)) ** 2 for g in groups for x in g) / (len(groups) * (k - 1))
    den = msb + (k - 1) * msw
    return (msb - msw) / den if den else None


def summarize(groups: list[list[float]]) -> dict:
    groups = [g for g in groups if len(g) > 1]
    within = st.mean(st.pstdev(g) for g in groups) if groups else None
    between = st.pstdev([st.mean(g) for g in groups]) if len(groups) > 1 else None
    return {"n": len(groups), "within_sd": within, "between_sd": between, "icc": icc1(groups)}


def main():
    src, out = sys.argv[1], sys.argv[2]
    games = int(sys.argv[3]) if len(sys.argv) > 3 else 6
    repeats = int(sys.argv[4]) if len(sys.argv) > 4 else 5
    eps = [json.loads(line) for line in open(src)][:games]
    jobs = {}
    for ep in eps:
        for k in range(len(ep["segments"]) + 1):
            s = ep["resets"][k]
            prev = next((r["summary"] for r in reversed(ep["resets"][:k]) if r.get("summary")), None)
            text = judge.prompt(ep["rules"], s, s["kind"] == "reset", s.get("summary"), prev)
            for i in range(repeats):
                jobs[f"{ep['seed']}:{k}:r{i}"] = text
    replies = asyncio.run(judge.judge_all(jobs, out + ".calls.jsonl", 12))

    per = {q: [] for q in ("tool", "ready", "safe", "near", "P", "V", "G_star", "D", "Q", "R", "n_flaws")}
    jaccard = []
    for ep in eps:
        n = len(ep["segments"]) + 1
        scored = []  # [k][i]
        for k in range(n):
            s = ep["resets"][k]
            review = s["kind"] == "reset"
            row = []
            for i in range(repeats):
                j = replies.get(f"{ep['seed']}:{k}:r{i}")
                row.append(judge.score(s, j, review, s.get("summary")) if j else None)
            scored.append(row)
            ok = [x for x in row if x]
            for part in ("tool", "ready", "safe", "near"):
                per[part].append([x["P"][part] for x in ok])
            per["P"].append([sum(x["P"].values()) for x in ok])
            for q in ("V", "G_star", "Q"):
                per[q].append([x[q] for x in ok])
            if review and s.get("summary"):
                per["D"].append([x["D"] for x in ok])
                per["n_flaws"].append([len(x["flaws"]) for x in ok])
                types = [{f.get("type") for f in x["flaws"]} for x in ok]
                for a, b in itertools.combinations(types, 2):
                    if a | b:
                        jaccard.append(len(a & b) / len(a | b))
        for t in range(n - 1):
            pairs = [(scored[t][i], scored[t + 1][i]) for i in range(repeats)]
            per["R"].append([judge.segment_reward(a, b) for a, b in pairs if a and b])
    report = {q: summarize(g) for q, g in per.items()}
    report["flaw_type_jaccard"] = st.mean(jaccard) if jaccard else None
    report["games"], report["repeats"], report["calls"] = len(eps), repeats, len(jobs)
    json.dump({"report": report, "groups": per}, open(out, "w"))
    for q, r in report.items():
        print(q, json.dumps(r) if isinstance(r, dict) else r)


if __name__ == "__main__":
    main()
