"""Re-score recorded segments with the judge, without a policy or a GPU.

Reads a Crafter episode log (one JSON line per segment, as the env writes with
segments on: notes, inventory, achievements, events, the local map, the nearest
resources and the tech tree's next step are all in the line) and runs the judge
prompt on each, printing the sub-scores and the rationale next to the segment's
achievement delta. For tuning the judge's context and rubric, and for the
"judge agreement with ground truth" measurement: scores should rank segments that
unlocked a tier above segments that wandered.

    GEMINI_API_KEY=... uv run scripts/crafter/judge_replay.py episodes.jsonl --limit 8 [--show-prompt]

With your saved Copilot CLI login (or COPILOT_GITHUB_TOKEN):
    uv run scripts/crafter/judge_replay.py episodes.jsonl --backend capi --model gpt-4.1
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
from types import SimpleNamespace

from crafter_env.judge import (
    CrafterGroupJudge,
    CrafterJudge,
    CrafterJudgeConfig,
    group_verdicts,
    judge_fields,
    judge_segment,
)


class ReplayTrace:
    def __init__(self, info):
        self.rewards, self.metrics, self.info, self.extra_usage = {}, {}, {"crafter": info}, None

    def record_reward(self, name, value, weight=1.0):
        self.rewards[name] = value * weight

    def record_metrics(self, metrics):
        self.metrics.update(metrics)

    def record_judge_call(self, **kwargs):
        self.info.setdefault("judge_calls", []).append(kwargs)


def info_from_line(line: dict) -> dict:
    unlocked = line.get("achievements_unlocked")
    if unlocked is None:  # a log from before the judge fields were written: rebuild from what is there
        unlocked = sorted(set(line.get("achievements_before") or []) | set(line.get("unlocked_at") or {}))
    return {**line, "achievements": unlocked, "steps": line.get("steps")}


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--limit", type=int, default=8)
    ap.add_argument("--backend", choices=["openai", "capi"], default="openai")
    ap.add_argument("--model")
    ap.add_argument("--base-url", help="OpenAI-compatible endpoint; CAPI uses GitHub's Copilot endpoint")
    ap.add_argument("--api-key-var", help="API key environment variable for OpenAI-compatible backends")
    ap.add_argument("--goal", default="collect_diamond")
    ap.add_argument("--mix", default="0.5,0.3,0.2")
    ap.add_argument("--world-radius", type=int, default=24)
    ap.add_argument("--show-prompt", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="print the prompt of the first picked segment and exit without calling the judge")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--mode", choices=["single", "group"], default="single",
                    help="group: judge the branches of each node together and compare with the single scores in the log")
    args = ap.parse_args()
    args.model = args.model or ("gpt-4.1" if args.backend == "capi" else "gemini-3.8-flash")
    args.base_url = args.base_url or (
        "https://api.githubcopilot.com"
        if args.backend == "capi"
        else "https://generativelanguage.googleapis.com/v1beta/openai/"
    )
    args.api_key_var = args.api_key_var or ("COPILOT_GITHUB_TOKEN" if args.backend == "capi" else "GEMINI_API_KEY")
    mix = tuple(float(x) for x in args.mix.split(","))
    lines = [json.loads(l) for l in open(args.log) if l.strip()]
    if args.mode == "group":
        return await replay_groups(args, lines, mix)
    segments = [l for l in lines if l.get("task") == "crafter" and l.get("depth") and l.get("summary") is not None]
    # a spread of outcomes: the most and least productive segments first
    segments.sort(key=lambda l: -len(l.get("segment_achievements") or []))
    picked = segments[: args.limit // 2] + segments[len(segments) - (args.limit - args.limit // 2):] if len(segments) > args.limit else segments
    judge = CrafterJudge(
        CrafterJudgeConfig(
            backend=args.backend,
            base_url=args.base_url,
            api_key_var=args.api_key_var,
            model=args.model,
            sampling={"temperature": 0.0},
        )
    )
    if (args.show_prompt or args.dry_run) and picked:
        print(judge.build_messages(**judge_fields(info_from_line(picked[0]), args.goal, args.world_radius)))
        print("=" * 80)
    if args.dry_run:
        print(f"{len(picked)} of {len(segments)} segments would be judged with {args.model}")
        return
    gate = asyncio.Semaphore(args.concurrency)
    traces = [ReplayTrace(info_from_line(l)) for l in picked]
    await asyncio.gather(*(judge_segment(judge, t, goal=args.goal, weight=1.0, mix=mix, world_radius=args.world_radius, gate=gate) for t in traces))
    rows = []
    for line, trace in zip(picked, traces):
        v = trace.info["crafter"]["judge"]
        rows.append((len(line.get("segment_achievements") or []), v))
        print(f"seed {line['seed']} d{line['depth']} delta {line.get('segment_achievements')} deaths {len(line.get('deaths') or [])} "
              f"| score {v['score']:.2f} progress {v.get('progress', float('nan')):.2f} plan {v.get('plan', float('nan')):.2f} "
              f"consistency {v.get('consistency', float('nan')):.2f} failed {v['failed']}\n   {v['rationale'][:300]}")
    ok = [(d, v["score"]) for d, v in rows if not v["failed"]]
    if len(ok) > 2 and len({d for d, _ in ok}) > 1:
        xs, ys = zip(*ok)
        mx, my = statistics.mean(xs), statistics.mean(ys)
        cov = sum((x - mx) * (y - my) for x, y in ok)
        sx, sy = statistics.pstdev(xs), statistics.pstdev(ys)
        print(f"\n{len(ok)} judged; correlation(score, achievement delta) = {cov / (len(ok) * sx * sy) if sx and sy else float('nan'):.2f}")
    usage = [c for t in traces for c in t.info.get("judge_calls", [])]
    spent = {k: sum(float(t.info["crafter"]["judge"].get(k, 0.0)) for t in traces) for k in ("input_tokens", "output_tokens", "thinking_tokens", "cost_usd")}
    print(f"{len(usage)} judge calls; failed: {sum(v['failed'] for _, v in rows)}; cost ${spent['cost_usd']:.4f} "
          f"(in {spent['input_tokens']:.0f}, out {spent['output_tokens']:.0f} tokens incl. {spent['thinking_tokens']:.0f} thinking)")
    record_replay(args, spent, len(usage))


async def replay_groups(args, lines, mix) -> None:
    """One comparative call per node over its recorded branches, next to the single scores already in the log."""
    import collections
    import time

    single = {l["node_id"]: l for l in lines if l.get("task") == "crafter-judge" and not l.get("judge_failed")}
    groups = collections.defaultdict(list)
    for l in lines:
        if l.get("task") == "crafter" and l.get("parent") and l.get("summary") is not None:
            groups[l["parent"]].append(l)
    full = [g for g in groups.values() if len(g) >= 2]
    full.sort(key=lambda g: -(max(len(x.get("segment_achievements") or []) for x in g) - min(len(x.get("segment_achievements") or []) for x in g)))
    picked = full[: max(1, args.limit // 4)]
    judge = CrafterGroupJudge(
        CrafterJudgeConfig(
            backend=args.backend,
            base_url=args.base_url,
            api_key_var=args.api_key_var,
            model=args.model,
            sampling={"temperature": 0.0},
        )
    )
    if args.show_prompt or args.dry_run:
        print(judge.build_messages(infos=[info_from_line(x) for x in picked[0]], goal=args.goal, world_radius=args.world_radius))
        print("=" * 80)
    if args.dry_run:
        print(f"{len(picked)} of {len(full)} nodes would be judged with {args.model}")
        return
    gate = asyncio.Semaphore(args.concurrency)
    spent: list = []

    async def one(g):
        t0 = time.time()
        costs = []
        out, call_s, err = await group_verdicts(judge, [info_from_line(x) for x in g], goal=args.goal, mix=mix, world_radius=args.world_radius, gate=gate, costs=costs)
        spent.extend(costs)
        return g, out, call_s, err

    pairs = []
    for g, out, call_s, err in await asyncio.gather(*(one(g) for g in picked)):
        print(f"node {g[0]['parent'][:10]} (seed {g[0]['seed']}, depth {g[0]['depth']}): {len(g)} branches, call {call_s:.1f} s{', FAILED: ' + err if err else ''}")
        for x, v in zip(g, out or [None] * len(g)):
            s1 = single.get(x["node_id"], {}).get("judge_score")
            pairs.append((s1, v["score"] if v else None, len(x.get("segment_achievements") or [])))
            one_s = "-" if s1 is None else f"{s1:.2f}"
            grp_s = "-" if not v else f"{v['score']:.2f}"
            why = (v or {}).get("rationale", "")[:120]
            print(f"   delta {str(x.get('segment_achievements')):40s} single {one_s}  group {grp_s}  {why}")
    ok = [(a, b, d) for a, b, d in pairs if a is not None and b is not None]
    if ok:
        import statistics
        def corr(xs, ys):
            mx, my = statistics.mean(xs), statistics.mean(ys)
            sx, sy = statistics.pstdev(xs), statistics.pstdev(ys)
            return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / (len(xs) * sx * sy) if sx and sy else float("nan")
    total = {k: sum(c[k] for c in spent) for k in ("input_tokens", "output_tokens", "thinking_tokens", "cost_usd")}
    print(f"{len(spent)} group calls; cost ${total['cost_usd']:.4f} (in {total['input_tokens']:.0f}, out {total['output_tokens']:.0f} tokens incl. {total['thinking_tokens']:.0f} thinking)")
    record_replay(args, total, len(spent))
    if ok:
        print(f"\n{len(ok)} branches: corr(single, group) {corr([a for a,_,_ in ok], [b for _,b,_ in ok]):.2f}; "
              f"corr(single, delta) {corr([a for a,_,_ in ok], [d for *_,d in ok]):.2f}; corr(group, delta) {corr([b for _,b,_ in ok], [d for *_,d in ok]):.2f}")


def record_replay(args, spent: dict, calls: int) -> None:
    import os
    import sys
    import time

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from judge_cost import append

    append({"at": time.time(), "kind": "replay", "note": f"replay {args.mode} {os.path.basename(os.path.dirname(args.log))}",
            "model": args.model, "calls": calls, "judged": calls, **spent})


if __name__ == "__main__":
    asyncio.run(main())
