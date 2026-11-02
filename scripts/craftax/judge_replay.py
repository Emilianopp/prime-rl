"""Re-grade saved Craftax segments with the notes judge (or the adherence judge), offline.

    python scripts/craftax/judge_replay.py <run dir> [--backend capi --model gpt-4.1] [--mode single|group]
        [--judge notes|adherence] [--limit N] [--out FILE]

`--judge adherence` grades the segments whose episode line holds an adherence digest (runs
after Milestone 3, depth 2+) from the line alone; no saved game is needed.

A run dir holds episodes.jsonl and state/<node_id>.pkl (the game saved at each segment's
cut: what the eval script copies back). Each segment's judge inputs are rebuilt from its
saved game (the map, the seen cells, nearest features, next step, the MAP claim check) and
its episode line (notes, record, timeline), so runs from before the env logged them can be
graded too. Group mode grades the saved siblings of one parent in one call. Prints one line
per segment (map_check, findings, plan, notes score, latency) and the spread within groups,
the number that matters to GRPO.
"""

from __future__ import annotations

import argparse
import asyncio
import collections
import json
import os
import pickle
import statistics as st
import time

os.environ.setdefault("JAX_PLATFORMS", "cpu")

from craftax_env import glyph, world  # noqa: E402
from craftax_env.judge import (  # noqa: E402
    CraftaxAdherenceJudge,
    CraftaxGroupAdherenceJudge,
    CraftaxGroupNotesJudge,
    CraftaxNotesJudge,
    _adherence_verdict,
    _verdict,
    adherence_fields,
    group_adherence_verdicts,
    group_verdicts,
    judge_fields,
)
from craftax_env.segments import parse_notes  # noqa: E402
from crafter_env.capi import CrafterJudgeConfig  # noqa: E402


def rebuild(line: dict, saved: dict) -> dict:
    """A segment's judge inputs, as play_segment leaves them in trace.info["craftax"]."""
    game = glyph.CraftaxGame.restore(saved["game"], None)
    explored, stations = saved["explored"], saved.get("stations") or []
    view = (game.floor, *map(int, game.position))  # the saved game is the state at the notes turn
    claims = world.check_map_claims(game, parse_notes(line.get("notes") or "")["MAP"], explored, view=view)
    return {**{k: line.get(k) for k in ("node_id", "parent", "depth", "seed", "notes", "record", "record_before", "events",
                                        "actions_hist", "forfeit_kinds", "turns")},
            "actions_hist": line.get("actions_hist") or line.get("actions") or {},
            "forfeit_kinds": line.get("forfeit_kinds") or {},
            "map_ascii": world.ascii_map(game, explored), "nearest": world.nearest(game, explored),
            "next_step": world.next_step(game, stations), "map_claims": claims.text(), "map_check": claims.score,
            "notes_view": world.view_note(view)}


async def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run")
    ap.add_argument("--backend", default="capi", choices=["capi", "openai"])
    ap.add_argument("--model", default=None)
    ap.add_argument("--mode", default="single", choices=["single", "group"])
    ap.add_argument("--judge", default="notes", choices=["notes", "adherence"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--out", default=None, help="append one JSON line per graded segment")
    args = ap.parse_args()
    model = args.model or ("gpt-4.1" if args.backend == "capi" else "gemini-3.8-flash")
    extra = {} if args.backend == "capi" else {"base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
                                                "api_key_var": "GEMINI_API_KEY"}
    if args.backend == "openai" and not os.environ.get("GEMINI_API_KEY") and os.path.exists(os.path.expanduser("~/.gemini_key")):
        os.environ["GEMINI_API_KEY"] = open(os.path.expanduser("~/.gemini_key")).read().strip()
    config = CrafterJudgeConfig(backend=args.backend, model=model, sampling={"temperature": 0.0}, **extra)
    lines = [json.loads(l) for l in open(os.path.join(args.run, "episodes.jsonl")) if l.strip()]
    segs = []
    if args.judge == "adherence":
        segs = [{**line, "turns": line.get("turns")} for line in lines
                if line.get("task") == "craftax" and (line.get("adherence_digest") or {}).get("subgoals")]
    for line in lines if args.judge == "notes" else []:
        if line.get("task") != "craftax" or not line.get("notes"):
            continue
        path = os.path.join(args.run, "state", f"{line['node_id']}.pkl")
        if os.path.exists(path):
            with open(path, "rb") as f:
                segs.append(rebuild(line, pickle.load(f)))
    if args.limit:
        segs = segs[: args.limit]
    print(f"{len(segs)} segments to grade; {args.judge} judge {args.backend}:{model} {args.mode}")
    gate = asyncio.Semaphore(args.concurrency)
    results: list[dict] = []

    async def single(info):
        t0 = time.time()
        async with gate:
            try:
                resp = await CraftaxNotesJudge(config).evaluate(**judge_fields(info))
                p = resp.parsed
                v = _verdict(info, p.findings, p.plan, p.rationale, False)
            except Exception as exc:
                v = _verdict(info, 0.5, 0.5, f"failed: {exc}", True)
        results.append({**v, "node_id": info["node_id"], "parent": info["parent"], "call_s": time.time() - t0})

    async def group(infos):
        out, call_s, err = await group_verdicts(CraftaxGroupNotesJudge(config), infos, gate=gate)
        for info, v in zip(infos, out or [_verdict(i, 0.5, 0.5, f"failed: {err}", True) for i in infos]):
            results.append({**v, "node_id": info["node_id"], "parent": info["parent"], "call_s": call_s})

    async def single_adherence(info):
        t0 = time.time()
        async with gate:
            try:
                resp = await CraftaxAdherenceJudge(config).evaluate(**adherence_fields(info))
                v = _adherence_verdict(info, resp.parsed, False)
            except Exception as exc:
                v = _adherence_verdict(info, None, True, f"failed: {exc}")
        results.append({**v, "node_id": info["node_id"], "parent": info["parent"], "call_s": time.time() - t0})

    async def group_adherence(infos):
        out, call_s, err = await group_adherence_verdicts(CraftaxGroupAdherenceJudge(config), infos, gate=gate)
        for info, v in zip(infos, out or [_adherence_verdict(i, None, True, f"failed: {err}") for i in infos]):
            results.append({**v, "node_id": info["node_id"], "parent": info["parent"], "call_s": call_s})

    if args.judge == "adherence":
        single, group = single_adherence, group_adherence
    if args.mode == "group":
        by_parent = collections.defaultdict(list)
        for info in segs:
            by_parent[info["parent"]].append(info)
        await asyncio.gather(*(group(g) if len(g) > 1 else single(g[0]) for g in by_parent.values()))
    else:
        await asyncio.gather(*(single(i) for i in segs))
    for r in sorted(results, key=lambda r: (str(r["parent"]), r["node_id"])):
        parts = (f"statuses {r.get('statuses')} adherence {r['score']:.3f}" if args.judge == "adherence" else
                 f"map_check {r['map_check']:.2f} findings {r['findings']:.2f} plan {r['plan']:.2f} notes {r['score']:.3f}")
        print(f"{r['node_id'][:8]} parent {str(r['parent'])[:8]} {parts} {'FAILED ' if r['failed'] else ''}{r['call_s']:.1f}s | {r['rationale'][:150]}")
    ok = [r for r in results if not r["failed"]]
    if ok:
        print(f"{args.judge} score: mean {st.mean(r['score'] for r in ok):.3f}, min {min(r['score'] for r in ok):.3f}, max {max(r['score'] for r in ok):.3f}; "
              f"call p50 {st.median(r['call_s'] for r in ok):.1f}s; failed {len(results) - len(ok)}")
        groups = collections.defaultdict(list)
        for r in ok:
            groups[r["parent"]].append(r["score"])
        spreads = [max(g) - min(g) for g in groups.values() if len(g) > 1]
        if spreads:
            print(f"within-group spread (max - min) over {len(spreads)} groups: mean {st.mean(spreads):.3f}, "
                  f"zero-spread groups {sum(s == 0 for s in spreads)}")
    if args.out:
        with open(args.out, "a") as f:
            for r in results:
                f.write(json.dumps({**r, "backend": args.backend, "model": model, "mode": args.mode, "judge": args.judge}) + "\n")


if __name__ == "__main__":
    asyncio.run(main())
