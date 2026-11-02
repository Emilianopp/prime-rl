"""Forfeited turns of Craftax runs, from their saved traces, and a ledger to follow them.

    python scripts/craftax/forfeit_report.py <run out dir>...              # per-run report
    python scripts/craftax/forfeit_report.py <run out dir>... --ledger     # also record them
    python scripts/craftax/forfeit_report.py --show                        # every run recorded

A run out dir is what scripts/craftax/eval.sbatch writes (cell.txt, run/*/monitors/file/
traces/stream/*.jsonl.zst). The traces carry each reply's content, reasoning and finish
reason, so this works on runs from before the env logged forfeit kinds itself
(craftax_env.forfeits, 2026-11-01), and the two classifications agree. The ledger is one
JSON line per run at $SCRATCH/factory/runs/craftax_forfeit_ledger.jsonl (override with
FORFEIT_LEDGER); eval.sbatch appends to it at the end of every run.
"""

from __future__ import annotations

import argparse
import datetime as dt
import glob
import io
import json
import os
import statistics as st
import sys

import zstandard

from craftax_env import glyph
from craftax_env.forfeits import BUCKET, COLLAPSE_STREAK, KINDS, ForfeitTracker, classify

LEDGER = os.environ.get("FORFEIT_LEDGER") or os.path.join(
    os.environ.get("SCRATCH", "."), "factory/runs/craftax_forfeit_ledger.jsonl"
)


def traces(run_dir: str):
    for path in sorted(glob.glob(os.path.join(run_dir, "run*/*/monitors/file/traces/stream/*.jsonl.zst"))):
        with open(path, "rb") as f:
            for line in io.TextIOWrapper(zstandard.ZstdDecompressor().stream_reader(f)):
                if line.strip():
                    yield from json.loads(line).get("traces") or []


def run_report(run_dir: str) -> dict | None:
    trackers: list[ForfeitTracker] = []
    replies = 0
    for tr in traces(run_dir):
        info = (tr.get("info") or {}).get("craftax") or {}
        max_actions = 1  # GlyphBench's single action; multi-action runs log it in the task
        max_actions = int(((tr.get("task") or {}).get("data") or {}).get("info", {}).get("max_actions", max_actions))
        nodes, t = tr["nodes"], ForfeitTracker()
        calls = tr.get("calls") or []
        for c in calls[: info.get("turns", len(calls))]:  # the turn cut off by a limit has no reply played
            m = nodes[c["node"]]["message"] if c.get("node") is not None else {}
            content, reasoning = m.get("content"), m.get("reasoning_content")
            indices, failure, _ = glyph.parse_actions(content or "", max_actions)
            t.update(None if indices else classify(content, reasoning, c.get("finish_reason"), failure))
        replies += t.turns
        trackers.append(t)
    if not trackers:
        return None
    n = len(trackers)
    kinds = {k: sum(t.kinds.get(k, 0) for t in trackers) for k in KINDS}
    forfeits = sum(kinds.values())
    share = [sum(t.kinds.values()) / max(t.turns, 1) for t in trackers]
    onsets = sorted(t.collapse_onset for t in trackers if t.collapse_onset is not None)
    width = max(len(t.buckets) for t in trackers)
    by_bucket = []
    for b in range(width):
        live = [t for t in trackers if len(t.buckets) > b]
        by_bucket.append(round(sum(t.buckets[b] for t in live) / max(1, sum(min(BUCKET, t.turns - BUCKET * b) for t in live)), 3))
    cell = os.path.join(run_dir, "cell.txt")
    return {
        "run": os.path.basename(os.path.normpath(run_dir)),
        "cell": open(cell).read().strip() if os.path.exists(cell) else "",
        "episodes": n,
        "replies": replies,
        "forfeit_rate": forfeits / max(replies, 1),
        "kinds": {k: v / max(forfeits, 1) for k, v in kinds.items() if v},
        "length_cut_rate": kinds["length"] / max(replies, 1),
        "episodes_lt5": sum(x < 0.05 for x in share) / n,
        "episodes_gt50": sum(x > 0.5 for x in share) / n,
        "collapsed": len(onsets) / n,
        "collapse_onset_p50": st.median(onsets) if onsets else None,
        "max_streak_p90": sorted(t.max_streak for t in trackers)[int(0.9 * (n - 1))],
        "by_bucket": by_bucket,
        "at": dt.datetime.now().isoformat(timespec="seconds"),
    }


def show(r: dict) -> None:
    print(f"== {r['run']}  {r['cell'][:160]}")
    print(f"{r['episodes']} episodes, {r['replies']} replies; forfeited {100 * r['forfeit_rate']:.1f}% "
          f"(length cuts {100 * r['length_cut_rate']:.1f}% of replies)")
    print("kinds: " + ", ".join(f"{k} {100 * v:.0f}%" for k, v in sorted(r["kinds"].items(), key=lambda kv: -kv[1])))
    print(f"episodes forfeiting <5%: {100 * r['episodes_lt5']:.0f}%, >50%: {100 * r['episodes_gt50']:.0f}%; "
          f"collapsed ({COLLAPSE_STREAK}+ in a row): {100 * r['collapsed']:.0f}%, onset p50 {r['collapse_onset_p50']}, "
          f"longest streak p90 {r['max_streak_p90']}")
    print(f"forfeit share by {BUCKET}-turn bucket: " + " ".join(f"{x:.2f}" for x in r["by_bucket"][:30]))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="*")
    ap.add_argument("--ledger", action="store_true", help="append each run's report to the ledger")
    ap.add_argument("--show", action="store_true", help="print the ledger (latest entry per run)")
    args = ap.parse_args()
    for run in args.runs:
        r = run_report(run)
        if r is None:
            print(f"== {run}: no traces, skipped", file=sys.stderr)
            continue
        show(r)
        if args.ledger:
            with open(LEDGER, "a") as f:
                f.write(json.dumps(r) + "\n")
    if args.show:
        if not os.path.exists(LEDGER):
            sys.exit(f"no ledger at {LEDGER}")
        latest = {}
        for line in open(LEDGER):
            if line.strip():
                r = json.loads(line)
                latest[r["run"]] = r
        print(f"{'run':>10} {'eps':>5} {'forfeit':>8} {'length':>7} {'<5%':>5} {'>50%':>5} {'collapsed':>9} {'onset':>6}  cell")
        for r in sorted(latest.values(), key=lambda r: r["run"]):
            print(f"{r['run']:>10} {r['episodes']:>5} {100 * r['forfeit_rate']:>7.1f}% {100 * r['length_cut_rate']:>6.1f}% "
                  f"{100 * r['episodes_lt5']:>4.0f}% {100 * r['episodes_gt50']:>4.0f}% {100 * r['collapsed']:>8.0f}% "
                  f"{str(r['collapse_onset_p50']):>6}  {r['cell'][14:110]}")


if __name__ == "__main__":
    main()
