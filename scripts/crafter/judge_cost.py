"""What the LLM judge has cost: per run, and across experiments in one ledger.

    python scripts/crafter/judge_cost.py <run out dir or episodes.jsonl>...   # per-run totals
    python scripts/crafter/judge_cost.py <run out dir> --append-ledger         # also record them
    python scripts/crafter/judge_cost.py --ledger                              # every run recorded so far

Each judge line in a run's episodes.jsonl carries the billed tokens and USD of its
call (`input_tokens`, `output_tokens` incl. thinking, `cost_usd`; in group mode each
branch carries an equal share of the one call). The ledger is one JSON line per run at
$SCRATCH/factory/runs/judge_ledger.jsonl (override with JUDGE_LEDGER); the job scripts
append to it at the end of every run, the replay script after every replay. Prices are
in crafter_env.judge.PRICES (from the Gemini pricing page); the provider's billing page
is the ground truth.
"""

from __future__ import annotations

import argparse
import collections
import datetime as dt
import json
import math
import os
import sys

LEDGER = os.environ.get("JUDGE_LEDGER") or os.path.join(os.environ.get("SCRATCH", "."), "factory/runs/judge_ledger.jsonl")
FIELDS = ("input_tokens", "output_tokens", "thinking_tokens", "cost_usd")


def run_totals(path: str) -> dict:
    log = path if path.endswith(".jsonl") else os.path.join(path, "episodes.jsonl")
    lines = [json.loads(l) for l in open(log) if l.strip()]
    judged = [l for l in lines if l.get("task") == "crafter-judge"]
    segs = [l for l in lines if l.get("task") == "crafter"]
    def val(j, k):  # a CAPI judge (Copilot quota, no per-token price) logs cost_usd = NaN
        v = float(j.get(k) or 0)
        return 0.0 if math.isnan(v) else v

    tot = {k: sum(val(j, k) for j in judged) for k in FIELDS}
    calls = sum(1.0 / max(1.0, float(j.get("group_size") or 1)) if j.get("mode") == "group" else 1.0 for j in judged)
    priced = sum(1 for j in judged if "cost_usd" in j and not math.isnan(float(j["cost_usd"] or 0)))
    out_dir = os.path.dirname(log)
    cell = open(os.path.join(out_dir, "cell.txt")).read().strip() if os.path.exists(os.path.join(out_dir, "cell.txt")) else ""
    return {
        "at": max((j.get("at", 0) for j in judged), default=0) or os.path.getmtime(log),
        "run": out_dir, "job": os.path.basename(out_dir.rstrip("/")), "kind": "run",
        "modes": dict(collections.Counter(j.get("mode", "single") for j in judged)),
        "segments": len(segs), "judged": len(judged), "priced": priced, "calls": round(calls, 2), **tot,
        "usd_per_segment": tot["cost_usd"] / max(1, len(judged)), "cell": cell[:200],
    }


def show(t: dict) -> None:
    unpriced = t["judged"] - t["priced"]
    print(f"{t['job']}: {t['judged']} segments judged in {t['calls']:.0f} calls {t['modes']}; "
          f"input {t['input_tokens'] / 1e6:.2f}M, output {t['output_tokens'] / 1e6:.2f}M tokens (thinking {t['thinking_tokens'] / 1e6:.2f}M); "
          f"${t['cost_usd']:.2f} (${t['usd_per_segment'] * 1000:.2f} per 1000 segments)"
          + (f"; {unpriced} judge lines carry no price (a CAPI judge, or logs from before cost logging)" if unpriced else ""))


def ledger() -> None:
    if not os.path.exists(LEDGER):
        print(f"no ledger yet at {LEDGER}")
        return
    rows = [json.loads(l) for l in open(LEDGER) if l.strip()]
    latest = {}
    for r in rows:  # a run appended twice counts once (the latest)
        latest[(r.get("kind"), r.get("run") or r.get("note"), r.get("at") if r.get("kind") != "run" else None)] = r
    rows = sorted(latest.values(), key=lambda r: r.get("at", 0))
    by_day = collections.defaultdict(float)
    by_kind = collections.defaultdict(float)
    for r in rows:
        day = dt.datetime.fromtimestamp(r.get("at", 0)).strftime("%Y-%m-%d")
        by_day[day] += r["cost_usd"]
        by_kind[r.get("kind", "run")] += r["cost_usd"]
        print(f"  {day} {r.get('kind', 'run'):7s} {r.get('job') or r.get('note', ''):24.24s} {r.get('judged', r.get('calls', 0)):6.0f} judged  "
              f"in {r['input_tokens'] / 1e6:6.2f}M out {r['output_tokens'] / 1e6:6.2f}M  ${r['cost_usd']:8.2f}")
    total = sum(r["cost_usd"] for r in rows)
    print(f"total ${total:.2f} over {len(rows)} entries; by day {dict((k, round(v, 2)) for k, v in by_day.items())}; "
          f"by kind {dict((k, round(v, 2)) for k, v in by_kind.items())}")


def append(entry: dict) -> None:
    os.makedirs(os.path.dirname(LEDGER), exist_ok=True)
    with open(LEDGER, "a") as f:
        f.write(json.dumps(entry) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="*")
    ap.add_argument("--append-ledger", action="store_true")
    ap.add_argument("--ledger", action="store_true")
    args = ap.parse_args()
    for r in args.runs:
        t = run_totals(r)
        show(t)
        if args.append_ledger and t["judged"]:
            append(t)
    if args.ledger or not args.runs:
        ledger()


if __name__ == "__main__":
    sys.exit(main())
