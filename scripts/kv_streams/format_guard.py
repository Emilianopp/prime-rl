"""Watch a turn_credit run from inside its job and stop it on rules fixed before launch;
keep its weights.

Reads the algorithm's per-episode log (`[...algo] log_path`) in windows of `--window`
finished episodes. The first `--baseline` windows set the reference (forfeits per game
turn f0, game reward per episode r0, sampled tokens per game turn l0); each later
window is compared through the mean of the last two windows:

  regressed     r < --min-reward x r0                    the play got worse
  drifted       l outside [1/--max-drift, --max-drift] x l0   the policy changed beyond format
  done          f <= --done-forfeit                     the format is fixed: stop early
  stalled       f > f0 - --min-gain after --patience windows   the format is not improving

On a stop it writes the reason to `<out>/STOP` (rl.sbatch's WATCH contract ends the
run) and exits. Every finished weight broadcast is hard-linked into `<out>/weights/
step_N` (every `--keep-every`-th kept, plus the latest), and `<out>/weights/final`
points at the latest.

    python format_guard.py --log <out>/format.jsonl --out <out> --runs <out>/run
"""

import argparse
import json
import os
import time
from collections import Counter
from pathlib import Path

FORFEITS = ("no_action_tag", "unknown_name", "action_in_reasoning", "think_tag_in_reply", "unclosed_action_tag")


def windows(log: Path, size: int) -> list[dict]:
    if not log.exists():
        return []
    episodes = [json.loads(line) for line in log.read_text().splitlines() if line.strip()]
    out = []
    for start in range(0, len(episodes) - size + 1, size):
        chunk = episodes[start : start + size]
        reasons = sum((Counter(e["reasons"]) for e in chunk), Counter())
        turns = max(1, sum(e["replies"] for e in chunk))
        out.append(
            {
                "episodes": f"{start + 1}-{start + size}",
                "policy": [min(e["policy"] or 0 for e in chunk), max(e["policy"] or 0 for e in chunk)],
                "forfeit": sum(reasons[r] for r in FORFEITS) / turns,
                "reward": sum(e["reward"] for e in chunk) / size,
                "tokens": sum(e["tokens"] for e in chunk) / max(1, sum(e["replies"] for e in chunk)),
                "reasons": dict(reasons.most_common()),
            }
        )
    return out


def verdict(ws: list[dict], args) -> str | None:
    if len(ws) < args.baseline + 2:
        return None
    base = ws[: args.baseline]
    f0 = sum(w["forfeit"] for w in base) / len(base)
    r0 = sum(w["reward"] for w in base) / len(base)
    l0 = sum(w["tokens"] for w in base) / len(base)
    last = ws[-2:]
    f = sum(w["forfeit"] for w in last) / 2
    r = sum(w["reward"] for w in last) / 2
    length = sum(w["tokens"] for w in last) / 2
    if r0 > 0 and r < args.min_reward * r0:
        return f"regressed: reward {r:.4f} < {args.min_reward} x {r0:.4f}"
    if not l0 / args.max_drift <= length <= l0 * args.max_drift:
        return f"drifted: tokens per reply {length:.0f} vs {l0:.0f}"
    if f <= args.done_forfeit:
        return f"done: forfeits {f:.3f} <= {args.done_forfeit} (from {f0:.3f})"
    if len(ws) - args.baseline >= args.patience and f > f0 - args.min_gain:
        return f"stalled: forfeits {f:.3f} not below {f0:.3f} - {args.min_gain} after {len(ws) - args.baseline} windows"
    return None


def keep_weights(runs: Path, weights: Path, keep_every: int) -> None:
    finished = sorted(
        (int(p.parent.name.split("_")[1]), p.parent) for p in runs.glob("**/broadcasts/step_*/.finished")
    )
    for step, src in finished:
        dst = weights / f"step_{step}"
        if dst.exists():
            continue
        tmp = weights / f".step_{step}.tmp"
        tmp.mkdir(parents=True, exist_ok=True)
        for f in src.iterdir():
            if f.is_file() and not f.name.startswith("."):
                try:
                    os.link(f, tmp / f.name)
                except FileExistsError:
                    pass
        tmp.rename(dst)
    kept = sorted(int(p.name.split("_")[1]) for p in weights.glob("step_*"))
    for step in kept[:-1]:
        if step % keep_every:
            for f in (weights / f"step_{step}").iterdir():
                f.unlink()
            (weights / f"step_{step}").rmdir()
    if kept:
        final = weights / "final"
        tmp = weights / ".final.tmp"
        tmp.unlink(missing_ok=True)
        tmp.symlink_to(f"step_{kept[-1]}")
        tmp.replace(final)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--runs", type=Path, required=True, help="where the run's broadcasts/ live")
    ap.add_argument("--window", type=int, default=16, help="episodes per window")
    ap.add_argument("--baseline", type=int, default=2, help="windows that set the reference")
    ap.add_argument("--done-forfeit", type=float, default=0.03)
    ap.add_argument("--min-reward", type=float, default=0.8)
    ap.add_argument("--max-drift", type=float, default=1.6)
    ap.add_argument("--min-gain", type=float, default=0.02)
    ap.add_argument("--patience", type=int, default=6, help="windows after the baseline before 'stalled'")
    ap.add_argument("--keep-every", type=int, default=4)
    ap.add_argument("--interval", type=float, default=30)
    args = ap.parse_args()
    seen = 0
    while True:
        keep_weights(args.runs, args.out / "weights", args.keep_every)
        ws = windows(args.log, args.window)
        for w in ws[seen:]:
            print(json.dumps(w), flush=True)
        seen = len(ws)
        reason = verdict(ws, args)
        if reason:
            (args.out / "STOP").write_text(reason + "\n")
            print(f"STOP {reason}", flush=True)
            return
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
