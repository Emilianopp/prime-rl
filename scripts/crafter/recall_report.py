"""Recall-probe report over beam chains (env.recall_probe = True, RECALL_PROBE=1 in crafter_beam.sbatch):
how well the MEMORY CHECK reply at the cut matches the segment's record, by how long before
the cut each death happened (the turn window evicts what is older than its size).

usage: recall_report.py <chain dir or episodes.jsonl>[:window] [...]   (one block per chain)

The window (turns kept in context) is read off the dir name (recall-w50 -> 50, anything else
-> no eviction) or given after a colon; it splits each segment's deaths into visible and evicted.
The strict measure (a death placed within 10 game steps) is followed by looser ones: a tolerance
sweep, the death count, whether any death is mentioned when every death was evicted, and the
cause alone.
"""
import collections
import json
import os
import re
import sys

_COUNT = re.compile(r"deaths?\s*:\s*(\d+|unknown)", re.I)
_DEATH = re.compile(r"death\s*\d*\s*:\s*step\s*[=:]?\s*(\d+|unknown)\s*,?\s*cause\s*[=:]?\s*([a-z_]+)", re.I)
_CAUSES = {"zombie": "zombie", "skeleton": "skeleton", "arrow": "skeleton", "lava": "lava", "starv": "starved",
           "hunger": "starved", "food": "starved", "thirst": "starved", "drink": "starved", "energy": "starved"}


def claims(reply):
    """The claimed deaths (step or None, cause) and the claimed count, re-parsed from the stored reply."""
    text = re.sub(r"<action>.*?</action>", "", reply or "", flags=re.S)
    cl = [(int(a) if a.isdigit() else None, next((c for k, c in _CAUSES.items() if b.lower().startswith(k)), "unknown")) for a, b in _DEATH.findall(text)]
    m = _COUNT.search(text)
    count = int(m.group(1)) if m and m.group(1).isdigit() else (len(cl) if cl else None)
    return cl, count


def placed(truth, cl, tol):
    """How many of the true deaths (game steps) have a claim within tol, each claim used once, closest first."""
    free = list(cl)
    hit = 0
    for t in truth:
        best = min((c for c in free if c[0] is not None and abs(c[0] - t) <= tol), key=lambda c: abs(c[0] - t), default=None)
        if best is not None:
            free.remove(best)
            hit += 1
    return hit


def loose(parsed, window):
    print(f"  -- looser readings (window {window if window else 'none: nothing evicted'})")
    vis = lambda d: window is None or d["turns_ago"] <= window
    for label, keep in (("visible", vis), ("evicted", lambda d: not vis(d))):
        truth = [[d["game_step"] for d in s["recall"]["deaths"] if keep(d)] for s in parsed]
        n = sum(len(t) for t in truth)
        if not n:
            continue
        cells = []
        for tol in (10, 25, 50, 100):
            hit = sum(placed(t, claims(s["recall"]["reply"])[0], tol) for s, t in zip(parsed, truth))
            cells.append(f"{tol:3d}: {pct(hit, n)}")
        print(f"     {label:>7} deaths placed within N steps | " + " | ".join(cells))
    some = [s for s in parsed if s["recall"]["deaths_true"] > 0]
    if some:
        rows = [(s["recall"]["deaths_true"], sum(vis(d) for d in s["recall"]["deaths"]), claims(s["recall"]["reply"])[1]) for s in some]
        known = [r for r in rows if r[2] is not None]
        print(f"     death count (segments with deaths, {len(some)}): mean true {sum(r[0] for r in rows) / len(rows):.2f}, visible {sum(r[1] for r in rows) / len(rows):.2f},"
              f" claimed {sum(r[2] for r in known) / max(1, len(known)):.2f}; exact {pct(sum(r[2] == r[0] for r in known), len(known))}, within 1 {pct(sum(abs(r[2] - r[0]) <= 1 for r in known), len(known))}")
        ev = [r for r in known if r[0] > r[1]]
        if ev:
            print(f"     segments with an evicted death ({len(ev)}): claimed count above the visible count {pct(sum(r[2] > r[1] for r in ev), len(ev))},"
                  f" at least the true count {pct(sum(r[2] >= r[0] for r in ev), len(ev))}")
        allev = [s for s in some if not any(vis(d) for d in s["recall"]["deaths"])]
        if allev:
            anyd = sum(1 for s in allev if (claims(s["recall"]["reply"])[1] or 0) >= 1)
            print(f"     segments whose deaths were all evicted ({len(allev)}): any death mentioned {pct(anyd, len(allev))}"
                  f"  (death-free segments mention one {pct(sum(1 for s in parsed if s['recall']['deaths_true'] == 0 and (claims(s['recall']['reply'])[1] or 0) >= 1), sum(1 for s in parsed if s['recall']['deaths_true'] == 0))})")
        # the cause alone: an evicted death's cause among the claimed causes once the visible deaths took theirs
        hit = n = 0
        for s in some:
            cl = collections.Counter(c for _, c in claims(s["recall"]["reply"])[0])
            for d in s["recall"]["deaths"]:
                if vis(d) and cl[d["cause"]] > 0:
                    cl[d["cause"]] -= 1
            for d in s["recall"]["deaths"]:
                if not vis(d):
                    n += 1
                    if cl[d["cause"]] > 0:
                        cl[d["cause"]] -= 1
                        hit += 1
        if n:
            print(f"     evicted deaths whose cause appears among the claims (beyond the visible deaths' causes): {pct(hit, n)}")

BUCKETS = [(0, 50, "<=50"), (51, 100, "51-100"), (101, 200, "101-200"), (201, 10**9, ">200")]


def rows(path):
    f = os.path.join(path, "episodes.jsonl") if os.path.isdir(path) else path
    for line in open(f):
        r = json.loads(line)
        if r.get("recall") is not None:
            yield r


def pct(a, b):
    return f"{100 * a / b:5.1f}% ({b})" if b else "    -   (0)"


def report(path):
    path, _, win = path.partition(":")
    m = re.search(r"w(\d+)", os.path.basename(path.rstrip("/")))
    window = int(win) if win else (int(m.group(1)) if m else None)
    segs = list(rows(path))
    print(f"\n== {path}: {len(segs)} probed segments")
    if not segs:
        return
    parsed = [s for s in segs if s["recall"]["parsed"]]
    print(f"  reply parsed (the fixed line form): {pct(len(parsed), len(segs))}")
    none = [s for s in parsed if s["recall"]["deaths_true"] == 0]
    some = [s for s in parsed if s["recall"]["deaths_true"] > 0]
    print(f"  segments with no death: 'deaths: 0' answered {pct(sum(s['recall']['count_correct'] for s in none), len(none))};"
          f" false deaths claimed per segment {sum(s['recall']['false_deaths'] for s in none) / max(1, len(none)):.2f}")
    print(f"  segments with deaths: count right {pct(sum(s['recall']['count_correct'] for s in some), len(some))};"
          f" false deaths per segment {sum(s['recall']['false_deaths'] for s in some) / max(1, len(some)):.2f}")
    by = collections.defaultdict(list)
    for s in parsed:
        for d in s["recall"]["deaths"]:
            for lo, hi, name in BUCKETS:
                if lo <= d["turns_ago"] <= hi:
                    by[name].append(d)
    print("  deaths by turns before the cut | step within 10 | step and cause right")
    for _, _, name in BUCKETS:
        ds = by.get(name, [])
        print(f"    {name:>7} | {pct(sum(d['matched'] for d in ds), len(ds))} | {pct(sum(d['cause_ok'] for d in ds), len(ds))}")
    claimed = sum(len(s["recall"]["achievements"]) for s in parsed)
    true = sum(s["recall"]["achievements_true"] for s in parsed)
    ok = sum(s["recall"]["achievements_step_ok"] for s in parsed)
    false = sum(s["recall"]["false_achievements"] for s in parsed)
    print(f"  achievements: {true} unlocked, {claimed} claimed, step within 10 for {pct(ok, claimed)}, not unlocked this segment {pct(false, claimed)}")
    loose(parsed, window)
    unparsed = [s for s in segs if not s["recall"]["parsed"]]
    if unparsed:
        print(f"  unparsed reply example: {unparsed[0]['recall']['reply'][:160]!r}")


if __name__ == "__main__":
    for p in sys.argv[1:]:
        report(p)
