"""Score a memory-prompt round (round.sh): per variant, how the model used its memory and
how it played, from the eval's traces (finished episodes: traces/stream/*.jsonl.zst, else
traces/live/*.jsonl) and episode log. The final memory is rebuilt by replaying every
reply through verifiers' Memory, as the harness applied it; each variant's dir gets
memory_samples.txt (final memories and post-reset openings) for reading.

usage: score.py ROUND_OUT_DIR
"""

import collections
import glob
import json
import os
import re
import statistics as st
import subprocess
import sys

from verifiers.v1.memory import Memory

OP = re.compile(r"<(write|rewrite|replace|delete|read)\b")
ACT = re.compile(r"<action>\s*([A-Z_]+)\s*</action>")
SUMMARY = re.compile(r"<summary>", re.I)


def episodes(vdir):
    """{episode id: [segment traces in order]} with each trace's messages."""
    out = {}
    for p in sorted(glob.glob(f"{vdir}/run/*/monitors/file/traces/stream/*.jsonl.zst")):
        raw = subprocess.run(["zstd", "-dc", p], capture_output=True, check=True).stdout
        for line in raw.splitlines():
            rec = json.loads(line)
            segs = sorted(rec["traces"], key=lambda t: t.get("info", {}).get("segment", 0))
            out[rec["id"]] = [(rec["task"]["data"]["idx"], [n["message"] for n in t["nodes"]]) for t in segs]
    live = collections.defaultdict(lambda: [None, []])
    for p in glob.glob(f"{vdir}/run/*/monitors/file/traces/live/*.jsonl"):
        idx, nodes = None, []
        for line in open(p):
            if line.strip():
                d = json.loads(line)
                if "open" in d:
                    idx = d["open"]["task"]["data"]["idx"]
                nodes += d.get("nodes", [])
        if idx is not None and nodes:
            live[(idx, os.path.basename(p))] = [idx, [n["message"] for n in nodes]]
    if not out and live:  # unfinished: one trace file per segment, ordered by first turn
        by_game = collections.defaultdict(list)
        for idx, msgs in live.values():
            by_game[idx].append(msgs)
        for idx, segs in by_game.items():
            segs.sort(key=lambda ms: _first_turn(ms))
            out[f"live-{idx}"] = [(idx, ms) for ms in segs]
    return out


def _first_turn(msgs):
    for m in msgs:
        t = re.search(r"turn (\d+)\]", m.get("content") or "") if m["role"] == "user" else None
        if t:
            return int(t.group(1))
    return 0


def text(m):
    c = m.get("content")
    return c if isinstance(c, str) and c != "None" else ""


def score_variant(vdir):
    eps = episodes(vdir)
    s = collections.Counter()
    mstats = collections.Counter()
    final_tok, final_keys, post_reset_read, resets = [], [], 0, 0
    samples = []
    for eid, segs in eps.items():
        mem = Memory(max_key_tokens=4096)
        for k, (idx, msgs) in enumerate(segs):
            replies = [m for m in msgs if m["role"] == "assistant"]
            for j, m in enumerate(replies):
                c = text(m)
                s["replies"] += 1
                ops = OP.findall(c)
                s["ops"] += len(ops)
                s["replies_with_op"] += bool(ops)
                for o in ops:
                    s[o] += 1
                s["ignored_in_reasoning"] += len(OP.findall(m.get("reasoning_content") or ""))
                if SUMMARY.search(c):
                    s["summaries"] += 1
                elif not ACT.search(c):
                    s["forfeits"] += 1
                mem.apply(c)
            if k > 0:
                resets += 1
                post_reset_read += any(OP.findall(text(m)) and "read" in OP.findall(text(m)) for m in replies[:3])
        mstats.update(mem.stats)
        toks = {key: len(v) // 4 for key, v in mem.entries.items()}
        final_keys.append(len(toks))
        final_tok.append(sum(toks.values()))
        if len(samples) < 3:
            samples.append((segs[0][0], mem.entries))
    games = []
    for line in open(f"{vdir}/episodes.jsonl") if os.path.exists(f"{vdir}/episodes.jsonl") else []:
        e = json.loads(line)
        games.append((len(e.get("achievements") or []), len(e.get("deaths") or []), e.get("turns")))
    with open(f"{vdir}/memory_samples.txt", "w") as f:
        for idx, entries in samples:
            f.write(f"===== world {idx}: {len(entries)} entries\n")
            for key, v in entries.items():
                f.write(f"--- [{key}] ({len(v) // 4} tok)\n{v.strip()}\n")
            f.write("\n")
    r = max(1, s["replies"])
    return {
        "games": len(eps), "replies/game": s["replies"] / max(1, len(eps)),
        "ops/reply": s["ops"] / r, "%replies_op": 100 * s["replies_with_op"] / r,
        "writes/reply": (s["write"] + s["rewrite"] + s["replace"]) / r, "reads/reply": s["read"] / r,
        "ign_reason/reply": s["ignored_in_reasoning"] / r,
        "%ops_ok": 100 * sum(v for k, v in mstats.items() if k.endswith(" ok")) / max(1, sum(mstats.values())),
        "%reset_read": 100 * post_reset_read / max(1, resets), "resets/game": resets / max(1, len(eps)),
        "keys_end": st.mean(final_keys) if final_keys else 0, "tok_end": st.mean(final_tok) if final_tok else 0,
        "%forfeit": 100 * s["forfeits"] / r,
        "ach": st.mean(g[0] for g in games) if games else float("nan"),
        "deaths": st.mean(g[1] for g in games) if games else float("nan"),
        "turns": st.mean(g[2] or 0 for g in games) if games else float("nan"),
    }


def main():
    out = sys.argv[1]
    rows = {}
    for vdir in sorted(glob.glob(f"{out}/*/")):
        name = os.path.basename(vdir.rstrip("/"))
        try:
            rows[name] = score_variant(vdir.rstrip("/"))
        except Exception as e:
            print(f"{name}: {e!r}", file=sys.stderr)
    cols = ["games", "replies/game", "ops/reply", "%replies_op", "writes/reply", "reads/reply", "ign_reason/reply", "%ops_ok",
            "%reset_read", "resets/game", "keys_end", "tok_end", "%forfeit", "ach", "deaths", "turns"]
    print("variant".ljust(14) + "".join(c.rjust(11) for c in cols))
    for name, r in sorted(rows.items(), key=lambda kv: -kv[1]["ops/reply"]):
        print(name.ljust(14) + "".join(f"{r[c]:11.2f}" for c in cols))
    json.dump(rows, open(f"{out}/score.json", "w"), indent=1)


if __name__ == "__main__":
    main()
