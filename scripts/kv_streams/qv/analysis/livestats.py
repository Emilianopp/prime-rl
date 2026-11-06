"""Matched early-play stats: new block (rl-qv-mem) vs first-run block A (rl-qv), same base
weights, same 64 worlds. For each world, A's first segment is cut to the same number of
replies the new block has so far. Also: A's finished segments (replies per segment, edits
at the reset reply vs ordinary replies, reads right after a reset).

usage: livestats.py NEW_LIVE_DIR A_LIVE_DIR"""
import collections
import glob
import json
import re
import statistics as st
import sys

OPS = re.compile(r"<(write|rewrite|replace|delete|read)\b")
EDIT = {"write", "rewrite", "replace", "delete"}
ACT = re.compile(r"<action>\s*([A-Z_]+)\s*</action>")
SUM = re.compile(r"<summary>", re.I)
TURN = re.compile(r"turn (\d+)\]")
MEMTOK = re.compile(r"\((\d+) tok\)")
PAIR = re.compile(r"\(\d+\s*,\s*\d+\)")
BODY = re.compile(r"<(?:write|rewrite)\b[^>]*>(.*?)</(?:write|rewrite)>", re.S)


def load(path):
    idx, nodes = None, []
    for line in open(path):
        if line.strip():
            d = json.loads(line)
            if "open" in d:
                idx = d["open"]["task"]["data"]["idx"]
            nodes += d.get("nodes", [])
    return idx, nodes


def txt(m):
    c = m.get("content")
    return c if isinstance(c, str) else ""


def segs(live):
    out = []
    for p in glob.glob(f"{live}/*.jsonl"):
        idx, nodes = load(p)
        if idx is None or not nodes:
            continue
        msgs = [n["message"] for n in nodes]
        users = [m for m in msgs if m["role"] == "user"]
        t0 = next((int(t.group(1)) for u in users for t in [TURN.search(txt(u))] if t), None)
        toks = sum(len(n.get("token_ids") or []) for n in nodes if n["message"]["role"] != "system")
        out.append(dict(idx=idx, msgs=msgs, t0=t0, toks=toks,
                        vis=sum("[Visible]" in txt(u) for u in users), users=len(users)))
    return out


def reply_stats(msgs, limit=None):
    s = collections.Counter()
    asst = [m for m in msgs if m["role"] == "assistant"]
    if limit is not None:
        asst = asst[:limit]
    for m in asst:
        c = txt(m)
        ops = OPS.findall(c)
        s["replies"] += 1
        s["edits"] += sum(o in EDIT for o in ops)
        s["reads"] += ops.count("read")
        s["reply_edit"] += any(o in EDIT for o in ops)
        s["coords"] += sum(len(PAIR.findall(b)) for b in BODY.findall(c))
        if SUM.search(c):
            s["summary"] += 1
        elif not ACT.search(c):
            s["forfeit"] += 1
        s["acts"] += len(ACT.findall(c))
    # memory size seen in the last observation within the window
    users = [m for m in msgs if m["role"] == "user"]
    if limit is not None:
        users = users[:limit + 1]
    last = next((txt(u) for u in reversed(users) if "[Memory]" in txt(u)), "")
    line = last[last.find("[Memory]"):].split("\n", 1)[0] if last else ""
    s["mem_tok"] = sum(int(x) for x in MEMTOK.findall(line))
    s["mem_keys"] = len(MEMTOK.findall(line))
    return s


def row(name, ss, n):
    r = max(1, ss["replies"])
    print(f"{name:26} games {n:3}  replies {ss['replies']:5}  edits/100rep {100 * ss['edits'] / r:5.1f}"
          f"  %rep_w_edit {100 * ss['reply_edit'] / r:4.1f}  reads/100rep {100 * ss['reads'] / r:5.1f}"
          f"  coords/100rep {100 * ss['coords'] / r:5.1f}  %forfeit {100 * ss['forfeit'] / r:4.1f}"
          f"  act/rep {ss['acts'] / r:4.2f}  mem tok/game {ss['mem_tok'] / max(1, n):6.0f}"
          f"  games w/ mem {ss['has_mem']}/{n}")


def main():
    new, a = segs(sys.argv[1]), segs(sys.argv[2])
    first_new = {s["idx"]: s for s in new if s["t0"] in (0, 1)}
    first_a = {s["idx"]: s for s in a if s["t0"] in (0, 1)}
    worlds = sorted(set(first_new) & set(first_a))
    tn, ta = collections.Counter(), collections.Counter()
    for w in worlds:
        k = sum(m["role"] == "assistant" for m in first_new[w]["msgs"])
        x, y = reply_stats(first_new[w]["msgs"]), reply_stats(first_a[w]["msgs"], k)
        x["has_mem"], y["has_mem"] = x["mem_keys"] > 0, y["mem_keys"] > 0
        tn.update(x), ta.update(y)
    print(f"MATCHED first segment, {len(worlds)} worlds, A cut to the new block's reply count per world")
    row("new  (g7_vis + Visible)", tn, len(worlds))
    row("A    (default prompt)", ta, len(worlds))
    vis = sum(s["vis"] for s in new), sum(s["users"] for s in new)
    print(f"new block: [Visible] in {vis[0]}/{vis[1]} observations; tokens/reply "
          f"{sum(s['toks'] for s in new) / max(1, sum(reply_stats(s['msgs'])['replies'] for s in new)):.0f}")

    # A's finished segments
    done = [s for s in a if reply_stats(s["msgs"])["summary"]]
    reps = [reply_stats(s["msgs"])["replies"] for s in done]
    print(f"\nFIRST RUN block A: {len(done)} finished segments; replies/segment median {st.median(reps):.0f} "
          f"(p10 {sorted(reps)[len(reps) // 10]}, p90 {sorted(reps)[9 * len(reps) // 10]}); "
          f"tokens/reply {sum(s['toks'] for s in done) / max(1, sum(reps)):.0f}")
    at_reset = ordinary = rep_ord = 0
    for s in done:
        asst = [m for m in s["msgs"] if m["role"] == "assistant"]
        for m in asst:
            e = sum(o in EDIT for o in OPS.findall(txt(m)))
            if SUM.search(txt(m)):
                at_reset += e
            else:
                ordinary += e
                rep_ord += 1
    print(f"  edits at reset replies {at_reset} ({at_reset / max(1, len(done)):.2f}/reset); "
          f"ordinary replies {ordinary} ({100 * ordinary / max(1, rep_ord):.1f}/100 replies)")
    later = [s for s in a if s["t0"] not in (0, 1, None)]
    rr = sum(any("read" in OPS.findall(txt(m)) for m in [x for x in s["msgs"] if x["role"] == "assistant"][:3]) for s in later)
    print(f"  segments after a reset: {len(later)}; read memory in first 3 replies: {rr}")


if __name__ == "__main__":
    main()
