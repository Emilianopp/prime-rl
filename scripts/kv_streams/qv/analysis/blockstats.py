"""Per-block play statistics for the qv chain: each block replays the same 64 worlds
(craftax#train0-63) from turn 0, so blocks compare policies on identical games.

usage: blockstats.py OUT_JSON  block_dir:label ...
Writes per-segment rows (json) for later tables; prints a summary."""
import collections
import glob
import json
import os
import re
import statistics as st
import sys

MEM_OPS = ("write", "rewrite", "replace", "delete", "read")
op_re = {k: re.compile(rf"<{k}\b") for k in MEM_OPS}
act_re = re.compile(r"<action>\s*([A-Z_]+)\s*</action>")
sum_re = re.compile(r"<summary>(.*?)</summary>", re.S)
mem_block = re.compile(r"\[Memory\](.*?)(?:\n\[Legend\]|\n\[Current Observation)", re.S)


def trace_rows(path):
    idx, nodes = None, []
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            if "open" in d:
                idx = d["open"]["task"]["data"]["idx"]
            nodes += d.get("nodes", [])
    msgs = [n["message"] for n in nodes]
    asst = [m for m in msgs if m["role"] == "assistant"]
    users = [m for m in msgs if m["role"] == "user"]
    ops = collections.Counter()
    acts = collections.Counter()
    forfeit = 0
    summary = None
    for m in asst:
        c = m.get("content") or ""
        for k, r in op_re.items():
            ops[k] += len(r.findall(c))
        a = act_re.findall(c)
        acts.update(a)
        s = sum_re.search(c)
        if s:
            summary = s.group(1).strip()
        elif not a:
            forfeit += 1
    turns = []
    for u in users:
        t = re.search(r"\[Current Observation — turn (\d+)\]", u.get("content") or "")
        if t:
            turns.append(int(t.group(1)))
    mem_entries = mem_tok = 0
    if users:
        c = users[-1].get("content") or ""
        line = c[c.find("[Memory]"):].split("\n", 1)[0] if "[Memory]" in c else ""
        if line and "no entries yet" not in line:
            toks = [int(x) for x in re.findall(r"\((\d+) tok\)", line)]
            mem_entries, mem_tok = len(toks), sum(toks)
    return dict(idx=idx, replies=len(asst), forfeit=forfeit, ops=dict(ops), n_actions=sum(acts.values()),
                acts=dict(acts), turn0=min(turns) if turns else None, turn1=max(turns) if turns else None,
                summary=summary, mem_tok=mem_tok, mem_entries=mem_entries)


def main():
    out = sys.argv[1]
    rows = []
    for spec in sys.argv[2:]:
        bdir, label = spec.split(":")
        qv = {}
        for l in open(f"{bdir}/qv.jsonl"):
            r = json.loads(l)
            qv[r["trace"]] = r
        tdir = glob.glob(f"{bdir}/run/*/monitors/file/traces/live")[0]
        for p in glob.glob(f"{tdir}/*.jsonl"):
            tid = os.path.basename(p)[:-6]
            try:
                row = trace_rows(p)
            except Exception as e:  # a trace still being written
                continue
            q = qv.get(tid)
            row.update(block=label, trace=tid, scored=bool(q and "B" in q["info"]))
            if q and "B" in q["info"]:
                i = q["info"]
                row.update(policy=q["policy"], R=q["reward"], B0=i["B"][0], B1=i["B"][1],
                           d0=i["deaths"][0], d1=i["deaths"][1], D=i["end"]["D"], flaws=i["flaws"],
                           episode=q["episode"])
            rows.append(row)
        print(label, len([r for r in rows if r["block"] == label]), "traces", file=sys.stderr)
    json.dump(rows, open(out, "w"))


if __name__ == "__main__":
    main()
