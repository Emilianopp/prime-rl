"""Block-by-block comparison: first RL run (rl-qv, default memory prompt) vs the memory run
(rl-qv-mem, g7_vis prompt + [Visible]). Each block replays the same 64 worlds from turn 0.

usage: cmp.py LABEL:BLOCK_DIR ..."""
import collections
import glob
import json
import os
import re
import statistics as st
import sys

OPS = re.compile(r"<(write|rewrite|replace|delete|read)\b")
EDIT = {"write", "rewrite", "replace", "delete"}
ACT = re.compile(r"<action>\s*([A-Z_]+)\s*</action>")
SUM = re.compile(r"<summary>", re.I)
TURN = re.compile(r"turn (\d+)\]")
MEMTOK = re.compile(r"\((\d+) tok\)")


def txt(m):
    c = m.get("content")
    return c if isinstance(c, str) else ""


def m(xs):
    xs = [x for x in xs if x is not None]
    return st.mean(xs) if xs else float("nan")


def block(bdir):
    qv = {}
    if os.path.exists(f"{bdir}/qv.jsonl"):
        for l in open(f"{bdir}/qv.jsonl"):
            r = json.loads(l)
            qv[r["trace"]] = r
    segs = []
    for p in glob.glob(f"{bdir}/run/*/monitors/file/traces/live/*.jsonl"):
        tid = os.path.basename(p)[:-6]
        idx, nodes = None, []
        try:
            for line in open(p):
                if line.strip():
                    d = json.loads(line)
                    if "open" in d:
                        idx = d["open"]["task"]["data"]["idx"]
                    nodes += d.get("nodes", [])
        except Exception:
            continue
        msgs = [n["message"] for n in nodes]
        asst = [x for x in msgs if x["role"] == "assistant"]
        users = [x for x in msgs if x["role"] == "user"]
        turns = [int(t.group(1)) for u in users for t in [TURN.search(txt(u))] if t]
        s = dict(idx=idx, tid=tid, replies=len(asst), t0=min(turns) if turns else None, t1=max(turns) if turns else None,
                 summ=any(SUM.search(txt(a)) for a in asst), q=qv.get(tid))
        s["edits_reset"] = sum(sum(o in EDIT for o in OPS.findall(txt(a))) for a in asst if SUM.search(txt(a)))
        s["edits_ord"] = sum(sum(o in EDIT for o in OPS.findall(txt(a))) for a in asst if not SUM.search(txt(a)))
        s["reads"] = sum(OPS.findall(txt(a)).count("read") for a in asst)
        s["read_first3"] = any("read" in OPS.findall(txt(a)) for a in asst[:3])
        s["forfeit"] = sum(not SUM.search(txt(a)) and not ACT.search(txt(a)) for a in asst)
        first = next((txt(u) for u in users if "[Memory]" in txt(u)), "")
        s["opened_with_mem"] = "[Memory]" in first and "no entries yet" not in first.split("\n", 1)[0]
        last = next((txt(u) for u in reversed(users) if "[Memory]" in txt(u)), "")
        line = last[last.find("[Memory]"):].split("\n", 1)[0] if last else ""
        s["mem_tok"] = sum(int(x) for x in MEMTOK.findall(line))
        segs.append(s)
    return segs


def state(bdir, tid):
    try:
        return json.load(open(f"{bdir}/qv_states/{tid}.end.json"))["state"]
    except FileNotFoundError:
        return None


def main():
    print("blk  updates  segs  R_mean  ach@end  pts@end  deaths/100t  forfeit%  | edits/reset  edits/100rep  "
          "%reset_w_mem  %read_after_reset(mem nonempty)  mem_tok@end  cost")
    for spec in sys.argv[1:]:
        label, upd, bdir = spec.split(":")
        segs = block(bdir)
        scored = [s for s in segs if s["q"] and "B" in s["q"]["info"]]
        games = collections.defaultdict(list)
        for s in scored:
            games[s["idx"]].append(s)
        achs = []
        for idx, ss in games.items():
            ss.sort(key=lambda s: s["t0"] or 0)
            stt = state(bdir, ss[-1]["tid"])
            if stt:
                achs.append((len(stt["achievements"]), stt["B"], stt["deaths"], stt["turn"]))
        done = [s for s in segs if s["summ"]]
        later = [s for s in segs if s["t0"] not in (0, 1, None)]
        later_mem = [s for s in later if s["opened_with_mem"]]
        rep = sum(s["replies"] for s in segs)
        ordrep = rep - len(done)
        cost = 0.0
        if os.path.exists(f"{bdir}/qv.jsonl"):
            for l in open(f"{bdir}/qv.jsonl"):
                cost = max(cost, json.loads(l).get("info", {}).get("cost", 0) or 0)
        print(f"{label:4} {upd:8} {len(scored):5}  {m([s['q']['reward'] for s in scored]):6.3f}  {m([a[0] for a in achs]):6.2f}"
              f"  {m([a[1] for a in achs]):6.1f}  {100 * sum(a[2] for a in achs) / max(1, sum(a[3] for a in achs)):9.2f}"
              f"  {100 * sum(s['forfeit'] for s in segs) / max(1, rep):7.1f}  | {m([s['edits_reset'] for s in done]):8.2f}"
              f"  {100 * sum(s['edits_ord'] for s in segs) / max(1, ordrep):10.2f}  {100 * len(later_mem) / max(1, len(later)):9.0f}"
              f"  {100 * m([s['read_first3'] for s in later] or [0]):9.0f}% ({100 * m([s['read_first3'] for s in later_mem] or [0]):3.0f}%)"
              f"  {m([s['mem_tok'] for s in done]):10.0f}  ${cost:.2f}")


if __name__ == "__main__":
    main()
