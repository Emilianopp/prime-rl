"""Judge-reward summary of one qv.jsonl (per policy version: R, dV, dB, D; cumulative cost).

usage: qvstat.py QV_JSONL"""
import json, statistics as st, sys
rows = [json.loads(l) for l in open(sys.argv[1])]
print("scored", len(rows), "null", sum(r["reward"] is None for r in rows))
by = {}
for r in rows:
    by.setdefault(r.get("policy"), []).append(r)
for pol, ch in sorted(by.items()):
    ch = [r for r in ch if r["reward"] is not None]
    i = [r["info"] for r in ch]
    dV = [x["end"]["V"] - x["start"]["V"] for x in i if x["start"].get("V") is not None and x["end"].get("V") is not None]
    dB = [x["B"][1] - x["B"][0] for x in i if x.get("B")]
    D = [x["end"]["D"] for x in i if x["end"].get("D") is not None]
    R = [r["reward"] for r in ch]
    print(f"policy {pol}: n {len(ch)}  R {st.mean(R):.3f} sd {st.pstdev(R):.2f}  at+1 {sum(x>=.999 for x in R)}  <=0 {sum(x<=0 for x in R)}  dV {st.mean(dV):.2f}  dB {st.mean(dB):.2f}  D {st.mean(D):.2f}")
print("cost %.3f calls %d" % (rows[-1]["info"]["cost"], rows[-1]["info"]["calls"]))
