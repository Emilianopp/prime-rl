"""Tables over blockstats.py rows for the first qv chain (blocks A-I): achievements,
deaths, memory and summaries per block. usage: blocktable.py RL_QV_DIR"""
import collections
import json
import statistics as st
import sys

K = sys.argv[1]
rows = json.load(open(f"{K}/blockstats.json"))
# the first chain (rl-qv) blocks A-I: job ids under $SCRATCH/kv-race/rl/rl-qv, 4 updates each
dirs = {"A": "11524553", "B": "11530252", "C": "11530254", "D": "11533232", "E": "11535773", "F": "11537664", "G": "11539189", "H": "11540951", "I": "11541997"}
upd = {"A": "0-3", "B": "4-7", "C": "8-11", "D": "12-15", "E": "16-19", "F": "20-23", "G": "24-27", "H": "28-31", "I": "32-33 (1h41)"}
BL = "ABCDEFGHI"


def m(xs):
    xs = [x for x in xs if x is not None]
    return st.mean(xs) if xs else float("nan")


def state(block, trace, kind="end"):
    try:
        return json.load(open(f"{K}/{dirs[block]}/qv_states/{trace}.{kind}.json"))["state"]
    except FileNotFoundError:
        return None


by = collections.defaultdict(list)
for r in rows:
    by[r["block"]].append(r)

print("=== per block: 64 games, same worlds, from turn 0")
print("blk upd        turns/game  pts@end  ach@end  deaths/100t  pts@turn600  pts@turn1200  forfeit%  act/reply")
games_ach = {}
for b in BL:
    rs = by[b]
    games = collections.defaultdict(list)
    for r in rs:
        if r["idx"] is not None and r["turn1"]:
            games[r["idx"]].append(r)
    fin, at600, at1200, dper = [], [], [], []
    achs = []
    for idx, segs in games.items():
        segs.sort(key=lambda r: r["turn0"] or 0)
        last = segs[-1]
        fin.append(last["turn1"])
        scored = [s for s in segs if s.get("scored")]
        if scored:
            ls = scored[-1]
            stt = state(b, ls["trace"])
            if stt:
                achs.append((len(stt["achievements"]), stt["B"], stt["deaths"], stt["turn"], set(stt["achievements"])))
            for T, acc in ((600, at600), (1200, at1200)):
                before = [s for s in scored if s["turn1"] <= T + 60]
                if before and max(s["turn1"] for s in segs) >= T:
                    acc.append(before[-1]["B1"])
    games_ach[b] = achs
    replies = sum(r["replies"] for r in rs)
    forf = sum(r["forfeit"] for r in rs)
    acts = sum(r["n_actions"] for r in rs)
    turns = m([a[3] for a in achs])
    deaths = m([a[2] for a in achs])
    print(f"{b}   {upd[b]:15} {st.median(fin):6.0f}  {m([a[1] for a in achs]):7.1f}  {m([a[0] for a in achs]):7.1f}"
          f"  {100 * sum(a[2] for a in achs) / max(1, sum(a[3] for a in achs)):9.2f}  {m(at600):9.1f}(n{len(at600)})"
          f"  {m(at1200):8.1f}(n{len(at1200)})  {100 * forf / max(1, replies):6.1f}  {acts / max(1, replies - forf):6.2f}")

print("\n=== achievements unlocked by share of games at end of block")
names = sorted({n for b in BL for a in games_ach[b] for n in a[4]})
for n in names:
    shares = [sum(n in a[4] for a in games_ach[b]) / max(1, len(games_ach[b])) for b in BL]
    if max(shares) > 0.04:
        print(f"  {n:22}" + " ".join(f"{s * 100:4.0f}" for s in shares))

print("\n=== memory and summaries (per segment, matched by game stage = points at segment start)")
print("blk stage   n   mem_ops  writes  entries  mem_tok  summary_words  D(flaws)  flaws/summary")
for stage, lo, hi in (("0-4", 0, 5), ("5-9", 5, 10), ("10+", 10, 999)):
    for b in BL:
        rs = [r for r in by[b] if r.get("scored") and lo <= r["B0"] < hi]
        if len(rs) < 8:
            continue
        ops = m([sum(r["ops"].values()) for r in rs])
        wr = m([r["ops"].get("write", 0) + r["ops"].get("rewrite", 0) + r["ops"].get("replace", 0) for r in rs])
        sw = m([len(r["summary"].split()) for r in rs if r["summary"]])
        print(f"{b}   {stage:5} {len(rs):4}  {ops:6.1f}  {wr:6.1f}  {m([r['mem_entries'] for r in rs]):6.1f}  {m([r['mem_tok'] for r in rs]):7.0f}"
              f"  {sw:10.0f}  {m([r['D'] for r in rs]):8.2f}  {m([len(r['flaws']) for r in rs if r['summary']]):8.2f}")

print("\n=== flaw types per summary")
for b in BL:
    rs = [r for r in by[b] if r.get("scored") and r["summary"]]
    c = collections.Counter(f for r in rs for f in r["flaws"])
    print(b, len(rs), {k: round(v / len(rs), 2) for k, v in c.most_common(7)})

print("\n=== action mix (% of played actions)")
for b in BL:
    c = collections.Counter()
    for r in by[b]:
        c.update(r["acts"])
    tot = sum(c.values())
    top = {k: round(100 * v / tot, 1) for k, v in c.most_common(9)}
    print(b, tot, top)
