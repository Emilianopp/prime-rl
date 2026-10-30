import json, sys, statistics as st, collections, re
for j in sys.argv[1:]:
    rows = [json.loads(l) for l in open(f"{j}/episodes.jsonl")]
    segs = [r for r in rows if r.get("task") == "crafter"]
    judges = {r["node_id"]: r for r in rows if r.get("task") != "crafter"}
    print(f"\n===== {j}")
    # within-group spread: deaths penalty vs achievement vs judge
    groups = collections.defaultdict(list)
    for r in segs: groups[(r["parent"], r["depth"])].append(r)
    sd_d, sd_a, sd_j = [], [], []
    for g in groups.values():
        if len(g) < 2: continue
        sd_d.append(st.pstdev(0.05*len(r["deaths"]) for r in g))
        sd_a.append(st.pstdev(len(r["segment_achievements"])/22 for r in g))
        js = [judges[r["node_id"]]["judge_score"] for r in g if r["node_id"] in judges]
        if len(js) > 1: sd_j.append(st.pstdev(js))
    print(f" within-node std: death penalty {st.mean(sd_d):.3f}  achievement term {st.mean(sd_a):.3f}  judge {st.mean(sd_j):.3f}  ({len(sd_d)} groups)")
    # judge vs deaths correlation
    xs = [(len(r["deaths"]), judges[r["node_id"]]["judge_score"], len(r["segment_achievements"])) for r in segs if r["node_id"] in judges]
    def corr(a, b):
        ma, mb = st.mean(a), st.mean(b); 
        return sum((x-ma)*(y-mb) for x,y in zip(a,b)) / (len(a)*st.pstdev(a)*st.pstdev(b))
    print(f" corr(judge, deaths) {corr([x[0] for x in xs],[x[1] for x in xs]):.2f}  corr(judge, seg achievements) {corr([x[2] for x in xs],[x[1] for x in xs]):.2f}  corr(deaths, seg ach) {corr([x[0] for x in xs],[x[2] for x in xs]):.2f}")
    # malformed distribution
    m = sorted(r["malformed_replies"] for r in segs)
    print(f" malformed p50 {m[len(m)//2]} p90 {m[int(.9*len(m))]} max {m[-1]}; segs with >50: {sum(x>50 for x in m)}")
    # event types mentioning respawn / base / sleep
    ev = collections.Counter()
    for r in segs:
        for e in r["events"]:
            ev[re.sub(r"step \d+: ", "", e).split(" at ")[0][:40]] += 1
    print(" event kinds:", dict(ev.most_common(14)))
    # deaths that came within 10 turns of a respawn at night
    deaths = [d for r in segs for d in r["deaths"]]
    print(f" deaths by daylight bucket: { {b: sum(int(d['daylight']*5)==b for d in deaths) for b in range(5)} } (0 = darkest)")
    # turns from the first death of a segment to the cut, i.e. how much of a segment is 'after the first death'
    ft = [r["steps"] - r["deaths"][0]["step"] for r in segs]
    print(f" turns after first death per seg: mean {st.mean(ft):.0f} of {st.mean(r['steps'] for r in segs):.0f}")
    # sample a death-loop segment: the worst
    w = max(segs, key=lambda r: len(r["deaths"]))
    print(f" worst segment seed {w['seed']} depth {w['depth']}: {len(w['deaths'])} deaths; steps/pos/respawn/since/hostile:")
    for d in w["deaths"][:12]: print("   ", d["step"], d["pos"], d["respawn"], d["since"], d["cause"], "night" if d["daylight"]<0.5 else "day", d["respawn_hostile"])
