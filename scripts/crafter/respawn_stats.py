import json, sys, statistics as st, collections
def pct(xs, q):
    xs = sorted(xs); 
    return xs[min(len(xs)-1, int(q*len(xs)))] if xs else float('nan')
for j in sys.argv[1:]:
    rows = [json.loads(l) for l in open(f"{j}/episodes.jsonl")]
    segs = [r for r in rows if r.get("task") == "crafter"]
    judges = [r for r in rows if r.get("task") != "crafter"]
    deaths = [d for r in segs for d in r["deaths"]]
    turns = sum(r["steps"] for r in segs)
    print(f"\n===== {j}: {len(segs)} segments, {turns} turns, {len(judges)} judge rows")
    print(f" deaths {len(deaths)}  per seg {len(deaths)/len(segs):.2f}  per 100 turns {100*len(deaths)/turns:.2f}")
    print(f" segs with 0 deaths {sum(not r['deaths'] for r in segs)}  with >=10 deaths {sum(len(r['deaths'])>=10 for r in segs)}  max deaths/seg {max(len(r['deaths']) for r in segs)}")
    since = [d["since"] for d in deaths]
    print(f" since-last-respawn: p10 {pct(since,.1)} p50 {pct(since,.5)} p90 {pct(since,.9)};  <=5: {sum(s<=5 for s in since)/len(since):.0%}  <=10: {sum(s<=10 for s in since)/len(since):.0%}  <=20: {sum(s<=20 for s in since)/len(since):.0%}")
    print(f" respawn with hostile within 8: {sum(bool(d['respawn_hostile']) for d in deaths)/len(deaths):.0%}")
    dist = [max(abs(d['pos'][0]-d['respawn'][0]), abs(d['pos'][1]-d['respawn'][1])) for d in deaths]
    print(f" respawn-death distance p10 {pct(dist,.1)} p50 {pct(dist,.5)} p90 {pct(dist,.9)}; same cell {sum(x==0 for x in dist)/len(dist):.0%}")
    print(f" causes {dict(collections.Counter(d['cause'] for d in deaths))}; at night {sum(d['daylight']<0.5 for d in deaths)/len(deaths):.0%}")
    # death chains: consecutive deaths whose death pos within 3 of the previous respawn
    chain = 0
    for r in segs:
        ds = r["deaths"]
        for a, b in zip(ds, ds[1:]):
            if max(abs(b['pos'][0]-a['respawn'][0]), abs(b['pos'][1]-a['respawn'][1])) <= 3: chain += 1
    print(f" deaths within 3 tiles of the previous respawn: {chain} ({chain/max(1,len(deaths)):.0%})")
    # vitals at death (starvation proxies)
    print(f" deaths with food/drink/energy <=2 at death: {sum(min(d['vitals']['food'],d['vitals']['drink'],d['vitals']['energy'])<=2 for d in deaths)/len(deaths):.0%}")
    # achievements
    sa = [r["segment_achievements"] for r in segs]; na = [len(x) for x in sa]
    print(f" segment achievements mean {st.mean(na):.2f} p50 {pct(na,.5)} max {max(na)}; unlocked total mean {st.mean(len(r['achievements_unlocked']) for r in segs):.2f} max {max(len(r['achievements_unlocked']) for r in segs)}")
    print(f" achievement counts: {dict(collections.Counter(a for x in sa for a in x).most_common())}")
    rw = [r["reward"] for r in segs]
    print(f" reward mean {st.mean(rw):.3f} p10 {pct(rw,.1):.3f} p50 {pct(rw,.5):.3f} p90 {pct(rw,.9):.3f}; death penalty share: mean deaths*0.05 = {0.05*len(deaths)/len(segs):.3f}")
    jr = [r.get("judge") or r.get("score") for r in judges]
    print(f" survived at cut {st.mean(r['survived'] for r in segs):.0%}; depth {dict(collections.Counter(r['depth'] for r in segs))}; malformed mean {st.mean(r['malformed_replies'] for r in segs):.1f} invalid {st.mean(r['invalid_actions'] for r in segs):.1f}")
    bydepth = collections.defaultdict(list)
    for r in segs: bydepth[r['depth']].append(len(r['deaths']))
    print(f" deaths/seg by depth: { {k: round(st.mean(v),1) for k,v in sorted(bydepth.items())} }")
    print(f" tools at end: pickaxe {sum(r['inventory']['wood_pickaxe']>0 for r in segs)} sword {sum(r['inventory']['wood_sword']>0 for r in segs)} ; table placed {sum('place_table' in r['achievements_unlocked'] for r in segs)} sleep {sum('wake_up' in r['achievements_unlocked'] for r in segs)}")
    print(" judge row keys:", list(judges[0].keys())[:20] if judges else None)
