"""Within-node training signal of a Crafter kve-summary run: what the branch advantages are
made of (achievement delta, judge score, deaths) and which sampled tokens carry them
(action tags, tag-less prose, the notes turn, lessons) under token-level loss aggregation.

usage: signal_analysis.py <run dir> [<run dir> ...]   (a dir with run/*/monitors/file/traces)
"""
import collections
import glob
import io
import json
import math
import re
import subprocess
import sys

A = re.compile(r"<action>\s*(\w+)\s*</action>")


def text(m):
    c = m.get("content")
    return c if isinstance(c, str) else " ".join(x.get("text", "") or "" for x in (c or []) if isinstance(x, dict))


def lines(path):
    blob = subprocess.run(["zstd", "-dc", path], capture_output=True, check=True).stdout
    return io.BytesIO(blob)


def category(content, last_assistant):
    if last_assistant and not A.search(content):
        return "notes"  # the end-of-segment summary turn
    if "<lesson>" in content:
        return "lesson"
    a = A.search(content)
    if a is None:
        return "empty" if len(content.strip()) < 20 else "prose"  # "<" + EOS, or a paragraph with no tag
    if "<notes>" in content:
        return "action+notes"  # spontaneous mid-segment notes next to the action
    return "action" if len(content.strip()) <= 40 else "action+text"


def load(run):
    adv = {}
    for f in glob.glob(f"{run}/run/*/monitors/file/traces/annotations/orch/*.jsonl.zst"):
        for line in lines(f):
            d = json.loads(line)
            adv[d["trace_id"]] = d["info"].get("advantage")
    traces = []
    for f in sorted(glob.glob(f"{run}/run/*/monitors/file/traces/stream/*.jsonl.zst")):
        for line in lines(f):
            d = json.loads(line)
            gid = d["group"]["id"]
            for tr in d["traces"]:
                c = (tr.get("info") or {}).get("crafter")
                if not c:
                    continue
                rw = tr["rewards"]
                comp = {k: v["score"] * v["weight"] for k, v in rw.items()}
                j = c.get("judge") or {}
                rec = dict(id=tr["id"], group=gid, step=tr["info"]["dispatch"]["step"], depth=c["depth"], parent=c["parent"],
                           comp=comp, total=sum(comp.values()), judge=j, deaths=len(c["deaths"]), seg_ach=len(c["segment_achievements"]),
                           steps=c["steps"], adv=adv.get(tr["id"]), cats=collections.defaultdict(lambda: [0, 0.0, 0.0, 0]), decision=[])
                nodes = [n for n in tr["nodes"] if n.get("message") and n["message"]["role"] == "assistant" and n.get("sampled")]
                for i, n in enumerate(nodes):
                    content = text(n["message"])
                    cat = category(content, i == len(nodes) - 1)
                    acc = rec["cats"][cat]
                    acc[3] += 1
                    lps = [lp for lp in (n.get("logprobs") or []) if lp is not None]  # one per masked token, in order
                    for lp in lps:
                        acc[0] += 1
                        acc[1] += 1.0 - math.exp(lp)
                        acc[2] += -lp
                    if cat == "action" and lps:
                        rec["decision"].append(math.exp(min(lps)))  # p of the least likely token: the action choice
                traces.append(rec)
    return traces


def corr(xs, ys):
    n = len(xs)
    if n < 3:
        return float("nan")
    mx, my = sum(xs) / n, sum(ys) / n
    sx = math.sqrt(sum((x - mx) ** 2 for x in xs)) or 1e-12
    sy = math.sqrt(sum((y - my) ** 2 for y in ys)) or 1e-12
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / (sx * sy)


def demean(traces, key):
    """value minus its group mean (what GRPO's advantage sees)."""
    g = collections.defaultdict(list)
    for t in traces:
        g[t["group"]].append(key(t))
    means = {k: sum(v) / len(v) for k, v in g.items()}
    return [key(t) - means[t["group"]] for t in traces]


def report(run):
    traces = load(run)
    untrained = [t for t in traces if t["adv"] is None]  # arrived after the last trainer step
    traces = [t for t in traces if t["adv"] is not None]
    print(f"\n=== {run}\n{len(untrained)} traces without an advantage (never trained on) dropped")
    groups = collections.defaultdict(list)
    for t in traces:
        groups[t["group"]].append(t)
    print(f"{len(traces)} traces in {len(groups)} groups; group sizes {collections.Counter(len(v) for v in groups.values())}; "
          f"same parent within group: {sum(len({t['parent'] for t in v}) == 1 for v in groups.values())}/{len(groups)}; "
          f"with advantage: {sum(t['adv'] is not None for t in traces)}")
    tot = [t["total"] for t in traces]
    print(f"reward: mean {sum(tot)/len(tot):.3f} sd {pstd(tot):.3f}")
    comps = sorted({k for t in traces for k in t["comp"]})
    # within-group vs between-group spread
    gmeans = [sum(t["total"] for t in v) / len(v) for v in groups.values()]
    within = demean(traces, lambda t: t["total"])
    print(f"  between-group sd {pstd(gmeans):.3f} | within-group sd {pstd(within):.3f} | groups with spread < 0.02: "
          f"{sum(max(t['total'] for t in v) - min(t['total'] for t in v) < 0.02 for v in groups.values())}/{len(groups)}")
    adv = [t["adv"] for t in traces]
    print(f"  advantage: sd {pstd(adv):.3f}, |adv| mean {sum(abs(a) for a in adv)/len(adv):.3f}, within-group reward vs advantage corr {corr(within, adv):.3f}")
    print("  what the advantage is made of (within-group demeaned components):")
    for k in comps:
        dm = demean(traces, lambda t, k=k: t["comp"].get(k, 0.0))
        print(f"    {k:13s} within-sd {pstd(dm):.3f}  corr with advantage {corr(dm, adv):+.3f}  share of advantage variance explained alone {corr(dm, adv)**2:.2f}")
    j_prog = [t["judge"].get("progress") for t in traces]
    if all(x is not None for x in j_prog):
        for sub in ("progress", "plan", "consistency"):
            dm = demean(traces, lambda t, s=sub: float(t["judge"].get(s) or 0.0))
            print(f"    judge.{sub:11s} within-sd {pstd(dm):.3f}  corr with advantage {corr(dm, adv):+.3f}")
    dm_ach = demean(traces, lambda t: t["seg_ach"])
    dm_judge = demean(traces, lambda t: t["comp"].get("judge", 0.0))
    dm_death = demean(traces, lambda t: t["deaths"])
    print(f"  within-group corr: judge vs achievements {corr(dm_judge, dm_ach):+.3f}; judge vs deaths {corr(dm_judge, dm_death):+.3f}; achievements vs deaths {corr(dm_ach, dm_death):+.3f}")
    # tokens
    cats = ["action", "action+text", "action+notes", "empty", "prose", "lesson", "notes"]
    agg = {c: [0, 0.0, 0.0, 0] for c in cats}
    push = {c: 0.0 for c in cats}
    mass = {c: 0.0 for c in cats}
    for t in traces:
        for c, (n, g, s, r) in t["cats"].items():
            if c not in agg:
                agg[c] = [0, 0.0, 0.0, 0]; push[c] = 0.0; mass[c] = 0.0
            agg[c][0] += n; agg[c][1] += g; agg[c][2] += s; agg[c][3] += r
            push[c] += (t["adv"] or 0.0) * g
            mass[c] += abs(t["adv"] or 0.0) * g
    ntok = sum(v[0] for v in agg.values()) or 1
    gtot = sum(v[1] for v in agg.values()) or 1e-9
    mtot = sum(mass.values()) or 1e-9
    print(f"  sampled tokens per trace {ntok/len(traces):.0f}; by reply category (token share | share of gradient mass sum(1-p) | mean 1-p per token | replies | signed push adv*(1-p), + = reinforced):")
    for c in agg:
        n, g, s, r = agg[c]
        if r == 0:
            continue
        print(f"    {c:12s} tokens {n/ntok:6.1%} | mass {g/gtot:6.1%} (|adv|-weighted {mass[c]/mtot:6.1%}) | 1-p {g/max(1,n):.3f} | replies {r:6d} ({n/max(1,r):.0f} tok each) | push {push[c]:+8.1f}")
    dec = [p for t in traces for p in t["decision"]]
    dec.sort()
    q = lambda f: dec[int(f * (len(dec) - 1))]
    print(f"  action choice (p of the sampled action token in pure-tag replies, n={len(dec)}): p10 {q(0.1):.2f} p25 {q(0.25):.2f} median {q(0.5):.2f} p75 {q(0.75):.2f}; "
          f"share with p < 0.5: {sum(p < 0.5 for p in dec)/len(dec):.1%}; share with p > 0.9: {sum(p > 0.9 for p in dec)/len(dec):.1%}")
    for c in ("prose", "notes", "action"):
        xs = [t["cats"][c][0] if c in t["cats"] else 0 for t in traces]
        print(f"  corr(advantage, {c} tokens) {corr(adv, xs):+.3f}; corr(advantage, all sampled tokens) {corr(adv, [sum(v[0] for v in t['cats'].values()) for t in traces]):+.3f}" if c == "prose" else f"  corr(advantage, {c} tokens) {corr(adv, xs):+.3f}")
    # per step
    print("  per step: n | reward mean | within-sd | adv sd | prose replies/trace | prose tokens share")
    by = collections.defaultdict(list)
    for t in traces:
        by[t["step"]].append(t)
    for s in sorted(by):
        v = by[s]
        w = demean(v, lambda t: t["total"])
        pt = sum(t["cats"]["prose"][0] if "prose" in t["cats"] else 0 for t in v)
        at = sum(sum(x[0] for x in t["cats"].values()) for t in v)
        pr = sum(t["cats"]["prose"][3] if "prose" in t["cats"] else 0 for t in v)
        print(f"    step {s:3d}: {len(v):3d} | {sum(t['total'] for t in v)/len(v):.3f} | {pstd(w):.3f} | {pstd([t['adv'] for t in v]):.3f} | {pr/len(v):5.1f} | {pt/max(1,at):.1%}")


def pstd(xs):
    xs = [x for x in xs if x is not None]
    if len(xs) < 2:
        return float("nan")
    m = sum(xs) / len(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / len(xs))


if __name__ == "__main__":
    for run in sys.argv[1:]:
        report(run)
