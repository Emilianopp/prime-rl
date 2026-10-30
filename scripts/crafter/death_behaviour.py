"""Behaviour around deaths, read off a run's rollout stream: usage death_behaviour.py <run dir>...

For each run dir ($SCRATCH/factory/runs/crafter-kve-summary/<job>): decompresses
run/*/monitors/file/traces/stream/*.jsonl.zst and reports the self-imitation rate,
what the policy did in the 3 turns before each death and after each respawn, whether
the first action after a respawn repeats the killing action, and whether a zombie was
adjacent in the last observation before the death."""
import collections, glob, io, json, re, statistics as st, subprocess, sys

def text(m):
    c = m.get("content")
    if c is None: return ""
    return c if isinstance(c, str) else " ".join(x.get("text", "") or "" for x in c if isinstance(x, dict))

GRID = re.compile(r"[.~Ts=,cidBFPCZS^@]{9}")
for run in sys.argv[1:]:
    files = sorted(glob.glob(f"{run}/run/*/monitors/file/traces/stream/*.jsonl.zst"))
    rep = tot = 0; base = collections.Counter(); runs = []; n = zadj = after_same = nd = 0
    before = collections.Counter(); after = collections.Counter(); div_pre = []; div_post = []
    for f in files:
        blob = subprocess.run(["zstd", "-dc", f], capture_output=True, check=True).stdout
        for line in io.BytesIO(blob):
            t = json.loads(line)
            for tr in t.get("traces", []):
                msgs = [nd_["message"] for nd_ in tr.get("nodes", []) if nd_.get("message")]
                acts = []; deaths = []
                for i, m in enumerate(msgs):
                    if m.get("role") == "assistant":
                        a = re.search(r"<action>(\w+)</action>", text(m)); acts.append(a.group(1) if a else None)
                        if i + 1 < len(msgs) and "You died" in text(msgs[i + 1]): deaths.append(len(acts) - 1)
                    if m.get("role") == "user" and "You died" in text(m):
                        n += 1
                        prev = [text(msgs[k]) for k in range(max(0, i - 6), i) if msgs[k].get("role") == "assistant"][-3:]
                        for p in prev:
                            a = re.search(r"<action>(\w+)", p); before[a.group(1) if a else "no-tag"] += 1
                        obs = text(msgs[i - 2]) if i >= 2 else ""
                        grid = [l for l in obs.split("\n") if len(l) == 9 and GRID.fullmatch(l)]
                        if len(grid) >= 9 and "@" in grid[4]:
                            x = grid[4].index("@"); nb = [grid[y][xx] for y in range(3, 6) for xx in range(max(0, x - 1), min(9, x + 2))]
                            zadj += "Z" in nb
                        nxt = [text(msgs[k]) for k in range(i + 1, min(len(msgs), i + 7)) if msgs[k].get("role") == "assistant"][:3]
                        for p in nxt:
                            a = re.search(r"<action>(\w+)", p); after[a.group(1) if a else "no-tag"] += 1
                acts = acts[:-1]  # the notes turn
                for a, b in zip(acts, acts[1:]):
                    if a and b: tot += 1; rep += a == b; base[b] += 1
                r = 1
                for a, b in zip(acts, acts[1:]):
                    if a == b and a: r += 1
                    else: runs.append(r); r = 1
                for d in deaths:
                    if d + 1 < len(acts) and acts[d] and acts[d + 1]: nd += 1; after_same += acts[d] == acts[d + 1]
                    div_pre.append(len(set(a for a in acts[max(0, d - 10):d + 1] if a))); div_post.append(len(set(a for a in acts[d + 1:d + 11] if a)))
    if not tot: print(f"{run}: no traces"); continue
    exp = sum((c / tot) ** 2 for c in base.values())
    print(f"===== {run}: {len(files)} stream files, {n} deaths in traces")
    print(f" P(action_t == action_t-1) = {rep/tot:.2f} (expected from the mix {exp:.2f}); runs of one action p90 {sorted(runs)[int(.9*len(runs))]} max {max(runs)}")
    print(f" zombie adjacent in the last observation before death: {zadj/max(1,n):.0%}")
    print(f" first action after a respawn = the killing action: {after_same/max(1,nd):.0%} ({nd})")
    print(f" distinct actions in the 10 turns before a death {st.mean(div_pre) if div_pre else 0:.1f}, in the 10 after {st.mean(div_post) if div_post else 0:.1f}")
    print(f" 3 turns before a death: {dict(before.most_common(5))}")
    print(f" 3 turns after a respawn: {dict(after.most_common(5))}")
    print(f" action mix: { {k: round(v/tot,2) for k,v in base.most_common(6)} }")
