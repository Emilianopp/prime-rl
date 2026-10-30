"""Death lessons, read off a run's rollout stream: usage lesson_analysis.py <run dir> [--samples N]

For a DEATH_LESSON=1 run ($SCRATCH/factory/runs/crafter-kve-summary/<job>):
1. Mode switching: at every observation that asks for a <lesson>, what the reply held
   (lesson + action, action only = stayed in action mode, lesson only = dropped the
   action, neither), plus <lesson> tags written on turns that did not ask for one
   (stuck in lesson mode) and lessons under 20 characters.
2. Relevance: each lesson against what actually happened (cause and night from
   episodes.jsonl, the last observation and the last 3 actions before the death):
   does it name the real killer, the night when it was night, the vital that hit 0;
   does the first action after it repeat the killing action. Prints N samples with
   the context so the texts can be read."""
import collections, glob, io, json, re, subprocess, sys, random

LESSON_REQUEST = "First write what you learned from this death inside <lesson>"
_LESSON = re.compile(r"<lesson>\s*(.*?)\s*(?:</lesson>|<action>|$)", re.DOTALL)
_ACTION = re.compile(r"<action>\s*(\w+)\s*</action>")
GRID = re.compile(r"[.~Ts=,cidBFPCZS^@]{9}")
WORDS = {"zombie": r"zombie", "skeleton": r"skeleton|arrow", "lava": r"lava", "starved": r"food|hungry|hunger|starv|drink|water|thirst|energy|sleep|tired|rest",
         "night": r"night|dark|dusk|evening|sleep"}

def text(m):
    c = m.get("content")
    if c is None: return ""
    return c if isinstance(c, str) else " ".join(x.get("text", "") or "" for x in c if isinstance(x, dict))

def act(reply):
    a = _ACTION.search(reply); return a.group(1) if a else None

def surroundings(obs):
    """What the last observation before the death showed: zombie/skeleton adjacent or in view, vitals."""
    grid = [l for l in obs.split("\n") if len(l) == 9 and GRID.fullmatch(l)]
    out = {"zombie_adjacent": False, "zombie_in_view": False, "skeleton_in_view": False, "lava_in_view": False}
    if len(grid) >= 9 and "@" in grid[4]:
        x = grid[4].index("@"); nb = [grid[y][xx] for y in range(3, 6) for xx in range(max(0, x - 1), min(9, x + 2))]
        out["zombie_adjacent"] = "Z" in nb
        allc = "".join(grid); out["zombie_in_view"] = "Z" in allc; out["skeleton_in_view"] = "S" in allc; out["lava_in_view"] = "^" in allc
    v = re.search(r"Vitals: (.*)", obs); out["vitals"] = v.group(1) if v else ""
    return out

samples = int(sys.argv[sys.argv.index("--samples") + 1]) if "--samples" in sys.argv else 12
args = [a for i, a in enumerate(sys.argv[1:], 1) if not a.startswith("--") and sys.argv[i - 1] != "--samples"]
for run in args:
    deaths_by_node = {}; order = {}
    try:
        eps = sorted((json.loads(l) for l in open(f"{run}/episodes.jsonl")), key=lambda e: e.get("start", 0))
        eps = [e for e in eps if e.get("task") == "crafter"]
        for i, e in enumerate(eps):
            deaths_by_node[e["node_id"]] = e.get("deaths") or []; order[e["node_id"]] = i * 5 // max(1, len(eps))
    except FileNotFoundError:
        pass
    files = sorted(glob.glob(f"{run}/run/*/monitors/file/traces/stream/*.jsonl.zst"))
    modes = collections.Counter(); spontaneous = 0; spontaneous_traces = 0; short = 0; traces = 0; action_first = 0
    rows = []  # one per lesson reply with context
    per_q = {}  # per quintile of the segment start order (~ training progress): mode counts, killer named, repeat, spontaneous
    for f in files:
        blob = subprocess.run(["zstd", "-dc", f], capture_output=True, check=True).stdout
        for line in io.BytesIO(blob):
            t = json.loads(line)
            for tr in t.get("traces", []):
                traces += 1
                msgs = [n["message"] for n in tr.get("nodes", []) if n.get("message")]
                deaths = deaths_by_node.get(tr.get("id")) or ((tr.get("info") or {}).get("crafter") or {}).get("deaths") or []
                q = per_q.setdefault(order.get(tr.get("id"), -1), {"m": collections.Counter(), "named": 0, "known": 0, "rep": 0, "repn": 0, "spont": 0, "traces": 0})
                q["traces"] += 1
                k = 0; spont = 0
                for i, m in enumerate(msgs):
                    if m.get("role") != "user": continue
                    asked = LESSON_REQUEST in text(m)
                    reply = text(msgs[i + 1]) if i + 1 < len(msgs) and msgs[i + 1].get("role") == "assistant" else ""
                    if "You died" in text(m):
                        death = deaths[k] if k < len(deaths) else {}; k += 1
                    if not asked:
                        if "<lesson>" in reply: spont += 1
                        continue
                    has_l, has_a = _LESSON.search(reply) is not None, _ACTION.search(reply) is not None
                    mode = "lesson+action" if has_l and has_a else "action only" if has_a else "lesson only" if has_l else "neither"
                    modes[mode] += 1; q["m"][mode] += 1
                    lesson = " ".join((_LESSON.search(reply).group(1) if has_l else "").split())
                    if has_l and has_a and reply.find("<action>") < reply.find("<lesson>"): action_first += 1
                    if has_l and len(lesson) < 20: short += 1
                    prev_obs = text(msgs[i - 2]) if i >= 2 else ""
                    prev_acts = [act(text(msgs[j])) for j in range(max(0, i - 6), i) if msgs[j].get("role") == "assistant"][-3:]
                    next_acts = [act(text(msgs[j])) for j in range(i + 1, min(len(msgs), i + 7)) if msgs[j].get("role") == "assistant"][:3]
                    if has_l and death.get("cause") in WORDS:
                        q["known"] += 1; q["named"] += bool(re.search(WORDS[death["cause"]], lesson, re.I))
                    if prev_acts and next_acts and prev_acts[-1] and next_acts[0]:
                        q["repn"] += 1; q["rep"] += prev_acts[-1] == next_acts[0]
                    rows.append({"trace": tr.get("id"), "lesson": lesson, "has_lesson": has_l, "has_action": has_a,
                                 "cause": death.get("cause"), "night": (death.get("daylight", 1.0) < 0.5) if death else None,
                                 "starved": [v for v in ("food", "drink", "energy") if (death.get("vitals") or {}).get(v, 1) == 0],
                                 "pos": death.get("pos"), "respawn": death.get("respawn"), **surroundings(prev_obs),
                                 "prev_acts": prev_acts, "next_acts": next_acts})
                spontaneous += spont; spontaneous_traces += spont > 0; q["spont"] += spont
    asked = sum(modes.values())
    print(f"===== {run}: {len(files)} stream files, {traces} traces, {asked} lesson requests")
    if not asked: continue
    print(" --- mode switching at the request")
    for kname, v in modes.most_common(): print(f"   {kname:14s} {v:4d} ({v/asked:.0%})")
    print(f"   action written before the lesson: {action_first}; lessons under 20 chars: {short}")
    print(f"   <lesson> on turns that did not ask for one: {spontaneous} replies in {spontaneous_traces}/{traces} traces")
    print(" --- per quintile of the segment start order: traces, requests: lesson+action, lesson only, action only, neither | killer named | repeat of killing action | spontaneous <lesson>")
    for qi in sorted(per_q):
        q = per_q[qi]; fm = q["m"]; tot = sum(fm.values())
        print(f"   q{qi + 1}: {q['traces']:3d} traces, {tot:4d} requests: {fm['lesson+action']:4d} {fm['lesson only']:3d} {fm['action only']:3d} {fm['neither']:3d} | {q['named']}/{q['known']} | {q['rep']}/{q['repn']} | {q['spont']}")
    with_l = [r for r in rows if r["has_lesson"]]
    known = [r for r in with_l if r["cause"]]
    print(f" --- relevance of the {len(with_l)} lessons ({len(known)} with the cause from episodes.jsonl)")
    by_cause = collections.defaultdict(list)
    for r in known:
        key = r["cause"]; pat = WORDS.get(key)
        by_cause[key].append(bool(pat and re.search(pat, r["lesson"], re.I)))
    for c, v in sorted(by_cause.items()): print(f"   names the killer | {c:8s}: {sum(v)}/{len(v)} ({sum(v)/len(v):.0%})")
    nights = [r for r in known if r["night"]]; days = [r for r in known if r["night"] is False]
    for name, grp in (("night deaths", nights), ("day deaths", days)):
        if grp: print(f"   mentions night/dark/sleep | {name}: {sum(bool(re.search(WORDS['night'], r['lesson'], re.I)) for r in grp)}/{len(grp)}")
    zadj = [r for r in with_l if r["zombie_adjacent"]]
    if zadj: print(f"   zombie adjacent before the death and the lesson says zombie: {sum(bool(re.search('zombie', r['lesson'], re.I)) for r in zadj)}/{len(zadj)}")
    wrong = [r for r in known if r["cause"] in ("zombie", "skeleton", "lava") and re.search(WORDS["starved"], r["lesson"], re.I) and not re.search(WORDS[r["cause"]], r["lesson"], re.I)]
    print(f"   blames hunger/thirst/energy for a zombie/skeleton/lava death: {len(wrong)}/{len(known)}")
    mentions_coords = sum(bool(re.search(r"\(\d+, ?\d+\)", r["lesson"])) for r in with_l)
    mentions_action = sum(bool(re.search(r"move_(left|right|up|down)|\bdo\b|sleep|place_|make_", r["lesson"])) for r in with_l)
    print(f"   lesson names a coordinate: {mentions_coords}/{len(with_l)}; names a concrete action: {mentions_action}/{len(with_l)}")
    rep = [(r["prev_acts"][-1] == r["next_acts"][0]) for r in rows if r["prev_acts"] and r["next_acts"] and r["prev_acts"][-1] and r["next_acts"][0]]
    rep_l = [(r["prev_acts"][-1] == r["next_acts"][0]) for r in with_l if r["prev_acts"] and r["next_acts"] and r["prev_acts"][-1] and r["next_acts"][0]]
    print(f"   first action after the death repeats the killing action: all {sum(rep)}/{len(rep)} ({sum(rep)/max(1,len(rep)):.0%}); with a lesson {sum(rep_l)}/{len(rep_l)} ({sum(rep_l)/max(1,len(rep_l)):.0%})")
    lens = sorted(len(r["lesson"]) for r in with_l)
    if lens: print(f"   lesson length chars: median {lens[len(lens)//2]}, p90 {lens[int(.9*len(lens))]}, max {lens[-1]}")
    dup = collections.Counter(r["lesson"] for r in with_l).most_common(3)
    print(f"   most repeated lesson texts: {[(n, t[:80]) for t, n in dup if n > 1]}")
    print(f" --- {samples} samples")
    random.seed(0)
    for r in random.sample(rows, min(samples, len(rows))):
        print(f"   [{r['cause']}{' night' if r['night'] else ''}{' starved:' + ','.join(r['starved']) if r['starved'] else ''}] at {r['pos']} -> {r['respawn']}; before: {r['prev_acts']}, zombie adjacent={r['zombie_adjacent']}, vitals {r['vitals']}")
        print(f"      lesson: {r['lesson'][:300] if r['has_lesson'] else '(none: ' + ('action only' if r['has_action'] else 'no tags') + ')'}")
        print(f"      after: {r['next_acts']}")
