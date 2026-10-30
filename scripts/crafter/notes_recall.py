"""Do the end-of-segment notes recall what the eviction window has dropped? For every notes
turn: each unambiguous achievement unlocked in the lineage is 'visible' (unlocked within the
last WINDOW turns of the segment), 'evicted' (unlocked earlier in this segment) or 'inherited'
(unlocked in an earlier segment, so only the previous notes could carry it); 'never' means the
notes name something the lineage never unlocked. Deaths likewise (visible/evicted).
usage: notes_recall.py <stream dir or run dir> [--window 50]"""
import collections, glob, io, json, re, subprocess, sys
WINDOW = 50
args = [a for a in sys.argv[1:] if not a.startswith("--")]
if "--window" in sys.argv: WINDOW = int(sys.argv[sys.argv.index("--window") + 1])
A = re.compile(r"<action>")
KEYS = {  # achievement -> regex on the notes (case-insensitive); only the unambiguous ones
    "place_table": r"\btable", "make_wood_pickaxe": r"wood(en)?[ _]pickaxe", "make_wood_sword": r"wood(en)?[ _]sword",
    "make_stone_pickaxe": r"stone[ _]pickaxe", "make_stone_sword": r"stone[ _]sword", "place_furnace": r"furnace",
    "collect_coal": r"\bcoal", "collect_iron": r"\biron\b", "collect_sapling": r"sapling", "eat_cow": r"\bcow",
    "collect_stone": r"\bstone\b", "defeat_zombie": r"(defeat|kill|slay|beat)\w*\s+(a |the )?zombie|zombie\w*\s+(defeat|kill|slain)", "wake_up": r"\b(slept|sleep|woke)", "place_plant": r"plant(ed)?\b",
}
def text(m):
    c = m.get("content"); return c if isinstance(c, str) else " ".join(x.get("text", "") or "" for x in (c or []) if isinstance(x, dict))
tally = collections.defaultdict(lambda: [0, 0])  # (kind) -> [mentioned, total]
never = [0, 0]; deaths = collections.defaultdict(lambda: [0, 0]); n = 0
for d in args:
    for f in sorted(glob.glob(f"{d}/**/stream/*.jsonl.zst", recursive=True)):
        for line in io.BytesIO(subprocess.run(["zstd", "-dc", f], capture_output=True, check=True).stdout):
            for tr in json.loads(line)["traces"]:
                c = (tr.get("info") or {}).get("crafter")
                if not c: continue
                nodes = [x for x in tr["nodes"] if x.get("message") and x["message"]["role"] == "assistant"]
                if not nodes: continue
                notes = text(nodes[-1]["message"])
                if A.search(notes) or "<notes>" not in notes: continue  # no notes turn
                n += 1; low = notes.lower(); steps = c["steps"]
                before = set(c.get("achievements_before") or [])
                at = c.get("unlocked_at") or {}
                for ach, pat in KEYS.items():
                    hit = re.search(pat, low) is not None
                    if ach in at: kind = "visible" if steps - at[ach] <= WINDOW else "evicted"
                    elif ach in before: kind = "inherited"
                    else:
                        never[1] += 1; never[0] += hit; continue
                    tally[kind][1] += 1; tally[kind][0] += hit
                dl = [dd["step"] for dd in c.get("deaths") or []]
                if dl:
                    mention = re.search(r"\b(died|death|killed by|respawn)", low) is not None
                    kind = "visible" if steps - max(dl) <= WINDOW else "evicted"
                    deaths[kind][1] += 1; deaths[kind][0] += mention
print(f"{n} notes turns, window {WINDOW} turns")
for k in ("visible", "evicted", "inherited"):
    m, t = tally[k]; print(f"  achievement {k:9s}: mentioned {m}/{t} ({m/max(1,t):.0%})")
print(f"  never unlocked but named: {never[0]}/{never[1]} ({never[0]/max(1,never[1]):.0%}) of the (segment x achievement) pairs that never happened")
for k in ("visible", "evicted"):
    m, t = deaths[k]; print(f"  deaths {k:8s} (last death within/before the window): mentioned {m}/{t} ({m/max(1,t):.0%})")
