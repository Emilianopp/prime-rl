"""How useful is the player's memory in the RL traces? Per block:
 - memory rebuilt per game (verifiers Memory, every reply in order across segments);
 - position facts at each segment end checked against that boundary's true map;
 - after a reset with memory present: did it read; did remembered positions show up in
   its next 15 replies (reasoning or reply); reward and points gained, read vs not read;
 - examples written to OUT_TXT: late-game memories and read episodes with what followed.

usage: useful.py OUT_TXT LABEL:BLOCK_DIR ..."""
import collections
import json
import re
import statistics as st
import sys

sys.path.insert(0, sys.path[0])
from cmp import block, txt  # noqa: E402
from verifiers.v1.memory import Memory  # noqa: E402

GLYPH = {"tree": "♣", "water": "≈", "stone": "S", "coal": "C", "iron": "I", "diamond": "D", "lava": "♨",
         "sand": "░", "table": "t", "furnace": "f", "ladder": "▼"}
WORD = re.compile(r"\b(" + "|".join(GLYPH) + r")s?\b", re.I)
PAIR = re.compile(r"\((\d+)\s*,\s*(\d+)\)")
OPS = re.compile(r"<(write|rewrite|replace|delete|read)\b")
READKEY = re.compile(r'<read\s+key\s*=\s*"([^"]+)"')


def m(xs):
    xs = [x for x in xs if x is not None]
    return st.mean(xs) if xs else float("nan")


def grid(state):
    rows = {}
    for line in state["map"].splitlines()[2:]:
        mm = re.match(r"\s*(\d+) (.*)", line)
        if mm:
            rows[int(mm.group(1))] = mm.group(2)
    return rows


def check(entries, state):
    n = near = exact = 0
    if not state or "Overworld" not in state["hud"]:
        return 0, 0, 0
    rows = grid(state)
    for text in entries.values():
        for piece in re.split(r"[;\n]|,\s(?=[A-Za-z])", text):
            w = WORD.search(piece)
            if not w:
                continue
            g = GLYPH[w.group(1).lower()]
            for a, b in PAIR.findall(piece):
                r, c = int(a), int(b)
                n += 1
                near += any(rows.get(rr, "")[cc:cc + 1] == g for rr in range(r - 1, r + 2) for cc in range(max(0, c - 1), c + 2))
                exact += rows.get(r, "")[c:c + 1] == g
    return n, near, exact


def load_msgs(bdir, tid):
    import glob
    p = glob.glob(f"{bdir}/run/*/monitors/file/traces/live/{tid}.jsonl")[0]
    nodes = []
    for line in open(p):
        if line.strip():
            nodes += json.loads(line).get("nodes", [])
    return [n["message"] for n in nodes]


def state(bdir, tid, kind):
    try:
        return json.load(open(f"{bdir}/qv_states/{tid}.{kind}.json"))["state"]
    except FileNotFoundError:
        return None


def main():
    out = open(sys.argv[1], "w")
    print("blk  pos_facts  within1  exact | resets(mem)  read%  used_pos%(read)  used_pos%(no read) | "
          "R read/no  dPts read/no  | facts/mem  mem_keys")
    for spec in sys.argv[2:]:
        label, bdir = spec.split(":")
        segs = block(bdir)
        games = collections.defaultdict(list)
        for s in segs:
            if s["t0"] is not None and s["idx"] is not None:
                games[s["idx"]].append(s)
        P = collections.Counter()
        after = []  # (read, used, R, dB)
        examples_mem, examples_read = [], []
        nfacts, nkeys = [], []
        for idx, ss in games.items():
            ss.sort(key=lambda s: s["t0"])
            mem = Memory(max_key_tokens=4096)
            for k, s in enumerate(ss):
                msgs = load_msgs(bdir, s["tid"])
                asst = [x for x in msgs if x["role"] == "assistant"]
                before = dict(mem.entries)
                mem_pairs = {p for v in before.values() for p in PAIR.findall(v)}
                if k and before:
                    read = s["read_first3"]
                    used = False
                    for a in asst[:15]:
                        t = txt(a) + (a.get("reasoning_content") or "")
                        if any(p in mem_pairs for p in PAIR.findall(t)):
                            used = True
                            break
                    q = s["q"]
                    R = q["reward"] if q and "B" in q["info"] else None
                    dB = q["info"]["B"][1] - q["info"]["B"][0] if q and "B" in q["info"] else None
                    after.append((read, used, R, dB))
                    if read and len(examples_read) < 4 and label.startswith("M"):
                        keys = [kk for a in asst[:3] for kk in READKEY.findall(txt(a))]
                        follow = []
                        for a in asst[:8]:
                            rc = (a.get("reasoning_content") or "").strip().replace("\n", " ")
                            follow.append(f"  REASON: {rc[:500]}\n  REPLY: {txt(a).strip()[:300]}")
                        examples_read.append(
                            f"===== {label} world {idx} reset #{k} (turn {s['t0']}): read {keys}; R={R} dPts={dB}\n"
                            + "".join(f"[{kk}]\n{before.get(kk, '(missing)')}\n" for kk in keys)
                            + "--- next replies\n" + "\n".join(follow) + "\n")
                for a in asst:
                    mem.apply(txt(a))
                if s["summ"]:
                    st_end = state(bdir, s["tid"], "end")
                    n, ne, ex = check(mem.entries, st_end)
                    P["n"] += n
                    P["near"] += ne
                    P["exact"] += ex
                    nfacts.append(sum(len(PAIR.findall(v)) for v in mem.entries.values()))
                    nkeys.append(len(mem.entries))
            if label.startswith("M") and len(examples_mem) < 3 and len(ss) >= 6 and mem.entries:
                st_last = state(bdir, ss[-1]["tid"], "end") or state(bdir, ss[-1]["tid"], "start")
                n, ne, ex = check(mem.entries, st_last)
                examples_mem.append(f"===== {label} world {idx}: memory after {len(ss)} segments "
                                    f"(positions {ne}/{n} within 1 tile)\n"
                                    + (f"TRUE HUD: {st_last['hud']}\nTRUE nearest: {st_last['nearest']}\n" if st_last else "")
                                    + "".join(f"--- [{kk}]\n{v.strip()}\n" for kk, v in mem.entries.items()))
        rd = [a for a in after if a[0]]
        nr = [a for a in after if not a[0]]
        print(f"{label:4} {P['n']:8}  {100 * P['near'] / max(1, P['n']):5.0f}%  {100 * P['exact'] / max(1, P['n']):4.0f}%"
              f" | {len(after):6}  {100 * len(rd) / max(1, len(after)):4.0f}%  {100 * m([a[1] for a in rd]):8.0f}%"
              f"  {100 * m([a[1] for a in nr]):8.0f}%   | {m([a[2] for a in rd]):5.2f}/{m([a[2] for a in nr]):5.2f}"
              f"  {m([a[3] for a in rd]):4.2f}/{m([a[3] for a in nr]):4.2f}  | {m(nfacts):6.1f}  {m(nkeys):5.1f}")
        out.write("\n".join(examples_mem) + "\n" + "\n".join(examples_read) + "\n")
    out.close()


if __name__ == "__main__":
    main()
