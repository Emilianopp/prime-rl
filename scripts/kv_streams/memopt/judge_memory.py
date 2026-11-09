"""Gemini grades the memory a memopt round's players kept (round.sh runs it after score.py
when GEMINI_API_KEY is set). The unit is a context reset: the memory as it stands when a
segment ends (rebuilt by replaying the replies through verifiers' Memory) next to the true
state there (env.state_log: HUD, the floor's full map, achievements). The judge sorts every
fact into durable / transient / dump / wrong / duplicate, checks remembered positions on
the map, and scores how much the memory would help right after the reset (usefulness) and
how much of it the player could not recover from its screen (interesting), 0-10 each. An
empty memory is graded too (a variant that writes nothing scores 0, not a gap).

usage: judge_memory.py ROUND_OUT_DIR [--per-game 4] [--model gemini-3.1-flash-lite] [--cap 3.0]
"""

import argparse
import asyncio
import collections
import glob
import json
import os
import statistics as st
import subprocess
import sys

import httpx
from verifiers.v1.memory import Memory

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "environments", "craftax_env"))
from craftax_env.qv import call  # noqa: E402  (Gemini's OpenAI-compatible endpoint, JSON out, retries)

PROMPT = """You grade the long-term MEMORY that a player of Craftax (the full Crafter-like roguelike) keeps for itself.

The player's context is being wiped right now. Afterwards it will see only this memory, a short
handoff summary and its current screen: a 9x11 window centered on itself. Everything else it
knew is gone. Positions are absolute (row, col), as in the HUD's Pos and in the map below.

=== TRUE STATE NOW
{hud}

=== THE FLOOR'S FULL MAP (rows and columns labeled; the player sees only its 9x11 window)
{map}
{legend}

Achievements unlocked: {achievements}

=== THE PLAYER'S MEMORY ({n_entries} entries, ~{n_tokens} tokens)
{memory}

Grade it. Return JSON with:
1. "facts": one item per distinct fact in the memory (merge a fact repeated across lines), each
   {{"text": short quote, "kind": one of
     "durable"   - still true and useful later: where off-screen resources or structures are,
                   confirmed rules or recipes, what failed and why, dangers, a concrete plan;
     "transient" - true now but useless soon (current vitals, inventory counts, what is on
                   screen this turn, a step count);
     "dump"      - copied observation/HUD text or a turn-by-turn log;
     "wrong"     - contradicts the true state or the game's rules;
     "duplicate" - restates another fact,
   "position_ok": true / false if it states a position (true when the map has that thing within
   1 tile of it), else null}}.
2. "usefulness": 0-10, how much this memory would help this player play well right after the
   reset compared with an empty memory (0 = no help or harmful, 10 = everything it needs).
3. "interesting": 0-10, how much it holds that the player could NOT get back from its screen
   (off-screen locations, mechanics it learned, lessons from failures).
4. "note": one sentence on the memory's biggest strength or flaw.
An empty memory gets "facts": [], usefulness 0, interesting 0."""


def snapshots(vdir):
    """(world, segment, memory entries, true state) at every reset of every finished episode."""
    out = []
    for p in sorted(glob.glob(f"{vdir}/run/*/monitors/file/traces/stream/*.jsonl.zst")):
        raw = subprocess.run(["zstd", "-dc", p], capture_output=True, check=True).stdout
        for line in raw.splitlines():
            rec = json.loads(line)
            segs = sorted(rec["traces"], key=lambda t: t.get("info", {}).get("segment", 0))
            mem = Memory(max_key_tokens=4096)
            for k, t in enumerate(segs[:-1]):  # the last segment ends on the clock, not at a reset
                for n in t["nodes"]:
                    m = n["message"]
                    if m["role"] == "assistant":
                        c = m.get("content")
                        mem.apply(c if isinstance(c, str) else "")
                f = f"{vdir}/states/{t['id']}.end.json"
                if os.path.exists(f):
                    out.append((rec["task"]["data"]["idx"], k, dict(mem.entries), json.load(open(f))["state"]))
    return out


def render(entries):
    if not entries:
        return "(empty)"
    return "\n".join(f"[{k}]\n{v.strip()}" for k, v in entries.items())


async def judge_all(items, model, cap, price_in=0.25, price_out=1.5):
    spent = {"cost": 0.0, "calls": 0}
    sem = asyncio.Semaphore(16)
    async with httpx.AsyncClient(headers={"Authorization": f"Bearer {os.environ['GEMINI_API_KEY']}"}) as client:
        async def one(item):
            variant, (world, k, entries, s) = item
            async with sem:
                if spent["cost"] >= cap:
                    return None
                text = PROMPT.format(hud=s["hud"], map=s["map"], legend=s["legend"],
                                     achievements=", ".join(s["achievements"]) or "none",
                                     n_entries=len(entries), n_tokens=sum(len(v) for v in entries.values()) // 4,
                                     memory=render(entries))
                try:
                    j, usage = await call(client, text, model)
                except Exception as e:
                    return {"variant": variant, "world": world, "segment": k, "error": repr(e)}
                p = usage.get("prompt_tokens", 0) or 0
                spent["cost"] += (p * price_in + ((usage.get("total_tokens", 0) or 0) - p) * price_out) / 1e6
                spent["calls"] += 1
                return {"variant": variant, "world": world, "segment": k, "entries": len(entries),
                        "tokens": sum(len(v) for v in entries.values()) // 4, **j}
        res = await asyncio.gather(*(one(i) for i in items))
    return [r for r in res if r], spent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--per-game", type=int, default=4)
    ap.add_argument("--model", default="gemini-3.1-flash-lite")
    ap.add_argument("--cap", type=float, default=3.0, help="dollars for this round")
    a = ap.parse_args()
    items = []
    for vdir in sorted(glob.glob(f"{a.out}/*/")):
        variant = os.path.basename(vdir.rstrip("/"))
        by_world = collections.defaultdict(list)
        for snap in snapshots(vdir.rstrip("/")):
            by_world[snap[0]].append(snap)
        for snaps in by_world.values():  # spread the graded resets over the game
            step = max(1, len(snaps) // a.per_game)
            items += [(variant, s) for s in snaps[::step][: a.per_game]]
    res, spent = asyncio.run(judge_all(items, a.model, a.cap))
    with open(f"{a.out}/memory_judge.jsonl", "w") as f:
        for r in res:
            f.write(json.dumps(r) + "\n")

    rows = collections.defaultdict(list)
    for r in res:
        if "error" not in r:
            rows[r["variant"]].append(r)
    cols = ["n", "usefulness", "interesting", "durable/reset", "%durable", "%wrong", "%junk", "%pos_ok", "tokens"]
    print(f"memory judge ({a.model}): {spent['calls']} calls, ${spent['cost']:.2f}")
    print("variant".ljust(28) + "".join(c.rjust(14) for c in cols))
    summary = {}
    for v, rs in sorted(rows.items(), key=lambda kv: -st.mean(r.get("usefulness", 0) for r in kv[1])):
        kinds = collections.Counter(f.get("kind") for r in rs for f in r.get("facts", []))
        tot = max(1, sum(kinds.values()))
        pos = [f.get("position_ok") for r in rs for f in r.get("facts", []) if f.get("position_ok") is not None]
        s = {"n": len(rs), "usefulness": st.mean(r.get("usefulness", 0) for r in rs),
             "interesting": st.mean(r.get("interesting", 0) for r in rs),
             "durable/reset": kinds["durable"] / len(rs), "%durable": 100 * kinds["durable"] / tot,
             "%wrong": 100 * kinds["wrong"] / tot, "%junk": 100 * (kinds["transient"] + kinds["dump"] + kinds["duplicate"]) / tot,
             "%pos_ok": 100 * sum(pos) / max(1, len(pos)), "tokens": st.mean(r["tokens"] for r in rs)}
        summary[v] = s
        print(v.ljust(28) + "".join(f"{s[c]:14.2f}" for c in cols))
    json.dump({"spent": spent, "variants": summary}, open(f"{a.out}/memory_judge.json", "w"), indent=1)


if __name__ == "__main__":
    main()
