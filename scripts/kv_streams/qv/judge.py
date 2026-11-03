"""The Q/V judge: Gemini grades a Craftax state and the handoff summary written there.

Per state it sees the rules, the true state (HUD, full map of the floor, nearest key
blocks, unlocked and locked achievements with their points) and, for a summary under
review, that summary and the one before it. It returns position parts (tool 0-4,
ready 0-3, safe 0-2, near 0-1), each locked achievement a good agent with an ideal
handoff could unlock by the next reset with its probability, the expected position
gain, an ideal handoff, and the summary's flaws (each with a quote and the
contradicting fact). From those, in code (`score`):

  V  = B + P                                  B: banked achievement points (1/3/5/8 by tier)
  G* = min(G_MAX, sum_j p_j points_j + expected position gain)
  D  = 1 - prod_flaws (1 - w_flaw)             (no summary under review: D = 1)
  Q  = V + G* (1 - D)                          (game start and end: Q = V)

and a segment's reward from the reset that opens it (t) to the one that closes it:
R_t = clip((Q_{t+1} - Q_t) / max(G*_t, 1), -1, 1) (`segment_reward`).

The key is read from GEMINI_API_KEY. As a script, judges every reset of replay.py's
output (calls cached next to the output, so a rerun resumes):

    python judge.py <replay.jsonl> <out.json> [concurrency]
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import sys
import time

import httpx

MODEL = "gemini-3.8-flash"
URL = "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
G_MAX = 10.0
FLAW_WEIGHTS = {
    "false_fact": 0.3,
    "made_up_rule": 0.2,
    "next_goal_wrong": 0.4,
    "next_goal_vague": 0.1,
    "location": 0.15,
    "impossible_step": 0.2,
    "urgent_ignored": 0.3,
    "dropped_fact": 0.1,
}
P_MAX = {"tool": 4, "ready": 3, "safe": 2, "near": 1}

INSTRUCTIONS = """\
You grade one moment of a Craftax game for a reinforcement-learning reward. The player
is a language model that only sees a small local view and a HUD; you see the true state,
including the full map of its current floor. The player's context is reset periodically;
before each reset it writes a handoff summary, and after the reset it sees only that
summary, the rules and the current observation (local view + HUD). The next reset comes
after roughly 120 more player turns (each turn plays up to 4 actions, so roughly 250-450
game steps).

Do these steps, in this order, inside the JSON fields below.

1. "diagnosis": from the true state and full map, say what stage the player is at, what
   the next milestones are (by the tech tree in the rules), what blocks them (missing
   tools or materials, no station nearby), what threatens the player (any vital <= 3,
   night, monsters, lava), and where the resources it needs are (absolute (row,col)).

2. "P": score the state with these fixed anchors (integers):
   tool  0 no tools | 1 wood tools | 2 stone tools | 3 iron tools | 4 diamond tools or the player is below floor 0
   ready 0 lacks the materials for its next craft or placement and no station is nearby
         | 1 lacks materials but the station is nearby, or has some of them | 2 has the materials
         | 3 has the materials and the needed station is adjacent (or no station is needed)
   safe  0 some vital <= 2, or a monster adjacent while health <= 4 | 1 some vital <= 4, or night in the open | 2 safe
   near  1 if the resources the next milestone needs are within ~10 cells (Manhattan), else 0

3. "ideal_handoff": the handoff summary you would write for the player now, 3-6 lines:
   state facts the player will need, the next subgoal, absolute positions of what it
   needs, threats to handle first.

4. "next_achievements": the LOCKED achievements (names exactly as listed) that a competent
   player starting from this state with your ideal handoff could unlock before the next
   reset, each with your probability that it does ("p", 0 to 1). List every one with a
   real chance, including unlikely ones with a small p. Be calibrated: these probabilities
   are checked against what actually happens, so account for the time it takes to walk,
   gather, survive the night and recover from danger. "position_gain": the expected rise
   in P by then (a number, may be fractional; at most 10 minus the current P total).

5. "flaws": only if a SUMMARY UNDER REVIEW is given. Check it against the true state, the
   rules and the previous summary, and list every flaw of these types:
   false_fact      states something the true state contradicts (inventory, tools,
                   achievements, deaths, position, what exists on the map)
   made_up_rule    claims a game rule or mechanic that is not true per the rules
   next_goal_wrong its next goal is wrong or missing, against your diagnosis (at most once)
   next_goal_vague its next goal is right but too vague to act on (at most once; not with next_goal_wrong)
   location        a position its next goal needs is missing, given relative to the local
                   view instead of absolute (row,col), or wrong (one per needed location)
   impossible_step a plan step that cannot be done with the current inventory/tools
   urgent_ignored  a vital <= 3 or nightfall is not addressed
   dropped_fact    a still-true, still-useful fact from the PREVIOUS SUMMARY is missing
   Each flaw needs "quote" (the exact words from the summary, or "" for something missing)
   and "fact" (the true state fact or rule that shows it is a flaw). Do not grade length,
   formatting or tone. No flaw without evidence.

Reply with JSON only:
{"diagnosis": "...", "P": {"tool": 0, "ready": 0, "safe": 0, "near": 0},
 "ideal_handoff": "...", "next_achievements": [{"name": "...", "p": 0.0}], "position_gain": 0.0,
 "flaws": [{"type": "...", "quote": "...", "fact": "..."}]}
"""


def prompt(rules: str, s: dict, review: bool = False, summary: str | None = None, prev: str | None = None) -> str:
    """The judge's prompt for state record `s` (replay.state_record); with `review`, the
    summary under review and the previous one."""
    near = "\n".join(f"  {n}: ({v[0]},{v[1]}), distance {v[2]}" for n, v in sorted(s["nearest"].items(), key=lambda x: x[1][2]))
    parts = [
        "=== RULES (the player's system prompt)\n" + rules[:12000],
        "=== TRUE STATE\n" + s["hud"],
        "Unlocked achievements (points): " + ", ".join(f"{a} ({p:g})" for a, p in s["achievements"].items()),
        "Locked achievements (points): " + ", ".join(f"{a} ({p:g})" for a, p in s["locked"].items()),
        f"Deaths so far: {s['deaths']} (a death drops the whole inventory and respawns the player near the spawn point)",
        "Nearest key blocks (row,col), Manhattan distance from the player:\n" + near,
        "Full map of the current floor (row labels on the left, column digits on top; the player is the arrow):\n"
        + s["map"] + "\nLegend:\n" + s["legend"],
    ]
    if review:
        parts.append("=== PREVIOUS SUMMARY (what the player was handed at the last reset)\n" + (prev or "(none)"))
        parts.append("=== SUMMARY UNDER REVIEW (written now)\n" + (summary or "(the player wrote no summary)"))
    else:
        parts.append("=== No summary to review at this point: leave \"flaws\" empty.")
    return "\n\n".join(parts) + "\n\n=== INSTRUCTIONS\n" + INSTRUCTIONS


def score(s: dict, j: dict, review: bool = False, summary: str | None = None) -> dict:
    """V, G*, D and Q of state record `s` from the judge's reply `j`."""
    P = {n: max(0, min(P_MAX[n], int(j.get("P", {}).get(n, 0) or 0))) for n in P_MAX}
    p_total = sum(P.values())
    nexts = []
    for a in j.get("next_achievements", []):
        name, p = (a.get("name"), a.get("p", 0)) if isinstance(a, dict) else (a, 1.0)
        if name in s["locked"] and name not in [n for n, _ in nexts]:
            nexts.append((name, max(0.0, min(1.0, float(p or 0)))))
    gain = sum(p * s["locked"][a] for a, p in nexts) + max(0.0, min(10.0 - p_total, float(j.get("position_gain", 0) or 0)))
    g_star = min(G_MAX, gain)
    flaws = j.get("flaws", []) if review and summary else []
    counts: dict[str, int] = {}
    keep = 1.0
    for f in flaws:
        t = f.get("type")
        if t not in FLAW_WEIGHTS:
            continue
        counts[t] = counts.get(t, 0) + 1
        if t in ("next_goal_wrong", "next_goal_vague") and counts[t] > 1:
            continue
        if t == "next_goal_vague" and counts.get("next_goal_wrong"):
            continue
        if t == "dropped_fact" and counts[t] > 3:
            continue
        keep *= 1.0 - FLAW_WEIGHTS[t]
    v = s["B"] + p_total
    if review:
        d = 1.0 - keep if summary else 1.0
        q = v + g_star * (1 - d)
    else:
        d, q = None, v
    return {"P": P, "V": v, "G_star": g_star,
            "next_achievements": [{"name": a, "p": p, "points": s["locked"][a]} for a, p in nexts],
            "D": d, "Q": q, "flaws": flaws, "ideal_handoff": j.get("ideal_handoff", ""), "diagnosis": j.get("diagnosis", "")}


def segment_reward(start: dict, end: dict) -> float:
    """R of the segment between two scored states."""
    return max(-1.0, min(1.0, (end["Q"] - start["Q"]) / max(start["G_star"], 1.0)))


async def call(client: httpx.AsyncClient, text: str, model: str = MODEL) -> tuple[dict, dict]:
    body = {"model": model, "messages": [{"role": "user", "content": text}],
            "response_format": {"type": "json_object"}, "temperature": 0.2}
    for attempt in range(5):
        try:
            r = await client.post(URL, json=body, timeout=300)
            if r.status_code == 429 or r.status_code >= 500:
                raise httpx.HTTPStatusError(str(r.status_code), request=r.request, response=r)
            r.raise_for_status()
            data = r.json()
            content = data["choices"][0]["message"]["content"]
            m = re.search(r"\{.*\}", content, re.S)
            return json.loads(m.group(0) if m else content), data.get("usage", {})
        except Exception:  # retry transient API and parse failures
            if attempt == 4:
                raise
            await asyncio.sleep(5 * 2 ** attempt)
    raise RuntimeError("unreachable")


async def judge_all(jobs: dict, cache_path: str, concurrency: int = 8, model: str = MODEL) -> dict:
    """Run `jobs` ({key: prompt text}) through the judge, `key` a JSON-able id; replies are
    appended to `cache_path` and reused on a rerun. Returns {key: reply}."""
    cache = {}
    if os.path.exists(cache_path):
        for line in open(cache_path):
            c = json.loads(line)
            cache[c["key"]] = c["judge"]
    todo = {k: p for k, p in jobs.items() if k not in cache}
    print(f"{len(todo)} judge calls ({len(jobs) - len(todo)} cached)", flush=True)
    sem = asyncio.Semaphore(concurrency)
    headers = {"Authorization": f"Bearer {os.environ['GEMINI_API_KEY']}"}
    async with httpx.AsyncClient(headers=headers) as client:
        with open(cache_path, "a") as f:
            async def run(key, text):
                async with sem:
                    t0 = time.time()
                    try:
                        j, usage = await call(client, text, model)
                    except Exception as exc:
                        print(f"{key}: FAILED {type(exc).__name__}: {str(exc)[:200]}", flush=True)
                        return
                    cache[key] = j
                    f.write(json.dumps({"key": key, "judge": j, "usage": usage, "secs": time.time() - t0}) + "\n")
                    f.flush()
            await asyncio.gather(*(run(k, p) for k, p in todo.items()))
    return {k: cache[k] for k in jobs if k in cache}


def main():
    src, out = sys.argv[1], sys.argv[2]
    conc = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    eps = [json.loads(line) for line in open(src)]
    jobs = {}
    for ep in eps:
        for k in range(len(ep["segments"]) + 1):  # start, then the end of each segment
            s = ep["resets"][k]
            prev = next((r["summary"] for r in reversed(ep["resets"][:k]) if r.get("summary")), None)
            jobs[f"{ep['seed']}:{k}"] = prompt(ep["rules"], s, s["kind"] == "reset", s.get("summary"), prev)
    replies = asyncio.run(judge_all(jobs, out + ".calls.jsonl", conc))
    results = []
    for ep in eps:
        n = len(ep["segments"]) + 1
        states = []
        for k in range(n):
            s, j = ep["resets"][k], replies.get(f"{ep['seed']}:{k}")
            states.append(score(s, j, s["kind"] == "reset", s.get("summary")) if j else None)
        segs = [{**{x: seg[x] for x in ("turns", "actions", "forfeits", "deaths", "unlocked")},
                 "R": segment_reward(states[t], states[t + 1]) if states[t] and states[t + 1] else None}
                for t, seg in enumerate(ep["segments"])]
        keep = ("kind", "k", "turn", "hud", "map", "legend", "achievements", "B", "deaths", "summary", "nearest")
        results.append({"seed": ep["seed"], "segments": segs,
                        "states": [{**{x: ep["resets"][k][x] for x in keep}, "judge": states[k]} for k in range(n)]})
    json.dump(results, open(out, "w"))
    print("wrote", out)


if __name__ == "__main__":
    main()
