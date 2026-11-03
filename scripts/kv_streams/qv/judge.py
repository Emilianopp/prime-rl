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
import sys
import time

import httpx

from craftax_env.qv import (  # noqa: E402, F401  (the judge itself; re-exported for the scripts)
    FLAW_WEIGHTS, G_MAX, INSTRUCTIONS, MODEL, P_MAX, call, prompt, score, segment_reward,
)


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
