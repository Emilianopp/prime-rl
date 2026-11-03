"""Replay Craftax eval games from their traces: the true game state at every reset.

The eval traces keep observations, not game states. A Craftax game is deterministic
given its seed and the actions played, so this replays an episode as CraftaxEnv.run
does (freeze on a forfeit, respawn with drop_all on a death; the player's memory
replays its tags) and records, at the game start, every summary turn and the game end:
the HUD, the full map of the current floor (no fog, absolute row/column labels), the
nearest key blocks, the achievements with their points and the summary written there.
Every replayed state is checked against the logged observation's HUD (position and
inventory); mismatches are counted per episode.

With ``capture`` it also keeps, at those resets, what a branch needs to play the game
on from there: an env save (``CraftaxEnv``'s ``state_dir`` format) and the token ids
of the summary turn's prompt as the KV-Streams turn window left it (the system prompt,
the last ``window`` complete turns and the summary request).

    python replay.py <eval_dir> <out.jsonl> [n_episodes] [stride] [offset]
"""

from __future__ import annotations

import glob
import json
import re
import subprocess
import sys
from collections import Counter

import verifiers.v1 as vf
from verifiers.v1.memory import Memory
from verifiers.v1.types import AssistantMessage, UserMessage, content_text

from craftax_env import glyph
from craftax_env.qv import full_map, nearest, points, state_record  # noqa: F401  (re-exported for the scripts)
from craftax_env.turn_rewards import SUMMARY_ASK

SUMMARY = re.compile(r"<summary>(.*?)</summary>", re.S | re.I)
POS = re.compile(r"Pos: \((\d+),(\d+)\)")
INV = re.compile(r"\[Inventory\] (.*)")
def hud_check(game: glyph.CraftaxGame, obs: str) -> bool:
    _l, _g, hud, _m = game.render()
    a, b = POS.search(hud), POS.search(obs)
    i, j = INV.search(hud), INV.search(obs)
    return bool(a and b and a.groups() == b.groups() and i and j and i.group(1).strip() == j.group(1).strip())


def window_prompt(nodes: list[dict], ask: int, summary_node: dict, window: int) -> list[int]:
    """Token ids of the summary turn's prompt under a KV-Streams turn window: the system
    prompt, the last `window` complete turns before the request, the request, and the
    reply's unsampled head (the assistant header and thinking opener)."""
    ids = list(nodes[0]["token_ids"]) if nodes[0]["message"]["role"] == "system" else []
    turns, i = [], ask - 1
    while i >= 1 and len(turns) < window:
        if nodes[i]["message"]["role"] == "assistant" and nodes[i - 1]["message"]["role"] == "user":
            turns.append(nodes[i - 1]["token_ids"] + nodes[i]["token_ids"])
            i -= 2
        else:
            i -= 1
    for t in reversed(turns):
        ids += t
    ids += nodes[ask]["token_ids"]
    mask, head = summary_node.get("mask") or [], []
    for tok, m in zip(summary_node["token_ids"], mask):
        if m:
            break
        head.append(tok)
    return ids + head


def replay(ep: dict, capture: set[int] | None = None, window: int = 50, max_key_tokens: int = 4096) -> dict:
    """Replay one episode (a stream line). ``capture``: reset indices (1 = the first
    summary turn) whose env save and summary prompt to keep."""
    info = ep["traces"][0]["task"]["data"]["info"]
    seed, max_actions = int(info["seed"]), int(info.get("max_actions", 1))
    game = glyph.CraftaxGame(seed, None)
    memory = Memory(max_key_tokens=max_key_tokens)
    deaths: list = []
    total, max_floor = 0.0, 0
    forfeits, actions, per_turn = Counter(), Counter(), Counter()
    unlocked_at: dict[str, int] = {}
    resets = [state_record(game, 0, "start", k=0)]
    segments = []  # per trace: game turns, actions, achievements unlocked, deaths
    captured = {}
    checked = mismatched = turns = 0
    inherited = None
    for t in sorted(ep["traces"], key=lambda t: (t.get("info") or {}).get("segment", 0)):
        trace = vf.Trace.model_validate(t)
        if not trace.branches:
            continue
        raw_nodes = t["nodes"]
        index = {id(n): i for i, n in enumerate(trace.nodes)}
        seg = {"segment": (t.get("info") or {}).get("segment", 0), "turns": 0, "actions": 0, "forfeits": 0,
               "unlocked": [], "deaths": 0}
        obs, obs_i = "", None
        for node in trace.branches[-1].nodes:
            m = node.message
            if isinstance(m, UserMessage):
                obs, obs_i = content_text(m.content), index[id(node)]
                if "[HUD]" in obs:
                    checked += 1
                    mismatched += not hud_check(game, obs)
                continue
            if not (isinstance(m, AssistantMessage) and node.sampled and any(node.mask)):
                continue
            reply = content_text(m.content) if m.content is not None else ""
            if SUMMARY_ASK.search(obs):
                s = SUMMARY.search(reply)
                k = len(resets)
                if capture and k in capture:
                    captured[k] = {
                        "save": {
                            "version": 1, "seed": seed, "block": 0, "game": game.dump(),
                            "total": total, "turns": turns, "max_floor": max_floor,
                            "forfeits": dict(forfeits), "actions": dict(actions), "per_turn": dict(per_turn),
                            "unlocked_at": dict(unlocked_at), "deaths": list(deaths), "played_seconds": 0.0,
                            "summary": None, "last_summary": None, "memory": dict(memory.entries),
                        },
                        "prompt_ids": window_prompt(raw_nodes, obs_i, raw_nodes[index[id(node)]], window),
                        "inherited": inherited,
                    }
                resets.append(state_record(game, len(deaths), "reset", s.group(1).strip() if s else None, k=k))
                memory.apply(reply)  # the summary turn may edit memory too
                inherited = s.group(1).strip() if s else None
                segments.append(seg)
                seg = None
                continue
            memory.apply(reply)
            turns += 1
            seg["turns"] += 1
            indices, failure, _ = glyph.parse_actions(reply, max_actions)
            if not indices:
                forfeits[failure] += 1
                per_turn[0] += 1
                seg["forfeits"] += 1
                done = game.forfeit()
            else:
                done, played = False, 0
                for i in indices:
                    step = game.step(i)
                    played += 1
                    actions[glyph.ACTION_NAMES[i]] += 1
                    total += step.reward
                    seg["actions"] += 1
                    seg["unlocked"] += step.unlocked
                    for name in step.unlocked:
                        unlocked_at.setdefault(name, game.turn)
                    max_floor = max(max_floor, game.floor)
                    done = step.done
                    if done:
                        break
                per_turn[played] += 1
            if done and game.health <= 0:
                deaths.append({"turn": game.turn, "floor": game.floor, "pos": list(game.position)})
                at = game.respawn("drop_all")
                deaths[-1]["respawn"] = list(at)
                seg["deaths"] += 1
        if seg is not None:
            segments.append(seg)
    resets.append(state_record(game, len(deaths), "end", k=len(resets)))
    rules = ep["traces"][0]["task"]["data"].get("system_prompt") or ""
    return {"seed": seed, "episode": ep["id"], "rules": rules, "resets": resets, "segments": segments,
            "hud_checked": checked, "hud_mismatched": mismatched, "captured": captured}


def episodes(eval_dir: str):
    for path in sorted(glob.glob(f"{eval_dir}/run/*/monitors/file/traces/stream/*.zst")):
        with subprocess.Popen(["zstdcat", path], stdout=subprocess.PIPE, text=True) as p:
            for line in p.stdout:
                yield json.loads(line)


def main():
    eval_dir, out = sys.argv[1], sys.argv[2]
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    stride = int(sys.argv[4]) if len(sys.argv) > 4 else 8
    offset = int(sys.argv[5]) if len(sys.argv) > 5 else 0
    done = 0
    with open(out, "w") as f:
        for i, ep in enumerate(episodes(eval_dir)):
            if i % stride != offset or done >= n:
                continue
            rec = replay(ep)
            rec.pop("captured")
            done += 1
            print(f"seed {rec['seed']}: resets {len(rec['resets'])} HUD mismatches "
                  f"{rec['hud_mismatched']}/{rec['hud_checked']}", flush=True)
            f.write(json.dumps(rec) + "\n")
            f.flush()


if __name__ == "__main__":
    main()
