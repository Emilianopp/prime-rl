"""The Q/V judge of a Craftax segment: the true state at its boundaries, Gemini's grade
of it and of the handoff summary written there, and the segment's reward.

At every reset the judge sees the rules, the true state (HUD, the full map of the floor
with absolute coordinates, the nearest key blocks, unlocked and locked achievements with
their points) and, when a summary was written there, that summary and the one before
it. It returns position parts (tool 0-4, ready 0-3, safe 0-2, near 0-1), each locked
achievement a good agent with an ideal handoff could unlock by the next reset with its
probability, the expected position gain, an ideal handoff, and the summary's flaws (each
with a quote and the contradicting fact). From those, in code (`score`):

  V  = B + P                                  B: banked achievement points (1/3/5/8 by tier)
  G* = min(G_MAX, sum_j p_j points_j + expected position gain)
  D  = 1 - prod_flaws (1 - w_flaw)             (no summary under review: D = 1)
  Q  = V + G* (1 - D)                          (game start and end: Q = V)
  R  = clip((Q_end - Q_start) / max(G*_start, 1), -1, 1)     a segment's reward

`state_record` is what the env writes at segment boundaries (`CraftaxEnvConfig.state_log`);
`score_segment` is the scorer of prime-rl's `trace_credit` algorithm.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import time
from pathlib import Path

import httpx
import numpy as np

from craftax_env import glyph

KEY_BLOCKS = ("tree", "water", "stone", "coal", "iron", "diamond", "sapphire", "ruby",
              "crafting table", "furnace", "lava")


def full_map(game: glyph.CraftaxGame) -> tuple[str, str]:
    """The current floor's 48x48 map, every tile revealed, with row/column labels."""
    import jax.numpy as jnp
    from craftax.craftax import ascii_renderer
    from craftax.craftax import glyphbench_api as api

    seen = jnp.ones((9, 48, 48), dtype=bool)
    chars = ascii_renderer.render_craftax_ascii_grid_full(game.state, seen)[0]
    rows = api.unicode_grid_to_string(chars).split("\n")
    head = "    " + "".join(str(c // 10) for c in range(48)) + "\n    " + "".join(str(c % 10) for c in range(48))
    body = "\n".join(f"{r:>3} {row}" for r, row in enumerate(rows))
    return head + "\n" + body, api.unicode_legend_from_char_grid(chars)


def nearest(game: glyph.CraftaxGame) -> dict[str, list]:
    """Nearest cell of each key block (and the down ladder) on this floor: [row, col, manhattan]."""
    from craftax.craftax.constants import ItemType

    level, (r, c) = game.floor, game.position
    out = {}

    def closest(name, cells):
        if len(cells):
            d = np.abs(cells - [r, c]).sum(1)
            i = int(d.argmin())
            out[name] = [int(cells[i][0]), int(cells[i][1]), int(d[i])]

    grid = np.asarray(game.state.map[level])
    for value, name in game._rt["block_names"].items():
        if name in KEY_BLOCKS:
            closest(name, np.argwhere(grid == value))
    closest("ladder down", np.argwhere(np.asarray(game.state.item_map[level]) == ItemType.LADDER_DOWN.value))
    return out


def points(game: glyph.CraftaxGame) -> dict[str, float]:
    """Every achievement's Craftax reward (1/3/5/8 by tier)."""
    w = np.asarray(game._rt["weights"])
    return {a.name.lower(): float(w[int(a.value)]) for a in game._rt["achievements"]}


def state_record(game: glyph.CraftaxGame, deaths: int, kind: str, summary: str | None = None, k: int | None = None) -> dict:
    """What the judge sees of a state."""
    _legend, _grid, hud, _msg = game.render()
    grid, legend = full_map(game)
    pts = points(game)
    return {
        "kind": kind, "k": k, "turn": game.turn, "timestep": game.timestep,
        "hud": hud, "map": grid, "legend": legend, "nearest": nearest(game),
        "achievements": {a: pts[a] for a in sorted(game.achievements)},
        "locked": {a: p for a, p in pts.items() if a not in game.achievements},
        "B": float(game.achievement_score), "deaths": deaths, "summary": summary,
    }


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


async def call(client: httpx.AsyncClient, text: str, model: str = MODEL, url: str = URL,
               params: dict | None = None) -> tuple[dict, dict]:
    """One judge call to an OpenAI-compatible chat endpoint; `params` are extra request
    fields (default temperature 0.2; reasoning models take e.g. reasoning_effort instead)."""
    body = {"model": model, "messages": [{"role": "user", "content": text}],
            "response_format": {"type": "json_object"}, **({"temperature": 0.2} if params is None else params)}
    for attempt in range(5):
        try:
            r = await client.post(url, json=body, timeout=300)
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




# --- the segment scorer (prime-rl's trace_credit) ---

SUMMARY_HEADER = "HANDOFF SUMMARY (written by you before your context was reset):"
"""Heads the inherited summary in a segment's system prompt (verifiers' SegmentsConfig default)."""
SUMMARY_ASK = re.compile(r"context is about to be reset", re.IGNORECASE)


def write_boundary(state_dir: str, game: glyph.CraftaxGame, deaths: int, end_of: str | None, start_of: str | None) -> None:
    """The env's side of the scorer: the true state where segment `end_of` ended and
    `start_of` began, one file per trace and side, written atomically."""
    folder = Path(os.path.expandvars(state_dir))
    folder.mkdir(parents=True, exist_ok=True)
    record = state_record(game, deaths, "boundary")
    for trace, side, extra in ((end_of, "end", {"next": start_of}), (start_of, "start", {})):
        if trace is None:
            continue
        tmp = folder / f".{trace}.{side}.{os.getpid()}"
        tmp.write_text(json.dumps({**extra, "state": record}))
        tmp.replace(folder / f"{trace}.{side}.json")


class _Run:
    """What the scorer keeps across the segments of a run (one orchestrator process)."""

    def __init__(self) -> None:
        self.handoff: dict[str, asyncio.Future] = {}  # next trace id -> the end score of its predecessor
        self.cost = 0.0
        self.calls = 0
        self.stopped: str | None = None
        self.client: httpx.AsyncClient | None = None


_RUN = _Run()


async def _boundary(folder: Path, trace: str, side: str, timeout: float) -> dict | None:
    path = folder / f"{trace}.{side}.json"
    deadline = time.time() + timeout
    while not path.exists():
        if time.time() > deadline:
            return None
        await asyncio.sleep(5)
    return json.loads(path.read_text())


def _system(trace) -> tuple[str, str | None]:
    """(rules, inherited summary) from the segment's system prompt."""
    system = trace.task.data.system_prompt or ""
    if not system:
        nodes = trace.branches[-1].nodes if trace.branches else trace.nodes
        system = next((m.message.content for m in nodes if m.message.role == "system"), "") or ""
    if SUMMARY_HEADER in system:
        rules, inherited = system.split(SUMMARY_HEADER, 1)
        return rules.strip(), inherited.strip()
    return system.strip(), None


def _written_summary(trace) -> str | None:
    """The handoff summary the segment wrote at its end, read as the segment switch reads
    it; None if its last turn was not a summary turn (the episode ended)."""
    from verifiers.v1.segments import _summary_text
    from verifiers.v1.types import content_text

    nodes = trace.branches[-1].nodes if trace.branches else trace.nodes
    last_user = None
    summary = None
    for node in nodes:
        role = node.message.role
        if role == "user":
            last_user = content_text(node.message.content)
        elif role == "assistant" and node.sampled and any(node.mask):
            if last_user is not None and SUMMARY_ASK.search(last_user):
                reply = content_text(node.message.content) if node.message.content is not None else ""
                summary = _summary_text(reply) or _summary_text(node.message.reasoning_content or "")
            else:
                summary = None
    return summary


async def _judge(text: str, model: str, price_in: float, price_out: float, url: str, key_var: str,
                 params: dict | None) -> dict:
    if _RUN.client is None:
        _RUN.client = httpx.AsyncClient(headers={"Authorization": f"Bearer {os.environ[key_var]}"})
    j, usage = await call(_RUN.client, text, model, url=url, params=params)
    prompt_tokens = usage.get("prompt_tokens", 0) or 0
    out_tokens = (usage.get("total_tokens", 0) or 0) - prompt_tokens  # thinking is billed as output
    _RUN.cost += (prompt_tokens * price_in + out_tokens * price_out) / 1e6
    _RUN.calls += 1
    return j


async def score_segment(
    trace,
    state_dir: str,
    model: str = "gemini-3.1-flash-lite",
    cost_cap: float = 10.0,
    price_in: float = 0.25,
    price_out: float = 1.5,
    stop_path: str | None = "$OUT/STOP",
    wait_seconds: float = 1800.0,
    url: str = URL,
    key_var: str = "GEMINI_API_KEY",
    params: dict | None = None,
    death_penalty: float = 0.0,
) -> tuple[float | None, dict]:
    """The reward of one segment (a trace of a game played in segments): R from the Q
    of its start (state + inherited summary) and of its end (state + the summary it
    wrote). Q at a reset is judged once, by the segment that wrote the summary there, and
    handed to the next segment. Past `cost_cap` dollars of judge calls (at `price_in` /
    `price_out` per million tokens) it judges nothing more and writes `stop_path`, which
    ends a run under rl.sbatch. The judge is `model` at the OpenAI-compatible `url`, with
    the API key in env var `key_var` and extra request fields `params` (see `call`).
    `death_penalty` is subtracted from R per death in the segment (before the clip).
    Returns (None, info) when it cannot score."""
    info: dict = {"trace": trace.id}
    if _RUN.stopped:
        return None, {**info, "skipped": _RUN.stopped}
    folder = Path(os.path.expandvars(state_dir))
    start_rec, end_rec = await asyncio.gather(
        _boundary(folder, trace.id, "start", wait_seconds), _boundary(folder, trace.id, "end", wait_seconds))
    if start_rec is None or end_rec is None:
        return None, {**info, "error": "no boundary state from the env"}
    nxt = end_rec.get("next")
    loop = asyncio.get_running_loop()
    handoff = _RUN.handoff.setdefault(nxt, loop.create_future()) if nxt else None
    rules, inherited = _system(trace)
    written = _written_summary(trace)
    start_state, end_state = start_rec["state"], end_rec["state"]
    try:
        start = None
        own = _RUN.handoff.pop(trace.id, None)
        if own is not None:
            try:
                start = await asyncio.wait_for(asyncio.shield(own), timeout=wait_seconds)
            except (asyncio.TimeoutError, Exception):
                start = None
        jobs = {"end": prompt(rules, end_state, written is not None, written, inherited)}
        if start is None:
            jobs["start"] = prompt(rules, start_state, inherited is not None, inherited, None)
        replies = dict(zip(jobs, await asyncio.gather(*(_judge(t, model, price_in, price_out, url, key_var, params) for t in jobs.values()))))
        if start is None:
            start = score(start_state, replies["start"], inherited is not None, inherited)
        end = score(end_state, replies["end"], written is not None, written)
    except Exception as exc:
        if handoff is not None and not handoff.done():
            handoff.set_result(None)
        return None, {**info, "error": f"{type(exc).__name__}: {exc}"[:300]}
    if handoff is not None and not handoff.done():
        handoff.set_result(end)
    if _RUN.cost >= cost_cap and not _RUN.stopped:
        _RUN.stopped = f"judge cost ${_RUN.cost:.2f} reached the ${cost_cap:.2f} cap after {_RUN.calls} calls"
        if stop_path:
            Path(os.path.expandvars(stop_path)).write_text(_RUN.stopped + "\n")
    died = max(0, int(end_state["deaths"]) - int(start_state["deaths"]))
    reward = max(-1.0, min(1.0, segment_reward(start, end) - death_penalty * died))
    keep = ("V", "G_star", "D", "Q", "P")
    return reward, {**info, "start": {k: start[k] for k in keep}, "end": {k: end[k] for k in keep},
                    "flaws": [f.get("type") for f in end["flaws"]], "wrote_summary": written is not None,
                    "inherited": inherited is not None, "B": [start_state["B"], end_state["B"]],
                    "deaths": [start_state["deaths"], end_state["deaths"]],
                    "cost": round(_RUN.cost, 4), "calls": _RUN.calls}
