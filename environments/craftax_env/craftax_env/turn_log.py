"""Per-turn log of a Craftax episode or segment, for replay (scripts/craftax/lineage_replay.py).

One gzipped JSON file per segment, ``<dir>/<node_id>.json.gz``, next to the episode log:
the header says where the segment starts (seed, parent, the game turn) and how deaths were
undone; ``turns`` holds every model turn in order: the observation the model answered, its
reasoning and reply, the actions the game played (none for a frozen forfeit) and where the
player stood after them; ``notes`` holds the notes turn(s). The game is deterministic given
the seed and the played actions (CraftaxGame splits its key once per upstream step), so a
replay from the root through every segment of a lineage rebuilds each frame; ``end_hash``
(the game state and key at the segment's end) checks that it did.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
from typing import Any

import numpy as np

FORMAT = 1


def state_hash(game: Any) -> str:
    """sha256 over the game's state arrays (in pytree order), its RNG key and turn counter."""
    jax = game._rt["jax"]
    h = hashlib.sha256()
    for leaf in jax.tree_util.tree_leaves(game.state):
        a = np.ascontiguousarray(np.asarray(leaf))
        h.update(str(a.dtype).encode())
        h.update(a.tobytes())
    h.update(np.ascontiguousarray(np.asarray(game._key)).tobytes())
    h.update(str(int(game.turn)).encode())
    return h.hexdigest()


def where(game: Any) -> list[int]:
    """[floor, row, col] of the player."""
    return [game.floor, *map(int, game.position)]


def directory(episode_log: str | None) -> str | None:
    """The turn-log directory that goes with an episode log: ``turns/`` beside it."""
    return os.path.join(os.path.dirname(os.path.abspath(episode_log)), "turns") if episode_log else None


def write(path: str, payload: dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.tmp{os.getpid()}"
    with gzip.open(tmp, "wt", compresslevel=6) as f:
        json.dump({"format": FORMAT, **payload}, f, separators=(",", ":"))
    os.replace(tmp, path)


def read(path: str) -> dict[str, Any]:
    with gzip.open(path, "rt") as f:
        return json.load(f)


class ReplayMismatch(RuntimeError):
    """The replayed game left the logged one (a different code version, or a lost step)."""


def replay_segment(game: Any, seg: dict[str, Any], on_turn=None, fail=None, label: str = "") -> Any:
    """Play one segment's logged turns on ``game`` (in place), calling ``on_turn(row, game)``
    after each and ``fail(text)`` on a difference from the log. Returns the game."""

    def bad(text: str) -> None:
        if fail is None:
            raise ReplayMismatch(text)
        fail(text)

    label = label or str(seg.get("node_id"))
    if seg.get("start_hash") and state_hash(game) != seg["start_hash"]:
        bad(f"{label}: start state differs from the log")
    safe = seg.get("respawn_safe_radius")
    for row in seg["turns"]:
        if row.get("frozen"):
            game.forfeit()
        else:
            for a in row["acts"]:
                game.step(int(a))
        if row.get("death"):
            at = game.respawn(seg["death_inventory"], safe)
            if list(at) != list(row["death"]["respawn"]):
                bad(f"{label} turn {row['turn']}: respawned at {list(at)}, log says {row['death']['respawn']}")
        if row.get("message") is not None:
            game.message = row["message"]
        if row.get("pos") is not None and where(game) != list(row["pos"]):
            bad(f"{label} turn {row['turn']}: player at {where(game)}, log says {row['pos']}")
        if on_turn is not None:
            on_turn(row, game)
    if seg.get("end_hash") and state_hash(game) != seg["end_hash"]:
        bad(f"{label}: end state differs from the log")
    return game


def new_game(seg: dict[str, Any]) -> Any:
    """The game a root segment starts from."""
    from craftax_env import glyph

    return glyph.CraftaxGame(int(seg["seed"]), seg.get("max_turns"))


def replay(segments: list[dict[str, Any]], on_turn=None, strict: bool = True):
    """Replay a lineage: ``segments`` are turn logs from the root's child down to a leaf, in
    order. Calls ``on_turn(k, row, game)`` after each turn of segment k (and once per segment
    with row None, before its first turn) and returns the game at the end. Checks every turn's
    position and each segment's start and end hash; ``strict`` raises ReplayMismatch on the
    first difference, else returns (game, differences)."""
    problems: list[str] = []
    game = new_game(segments[0])
    for k, seg in enumerate(segments):
        if on_turn is not None:
            on_turn(k, None, game)
        replay_segment(game, seg, None if on_turn is None else (lambda row, g, k=k: on_turn(k, row, g)),
                       None if strict else problems.append, label=f"segment {k} ({seg.get('node_id')})")
    return game if strict else (game, problems)
