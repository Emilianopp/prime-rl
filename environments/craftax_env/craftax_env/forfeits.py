"""What kind of reply was forfeited, and whether the episode collapsed into forfeiting.

Measured on the base model (job 11470355, 2026-11-01): only ~5% of forfeits are length
cuts. Most are replies that stop by themselves without an action tag, and they cluster:
about a fifth of episodes fall into a streak that lasts to the end. This module tracks
that per episode, so runs (and RL steps) can be compared. Pure Python, no JAX, so the
offline report can load it by path.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Any

KINDS = (
    "length",  # the reply hit the output-token cap
    "reasoning_only",  # stopped by itself inside the reasoning: empty content, no tag
    "action_in_reasoning",  # empty content, the <action> tag only inside the reasoning
    "content_no_tag",  # content without any <action> tag (often garbage)
    "malformed_tag",  # an opening <action> without a complete tag
    "empty",  # no reasoning, no content
    "unknown_name",  # a complete tag with a name that is not an action
)
COLLAPSE_STREAK = 10
"""A run of this many consecutive forfeits counts as a collapse."""
BUCKET = 10
"""Turns per bucket of the per-episode forfeit histogram."""

_ACTION = re.compile(r"<\s*action\s*>(.*?)<\s*/\s*action\s*>", re.DOTALL | re.IGNORECASE)
_OPEN = re.compile(r"<\s*action\s*>", re.IGNORECASE)


def classify(content: str | None, reasoning: str | None, finish_reason: str | None, failure: str | None) -> str:
    """The kind of one forfeited reply. `failure` is `glyph.parse_actions`' reason."""
    if failure == "unknown_name":
        return "unknown_name"
    if finish_reason == "length":
        return "length"
    content, reasoning = (content or "").strip(), (reasoning or "").strip()
    if not content:
        if _ACTION.search(reasoning):
            return "action_in_reasoning"
        return "reasoning_only" if reasoning else "empty"
    if _OPEN.search(content):
        return "malformed_tag"
    return "content_no_tag"


def reply_parts(segment: Any, trace: Any) -> tuple[str | None, str | None, str | None]:
    """(content, reasoning, finish_reason) of the reply just received; Nones where the
    harness does not expose them (the reasoning then falls back to `last_reply`)."""
    content = reasoning = finish = None
    for message in reversed(getattr(segment, "messages", None) or []):
        if getattr(message, "role", None) == "assistant":
            content, reasoning = getattr(message, "content", None), getattr(message, "reasoning_content", None)
            break
    else:
        content = getattr(segment, "last_reply", None)
    calls = getattr(trace, "calls", None) or []
    if calls:
        finish = getattr(calls[-1], "finish_reason", None)
    return content, reasoning, finish


class ForfeitTracker:
    """Per-episode forfeit kinds, streaks and the per-bucket histogram."""

    def __init__(self) -> None:
        self.kinds: Counter[str] = Counter()
        self.turns = 0
        self.streak = 0
        self.max_streak = 0
        self.collapses = 0
        self.collapse_onset: int | None = None  # turn (0-based) at which the first collapse began
        self.buckets: list[int] = []

    def update(self, kind: str | None) -> None:
        """One reply: `kind` is None for a reply whose action was played."""
        b = self.turns // BUCKET
        if b >= len(self.buckets):
            self.buckets.append(0)
        if kind is None:
            self.streak = 0
        else:
            self.kinds[kind] += 1
            self.buckets[b] += 1
            self.streak += 1
            self.max_streak = max(self.max_streak, self.streak)
            if self.streak == COLLAPSE_STREAK:
                self.collapses += 1
                if self.collapse_onset is None:
                    self.collapse_onset = self.turns - COLLAPSE_STREAK + 1
        self.turns += 1

    def metrics(self) -> dict[str, float]:
        out = {f"forfeit_{k}": float(self.kinds.get(k, 0)) for k in KINDS}
        out.update({
            "max_forfeit_streak": float(self.max_streak),
            "forfeit_collapsed": 1.0 if self.collapses else 0.0,
            "collapse_onset_turn": float(self.collapse_onset if self.collapse_onset is not None else -1),
        })
        return out

    def log(self) -> dict[str, Any]:
        return {
            "forfeit_kinds": {k: self.kinds[k] for k in KINDS if self.kinds.get(k)},
            "max_forfeit_streak": self.max_streak,
            "forfeit_collapses": self.collapses,
            "collapse_onset": self.collapse_onset,
            "forfeit_buckets": self.buckets,
        }
