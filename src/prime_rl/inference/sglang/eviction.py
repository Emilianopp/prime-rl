"""KV-Streams eviction plans: which tokens of a trace's session die at each call.

The engine (SGLang ``kv_eviction``) evicts any sorted, disjoint spans of a session's
resident tokens and the trainer masks any spans per call (``kv_death``), so what to
evict is up to the plans. ``SessionTable`` keeps the bookkeeping - the dead spans,
survivor coordinates, the engine request and the training record - and asks each of
its plans which chronological spans should be dead from a call on:

* ``TurnWindow``: the model-agnostic turn schedule (``r{max_turns}-{stride}``).
* model plans (``model_evictions``): tokens of one model's earlier turns that its
  chat template drops and the append-only token stream keeps, where attending to
  them makes the model worse.

Self-contained (no ``prime_rl`` imports): the SGLang server imports it from its own
environment.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass
class CallView:
    """A call as an eviction plan sees it. Positions are chronological: the trace's
    token stream, dead tokens included."""

    prompt: list[int]
    """The session's resident tokens followed by this call's new tail."""
    ends: list[int]
    """Positions just after each message end in ``prompt``."""
    protected: int
    """The protected prefix: never evicted."""
    limit: int
    """Only tokens before it have KV and can die (the last sampled token has none yet)."""
    dead: list[list[int]]
    """Spans already dead, sorted and disjoint."""
    state: dict[str, Any]
    """The plan's own state for this trace session, kept across the session's calls."""


class EvictionPlan(Protocol):
    def evict(self, call: CallView) -> list[list[int]]:
        """Chronological ``[start, end)`` spans that should be dead from this call on.
        They may overlap tokens already dead; parts outside ``[protected, limit)`` are
        ignored."""
        ...


@dataclass
class TurnWindow:
    """The kv-eviction turn schedule (``r{max_turns}-{stride}``), on token ids.

    The first message (through its end marker) is protected. Every two later
    message ends close one turn. Once ``max_turns`` turns are complete, the
    ``stride`` oldest live turns die, and again every ``stride`` turns after
    that, so the live window saws between ``max_turns - stride`` and
    ``max_turns - 1`` completed turns plus the turn in progress.
    """

    max_turns: int
    stride: int
    message_end_id: int
    """Token id that ends a chat message (``<|im_end|>`` for Qwen)."""

    def __post_init__(self):
        if self.max_turns < 1 or not 1 <= self.stride <= self.max_turns:
            raise ValueError(f"need max_turns >= 1 and 1 <= stride <= max_turns, got {self.max_turns}, {self.stride}")

    def live_turns(self, completed: int) -> int:
        if completed < self.max_turns:
            return completed
        return completed - ((completed - self.max_turns) // self.stride + 1) * self.stride

    def message_ends(self, token_ids: list[int], offset: int = 0) -> list[int]:
        """Positions just after each message end marker in ``token_ids``, which start at ``offset``."""
        end = self.message_end_id
        return [offset + i + 1 for i, token in enumerate(token_ids) if token == end]

    def layout(self, token_ids: list[int]) -> tuple[int, int]:
        """(protected prefix length, chronological tokens after it that are dead)."""
        return self.layout_ends(self.message_ends(token_ids))

    def layout_ends(self, ends: list[int]) -> tuple[int, int]:
        """``layout`` from the prompt's message end positions."""
        if not ends:
            return 0, 0
        protected, turn_ends = ends[0], ends[2::2]
        dead_turns = len(turn_ends) - self.live_turns(len(turn_ends))
        return protected, (turn_ends[dead_turns - 1] - protected if dead_turns else 0)

    def evict(self, call: CallView) -> list[list[int]]:
        protected, dead = self.layout_ends(call.ends)
        # A retired turn that ends past the limit waits for the next call.
        return [[protected, protected + dead]] if dead and protected + dead <= call.limit else []


def occurrences(token_ids: list[int], start: int, stop: int, marker: tuple[int, ...]) -> tuple[list[list[int]], int]:
    """Every ``marker`` in ``token_ids[start:stop]`` as a ``[start, end)`` span, and
    where the next scan starts: one straddling ``stop`` is looked for again."""
    spans, i, n = [], start, len(marker)
    while True:
        try:
            i = token_ids.index(marker[0], i, stop)
        except ValueError:
            return spans, max(start, stop)
        if i + n > stop:
            return spans, i
        if tuple(token_ids[i : i + n]) == marker:
            spans.append([i, i + n])
            i += n
        else:
            i += 1


def merge(spans: list[list[int]]) -> list[list[int]]:
    """Sorted, disjoint union of ``spans`` (touching spans join)."""
    out: list[list[int]] = []
    for start, end in sorted(spans):
        if out and start <= out[-1][1]:
            out[-1][1] = max(out[-1][1], end)
        else:
            out.append([start, end])
    return out


def subtract(spans: list[list[int]], dead: list[list[int]]) -> list[list[int]]:
    """The parts of sorted, disjoint ``spans`` outside sorted, disjoint ``dead``."""
    out = []
    for start, end in spans:
        for dead_start, dead_end in dead:
            if dead_end <= start or dead_start >= end:
                continue
            if dead_start > start:
                out.append([start, dead_start])
            start = max(start, dead_end)
        if end > start:
            out.append([start, end])
    return out


def survivor(position: int, dead: list[list[int]]) -> int:
    """``position`` (chronological) in resident coordinates: minus the dead tokens before it."""
    return position - sum(max(0, min(end, position) - start) for start, end in dead if start < position)
