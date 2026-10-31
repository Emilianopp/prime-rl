"""Per-turn format and memory rewards for a Craftax segment, read off its trace.

The game reward credits a whole episode; these credit single replies, so a reply that
forfeits its turn is pushed down without touching the turns around it. Every signal is
deterministic and comes from the trace itself: the reply (parsed with the env's own
action parser) and the next observation, which carries the env's verdict on the reply
(the [Memory] block's result lines, the entries it read, the key index).

Penalties: a game turn with no playable action (no tag, or an unknown name; reported as
`action_in_reasoning` when the tag only appears in the reasoning), memory tags left in
the reasoning, reading an entry that does not exist, creating more than
`max_new_keys` entries in one segment, and a summary turn without a `<summary>` in the
reply. Bonuses, at most once per segment: reading an existing entry in the first
`read_window` turns of a segment that opened with memory, and the first saved edit
(write, rewrite or replace). Any other reply scores 0.

    scores = turn_rewards(trace)   # [(node index, reward, [reasons])] for scored replies
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, replace

import verifiers.v1 as vf
from verifiers.v1.types import AssistantMessage, UserMessage, content_text

from craftax_env import glyph

SUMMARY_ASK = re.compile(r"context is about to be reset", re.IGNORECASE)
SUMMARY = re.compile(r"<summary>\s*\S.*?</summary>", re.DOTALL | re.IGNORECASE)
INDEX = re.compile(r"^\[Memory\] (.*)$", re.MULTILINE)
INDEX_KEY = re.compile(r"(.+?) \([\d,]+ tok\)$")
RESULT = re.compile(r'^(write|rewrite|replace|delete|read) "(.+?)": (.*)$', re.MULTILINE)
READ_BLOCK = re.compile(r'^\[Memory "(.+?)"\]$', re.MULTILINE)
MEMORY_IN_REASONING = "memory tags inside your reasoning were ignored"


@dataclass
class TurnRewardConfig:
    forfeit: float = -1.0
    memory_in_reasoning: float = -0.5
    read_missing: float = -0.5
    key_spam: float = -0.5
    """Per entry created beyond `max_new_keys` in one segment."""
    max_new_keys: int = 5
    missing_summary: float = -1.0
    read_after_reset: float = 1.0
    read_window: int = 3
    """Game turns at the start of a segment in which a read earns `read_after_reset`."""
    first_write: float = 0.5
    max_actions: int | None = None
    """Actions a reply may play, as the env parsed them; None reads the task's info (a
    trace loaded back from a stream keeps only the base task fields: pass it then)."""


@dataclass
class TurnScore:
    node: int
    """Index of the reply in `trace.nodes`."""
    reward: float = 0.0
    reasons: list[str] = field(default_factory=list)

    def add(self, value: float, reason: str) -> None:
        if value:
            self.reward += value
            self.reasons.append(reason)


def index_keys(observation: str) -> set[str] | None:
    """The keys the observation's [Memory] index lists (None: no memory block)."""
    m = INDEX.search(observation)
    if m is None:
        return None
    if m.group(1).startswith("no entries yet"):
        return set()
    return {k.group(1) for part in m.group(1).split(" · ") if (k := INDEX_KEY.match(part.strip()))}


def turn_rewards(trace: vf.Trace, config: TurnRewardConfig | None = None, **overrides) -> list[TurnScore]:
    """Score every reply of the trace's last branch (see the module doc)."""
    config = replace(config or TurnRewardConfig(), **overrides)
    branches = trace.branches
    if not branches:
        return []
    info = getattr(trace.task.data, "info", None) or {}
    max_actions = config.max_actions or int(info.get("max_actions", 1))
    nodes = list(trace.nodes)
    index_of = {id(node): i for i, node in enumerate(nodes)}
    path = branches[-1].nodes
    # (reply node, the observation before it, the observation after it)
    turns: list[tuple[vf.MessageNode, str, str | None]] = []
    before: str | None = None
    for k, node in enumerate(path):
        message = node.message
        if isinstance(message, UserMessage):
            before = content_text(message.content)
        elif isinstance(message, AssistantMessage) and node.sampled and any(node.mask):
            after = next((content_text(n.message.content) for n in path[k + 1 :] if isinstance(n.message, UserMessage)), None)
            turns.append((node, before or "", after))

    opened_with = index_keys(turns[0][1]) if turns else None
    scores: list[TurnScore] = []
    game_turns = new_keys = 0
    read_paid = write_paid = False
    for node, obs, after in turns:
        message = node.message
        reply = message.content or ""
        reasoning = message.reasoning_content or ""
        score = TurnScore(index_of[id(node)])
        scores.append(score)
        if SUMMARY_ASK.search(obs):
            if not SUMMARY.search(reply):
                score.add(config.missing_summary, "missing_summary")
            continue  # its memory results land in the next segment
        game_turns += 1
        indices, failure, _ = glyph.parse_actions(reply, max_actions)
        if not indices:
            in_reasoning = failure == "no_action_tag" and glyph.parse_actions(reasoning, max_actions)[0]
            score.add(config.forfeit, "action_in_reasoning" if in_reasoning else failure or "forfeit")
        if after is None:
            continue
        if MEMORY_IN_REASONING in after:
            score.add(config.memory_in_reasoning, "memory_in_reasoning")
        results = RESULT.findall(after)
        for op, _, outcome in results:
            if op == "read" and outcome.startswith("no such entry"):
                score.add(config.read_missing, "read_missing")
        if not write_paid and any(op in ("write", "rewrite", "replace") and outcome.startswith("saved") for op, _, outcome in results):
            write_paid = True
            score.add(config.first_write, "first_write")
        if not read_paid and opened_with and game_turns <= config.read_window and READ_BLOCK.search(after):
            read_paid = True
            score.add(config.read_after_reset, "read_after_reset")
        keys_before, keys_after = index_keys(obs), index_keys(after)
        if keys_before is not None and keys_after is not None:
            created = len(keys_after - keys_before)
            spam = max(0, new_keys + created - config.max_new_keys) - max(0, new_keys - config.max_new_keys)
            new_keys += created
            score.add(config.key_spam * spam, "key_spam")
    return scores


__all__ = ["TurnRewardConfig", "TurnScore", "index_keys", "turn_rewards"]
