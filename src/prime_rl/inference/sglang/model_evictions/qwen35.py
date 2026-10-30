"""Qwen3.5: evict earlier turns' empty think blocks.

With thinking disabled the generation prompt prefills an empty
``<think>\\n\\n</think>\\n\\n`` before every reply. The chat template renders it only
for assistant turns after the last user query; the renderer keeps every turn's block
in the token stream so it stays append-only. Left attended, a history of empty blocks
makes Qwen3.5 end replies early and then imitate that (Crafter, thinking disabled: 37%
of replies without an action, 2% with this plan). A block dies once a later user
query arrives; a tool response is rendered as a user message but is not a query.

Reasoning (thinking enabled) is kept, although the template drops it too: it is the
model's memory of its plans across turns, and evicting it costs more than the context
it frees (Crafter, thinking enabled, 300 steps: 5.4 achievements kept, 4.0 evicted).
"""

from __future__ import annotations

from dataclasses import dataclass

try:
    from ..eviction import CallView, occurrences
except ImportError:  # the SGLang server runs as a script, its directory on sys.path
    from eviction import CallView, occurrences

MARKERS = {
    "empty_think": "<think>\n\n</think>\n\n",
    "user": "<|im_start|>user\n",
    "tool_response": "<tool_response>",
}


@dataclass
class Plan:
    empty_think_ids: tuple[int, ...]
    user_ids: tuple[int, ...]
    tool_response_ids: tuple[int, ...]

    def __post_init__(self):
        for name in ("empty_think_ids", "user_ids", "tool_response_ids"):
            setattr(self, name, tuple(getattr(self, name)))
            if not getattr(self, name):
                raise ValueError(f"{name} must not be empty")

    def _is_query(self, prompt: list[int], i: int) -> bool:
        """A user message starts at ``i`` and is not a tool response."""
        n = len(self.user_ids)
        if tuple(prompt[i : i + n]) != self.user_ids:
            return False
        return tuple(prompt[i + n : i + n + len(self.tool_response_ids)]) != self.tool_response_ids

    def evict(self, call: CallView) -> list[list[int]]:
        state = call.state
        # Empty blocks in what the engine holds wait for a later query.
        start = max(state.get("scan", call.protected), call.protected)
        found, state["scan"] = occurrences(call.prompt, start, call.limit, self.empty_think_ids)
        pending = state.setdefault("pending", [])
        pending.extend(found)
        # The last user query, this call's new tail included (whole messages: no
        # header straddles the end).
        last = state.get("last_query", -1)
        for i in range(state.get("query_scan", call.protected), len(call.prompt)):
            if self._is_query(call.prompt, i):
                last = i
        state["query_scan"], state["last_query"] = len(call.prompt), last
        dying = [span for span in pending if span[1] <= last]
        state["pending"] = [span for span in pending if span[1] > last]
        return dying
