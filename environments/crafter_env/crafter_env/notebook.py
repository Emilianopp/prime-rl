"""A notebook the model writes with text tools in its reply: a short list of
memories and a scratchpad, shown back at the top of every observation.

Game-agnostic: any turn-based env can keep one per episode. Under KV-Streams
eviction old turns drop out of the model's context; whatever it wrote here
comes back every turn (and survives a respawn), so it is the model's own
long-term memory.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

_REMEMBER = re.compile(r"<remember>(.*?)</remember>", re.DOTALL)
_FORGET = re.compile(r"<forget>\s*(\d+)\s*</forget>")
_SCRATCHPAD = re.compile(r"<scratchpad>(.*?)</scratchpad>", re.DOTALL)


def instructions(slots: int, memory_chars: int, scratchpad_chars: int) -> str:
    """The system-prompt section that explains the notebook tools."""
    return f"""\
Notebook tools (optional; write them in the same reply as your action, they
cost no game step):
- <remember>note</remember> saves a note of at most {memory_chars} characters to
  your memories, e.g. <remember>table at (30, 28)</remember> or
  <remember>water 5 steps east of spawn</remember>. Write one when you place a
  table or furnace, find water, stone, coal, iron or diamond, or learn what
  killed you. You keep at most {slots}; when full, the oldest is dropped.
- <forget>N</forget> deletes memory number N.
- <scratchpad>text</scratchpad> replaces your scratchpad (at most
  {scratchpad_chars} characters): your current goal and plan, e.g.
  <scratchpad>goal: wood pickaxe. need 1 more wood, then place table</scratchpad>.
Your memories and scratchpad are shown at the top of every observation. Older
turns may drop out of your context, so write down anything you will need later."""


def _clean(text: str) -> str:
    return " ".join(text.split())


@dataclass
class Notebook:
    slots: int = 10
    memory_chars: int = 120
    scratchpad_chars: int = 400
    memories: list[str] = field(default_factory=list)
    scratchpad: str = ""
    writes: int = 0
    forgets: int = 0
    scratchpad_updates: int = 0

    def apply(self, reply: str) -> list[str]:
        """Run the notebook tools in `reply`, in order: forgets, remembers, then the
        last scratchpad. Returns one feedback line per tool call."""
        feedback = []
        for number in sorted({int(n) for n in _FORGET.findall(reply)}, reverse=True):
            if 1 <= number <= len(self.memories):
                del self.memories[number - 1]
                self.forgets += 1
                feedback.append(f"Forgot memory {number}.")
            else:
                feedback.append(f"No memory {number} to forget.")
        for raw in _REMEMBER.findall(reply):
            note = _clean(raw)
            if not note:
                continue
            cut = len(note) > self.memory_chars
            note = note[: self.memory_chars]
            if len(self.memories) >= self.slots:
                dropped = self.memories.pop(0)
                feedback.append(f"Memory full: dropped the oldest ({dropped}).")
            self.memories.append(note)
            self.writes += 1
            feedback.append(f"Saved memory {len(self.memories)}" + (f" (cut to {self.memory_chars} characters)." if cut else "."))
        pads = _SCRATCHPAD.findall(reply)
        if pads:
            text = _clean(pads[-1])
            cut = len(text) > self.scratchpad_chars
            self.scratchpad = text[: self.scratchpad_chars]
            self.scratchpad_updates += 1
            feedback.append("Scratchpad updated" + (f" (cut to {self.scratchpad_chars} characters)." if cut else "."))
        return feedback

    def render(self) -> str:
        lines = [f"Memories ({len(self.memories)}/{self.slots}):"]
        lines += [f"  {i}. {m}" for i, m in enumerate(self.memories, 1)] or ["  (none)"]
        lines.append(f"Scratchpad: {self.scratchpad or '(empty)'}")
        return "\n".join(lines)

    def summary(self) -> dict:
        return {
            "memories": list(self.memories),
            "scratchpad": self.scratchpad,
            "writes": self.writes,
            "forgets": self.forgets,
            "scratchpad_updates": self.scratchpad_updates,
        }
