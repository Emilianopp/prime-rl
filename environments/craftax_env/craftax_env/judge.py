"""An LLM judge of how the player behaves in a Craftax segment, added to the reward.

The game reward only counts achievements; it says nothing about *how* the player got
there (or failed to). This judge reads one segment — every turn's HUD, feedback message,
map, the player's reasoning (cut) and reply — against the game's rules (the segment's
system prompt, handoff summary included) and scores the behaviour 0..10: learning from
the feedback messages, not looping, beliefs that match the rules, survival, and memory
use when the player has a memory. Plug it into a run:

    [[env.taskset.task.judges]]
    id = "craftax_env.judge"
    weight = 0.02          # the game reward is ~0.01-0.05 per game: keep the judge comparable
    on_error = "skip"      # a judge outage must not fail an hour-long game

It calls Gemini through its OpenAI-compatible endpoint, the key read from GEMINI_API_KEY.
Each segment is its own trace, so a segmented game is judged once per segment.
"""

import re

import verifiers.v1 as vf
from verifiers.v1.types import AssistantMessage, SystemMessage, UserMessage, content_text

CRITERIA = """\
Judge the player's behaviour in this stretch of play, using the game's rules above as the
ground truth. Weigh, roughly in this order:
1. Learning from feedback: each observation's [Message] says what the last actions did and
   why an action failed (e.g. "stone needs a wood pickaxe", "nothing collected from grass").
   Does the player change its approach after a failure, or repeat it?
2. Not looping: long runs of the same failing action, of walking back and forth, or of
   re-planning without acting are bad.
3. Grounded beliefs: claims in its reasoning, memory entries and summary should match the
   rules and what it observed. Acting on a belief the rules or the feedback contradict
   (e.g. "trees are decorative", "saplings must grow before trees give wood") is bad.
4. Progress and survival: working toward the next tool or achievement, and eating,
   drinking and sleeping before those run out.
5. Memory (only if the player has one; its edits are tags in the reply): saving correct,
   reusable facts and reading them back after a context reset is good; saving wrong rules,
   stale positions, or one new entry per failed attempt is bad.
Judge the behaviour, not luck: a player that acts sensibly but finds no resources should
still score well."""

TURN = re.compile(r"\[Current Observation — turn (\d+)\]")
DROP = re.compile(r"^\[Legend\]\n(?:.+\n)*?\n|^\[Potions\].*\n|^XP:.*\n|^Step:.*\n", re.M)
INSTRUCTION = re.compile(r"\n\nNow reason as much as useful.*", re.S)


class CraftaxJudgeConfig(vf.TraceJudgeConfig):
    id: vf.ID = "craftax_env.judge"
    name: str = "craftax_behaviour"
    model: str = "gemini-3.8-flash"
    base_url: str = "https://generativelanguage.googleapis.com/v1beta/openai/"
    api_key_var: str = "GEMINI_API_KEY"
    criteria: str = CRITERIA
    reasoning_chars: int = 800
    """Each reply's reasoning is cut to its first this many characters (0: none)."""
    grid: bool = True
    """Show each observation's map grid."""


class CraftaxJudge(vf.TraceJudge[CraftaxJudgeConfig]):
    def observation(self, text: str) -> str:
        """A turn's observation without its legend, rarely-useful HUD lines, the repeated
        reply instruction and (with `grid = false`) the map."""
        text = INSTRUCTION.sub("", DROP.sub("", text))
        if not self.config.grid:
            text = re.sub(r"\[Grid\]\n.*?(?=\[HUD\])", "", text, flags=re.S)
        return text.strip()

    def render(self, task: vf.TaskData, trace: vf.Trace) -> str:
        branches = trace.branches
        blocks: list[str] = []
        for message in branches[-1].messages if branches else []:
            if isinstance(message, SystemMessage):
                continue
            if isinstance(message, UserMessage):
                text = content_text(message.content)
                blocks.append(f"=== GAME{' (turn ' + m.group(1) + ')' if (m := TURN.search(text)) else ''}\n{self.observation(text)}")
            elif isinstance(message, AssistantMessage):
                lines = ["=== PLAYER"]
                if self.config.reasoning and self.config.reasoning_chars and message.reasoning_content:
                    thought = message.reasoning_content.strip()
                    cut = len(thought) - self.config.reasoning_chars
                    if cut > 0:
                        thought = f"{thought[: self.config.reasoning_chars]} [... {cut:,} more characters]"
                    lines.append(f"(reasoning) {thought}")
                lines.append(f"(reply) {(message.content or '').strip() or '[empty reply]'}")
                blocks.append("\n".join(lines))
        return "\n\n".join(blocks)


__all__ = ["CraftaxJudge"]
