"""Per-turn format and memory rewards: what each reply earns from its text and the env's
answer in the next observation."""

import verifiers.v1 as vf
from verifiers.v1.graph import MessageNode
from verifiers.v1.types import AssistantMessage, SystemMessage, UserMessage

from craftax_env.turn_rewards import index_keys, turn_rewards

OBS = "Now reason as much as useful, then end with 1 to 4 <action> tags."


def trace(*exchanges: tuple[str, str, str]) -> vf.Trace:
    """(observation, reasoning, reply) per turn."""
    messages = [SystemMessage(content="RULES")]
    for obs, reasoning, reply in exchanges:
        messages += [UserMessage(content=obs), AssistantMessage(content=reply, reasoning_content=reasoning)]
    nodes = []
    for i, m in enumerate(messages):
        sampled = isinstance(m, AssistantMessage)
        nodes.append(MessageNode(parent=i - 1 if i else None, message=m, sampled=sampled, token_ids=[1, 2], mask=[sampled] * 2))
    return vf.Trace(
        agent=vf.AgentInfo(config=vf.AgentConfig()),
        task=vf.TraceTask(type="Task", data=vf.TaskData(idx=0, prompt=None)),
        nodes=nodes,
    )


def rewards(t: vf.Trace, **overrides) -> list[tuple[float, list[str]]]:
    return [(s.reward, s.reasons) for s in turn_rewards(t, max_actions=4, **overrides)]


def test_forfeits_and_where_the_tag_was():
    t = trace(
        (f"[Memory] no entries yet: write it down\n\n{OBS}", "", "<action>DO</action>"),
        (OBS, "I will <action>DO</action>", "thinking out loud"),
        (OBS, "", "no tag at all"),
        (OBS, "", "<action>FLY</action>"),
    )
    assert rewards(t) == [(0.0, []), (-1.0, ["action_in_reasoning"]), (-1.0, ["no_action_tag"]), (-1.0, ["unknown_name"])]


def test_memory_results_in_the_next_observation():
    t = trace(
        ("[Memory] TOOLS (10 tok)\n\n" + OBS, "", '<read key="TOOLS"/><action>DO</action>'),
        ('[Memory] TOOLS (10 tok)\n[Memory "TOOLS"]\nwood pickaxe first\n[/Memory "TOOLS"]\n\n' + OBS, "",
         '<write key="PLAN">mine</write><read key="NOPE"/><action>DO</action>'),
        ('[Memory] TOOLS (10 tok) · PLAN (1 tok)\nwrite "PLAN": saved (1 tok)\nread "NOPE": no such entry\n\n' + OBS, "",
         '<write key="PLAN">more</write><action>DO</action>'),
        ('[Memory] TOOLS (10 tok) · PLAN (2 tok)\nwrite "PLAN": saved (2 tok)\nmemory tags inside your reasoning were ignored: put them in your reply\n\n' + OBS, "", "<action>DO</action>"),
    )
    assert rewards(t) == [
        (1.0, ["read_after_reset"]),  # the segment opened with an entry and read it on turn 1
        (0.0, ["read_missing", "first_write"]),  # -0.5 + 0.5
        (-0.5, ["memory_in_reasoning"]),  # the second write earns nothing
        (0.0, []),  # last reply: no next observation
    ]


def test_no_read_bonus_without_memory_or_after_the_window():
    empty = trace(
        ("[Memory] no entries yet: x\n\n" + OBS, "", "<action>DO</action>"),
        ('[Memory] no entries yet: x\n[Memory "A"]\nx\n[/Memory "A"]\n\n' + OBS, "", "<action>DO</action>"),
    )
    assert [r for r, _ in rewards(empty)] == [0.0, 0.0]
    late = trace(
        ("[Memory] A (1 tok)\n\n" + OBS, "", "<action>DO</action>"),
        ("[Memory] A (1 tok)\n\n" + OBS, "", "<action>DO</action>"),
        ('[Memory] A (1 tok)\n[Memory "A"]\nx\n[/Memory "A"]\n\n' + OBS, "", "<action>DO</action>"),
    )
    assert [r for r, _ in rewards(late, read_window=1)] == [0.0, 0.0, 0.0]


def test_key_spam_counts_entries_past_the_cap():
    def index(n):
        return "[Memory] " + " · ".join(f"K{i} (1 tok)" for i in range(n)) if n else "[Memory] no entries yet: x"

    t = trace(
        (f"{index(0)}\n\n{OBS}", "", "<action>DO</action>"),
        (f"{index(4)}\n\n{OBS}", "", "<action>DO</action>"),
        (f"{index(7)}\n\n{OBS}", "", "<action>DO</action>"),
    )
    assert [r for r, _ in rewards(t, max_new_keys=5)] == [0.0, -1.0, 0.0]
    assert index_keys(index(2)) == {"K0", "K1"} and index_keys(OBS) is None


def test_summary_turns():
    ask = OBS + "\n\nYour context is about to be reset, and this reply is not a game turn."
    t = trace(
        (OBS, "", "<action>DO</action>"),
        (ask, "<summary>only in my head</summary>", "<action>DO</action>"),
    )
    assert rewards(t) == [(0.0, []), (-1.0, ["missing_summary"])]
    ok = trace((ask, "", "<summary>STATUS: alive</summary>"))
    assert rewards(ok) == [(0.0, [])]
