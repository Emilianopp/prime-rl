"""The Craftax behaviour judge: what it shows the judge model, and plugging it by id."""

import verifiers.v1 as vf
from verifiers.v1.graph import MessageNode
from verifiers.v1.types import AssistantMessage, SystemMessage, UserMessage

from craftax_env.judge import CraftaxJudge, CraftaxJudgeConfig

OBSERVATION = """[Memory] no entries yet

[Legend]
c — cow
♣ — tree

[Current Observation — turn 7]
[Grid]
·♣·
·↓·
[HUD]
Step: 7    Upstream timestep: 7
HP: 9/9    Food: 4/9    Drink: 7/9    Energy: 5/9    Mana: 9/9
XP: 0    Str: 1    Dex: 1    Int: 1
Floor: 0 (Overworld)    Pos: (22,22)    Facing: down    Light: Day
[Inventory] Wood:1
[Potions] Red:0
[Message]
Played 1 action, reward +0.000. 1. DO: nothing collected from grass (it drops a sapling 10% of the time).

Now reason as much as useful, then end with 1 to 4 <action> tags."""


def trace() -> vf.Trace:
    messages = [
        SystemMessage(content="RULES: PLACE_TABLE needs wood."),
        UserMessage(content=OBSERVATION),
        AssistantMessage(content="<action>DO</action>", reasoning_content="x" * 50),
    ]
    return vf.Trace(
        agent=vf.AgentInfo(config=vf.AgentConfig()),
        task=vf.TraceTask(type="Task", data=vf.TaskData(idx=0, prompt=None)),
        nodes=[MessageNode(parent=i - 1 if i else None, message=m, sampled=isinstance(m, AssistantMessage)) for i, m in enumerate(messages)],
    )


def test_the_render_keeps_what_matters():
    t = trace()
    judge = CraftaxJudge(CraftaxJudgeConfig(reasoning_chars=10))
    text = judge.render(t.task.data, t)
    assert "=== GAME (turn 7)" in text and "nothing collected from grass" in text and "·♣·" in text
    assert "[Legend]" not in text and "Now reason" not in text and "[Potions]" not in text and "XP:" not in text
    assert "(reasoning) xxxxxxxxxx [... 40 more characters]" in text and "(reply) <action>DO</action>" in text
    assert "·♣·" not in CraftaxJudge(CraftaxJudgeConfig(grid=False)).render(t.task.data, t)
    assert judge.fields(t.task.data, t)["task"] == "RULES: PLACE_TABLE needs wood."


def test_plugged_by_id_with_gemini_defaults():
    config = vf.TaskConfig.model_validate({"judges": [{"id": "craftax_env.judge", "weight": 0.02}]})
    judge = config.judges[0]
    assert isinstance(judge, CraftaxJudgeConfig) and judge.api_key_var == "GEMINI_API_KEY"
    assert judge.weight == 0.02 and CraftaxJudge(judge).reward_name == "craftax_behaviour"
