"""forfeit_penalty: the env names every reply with no valid action in trace.info["node_penalties"]."""

import asyncio
import json
from contextlib import asynccontextmanager
from types import SimpleNamespace

from craftax_env.taskset import CraftaxConfig, CraftaxEnv, CraftaxEnvConfig, CraftaxTaskset, request_node_penalties

# turn 1 no tag (forfeit), 2 plays, 3 unknown name (forfeit), 4 plays, 5 no tag (forfeit), 6 plays
REPLIES = ["no tags", "<action>DO</action>", "<action>JUMP</action>", "<action>MOVE_LEFT</action>"]


class Scripted:
    """Each turn appends an observation node and a sampled reply node, as a real trace does."""

    def __init__(self):
        self.received = []
        self.trace = SimpleNamespace(
            rewards={}, metrics={}, info={}, num_total_tokens=0, nodes=[],
            record_reward=lambda name, value, weight: self.trace.rewards.update({name: value}),
            record_metrics=lambda metrics: self.trace.metrics.update(metrics),
        )

    async def turn(self, message=None):
        self.received.append(message)
        reply = REPLIES[(len(self.received) - 1) % len(REPLIES)]
        self.trace.nodes.append(SimpleNamespace(sampled=False, text=message))
        self.trace.nodes.append(SimpleNamespace(sampled=True, text=reply))
        return SimpleNamespace(last_reply=reply, terminated=False)


def run(tmp_path, **env_config):
    task = next(iter(CraftaxTaskset(CraftaxConfig(num_examples=1, max_output_tokens=2048)).load()))
    interaction = Scripted()

    @asynccontextmanager
    async def open_interaction(_task):
        yield interaction

    log = tmp_path / "episodes.jsonl"
    env = SimpleNamespace(config=CraftaxEnvConfig(episode_log=str(log), max_episode_steps=6, **env_config))
    for name, member in vars(CraftaxEnv).items():
        if callable(member) and not name.startswith("__"):
            setattr(env, name, member.__get__(env))
    asyncio.run(CraftaxEnv.run(env, task, SimpleNamespace(player=SimpleNamespace(interaction=open_interaction))))
    return interaction, json.loads(log.read_text())


def test_forfeit_penalty_names_the_failed_replies(tmp_path):
    interaction, line = run(tmp_path, forfeit_penalty=1.0)
    trace = interaction.trace
    pairs = trace.info["node_penalties"]
    failed = sum(line["forfeits"].values())
    assert failed >= 2 and len(pairs) == failed == trace.metrics["forfeit_penalized"]
    for index, penalty in pairs:
        node = trace.nodes[index]
        assert node.sampled and penalty == 1.0
        assert "<action>DO</action>" not in node.text and "<action>MOVE_LEFT</action>" not in node.text


def test_forfeit_penalty_with_retry_names_failed_retries_too(tmp_path):
    interaction, line = run(tmp_path, forfeit_penalty=0.5, action_retry=1)
    pairs = interaction.trace.info["node_penalties"]
    texts = [interaction.trace.nodes[i].text for i, _ in pairs]
    # first replies that failed (rescued or not) plus retries that failed
    assert len(pairs) == sum(line["first_try_forfeits"].values()) + (line["action_retries"] - line["action_retry_rescued"])
    assert all(t in ("no tags", "<action>JUMP</action>") for t in texts)


def test_forfeit_penalty_off_by_default(tmp_path):
    interaction, line = run(tmp_path)
    assert "node_penalties" not in interaction.trace.info
    assert interaction.trace.metrics["forfeit_penalized"] == 0.0


def test_request_node_penalties_maps_ordinals_to_node_indexes():
    nodes = [SimpleNamespace(sampled=s) for s in (False, True, False, True, False, True)]
    trace = SimpleNamespace(info={}, nodes=nodes)
    assert request_node_penalties(trace, [1, 3], 0.7) == 2
    assert trace.info["node_penalties"] == [[1, 0.7], [5, 0.7]]
