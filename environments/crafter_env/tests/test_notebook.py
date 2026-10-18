import asyncio
from contextlib import asynccontextmanager
from types import SimpleNamespace

from crafter_env.notebook import Notebook
from crafter_env.taskset import CrafterConfig, CrafterEnv, CrafterEnvConfig, CrafterTaskset


def test_remember_forget_and_scratchpad():
    book = Notebook(slots=2, memory_chars=10, scratchpad_chars=8)
    assert book.apply("<remember>table at  (3, 4)</remember><action>noop</action>") == [
        "Saved memory 1 (cut to 10 characters)."
    ]
    book.apply("<remember>water east</remember><remember>cows north</remember>")
    assert book.memories == ["water east", "cows north"]  # the oldest dropped when full
    assert book.apply("<forget>1</forget><forget>9</forget>") == ["No memory 9 to forget.", "Forgot memory 1."]
    assert book.apply("<scratchpad>first</scratchpad><scratchpad>get 2 wood then table</scratchpad>") == [
        "Scratchpad updated (cut to 8 characters)."
    ]
    assert book.memories == ["cows north"] and book.scratchpad == "get 2 wo"
    assert book.render() == "Memories (1/2):\n  1. cows north\nScratchpad: get 2 wo"
    assert book.summary()["writes"] == 3 and book.summary()["forgets"] == 1


def test_notebook_is_explained_and_shown_every_turn():
    task = next(iter(CrafterTaskset(CrafterConfig(num_examples=1, notebook=True)).load()))
    assert "<remember>fact</remember>" in task.data.system_prompt
    replies = iter(
        ["<remember>spawn has water to the east</remember><action>do</action>"]
        + ["<scratchpad>find trees</scratchpad><action>move_left</action>"]
        + ["<action>noop</action>"] * 10
    )
    received = []
    trace = SimpleNamespace(
        rewards={}, metrics={}, info={}, num_total_tokens=0,
        record_reward=lambda name, value, weight: trace.rewards.update({name: value}),
        record_metrics=lambda metrics: trace.metrics.update(metrics),
    )

    class Interaction:
        def __init__(self):
            self.trace = trace

        async def turn(self, message=None):
            received.append(message)
            return SimpleNamespace(last_reply=next(replies), terminated=False)

    @asynccontextmanager
    async def open_interaction(_task):
        yield Interaction()

    agents = SimpleNamespace(player=SimpleNamespace(interaction=open_interaction))
    asyncio.run(CrafterEnv.run(SimpleNamespace(config=CrafterEnvConfig(max_episode_steps=6)), task, agents))
    assert received[0].startswith("Memories (0/10):\n  (none)\nScratchpad: (empty)")
    assert "Saved memory 1." in received[1] and "1. spawn has water to the east" in received[1]
    assert "Scratchpad updated." in received[2] and "Scratchpad: find trees" in received[2]
    assert all("1. spawn has water to the east" in m for m in received[1:])  # every later turn
    assert trace.info["crafter"]["notebook"]["memories"] == ["spawn has water to the east"]
    assert trace.metrics["memory_writes"] == 1.0
