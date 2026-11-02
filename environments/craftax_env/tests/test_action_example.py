"""action_example: the exact action-tag format, with a real Craftax action, in the system
prompt and every turn's closing request (opt-in; off keeps GlyphBench's text)."""

import asyncio

from craftax_env import glyph
from craftax_env.taskset import CraftaxConfig, CraftaxTaskset

from test_segments import SegmentPlayer, make


def test_default_keeps_glyphbench_text():
    sp = glyph.system_prompt(True, 1024, "freeze", respawn=True)
    assert "<action>MOVE_FORWARD</action>" in sp and "Only the tag counts" not in sp
    game = glyph.CraftaxGame(1000042, None)
    assert glyph.user_turn(game, True).endswith(
        "Now reason as much as useful, then end with a final <action> tag containing "
        "exactly one action name from the system prompt's action list.")


def test_system_prompt_shows_a_craftax_action():
    sp = glyph.system_prompt(True, 1024, "freeze", respawn=True, action_example=True)
    assert "MOVE_FORWARD" not in sp and "MOVE_LEFT" in glyph.ACTION_NAMES
    assert "the last line of your reply is:\n  <action>MOVE_LEFT</action>\n" in sp
    assert "`Action: MOVE_LEFT`" in sp
    # the rest of the format block is unchanged
    base = glyph.system_prompt(True, 1024, "freeze", respawn=True)
    assert sp.split("RESPONSE FORMAT")[0] == base.split("RESPONSE FORMAT")[0]
    assert sp.split("Failure modes:")[1] == base.split("Failure modes:")[1]


def test_closing_request_names_the_format():
    game = glyph.CraftaxGame(1000042, None)
    turn = glyph.user_turn(game, True, action_example=True)
    assert turn.endswith("in this exact format:\n<action>MOVE_LEFT</action>")
    assert "Now reason as much as useful" in turn
    multi = glyph.user_turn(game, True, max_actions=3, action_example=True)
    assert multi.endswith("<action>MOVE_LEFT</action>\n<action>DO</action>")
    assert glyph.user_turn(game, True, closing=False, action_example=True) == glyph.user_turn(game, True, closing=False)
    for reply in ("<action>MOVE_LEFT</action>", "reasoning...\n<action>MOVE_LEFT</action>"):
        assert glyph.parse_actions(reply, 1)[0] == [glyph.ACTION_NAMES.index("MOVE_LEFT")]


def test_taskset_and_env_pass_it_through(tmp_path):
    task = next(iter(CraftaxTaskset(CraftaxConfig(num_examples=1, max_output_tokens=1024, respawn=True,
                                                  action_example=True)).load()))
    assert task.data.info["action_example"] is True and "Only the tag counts" in task.data.system_prompt
    _, env = make(tmp_path, segment_turns=3)
    player = SegmentPlayer()
    asyncio.run(env.play_segment(task, player))
    actions = [m for m in player.received if not m.startswith("END OF SEGMENT")]
    assert actions and all(m.endswith("in this exact format:\n<action>MOVE_LEFT</action>") for m in actions)
