"""env.recipes: the RECIPES block in the system prompt and the reason for a failed build."""
from craftax_env import glyph
from craftax_env.taskset import _BUILDS


def _prompt(**kw):
    return glyph.system_prompt(True, 1024, "freeze", True, 1, action_example=True, **kw)


def test_off_keeps_the_prompt():
    assert _prompt() == _prompt(recipes=False)
    assert "RECIPES" not in _prompt()


def test_block_states_the_rules_the_code_uses():
    text = _prompt(recipes=True)
    assert glyph.RECIPES_PROMPT in text
    assert text.index(glyph.RECIPES_PROMPT) < text.index(glyph.OBSERVATION_CONVENTIONS)
    cost, stations, _ = glyph.RECIPES["MAKE_IRON_PICKAXE"]
    assert cost == {"wood": 1, "stone": 1, "iron": 1, "coal": 1} and stations == ("table", "furnace")
    assert glyph.PLACEMENTS["PLACE_TABLE"] == {"wood": 2} and glyph.PLACEMENTS["PLACE_FURNACE"] == {"stone": 1}
    assert "MAKE_IRON_PICKAXE" in _BUILDS and "PLACE_FURNACE" in _BUILDS and "DO" not in _BUILDS


def test_failed_build_reason():
    game = glyph.CraftaxGame(seed=3)
    index = glyph.ACTION_NAMES.index("MAKE_IRON_PICKAXE")
    before = game.snapshot()
    game.step(index)
    text = game.feedback(index, before, game.snapshot())
    assert ": nothing made: needs" in text and "a furnace within one step" in text and "1 iron (you have 0)" in text
    index = glyph.ACTION_NAMES.index("PLACE_TABLE")
    before = game.snapshot()
    game.step(index)
    assert ": nothing placed: needs 2 wood (you have 0)" in game.feedback(index, before, game.snapshot())
