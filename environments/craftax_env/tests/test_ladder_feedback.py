"""Action feedback says why DESCEND or ASCEND left the player on the same floor (upstream main 5905f25c9)."""
from craftax_env import glyph


def _try(game, name):
    index = glyph.ACTION_NAMES.index(name)
    before = game.snapshot()
    game.step(index)
    return game.feedback(index, before, game.snapshot())


def test_overworld_start_names_the_reason():
    game = glyph.CraftaxGame(seed=3)
    assert game.floor == 0
    assert _try(game, "ASCEND") == "ASCEND: no effect: there is no floor above the Overworld."
    text = _try(game, "DESCEND")
    assert text.startswith("DESCEND: no effect: you are not on a ladder down (▼)")


def test_reasons_by_the_cell_under_the_player():
    game = glyph.CraftaxGame(seed=3)
    rt = game._rt
    down = next(iter(rt["ladder_down"]))
    closed = game._ladder_why("DESCEND", {"under": down, "kills": rt["kills_required"] - 2, "floor": 1})
    assert "closed: kill 2 more monsters" in closed
    assert "only goes up (ASCEND)" in game._ladder_why("DESCEND", {"under": rt["ladder_up"], "kills": 0, "floor": 1})
    assert "only goes down (DESCEND)" in game._ladder_why("ASCEND", {"under": down, "kills": 0, "floor": 1})
    assert "not on a ladder up" in game._ladder_why("ASCEND", {"under": -1, "kills": 0, "floor": 1})
