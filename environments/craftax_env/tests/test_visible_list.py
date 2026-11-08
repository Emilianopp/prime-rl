"""visible_list: the [Visible] block (glyph.visible) and its system-prompt note (taskset.visible_list)."""

import re

import numpy as np

from craftax_env import glyph, world


def test_visible_places_match_the_true_map():
    game = glyph.CraftaxGame(42, 1000)
    legend, grid, _hud, _msg = game.render()
    text = glyph.visible(legend, grid, game.floor, game.position)
    tiles = np.asarray(game.state.map[game.floor])
    names = game._rt["block_names"]
    checked = 0
    for part in text.split("; "):
        kind = part.split(" F")[0].lower()
        for f, r, c in re.findall(r"F(\d+) \((\d+),(\d+)\)", part):
            assert int(f) == game.floor
            real = names.get(int(tiles[int(r), int(c)]), "?")
            assert kind.split()[0] in real or real.split()[0] in kind, (part, real)
            checked += 1
    assert checked > 0
    # every place carries its floor, so the notes parser never reads one as a view-frame cell
    assert not re.search(r"(?<!F\d )\(\d+,\d+\)", text.replace(", F", " F"))
    places = world.parse_places(text, game.floor, view=(game.floor, *game.position))
    assert len(places) == checked


def test_visible_leaves_out_ground_and_player():
    legend = "↑ — you, facing up\n· — grass or path\n░ — sand\n♣ — tree"
    grid = "·░·\n·↑♣\n···"
    assert glyph.visible(legend, grid, 0, (10, 10)) == "tree F0 (10,11)"
    assert glyph.visible(legend, "···\n·↑·\n░░░", 0, (10, 10)) == "nothing but ground"


def test_visible_list_off_is_glyphbench():
    game = glyph.CraftaxGame(42, 1000)
    assert "[Visible]" not in glyph.user_turn(game, True)
    on = glyph.user_turn(game, True, visible_list=True)
    assert on.index("[Grid]") < on.index("[Visible]") < on.index("[HUD]")
    assert "[Visible]" in glyph.user_turn(game, True, closing=False, visible_list=True)
    base = glyph.system_prompt(True, 2048, "freeze")
    assert "[Visible]" not in base
    assert glyph.VISIBLE_CONVENTION in glyph.system_prompt(True, 2048, "freeze", visible_list=True)


def test_taskset_passes_visible_list():
    from craftax_env.taskset import CraftaxConfig, CraftaxTaskset

    task = next(iter(CraftaxTaskset(CraftaxConfig(num_examples=1, visible_list=True)).load()))
    assert task.data.info["visible_list"] is True
    assert glyph.VISIBLE_CONVENTION in task.data.system_prompt
