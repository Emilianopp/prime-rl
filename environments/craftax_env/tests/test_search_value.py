"""world.search_value: what selection = "puct" ranks saved games by."""
import numpy as np

from craftax_env import glyph, world


def _with(game, **items):
    inv = game.state.inventory
    game.state = game.state.replace(inventory=inv.replace(**{k: getattr(inv, k) * 0 + v for k, v in items.items()}))


def test_search_value_orders_states():
    game = glyph.CraftaxGame(seed=3)
    explored = np.zeros(np.asarray(game.state.map).shape, dtype=bool)
    fresh = world.search_value(game, explored, game.achievement_score)
    assert 0 < fresh < 1.0  # no achievements, no tools: only the nearness and health terms
    game.achievements = {"collect_wood"}  # next ladder step: place_table (2 wood)
    no_wood = world.search_value(game, explored, 1.0)
    _with(game, wood=2)
    with_wood = world.search_value(game, explored, 1.0)
    assert 1.0 < no_wood < with_wood < 2.0  # materials toward the next step break ties, under one point
    _with(game, pickaxe=2)
    with_pickaxe = world.search_value(game, explored, 1.0)
    assert with_pickaxe > with_wood + 1.5  # a stone pickaxe held is worth two points: a death that drops it shows
    assert world.search_value(game, explored, 5.0) == with_pickaxe + 4.0
