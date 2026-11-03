"""What the judge knows about the world: privileged, judge-only facts (docs/craftax-segments-judge.md).

- ``ascii_map``: a whole floor, one letter per cell (upper case: a cell the agent saw;
  lower case: never seen), ladders ``>``/``<``, the player ``@``, with row and col index
  headers that are the same numbers as ``glyph.place``.
- ``nearest``: the nearest cell of each feature on the current floor, seen or not.
- ``next_step``: the first unmet step of the tech ladder and what it still needs.
- ``check_map_claims``: the policy's MAP lines checked against the real map.

Every place the env writes uses ``glyph.place`` (``F<floor> (row,col)``), as the record and
the timeline do. The notes write places in the view frame instead: (row,col) in the 9x11
[Grid] of the observation the notes turn shows, the player at ``VIEW_CENTER``.
``parse_places(..., view=...)`` turns those into map places.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from craftax_env.glyph import PLACEMENTS, RECIPES, place

# Craftax BlockType names -> one map letter. Several blocks share a letter (realm grass
# folds into grass, realm trees into trees); `_FEATURES` below names what the judge looks up.
LETTERS = {
    "invalid": "?", "out of bounds": "#", "grass": "G", "water": "W", "stone": "S", "tree": "T",
    "wood": "T", "path": "P", "coal": "C", "iron": "I", "diamond": "D", "crafting table": "B",
    "furnace": "F", "sand": "A", "lava": "L", "plant": "Q", "ripe plant": "Q", "wall": "X",
    "darkness": "K", "wall moss": "X", "stalagmite": "Y", "sapphire": "Z", "ruby": "R",
    "chest": "H", "fountain": "O", "fire grass": "G", "ice grass": "G", "gravel": "V",
    "fire tree": "T", "ice shrub": "T", "enchantment table fire": "E", "enchantment table ice": "E",
    "necromancer": "N", "grave": "M", "grave2": "M", "grave3": "M", "necromancer vulnerable": "N",
}
LEGEND = (
    "G grass, W water, S stone, T tree, P path, C coal, I iron, D diamond, Z sapphire, R ruby, "
    "A sand, L lava, Q plant, X wall, Y stalagmite, K darkness, V gravel, B crafting table, "
    "F furnace, E enchantment table, H chest, O fountain, N necromancer, M grave, "
    "> ladder down, < ladder up, @ the player, # map edge"
)

# Feature -> the block names (or ladder items) that count as it, for `nearest` and the claims.
_FEATURES: dict[str, tuple[str, ...]] = {
    "water": ("water",), "tree": ("tree", "fire tree", "ice shrub", "wood"), "stone": ("stone",),
    "coal": ("coal",), "iron": ("iron",), "diamond": ("diamond",), "sapphire": ("sapphire",), "ruby": ("ruby",),
    "lava": ("lava",), "table": ("crafting table",), "furnace": ("furnace",), "chest": ("chest",),
    "fountain": ("fountain",), "enchantment table": ("enchantment table fire", "enchantment table ice"),
    "plant": ("plant", "ripe plant"), "sand": ("sand",), "path": ("path",), "grass": ("grass", "fire grass", "ice grass"),
    "ladder down": ("@ladder_down",), "ladder up": ("@ladder_up",), "ladder": ("@ladder_down", "@ladder_up"),
}
_NEAREST_ORDER = ("water", "tree", "stone", "coal", "iron", "diamond", "sapphire", "ruby", "table", "furnace",
                  "ladder down", "ladder up", "chest", "fountain", "enchantment table", "lava")
# Words in a MAP line -> the feature it claims (first match wins: specific before generic).
_SYNONYMS: tuple[tuple[tuple[str, ...], str], ...] = (
    (("ladder down", "down ladder", "downward ladder", "descend", "ladder (down)", "ladder_down"), "ladder down"),
    (("ladder up", "up ladder", "upward ladder", "ascend", "ladder (up)", "ladder_up"), "ladder up"),
    (("ladder",), "ladder"),
    (("enchant",), "enchantment table"),
    (("crafting table", "table", "workbench"), "table"),
    (("furnace",), "furnace"),
    (("diamond",), "diamond"), (("sapphire",), "sapphire"), (("ruby",), "ruby"),
    (("coal",), "coal"), (("iron",), "iron"),
    (("lava",), "lava"), (("chest",), "chest"), (("fountain",), "fountain"),
    (("water", "lake", "river", "pond", "drink"), "water"),
    (("tree", "wood", "forest"), "tree"),
    (("stone", "rock", "mountain", "cave wall"), "stone"),
    (("sapling", "plant"), "plant"), (("sand", "beach"), "sand"), (("path", "tunnel"), "path"), (("grass",), "grass"),
)
_CREATURES = ("zombie", "cow", "skeleton", "enemy", "enemies", "mob", "monster", "creature", "gnome", "orc",
              "lizard", "kobold", "troll", "knight", "archer", "pigman", "elemental", "bat", "snail", "spider")
# F0 (19,29) / F0 (19, 29) / floor 0 (19,29) / (19,29) on F0 / (19,29) floor 0; a range "(24,24-25)" takes its start.
_PLACE = re.compile(r"(?:\bF\s*(\d+)|\bfloor\s*(\d+))?[^()\d]{0,12}\(\s*(\d+)\s*,\s*(\d+)(?:\s*-\s*\d+)?\s*\)(?:\s*(?:on|at|in)?\s*(?:F\s*(\d+)|floor\s*(\d+)))?", re.IGNORECASE)


VIEW_ROWS, VIEW_COLS = 9, 11
VIEW_CENTER = (4, 5)  # the player's arrow in the [Grid]


def view_to_map(view: tuple[int, int, int], r: int, c: int) -> tuple[int, int, int]:
    """A view-frame cell (r, c) seen with the player at `view` = (floor, row, col)."""
    floor, pr, pc = view
    return floor, pr + r - VIEW_CENTER[0], pc + c - VIEW_CENTER[1]


def view_note(view: tuple[int, int, int]) -> str:
    """How to read the notes' places, for the judges and the next segment."""
    return (f"places in the notes are (row,col) in the agent's 9x11 view at the end of the segment, the agent "
            f"at view (4,5) = {place(view[0], view[1:])}: view (r,c) is F{view[0]} (R,C) with "
            f"R = {view[1]} + r - 4, C = {view[2]} + c - 5")


def _rt_names(game: Any) -> dict[int, str]:
    return game._rt["block_names"]


def _feature_mask(game: Any, floor: int, feature: str) -> np.ndarray:
    """Cells of `floor` that are the feature (blocks, or ladder items)."""
    import craftax.craftax.constants as C

    names = _FEATURES[feature]
    blocks = np.asarray(game.state.map[floor])
    mask = np.zeros(blocks.shape, dtype=bool)
    block_ids = [i for i, n in _rt_names(game).items() if n in names]
    if block_ids:
        mask |= np.isin(blocks, block_ids)
    items = np.asarray(game.state.item_map[floor])
    if "@ladder_down" in names:
        mask |= np.isin(items, [C.ItemType.LADDER_DOWN.value, C.ItemType.LADDER_DOWN_BLOCKED.value])
    if "@ladder_up" in names:
        mask |= items == C.ItemType.LADDER_UP.value
    return mask


def ascii_map(game: Any, explored: np.ndarray, floor: int | None = None) -> str:
    """One floor as letters, with col index headers (tens, units) and row numbers."""
    import craftax.craftax.constants as C

    floor = game.floor if floor is None else floor
    blocks = np.asarray(game.state.map[floor])
    items = np.asarray(game.state.item_map[floor])
    seen = explored[floor]
    names = _rt_names(game)
    rows, cols = blocks.shape
    down = (C.ItemType.LADDER_DOWN.value, C.ItemType.LADDER_DOWN_BLOCKED.value)
    pr, pc = game.position if floor == game.floor else (-1, -1)
    out = ["    " + "".join(str(c // 10) for c in range(cols)), "    " + "".join(str(c % 10) for c in range(cols))]
    for r in range(rows):
        line = []
        for c in range(cols):
            if (r, c) == (pr, pc):
                ch = "@"
            elif items[r, c] in down:
                ch = ">"
            elif items[r, c] == C.ItemType.LADDER_UP.value:
                ch = "<"
            else:
                ch = LETTERS.get(names.get(int(blocks[r, c]), "invalid"), "?")
                if ch.isalpha() and not seen[r, c]:
                    ch = ch.lower()
            line.append(ch)
        out.append(f"{r:>3} " + "".join(line))
    return "\n".join(out)


def nearest(game: Any, explored: np.ndarray, radius: int = 24) -> str:
    """The nearest cell of each feature within `radius` tiles (Chebyshev) on the current floor."""
    floor = game.floor
    pr, pc = game.position
    out = []
    for feature in _NEAREST_ORDER:
        mask = _feature_mask(game, floor, feature)
        cells = np.argwhere(mask)
        if not len(cells):
            continue
        d = np.maximum(np.abs(cells[:, 0] - pr), np.abs(cells[:, 1] - pc))
        i = int(np.argmin(d))
        if d[i] > radius:
            continue
        r, c = (int(v) for v in cells[i])
        out.append(f"{feature} {place(floor, (r, c))} dist {int(d[i])} {'seen' if explored[floor][r, c] else 'unseen'}")
    return "; ".join(out) or "none within range"


# --- the tech ladder ------------------------------------------------------------------

_TOOL_FOR = {"stone": 1, "coal": 1, "iron": 2, "diamond": 3, "sapphire": 4, "ruby": 4}
_PICKAXE = {1: "a wood pickaxe", 2: "a stone pickaxe", 3: "an iron pickaxe", 4: "a diamond pickaxe"}
LADDER = (
    ("collect_wood", None), ("place_table", "PLACE_TABLE"), ("make_wood_pickaxe", "MAKE_WOOD_PICKAXE"),
    ("collect_stone", "stone"), ("make_stone_pickaxe", "MAKE_STONE_PICKAXE"), ("place_furnace", "PLACE_FURNACE"),
    ("collect_coal", "coal"), ("collect_iron", "iron"), ("make_iron_pickaxe", "MAKE_IRON_PICKAXE"),
    ("make_iron_sword", "MAKE_IRON_SWORD"), ("collect_diamond", "diamond"), ("enter_dungeon", "DESCEND"),
)


def _nearest_station(game: Any, kind: str, stations: list[dict[str, Any]]) -> str:
    floor = game.floor
    pr, pc = game.position
    mask = _feature_mask(game, floor, "table" if kind == "table" else "furnace")
    cells = np.argwhere(mask)
    if not len(cells):
        return f"no {kind} on this floor"
    d = np.maximum(np.abs(cells[:, 0] - pr), np.abs(cells[:, 1] - pc))
    i = int(np.argmin(d))
    return f"nearest {kind} {place(floor, cells[i])}, dist {int(d[i])}"


def next_step(game: Any, stations: list[dict[str, Any]] | None = None) -> str:
    """The first unmet step of the ladder (wood -> ... -> diamond -> the Dungeon -> deeper) and
    what it still needs, from the game's own recipes and the inventory."""
    import craftax.craftax.constants as C

    stations = stations or []
    have = set(game.achievements)
    snap = game.snapshot()
    items = snap["items"]
    for name, how in LADDER:
        if name in have:
            continue
        needs: list[str] = []
        if how is None:
            return f"{name} (DO facing a tree): requirements met"
        if how in RECIPES:
            cost, stations_needed, _ = RECIPES[how]
            needs += [f"{n} {k} (has {items[k]})" for k, n in cost.items() if items[k] < n]
            for s in stations_needed:
                if not snap[f"near_{s}"]:
                    needs.append(f"a {s} within one step ({_nearest_station(game, s, stations)})")
            label = f"{name} ({how})"
        elif how in PLACEMENTS:
            # Craftax places a furnace anywhere (place_block checks only the stone): no table needed.
            needs += [f"{n} {k} (has {items[k]})" for k, n in PLACEMENTS[how].items() if items[k] < n]
            label = f"{name} ({how})"
        elif how == "DESCEND":
            mask = _feature_mask(game, 0, "ladder down")
            cells = np.argwhere(mask)
            where = place(0, cells[0]) if len(cells) else "unknown"
            label = f"{name} (stand on the Overworld's down-ladder and DESCEND; it is open)"
            needs.append(f"reach the ladder at {where}")
        else:  # collect a material
            tool = _TOOL_FOR[how]
            if items["pickaxe"] < tool:
                needs.append(f"{_PICKAXE[tool]} (has pickaxe level {items['pickaxe']})")
            label = f"{name} (DO facing {how})"
        return f"{label}: " + ("needs " + ", ".join(needs) if needs else "requirements met right now")
    floor = game.floor
    killed = int(np.asarray(game.state.monsters_killed[floor]))
    if floor >= 1 and killed < C.MONSTERS_KILLED_TO_CLEAR_LEVEL:
        return f"on floor {floor}: kill monsters until the down-ladder opens ({killed}/{C.MONSTERS_KILLED_TO_CLEAR_LEVEL}), then DESCEND"
    return f"on floor {floor}: the down-ladder is open; find it and DESCEND (diamond tools and armour once diamonds allow)"


# --- the MAP claims -------------------------------------------------------------------


@dataclass
class ClaimCheck:
    correct: int = 0
    wrong: int = 0
    invented: int = 0
    unchecked: int = 0
    lines: list[str] = field(default_factory=list)

    @property
    def claims(self) -> int:
        return self.correct + self.wrong + self.invented

    @property
    def score(self) -> float:
        """map_check in [0, 1]: (correct - 0.5 wrong - 2 invented) / max(claims, 3)."""
        return float(min(1.0, max(0.0, (self.correct - 0.5 * self.wrong - 2 * self.invented) / max(self.claims, 3))))

    def text(self) -> str:
        head = (f"{self.correct} correct, {self.wrong} wrong place, {self.invented} invented, "
                f"{self.unchecked} not checkable; map_check = {self.score:.2f}")
        return "\n".join([head, *self.lines]) if self.lines else head


def claim_feature(text: str) -> str | None:
    low = text.lower()
    for words, feature in _SYNONYMS:
        if any(w in low for w in words):
            return feature
    return None


def parse_places(text: str, default_floor: int, view: tuple[int, int, int] | None = None) -> list[tuple[int, int, int]]:
    """Every place in a line, as (floor, row, col), normalized from the near-misses the
    prompt does not ask for (``F0 (19, 29)``, ``floor 0 (19,29)``, ``(19,29) on F0``).
    With `view` (floor, row, col of the player when the notes were written), a place without
    a floor tag and inside the 9x11 view is a view-frame cell and is converted; a place with
    a floor tag, or outside the view, is a map place."""
    return [p for p, _ in _places(text, default_floor, view)]


# "row 6, col 7" / "row 6 column 7" / "(row 6, col 7)": a single cell written out (ranges such as
# "rows 3-8" are left alone).
_ROWCOL = re.compile(r"(?:\bF\s*(\d+)\s*)?\brow\s+(\d+)(?!\s*-)\s*,?\s*(?:and\s+)?col(?:umn)?\s+(\d+)(?!\s*-)", re.IGNORECASE)


def _places(text: str, default_floor: int, view: tuple[int, int, int] | None) -> list[tuple[tuple[int, int, int], str | None]]:
    """(place, the view cell as written or None) for every place in a line."""
    out = []
    found = [(m.start(), next((g for g in (m.group(1), m.group(2), m.group(5), m.group(6)) if g is not None), None),
              int(m.group(3)), int(m.group(4))) for m in _PLACE.finditer(text)]
    found += [(m.start(), m.group(1), int(m.group(2)), int(m.group(3))) for m in _ROWCOL.finditer(text)]
    for _, f, r, c in sorted(found, key=lambda x: x[0]):
        if view is not None and f is None and r < VIEW_ROWS and c < VIEW_COLS:
            out.append((view_to_map(view, r, c), f"view ({r},{c})"))
        else:
            out.append(((int(f) if f is not None else default_floor, r, c), None))
    return out


def check_map_claims(game: Any, map_lines: list[str], explored: np.ndarray, default_floor: int | None = None,
                     tolerance: int = 1, near: int = 5, view: tuple[int, int, int] | None = None) -> ClaimCheck:
    """Each MAP line's (feature, place) against the cells of the real map the agent SAW:
    *correct* if a seen cell of the feature is within `tolerance` tiles (Chebyshev), *wrong* if
    one is within `near` tiles (the place is off), *invented* otherwise - including a claim
    that matches only cells the agent never saw (dense features such as trees would otherwise
    match a made-up place by chance; smoke job 11487521's notes gave 9x11-view coordinates
    that "hit" trees in rows the agent never visited). Lines naming a creature (they move), no
    known feature or no place are *not checkable*. A line with several places checks each.
    `view`: see ``parse_places``; a converted place is labelled ``view (r,c) = F0 (R,C)``."""
    default_floor = game.floor if default_floor is None else default_floor
    out = ClaimCheck()
    n_floors = int(np.asarray(game.state.map).shape[0])
    for line in map_lines:
        low = line.lower()
        places = _places(line, default_floor, view)
        feature = claim_feature(line)
        if any(w in low for w in _CREATURES) and feature in (None, "grass", "path"):
            out.unchecked += 1
            out.lines.append(f"- not checkable (a creature): {line.strip()}")
            continue
        if feature is None or not places:
            out.unchecked += 1
            why = "no place" if feature is not None else "no known feature"
            out.lines.append(f"- not checkable ({why}): {line.strip()}")
            continue
        for (floor, r, c), seen_as in places:
            label = f"{feature} {seen_as + ' = ' if seen_as else ''}{place(floor, (r, c))}"
            if not 0 <= floor < n_floors:
                out.invented += 1
                out.lines.append(f"x invented: {label} (no floor {floor})")
                continue
            real = _feature_mask(game, floor, feature)
            cells = np.argwhere(real & explored[floor])
            if not len(cells):
                out.invented += 1
                why = f"you never saw {feature} on floor {floor}" if real.any() else f"no {feature} on floor {floor}"
                out.lines.append(f"x invented: {label} ({why})")
                continue
            d = np.maximum(np.abs(cells[:, 0] - r), np.abs(cells[:, 1] - c))
            i = int(np.argmin(d))
            truth = place(floor, cells[i])  # the nearest SEEN cell of the feature
            if d[i] <= tolerance:
                out.correct += 1
                out.lines.append(f"ok: {label}")
            elif d[i] <= near:
                out.wrong += 1
                out.lines.append(f"x wrong place: {label} (nearest {feature} {truth}, {int(d[i])} tiles off)")
            else:
                out.invented += 1
                unseen = np.argwhere(real & ~explored[floor])
                guessed = len(unseen) and int(np.min(np.maximum(np.abs(unseen[:, 0] - r), np.abs(unseen[:, 1] - c)))) <= tolerance
                note = "; there is one there, but you never saw it" if guessed else ""
                out.lines.append(f"x invented: {label} (nearest seen {feature} {truth}, {int(d[i])} tiles away{note})")
    return out


# --- the known places (code-written MAP, `code_facts`) ----------------------------------
# Under a turn-window eviction the model writes its notes from the last ~50 turns only, so its
# MAP restates its own view (2026-11-02: 84% of the claims were view cells, 0 of 101 segments
# located a needed resource out of view). The game writes the places instead, from every cell
# the agent has seen; the notes keep what code cannot write (plan, lessons, survival rules).

PROVIDES = {"make_wood_pickaxe": "pick1", "make_stone_pickaxe": "pick2", "make_iron_pickaxe": "pick3",
            "make_diamond_pickaxe": "pick4", "place_table": "table", "place_furnace": "furnace"}
"""The capability an achievement's craft or placement gives."""
REQUIRES: dict[str, tuple[str, ...]] = {
    "make_wood_pickaxe": ("table",), "make_wood_sword": ("table",),
    "collect_stone": ("pick1",), "collect_coal": ("pick1",), "place_stone": ("pick1",), "place_furnace": ("pick1",),
    "make_stone_pickaxe": ("table", "pick1"), "make_stone_sword": ("table", "pick1"), "make_arrow": ("table", "pick1"),
    "make_torch": ("table", "pick1"), "collect_iron": ("pick2",),
    "make_iron_pickaxe": ("table", "furnace", "pick2"), "make_iron_sword": ("table", "furnace", "pick2"),
    "make_iron_armour": ("table", "furnace", "pick2"), "collect_diamond": ("pick3",),
    "make_diamond_pickaxe": ("table", "pick3"), "make_diamond_sword": ("table", "pick3"),
    "collect_sapphire": ("pick4",), "collect_ruby": ("pick4",),
}
"""The capabilities (pickaxe level, a table or furnace on the floor) an achievement needs; materials aside."""
_CAP_PROVIDER = {v: k for k, v in PROVIDES.items()}
_COST = {"make_wood_pickaxe": "MAKE_WOOD_PICKAXE", "make_stone_pickaxe": "MAKE_STONE_PICKAXE",
         "make_iron_pickaxe": "MAKE_IRON_PICKAXE", "make_wood_sword": "MAKE_WOOD_SWORD",
         "make_stone_sword": "MAKE_STONE_SWORD", "make_iron_sword": "MAKE_IRON_SWORD",
         "make_diamond_pickaxe": "MAKE_DIAMOND_PICKAXE", "make_diamond_sword": "MAKE_DIAMOND_SWORD",
         "make_arrow": "MAKE_ARROW", "make_torch": "MAKE_TORCH", "make_iron_armour": "MAKE_IRON_ARMOUR",
         "place_table": "PLACE_TABLE", "place_furnace": "PLACE_FURNACE", "place_stone": "PLACE_STONE"}
_MATERIAL_FEATURE = {"wood": "tree", "stone": "stone", "coal": "coal", "iron": "iron", "diamond": "diamond"}
_COLLECT_FEATURE = {"collect_wood": "tree", "collect_stone": "stone", "collect_coal": "coal", "collect_iron": "iron",
                    "collect_diamond": "diamond", "collect_sapphire": "sapphire", "collect_ruby": "ruby",
                    "enter_dungeon": "ladder down"}
KNOWN_FEATURES = ("tree", "water", "stone", "coal", "iron", "diamond", "sapphire", "ruby", "table", "furnace",
                  "ladder down", "ladder up", "chest", "fountain", "enchantment table", "lava")
"""What the known places list, in this order after the needed ones."""


def capabilities(game: Any, explored: np.ndarray) -> set[str]:
    """pick1..pick4 from the inventory's pickaxe level; table / furnace if one stands on a seen cell of this floor."""
    items = game.snapshot()["items"]
    out = {f"pick{k}" for k in range(1, int(items["pickaxe"]) + 1)}
    for kind in ("table", "furnace"):
        if (_feature_mask(game, game.floor, kind) & explored[game.floor]).any():
            out.add(kind)
    return out


def _cost(name: str) -> dict[str, int]:
    action = _COST.get(name)
    if action in RECIPES:
        return RECIPES[action][0]
    return PLACEMENTS.get(action, {}) if action else {}


def critical_path(game: Any, explored: np.ndarray, steps: int = 1) -> tuple[list[str], set[str], set[str]]:
    """The next `steps` unmet LADDER steps (a pickaxe lost at a death counts as unmet) with, recursively,
    the providers of the capabilities they lack: (achievements, materials still missing, features to find)."""
    caps, have = capabilities(game, explored), set(game.achievements)
    items = game.snapshot()["items"]
    heads = []
    for name, _ in LADDER:
        lost = name in PROVIDES and PROVIDES[name].startswith("pick") and PROVIDES[name] not in caps
        if name not in have or lost:
            heads.append(name)
            if len(heads) == steps:
                break
    achs: list[str] = []

    def add(name: str, depth: int = 0) -> None:
        if name in achs or depth > 6:
            return
        for cap in REQUIRES.get(name, ()):
            if cap not in caps and cap in _CAP_PROVIDER:
                add(_CAP_PROVIDER[cap], depth + 1)
        achs.append(name)  # prerequisites first

    for h in heads:
        add(h)
    need: dict[str, int] = {}  # summed over the steps: 2 wood for the table + 1 for the pickaxe = 3
    for a in achs:
        for k, n in _cost(a).items():
            need[k] = need.get(k, 0) + n
    mats = {k for k, n in need.items() if items.get(k, 0) < n}
    feats = {_MATERIAL_FEATURE[m] for m in mats if m in _MATERIAL_FEATURE} | {_COLLECT_FEATURE[a] for a in achs if a in _COLLECT_FEATURE}
    return achs, mats, feats


_COMPASS = {(-1, 0): "north", (1, 0): "south", (0, 1): "east", (0, -1): "west",
            (-1, 1): "north-east", (-1, -1): "north-west", (1, 1): "south-east", (1, -1): "south-west"}


def direction(dr: int, dc: int) -> str:
    """A compass word for an offset (row grows south, col grows east); diagonal when neither axis dominates 2:1."""
    if dr == dc == 0:
        return "here"
    sr = int(np.sign(dr)) if abs(dr) * 2 >= abs(dc) else 0
    sc = int(np.sign(dc)) if abs(dc) * 2 >= abs(dr) else 0
    return _COMPASS[(sr, sc)]


OTHER_FLOOR_FEATURES = ("ladder up", "ladder down", "table", "furnace", "fountain", "enchantment table")
"""What KNOWN PLACES lists for the other floors seen (with the features needed next)."""


def other_floors(game: Any, explored: np.ndarray, needed: set[str] | frozenset = frozenset()) -> list[str]:
    """One line per OTHER floor the player has seen: its ladders, stations and the features needed next,
    one seen cell each (the one nearest that floor's up-ladder, where a descent arrives)."""
    from craftax_env.glyph import FLOOR_NAMES

    out = []
    for f in range(explored.shape[0]):
        if f == game.floor or not explored[f].any():
            continue
        up = np.argwhere(_feature_mask(game, f, "ladder up") & explored[f])
        ref = up[0] if len(up) else np.array([24, 24])
        parts = []
        for feature in [x for x in KNOWN_FEATURES if x in OTHER_FLOOR_FEATURES or x in needed]:
            cells = np.argwhere(_feature_mask(game, f, feature) & explored[f])
            if len(cells):
                r, c = (int(v) for v in cells[np.argmin(np.abs(cells - ref).max(axis=1))])
                parts.append(f"{feature} {place(f, (r, c))}" + (" <- needed next" if feature in needed else ""))
        name = FLOOR_NAMES[f] if f < len(FLOOR_NAMES) else str(f)
        out.append(f"- F{f} ({name}): " + ("; ".join(parts) or "no ladder or station seen"))
    return out


def known_places(game: Any, explored: np.ndarray, per_feature: int = 2) -> str:
    """The code-written MAP: for each feature, the nearest SEEN cells on the current floor (place, distance,
    direction, whether it is in the 9x11 view); the features the next tech steps need come first, marked."""
    floor = game.floor
    pr, pc = (int(v) for v in game.position)
    r0, c0 = pr - VIEW_CENTER[0], pc - VIEW_CENTER[1]
    _, _, needed = critical_path(game, explored, steps=2)
    if game.snapshot()["vitals"]["drink"] <= 4:
        needed.add("water")
    order = [f for f in KNOWN_FEATURES if f in needed] + [f for f in KNOWN_FEATURES if f not in needed]
    lines, unseen = [], []
    for feature in order:
        cells = np.argwhere(_feature_mask(game, floor, feature) & explored[floor])
        if not len(cells):
            if feature in needed:
                unseen.append(feature)
            continue
        d = np.maximum(np.abs(cells[:, 0] - pr), np.abs(cells[:, 1] - pc))
        picks = []
        for i in np.argsort(d, kind="stable"):
            r, c = (int(v) for v in cells[i])
            if any(max(abs(r - a), abs(c - b)) <= 2 for a, b in picks):  # one entry per patch
                continue
            picks.append((r, c))
            if len(picks) == per_feature:
                break
        parts = []
        for r, c in picks:
            dist = max(abs(r - pr), abs(c - pc))
            where = "in your view" if r0 <= r < r0 + VIEW_ROWS and c0 <= c < c0 + VIEW_COLS else direction(r - pr, c - pc)
            parts.append(f"{place(floor, (r, c))} {dist} tiles {where}" if where != "in your view" else f"{place(floor, (r, c))} in your view")
        lines.append(f"- {feature}: " + "; ".join(parts) + ("   <- needed next" if feature in needed else ""))
    head = (f"KNOWN PLACES on F{floor} (written by the game from every cell you have seen on this floor; "
            f"you are at {place(floor, (pr, pc))}; rows grow south, cols grow east)")
    tail = [f"- needed next but never seen on this floor: {', '.join(unseen)} (search unexplored ground)"] if unseen else []
    others = other_floors(game, explored, needed)
    if others:
        tail += ["Other floors you have seen (written by the game):", *others]
    return "\n".join([head, *(lines or ["- nothing notable seen yet"]), *tail])
