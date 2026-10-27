"""Text-only Pokémon FireRed environment (mGBA, headless, unthrottled).

reset() -> obs, step(buttons) -> (obs, StepInfo), observe(), milestones, score, save_state(path).

Observation = status (location, position, party, bag, badges) + decoded
screen text (dialogue, menus, battle menus, ▶ cursor) + battle state + a
semantic map of the current area with doors labeled by warp destination.
"""

from __future__ import annotations

import os
import struct
import zlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from pokemon_env.firered.game import BEHAVIORS, C, DATA, SYM, FireRedState, GBAMemory, decode_text, map_name, pretty
from pokemon_env.firered.screen_text import WAITING_STATES, FontBank, read_screen_text

INFO_LEVELS = ("raw", "semantic", "oracle")
BUTTONS = ("up", "down", "left", "right", "a", "b", "start", "select", "l", "r")
DIRECTIONS = ("up", "down", "left", "right")
SCRIPT_BUSY = {v & ~1 for v in DATA["script_busy"].values()}
BATTLE_INPUT = {addr & ~1 for _, addr in DATA["battle_input"]}
SCRIPT_MODE_NATIVE = 2

LEGEND = (
    "@=you .=walkable #=blocked ~=tall grass W=water D=door/stairs/warp "
    "1-9,0,a-z=labeled door (see list) N=NPC O=item ball !=sign t=cuttable tree r=smashable rock "
    "b=strength boulder →←↑↓=one-way ledge C=counter (talk across it) P=PC B=shelf T=TV *=spin/slide tile"
)
DOOR_LABELS = list("1234567890") + list("acdefghijklmnopqsuvwxyz")  # skip b,r,t (objects)
_BUILDING_HINTS = [("PokemonCenter", "Pokémon Center: heal party, PC"), ("Mart", "Poké Mart: buy items"), ("Gym", "Gym")]

_WATER = {"MB_POND_WATER", "MB_FAST_WATER", "MB_DEEP_WATER", "MB_WATERFALL", "MB_OCEAN_WATER",
          "MB_CYCLING_ROAD_WATER", "MB_UNUSED_WATER", "MB_EASTWARD_CURRENT", "MB_WESTWARD_CURRENT",
          "MB_NORTHWARD_CURRENT", "MB_SOUTHWARD_CURRENT"}
_LEDGES = {"MB_JUMP_EAST": "→", "MB_JUMP_WEST": "←", "MB_JUMP_NORTH": "↑", "MB_JUMP_SOUTH": "↓"}
_SPECIAL = {"MB_COUNTER": "C", "MB_PC": "P", "MB_BOOKSHELF": "B", "MB_POKEMART_SHELF": "B", "MB_TELEVISION": "T"}
_OBJECT_GLYPH = {C["OBJ_EVENT_GFX_ITEM_BALL"]: "O", C["OBJ_EVENT_GFX_CUT_TREE"]: "t",
                 C["OBJ_EVENT_GFX_ROCK_SMASH_ROCK"]: "r", C["OBJ_EVENT_GFX_PUSHABLE_BOULDER"]: "b"}


@dataclass
class StepInfo:
    frames: int
    location: str
    coords: Tuple[int, int]
    new_milestones: List[str] = field(default_factory=list)


def _rel(dx: int, dy: int) -> str:
    """Relative offset -> '3 left, 1 up' (screen axes: +x right, +y down)."""
    parts = []
    if dx:
        parts.append(f"{abs(dx)} {'right' if dx > 0 else 'left'}")
    if dy:
        parts.append(f"{abs(dy)} {'down' if dy > 0 else 'up'}")
    return ", ".join(parts) if parts else "here"


def _cell_glyph(behavior: int, collision: int) -> str:
    name = BEHAVIORS.get(behavior, "")
    if name in _LEDGES:
        return _LEDGES[name]
    if name in _WATER:
        return "W"
    if any(k in name for k in ("DOOR", "WARP", "LADDER", "STAIR", "ESCALATOR")):
        return "D"
    if name in _SPECIAL:
        return _SPECIAL[name]
    if collision:
        return "#"
    if name == "MB_TALL_GRASS":
        return "~"
    if name.startswith(("MB_SPIN_", "MB_SLIDE_", "MB_WALK_")):
        return "*"
    return "."


def _is_warp_tile(behavior: int) -> bool:
    name = BEHAVIORS.get(behavior, "")
    return any(k in name for k in ("DOOR", "WARP", "LADDER", "STAIR", "ESCALATOR"))


def _warp_hint(behavior: int) -> str:
    """Stairs and exit mats only fire when you press toward them while standing on the tile."""
    name = BEHAVIORS.get(behavior, "")
    for key, d in (("EAST", "right"), ("WEST", "left"), ("NORTH", "up"), ("SOUTH", "down")):
        if name == f"MB_{key}_ARROW_WARP":
            return f" (step on it, then press {d})"
    if name.endswith("_STAIR_WARP"):
        return f" (step on it, then press {'right' if 'RIGHT' in name else 'left'})"
    return ""


# (milestone id, predicate(location, party_size, badge_count))
MILESTONES = [
    ("PALLET_TOWN", lambda l, p, b: l == "PalletTown"),
    ("OAKS_LAB", lambda l, p, b: l == "PalletTown_ProfessorOaksLab"),
    ("STARTER_OBTAINED", lambda l, p, b: p >= 1),
    ("ROUTE_1", lambda l, p, b: l == "Route1"),
    ("VIRIDIAN_CITY", lambda l, p, b: l == "ViridianCity"),
    ("ROUTE_2", lambda l, p, b: l == "Route2"),
    ("VIRIDIAN_FOREST", lambda l, p, b: l == "ViridianForest"),
    ("PEWTER_CITY", lambda l, p, b: l == "PewterCity"),
    ("PEWTER_GYM", lambda l, p, b: l == "PewterCity_Gym"),
    ("BOULDER_BADGE", lambda l, p, b: b >= 1),
    ("ROUTE_3", lambda l, p, b: l == "Route3"),
    ("MT_MOON", lambda l, p, b: l.startswith("MtMoon")),
    ("CERULEAN_CITY", lambda l, p, b: l == "CeruleanCity"),
    ("CERULEAN_GYM", lambda l, p, b: l == "CeruleanCity_Gym"),
    ("CASCADE_BADGE", lambda l, p, b: b >= 2),
    ("VERMILION_CITY", lambda l, p, b: l == "VermilionCity"),
    ("SS_ANNE", lambda l, p, b: l.startswith("SSAnne")),
    ("VERMILION_GYM", lambda l, p, b: l == "VermilionCity_Gym"),
    ("THUNDER_BADGE", lambda l, p, b: b >= 3),
    ("ROCK_TUNNEL", lambda l, p, b: l.startswith("RockTunnel")),
    ("LAVENDER_TOWN", lambda l, p, b: l == "LavenderTown"),
    ("CELADON_CITY", lambda l, p, b: l == "CeladonCity"),
    ("CELADON_GYM", lambda l, p, b: l == "CeladonCity_Gym"),
    ("RAINBOW_BADGE", lambda l, p, b: b >= 4),
    ("POKEMON_TOWER", lambda l, p, b: l.startswith("PokemonTower")),
    ("FUCHSIA_CITY", lambda l, p, b: l == "FuchsiaCity"),
    ("FUCHSIA_GYM", lambda l, p, b: l == "FuchsiaCity_Gym"),
    ("SOUL_BADGE", lambda l, p, b: b >= 5),
    ("SAFFRON_CITY", lambda l, p, b: l == "SaffronCity"),
    ("SILPH_CO", lambda l, p, b: l.startswith("SilphCo")),
    ("SAFFRON_GYM", lambda l, p, b: l == "SaffronCity_Gym"),
    ("MARSH_BADGE", lambda l, p, b: b >= 6),
    ("CINNABAR_ISLAND", lambda l, p, b: l == "CinnabarIsland"),
    ("CINNABAR_GYM", lambda l, p, b: l == "CinnabarIsland_Gym"),
    ("VOLCANO_BADGE", lambda l, p, b: b >= 7),
    ("VIRIDIAN_GYM", lambda l, p, b: l == "ViridianCity_Gym" and b >= 7),
    ("EARTH_BADGE", lambda l, p, b: b >= 8),
    ("VICTORY_ROAD", lambda l, p, b: l.startswith("VictoryRoad")),
    ("INDIGO_PLATEAU", lambda l, p, b: l.startswith("IndigoPlateau")),
    ("ELITE_FOUR", lambda l, p, b: l.startswith("PokemonLeague_")),
    ("CHAMPION", lambda l, p, b: l in ("PokemonLeague_ChampionsRoom", "PokemonLeague_HallOfFame")),
]


class FireRedGame:
    def __init__(
        self,
        rom_path: str,
        state_path: Optional[str] = None,
        info_level: str = "semantic",
        view: str = "full",
        radius: int = 7,
        spaced: bool = True,
        settle_max: int = 300,
        settle_stable: int = 10,
    ):
        if info_level not in INFO_LEVELS:
            raise ValueError(f"info_level must be one of {INFO_LEVELS}")
        if not os.path.exists(rom_path):
            raise FileNotFoundError(f"ROM not found at {rom_path}")
        import mgba.core
        import mgba.log

        mgba.log.silence()
        self.core = mgba.core.load_path(rom_path)
        self.core.reset()
        with open(rom_path, "rb") as f:
            self.mem = GBAMemory(self.core, f.read())
        self.state = FireRedState(self.mem)
        self.font = FontBank(self.mem)
        self.keys = {"up": self.core.KEY_UP, "down": self.core.KEY_DOWN, "left": self.core.KEY_LEFT,
                     "right": self.core.KEY_RIGHT, "a": self.core.KEY_A, "b": self.core.KEY_B,
                     "start": self.core.KEY_START, "select": self.core.KEY_SELECT,
                     "l": self.core.KEY_L, "r": self.core.KEY_R}
        self.state_path = state_path if state_path and os.path.exists(state_path) else None
        self._initial_state = open(self.state_path, "rb").read() if self.state_path else None
        self.info_level, self.view, self.radius, self.spaced = info_level, view, radius, spaced
        self.settle_max, self.settle_stable = settle_max, settle_stable
        self.steps = 0
        self.total_frames = 0
        self.milestones: Dict[str, int] = {}
        self.events: List[str] = []   # battle messages shown since the last step began
        self._last_msg = ""

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def reset(self, idle_frames: int = 0) -> str:
        """Load the start state; `idle_frames` extra frames shift the game's RNG (FireRed
        advances it every frame), so tasks sharing one start state meet different
        encounters and damage rolls."""
        self.core.reset()
        if self._initial_state is not None:
            self.core.load_raw_state(self._initial_state)
        self.steps = 0
        self.total_frames = 0
        self.milestones = {}
        self._run(2 + idle_frames)
        self._settle()
        self._update_milestones()
        return self.observe()

    def close(self):
        self.core = None

    def save_state(self, path: str):
        with open(path, "wb") as f:
            f.write(bytes(self.core.save_raw_state()))

    # ------------------------------------------------------------------
    # Stepping
    # ------------------------------------------------------------------

    def _run(self, frames: int):
        for _ in range(frames):
            self.core.run_frame()

    def step(self, buttons: List[str]) -> Tuple[str, StepInfo]:
        frames = 0
        self.events = []
        for b in buttons:
            b = b.lower().strip()
            if b not in BUTTONS:
                raise ValueError(f"unknown button {b!r}; valid: {BUTTONS}")
            frames += self._press(b) + self._settle()
        self.steps += 1
        self.total_frames += frames
        new = self._update_milestones()
        return self.observe(), StepInfo(frames=frames, location=self.state.location_name(),
                                        coords=self.state.player_xy(), new_milestones=new)

    def _press(self, b: str) -> int:
        key = self.keys[b]
        self.mem.refresh()
        if b in DIRECTIONS and self.state.mode() == "overworld":
            # Hold until the step starts (a short tap only turns the player in Gen 3).
            start = self.state.player_xy()
            self.core.set_keys(key)
            n = 0
            while n < 24:
                self.core.run_frame()
                n += 1
                self.mem.refresh()
                if self.state.player_xy() != start:
                    break
            self.core.clear_keys(key)
            return n
        self.core.set_keys(key)
        self._run(4)
        self.core.clear_keys(key)
        return 4

    def _snapshot(self) -> Tuple[tuple, int]:
        """(game-state signature, window-pixels CRC)."""
        m = self.mem
        printers = m.read(SYM["sTextPrinters"], 0x480)
        # per printer: currentChar pointer + active + state (not its arrow-animation counters)
        progress = b"".join(printers[i:i + 4] + printers[i + 0x1B:i + 0x1D] for i in range(0, 0x480, 0x24))
        win_crc = 0
        for wid in range(32):
            raw = m.read(SYM["gWindows"] + 12 * wid, 12)
            if raw[0] != 0xFF and raw[8:12] != b"\0\0\0\0":
                ptr = int.from_bytes(raw[8:12], "little")
                if ptr >> 24 == 2:
                    win_crc = zlib.crc32(m.read(ptr, raw[3] * raw[4] * 32), win_crc)
        sig = (
            progress,
            m.read(self.state.player_object(), 0x24),
            m.read(SYM["gPlayerAvatar"], 8),
            m.read(SYM["gMain"], 8),
            m.read(SYM["gPaletteFade"], 16),
            m.read(SYM["gBattleControllerExecFlags"], 4),
        )
        return sig, win_crc

    def _log_battle_message(self):
        """Battle messages often auto-advance within one step; keep them as events."""
        if self.state.mode() != "battle":
            self._last_msg = ""
            return
        # only text the message window's printer is actually printing (the buffer is also
        # reused to draw health-box names)
        p = self.mem.read(SYM["sTextPrinters"], 0x24)
        cur = int.from_bytes(p[0:4], "little")
        start = SYM["gDisplayedStringBattle"]
        msg = decode_text(self.mem.read(start, 0x12C))
        printing = p[0x1B] and start <= cur < start + 0x12C
        sentence = " " in msg and not msg.endswith(("♂", "♀"))  # not a bare health-box name
        if not (printing or sentence):
            return
        if msg and msg != self._last_msg:
            self._last_msg = msg
            if not msg.startswith("What will"):
                self.events.append(msg)

    def _busy(self) -> bool:
        """A cutscene is playing: the running script waits on movement/fade/sound/timer, or an
        object is executing a scripted (held) movement."""
        m = self.mem
        if self.state.mode() == "battle":
            # busy unless the player's controller waits for a choice or a message waits for A
            if (m.u32(SYM["gBattlerControllerFuncs"]) & ~1) in BATTLE_INPUT:
                return False
            printers = m.read(SYM["sTextPrinters"], 0x480)
            return not any(printers[i + 0x1B] and printers[i + 0x1C] in WAITING_STATES for i in range(0, 0x480, 0x24))
        ctx = SYM["sGlobalScriptContext"]
        if m.u8(SYM["sGlobalScriptContextStatus"]) == 0 and m.u8(ctx + 1) == SCRIPT_MODE_NATIVE \
                and (m.u32(ctx + 4) & ~1) in SCRIPT_BUSY:
            return True
        objs = m.read(SYM["gObjectEvents"], 0x24 * 16)
        return any((objs[i] & 0x41) == 0x41 and not objs[i] & 0x80 for i in range(0, 0x24 * 16, 0x24))

    def _settle(self) -> int:
        """Run until the game waits for input: no cutscene running, state unchanged for settle_stable
        frames, and window pixels static or only cycling through a few states (blinking ▼ arrow).
        Gives up after settle_max idle frames, or 5x that while a cutscene keeps running."""
        prev, stable, crcs = None, 0, []
        for i in range(1, 5 * self.settle_max + 1):
            self.core.run_frame()
            self.mem.refresh()
            self._log_battle_message()
            if self._busy():
                prev, stable, crcs = None, 0, []
                continue
            if i > self.settle_max and stable == 0:
                return i
            sig, crc = self._snapshot()
            if sig == prev:
                stable += 1
                crcs.append(crc)
            else:
                stable, crcs = 0, [crc]
            prev = sig
            if stable >= self.settle_stable:
                recent = crcs[-self.settle_stable:]
                if len(set(recent)) == 1 or (len(crcs) >= 40 and len(set(crcs[-40:])) <= 4):
                    return i
        return 5 * self.settle_max

    # ------------------------------------------------------------------
    # Milestones
    # ------------------------------------------------------------------

    def _update_milestones(self) -> List[str]:
        self.mem.refresh()
        if self.state.sb1 >> 24 != 2:  # save block not set up yet (title screen / intro)
            return []
        loc = self.state.location_name()
        party = self.state.party()
        nbadges = len(self.state.badges())
        new = []
        for mid, pred in MILESTONES:
            if mid not in self.milestones and pred(loc, len(party), nbadges):
                self.milestones[mid] = self.steps
                new.append(mid)
        return new

    @property
    def score(self) -> int:
        return len(self.milestones)

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------

    def observe(self) -> str:
        self.mem.refresh()
        mode = self.state.mode()
        in_game = self.state.sb1 >> 24 == 2
        parts = [self._status_block(mode, in_game)]
        if mode == "battle":
            parts.append(self._battle_block())
        if self.events:
            parts.append("=== MESSAGES SINCE YOUR LAST ACTION ===\n" + "\n".join(self.events[-12:]))
        windows = read_screen_text(self.mem, self.font, in_battle=mode == "battle")
        if windows:
            lines = []
            for w in windows:
                lines += w.lines
                if w.waiting_for_button and not (w.lines and w.lines[-1].endswith("▼")):
                    lines.append("▼ (press a to continue)")
            cursor = self._cursor_note(mode)
            if cursor:
                lines.append(cursor)
            parts.append("=== SCREEN TEXT ===\n" + "\n".join(lines))
        if mode == "overworld" and in_game:
            parts.append(self._map_block())
        return "\n\n".join(p for p in parts if p)

    def _cursor_note(self, mode: str) -> str:
        """Cursors drawn as tiles/highlights rather than a ▶ glyph (battle menus, party screen)."""
        m = self.mem
        if mode == "battle":
            scroll = m.u16(SYM["gBattle_BG0_Y"])
            if scroll == 160:  # action menu
                pos = m.u8(SYM["gActionSelectionCursor"])
                return f"▶ cursor on: {['FIGHT', 'BAG', 'POKéMON', 'RUN'][pos & 3]}"
            if scroll == 320:  # move menu
                pos = m.u8(SYM["gMoveSelectionCursor"]) & 3
                me = self.state.battle_mon(0)
                moves = [mv for mv, _ in me["moves"]] if me else []
                return f"▶ cursor on: {moves[pos] if pos < len(moves) else f'move slot {pos + 1}'}"
        elif (self.state.callback2() & ~1) == (SYM["CB2_UpdatePartyMenu"] & ~1):
            slot = struct.unpack("<b", m.read(SYM["gPartyMenu"] + 9, 1))[0]
            party = self.state.party()
            if 0 <= slot < len(party):
                return f"▶ cursor on: party slot {slot + 1} ({party[slot]['nickname'] or party[slot]['species']})"
            if slot in (6, 7):
                return f"▶ cursor on: {'CANCEL' if slot == 7 else 'CONFIRM'}"
        return ""

    def _status_block(self, mode: str, in_game: bool) -> str:
        if not in_game:
            return f"=== STATUS ===\nStep {self.steps} | Mode: {mode} (title / intro screens)"
        s = self.state
        x, y = s.player_xy()
        loc = s.location_name()
        region = s.region_name()
        place = pretty(loc)
        if region and not place.upper().startswith(region.upper()):
            place = f"{region} — {place}"
        lines = ["=== STATUS ===",
                 f"Step {self.steps} | Mode: {mode} | Location: {place} | Position: ({x},{y}) facing {s.facing()}"]
        badges = s.badges()
        lines.append(f"Badges: {', '.join(badges) if badges else 'none'} | Money: ¥{s.money()}")
        party = s.party()
        if party:
            for i, p in enumerate(party, 1):
                nick = f" ({p['species']})" if p["nickname"] and p["nickname"] != p["species"] else ""
                lines.append(f"Party {i}: {p['nickname'] or p['species']}{nick} Lv{p['level']} HP {p['hp']}/{p['max_hp']} "
                             f"{p['status']} [{'/'.join(p['types'])}] moves: "
                             + ", ".join(f"{m} {pp}PP" for m, pp in p["moves"]))
        else:
            lines.append("Party: (empty)")
        for pocket, items in s.bag().items():
            lines.append(f"Bag/{pocket}: " + ", ".join(f"{n} x{q}" for n, q in items))
        return "\n".join(lines)

    def _battle_block(self) -> str:
        s = self.state
        me, opp = s.battle_mon(0), s.battle_mon(1)
        lines = [f"=== BATTLE ({'trainer' if s.is_trainer_battle() else 'wild'}) ==="]
        if opp:
            pct = 100 * opp["hp"] / max(opp["max_hp"], 1)
            line = f"Opponent: {opp['species']} Lv{opp['level']} HP {pct:.0f}% {opp['status']} [{'/'.join(opp['types'])}]"
            if self.info_level == "oracle":
                line += f" HP {opp['hp']}/{opp['max_hp']} moves: " + ", ".join(m for m, _ in opp["moves"])
            lines.append(line)
        if me:
            lines.append(f"Your active: {me['nickname'] or me['species']} Lv{me['level']} HP {me['hp']}/{me['max_hp']} "
                         f"{me['status']} moves: " + ", ".join(f"{m} {pp}PP" for m, pp in me["moves"]))
        return "\n".join(lines)

    def _map_block(self) -> str:
        s = self.state
        cells = s.grid()
        h, w = len(cells), len(cells[0]) if cells else 0
        if not h:
            return ""
        px, py = s.player_xy()
        grid = [[_cell_glyph(b, c) for b, c in row] for row in cells]

        for e in s.bg_events():
            if 0 <= e["y"] < h and 0 <= e["x"] < w:
                if e["kind"] <= 4:
                    grid[e["y"]][e["x"]] = "!"
                elif e["kind"] == 7 and self.info_level == "oracle":
                    grid[e["y"]][e["x"]] = "?"

        # Warp events on plain floor never fire; keep only tiles whose behavior is a warp
        # (door, stairs, ladder, exit mat), unless a destination has no such tile.
        by_dest: Dict[Tuple[int, int], List[Tuple[int, int]]] = {}
        for wv in sorted(s.warps(), key=lambda e: (e["y"], e["x"])):
            if 0 <= wv["y"] < h and 0 <= wv["x"] < w:
                by_dest.setdefault((wv["group"], wv["num"]), []).append((wv["x"], wv["y"]))
        dest_label: Dict[Tuple[int, int], str] = {}
        doors: Dict[Tuple[int, int], List[Tuple[int, int]]] = {}
        for dest, tiles in by_dest.items():
            live = [(x, y) for x, y in tiles if _is_warp_tile(cells[y][x][0])] or tiles
            doors[dest] = live
            if len(dest_label) < len(DOOR_LABELS):
                dest_label[dest] = DOOR_LABELS[len(dest_label)]
            for x, y in live:
                grid[y][x] = dest_label.get(dest, "D") if self.info_level != "raw" else "D"

        objs = []
        for o in s.objects():
            if 0 <= o["y"] < h and 0 <= o["x"] < w:
                g = _OBJECT_GLYPH.get(o["gfx"], "N")
                grid[o["y"]][o["x"]] = g
                objs.append((g, o["x"], o["y"]))
        if 0 <= py < h and 0 <= px < w:
            grid[py][px] = "@"

        if self.view == "local":
            x0, x1 = max(0, px - self.radius), min(w, px + self.radius + 1)
            y0, y1 = max(0, py - self.radius), min(h, py + self.radius + 1)
        else:
            x0, x1, y0, y1 = 0, w, 0, h
        sep = " " if self.spaced else ""
        pad = " " * 4
        out = [f"=== MAP: {pretty(s.location_name())} ({w}x{h}, "
               f"{'full map' if self.view == 'full' else f'local view, radius {self.radius}'}; x→ columns, y↓ rows) ===",
               pad + sep.join(str((x // 10) % 10) for x in range(x0, x1)),
               pad + sep.join(str(x % 10) for x in range(x0, x1)),
               *[f"{y:>3} " + sep.join(grid[y][x0:x1]) for y in range(y0, y1)],
               f"Legend: {LEGEND}"]
        if self.info_level == "raw":
            return "\n".join(out)

        poi = ["=== POINTS OF INTEREST (offsets from you) ==="]
        for dest, label in dest_label.items():
            nearest = min(doors[dest], key=lambda p: abs(p[0] - px) + abs(p[1] - py))
            if dest == (0x7F, 0x7F):
                name = f"exit → {pretty(s.dynamic_warp_name())}"
            else:
                name = pretty(map_name(*dest))
                for key, hint in _BUILDING_HINTS:
                    if key in map_name(*dest):
                        name += f" ({hint})"
                        break
            how = _warp_hint(cells[nearest[1]][nearest[0]][0]) if 0 <= nearest[1] < h and 0 <= nearest[0] < w else ""
            poi.append(f"[{label}] {name} at {nearest}: {_rel(nearest[0] - px, nearest[1] - py)}{how}")
        for d, dest in s.connections().items():
            poi.append(f"Map edge {d} → {pretty(dest)}")
        kinds = {"O": "Item ball", "t": "Cuttable tree", "r": "Smashable rock", "b": "Boulder", "N": "NPC"}
        for g, ox, oy in sorted(objs, key=lambda o: abs(o[1] - px) + abs(o[2] - py))[:8]:
            poi.append(f"{kinds[g]} at ({ox},{oy}): {_rel(ox - px, oy - py)}")
        return "\n".join(out) + "\n\n" + "\n".join(poi)
