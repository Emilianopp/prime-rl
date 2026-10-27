"""Pokémon FireRed (US v1.0) game-state readers over mGBA memory.

Addresses and constants come from data.json (generated from pret/pokefirered
by gen_data.py). All reads are exact struct reads; nothing is scanned or
guessed. ROM tables are read from the ROM file itself.
"""

from __future__ import annotations

import json
import os
import struct
from typing import Dict, List, Optional, Tuple

DATA = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "data.json")))
SYM: Dict[str, int] = DATA["symbols"]
C = DATA["constants"]
CHARMAP: Dict[int, str] = {int(k): v for k, v in DATA["charmap"].items()}
MAP_GROUPS: List[List[str]] = DATA["map_groups"]
BEHAVIORS: Dict[int, str] = {int(k): v for k, v in DATA["metatile_behaviors"].items()}

MAP_OFFSET = C["MAP_OFFSET"]
FACING = {1: "down", 2: "up", 3: "left", 4: "right"}
BADGES = ["Boulder", "Cascade", "Thunder", "Rainbow", "Soul", "Marsh", "Volcano", "Earth"]
# Gen 3 party-data substructure orders, indexed by personality % 24 (G=growth A=attacks E=EVs M=misc)
SUBSTRUCT_ORDERS = ["GAEM", "GAME", "GEAM", "GEMA", "GMAE", "GMEA", "AGEM", "AGME", "AEGM", "AEMG", "AMGE", "AMEG",
                    "EGAM", "EGMA", "EAGM", "EAMG", "EMGA", "EMAG", "MGAE", "MGEA", "MAGE", "MAEG", "MEGA", "MEAG"]


# Argument bytes after 0xFC <code> (EXT_CTRL_CODE_* in pokefirered text.c)
_EXT_CTRL_ARGS = {0x01: 1, 0x02: 1, 0x03: 1, 0x04: 3, 0x05: 1, 0x06: 1, 0x07: 0, 0x08: 1, 0x09: 0, 0x0A: 0,
                  0x0B: 2, 0x0C: 1, 0x0D: 1, 0x0E: 1, 0x0F: 0, 0x10: 2, 0x11: 1, 0x12: 1, 0x13: 1, 0x14: 1,
                  0x15: 0, 0x16: 0, 0x17: 0, 0x18: 0}


def decode_text(raw: bytes) -> str:
    """Decode a Gen 3 string, skipping control codes and their arguments."""
    out, i = [], 0
    while i < len(raw):
        b = raw[i]
        i += 1
        if b == 0xFF:  # EOS
            break
        if b in (0xFE, 0xFA, 0xFB):  # newline, scroll, new paragraph
            out.append(" ")
        elif b == 0xFC:  # extended control code + args
            code = raw[i] if i < len(raw) else 0
            i += 1 + _EXT_CTRL_ARGS.get(code, 0)
        elif b in (0xF8, 0xF9, 0xFD):  # keypad icon / extra symbol / unexpanded placeholder: 1 arg
            i += 1
        elif b >= 0xF7:
            continue
        else:
            out.append(CHARMAP.get(b, ""))
    return " ".join("".join(out).split())


def map_name(group: int, num: int) -> str:
    try:
        return MAP_GROUPS[group][num]
    except (IndexError, TypeError):
        return f"MAP_{group}_{num}"


def pretty(name: str) -> str:
    """'PalletTown_ProfessorOaksLab' -> 'Pallet Town Professor Oaks Lab'."""
    out = ""
    for i, ch in enumerate(name.replace("_", " ")):
        prev = name[i - 1] if i else " "
        nxt = name[i + 1] if i + 1 < len(name) else ""
        if i and ch.isupper() and prev not in " _" and (prev.islower() or (nxt.islower() and not prev.isdigit())):
            out += " "
        elif i and ch.isdigit() and prev.islower():
            out += " "
        out += ch
    return " ".join(out.split())


class GBAMemory:
    """Bulk access to EWRAM/IWRAM/VRAM (snapshotted on refresh) and the ROM file."""

    def __init__(self, core, rom_bytes: bytes):
        from mgba._pylib import ffi

        self._ffi = ffi
        self.core = core
        self.rom = rom_bytes
        self._mcore = core.memory.u8._core
        self.refresh()

    def _block(self, region: int):
        """Live (zero-copy) view of an emulator memory block; slicing copies just that slice."""
        size = self._ffi.new("size_t *")
        ptr = self._ffi.cast("uint8_t *", self._mcore.getMemoryBlock(self._mcore, region, size))
        return self._ffi.buffer(ptr, size[0])

    def refresh(self):
        """Kept for callers; the views are live, so there is nothing to re-read."""
        if not hasattr(self, "ewram"):
            self.ewram = self._block(2)
            self.iwram = self._block(3)
            self.vram = self._block(6)

    def io16(self, addr: int) -> int:
        return self.core.memory.u16[addr]

    def read(self, addr: int, n: int) -> bytes:
        region = addr >> 24
        if region == 2:
            off = addr & 0x3FFFF
            return self.ewram[off:off + n]
        if region == 3:
            off = addr & 0x7FFF
            return self.iwram[off:off + n]
        if region in (8, 9):
            off = addr - 0x08000000
            return self.rom[off:off + n]
        if region == 6:
            off = addr & 0x1FFFF
            return self.vram[off:off + n]
        raise ValueError(f"unsupported address {addr:#x}")

    def u8(self, a): return self.read(a, 1)[0]
    def u16(self, a): return struct.unpack_from("<H", self.read(a, 2))[0]
    def s16(self, a): return struct.unpack_from("<h", self.read(a, 2))[0]
    def u32(self, a): return struct.unpack_from("<I", self.read(a, 4))[0]
    def s32(self, a): return struct.unpack_from("<i", self.read(a, 4))[0]


class FireRedState:
    """Typed readers for the game state. Call mem.refresh() before reading a new frame."""

    def __init__(self, mem: GBAMemory):
        self.mem = mem
        self._attr_cache: Dict[int, List[int]] = {}

    # --- pointers ---------------------------------------------------------
    @property
    def sb1(self) -> int: return self.mem.u32(SYM["gSaveBlock1Ptr"])
    @property
    def sb2(self) -> int: return self.mem.u32(SYM["gSaveBlock2Ptr"])

    def callback2(self) -> int:
        return self.mem.u32(SYM["gMain"] + 4)

    def mode(self) -> str:
        cb = self.callback2() & ~1
        if cb == SYM["BattleMainCB2"] & ~1:
            return "battle"
        if cb == SYM["CB2_Overworld"] & ~1:
            return "overworld"
        return "menu"

    # --- location ---------------------------------------------------------
    def location(self) -> Tuple[int, int]:
        b = self.mem.read(self.sb1 + 4, 2)
        return b[0], b[1]

    def location_name(self) -> str:
        return map_name(*self.location())

    def region_name(self) -> str:
        sec = self.mem.u8(SYM["gMapHeader"] + 0x14)
        idx = sec - C["MAPSEC_KANTO_START"]
        if 0 <= idx < 109:
            ptr = self.mem.u32(SYM["sMapNames"] + 4 * idx)
            return decode_text(self.mem.read(ptr, 32)).title()
        return ""

    def player_object(self) -> int:
        oid = self.mem.u8(SYM["gPlayerAvatar"] + 5)
        return SYM["gObjectEvents"] + 0x24 * oid

    def player_xy(self) -> Tuple[int, int]:
        o = self.player_object()
        return self.mem.s16(o + 0x10) - MAP_OFFSET, self.mem.s16(o + 0x12) - MAP_OFFSET

    def facing(self) -> str:
        return FACING.get(self.mem.u8(self.player_object() + 0x18) & 0xF, "?")

    # --- map --------------------------------------------------------------
    def _layout(self) -> int:
        return self.mem.u32(SYM["gMapHeader"])

    def map_size(self) -> Tuple[int, int]:
        lay = self._layout()
        return self.mem.s32(lay), self.mem.s32(lay + 4)

    def _attributes(self, tileset_ptr: int, count: int) -> List[int]:
        if tileset_ptr not in self._attr_cache:
            attrs_ptr = self.mem.u32(tileset_ptr + 0x14)
            raw = self.mem.read(attrs_ptr, 4 * count)
            self._attr_cache[tileset_ptr] = list(struct.unpack(f"<{len(raw) // 4}I", raw))
        return self._attr_cache[tileset_ptr]

    def grid(self) -> List[List[Tuple[int, int]]]:
        """[(behavior, collision)] for every cell of the current map (map coordinates)."""
        lay = self._layout()
        primary = self._attributes(self.mem.u32(lay + 0x10), C["NUM_METATILES_IN_PRIMARY"])
        secondary = self._attributes(self.mem.u32(lay + 0x14), 1024 - C["NUM_METATILES_IN_PRIMARY"])
        w, h = self.map_size()
        xs = self.mem.s32(SYM["VMap"])
        vmap_ptr = self.mem.u32(SYM["VMap"] + 8)
        rows = []
        for y in range(h):
            raw = self.mem.read(vmap_ptr + 2 * ((y + MAP_OFFSET) * xs + MAP_OFFSET), 2 * w)
            row = []
            for e in struct.unpack(f"<{w}H", raw):
                mid = e & 0x3FF
                attr = primary[mid] if mid < C["NUM_METATILES_IN_PRIMARY"] else \
                    secondary[mid - C["NUM_METATILES_IN_PRIMARY"]] if mid - C["NUM_METATILES_IN_PRIMARY"] < len(secondary) else 0
                row.append((attr & 0x1FF, (e >> 10) & 3))
            rows.append(row)
        return rows

    def _events(self) -> int:
        return self.mem.u32(SYM["gMapHeader"] + 4)

    def warps(self) -> List[dict]:
        ev = self._events()
        n = self.mem.u8(ev + 1)
        ptr = self.mem.u32(ev + 8)
        out = []
        for i in range(n):
            x, y, _elev, _wid, num, grp = struct.unpack("<hhBBBB", self.mem.read(ptr + 8 * i, 8))
            out.append({"x": x, "y": y, "group": grp, "num": num})
        return out

    def bg_events(self) -> List[dict]:
        ev = self._events()
        n = self.mem.u8(ev + 3)
        ptr = self.mem.u32(ev + 16)
        out = []
        for i in range(n):
            x, y, _elev, kind = struct.unpack("<HHBB", self.mem.read(ptr + 12 * i, 6))
            out.append({"x": x, "y": y, "kind": kind})
        return out

    def connections(self) -> Dict[str, str]:
        ptr = self.mem.u32(SYM["gMapHeader"] + 0x0C)
        if not ptr:
            return {}
        n = self.mem.s32(ptr)
        arr = self.mem.u32(ptr + 4)
        out = {}
        for i in range(max(0, min(n, 8))):
            b = self.mem.read(arr + 12 * i, 12)
            d = C["CONNECTIONS"].get(str(b[0]))
            if d:
                out[d] = map_name(b[8], b[9])
        return out

    def dynamic_warp_name(self) -> str:
        b = self.mem.read(self.sb1 + 0x14, 2)
        return map_name(b[0], b[1])

    def objects(self) -> List[dict]:
        """Visible non-player object events (NPCs, item balls, cut trees, boulders)."""
        out = []
        for i in range(16):
            base = SYM["gObjectEvents"] + 0x24 * i
            flags = self.mem.read(base, 4)
            active, invisible, is_player = flags[0] & 1, flags[1] & 0x20, flags[2] & 1
            if not active or invisible or is_player:
                continue
            out.append({
                "x": self.mem.s16(base + 0x10) - MAP_OFFSET,
                "y": self.mem.s16(base + 0x12) - MAP_OFFSET,
                "gfx": self.mem.u8(base + 5),
            })
        return out

    # --- party / bag / progress -------------------------------------------
    def _species_name(self, sp: int) -> str:
        return decode_text(self.mem.read(SYM["gSpeciesNames"] + 11 * sp, 11))

    def _move_name(self, mv: int) -> str:
        return decode_text(self.mem.read(SYM["gMoveNames"] + 13 * mv, 13))

    def _item_name(self, it: int) -> str:
        return decode_text(self.mem.read(SYM["gItems"] + 0x2C * it, 14))

    def _type_name(self, t: int) -> str:
        return decode_text(self.mem.read(SYM["gTypeNames"] + 7 * t, 7)).title()

    def species_types(self, sp: int) -> List[str]:
        t1, t2 = self.mem.read(SYM["gSpeciesInfo"] + 0x1C * sp + 6, 2)
        return [self._type_name(t1)] if t1 == t2 else [self._type_name(t1), self._type_name(t2)]

    @staticmethod
    def status_name(s: int) -> str:
        if s & 0x7:
            return "SLP"
        for bit, name in ((0x8, "PSN"), (0x10, "BRN"), (0x20, "FRZ"), (0x40, "PAR"), (0x80, "TOX")):
            if s & bit:
                return name
        return "OK"

    def party(self) -> List[dict]:
        n = min(self.mem.u8(SYM["gPlayerPartyCount"]), 6)
        out = []
        for i in range(n):
            raw = self.mem.read(SYM["gPlayerParty"] + 100 * i, 100)
            pid, otid = struct.unpack_from("<II", raw, 0)
            key = pid ^ otid
            enc = struct.unpack_from("<12I", raw, 32)
            dec = struct.pack("<12I", *[w ^ key for w in enc])
            order = SUBSTRUCT_ORDERS[pid % 24]
            sub = {k: dec[12 * j:12 * j + 12] for j, k in enumerate(order)}
            species = struct.unpack_from("<H", sub["G"], 0)[0]
            moves = struct.unpack_from("<4H", sub["A"], 0)
            pp = sub["A"][8:12]
            status, level = struct.unpack_from("<IB", raw, 80)
            hp, max_hp = struct.unpack_from("<HH", raw, 86)
            out.append({
                "nickname": decode_text(raw[8:18]),
                "species": self._species_name(species),
                "species_id": species,
                "level": level, "hp": hp, "max_hp": max_hp,
                "status": self.status_name(status),
                "types": self.species_types(species),
                "moves": [(self._move_name(m), p) for m, p in zip(moves, pp) if m],
            })
        return out

    def _flag(self, flag: int) -> bool:
        return bool(self.mem.u8(self.sb1 + C["SB1_FLAGS"] + (flag >> 3)) >> (flag & 7) & 1)

    def badges(self) -> List[str]:
        return [name for i, name in enumerate(BADGES) if self._flag(C["FLAG_BADGE01_GET"] + i)]

    def _key(self) -> int:
        return self.mem.u32(self.sb2 + C["SB2_ENCRYPTION_KEY"])

    def money(self) -> int:
        return self.mem.u32(self.sb1 + C["SB1_MONEY"]) ^ self._key()

    def bag(self) -> Dict[str, List[Tuple[str, int]]]:
        key = self._key() & 0xFFFF
        out = {}
        for pocket, off, count in C["SB1_BAG_POCKETS"]:
            raw = self.mem.read(self.sb1 + off, 4 * count)
            items = []
            for j in range(count):
                item, qty = struct.unpack_from("<HH", raw, 4 * j)
                if item:
                    items.append((self._item_name(item), qty ^ key))
            if items:
                out[pocket] = items
        return out

    # --- battle -----------------------------------------------------------
    def battle_mon(self, battler: int) -> Optional[dict]:
        raw = self.mem.read(SYM["gBattleMons"] + 0x58 * battler, 0x58)
        species = struct.unpack_from("<H", raw, 0)[0]
        if not species:
            return None
        moves = struct.unpack_from("<4H", raw, 0x0C)
        hp, level, _f, max_hp = struct.unpack_from("<HBBH", raw, 0x28)
        return {
            "species": self._species_name(species),
            "nickname": decode_text(raw[0x30:0x3B]),
            "level": level, "hp": hp, "max_hp": max_hp,
            "types": [self._type_name(raw[0x21])] if raw[0x21] == raw[0x22] else
                     [self._type_name(raw[0x21]), self._type_name(raw[0x22])],
            "status": self.status_name(struct.unpack_from("<I", raw, 0x4C)[0]),
            "moves": [(self._move_name(m), p) for m, p in zip(moves, raw[0x24:0x28]) if m],
        }

    def is_trainer_battle(self) -> bool:
        return bool(self.mem.u32(SYM["gBattleTypeFlags"]) & 0x8)  # BATTLE_TYPE_TRAINER
