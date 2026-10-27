"""Read on-screen text in FireRed by matching text-window pixels against the ROM font.

GBA text has no character grid: the game renders glyphs into per-window 4bpp
pixel buffers (gWindows[i].tileData). This module rebuilds the game's own
Latin font bitmaps from the ROM and decodes each visible window line by line.
Every menu, dialogue box, shop list and battle menu is plain text drawn this
way, including the ▶ selection cursor, so one decoder covers all of them.
"""

from __future__ import annotations

import re
import struct
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from pokemon_env.firered.game import CHARMAP, SYM, GBAMemory

FONTS = [  # (glyph table, width table, tiles: (src byte offset, dest x, dest y), glyph stride bytes, height)
    ("sFontNormalLatinGlyphs", "sFontNormalLatinGlyphWidths", ((0, 0, 0), (16, 8, 0), (32, 0, 8), (48, 8, 8)), 64),
    ("sFontNormalCopy1LatinGlyphs", "sFontNormalCopy1LatinGlyphWidths", ((0, 0, 0), (16, 8, 0), (32, 0, 8), (48, 8, 8)), 64),
    ("sFontMaleLatinGlyphs", "sFontMaleLatinGlyphWidths", ((0, 0, 0), (16, 8, 0), (32, 0, 8), (48, 8, 8)), 64),
    ("sFontFemaleLatinGlyphs", "sFontFemaleLatinGlyphWidths", ((0, 0, 0), (16, 8, 0), (32, 0, 8), (48, 8, 8)), 64),
    ("sFontSmallLatinGlyphs", "sFontSmallLatinGlyphWidths", ((0, 0, 0), (16, 0, 8)), 32),
]
BAND = 14  # glyph rows used by the Latin fonts; a taller band picks up the next line
POW = (1 << np.arange(BAND, dtype=np.int64))
TEXT_PRINTER_SIZE = 0x24
# Text-printer render states that wait for A/B before continuing: WAIT, CLEAR, SCROLL_START
WAITING_STATES = (1, 2, 3)


@dataclass(frozen=True)
class Glyph:
    char: str
    width: int
    lead: int        # blank columns before the first ink column
    cols: tuple      # per-column ink bitmask (bit r = row r of the 16-row cell)
    ink: int


class FontBank:
    def __init__(self, mem: GBAMemory):
        offsets = mem.read(SYM["sFontHalfRowOffsets"], 256)
        # LUT index -> 4 pixel values (0 bg, 1 fg, 2 shadow), leftmost pixel first
        lut = [(i // 27, (i // 9) % 3, (i // 3) % 3, i % 3) for i in range(81)]
        self.byte_px = [lut[offsets[b]] for b in range(256)]
        self.by_first_col: Dict[int, List[Glyph]] = {}
        self.space_width = 6
        seen = set()
        for glyphs_sym, widths_sym, tiles, stride in FONTS:
            widths = mem.read(SYM[widths_sym], 256)
            base = SYM[glyphs_sym]
            for gid in range(256):
                ch = CHARMAP.get(gid)
                w = widths[gid]
                if not ch or not w or w > 16:
                    continue
                if gid == 0 and glyphs_sym == "sFontNormalLatinGlyphs":
                    self.space_width = w
                px = self._decode(mem.read(base + stride * gid, stride), tiles)
                mask = (px == 1)[:, :w]
                cols = tuple(int(v) for v in (mask[:BAND].astype(np.int64) * POW[:, None]).sum(0))
                ink = int(mask.sum())
                if not ink:
                    continue
                lead = next(i for i, c in enumerate(cols) if c)
                key = (ch, cols)
                if key in seen:
                    continue
                seen.add(key)
                g = Glyph(ch, w, lead, cols, ink)
                self.by_first_col.setdefault(cols[lead], []).append(g)
        for lst in self.by_first_col.values():
            lst.sort(key=lambda g: -g.ink)

    def _decode(self, raw: bytes, tiles) -> np.ndarray:
        px = np.zeros((16, 16), dtype=np.uint8)
        for src, dx, dy in tiles:
            for r in range(8):
                word = struct.unpack_from("<H", raw, src + 2 * r)[0]
                left, right = self.byte_px[word >> 8], self.byte_px[word & 0xFF]
                px[dy + r, dx:dx + 4] = left
                px[dy + r, dx + 4:dx + 8] = right
        return px


def _window_pixels(mem: GBAMemory, ptr: int, w: int, h: int) -> np.ndarray:
    raw = np.frombuffer(mem.read(ptr, w * h * 32), dtype=np.uint8)
    lo, hi = raw & 0xF, raw >> 4
    nib = np.stack([lo, hi], axis=-1).reshape(h, w, 8, 8)  # tile row, tile col, pixel row, pixel col
    return nib.transpose(0, 2, 1, 3).reshape(h * 8, w * 8)


def _decode_line(colbits: List[int], font: FontBank) -> Tuple[str, int, int, int]:
    """Decode one text row -> (text, matched ink, first x, end x)."""
    out, matched, x, last_end, W = [], 0, 0, None, len(colbits)
    first_x = None
    while x < W:
        c = colbits[x]
        if not c:
            x += 1
            continue
        best = None
        for g in font.by_first_col.get(c, ()):
            origin = x - g.lead
            if origin < 0:
                continue
            seg = colbits[origin:origin + g.width]
            seg = seg + [0] * (g.width - len(seg))
            if tuple(seg) == g.cols and (best is None or g.ink > best[0].ink):
                best = (g, origin)
        if best is None:
            # Touching glyphs (e.g. "tt" crossbars): accept the largest glyph whose
            # ink is fully present, ignoring a neighbour's ink in its columns.
            for g in font.by_first_col.get(c, ()):
                origin = x - g.lead
                seg = colbits[origin:origin + g.width]
                if len(seg) == g.width and g.ink >= 6 and all(s & t == t for s, t in zip(seg, g.cols)):
                    best = (g, origin)
                    break  # candidates are sorted by ink, largest first
        if best is None:
            x += 1
            continue
        g, origin = best
        if last_end is not None and origin - last_end >= max(2, font.space_width - 2):
            out.append(" " * max(1, round((origin - last_end) / font.space_width)))
        out.append(g.char)
        matched += g.ink
        first_x = origin if first_x is None else first_x
        last_end = origin + g.width
        x = last_end
    return "".join(out), matched, first_x or 0, last_end or 0


def _decode_mask(mask: np.ndarray, font: FontBank) -> Tuple[List[Tuple[int, int, int, str]], int]:
    """Decode all text rows of one ink color -> ([(y, x0, x1, text)], matched ink)."""
    H = mask.shape[0]
    padded = np.vstack([mask, np.zeros((BAND, mask.shape[1]), dtype=bool)])
    row_ink = mask.any(1)
    lines, total, y = [], 0, 0
    while y < H:
        if not row_ink[y]:
            y += 1
            continue
        best = None
        for y0 in range(max(0, y - 6), y + 1):
            band = padded[y0:y0 + BAND].astype(np.int64)
            colbits = (band * POW[:, None]).sum(0).tolist()
            text, matched, x0, x1 = _decode_line(colbits, font)
            if best is None or matched > best[2]:
                best = (y0, text, matched, x0, x1)
        y0, text, matched, x0, x1 = best
        if matched:
            lines.append((y0, x0, x1, text))
            total += matched
            y = max(y + 1, y0 + 13)
        else:
            y += 1
    return lines, total


_O_FOR_ZERO = re.compile(r"(?<=[A-Z])0|0(?=[A-Z])")
_CONTINUE_ARROW = re.compile(r"([.!?…’”])\s*v$")


def _join_rows(segments: List[Tuple[int, int, int, str]], space_width: int) -> List[str]:
    """Group segments into rows (same y within 4 px) and join them left to right."""
    rows: List[List[Tuple[int, int, int, str]]] = []
    for seg in sorted(segments):
        if rows and abs(seg[0] - rows[-1][0][0]) <= 4:
            rows[-1].append(seg)
        else:
            rows.append([seg])
    out = []
    for row in rows:
        row.sort(key=lambda t: t[1])
        text, end = "", None
        for _y, x0, x1, t in row:
            if end is not None and x0 - end >= max(2, space_width - 2):
                text += " " * max(1, round((x0 - end) / space_width))
            text += t
            end = x1
        compact = text.replace(" ", "")
        if len(set(compact)) == 1 and (len(compact) > 2 or compact in "jil|"):
            continue  # HP bars / empty party slots drawn as pixels that happen to match a glyph
        text = _O_FOR_ZERO.sub("O", text)  # O and 0 share a glyph shape in some fonts
        text = _CONTINUE_ARROW.sub(r"\1 ▼", text)  # the blinking continue arrow matches "v"
        if text.strip():
            out.append(text)
    return out


@dataclass
class WindowText:
    window_id: int
    left: int
    top: int
    lines: List[str]
    waiting_for_button: bool


def _tilemap_entry(mem: GBAMemory, sbb: int, size: int, col: int, row: int) -> int:
    """Tilemap entry at (col, row) for a text BG of any screen size (32x32 blocks)."""
    wide, tall = size in (1, 3), size in (2, 3)
    col %= 64 if wide else 32
    row %= 64 if tall else 32
    block = (row // 32) * (2 if wide else 1) + col // 32
    return struct.unpack_from("<H", mem.vram, (sbb + block) * 0x800 + 2 * ((row % 32) * 32 + col % 32))[0]


def _scroll(mem: GBAMemory, bg: int, in_battle: bool) -> Tuple[int, int]:
    if in_battle and bg == 0:  # battle code scrolls BG0 through its own variables
        return mem.u16(SYM["gBattle_BG0_X"]), mem.u16(SYM["gBattle_BG0_Y"])
    cfg = SYM["sGpuBgConfigs2"] + 16 * bg
    return mem.u32(cfg + 8) >> 8, mem.u32(cfg + 12) >> 8


def _visible(mem: GBAMemory, bg: int, left: int, top: int, w: int, h: int, base_block: int,
             in_battle: bool) -> bool:
    """The window's BG layer is on, its tiles are in the tilemap, and it lies in the scrolled viewport."""
    if not mem.io16(0x04000000) & (1 << (8 + bg)):
        return False
    bgcnt = mem.io16(0x04000008 + 2 * bg)
    sbb, size = (bgcnt >> 8) & 0x1F, bgcnt >> 14
    width_px, height_px = (512 if size in (1, 3) else 256), (512 if size in (2, 3) else 256)
    sx, sy = _scroll(mem, bg, in_battle)
    # window rectangle relative to the viewport (wrapping around the BG)
    wy = (top * 8 - sy) % height_px
    wx = (left * 8 - sx) % width_px
    if wy >= 160 and wy + h * 8 <= height_px or wx >= 240 and wx + w * 8 <= width_px:
        return False
    for ty, tx in ((0, 0), (h // 2, w // 2), (h - 1, w - 1)):
        if (_tilemap_entry(mem, sbb, size, left + tx, top + ty) & 0x3FF) == ((base_block + ty * w + tx) & 0x3FF):
            return True
    return False


def read_screen_text(mem: GBAMemory, font: FontBank, in_battle: bool = False) -> List[WindowText]:
    out = []
    for wid in range(32):
        bg, left, top, w, h, _pal, base_block, ptr = struct.unpack("<6BHI", mem.read(SYM["gWindows"] + 12 * wid, 12))
        if bg == 0xFF or not ptr or not w or not h or (ptr >> 24) != 2:
            continue
        if not _visible(mem, bg, left, top, w, h, base_block, in_battle):
            continue
        px = _window_pixels(mem, ptr, w, h)
        counts = np.bincount(px.ravel(), minlength=16)
        bg_color = int(counts.argmax())
        decoded = []
        for color in np.argsort(-counts):
            color = int(color)
            if color == bg_color or counts[color] < 6:
                continue
            lines, score = _decode_mask(px == color, font)
            if score:
                decoded.append((score, lines))
        if not decoded:
            continue
        decoded.sort(key=lambda d: -d[0])
        # Main text color first; other colors (▶ cursor, greyed options, speaker names) only
        # where they don't overlap text already decoded.
        segments = list(decoded[0][1])
        for _score, lines in decoded[1:]:
            for seg in lines:
                if all(abs(seg[0] - o[0]) > 4 or seg[2] <= o[1] or seg[1] >= o[2] for o in segments):
                    segments.append(seg)
        printer = SYM["sTextPrinters"] + TEXT_PRINTER_SIZE * wid
        p = mem.read(printer, TEXT_PRINTER_SIZE)
        waiting = bool(p[0x1B]) and p[0x1C] in WAITING_STATES
        lines = _join_rows(segments, font.space_width)
        if lines and (waiting or re.search(r"[a-z]v$", lines[-1])) and lines[-1].endswith("v"):
            lines[-1] = lines[-1][:-1].rstrip() + " ▼"  # continue arrow drawn right after a word
        out.append(WindowText(wid, left, top, lines, waiting))
    out.sort(key=lambda wt: (wt.top, wt.left))
    return out
