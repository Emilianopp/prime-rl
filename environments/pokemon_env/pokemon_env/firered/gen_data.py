"""Generate data.json (addresses + constants for Pokémon FireRed US v1.0, BPRE rev 0).

Source: the pret/pokefirered decompilation (symbols branch for addresses).
Usage:
    python -m pokemon_env.firered.gen_data <pokefirered checkout or fetched files dir>

Expected files under that dir: pokefirered.sym, charmap.txt,
data/maps/map_groups.json, include/constants/metatile_behaviors.h.
"""

import json
import os
import re
import sys

SYMBOLS = [
    # RAM
    "gSaveBlock1Ptr", "gSaveBlock2Ptr", "gPlayerParty", "gPlayerPartyCount", "gMapHeader", "VMap",
    "gObjectEvents", "gPlayerAvatar", "gMain", "gBattleTypeFlags", "gBattleMons", "gBattlersCount",
    "gBattlerPartyIndexes", "gBattleControllerExecFlags", "gBattleOutcome", "gActionSelectionCursor",
    "gMoveSelectionCursor", "gWindows", "sTextPrinters", "gPaletteFade", "gStringVar4",
    "gDisplayedStringBattle", "gBattle_BG0_X", "gBattle_BG0_Y", "sGpuBgConfigs2", "gBattlerControllerFuncs",
    # ROM tables
    "gSpeciesNames", "gMoveNames", "gItems", "gTypeNames", "gSpeciesInfo", "gMapGroups", "sMapNames",
    "sFontHalfRowOffsets",
    "sFontSmallLatinGlyphs", "sFontSmallLatinGlyphWidths",
    "sFontNormalCopy1LatinGlyphs", "sFontNormalCopy1LatinGlyphWidths",
    "sFontNormalLatinGlyphs", "sFontNormalLatinGlyphWidths",
    "sFontMaleLatinGlyphs", "sFontMaleLatinGlyphWidths",
    "sFontFemaleLatinGlyphs", "sFontFemaleLatinGlyphWidths",
    # code (main-loop callbacks used to detect game mode)
    "BattleMainCB2", "CB2_Overworld", "CB2_UpdatePartyMenu", "gPartyMenu",
    # overworld script engine: what the running script is waiting on
    "sGlobalScriptContext", "sGlobalScriptContextStatus",
]
# Native wait functions that mean "a cutscene is still playing" (not waiting for the player).
SCRIPT_BUSY = [
    "WaitForMovementFinish", "IsPaletteNotActive", "IsFreezePlayerFinished", "WaitForSoundEffectFinish",
    "WaitForFieldEffectFinish", "WaitForFanfareFinish", "RunPauseTimer", "IsFreezeSelectedObjectAndPlayerFinished",
    "IsDoorAnimationStopped", "IsCryFinished", "IsBGMPausedOrStopped",
]
# Battle-controller states that wait for the player's input (all copies: player + Oak/old-man controllers).
BATTLE_INPUT = ["HandleInputChooseAction", "HandleInputChooseTarget", "HandleInputChooseMove",
                "HandleMoveSwitching", "WaitForMonSelection", "CompleteWhenChoseItem"]


def main(src):
    syms = {}
    for line in open(os.path.join(src, "pokefirered.sym")):
        parts = line.split()
        if len(parts) == 4 and parts[3] in SYMBOLS + SCRIPT_BUSY and parts[3] not in syms:
            syms[parts[3]] = int(parts[0], 16)
    battle_input = []
    for line in open(os.path.join(src, "pokefirered.sym")):
        parts = line.split()
        if len(parts) == 4 and (parts[3] in BATTLE_INPUT or "HandleInputChoose" in parts[3]):
            battle_input.append([parts[3], int(parts[0], 16)])
    missing = [s for s in SYMBOLS + SCRIPT_BUSY if s not in syms]
    if missing:
        sys.exit(f"missing symbols: {missing}")

    charmap = {}
    for line in open(os.path.join(src, "charmap.txt")):
        m = re.match(r"^'(.+)'\s*=\s*([0-9A-F]{2})\s*$", line.strip())
        if m and m.group(1) != "\\'" and int(m.group(2), 16) not in charmap:
            charmap[int(m.group(2), 16)] = m.group(1)
    charmap[0x53] = "Pk"  # PK / MN ligature glyphs
    charmap[0x54] = "Mn"
    charmap[0x34] = "Lv"

    groups = json.load(open(os.path.join(src, "data/maps/map_groups.json")))
    map_groups = [groups[g] for g in groups["group_order"]]

    behaviors = {}
    for line in open(os.path.join(src, "include/constants/metatile_behaviors.h")):
        m = re.match(r"#define (MB_\w+)\s+(0x[0-9A-Fa-f]+)", line)
        if m:
            behaviors[int(m.group(2), 16)] = m.group(1)

    out = {
        "game": "Pokemon FireRed (US) v1.0 BPRE rev0, sha1 41cb23d8dccc8ebd7c649cd8fbb58eeace6e2fdc",
        "source": "pret/pokefirered (symbols branch + headers)",
        "symbols": {k: syms[k] for k in SYMBOLS},
        "script_busy": {k: syms[k] for k in SCRIPT_BUSY},
        "battle_input": battle_input,
        "charmap": {str(k): v for k, v in sorted(charmap.items())},
        "map_groups": map_groups,
        "metatile_behaviors": {str(k): v for k, v in sorted(behaviors.items())},
        "constants": {
            "MAP_OFFSET": 7, "NUM_METATILES_IN_PRIMARY": 640, "MAPSEC_KANTO_START": 0x58,
            "OBJ_EVENT_GFX_ITEM_BALL": 92, "OBJ_EVENT_GFX_CUT_TREE": 95,
            "OBJ_EVENT_GFX_ROCK_SMASH_ROCK": 96, "OBJ_EVENT_GFX_PUSHABLE_BOULDER": 97,
            "FLAG_BADGE01_GET": 0x820, "SB1_FLAGS": 0xEE0, "SB1_MONEY": 0x290, "SB2_ENCRYPTION_KEY": 0xF20,
            "SB1_BAG_POCKETS": [["Items", 0x310, 42], ["Key Items", 0x3B8, 30], ["Poké Balls", 0x430, 13],
                                ["TMs/HMs", 0x464, 58], ["Berries", 0x54C, 43]],
            "CONNECTIONS": {"1": "south", "2": "north", "3": "west", "4": "east"},
        },
    }
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    print(f"wrote {path}: {len(syms)} symbols, {len(charmap)} chars, {sum(map(len, map_groups))} maps, "
          f"{len(behaviors)} behaviors")


if __name__ == "__main__":
    main(sys.argv[1])
