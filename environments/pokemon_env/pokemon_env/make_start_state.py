"""Play FireRed's new-game intro once and save a start state in the player's bedroom.

    python -m pokemon_env.make_start_state --rom $POKEMON_FIRERED_ROM --out firered_start.state

Uses the default player name, rival GREEN, and sets the in-game options text speed
FAST and battle animations OFF (both choices a player can make in OPTION).
"""

import argparse
import os

from pokemon_env.firered.emulator import FireRedGame
from pokemon_env.firered.intro import play_intro


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rom", default=os.environ.get("POKEMON_FIRERED_ROM"), required=not os.environ.get("POKEMON_FIRERED_ROM"))
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    play_intro(FireRedGame(rom_path=args.rom, state_path=None), args.out)


if __name__ == "__main__":
    main()
