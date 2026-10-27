"""Play FireRed's new-game intro and save a state in the player's bedroom.

Uses the decoded screen text to make choices: Start on the title screen,
NEW GAME, the default boy character, the default player name (OK on an empty keyboard) and the first preset rival name,
YES to confirmations, and A through all of Oak's speech. Then sets the in-game
options text speed FAST and battle animations OFF before saving.
"""

from __future__ import annotations

import os
import re
import time

from pokemon_env.firered.screen_text import read_screen_text

TARGET = "PalletTown_PlayersHouse_2F"


def set_fast_options(env):
    """In-game OPTION settings a player can pick: text speed FAST, battle animations OFF."""
    opts = env.state.sb2 + 0x14
    m = env.core.memory.u8
    m[opts] = (m[opts] & ~0x7) | 2          # optionsTextSpeed = OPTIONS_TEXT_SPEED_FAST
    m[opts + 1] = m[opts + 1] | 0x4         # optionsBattleSceneOff = 1
    env.mem.refresh()
    assert env.mem.u8(opts) & 0x7 == 2 and env.mem.u8(opts + 1) & 0x4, "option write did not stick"


def play_intro(env, out_path: str, max_presses: int = 3000):
    env.core.reset()
    env._run(10)
    t0 = time.perf_counter()
    picked_names = 0
    for i in range(max_presses):
        env.mem.refresh()
        st = env.state
        if st.mode() == "overworld" and st.sb1 >> 24 == 2 and st.location_name() == TARGET:
            windows = read_screen_text(env.mem, env.font)
            if not windows:
                set_fast_options(env)
                env._settle()
                env.save_state(out_path)
                print(f"saved {out_path} after {i} presses, {time.perf_counter() - t0:.1f}s; "
                      f"location {st.location_name()} at {st.player_xy()}")
                return
        text = " | ".join(l for w in read_screen_text(env.mem, env.font) for l in w.lines)
        if re.search(r"▶\s*NEW NAME", text):
            env._press("down")        # cursor from NEW NAME to the first preset name
            env._settle()
            picked_names += 1
        elif "NAME?" in text and "A B C" in text:
            env._press("start")       # naming keyboard: jump to OK, keep the default name
            env._settle()
        env._press("start" if i < 40 and not text else "a")
        env._settle()
        if i % 50 == 0 or os.environ.get("INTRO_DEBUG"):
            print(f"[{i}] mode={st.mode()} text={text[:100]!r}")
    raise RuntimeError("intro did not reach the player's room; inspect with `play --state ''`")
