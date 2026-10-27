"""One FireRed game per process, driven by JSON lines on stdin/stdout.

mGBA's Python bindings route every emulated frame through cffi "extern Python"
callbacks, whose start-up spins forever on Python >= 3.12. The env therefore can
run each episode's emulator in this worker under a Python where they work
(3.10/3.11 with `mgba` importable) — see `PokemonEnvConfig.emulator_python`.
A process per episode also keeps emulation off the env server's GIL.

Requests: {"cmd": "start", "kwargs": {...}} | {"cmd": "reset", "idle_frames": k}
| {"cmd": "step", "buttons": [...]} | {"cmd": "summary"} | {"cmd": "close"}.
Every reply is one JSON line: {"ok": true, ...} or {"ok": false, "error": "..."}.
"""

import json
import sys
import traceback


def handle(game, request):
    """Shared by the worker and the in-process game: one request -> one reply dict."""
    cmd = request["cmd"]
    if cmd == "reset":
        return {"observation": game.reset(idle_frames=request.get("idle_frames", 0))}
    if cmd == "step":
        loc_before, xy_before = game.state.location_name(), game.state.player_xy()
        observation, step = game.step(request["buttons"])
        return {
            "observation": observation,
            "frames": step.frames,
            "location": step.location,
            "coords": list(step.coords),
            "location_before": loc_before,
            "coords_before": list(xy_before),
            "new_milestones": step.new_milestones,
        }
    if cmd == "summary":
        return {"milestones": dict(game.milestones), "party": game.state.party()}
    raise ValueError(f"unknown command {cmd!r}")


def main():
    from pokemon_env.firered.emulator import FireRedGame

    out = sys.stdout
    sys.stdout = sys.stderr  # stray prints must not corrupt the protocol
    game = None
    for line in sys.stdin:
        request = json.loads(line)
        try:
            if request["cmd"] == "start":
                game = FireRedGame(**request["kwargs"])
                reply = {}
            elif request["cmd"] == "close":
                out.write(json.dumps({"ok": True}) + "\n")
                out.flush()
                return
            else:
                reply = handle(game, request)
            reply["ok"] = True
        except Exception as e:  # report, keep serving: the env decides what to do
            reply = {"ok": False, "error": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc()}
        out.write(json.dumps(reply) + "\n")
        out.flush()


if __name__ == "__main__":
    main()
