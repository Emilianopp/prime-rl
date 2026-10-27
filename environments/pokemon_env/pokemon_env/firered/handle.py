"""Async handles on one FireRed game: in this process, or in a worker process.

Both expose the same coroutines: start, reset, step, summary, close.
"""

import asyncio
import json
import os
import sys
from pathlib import Path

from pokemon_env.firered.worker import handle


class LocalGame:
    """The emulator in this process (needs a Python where mGBA's bindings run: <= 3.11)."""

    def __init__(self):
        self.game = None

    async def start(self, **kwargs):
        from pokemon_env.firered.emulator import FireRedGame

        self.game = await asyncio.to_thread(FireRedGame, **kwargs)

    async def request(self, **request) -> dict:
        return await asyncio.to_thread(handle, self.game, request)

    async def close(self):
        if self.game is not None:
            await asyncio.to_thread(self.game.close)
            self.game = None


class RemoteGame:
    """The emulator in a worker process running `python`."""

    def __init__(self, python: str, extra_path: str = ""):
        self.python, self.extra_path, self.proc = python, extra_path, None

    async def start(self, **kwargs):
        package_root = str(Path(__file__).resolve().parents[2])  # environments/pokemon_env
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join(p for p in (package_root, self.extra_path) if p)
        self.proc = await asyncio.create_subprocess_exec(
            self.python, "-s", "-m", "pokemon_env.firered.worker",
            stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=sys.stderr,
            env=env, limit=1 << 24,
        )
        await self.request(cmd="start", kwargs=kwargs)

    async def request(self, **request) -> dict:
        self.proc.stdin.write((json.dumps(request) + "\n").encode())
        await self.proc.stdin.drain()
        line = await self.proc.stdout.readline()
        if not line:
            raise RuntimeError(f"pokemon worker exited (code {await self.proc.wait()}) during {request['cmd']!r}")
        reply = json.loads(line)
        if not reply.pop("ok"):
            raise RuntimeError(f"pokemon worker {request['cmd']!r} failed: {reply['error']}\n{reply.get('traceback', '')}")
        return reply

    async def close(self):
        if self.proc is None or self.proc.returncode is not None:
            return
        try:
            await asyncio.wait_for(self.request(cmd="close"), timeout=10)
            await asyncio.wait_for(self.proc.wait(), timeout=10)
        except Exception:
            self.proc.kill()
            await self.proc.wait()


def open_game(python: str | None, extra_path: str = ""):
    """A worker under `python` when given, else the emulator in this process."""
    return RemoteGame(python, extra_path) if python else LocalGame()
