"""Poor man's sampling profiler for a process that cannot be ptraced (Mila sets ptrace_scope = 2).

A daemon thread samples ``sys._current_frames()`` (GIL-safe, unlike faulthandler's dumps, which
walk other threads' frames without the GIL and crashed an env worker on Python 3.12) every
``interval`` seconds and every ``flush`` seconds rewrites ``<path>.<pid>`` with cumulative counts
(innermost frame per busy thread sample, frames seen, the most common stacks) and appends one
line per window to ``<path>.<pid>.windows`` (busy share of the main thread, its time by owner
package, and the time the cyclic GC spent in that window) so a slowdown over a long run can be
read off as a trend."""

import collections
import gc
import os
import sys
import threading
import time
import traceback

_IDLE = ("selectors.py", "threading.py:", "queue.py", "concurrent/futures/thread.py", "/socket.py", "/ssl.py",
         "multiprocessing/connection.py", "clients/train.py:224")  # :224 = waiting for the renderer lock
_OWNERS = ("openai/", "httpx/", "httpcore/", "aiohttp/", "interception/", "clients/", "harness", "rollout.py",
           "graph.py", "serve/", "renderers/", "transformers/", "tokenizers/", "crafter_env/", "crafter/", "pydantic",
           "json/", "asyncio/", "typing.py")


def _frames(frame, limit=25):
    out = []
    for fr, line in traceback.walk_stack(frame):
        out.append(f"{fr.f_code.co_filename}:{line} {fr.f_code.co_name}")
        if len(out) >= limit:
            break
    return out


def _short(s: str) -> str:
    for marker in ("/site-packages/", "/src/", "/python3.12/"):
        if marker in s:
            s = s.split(marker, 1)[1]
    return s


def _owner(frames) -> str:
    """The innermost frame's package family (asyncio/typing only when nothing else is on the stack)."""
    for f in frames:
        for o in _OWNERS[:-2]:
            if o in f:
                return o
    for f in frames:
        for o in _OWNERS[-2:]:
            if o in f:
                return o
    return "other"


def start(path: str, interval: float = 0.25, flush: float = 30.0) -> None:
    out = f"{path}.{os.getpid()}"
    inner = collections.Counter()
    seen = collections.Counter()
    stacks = collections.Counter()
    gc_time = {"total": 0.0, "start": 0.0, "gen2": 0, "gen2_time": 0.0}
    state = {"samples": 0, "busy": 0, "started": time.time()}
    main_tid = threading.main_thread().ident

    def on_gc(phase, info):
        if phase == "start":
            gc_time["start"] = time.perf_counter()
        else:
            dt = time.perf_counter() - gc_time["start"]
            gc_time["total"] += dt
            if info.get("generation") == 2:
                gc_time["gen2"] += 1
                gc_time["gen2_time"] += dt

    gc.callbacks.append(on_gc)

    def run():
        me = threading.get_ident()
        last = time.time()
        w_samples = w_main_busy = 0
        w_owner = collections.Counter()
        w_gc = w_gc2 = 0.0
        w_gen2 = 0
        while True:
            time.sleep(interval)
            try:
                for tid, frame in sys._current_frames().items():
                    if tid == me:
                        continue
                    fs = _frames(frame)
                    if not fs or any(m in fs[0] for m in _IDLE):
                        continue
                    state["busy"] += 1
                    inner[_short(fs[0])] += 1
                    for f in set(fs):
                        seen[_short(f)] += 1
                    stacks[" <- ".join(_short(f) for f in fs[:14])] += 1
                    if tid == main_tid:
                        w_main_busy += 1
                        w_owner[_owner(fs)] += 1
                state["samples"] += 1
                w_samples += 1
                if time.time() - last >= flush:
                    now = time.time()
                    window = now - last
                    last = now
                    with open(out, "w") as fh:
                        fh.write(f"pid {os.getpid()} samples {state['samples']} busy-thread-samples {state['busy']} "
                                 f"elapsed {now - state['started']:.0f}s gc {gc_time['total']:.1f}s "
                                 f"(gen2 {gc_time['gen2']} collections, {gc_time['gen2_time']:.1f}s) "
                                 f"threshold {gc.get_threshold()} counts {gc.get_count()}\n\n== innermost frames\n")
                        for k, n in inner.most_common(30):
                            fh.write(f"{n:6d}  {k}\n")
                        fh.write("\n== frames present anywhere in the stack\n")
                        for k, n in seen.most_common(50):
                            fh.write(f"{n:6d}  {k}\n")
                        fh.write("\n== most common stacks (innermost first)\n")
                        for k, n in stacks.most_common(10):
                            fh.write(f"{n:6d}  " + "\n        ".join(k.split(" <- ")) + "\n\n")
                    owners = " ".join(f"{k}={n / max(w_samples, 1):.2f}" for k, n in w_owner.most_common(8))
                    with open(out + ".windows", "a") as fh:
                        fh.write(f"{time.strftime('%H:%M:%S')} window {window:.0f}s main-busy {w_main_busy / max(w_samples, 1):.2f} "
                                 f"gc {gc_time['total'] - w_gc:.1f}s gen2 {gc_time['gen2'] - w_gen2}x {gc_time['gen2_time'] - w_gc2:.1f}s "
                                 f"| {owners}\n")
                    w_gc, w_gc2, w_gen2 = gc_time["total"], gc_time["gen2_time"], gc_time["gen2"]
                    w_samples = w_main_busy = 0
                    w_owner = collections.Counter()
            except Exception:  # never let the profiler hurt the process
                pass

    threading.Thread(target=run, name="crafter-sampler", daemon=True).start()
