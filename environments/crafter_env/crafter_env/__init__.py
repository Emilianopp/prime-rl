import os

from crafter_env.taskset import CrafterEnv, CrafterTaskset

__all__ = ["CrafterEnv", "CrafterTaskset"]

# Poor man's sampling profiler for an env worker that cannot be ptraced (Mila sets
# ptrace_scope = 2): CRAFTER_STACK_DUMP=<file> makes every process that imports this
# package sample its threads' Python stacks (sys._current_frames, GIL-safe) and write
# aggregated counts to <file>.<pid> every CRAFTER_STACK_DUMP_S seconds. (faulthandler's
# dump_traceback_later crashed the worker on Python 3.12: it walks frames without the GIL.)
if os.environ.get("CRAFTER_STACK_DUMP"):
    from crafter_env._sampler import start as _start_sampler

    _start_sampler(os.environ["CRAFTER_STACK_DUMP"], flush=float(os.environ.get("CRAFTER_STACK_DUMP_S", "30")))
