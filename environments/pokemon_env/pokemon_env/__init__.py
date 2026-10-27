# Lazy, so the emulator worker (pokemon_env.firered.worker) can run under a Python
# without verifiers installed.
__all__ = ["PokemonEnv", "PokemonTaskset"]


def __getattr__(name):
    if name in __all__:
        from pokemon_env import taskset

        return getattr(taskset, name)
    raise AttributeError(f"module 'pokemon_env' has no attribute {name!r}")
