"""The orchestrator checkpoint carries the saved games its snapshot points at."""

import os

import torch

from prime_rl.configs.orchestrator import CheckpointConfig
from prime_rl.orchestrator.ckpt import CheckpointManager, saved_game_paths


def test_checkpoint_bundles_referenced_saved_games(tmp_path):
    games = tmp_path / "states"
    games.mkdir()
    for name in ("a.pkl", "b.pkl", "unreferenced.pkl"):
        (games / name).write_bytes(name.encode())
    state = {"progress": {"step": 3}, "train_source": {"rng": (3, (1,), None), "envs": {"craftax": {"sampler": {
        "frontier": [{"id": "x", "state_path": str(games / "a.pkl")}],
        "levels": [{"root": "r", "depth": 4, "candidates": [], "inflight": [
            {"id": "y", "state_path": str(games / "b.pkl")}, {"id": "root", "state_path": None}]}],
    }}}}}
    assert saved_game_paths(state) == {str(games / "a.pkl"), str(games / "b.pkl")}
    manager = CheckpointManager(tmp_path / "run", CheckpointConfig(interval=1))
    manager.save_state(state, step=3)
    bundle = manager.get_ckpt_path(3) / "states"
    assert sorted(os.listdir(bundle)) == ["a.pkl", "b.pkl"] and (bundle / "b.pkl").read_bytes() == b"b.pkl"
    assert torch.load(manager.get_ckpt_path(3) / "progress.pt", weights_only=False)["progress"]["step"] == 3


def test_checkpoint_survives_a_saved_game_already_gone(tmp_path):
    state = {"train_source": {"envs": {"e": {"frontier": [{"state_path": str(tmp_path / "gone.pkl")}]}}}}
    manager = CheckpointManager(tmp_path / "run", CheckpointConfig(interval=1))
    manager.save_state(state, step=1)
    assert (manager.get_ckpt_path(1) / "progress.pt").exists() and os.listdir(manager.get_ckpt_path(1) / "states") == []
