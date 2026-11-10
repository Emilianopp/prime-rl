"""PopulationSampler: fresh seeds up to the population size, best-branch continuation,
recycling of weak mature lineages while the young share is short, retries, resume."""

from __future__ import annotations

import os
from types import SimpleNamespace

from prime_rl.configs.orchestrator import PopulationSamplerConfig
from prime_rl.orchestrator.curriculum.samplers.population import PopulationSampler


class FakeTask:
    def __init__(self, seed: int, info: dict | None = None, name: str | None = None):
        self.data = SimpleNamespace(info=info if info is not None else {"seed": seed}, name=name or f"craftax#{seed}")

    def with_data(self, info, name):
        return FakeTask(info["seed"], info, name)


def _sampler(tmp_path, **kw) -> PopulationSampler:
    cfg = PopulationSamplerConfig(state_dir=str(tmp_path), **kw)
    return PopulationSampler(cfg, [FakeTask(s) for s in range(100)])


def _group(task, tmp_path, outcomes):
    """outcomes: (reward, points, continues) per branch; each branch writes its save."""
    episodes = []
    for reward, points, continues in outcomes:
        path = os.path.join(str(tmp_path), f"{os.urandom(6).hex()}.pkl")
        open(path, "w").close()
        trace = SimpleNamespace(
            has_error=False,
            info={"craftax": {"state_path": path, "continues": continues, "points": points, "ended_by": "segment_end",
                              "game_steps": 100},
                  "score": {"reward": reward, "end": {"Q": points, "G_star": 5.0}}},
        )
        episodes.append(SimpleNamespace(task=task, ok=True, traces=[trace]))
    return episodes


def test_fills_population_then_waits(tmp_path):
    s = _sampler(tmp_path, lineages=3)
    tasks = [next(s) for _ in range(3)]
    assert all(t is not None for t in tasks)
    assert {t.data.info["tree"]["seed"] for t in tasks} == {0, 1, 2}
    assert next(s) is None  # every game in flight


def test_best_branch_continues_and_others_deleted(tmp_path):
    s = _sampler(tmp_path, lineages=1)
    task = next(s)
    group = _group(task, tmp_path, [(0.1, 1, True), (0.5, 3, True), (-0.2, 0, True), (0.3, 2, True)])
    s.observe(group)
    paths = [ep.traces[0].info["craftax"]["state_path"] for ep in group]
    assert [os.path.exists(p) for p in paths] == [False, True, False, False]
    nxt = next(s)
    node = nxt.data.info["tree"]
    assert node["depth"] == 1 and node["load"] == paths[1] and node["start_score"]["Q"] == 3
    # the child's group consumes the parent's save
    s.observe(_group(nxt, tmp_path, [(0.0, 3, True)] * 4))
    assert not os.path.exists(paths[1])


def test_recycles_weak_mature_lineage_when_young_share_is_short(tmp_path):
    s = _sampler(tmp_path, lineages=4, grace=1, window=1, young_frac=0.5, cull_frac=0.5, min_mature=4)
    tasks = {t.data.info["tree"]["lineage"]: t for t in (next(s) for _ in range(4))}
    ids = sorted(tasks)
    # first segment: everyone reaches grace (depth 1); no one young is left
    gains = {ids[0]: 5, ids[1]: 4, ids[2]: 3, ids[3]: 0}
    for lid in ids[:3]:
        s.observe(_group(tasks[lid], tmp_path, [(0.0, gains[lid], True)] * 4))
    assert len(s.lineages) == 4
    # the weakest (0 points) reports last: mature 4, young 0 < 2 -> bottom half -> recycled
    s.observe(_group(tasks[ids[3]], tmp_path, [(0.0, 0, True)] * 4))
    assert ids[3] not in s.lineages and s.counts["recycled"] == 1
    fresh = [next(s) for _ in range(4)]
    depths = sorted(t.data.info["tree"]["depth"] for t in fresh if t is not None)
    assert depths == [0, 1, 1, 1]  # three continuations and one fresh seed


def test_strong_lineage_is_not_recycled(tmp_path):
    s = _sampler(tmp_path, lineages=4, grace=1, window=1, young_frac=0.5, cull_frac=0.5, min_mature=2)
    tasks = {t.data.info["tree"]["lineage"]: t for t in (next(s) for _ in range(4))}
    ids = sorted(tasks)
    for lid, g in zip(ids, [0, 1, 2, 9]):
        s.observe(_group(tasks[lid], tmp_path, [(0.0, g, True)] * 4))
    assert ids[3] in s.lineages  # top progress survives


def test_retry_then_end(tmp_path):
    s = _sampler(tmp_path, lineages=1, max_retries=1)
    task = next(s)
    s.observe(_group(task, tmp_path, [(None, 0, False)] * 4))  # no branch can continue
    assert s.counts["retried"] == 1
    again = next(s)
    assert again.data.info["tree"]["depth"] == 0
    s.observe(_group(again, tmp_path, [(None, 0, False)] * 4))
    assert s.counts["ended"] == 1 and not s.lineages


def test_stale_attempt_ignored_and_resume(tmp_path):
    s = _sampler(tmp_path, lineages=2)
    a, b = next(s), next(s)
    s.observe(_group(a, tmp_path, [(0.2, 1, True)] * 4))
    s.observe(_group(a, tmp_path, [(0.9, 9, True)] * 4))  # duplicate result of an old attempt
    assert s.lineages[a.data.info["tree"]["lineage"]]["points"] == [0.0, 1.0]
    s2 = PopulationSampler(s.config, [FakeTask(x) for x in range(100)])
    assert len(s2.lineages) == 2
    assert all(lin["status"] == "ready" for lin in s2.lineages.values())  # b was in flight: replayed
    assert s2.root_cursor == 2
