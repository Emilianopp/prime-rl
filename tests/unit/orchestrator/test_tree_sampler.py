"""The tree sampler: nodes out, branches in, a beam per root and depth, states deleted."""

import os
import uuid

import pytest
import verifiers.v1 as vf
from pydantic import ValidationError

from prime_rl.configs.orchestrator import CurriculumConfig, TreeSamplerConfig
from prime_rl.orchestrator.curriculum import Curriculum
from prime_rl.orchestrator.curriculum.samplers import TreeSampler


class Data(vf.TaskData):
    info: dict


def make_root(i: int) -> vf.Task:
    return vf.Task(Data(idx=i, name=f"crafter#train{i}", system_prompt="You are playing Crafter.", info={"seed": 42 + i}))


def branch(task: vf.Task, tmp_path, *, score: float, continues: bool = True, final: bool = False, ok: bool = True) -> vf.Episode:
    tid = uuid.uuid4().hex
    path = None
    if continues or final:
        path = str(tmp_path / f"{tid}.pkl")
        with open(path, "wb") as f:
            f.write(b"world")
    trace = vf.Trace(
        task=vf.TraceTask(type=type(task).__name__, data=task.data, key=task.key, hash=task.hash),
        agent=vf.AgentInfo(config=vf.AgentConfig()),
        nodes=[vf.MessageNode(message=vf.AssistantMessage(content="answer"), sampled=True)],
        rewards={"judge": vf.Reward(score=score)},
        ok=ok,
    )
    trace.info["crafter"] = {
        "node_id": tid, "continues": continues and not final, "final_goal_reached": final, "state_path": path,
        "state_sha256": "x", "summary": f"notes {score}", "achievements": ["collect_wood"] + (["collect_diamond"] if final else []),
        "game_step": 100,
    }
    return vf.Episode(env=vf.EnvInfo(id="crafter", name="crafter"), task=trace.task, traces=[trace], ok=ok)


def test_beam_keep_one_expands_the_best_branch_and_deletes_the_rest(tmp_path):
    sampler = TreeSampler(TreeSamplerConfig(keep=1, max_depth=3, state_dir=str(tmp_path)), [make_root(0), make_root(1)])
    root = next(sampler)
    node = root.data.info["node"]
    assert node["depth"] == 0 and root.data.system_prompt == "You are playing Crafter." and sampler.roots_started == 1
    group = [branch(root, tmp_path, score=s) for s in (0.1, 0.7, 0.3, 0.7)]
    sampler.observe(group)
    assert len(sampler.frontier) == 1 and sampler.frontier[0]["score"] == 0.7
    kept = sampler.frontier[0]["state_path"]
    assert os.path.exists(kept) and sum(os.path.exists(e.traces[0].info["crafter"]["state_path"]) for e in group) == 1
    child = next(sampler)
    assert child.data.info["node"]["depth"] == 1 and child.data.info["node"]["state_path"] == kept
    assert child.data.system_prompt.startswith("You are playing Crafter.\n\n") and child.data.system_prompt.endswith("notes 0.7")
    assert child.data.name == "crafter#train0/d1/" + child.data.info["node"]["id"] and child.key != root.key
    assert sampler.metrics()["frontier"] == 0 and sampler.metrics()["deepest_depth"] == 1
    # Nothing ready while the child is in flight: the next task is a new root.
    other = next(sampler)
    assert other.data.info["node"]["depth"] == 0 and other.data.info["seed"] == 43 and sampler.roots_started == 2
    # The child's parent state is deleted once its group reports.
    sampler.observe([branch(child, tmp_path, score=0.5)])
    assert not os.path.exists(kept) and sampler.frontier[0]["depth"] == 2


def test_beam_keep_two_waits_for_the_depth_then_releases_greedily(tmp_path):
    sampler = TreeSampler(TreeSamplerConfig(keep=2, max_depth=5), [make_root(0)])
    root = next(sampler)
    sampler.observe([branch(root, tmp_path, score=s) for s in (0.2, 0.9, 0.4, 0.6)])
    a, b = next(sampler), next(sampler)  # the depth's two best, ranked
    assert [a.data.info["node"]["score"], b.data.info["node"]["score"]] == [0.9, 0.6]
    assert a.data.info["node"]["lineage"] == [root.data.info["node"]["id"], a.data.info["node"]["id"]]
    # Depth 2 expects two groups. The first arrives; nothing is closed yet, so a request is served greedily.
    sampler.observe([branch(a, tmp_path, score=s) for s in (0.3, 0.8, 0.1, 0.5)])
    assert sampler.frontier == [] and sampler.levels[(a.data.info["node"]["root"], 2)]["observed"] == 1
    greedy = next(sampler)
    assert greedy.data.info["node"]["depth"] == 2 and greedy.data.info["node"]["score"] == 0.8
    # The second group arrives with a better branch: the remaining beam slot goes to it.
    sampler.observe([branch(b, tmp_path, score=s) for s in (0.95, 0.2, 0.2, 0.2)])
    assert [n["score"] for n in sampler.frontier] == [0.95] and (b.data.info["node"]["root"], 2) not in sampler.levels


def test_final_goal_stops_the_lineage_and_is_recorded(tmp_path):
    sampler = TreeSampler(TreeSamplerConfig(keep=1, max_depth=10, state_dir=str(tmp_path)), [make_root(0), make_root(1)])
    root = next(sampler)
    sampler.observe([branch(root, tmp_path, score=0.5), branch(root, tmp_path, score=1.0, final=True)])
    assert sampler.frontier == [] and len(sampler.found) == 1 and sampler.found[0]["depth"] == 1
    assert sampler.metrics()["deepest_tier"] == 9.0 and not list(tmp_path.glob("*.pkl"))
    assert next(sampler).data.info["seed"] == 43  # the next root


def test_final_goal_at_depth_three_records_the_full_lineage(tmp_path):
    sampler = TreeSampler(TreeSamplerConfig(keep=1, max_depth=10, state_dir=str(tmp_path)), [make_root(0)])
    task = next(sampler)
    path = [task.data.info["node"]["id"]]
    for _ in range(2):
        sampler.observe([branch(task, tmp_path, score=0.2), branch(task, tmp_path, score=0.8)])
        task = next(sampler)
        path.append(task.data.info["node"]["id"])
        assert task.data.info["node"]["lineage"] == path
    final = branch(task, tmp_path, score=1.0, final=True)
    sampler.observe([branch(task, tmp_path, score=0.3), final])
    assert len(sampler.found) == 1 and sampler.found[0]["depth"] == 3
    assert sampler.found[0]["lineage"] == [*path, final.traces[0].info["crafter"]["node_id"]]


def test_max_depth_leaves_and_errored_branches(tmp_path):
    sampler = TreeSampler(TreeSamplerConfig(keep=1, max_depth=1, state_dir=str(tmp_path)), [make_root(0)])
    root = next(sampler)
    sampler.observe([branch(root, tmp_path, score=0.9), branch(root, tmp_path, score=0.1, ok=False)])
    assert sampler.frontier == [] and not list(tmp_path.glob("*.pkl"))  # depth 1 is a leaf: nothing to expand


def test_checkpoint_resume_continues_the_frontier(tmp_path):
    config = CurriculumConfig(sampler=TreeSamplerConfig(keep=1, max_depth=4))
    roots = [make_root(i) for i in range(3)]
    curriculum = Curriculum(config, roots)
    root = next(curriculum.sampler)
    curriculum.on_result([branch(root, tmp_path, score=0.4), branch(root, tmp_path, score=0.6)])
    state = curriculum.state_dict()
    resumed = Curriculum(config, roots)
    resumed.load_state_dict(state)
    child = next(resumed.sampler)
    assert child.data.info["node"]["depth"] == 1 and child.data.info["node"]["score"] == 0.6
    assert child.data.info["seed"] == 42 and next(resumed.sampler).data.info["seed"] == 43


def test_max_roots_waits_for_an_open_world_instead_of_starting_one(tmp_path):
    sampler = TreeSampler(TreeSamplerConfig(keep=1, max_depth=100, max_roots=2), [make_root(i) for i in range(4)])
    a, b = next(sampler), next(sampler)
    assert sampler.roots_started == 2 and sampler.metrics()["open_roots"] == 2
    assert next(sampler) is None  # both worlds in flight, the cap is reached
    sampler.observe([branch(a, tmp_path, score=s) for s in (0.2, 0.6)])
    child = next(sampler)  # the open world goes deeper; no third root
    assert child.data.info["node"]["depth"] == 1 and child.data.info["node"]["root"] == a.data.info["node"]["root"]
    assert next(sampler) is None and sampler.roots_started == 2
    # A world that ends (no branch continues) frees its slot for a new root.
    sampler.observe([branch(b, tmp_path, score=0.1, continues=False)])
    fresh = next(sampler)
    assert fresh.data.info["node"]["depth"] == 0 and sampler.roots_started == 3


def test_checkpoint_while_a_node_is_in_flight_replays_it_on_resume(tmp_path):
    # With max_roots and keep=1 every open world is usually in flight when a checkpoint is taken: the
    # frontier is empty and the next depth has no candidates yet. A resume must play those nodes again.
    config = CurriculumConfig(sampler=TreeSamplerConfig(keep=1, max_depth=100, max_roots=2, state_dir=str(tmp_path)))
    roots = [make_root(i) for i in range(4)]
    curriculum = Curriculum(config, roots)
    a, b = next(curriculum.sampler), next(curriculum.sampler)
    curriculum.on_result([branch(a, tmp_path, score=s) for s in (0.3, 0.8)])
    child = next(curriculum.sampler)  # world a at depth 1, in flight
    sent = child.data.info["node"]
    assert curriculum.sampler.frontier == [] and os.path.exists(sent["state_path"])
    curriculum.sampler._sweep()  # an in-flight node's saved game is referenced: the sweep keeps it
    assert os.path.exists(sent["state_path"])
    state = curriculum.state_dict()
    resumed = Curriculum(config, roots)
    resumed.load_state_dict(state)
    assert resumed.sampler.metrics()["replayed_on_resume"] == 2 and resumed.sampler.metrics()["open_roots"] == 2
    again = [next(resumed.sampler), next(resumed.sampler)]
    nodes = sorted((t.data.info["node"] for t in again), key=lambda n: n["depth"])
    # world b's root and world a's depth-1 node are played again; no new world starts
    assert [n["depth"] for n in nodes] == [0, 1] and nodes[1]["state_path"] == sent["state_path"] and nodes[1]["score"] == 0.8
    assert next(resumed.sampler) is None and resumed.sampler.roots_started == 2
    # the replayed node reports and its world goes deeper as before
    replay = next(t for t in again if t.data.info["node"]["depth"] == 1)
    resumed.on_result([branch(replay, tmp_path, score=0.5)])
    assert next(resumed.sampler).data.info["node"]["depth"] == 2


def test_checkpoint_without_inflight_field_still_loads(tmp_path):
    config = CurriculumConfig(sampler=TreeSamplerConfig(keep=1, max_depth=10))
    roots = [make_root(i) for i in range(2)]
    curriculum = Curriculum(config, roots)
    next(curriculum.sampler)
    state = curriculum.state_dict()
    for level in state["sampler"]["levels"]:
        level.pop("inflight", None)  # an older checkpoint
    resumed = Curriculum(config, roots)
    resumed.load_state_dict(state)
    assert resumed.sampler.frontier == [] and next(resumed.sampler).data.info["seed"] == 43


def test_depth_schedule_gives_each_world_the_cap_of_the_step_it_started(tmp_path):
    config = TreeSamplerConfig(keep=1, max_depth=1, max_depth_schedule=[(2, 3)], state_dir=str(tmp_path))
    sampler = TreeSampler(config, [make_root(i) for i in range(3)])
    old = next(sampler)  # step 0, before the first entry: max_depth
    assert old.data.info["node"]["max_depth"] == 1 and sampler.metrics()["max_depth_now"] == 1
    sampler.set_step(2)
    assert sampler.metrics()["max_depth_now"] == 3
    # The world started at step 0 keeps its cap of 1: its depth-1 branches are leaves.
    sampler.observe([branch(old, tmp_path, score=s) for s in (0.2, 0.6)])
    assert sampler.frontier == []
    new = next(sampler)  # its slot starts a new world, on the new cap
    assert new.data.info["node"]["depth"] == 0 and new.data.info["node"]["max_depth"] == 3
    task = new
    for depth in (1, 2):
        sampler.observe([branch(task, tmp_path, score=0.5)])
        task = next(sampler)
        assert task.data.info["node"]["depth"] == depth and task.data.info["node"]["max_depth"] == 3
    sampler.observe([branch(task, tmp_path, score=0.5)])  # depth 3, its cap: a leaf
    assert sampler.frontier == [] and not list(tmp_path.glob("*.pkl"))
    assert next(sampler).data.info["node"]["depth"] == 0 and sampler.roots_started == 3


def test_depth_schedule_cap_survives_a_checkpoint(tmp_path):
    config = CurriculumConfig(sampler=TreeSamplerConfig(keep=1, max_depth=1, max_depth_schedule=[(0, 3)]))
    roots = [make_root(i) for i in range(2)]
    curriculum = Curriculum(config, roots)
    root = next(curriculum.sampler)
    curriculum.on_result([branch(root, tmp_path, score=0.5)])
    # Resumed without the schedule: the open world still plays to the cap it started with.
    resumed = Curriculum(CurriculumConfig(sampler=TreeSamplerConfig(keep=1, max_depth=1)), roots)
    resumed.load_state_dict(curriculum.state_dict())
    child = next(resumed.sampler)
    assert child.data.info["node"]["depth"] == 1 and child.data.info["node"]["max_depth"] == 3
    resumed.on_result([branch(child, tmp_path, score=0.5)])
    assert next(resumed.sampler).data.info["node"]["depth"] == 2


def test_depth_schedule_must_increase():
    with pytest.raises(ValidationError):
        TreeSamplerConfig(max_depth_schedule=[(50, 25), (10, 50)])
    with pytest.raises(ValidationError):
        TreeSamplerConfig(max_depth_schedule=[(0, 0)])
    assert TreeSamplerConfig(max_depth_schedule=[[0, 10], [50, 25]]).max_depth_schedule == [(0, 10), (50, 25)]


def test_train_source_passes_the_step_to_the_depth_schedule():
    from types import SimpleNamespace

    from prime_rl.orchestrator.train_source import TrainSource

    roots = [make_root(i) for i in range(3)]
    sampler_config = TreeSamplerConfig(keep=1, max_depth=10, max_depth_schedule=[(0, 10), (50, 25), (100, 50)])
    config = SimpleNamespace(ratio=1.0, curriculum=CurriculumConfig(sampler=sampler_config))
    source = TrainSource([SimpleNamespace(name="craftax", tasks=roots, num_tasks=len(roots), config=config)])
    caps = [source.next_task(step=step).task.data.info["node"]["max_depth"] for step in (3, 60, 120)]
    assert caps == [10, 25, 50]


# --- selection = "puct" (TTT-Discover's reuse rule) ---------------------------------------------


def valued(task: vf.Task, tmp_path, *, value: float, reward: float = 0.0) -> vf.Episode:
    episode = branch(task, tmp_path, score=reward)
    episode.traces[0].info["crafter"]["search_value"] = value
    return episode


def puct_sampler(tmp_path, **kw) -> TreeSampler:
    config = TreeSamplerConfig(selection="puct", state_dir=str(tmp_path), value_key="search_value",
                               **{"fresh_root_share": 0.0, "max_depth": 5, "puct_scope": "archive", **kw})
    return TreeSampler(config, [make_root(i) for i in range(4)])


def state_of(episode: vf.Episode) -> str:
    return episode.traces[0].info["crafter"]["state_path"]


def test_puct_archives_the_best_branches_by_state_value_and_keeps_the_parent(tmp_path):
    sampler = puct_sampler(tmp_path)
    root = next(sampler)  # an empty archive starts a world
    assert root.data.info["node"]["depth"] == 0 and sampler.fresh_issued == 1
    group = [valued(root, tmp_path, value=v, reward=r) for v, r in ((1.0, 0.9), (5.0, 0.1), (3.0, 0.5), (2.0, 0.7))]
    sampler.observe(group)
    assert sorted(n["score"] for n in sampler.archive.values()) == [3.0, 5.0]  # the value, not the reward
    assert [os.path.exists(state_of(e)) for e in group] == [False, True, True, False]
    assert sampler.puct_expansions == 0  # a fresh root is not an archive expansion
    first = next(sampler)
    node = first.data.info["node"]
    assert node["depth"] == 1 and node["state_path"] == state_of(group[1]) and first.data.system_prompt.endswith("notes 0.1")
    children = [valued(first, tmp_path, value=v) for v in (4.0, 6.0, 0.5)]
    sampler.observe(children)
    parent = sampler.archive[node["id"]]
    assert os.path.exists(parent["state_path"])  # the game before a death can be played again
    assert parent["n"] == 1 and parent["m"] == 6.0 and sampler.puct_expansions == 1
    assert len(sampler.archive) == 4 and not os.path.exists(state_of(children[2]))
    assert sampler.metrics()["archive"] == 4 and sampler.metrics()["archive_unvisited"] == 3


def test_puct_comes_back_to_the_parent_when_its_branches_lost_everything(tmp_path):
    sampler = puct_sampler(tmp_path, puct_c=0.0)
    root = next(sampler)
    sampler.observe([valued(root, tmp_path, value=v) for v in (5.0, 3.0)])
    best = next(sampler)
    assert best.data.info["node"]["score"] == 5.0
    sampler.observe([valued(best, tmp_path, value=v) for v in (1.0, 0.5)])  # both branches died: tools gone
    # Q(best) is now its best branch (1.0): the untouched 3.0 node is next, and the dead branches are last.
    picks = []
    for _ in range(2):
        task = next(sampler)
        picks.append(task.data.info["node"]["score"])
        sampler.observe([valued(task, tmp_path, value=task.data.info["node"]["score"] - 0.1)])
    assert picks[0] == 3.0 and 0.5 not in picks


def test_puct_exploration_bonus_spreads_the_picks(tmp_path):
    def picks(c: float) -> set[float]:
        sampler = puct_sampler(tmp_path, puct_c=c, puct_keep_children=4, max_depth=2)
        root = next(sampler)
        sampler.observe([valued(root, tmp_path, value=v) for v in (5.0, 4.9, 4.8, 4.7)])
        seen = set()
        for _ in range(12):
            task = next(sampler)
            seen.add(task.data.info["node"]["score"])
            sampler.observe([valued(task, tmp_path, value=6.0)])  # leaves at the cap: every node's Q becomes 6
        return seen

    assert picks(0.0) == {5.0}  # greedy: the first node to show a 6 is expanded for ever
    # The bonus is in units of the archive's value range (0.3 here, the leaves' 6 is outside it), so it takes a
    # large c to beat a gap of 1.1; it shrinks with a node's visits, so the under-visited nodes get their turn.
    assert len(picks(20.0)) >= 3


def test_puct_blocks_the_lineage_in_flight_and_starts_a_world_when_nothing_is_free(tmp_path):
    sampler = puct_sampler(tmp_path)
    root = next(sampler)
    sampler.observe([valued(root, tmp_path, value=v) for v in (5.0, 3.0)])
    a, b = next(sampler), next(sampler)
    assert {a.data.info["node"]["score"], b.data.info["node"]["score"]} == {5.0, 3.0}  # the same node is not sent twice
    assert next(sampler).data.info["node"]["depth"] == 0  # both in flight: a new world
    sampler.observe([valued(a, tmp_path, value=7.0)])
    child = next(sampler)  # a's branch is free; a itself is free again too
    assert child.data.info["node"]["parent"] == a.data.info["node"]["id"]
    assert next(sampler).data.info["node"]["depth"] == 0  # a is an ancestor of the node in flight, b is in flight


def test_puct_fresh_root_share(tmp_path):
    sampler = puct_sampler(tmp_path, fresh_root_share=0.5, puct_keep_children=4)
    depths = []
    for _ in range(40):
        task = next(sampler)
        depths.append(task.data.info["node"]["depth"])
        sampler.observe([valued(task, tmp_path, value=float(len(depths)))])
    assert 18 <= sum(d == 0 for d in depths) <= 22 and sampler.metrics()["fresh_root_share"] == pytest.approx(0.5, abs=0.05)


def test_puct_archive_size_evicts_the_lowest_values_and_their_states(tmp_path):
    sampler = puct_sampler(tmp_path, archive_size=3, puct_keep_children=4)
    root = next(sampler)
    group = [valued(root, tmp_path, value=v) for v in (1.0, 2.0, 3.0, 4.0)]
    sampler.observe(group)
    assert sorted(n["score"] for n in sampler.archive.values()) == [2.0, 3.0, 4.0]
    assert not os.path.exists(state_of(group[0])) and sampler.archive_evicted == 1
    assert len(os.listdir(tmp_path)) == 3


def test_puct_depth_cap_and_checkpoint(tmp_path):
    sampler = puct_sampler(tmp_path, max_depth=2)
    root = next(sampler)
    sampler.observe([valued(root, tmp_path, value=1.0)])
    first = next(sampler)
    sampler.observe([valued(first, tmp_path, value=2.0)])  # depth 2 = the cap: a leaf, not archived
    assert len(sampler.archive) == 1
    pending = next(sampler)  # in flight at the checkpoint
    assert pending.data.info["node"]["depth"] == 1
    state = sampler.state_dict()
    resumed = puct_sampler(tmp_path, max_depth=2)
    resumed.load_state_dict(state)
    assert set(resumed.archive) == set(sampler.archive) and resumed.puct_expansions == 1 and not resumed.puct_inflight
    assert next(resumed).data.info["node"]["id"] == pending.data.info["node"]["id"]  # free again after a resume


def test_beam_is_unchanged_by_the_value_key(tmp_path):
    sampler = TreeSampler(TreeSamplerConfig(keep=1, max_depth=3, state_dir=str(tmp_path), value_key="search_value"), [make_root(0)])
    root = next(sampler)
    sampler.observe([valued(root, tmp_path, value=9.0, reward=0.1), valued(root, tmp_path, value=1.0, reward=0.8)])
    assert sampler.frontier[0]["score"] == 0.8 and not sampler.archive


def world_sampler(tmp_path, **kw) -> TreeSampler:
    config = TreeSamplerConfig(selection="puct", state_dir=str(tmp_path), value_key="search_value",
                               **{"fresh_root_share": 0.5, "max_depth": 50, "max_roots": 3, **kw})
    return TreeSampler(config, [make_root(i) for i in range(200)])


def run_worlds(sampler: TreeSampler, tmp_path, steps: int, value) -> list[vf.Task]:
    """One expansion at a time; `value(task)` gives the two branch values."""
    issued = []
    for _ in range(steps):
        task = next(sampler)
        issued.append(task)
        sampler.observe([valued(task, tmp_path, value=v) for v in value(task)])
    return issued


def test_puct_world_scope_spreads_expansions_over_worlds_whatever_their_value(tmp_path):
    sampler = world_sampler(tmp_path, world_patience=1000)
    # world 0 is worth 100x the others and every step adds value: the archive-wide rule would only pick it
    def value(task):
        node = task.data.info["node"]
        gain = 100.0 if node["root"].startswith("0-") or node["root"] == sampler_first[0] else 1.0
        return (gain * (node["depth"] + 1), gain * (node["depth"] + 1) - 0.5)
    sampler_first = [None]
    first = next(sampler)
    sampler_first[0] = first.data.info["node"]["root"]
    sampler.observe([valued(first, tmp_path, value=v) for v in (100.0, 99.5)])
    issued = run_worlds(sampler, tmp_path, 60, value)
    assert len(sampler.worlds) == 3
    counts = [w["n"] for w in sampler.worlds.values()]
    assert max(counts) - min(counts) <= 1  # level, not by value
    picked = [t.data.info["node"]["root"] for t in issued if t.data.info["node"]["depth"] > 0]
    assert len(set(picked)) == 3


def test_puct_world_scope_one_expansion_in_flight_per_world(tmp_path):
    sampler = world_sampler(tmp_path, fresh_root_share=0.0, max_roots=2)
    roots = [next(sampler), next(sampler)]  # an empty archive starts worlds
    for task in roots:
        sampler.observe([valued(task, tmp_path, value=v) for v in (2.0, 1.0)])
    a, b = next(sampler), next(sampler)
    assert a.data.info["node"]["depth"] == 1 and b.data.info["node"]["depth"] == 1
    assert a.data.info["node"]["root"] != b.data.info["node"]["root"]
    assert next(sampler).data.info["node"]["depth"] == 0  # both worlds busy: a new world is played instead


def test_puct_world_scope_retires_a_stalled_world_and_admits_a_new_one(tmp_path):
    sampler = world_sampler(tmp_path, fresh_root_share=0.0, max_roots=1, world_patience=3)
    root = next(sampler)
    first = root.data.info["node"]["root"]
    group = [valued(root, tmp_path, value=v) for v in (5.0, 4.0)]
    sampler.observe(group)
    kept = [state_of(e) for e in group]
    for _ in range(3):  # no expansion beats 5.0
        task = next(sampler)
        assert task.data.info["node"]["root"] == first
        sampler.observe([valued(task, tmp_path, value=v) for v in (1.0, 0.5)])
    assert first not in sampler.worlds and sampler.worlds_retired == 1 and not sampler.archive
    assert not any(os.path.exists(path) for path in kept)
    task = next(sampler)
    assert task.data.info["node"]["depth"] == 0
    sampler.observe([valued(task, tmp_path, value=v) for v in (2.0, 1.0)])
    assert list(sampler.worlds) == [task.data.info["node"]["root"]]


def test_puct_world_scope_full_archive_plays_new_worlds_without_keeping_them(tmp_path):
    sampler = world_sampler(tmp_path, fresh_root_share=1.0, max_roots=1)
    root = next(sampler)
    sampler.observe([valued(root, tmp_path, value=v) for v in (5.0, 4.0)])
    extra = next(sampler)
    group = [valued(extra, tmp_path, value=v) for v in (9.0, 8.0)]
    sampler.observe(group)
    assert len(sampler.worlds) == 1 and len(sampler.archive) == 2
    assert not any(os.path.exists(state_of(e)) for e in group)


def test_puct_world_scope_checkpoint_round_trip(tmp_path):
    sampler = world_sampler(tmp_path)
    run_worlds(sampler, tmp_path, 12, lambda task: (task.data.info["node"]["depth"] + 1.0, 0.5))
    restored = world_sampler(tmp_path)
    restored.load_state_dict(sampler.state_dict())
    assert restored.worlds == sampler.worlds and set(restored.archive) == set(sampler.archive)
