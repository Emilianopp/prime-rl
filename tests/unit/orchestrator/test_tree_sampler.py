"""The tree sampler: nodes out, branches in, a beam per root and depth, states deleted."""

import os
import uuid

import verifiers.v1 as vf

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
