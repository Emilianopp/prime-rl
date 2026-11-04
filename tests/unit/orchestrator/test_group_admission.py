"""Atomic group admission: a fresh group opens only with permits for every member, its permits
are reserved on the group, and a dropped group hands the unused ones back."""

import asyncio
import uuid
from types import SimpleNamespace

import pytest

from prime_rl.orchestrator.dispatcher import Dispatcher
from prime_rl.orchestrator.types import GroupState


class FakeEnvs:
    def __init__(self, group_size):
        self._envs = {"craftax": SimpleNamespace(config=SimpleNamespace(group_size=group_size), uses_live_policy=True)}

    @property
    def configs(self):
        return [e.config for e in self._envs.values()]

    def get(self, name):
        return self._envs[name]

    def __iter__(self):
        return iter(self._envs.values())


class FakeSource:
    def __init__(self):
        self.pops = 0

    def next_task(self, *, step):
        self.pops += 1
        return SimpleNamespace(env_name="craftax", task=SimpleNamespace(key=f"t{self.pops}"), step=step, rollouts=None, group_id=None)


def make(group_size=8, max_inflight=12):
    d = Dispatcher(train_envs=FakeEnvs(group_size), eval_envs=None, train_source=FakeSource(), eval_source=None,
                   policy_clients=None, policy=SimpleNamespace(version=0, model_name="m"), progress=SimpleNamespace(step=1),
                   initial_max_inflight=max_inflight, max_inflight_ceiling=max_inflight, dispatch_per_minute=None,
                   max_off_policy_steps=8, run_id="r", run_name=None)
    dispatched = []

    async def stub(group_id, group):  # the real one builds the env request; here only the permit accounting
        await d._take_permit(group)
        dispatched.append(group_id)
        return True

    d.schedule_group_episode = stub
    return d, dispatched


def test_fresh_group_needs_permits_for_every_member():
    d, dispatched = make(group_size=8, max_inflight=12)
    assert asyncio.run(d.try_schedule("train"))  # opens group 1: 8 permits reserved, 1 member dispatched
    gid = dispatched[0]
    assert d.current_inflight == 8 and d.groups[gid].reserved == 7 and d.available_permits == 4
    for _ in range(7):  # the other members dispatch on reserved permits, no new ones
        assert asyncio.run(d.try_schedule("train"))
    assert d.current_inflight == 8 and d.groups[gid].reserved == 0 and d.groups[gid].episodes_to_schedule == 0
    assert d.train_source.pops == 1
    assert not asyncio.run(d.try_schedule("train"))  # 4 permits left < 8: no half group, and the source keeps its node
    assert d.train_source.pops == 1 and d.current_inflight == 8


def test_reserved_members_dispatch_when_permits_are_exhausted():
    d, dispatched = make(group_size=4, max_inflight=4)
    assert asyncio.run(d.try_schedule("train"))
    assert d.available_permits == 0 and d.reserved_work
    for _ in range(3):
        assert asyncio.run(d.try_schedule("train"))
    assert len(dispatched) == 4 and not d.reserved_work


def test_dropping_a_half_dispatched_group_returns_its_reserved_permits():
    d, dispatched = make(group_size=8, max_inflight=16)
    assert asyncio.run(d.try_schedule("train"))
    gid = dispatched[0]
    assert d.current_inflight == 8

    async def drop():
        return await d.drop_group(gid, reason="overload")

    cancelled = asyncio.run(drop())
    assert cancelled == 7 and gid not in d.groups
    assert d.current_inflight == 1  # the 7 reserved permits came back; the dispatched member releases its own when its task ends


def test_legacy_group_without_reservation_still_needs_a_permit():
    d, dispatched = make(group_size=4, max_inflight=1)
    gid = uuid.uuid4()
    d.groups[gid] = GroupState(kind="train", env_name="craftax", task=SimpleNamespace(key="t"), step=1,
                               episodes_to_schedule=2, target_episodes=2)
    assert asyncio.run(d.try_schedule("train")) and d.current_inflight == 1
    assert not asyncio.run(d.try_schedule("train"))  # cap 1: the second member waits for a permit
