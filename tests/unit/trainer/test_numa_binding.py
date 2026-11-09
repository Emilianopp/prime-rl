import sys
import types
from pathlib import Path

import pytest

from prime_rl.trainer import utils


@pytest.fixture
def numa(monkeypatch):
    """GPU 0 on NUMA node 1 (CPUs 8-15); returns the CPU set the binding sets."""
    fake_nvml = types.SimpleNamespace(
        nvmlInit=lambda: None,
        nvmlShutdown=lambda: None,
        nvmlDeviceGetHandleByIndex=lambda index: index,
        nvmlDeviceGetPciInfo=lambda handle: types.SimpleNamespace(busId=b"00000000:41:00.0"),
    )
    monkeypatch.setitem(sys.modules, "pynvml", fake_nvml)
    monkeypatch.setattr(utils.torch.cuda, "current_device", lambda: 0)
    sysfs = {
        "/sys/bus/pci/devices/0000:41:00.0/numa_node": "1\n",
        "/sys/devices/system/node/node1/cpulist": "8-15\n",
    }
    monkeypatch.setattr(Path, "read_text", lambda self: sysfs[str(self)])
    bound = []
    monkeypatch.setattr(utils.os, "sched_setaffinity", lambda pid, cpus: bound.append(set(cpus)))
    return bound


def test_binds_to_the_allowed_cpus_of_the_gpus_node(numa, monkeypatch):
    monkeypatch.setattr(utils.os, "sched_getaffinity", lambda pid: set(range(4)) | {10, 11})
    utils.bind_process_to_gpu_numa_node()
    assert numa == [{10, 11}]


def test_skips_binding_when_no_cpu_of_the_gpus_node_is_allowed(numa, monkeypatch):
    # A Slurm cpuset holding only node 0's CPUs: binding to node 1 would fail with EINVAL.
    monkeypatch.setattr(utils.os, "sched_getaffinity", lambda pid: set(range(8)))
    utils.bind_process_to_gpu_numa_node()
    assert numa == []
