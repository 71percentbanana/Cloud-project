"""Regression tests for the simulator fixes made during the October 2026 audit."""

import copy

import numpy as np

from src.models.device import VolunteerDevice, DeviceType, PowerState
from src.models.task import Task, ReplicaInstance, ExecutionRole
from src.scheduling.schedulers import (
    CapabilityOnlyScheduler,
    IndependenceReplicationScheduler,
    AdaptiveStabilityScheduler,
)
from src.simulation.environment import SimulationEnvironment
from src.simulation.generator import generate_heterogeneous_population, generate_workload


def _device(i, **kw):
    base = dict(device_id=i, device_type=DeviceType.DESKTOP, cores=5, ram_gb=16.0,
                uplink_mbps=100.0, baseline_mtbf=1e9, group_id=None)
    base.update(kw)
    return VolunteerDevice(**base)


def test_battery_exhaustion_causes_real_outage():
    rng = np.random.default_rng(0)
    d = _device(0, device_type=DeviceType.LAPTOP, power_state=PowerState.BATTERY,
                battery_level=0.01, mean_repair_time=50.0)
    env = SimulationEnvironment([d], [], CapabilityOnlyScheduler(), total_steps=1, seed=0)
    env.rng = rng
    env._update_devices()
    assert not d.is_available
    env._update_devices()
    assert not d.is_available, "device must stay down for its repair time, not recover next step"


def test_unschedulable_task_waits_instead_of_vanishing():
    d = _device(0, is_available=False)
    d.time_until_next_event = 3.0
    task = Task(task_id=0, req_cores=2, req_ram_gb=4.0, req_uplink_mbps=1.0,
                expected_duration_steps=12.0, deadline_factor=3.0, arrival_step=0.0)
    env = SimulationEnvironment([d], [task], CapabilityOnlyScheduler(), seed=0)
    pct, *_ = env.run()
    assert pct == 100.0


def test_wasted_cpu_credits_checkpointed_progress():
    r = ReplicaInstance(replica_id="p", device_id=0, start_progress=0.0, current_progress=0.8,
                        cpu_time_spent=800.0, banked_progress=0.5)
    assert abs(r.wasted_cpu() - 300.0) < 1e-6
    r.banked_progress = 0.0
    assert r.wasted_cpu() == 800.0


def test_restart_without_checkpoint_starts_from_zero():
    task = Task(task_id=0, expected_duration_steps=10.0, deadline_factor=3.0)
    task.completed_progress = 0.7
    task.replicas.append(ReplicaInstance(replica_id="p", device_id=0, current_progress=0.7, active=False))
    d = _device(0)
    env = SimulationEnvironment([d], [], CapabilityOnlyScheduler(), total_steps=1, seed=0)
    env.running_tasks = [task]
    task.state = task.state.RUNNING
    env._step_simulation()
    restarted = [r for r in task.replicas if r.active]
    assert restarted and restarted[0].start_progress == 0.0


def test_baselines_do_not_use_cloud_unless_enabled():
    for seed in range(3):
        rng = np.random.default_rng(seed)
        devices = generate_heterogeneous_population(10, "high", rng=rng)
        tasks = generate_workload(rng=np.random.default_rng(seed))
        for sched in (CapabilityOnlyScheduler(), IndependenceReplicationScheduler()):
            _, _, cloud, *_ = SimulationEnvironment(copy.deepcopy(devices), copy.deepcopy(tasks), sched, seed=seed).run()
            assert cloud == 0.0
    assert CapabilityOnlyScheduler(use_cloud_fallback=True).use_cloud_fallback
    assert AdaptiveStabilityScheduler().name == "Proposed"


def test_run_lasts_until_last_deadline():
    tasks = generate_workload(rng=np.random.default_rng(1))
    env = SimulationEnvironment([_device(0)], tasks, CapabilityOnlyScheduler(), seed=0)
    assert env.total_steps > max(t.deadline_step for t in tasks)
    assert 100 <= len(tasks) <= 210  # ~151 arrivals in one week


def test_device_runs_one_replica_at_a_time():
    import collections
    devices = generate_heterogeneous_population(30, "high", rng=np.random.default_rng(1000))
    tasks = generate_workload(rng=np.random.default_rng(1000))
    for sched in (CapabilityOnlyScheduler(), IndependenceReplicationScheduler(), AdaptiveStabilityScheduler()):
        env = SimulationEnvironment(copy.deepcopy(devices), copy.deepcopy(tasks), sched, seed=0)
        step = env._step_simulation

        def checked_step():
            step()
            per_device = collections.Counter(
                r.device_id for t in env.running_tasks for r in t.replicas if r.active and not r.is_cloud)
            assert max(per_device.values(), default=0) <= 1

        env._step_simulation = checked_step
        env.run()
