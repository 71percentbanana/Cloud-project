"""
Run independent simulation seeds in parallel worker processes.
Each seed is self-contained and seeded explicitly, so results are identical to a serial run.
"""

import os
from concurrent.futures import ProcessPoolExecutor
from typing import Optional

import numpy as np

from src.simulation.environment import SimulationEnvironment
from src.simulation.generator import generate_heterogeneous_population, generate_workload
from src.simulation.metrics import MetricCollector

_executor: Optional[ProcessPoolExecutor] = None


def _pool() -> ProcessPoolExecutor:
    global _executor
    if _executor is None:
        workers = int(os.environ.get("SIM_WORKERS", os.cpu_count() or 1))
        _executor = ProcessPoolExecutor(max_workers=max(1, workers))
    return _executor


def _run_one(job):
    scheduler, instability, num_devices, seed_offset, seed, shared_rng = job
    if shared_rng:
        rng = np.random.default_rng(seed + seed_offset)
        devices = generate_heterogeneous_population(num_devices=num_devices, instability=instability, rng=rng)
        tasks = generate_workload(rng=rng)
    else:
        devices = generate_heterogeneous_population(
            num_devices=num_devices, instability=instability, rng=np.random.default_rng(seed + seed_offset))
        tasks = generate_workload(rng=np.random.default_rng(seed + seed_offset))
    return SimulationEnvironment(devices=devices, tasks=tasks, scheduler=scheduler, seed=seed).run()


def run_seeds(scheduler, instability: str, num_devices: int, seed_offset: int,
              num_seeds: int, shared_rng: bool = False) -> MetricCollector:
    """Simulate num_seeds independent runs and collect their metrics in seed order."""
    jobs = [(scheduler, instability, num_devices, seed_offset, seed, shared_rng) for seed in range(num_seeds)]
    collector = MetricCollector()
    for result in _pool().map(_run_one, jobs, chunksize=4):
        collector.add_run_result(*result)
    return collector
