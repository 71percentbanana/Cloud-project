"""
Experiment 1: Main Simulation Results (Table 1).
Compares 4 scheduling methods across Stable, Moderate, and High instability conditions.
Paper Section 14 & Table 1.
"""

import copy
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
from src.simulation.generator import generate_heterogeneous_population, generate_workload
from src.simulation.environment import SimulationEnvironment
from src.simulation.metrics import MetricCollector
from src.simulation.parallel import run_seeds
from src.scheduling.schedulers import (
    CapabilityOnlyScheduler,
    CapabilityStabilityScheduler,
    IndependenceReplicationScheduler,
    AdaptiveStabilityScheduler,
)


def run_main_results_experiment(num_seeds: int = 300, num_devices: int = 30, num_tasks: Optional[int] = None,
                                cloud_baselines: bool = False) -> pd.DataFrame:
    """
    Executes Table 1 experiment:
    Conditions: Stable, Moderate, High
    Methods: Capability only, Capability + Stability, Independence replication, Proposed

    With cloud_baselines=True the three baselines may also resume lost tasks in the
    cloud (from their latest checkpoint, if any), giving an equal-cloud-access comparison.
    """
    conditions = ["Stable", "Moderate", "High"]
    rows = []
    
    for cond in conditions:
        instability_key = cond.lower()
        
        schedulers = [
            CapabilityOnlyScheduler(use_cloud_fallback=cloud_baselines),
            CapabilityStabilityScheduler(use_cloud_fallback=cloud_baselines),
            IndependenceReplicationScheduler(use_cloud_fallback=cloud_baselines),
            AdaptiveStabilityScheduler(),
        ]
        
        for sched in schedulers:
            collector = run_seeds(sched, instability_key, num_devices, seed_offset=1000, num_seeds=num_seeds)
            summary = collector.summarize()
            
            rows.append({
                "Condition": cond,
                "Method": sched.name,
                "Deadline (%)": f"{summary['deadline_pct'].mean:.2f} ± {summary['deadline_pct'].ci95:.2f}",
                "Wasted CPU": f"{int(round(summary['wasted_cpu'].mean))} ± {int(round(summary['wasted_cpu'].ci95))}",
                "Cloud (inst-sec)": f"{int(round(summary['cloud_inst_sec'].mean))}" if summary['cloud_inst_sec'].mean > 0 else "–",
                "Deadline_Mean": summary['deadline_pct'].mean,
                "Deadline_CI95": summary['deadline_pct'].ci95,
                "WastedCPU_Mean": summary['wasted_cpu'].mean,
                "WastedCPU_CI95": summary['wasted_cpu'].ci95,
                "Cloud_Mean": summary['cloud_inst_sec'].mean,
                "Cloud_CI95": summary['cloud_inst_sec'].ci95,
                "Replicas_Mean": summary['replica_launches'].mean,
            })
            
    df = pd.DataFrame(rows)
    return df


def run_tuned_proposed_experiment(num_seeds: int = 300, num_devices: int = 30, r2: float = 0.40, r3: float = 0.70) -> pd.DataFrame:
    """
    Evaluate the proposed scheduler with hedging thresholds chosen from the threshold-sensitivity
    experiment. That experiment uses a separate seed range (5000+), so this run on the main seeds
    (1000+) is an out-of-sample check of the chosen operating point.
    """
    from src.config import ExecutionThresholds
    rows = []
    for cond in ["Stable", "Moderate", "High"]:
        sched = AdaptiveStabilityScheduler(thresholds=ExecutionThresholds(R1=0.05, R2=r2, R3=r3))
        summary = run_seeds(sched, cond.lower(), num_devices, seed_offset=1000, num_seeds=num_seeds).summarize()
        rows.append({
            "Condition": cond,
            "Method": f"Proposed (R2={r2:.2f}, R3={r3:.2f})",
            "Deadline_Mean": summary['deadline_pct'].mean,
            "Deadline_CI95": summary['deadline_pct'].ci95,
            "WastedCPU_Mean": summary['wasted_cpu'].mean,
            "WastedCPU_CI95": summary['wasted_cpu'].ci95,
            "Cloud_Mean": summary['cloud_inst_sec'].mean,
            "Cloud_CI95": summary['cloud_inst_sec'].ci95,
            "Replicas_Mean": summary['replica_launches'].mean,
        })
    return pd.DataFrame(rows)
