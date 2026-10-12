"""
Experiment 6: Scaling with Volunteer Population (Table 6).
Evaluates proposed scheduler under High instability across 10, 20, 30, 40, and 50 devices.
Paper Section 18 & Table 6.
"""

import copy
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
from src.simulation.generator import generate_heterogeneous_population, generate_workload
from src.simulation.environment import SimulationEnvironment
from src.simulation.metrics import MetricCollector
from src.simulation.parallel import run_seeds
from src.scheduling.schedulers import AdaptiveStabilityScheduler


def run_scaling_experiment(num_seeds: int = 300, num_tasks: Optional[int] = None) -> pd.DataFrame:
    """
    Executes Table 6 experiment:
    Devices: 10, 20, 30, 40, 50
    """
    device_counts = [10, 20, 30, 40, 50]
    rows = []
    
    for n_dev in device_counts:
        sched = AdaptiveStabilityScheduler()
        collector = run_seeds(sched, "high", n_dev, seed_offset=6000, num_seeds=num_seeds)
        summary = collector.summarize()
        
        rows.append({
            "Devices": n_dev,
            "Deadline (%)": f"{summary['deadline_pct'].mean:.2f} ± {summary['deadline_pct'].ci95:.2f}",
            "Wasted CPU": f"{int(round(summary['wasted_cpu'].mean))} ± {int(round(summary['wasted_cpu'].ci95))}",
            "Cloud": f"{int(round(summary['cloud_inst_sec'].mean))} ± {int(round(summary['cloud_inst_sec'].ci95))}",
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
