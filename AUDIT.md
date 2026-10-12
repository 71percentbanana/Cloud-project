# Simulator Audit (October 2026)

Before regenerating every result table, we audited the simulator. This file lists each defect we found, why it mattered, and how it was fixed.

**Rule followed:** we fixed only behaviour that was wrong: it contradicted the paper's stated design, the code's own documentation, or basic accounting. We did not change any parameter to move results toward a desired outcome. All model parameters in `src/config.py` are unchanged.

Every fix except the import error (#1) has a regression test in `tests/test_audit_fixes.py`. All of those tests fail on the original code and pass after the fixes.

## Defects fixed

| # | Defect | Effect on results | Fix |
|---|--------|-------------------|-----|
| 1 | `experiments/co_availability_eval.py` used `Optional` without importing it | `run_experiments.py` crashed at import, so the committed results could not have come from this exact code | Added the import |
| 2 | Battery exhaustion set the laptop unavailable, but the recovery branch made it available again in the same step | Battery death, the main laptop-specific interruption, had no effect | The device stays down for a repair period and returns on AC power |
| 3 | A task that arrived when no capable device was free was removed from the queue and never retried | Tasks vanished and were counted as missed deadlines | The task stays queued until a device frees up or its deadline passes |
| 4 | When every replica of a task was lost, recovery restarted from `completed_progress` even if no checkpoint existed | Baselines kept their lost progress for free (implicit checkpointing) | Without a checkpoint, recovery restarts from 0 |
| 5 | `completed_progress` kept the progress of replicas that had already failed | A later checkpoint could record progress that no surviving execution had reached | Task progress is the maximum over live replicas and the latest checkpoint |
| 6 | Wasted CPU counted all of a failed replica's CPU time, including work saved in a checkpoint and reused by the resumed execution | Checkpointing could never reduce wasted CPU, which contradicts the metric's definition | Work saved in a checkpoint is not counted as wasted; all work on a task that misses its deadline is |
| 7 | Baselines got cloud recovery implicitly through `getattr(scheduler, "use_cloud_fallback", True)` | Contradicted the paper's design, where only the proposed scheduler uses the cloud | Cloud access is an explicit scheduler option. Baselines run without it (Table 1) and with it (Table 1b, equal cloud access) |
| 8 | Correlated group outages in the evaluation used fixed rates (0.5%/step, 15-step duration) that differed from the process that generated the availability history | The co-availability estimator learned from one outage process and was tested on another | Evaluation outages use the same group MTBF and repair time as the history |
| 9 | The run stopped at a fixed 2,880 steps (10 days) with about 216 arrivals; tasks arriving near the end were cut off and counted as failures | Did not match the paper's one-week, 150-task setup, and penalised late arrivals | Arrivals span one week (2,016 steps, about 151 tasks at λ = 0.075), and the run continues until the last deadline |
| 10 | Table 4 selected replicas using a device history and then measured joint interruption on that same history | In-sample evaluation flattered co-availability-aware placement | Placement uses the first half of the history; joint interruption is measured on the held-out second half |
| 11 | A device could run any number of task replicas at once, each at full speed | Capability-only placed 74% of tasks on one device with up to 10 concurrent replicas, so one interruption could kill all of them | Each device runs one replica at a time; tasks wait when no capable device is free |

## Other changes

- `src/simulation/parallel.py` runs seeds in parallel worker processes. Each seed is seeded explicitly, so results match a serial run.
- The main and threshold tables now also report cloud use and replicas per task.
- `run_experiments.py` no longer prints the draft paper's numbers as comparison targets, because no code can reproduce them.

## Known modelling limitations (not changed)

These are properties of the method as described in the paper, so they were reported rather than altered:

- The risk model's session-age factor assumes the hazard grows with uptime, but the simulated interruptions are memoryless.
- The network-instability multiplier (×1.8 for fluctuating or 1 Mbps links) has no counterpart in the simulated interruption process.
- Independence replication is static two-way replication: the primary plus one replica, both launched at dispatch and ranked by recent availability.
