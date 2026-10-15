"""diagnostics.py

Diagnostic AC pre-cooling start-time sweep on a deterministic workday.

Implements Point 1:
  - Forces day_type = DAY_WORKDAY and zero schedule jitter (schedule_jitter_max_steps = 0).
  - Prints day_type and activity per step for all 48 steps.
  - Sweeps AC start time over [14:00, 15:00, 16:00, 17:00, 18:00] (AC fan-low until 23:00,
    sensible lighting rule).
  - Evaluates energy (kWh), cost (INR), D_T, and total reward for each, identifying the best start time.
  - Prints per-time-block breakdown for the best start time and the 18:00 baseline run.
"""

from __future__ import annotations

import sys
from typing import Any
import numpy as np

from config import (
    ACTIVITIES,
    ACTIVITY_ASLEEP,
    ACTIVITY_AWAKE,
    ACTIVITY_AWAY,
    DAY_TYPES,
    DAY_WORKDAY,
    EnvConfig,
    LIGHT_OFF,
    LIGHT_ON,
    MODE_AC_FAN_LOW,
    MODE_FAN_LOW,
    MODE_OFF,
)
from env import RoomEnv, encode_action
from schedule import generate_daily_schedule, is_dark


def run_workday_simulation(
    ac_start_hour: float,
    seed: int = 100,
    config: EnvConfig | None = None,
) -> dict[str, Any]:
    """Simulate a single workday episode with AC starting at a specified hour.

    Args:
        ac_start_hour: Hour of day when AC is switched on (e.g. 14.0 .. 18.0).
        seed: Seed for reproducible exogenous noise.
        config: EnvConfig instance with zero jitter.

    Returns:
        dict[str, Any]: Aggregated and per-block metrics over the 48-step day.
    """
    cfg = config if config is not None else EnvConfig(schedule_jitter_max_steps=0)
    env = RoomEnv(cfg)
    env.reset(seed=seed)

    # Strictly force workday and deterministic schedule with zero jitter
    env.day_type = DAY_WORKDAY
    env.activity_schedule = generate_daily_schedule(DAY_WORKDAY, env.rng, cfg)

    num_blocks = int(round(24.0 / cfg.time_block_hours))
    block_energy = [0.0] * num_blocks
    block_dt = [0.0] * num_blocks
    block_rewards = [0.0] * num_blocks
    block_activities = [[] for _ in range(num_blocks)]

    total_energy = 0.0
    total_cost = 0.0
    total_d_t = 0.0
    total_d_l = 0.0
    total_reward = 0.0
    switches = 0
    t_return_temp = 0.0

    ac_start_step = int(round(ac_start_hour / cfg.dt))
    return_step = int(round(cfg.workday_return / cfg.dt))

    truncated = False
    step = 0

    while not truncated and step < cfg.episode_steps:
        hour = step * cfg.dt
        block_idx = min(int(hour // cfg.time_block_hours), num_blocks - 1)
        activity = int(env.activity_schedule[step])
        block_activities[block_idx].append(activity)
        dark = is_dark(hour, cfg)

        # Light policy: light ON when awake and dark, else OFF
        light = LIGHT_ON if (activity == ACTIVITY_AWAKE and dark) else LIGHT_OFF

        # Cooling policy:
        # - While away before ac_start_step: OFF
        # - From ac_start_step onward until 23:00 (sleep): AC_FAN_LOW
        # - While asleep (>= 23:00 or < 07:00): FAN_LOW
        if step < ac_start_step:
            cooling = MODE_OFF
        elif hour < cfg.workday_sleep and activity != ACTIVITY_ASLEEP:
            cooling = MODE_AC_FAN_LOW
        else:  # Asleep
            cooling = MODE_FAN_LOW

        action = encode_action(cooling, light)
        _, reward, _, truncated, info = env.step(action)

        step_energy = info["energy_kwh"]
        step_dt = info["D_T"]

        total_energy += step_energy
        total_cost += info["cost"]
        total_d_t += step_dt
        total_d_l += info["D_L"]
        total_reward += reward

        block_energy[block_idx] += step_energy
        block_dt[block_idx] += step_dt
        block_rewards[block_idx] += reward

        if info["switched"]:
            switches += 1

        if step == return_step:
            t_return_temp = info["indoor_temp"]

        step += 1

    modal_activity = []
    for acts in block_activities:
        if acts:
            modal_activity.append(ACTIVITIES[max(set(acts), key=acts.count)])
        else:
            modal_activity.append("unknown")

    return {
        "ac_start_hour": ac_start_hour,
        "energy_kwh": total_energy,
        "cost_inr": total_cost,
        "discomfort_dt": total_d_t,
        "discomfort_dl": total_d_l,
        "total_reward": total_reward,
        "switches": switches,
        "temp_at_return": t_return_temp,
        "block_energy": block_energy,
        "block_dt": block_dt,
        "block_rewards": block_rewards,
        "modal_activity": modal_activity,
        "schedule": env.activity_schedule.copy(),
    }


def print_step_schedule(schedule: np.ndarray, config: EnvConfig) -> None:
    """Print the exact activity per step for the deterministic workday."""
    print("=" * 80)
    print(" WORKDAY SCHEDULE PER STEP (Zero Jitter)")
    print(f" Day Type: {DAY_TYPES[DAY_WORKDAY]} (0)")
    print("=" * 80)
    print(f"{'Step':<6} | {'Hour':<8} | {'Activity ID':<12} | {'Activity Name':<14} | {'Dark':<6}")
    print("-" * 55)
    for s in range(config.episode_steps):
        hour = s * config.dt
        act_id = int(schedule[s])
        act_name = ACTIVITIES[act_id]
        dark = is_dark(hour, config)
        print(f"{s:<6} | {hour:04.1f}h  | {act_id:<12} | {act_name:<14} | {str(dark):<6}")
    print("-" * 55)


def main() -> None:
    cfg = EnvConfig(schedule_jitter_max_steps=0)

    # 1. Sweep AC start hour over 14:00, 15:00, 16:00, 17:00, 18:00
    sweep_hours = [14.0, 15.0, 16.0, 17.0, 18.0]

    sweep_results = []
    for h in sweep_hours:
        res = run_workday_simulation(ac_start_hour=h, seed=100, config=cfg)
        sweep_results.append(res)

    # Print schedule once
    print_step_schedule(sweep_results[0]["schedule"], cfg)

    # 2. Print AC Start Time Sweep Results
    print("\n" + "=" * 90)
    print(" AC PRE-COOLING START TIME SWEEP (Workday, Zero Jitter, Seed=100)")
    print("=" * 90)
    header = f"{'AC Start Hour':<15} | {'Energy (kWh)':<15} | {'Cost (INR)':<15} | {'Discomfort D_T':<18} | {'Total Reward':<16}"
    print(header)
    print("-" * 90)

    best_res = sweep_results[0]
    for r in sweep_results:
        line = (
            f"{r['ac_start_hour']:>5.1f}:00         | "
            f"{r['energy_kwh']:>6.3f} kWh      | "
            f"INR {r['cost_inr']:>6.2f}     | "
            f"{r['discomfort_dt']:>8.3f}         | "
            f"{r['total_reward']:>8.2f}"
        )
        print(line)
        if r["total_reward"] > best_res["total_reward"]:
            best_res = r

    print("-" * 90)
    print(f" BEST AC START TIME: {best_res['ac_start_hour']:.1f}:00 with Total Reward = {best_res['total_reward']:.2f}")
    print("=" * 90)

    # 3. Per-Time-Block Breakdown for Best Run vs 18:00 Baseline
    res_18 = next(r for r in sweep_results if r["ac_start_hour"] == 18.0)
    num_blocks = len(best_res["block_energy"])

    print(f"\n PER-TIME-BLOCK BREAKDOWN: Best ({best_res['ac_start_hour']:.1f}:00) vs Baseline (18:00)")
    print("-" * 95)
    print(f"{'Block':<6} | {'Hours':<11} | {'Activity':<8} | {'E Best (kWh)':<13} | {'E 18:00 (kWh)':<13} | {'D_T Best':<11} | {'D_T 18:00':<11} | {'Delta D_T':<10}")
    print("-" * 95)
    for b in range(num_blocks):
        h_start = b * cfg.time_block_hours
        h_end = (b + 1) * cfg.time_block_hours
        h_str = f"{h_start:04.1f}-{h_end:04.1f}"
        act = best_res["modal_activity"][b]
        e_best = best_res["block_energy"][b]
        e_18 = res_18["block_energy"][b]
        dt_best = best_res["block_dt"][b]
        dt_18 = res_18["block_dt"][b]
        diff_dt = dt_best - dt_18

        print(f" {b:<5} | {h_str:<11} | {act:<8} | {e_best:>11.3f}   | {e_18:>11.3f}   | {dt_best:>9.2f}   | {dt_18:>9.2f}   | {diff_dt:>+9.2f}")
    print("-" * 95)


if __name__ == "__main__":
    main()
