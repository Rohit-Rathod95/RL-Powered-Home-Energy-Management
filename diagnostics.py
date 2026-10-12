"""diagnostics.py

Diagnostic comparison of pre-cooling policies on a workday.

Compares:
  Policy 1: "AC on from 18:00" (AC starts when household returns).
  Policy 2: "AC on from 17:00" (Pre-cooling 1 hour before household returns).

Evaluates total energy (kWh), electricity cost (₹), thermal discomfort (D_T),
and visual discomfort (D_L) under identical exogenous workday conditions.
Runs once next_temperature() and compute_reward() in dynamics.py are implemented.
"""

from __future__ import annotations

import sys
import numpy as np

from config import (
    ACTIVITY_AWAKE,
    ACTIVITY_AWAY,
    DAY_WORKDAY,
    EnvConfig,
    LIGHT_OFF,
    LIGHT_ON,
    MODE_AC_FAN_LOW,
    MODE_FAN_LOW,
    MODE_OFF,
)
from env import RoomEnv, encode_action
from schedule import is_dark


def run_workday_simulation(ac_start_hour: float, seed: int = 42) -> dict[str, float]:
    """Simulate a single workday episode with AC starting at a specified hour.

    Args:
        ac_start_hour: Hour of day when AC is switched on (e.g., 17.0 or 18.0).
        seed: Seed for reproducible exogenous conditions.

    Returns:
        dict[str, float]: Aggregated metrics over the 48-step day.
    """
    config = EnvConfig()
    env = RoomEnv(config)
    env.reset(seed=seed)

    # Force workday to ensure schedule matches the test scenario
    env.day_type = DAY_WORKDAY

    total_energy = 0.0
    total_cost = 0.0
    total_d_t = 0.0
    total_d_l = 0.0
    switches = 0
    t_return_temp = 0.0

    ac_start_step = int(round(ac_start_hour / config.dt))
    return_step = int(round(config.workday_return / config.dt))

    truncated = False
    step = 0

    while not truncated and step < config.episode_steps:
        hour = step * config.dt
        activity = int(env.activity_schedule[step])
        dark = is_dark(hour, config)

        # Light policy: light ON when awake and dark, else OFF
        light = LIGHT_ON if (activity == ACTIVITY_AWAKE and dark) else LIGHT_OFF

        # Cooling policy:
        # - While away before ac_start_step: OFF
        # - From ac_start_step onward through evening: AC_FAN_LOW
        # - While asleep: FAN_LOW
        if step < ac_start_step:
            cooling = MODE_OFF
        elif activity == ACTIVITY_AWAY or activity == ACTIVITY_AWAKE:
            cooling = MODE_AC_FAN_LOW
        else:  # ACTIVITY_ASLEEP
            cooling = MODE_FAN_LOW

        action = encode_action(cooling, light)
        _, _, _, truncated, info = env.step(action)

        total_energy += info["energy_kwh"]
        total_cost += info["cost"]
        total_d_t += info["D_T"]
        total_d_l += info["D_L"]
        if info["switched"]:
            switches += 1

        if step == return_step:
            t_return_temp = info["indoor_temp"]

        step += 1

    return {
        "ac_start_hour": ac_start_hour,
        "energy_kwh": total_energy,
        "cost_inr": total_cost,
        "discomfort_dt": total_d_t,
        "discomfort_dl": total_d_l,
        "switches": switches,
        "temp_at_return": t_return_temp,
    }


def main() -> None:
    """Run comparative diagnostics between standard cooling and pre-cooling."""
    print("=" * 70)
    print(" Home Electricity Saver: Phase 1 Diagnostics")
    print(" Comparing 'AC on from 17:00' (Pre-cooling) vs 'AC on from 18:00'")
    print("=" * 70)

    try:
        results_18 = run_workday_simulation(ac_start_hour=18.0, seed=100)
        results_17 = run_workday_simulation(ac_start_hour=17.0, seed=100)
    except NotImplementedError as e:
        print("\n[NOTE] Mode B stubs are not yet implemented:")
        print(f"       {e}")
        print("\nImplement next_temperature() and compute_reward() in dynamics.py,")
        print("then rerun this script to see the pre-cooling diagnostic comparison.\n")
        sys.exit(0)

    print(f"{'Metric':<32} | {'AC at 18:00 (Baseline)':<20} | {'AC at 17:00 (Pre-cool)':<20}")
    print("-" * 78)
    print(f"{'Daily Energy (kWh)':<32} | {results_18['energy_kwh']:<20.3f} | {results_17['energy_kwh']:<20.3f}")
    print(f"{'Electricity Cost (INR)':<32} | {results_18['cost_inr']:<20.2f} | {results_17['cost_inr']:<20.2f}")
    print(f"{'Thermal Discomfort (D_T)':<32} | {results_18['discomfort_dt']:<20.3f} | {results_17['discomfort_dt']:<20.3f}")
    print(f"{'Temp at 18:00 return (deg C)':<32} | {results_18['temp_at_return']:<20.2f} | {results_17['temp_at_return']:<20.2f}")
    print(f"{'Cooling Mode Switches':<32} | {results_18['switches']:<20d} | {results_17['switches']:<20d}")
    print("-" * 78)

    delta_cost = results_17["cost_inr"] - results_18["cost_inr"]
    delta_dt = results_17["discomfort_dt"] - results_18["discomfort_dt"]
    print(f"Trade-off: Pre-cooling changed cost by {delta_cost:+.2f} INR and D_T by {delta_dt:+.3f}.")
    print("=" * 70)


if __name__ == "__main__":
    main()
