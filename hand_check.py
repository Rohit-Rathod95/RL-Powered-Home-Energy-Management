"""hand_check.py

Pencil-and-paper verification script for Home Electricity Saver (Phase 1).

Prints a step-by-step trace of 4 consecutive environment steps, showing every
intermediate mathematical quantity (T_out, E, cost, T_eff, D_T, D_L, reward,
T[t+1], and discrete state tuple) for manual verification.
"""

from __future__ import annotations

import numpy as np

from config import (
    ACTIVITIES,
    COOLING_MODES,
    DAY_TYPES,
    EnvConfig,
    REGIMES,
)
from discretize import discretize_state, get_temp_bin, get_time_block
from dynamics import (
    compute_effective_temp,
    compute_energy,
    compute_outdoor_temp,
)
from env import decode_action, encode_action
from schedule import generate_daily_schedule, is_dark


def main() -> None:
    print("=" * 80)
    print(" Home Electricity Saver: 4-Step Pencil-and-Paper Hand Check")
    print("=" * 80)

    config = EnvConfig()
    seed = 42
    rng = np.random.default_rng(seed)

    day_type = int(rng.choice([0, 1]))
    weather_regime = int(rng.choice([0, 1]))
    schedule = generate_daily_schedule(day_type, rng, config)
    outdoor_noises = rng.normal(0.0, config.outdoor_noise_std, size=config.episode_steps)

    print(f"Episode Setup (seed={seed}):")
    print(f"  Day type:       {DAY_TYPES[day_type]} ({day_type})")
    print(f"  Weather regime: {REGIMES[weather_regime]} ({weather_regime})")
    print(f"  Initial temp:   {config.t_mean:.2f} deg C")
    print(f"  Config dt:      {config.dt:.2f} h, tau={config.tau:.1f} h, g_occ={config.g_occ:.2f} deg C")
    print(f"  Price:          {config.price_per_kwh:.2f} INR/kWh")
    print(f"  Comfort band:   [{config.t_comfort_lo:.1f}, {config.t_comfort_hi:.1f}] deg C")
    print("=" * 80)

    # 4 distinct test actions:
    # Step 0: action 0 (off, light off)
    # Step 1: action 3 (fan_low, light on)
    # Step 2: action 6 (cooler_fan, light off)
    # Step 3: action 8 (ac_fan_low, light off)
    test_actions = [
        encode_action(cooling_mode=0, light_on=0),
        encode_action(cooling_mode=1, light_on=1),
        encode_action(cooling_mode=3, light_on=0),
        encode_action(cooling_mode=4, light_on=0),
    ]

    current_temp = float(config.t_mean)
    prev_mode = 0

    for step_idx, action in enumerate(test_actions):
        hour = step_idx * config.dt
        cooling_mode, light_on = decode_action(action)
        switched = (cooling_mode != prev_mode)
        activity = int(schedule[step_idx])
        occupied = (activity != 0)
        noise = float(outdoor_noises[step_idx])
        dark = is_dark(hour, config)

        t_out = compute_outdoor_temp(hour, noise, config)
        energy_kwh = compute_energy(cooling_mode, light_on, config)
        cost = config.price_per_kwh * energy_kwh
        t_eff = compute_effective_temp(current_temp, cooling_mode, config)

        # Comfort calculations
        d_t = occupied * (max(0.0, t_eff - config.t_comfort_hi)**2 + max(0.0, config.t_comfort_lo - t_eff)**2)
        d_l = 1.0 if (activity == 1 and dark and not light_on) else 0.0
        r_step = - cost - (config.lambda_t * d_t) - (config.lambda_l * d_l) - (config.lambda_s * (1.0 if switched else 0.0))

        # Thermal dynamics calculations
        passive_delta = (config.dt / config.tau) * (t_out - current_temp)
        occ_delta = config.g_occ if occupied else 0.0

        if cooling_mode == 3:  # Cooler
            hum_factor = config.humidity_factor_dry if weather_regime == 0 else config.humidity_factor_humid
            cooling_delta = config.c_cooler * (1.0 - hum_factor) * max(0.0, current_temp - config.cooler_temp_floor)
        elif cooling_mode == 4:  # AC
            cooling_delta = config.c_ac * max(0.0, current_temp - config.ac_setpoint)
        else:
            cooling_delta = 0.0

        next_temp_computed = current_temp + passive_delta + occ_delta - cooling_delta

        time_block = get_time_block(step_idx, config)
        temp_bin = get_temp_bin(current_temp, config.temp_bins)
        state_tuple = discretize_state(step_idx, activity, day_type, current_temp, weather_regime, config, prev_mode)

        print(f"\n--- STEP {step_idx} (Hour: {hour:04.1f}, Action: {action}) ---")
        print(f"  Mode:             {COOLING_MODES[cooling_mode]} (mode={cooling_mode}), Light: {'ON' if light_on else 'OFF'}")
        print(f"  Switched mode:    {switched} (prev={COOLING_MODES[prev_mode]})")
        print(f"  Activity:         {ACTIVITIES[activity]} (occupied={occupied}), Dark: {dark}")
        print(f"  Outdoor T_out:    {t_out:.4f} deg C (noise={noise:+.4f} deg C)")
        print(f"  Indoor T[t]:      {current_temp:.4f} deg C")
        print(f"  Effective T_eff:  {t_eff:.4f} deg C (wind-chill delta={current_temp - t_eff:.2f} deg C)")
        print(f"  Energy & Cost:    {energy_kwh:.4f} kWh -> Cost: INR {cost:.4f}")
        print(f"  Discomfort:       D_T = {d_t:.4f}, D_L = {d_l:.4f}")
        print(f"  Target Reward:    r = -{cost:.4f} - ({config.lambda_t}*{d_t:.4f}) - ({config.lambda_l}*{d_l:.4f}) = {r_step:.4f}")
        print("  Temperature Transition:")
        print(f"    Passive delta:  +{passive_delta:+.4f} deg C  [(dt/tau)*(T_out - T) = ({config.dt}/{config.tau})*({t_out:.2f} - {current_temp:.2f})]")
        print(f"    Occupant gain:  +{occ_delta:+.4f} deg C")
        print(f"    Active cooling: -{cooling_delta:.4f} deg C")
        print(f"    => T[t+1]:      {next_temp_computed:.4f} deg C")
        print(f"  Discrete State:   {state_tuple} [block={time_block}, act={activity}, day={day_type}, bin={temp_bin}, regime={weather_regime}]")

        prev_mode = cooling_mode
        current_temp = next_temp_computed

    print("\n" + "=" * 80)
    print(" Hand Check Complete. Compare these exact intermediate values with your notes.")
    print("=" * 80)


if __name__ == "__main__":
    main()
