"""steady_state.py

Analytical and simulated steady-state parameter derivation for Home Electricity Saver.

Section 6 / A5:
Computes T_settle analytically from:
    T_settle = (L * T_out + c * T_target + g) / (L + c),  L = dt / tau  (c = 0 for no cooling)
and by iterating next_temperature() 300 times at constant outdoor temperature.
Computes 90% closing time in minutes and validates all Phase 1 thermal calibration constraints.
"""

from __future__ import annotations

from typing import Any
import numpy as np

from config import (
    EnvConfig,
    MODE_AC_FAN_LOW,
    MODE_COOLER_FAN,
    MODE_OFF,
    REGIME_DRY,
    REGIME_HUMID,
)
from dynamics import compute_effective_temp, next_temperature


def compute_steady_state_table(config: EnvConfig | None = None) -> list[dict[str, Any]]:
    """Compute steady-state temperatures and closing times for all test cases.

    Cases evaluated:
      1. No cooling (MODE_OFF)
      2. Cooler dry (MODE_COOLER_FAN, REGIME_DRY)
      3. Cooler humid (MODE_COOLER_FAN, REGIME_HUMID)
      4. AC (MODE_AC_FAN_LOW)

    At outdoor temperatures:
      - Mean: t_mean (default 32 °C)
      - Peak: t_mean + t_amplitude (default 38 °C)

    Returns:
        list[dict[str, Any]]: List of dictionary records with analytical and simulation results.
    """
    cfg = config if config is not None else EnvConfig()
    L = cfg.dt / cfg.tau
    g = cfg.g_occ

    cases = [
        {
            "name": "No cooling",
            "mode": MODE_OFF,
            "regime": REGIME_DRY,
            "c": 0.0,
            "t_target": 0.0,
        },
        {
            "name": "Cooler (Dry)",
            "mode": MODE_COOLER_FAN,
            "regime": REGIME_DRY,
            "c": cfg.c_cooler * (1.0 - cfg.humidity_factor_dry),
            "t_target": cfg.cooler_temp_floor,
        },
        {
            "name": "Cooler (Humid)",
            "mode": MODE_COOLER_FAN,
            "regime": REGIME_HUMID,
            "c": cfg.c_cooler * (1.0 - cfg.humidity_factor_humid),
            "t_target": cfg.cooler_temp_floor,
        },
        {
            "name": "AC (Fan Low)",
            "mode": MODE_AC_FAN_LOW,
            "regime": REGIME_DRY,
            "c": cfg.c_ac,
            "t_target": cfg.ac_setpoint,
        },
    ]

    outdoor_temps = [
        ("Mean (32°C)", cfg.t_mean, 9.0),                     # hour 9.0 gives cos(2*pi*(9-15)/24) = cos(-pi/2) = 0 -> T_out = t_mean
        ("Peak (38°C)", cfg.t_mean + cfg.t_amplitude, 15.0),  # hour 15.0 gives cos(0) = 1 -> T_out = t_mean + amplitude
    ]

    records = []
    for out_name, t_out, hour in outdoor_temps:
        for case in cases:
            c = case["c"]
            t_target = case["t_target"]

            # Analytical T_settle
            t_settle_ana = (L * t_out + c * t_target + g) / (L + c)
            t_eff_ana = compute_effective_temp(t_settle_ana, case["mode"], cfg)

            # 90% closing time (discrete-time exponential decay)
            decay_rate = 1.0 - (L + c)
            if 0.0 < decay_rate < 1.0:
                n_steps_90 = np.log(0.10) / np.log(decay_rate)
                closing_time_min = float(n_steps_90 * cfg.dt * 60.0)
            else:
                closing_time_min = 0.0

            # Simulation T_settle (300 iterations at constant outdoor temperature)
            temp_sim = t_out
            for _ in range(300):
                temp_sim = next_temperature(
                    current_temp=temp_sim,
                    cooling_mode=case["mode"],
                    weather_regime=case["regime"],
                    hour=hour,
                    occupied=True,
                    outdoor_noise=0.0,
                    config=cfg,
                )
            t_eff_sim = compute_effective_temp(temp_sim, case["mode"], cfg)

            records.append({
                "condition": out_name,
                "case": case["name"],
                "t_out": t_out,
                "t_settle_ana": float(t_settle_ana),
                "t_settle_sim": float(temp_sim),
                "diff": float(abs(temp_sim - t_settle_ana)),
                "t_eff_ana": float(t_eff_ana),
                "t_eff_sim": float(t_eff_sim),
                "closing_time_min": closing_time_min,
                "closing_time_hours": closing_time_min / 60.0,
            })

    return records


def print_table(records: list[dict[str, Any]], config: EnvConfig | None = None) -> None:
    """Print a clean Markdown-compatible table of steady-state results."""
    cfg = config if config is not None else EnvConfig()
    print("=" * 105)
    print(" STEADY-STATE THERMAL CALIBRATION TABLE (Phase 1 Parameter Derivation)")
    print(f" Config: tau={cfg.tau:.1f}h, dt={cfg.dt:.2f}h (L={cfg.dt/cfg.tau:.3f}), g_occ={cfg.g_occ:.2f}°C, "
          f"c_ac={cfg.c_ac:.2f}, c_cooler={cfg.c_cooler:.2f}")
    print(f" Comfort Band: [{cfg.t_comfort_lo:.1f}, {cfg.t_comfort_hi:.1f}] °C | Body Heat Offset: {cfg.g_occ * cfg.tau / cfg.dt:.2f} °C")
    print("=" * 105)
    header = (
        f"{'Condition':<14} | {'Case':<16} | {'T_out':<7} | {'T_settle (Ana)':<14} | "
        f"{'T_settle (Sim)':<14} | {'Diff (°C)':<9} | {'T_eff':<7} | {'90% Close Time':<15}"
    )
    print(header)
    print("-" * 105)
    for r in records:
        line = (
            f"{r['condition']:<14} | {r['case']:<16} | {r['t_out']:>5.1f}°C | {r['t_settle_ana']:>12.2f}°C | "
            f"{r['t_settle_sim']:>12.2f}°C | {r['diff']:>8.5f} | {r['t_eff_ana']:>5.2f}°C | "
            f"{r['closing_time_min']:>5.1f}m ({r['closing_time_hours']:.2f}h)"
        )
        print(line)
    print("=" * 105)


if __name__ == "__main__":
    records = compute_steady_state_table()
    print_table(records)
