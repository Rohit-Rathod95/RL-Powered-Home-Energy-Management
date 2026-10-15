"""dynamics.py

Physical dynamics and reward functions for Home Electricity Saver (Phase 1).

In accordance with Phase 1 Mode B:
- compute_outdoor_temp(), compute_effective_temp(), and compute_energy() are fully implemented.
- next_temperature() and compute_reward() are clearly marked TODO stubs with full docstrings
  detailing inputs, outputs, units, and the corresponding design.md sections.
"""

from __future__ import annotations

import numpy as np

from config import (
    ACTIVITY_AWAKE,
    ACTIVITY_AWAY,
    EnvConfig,
    MODE_AC_FAN_LOW,
    MODE_COOLER_FAN,
    MODE_FAN_HIGH,
    MODE_FAN_LOW,
    MODE_OFF,
    REGIME_DRY,
    REGIME_HUMID,
)


def compute_outdoor_temp(hour: float, noise: float, config: EnvConfig) -> float:
    """Compute outdoor temperature at a given hour using a cosine wave plus noise.

    Formula (design.md Section 6):
        T_out[t] = T_mean + A * cos(2 * pi * (hour - hour_peak) / 24) + noise

    Peak is exactly at hour == config.t_peak_hour.
    Trough is exactly at (hour_peak - 12) % 24.

    Args:
        hour: Hour of the day in [0.0, 24.0).
        noise: Pre-sampled zero-mean Gaussian noise in °C.
        config: EnvConfig specifying t_mean, t_amplitude, and t_peak_hour.

    Returns:
        float: Outdoor air temperature in °C.
    """
    cosine_arg = 2.0 * np.pi * (hour - config.t_peak_hour) / 24.0
    return float(config.t_mean + config.t_amplitude * np.cos(cosine_arg) + noise)


def compute_effective_temp(indoor_temp: float, cooling_mode: int, config: EnvConfig) -> float:
    """Compute the perceived (felt) temperature used for thermal comfort.

    Formula (design.md Section 6):
        T_eff = T - k_fan(mode)

    Air motion from fans lowers perceived temperature without changing air temperature.
    Modes that include a fan (cooler_fan, ac_fan_low) also include fan_low wind-chill.

    Args:
        indoor_temp: Continuous indoor air temperature in °C.
        cooling_mode: Integer cooling mode (0..4).
        config: EnvConfig specifying k_fan_low and k_fan_high.

    Returns:
        float: Effective temperature in °C.
    """
    if cooling_mode == MODE_OFF:
        k_fan = 0.0
    elif cooling_mode == MODE_FAN_LOW:
        k_fan = config.k_fan_low
    elif cooling_mode == MODE_FAN_HIGH:
        k_fan = config.k_fan_high
    elif cooling_mode == MODE_COOLER_FAN:
        k_fan = config.k_fan_low  # Cooler includes internal fan circulation
    elif cooling_mode == MODE_AC_FAN_LOW:
        k_fan = config.k_fan_low  # AC includes internal fan circulation
    else:
        k_fan = 0.0

    return float(indoor_temp - k_fan)


def compute_energy(cooling_mode: int, light_on: int, config: EnvConfig) -> float:
    """Compute electrical energy consumed in one step.

    Formula (design.md Section 7):
        E[t] = P(mode) * dt + P_light * light_on * dt   (kWh)

    Args:
        cooling_mode: Integer cooling mode (0..4).
        light_on: Binary light indicator (0 = off, 1 = on).
        config: EnvConfig containing appliance power ratings (kW) and step dt (h).

    Returns:
        float: Energy consumed during the 30-minute step in kWh.
    """
    if cooling_mode == MODE_OFF:
        p_mode = 0.0
    elif cooling_mode == MODE_FAN_LOW:
        p_mode = config.p_fan_low
    elif cooling_mode == MODE_FAN_HIGH:
        p_mode = config.p_fan_high
    elif cooling_mode == MODE_COOLER_FAN:
        p_mode = config.p_cooler_fan
    elif cooling_mode == MODE_AC_FAN_LOW:
        p_mode = config.p_ac_fan_low
    else:
        p_mode = 0.0

    p_light_used = config.p_light if light_on else 0.0
    return float((p_mode + p_light_used) * config.dt)


def next_temperature(
    current_temp: float,
    cooling_mode: int,
    weather_regime: int,
    hour: float,
    occupied: bool,
    outdoor_noise: float,
    config: EnvConfig,
) -> float:
    """TODO (Mode B Stub): Pure dynamics function for indoor room temperature T[t+1].

    Implements design.md Section 6:
        T[t+1] = T[t] + (dt / tau) * (T_out[t] - T[t]) + g_occ * occupied[t] - cooling(mode, regime, T[t])

    Equations & Specifications:
    ---------------------------
    1. Outdoor temperature:
       T_out[t] = compute_outdoor_temp(hour, outdoor_noise, config)

    2. Passive heat transfer & occupancy:
       delta_passive = (config.dt / config.tau) * (T_out - current_temp)
       delta_occ = config.g_occ if occupied else 0.0

    3. Active cooling terms (cooling rates c_cooler and c_ac are fractions in (0, 1)):
       - fan (MODE_OFF, MODE_FAN_LOW, MODE_FAN_HIGH):
           cooling = 0.0  (fans do not change air temperature)
       - evaporative cooler (MODE_COOLER_FAN):
           humidity_factor = (config.humidity_factor_dry if weather_regime == REGIME_DRY
                              else config.humidity_factor_humid)
           gap = max(0.0, current_temp - config.cooler_temp_floor)
           cooling = config.c_cooler * (1.0 - humidity_factor) * gap
       - AC (MODE_AC_FAN_LOW):
           gap = max(0.0, current_temp - config.ac_setpoint)
           cooling = config.c_ac * gap

       Note on no-overshoot guarantee:
       Because c_cooler, c_ac in (0, 1) and (1 - humidity_factor) <= 1, the cooling term
       strictly pulls the temperature toward the floor/setpoint without overshooting it.

    Args:
        current_temp: Current continuous indoor temperature T[t] in °C.
        cooling_mode: Integer cooling mode in [0, 4] (MODE_OFF, MODE_FAN_LOW, ...).
        weather_regime: Weather regime in [0, 1] (0 = dry, 1 = humid).
        hour: Current hour of the day in [0.0, 24.0).
        occupied: Boolean flag (True if activity != away, False if away).
        outdoor_noise: Pre-sampled Gaussian noise value in °C for T_out.
        config: EnvConfig specifying tau, g_occ, c_cooler, c_ac, setpoints, etc.

    Returns:
        float: Next indoor temperature T[t+1] in °C.
    """
    t_out = compute_outdoor_temp(hour, outdoor_noise, config)
    delta_passive = (config.dt / config.tau) * (t_out - current_temp)
    delta_occ = config.g_occ if occupied else 0.0

    if cooling_mode in (MODE_OFF, MODE_FAN_LOW, MODE_FAN_HIGH):
        cooling = 0.0
    elif cooling_mode == MODE_COOLER_FAN:
        humidity_factor = (
            config.humidity_factor_dry if weather_regime == REGIME_DRY
            else config.humidity_factor_humid
        )
        gap = max(0.0, current_temp - config.cooler_temp_floor)
        cooling = config.c_cooler * (1.0 - humidity_factor) * gap
    elif cooling_mode == MODE_AC_FAN_LOW:
        gap = max(0.0, current_temp - config.ac_setpoint)
        cooling = config.c_ac * gap
    else:
        cooling = 0.0

    return float(current_temp + delta_passive + delta_occ - cooling)


def compute_reward(
    effective_temp: float,
    activity: int,
    is_dark: bool,
    light_on: int,
    energy_kwh: float,
    switched: bool,
    config: EnvConfig,
) -> tuple[float, dict[str, float]]:
    """Reward function and component breakdown per design.md sections 7-9.

    r[t] = - cost[t] - lambda_T * D_T[t] - lambda_L * D_L[t] - lambda_S * switched[t]

    Args:
        effective_temp: Perceived effective temperature T_eff in °C.
        activity: Occupancy activity index (0 = away, 1 = awake, 2 = asleep).
        is_dark: Boolean indicating whether outside daylight is absent.
        light_on: Binary indicator of whether room light is turned on (0 or 1).
        energy_kwh: Electrical energy consumed in this step (from compute_energy).
        switched: Boolean indicating whether cooling mode changed from previous step.
        config: EnvConfig instance containing tariff and lambda penalty weights.

    Returns:
        tuple[float, dict[str, float]]:
            - reward (float): Scalar reward r[t].
            - components (dict): Dictionary with keys:
                "cost": float,
                "D_T": float,
                "D_L": float,
                "switched": float (1.0 or 0.0)
    """
    cost = float(config.price_per_kwh * energy_kwh)

    occupied = (activity != ACTIVITY_AWAY)
    if occupied:
        d_t_hi = max(0.0, effective_temp - config.t_comfort_hi)
        d_t_lo = max(0.0, config.t_comfort_lo - effective_temp)
        d_t = float(d_t_hi ** 2 + d_t_lo ** 2)
    else:
        d_t = 0.0

    d_l = 1.0 if (activity == ACTIVITY_AWAKE and is_dark and not light_on) else 0.0

    switched_flag = 1.0 if (config.include_prev_mode and switched) else 0.0
    switched_penalty = float(config.lambda_s * switched_flag)

    total_reward = float(-cost - (config.lambda_t * d_t) - (config.lambda_l * d_l) - switched_penalty)

    components = {
        "cost": cost,
        "D_T": d_t,
        "D_L": d_l,
        "switched": switched_flag,
    }
    return total_reward, components
