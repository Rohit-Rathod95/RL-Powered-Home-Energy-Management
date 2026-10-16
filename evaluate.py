"""evaluate.py

Evaluation pipeline for Home Electricity Saver (Phase 2).

Implements B2:
  - run_episodes(policy, seeds, config) returning detailed per-episode metrics:
      * energy_kwh: Total daily energy consumed (kWh)
      * cost: Total daily electricity cost (₹)
      * total_d_t: Total thermal discomfort (squared °C)
      * occupied_hours_uncomfortable: Total occupied hours where T_eff outside [t_comfort_lo, t_comfort_hi]
      * occupied_uncomfortable_share: Share of occupied steps that were uncomfortable
      * light_needed_but_off_share: Share of awake-and-dark steps where light was off
      * energy_while_away: Energy consumed while household is away (kWh)
      * switches_per_day: Count of cooling mode switches during the day
      * total_reward: Sum of scalar rewards over the 48-step day
  - Disjoint train / validation / evaluation seeds (never mix):
      TRAIN_SEEDS = range(0, 1000)
      VALIDATION_SEEDS = range(20000, 20050)
      EVAL_SEEDS = range(10000, 10200)
  - summarize_metrics(): Calculates mean and std across evaluation seeds.
"""

from __future__ import annotations

from typing import Any
import inspect
import numpy as np
import pandas as pd

from baselines import Policy
from config import ACTIVITY_AWAKE, ACTIVITY_AWAY, EnvConfig
from env import RoomEnv
from schedule import is_dark

# Strict train / validation / eval seed partition (never mix)
TRAIN_SEEDS = range(0, 1000)
VALIDATION_SEEDS = range(20000, 20050)
EVAL_SEEDS = range(10000, 10200)


def run_episodes(
    policy: Policy,
    seeds: list[int] | range,
    config: EnvConfig | None = None,
) -> pd.DataFrame:
    """Run evaluation episodes for a given policy across a sequence of seeds.

    Args:
        policy: Policy instance with select_action(state).
        seeds: Iterable of integer seeds.
        config: EnvConfig instance.

    Returns:
        pd.DataFrame: DataFrame with one row per seed containing all episode metrics.
    """
    cfg = config if config is not None else EnvConfig()
    env = RoomEnv(cfg)

    records: list[dict[str, Any]] = []

    # Check if policy's select_action accepts a 'step' parameter (e.g. clock policies)
    try:
        sig = inspect.signature(policy.select_action)
        policy_takes_step = "step" in sig.parameters
    except Exception:
        policy_takes_step = False

    for seed in seeds:
        if hasattr(policy, "reset"):
            policy.reset()

        state, info = env.reset(seed=seed)

        ep_energy = 0.0
        ep_cost = 0.0
        ep_d_t = 0.0
        ep_reward = 0.0
        ep_switches = 0
        ep_energy_away = 0.0
        ep_degree_hours_above = 0.0

        occupied_steps = 0
        uncomfortable_occupied_steps = 0
        light_needed_steps = 0
        light_needed_but_off_steps = 0

        truncated = False
        while not truncated:
            step_now = env.step_idx
            if policy_takes_step:
                action = policy.select_action(state, step=step_now)
            else:
                action = policy.select_action(state)

            next_state, reward, terminated, truncated, step_info = env.step(action)

            # Accumulate scalars
            ep_reward += reward
            ep_energy += step_info["energy_kwh"]
            ep_cost += step_info["cost"]
            ep_d_t += step_info["D_T"]
            if step_info["switched"]:
                ep_switches += 1

            act = step_info["activity"]
            t_eff = step_info["effective_temp"]
            step_idx = step_info["step"] - 1
            hour = step_idx * cfg.dt
            dark = is_dark(hour, cfg)

            # Occupancy and comfort
            if act != ACTIVITY_AWAY:
                occupied_steps += 1
                if t_eff < cfg.t_comfort_lo or t_eff > cfg.t_comfort_hi:
                    uncomfortable_occupied_steps += 1
                if t_eff > cfg.t_comfort_hi:
                    ep_degree_hours_above += (t_eff - cfg.t_comfort_hi) * cfg.dt
            else:
                ep_energy_away += step_info["energy_kwh"]

            # Visual lighting requirement: awake and dark
            if act == ACTIVITY_AWAKE and dark:
                light_needed_steps += 1
                if step_info["D_L"] > 0.0:
                    light_needed_but_off_steps += 1

            state = next_state

        occ_hours_uncomp = uncomfortable_occupied_steps * cfg.dt
        occ_uncomp_share = (
            (uncomfortable_occupied_steps / occupied_steps) if occupied_steps > 0 else 0.0
        )
        light_needed_share = (
            (light_needed_but_off_steps / light_needed_steps) if light_needed_steps > 0 else 0.0
        )

        records.append({
            "seed": seed,
            "energy_kwh": ep_energy,
            "cost": ep_cost,
            "total_d_t": ep_d_t,
            "occupied_hours_uncomfortable": occ_hours_uncomp,
            "occupied_uncomfortable_share": occ_uncomp_share,
            "degree_hours_above_comfort": ep_degree_hours_above,
            "light_needed_but_off_share": light_needed_share,
            "energy_while_away": ep_energy_away,
            "switches_per_day": ep_switches,
            "total_reward": ep_reward,
        })

    return pd.DataFrame(records)


def summarize_metrics(df: pd.DataFrame, controller_name: str) -> dict[str, Any]:
    """Calculate mean and standard deviation for each metric across evaluation episodes.

    Args:
        df: DataFrame returned by run_episodes().
        controller_name: Name of the controller.

    Returns:
        dict[str, Any]: Summary dictionary with mean and std for each metric.
    """
    metrics = [
        "energy_kwh",
        "cost",
        "total_d_t",
        "occupied_hours_uncomfortable",
        "occupied_uncomfortable_share",
        "degree_hours_above_comfort",
        "light_needed_but_off_share",
        "energy_while_away",
        "switches_per_day",
        "total_reward",
    ]

    summary: dict[str, Any] = {"controller": controller_name}
    for m in metrics:
        summary[f"{m}_mean"] = float(df[m].mean())
        summary[f"{m}_std"] = float(df[m].std())

    return summary


def compute_paired_differences(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    metric: str = "total_reward",
) -> dict[str, float]:
    """Compute paired per-day differences between controller A and controller B.

    Delta_i = A_i - B_i for each evaluation seed.

    Returns:
        dict[str, float]:
            - mean_diff: Sample mean difference
            - std_diff: Sample standard deviation of differences
            - se_diff: Standard error of the mean difference
            - ci_95_lower: Lower bound of 95% confidence interval
            - ci_95_upper: Upper bound of 95% confidence interval
    """
    merged = pd.merge(df_a[["seed", metric]], df_b[["seed", metric]], on="seed", suffixes=("_a", "_b"))
    diffs = merged[f"{metric}_a"] - merged[f"{metric}_b"]
    n = len(diffs)
    mean_diff = float(diffs.mean())
    std_diff = float(diffs.std(ddof=1))
    se_diff = float(std_diff / np.sqrt(n))

    # 95% CI critical value for t-distribution (df = n - 1)
    # For n = 200, df = 199 -> t_crit ~ 1.972 (or 1.960 for normal)
    t_crit = 1.972 if n >= 100 else 2.0
    ci_lower = float(mean_diff - t_crit * se_diff)
    ci_upper = float(mean_diff + t_crit * se_diff)

    return {
        "n": float(n),
        "mean_diff": mean_diff,
        "std_diff": std_diff,
        "se_diff": se_diff,
        "ci_95_lower": ci_lower,
        "ci_95_upper": ci_upper,
    }

