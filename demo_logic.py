"""Pure logic for the interactive Home Electricity Saver demonstration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable
import json

import numpy as np

from baselines import (
    AlwaysOffPolicy,
    TunedPrecoolPolicy,
    TunedThresholdPolicy,
)
from config import (
    ACTIVITIES,
    ACTIVITY_AWAY,
    ACTIVITY_AWAKE,
    COOLING_MODES,
    DAY_TYPES,
    EnvConfig,
    REGIMES,
)
from discretize import discretize_state
from dynamics import compute_energy, compute_reward, next_temperature
from env import RoomEnv, decode_action
from evaluate import EVAL_SEEDS, TRAIN_SEEDS, VALIDATION_SEEDS
from qlearning import q_update, select_action, state_to_index
from schedule import is_dark


PLACEHOLDER_BANNER = (
    "Simulator with uncalibrated placeholder parameters; results show how methods "
    "compare inside the simulation, not measured savings."
)


@dataclass
class ReplayResult:
    rows: list[dict]
    metrics: dict[str, float]
    final_state: tuple[int, ...]


def load_baseline_params(path: str | Path = "baseline_params.json") -> dict | None:
    """Load persisted tuning parameters, returning None when unavailable."""
    file_path = Path(path)
    if not file_path.exists():
        return None
    with file_path.open(encoding="utf-8") as handle:
        return json.load(handle)


def make_baseline_policies(
    params: dict,
    config: EnvConfig | None = None,
) -> tuple[TunedThresholdPolicy, TunedPrecoolPolicy]:
    """Build baseline policies from persisted parameters without re-tuning."""
    cfg = config if config is not None else EnvConfig()
    threshold_data = params["tuned_threshold"]
    threshold = TunedThresholdPolicy(
        mode_bin_2=int(threshold_data["mode_bin_2"]),
        mode_bin_3=int(threshold_data["mode_bin_3"]),
        mode_bin_4_dry=int(threshold_data["mode_bin_4_dry"]),
        mode_bin_4_humid=int(threshold_data["mode_bin_4_humid"]),
        precool_mode=int(threshold_data["precool_mode"]),
        config=cfg,
    )
    precool_data = params["tuned_precool_block"]
    precool = TunedPrecoolPolicy(
        base_policy=threshold,
        lead_steps=int(precool_data["lead_steps"]),
        trigger_bin=int(precool_data["trigger_bin"]),
        precool_mode=int(precool_data["precool_mode"]),
        config=cfg,
    )
    return threshold, precool


def state_words(state: tuple[int, ...], raw_temp: float, config: EnvConfig) -> dict[str, str]:
    """Describe a real discretized state for display."""
    block, activity, day_type, temp_bin, regime = state[:5]
    edges = (-np.inf,) + config.temp_bins + (np.inf,)
    low, high = edges[temp_bin], edges[temp_bin + 1]
    temp_range = f"{low:g} to {high:g} °C"
    return {
        "time": f"{block * config.time_block_hours:g}:00–"
        f"{(block + 1) * config.time_block_hours:g}:00",
        "activity": ACTIVITIES[activity],
        "day_type": DAY_TYPES[day_type],
        "temperature_bin": temp_range,
        "weather": REGIMES[regime],
        "raw_temperature": f"{raw_temp:.2f} °C",
    }


def action_words(action: int) -> dict[str, str]:
    """Describe one of the environment's ten encoded actions."""
    mode, light = decode_action(int(action))
    return {
        "mode": COOLING_MODES[mode],
        "light": "on" if light else "off",
        "label": f"{COOLING_MODES[mode]} + light {'on' if light else 'off'}",
    }


def q_update_preview(
    q_table: np.ndarray,
    s_idx: int,
    action: int,
    reward: float,
    next_s_idx: int,
    terminated: bool,
    alpha: float,
    gamma: float,
) -> dict[str, float | bool | np.ndarray]:
    """Compute the project's update on a copy and report every displayed term."""
    before = float(q_table[s_idx, action])
    next_max = 0.0 if terminated else float(q_table[next_s_idx].max())
    target = float(reward + gamma * next_max)
    copied = q_table.copy()
    q_update(copied, s_idx, action, reward, next_s_idx, terminated, alpha, gamma)
    return {
        "before": before,
        "reward": float(reward),
        "gamma_next_max": float(gamma * next_max),
        "target": target,
        "alpha": float(alpha),
        "after": float(copied[s_idx, action]),
        "terminated": bool(terminated),
        "table": copied,
    }


def _empty_metrics() -> dict[str, float]:
    return {
        "energy_kwh": 0.0,
        "comfort_violations_hours": 0.0,
        "light_adequacy": 0.0,
        "total_reward": 0.0,
        "occupied_steps": 0.0,
        "light_needed_steps": 0.0,
        "light_on_needed_steps": 0.0,
    }


def replay_episode(
    policy,
    seed: int,
    config: EnvConfig | None = None,
    max_steps: int | None = None,
) -> ReplayResult:
    """Replay a real environment day and collect display-ready rows and metrics."""
    cfg = config if config is not None else EnvConfig()
    env = RoomEnv(cfg)
    if hasattr(policy, "reset"):
        policy.reset()
    state, _ = env.reset(seed=int(seed))
    rows: list[dict] = []
    metrics = _empty_metrics()
    truncated = False
    while not truncated and (max_steps is None or len(rows) < max_steps):
        step = env.step_idx
        raw_temp = env.indoor_temp
        action = int(policy.select_action(state))
        next_state, reward, terminated, truncated, info = env.step(action)
        dark = is_dark(step * cfg.dt, cfg)
        activity = int(info["activity"])
        metrics["energy_kwh"] += float(info["energy_kwh"])
        metrics["total_reward"] += float(reward)
        if activity != ACTIVITY_AWAY:
            metrics["occupied_steps"] += 1.0
            if info["effective_temp"] < cfg.t_comfort_lo or info["effective_temp"] > cfg.t_comfort_hi:
                metrics["comfort_violations_hours"] += cfg.dt
        if activity == ACTIVITY_AWAKE and dark:
            metrics["light_needed_steps"] += 1.0
            if decode_action(action)[1] == 1:
                metrics["light_on_needed_steps"] += 1.0
        rows.append({
            "step": step,
            "hour": step * cfg.dt,
            "state": state,
            "raw_indoor_temp": raw_temp,
            "action": action,
            "next_state": next_state,
            "reward": float(reward),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "info": info,
            "dark": dark,
        })
        state = next_state
    needed = metrics["light_needed_steps"]
    metrics["light_adequacy"] = (
        metrics["light_on_needed_steps"] / needed if needed else 1.0
    )
    return ReplayResult(rows, metrics, state)


def compare_seed(
    selected_policy,
    threshold_policy,
    seed: int,
    config: EnvConfig | None = None,
) -> dict[str, dict[str, float]]:
    """Return same-seed headline metrics for selected, off, and threshold policies."""
    cfg = config if config is not None else EnvConfig()
    policies = {
        "selected": selected_policy,
        "always_off": AlwaysOffPolicy(),
        "tuned_threshold": threshold_policy,
    }
    return {
        name: replay_episode(policy, seed, cfg).metrics
        for name, policy in policies.items()
    }


def q_table_action_policy(q_table: np.ndarray, config: EnvConfig | None = None):
    """Create a greedy policy using the project's epsilon-greedy implementation."""
    cfg = config if config is not None else EnvConfig()
    rng = np.random.default_rng(42)

    class _Policy:
        def select_action(self, state: tuple[int, ...]) -> int:
            return select_action(q_table, state_to_index(state, cfg), 0.0, rng)

    return _Policy()


def one_step_previews(
    state: tuple[int, ...],
    indoor_temp: float,
    actions: Iterable[int],
    config: EnvConfig | None = None,
) -> list[dict[str, float | int | str]]:
    """Evaluate hypothetical actions with real pure dynamics and zero noise."""
    cfg = config if config is not None else EnvConfig()
    block, activity, _, _, regime = state[:5]
    hour = block * cfg.time_block_hours
    dark = is_dark(hour, cfg)
    rows = []
    for action in actions:
        mode, light = decode_action(int(action))
        energy = compute_energy(mode, light, cfg)
        effective = indoor_temp - (
            cfg.k_fan_low if mode in (1, 3, 4)
            else cfg.k_fan_high if mode == 2 else 0.0
        )
        reward, components = compute_reward(
            effective, activity, dark, light, energy, mode != 0, cfg
        )
        next_temp = next_temperature(
            indoor_temp, mode, regime, hour, activity != ACTIVITY_AWAY, 0.0, cfg
        )
        rows.append({
            "action": int(action),
            "label": action_words(int(action))["label"],
            "energy_kwh": float(energy),
            "next_temperature": float(next_temp),
            "reward": float(reward),
            "cost": components["cost"],
            "D_T": components["D_T"],
            "D_L": components["D_L"],
        })
    return rows


def search_scenarios(
    q_table: np.ndarray,
    config: EnvConfig | None = None,
    seeds: Iterable[int] = EVAL_SEEDS,
) -> list[dict]:
    """Find the first reachable real evaluation state for each guided scenario."""
    cfg = config if config is not None else EnvConfig()
    seed_list = [int(seed) for seed in seeds]
    if any(seed in set(TRAIN_SEEDS) or seed in set(VALIDATION_SEEDS) for seed in seed_list):
        raise ValueError("Scenario search requires EVAL_SEEDS only.")
    policy = q_table_action_policy(q_table, cfg)

    def predicates() -> list[tuple[str, Callable[[dict], bool]]]:
        return [
            ("Hot room, occupant home and awake",
             lambda r: r["info"]["activity"] == ACTIVITY_AWAKE and r["info"]["effective_temp"] > cfg.t_comfort_hi),
            ("Empty room at midday",
             lambda r: r["info"]["activity"] == ACTIVITY_AWAY and 10.0 <= r["hour"] < 14.0),
            ("Empty room shortly before return on a workday",
             lambda r: r["info"]["activity"] == ACTIVITY_AWAY and r["info"]["day_type"] == 0
             and 16.0 <= r["hour"] < 18.0),
            ("Awake in the dark",
             lambda r: r["info"]["activity"] == ACTIVITY_AWAKE and r["dark"]),
            ("Comfortable room, occupant home",
             lambda r: r["info"]["activity"] != ACTIVITY_AWAY
             and cfg.t_comfort_lo <= r["info"]["effective_temp"] <= cfg.t_comfort_hi),
        ]

    found: dict[str, dict] = {}
    for seed in seed_list:
        result = replay_episode(policy, seed, cfg)
        for row in result.rows:
            for name, predicate in predicates():
                if name not in found and predicate(row):
                    action = int(row["action"])
                    mode, light = decode_action(action)
                    intended = {
                        "Empty room at midday": mode == 0,
                        "Empty room shortly before return on a workday": mode != 0,
                        "Awake in the dark": light == 1,
                        "Comfortable room, occupant home": mode == 0,
                    }.get(name, True)
                    found[name] = {
                        "scenario": name,
                        "seed": seed,
                        "step": row["step"],
                        "state": row["state"],
                        "action": action,
                        "action_label": action_words(action)["label"],
                        "reward": row["reward"],
                        "info": row["info"],
                        "contradiction": not intended,
                    }
    return [found[name] for name, _ in predicates() if name in found]


def train_with_progress(
    total_episodes: int,
    config: EnvConfig | None = None,
    agent_seed: int = 42,
    alpha: float = 0.1,
    gamma: float = 0.97,
    validation_frequency: int = 100,
    callback: Callable[[int, int, float, float, np.ndarray, np.ndarray], None] | None = None,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Train in session memory using only TRAIN episodes and VALIDATION evaluation."""
    cfg = config if config is not None else EnvConfig()
    rng = np.random.default_rng(agent_seed)
    n_states = int(np.prod(__import__("discretize").get_state_space_dims(cfg)))
    q_table = np.zeros((n_states, 10), dtype=np.float64)
    visits = np.zeros((n_states, 10), dtype=np.int64)
    env = RoomEnv(cfg)
    train_list = list(TRAIN_SEEDS)
    curve: list[dict] = []
    decay = max(1, int(total_episodes * 0.70))
    for episode in range(1, total_episodes + 1):
        epsilon = 1.0 - min(1.0, episode / decay) * 0.95
        seed = int(rng.choice(train_list))
        state, _ = env.reset(seed=seed)
        s_idx = state_to_index(state, cfg)
        total_reward = 0.0
        truncated = False
        while not truncated:
            action = select_action(q_table, s_idx, epsilon, rng)
            visits[s_idx, action] += 1
            next_state, reward, terminated, truncated, _ = env.step(action)
            next_idx = state_to_index(next_state, cfg)
            q_update(q_table, s_idx, action, reward, next_idx, terminated, alpha, gamma)
            s_idx = next_idx
            total_reward += reward
        if episode % validation_frequency == 0 or episode == total_episodes:
            val_policy = q_table_action_policy(q_table, cfg)
            val_rewards = [
                replay_episode(val_policy, seed, cfg).metrics["total_reward"]
                for seed in VALIDATION_SEEDS
            ]
            val_mean = float(np.mean(val_rewards))
            record = {"episode": episode, "epsilon": epsilon, "val_reward_mean": val_mean}
            curve.append(record)
            if callback is not None:
                callback(episode, total_episodes, val_mean, total_reward, q_table, visits)
    return q_table, visits, curve
