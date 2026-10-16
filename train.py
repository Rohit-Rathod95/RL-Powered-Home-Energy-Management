"""train.py

Training pipeline for Tabular Q-Learning on Home Electricity Saver.

Implements B3-B6:
  - Linear epsilon decay from 1.0 to 0.05 over first 70% of episodes.
  - Draws seeds from TRAIN_SEEDS (0..999) using agent's random generator.
  - Proper bootstrap on truncation (terminated=False at step 48).
  - Periodic evaluation on VALIDATION_SEEDS (20000..20049) every 500 episodes.
  - Final evaluation on EVAL_SEEDS (10000..10199).
  - Exports q_table.npy, curves.csv, policy.csv, and learning_curve.png.
  - Supports --smoke flag for fast test execution (500 episodes).
"""

from __future__ import annotations

import argparse
import sys
from typing import Any
import numpy as np
import pandas as pd

from config import ACTIVITY_AWAKE, COOLING_MODES, EnvConfig
from env import RoomEnv, decode_action
from evaluate import EVAL_SEEDS, TRAIN_SEEDS, VALIDATION_SEEDS, run_episodes, summarize_metrics
from plot_training import plot_learning_curve
from qlearning import (
    get_num_states,
    index_to_state,
    init_q_table,
    init_visit_counts,
    q_update,
    select_action,
    state_to_index,
)

class QLearningPolicy:
    """Policy wrapping a learned Q-table for evaluation."""

    name: str = "Q-Learning"

    def __init__(
        self,
        q_table: np.ndarray,
        config: EnvConfig | None = None,
        epsilon: float = 0.0,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.q_table = q_table
        self.config = config if config is not None else EnvConfig()
        self.epsilon = epsilon
        self.rng = rng if rng is not None else np.random.default_rng(42)

    def select_action(self, state: tuple[int, ...]) -> int:
        s_idx = state_to_index(state, self.config)
        return select_action(self.q_table, s_idx, self.epsilon, self.rng)


def run_validation(
    policy: QLearningPolicy,
    val_seeds: list[int] | range,
    config: EnvConfig,
) -> tuple[float, float]:
    """Evaluate greedy policy on validation seeds and return mean and std reward."""
    df_val = run_episodes(policy, val_seeds, config=config)
    return float(df_val["total_reward"].mean()), float(df_val["total_reward"].std())


def export_policy_table(
    q_table: np.ndarray,
    visit_counts: np.ndarray,
    config: EnvConfig,
    output_path: str = "policy.csv",
) -> None:
    """Export greedy policy and visit counts per state to CSV."""
    n_states = q_table.shape[0]
    records = []

    for s_idx in range(n_states):
        state = index_to_state(s_idx, config)
        total_visits = int(visit_counts[s_idx].sum())
        greedy_act = int(np.argmax(q_table[s_idx]))
        act_visits = int(visit_counts[s_idx, greedy_act])

        cooling_mode, light_on = decode_action(greedy_act)
        action_name = f"{COOLING_MODES[cooling_mode]}+light_{'on' if light_on else 'off'}"

        records.append({
            "state_index": s_idx,
            "time_block": state[0],
            "activity": state[1],
            "day_type": state[2],
            "temp_bin": state[3],
            "weather_regime": state[4],
            "greedy_action": greedy_act,
            "action_name": action_name,
            "action_visits": act_visits,
            "total_state_visits": total_visits,
            "never_visited": total_visits == 0,
        })

    df = pd.DataFrame(records)
    df.to_csv(output_path, index=False)
    print(f" Policy table successfully exported to: {output_path} ({df['never_visited'].sum()} states unvisited)")


def train(
    # PLACEHOLDER: Total number of training episodes
    total_episodes: int = 20000,
    # PLACEHOLDER: Q-learning step size learning rate alpha
    alpha: float = 0.1,
    # PLACEHOLDER: Bellman discount factor gamma
    gamma: float = 0.97,
    # PLACEHOLDER: Initial exploration rate epsilon
    epsilon_start: float = 1.0,
    # PLACEHOLDER: Final exploration rate epsilon
    epsilon_end: float = 0.05,
    # PLACEHOLDER: Fraction of episodes over which epsilon decays linearly
    epsilon_decay_fraction: float = 0.70,
    # PLACEHOLDER: Frequency (in episodes) for validation evaluation
    val_eval_freq: int = 500,
    agent_seed: int = 42,
    config: EnvConfig | None = None,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Train tabular Q-learning agent on Home Electricity Saver."""
    cfg = config if config is not None else EnvConfig()
    agent_rng = np.random.default_rng(agent_seed)

    n_states = get_num_states(cfg)
    q_table = init_q_table(n_states)
    visit_counts = init_visit_counts(n_states)

    env = RoomEnv(cfg)
    train_seeds_list = list(TRAIN_SEEDS)

    decay_episodes = max(1, int(total_episodes * epsilon_decay_fraction))
    curves_records: list[dict[str, Any]] = []

    print("=" * 80)
    print(f" Starting Q-Learning Training: {total_episodes} episodes, alpha={alpha}, gamma={gamma}")
    print(f" State space: {n_states} states, Actions: 10")
    print(f" Epsilon schedule: {epsilon_start} -> {epsilon_end} over first {decay_episodes} episodes")
    print("=" * 80)

    for ep in range(1, total_episodes + 1):
        # Calculate current epsilon via linear decay
        if ep <= decay_episodes:
            epsilon = epsilon_start - (ep / decay_episodes) * (epsilon_start - epsilon_end)
        else:
            epsilon = epsilon_end

        # Draw seed from TRAIN_SEEDS using agent's RNG
        ep_seed = int(agent_rng.choice(train_seeds_list))
        obs, info = env.reset(seed=ep_seed)
        s_idx = state_to_index(obs, cfg)

        ep_reward = 0.0
        truncated = False

        while not truncated:
            action = select_action(q_table, s_idx, epsilon, agent_rng)
            visit_counts[s_idx, action] += 1

            next_obs, reward, terminated, truncated, step_info = env.step(action)
            ep_reward += reward
            s_next_idx = state_to_index(next_obs, cfg)

            # Tabular Bellman update
            # Note: terminated is False on truncation, correctly triggering bootstrapping
            q_update(
                q_table=q_table,
                s_idx=s_idx,
                a=action,
                r=reward,
                s_next_idx=s_next_idx,
                terminated=terminated,
                alpha=alpha,
                gamma=gamma,
            )

            s_idx = s_next_idx

        # Periodic validation evaluation
        if ep % val_eval_freq == 0 or ep == total_episodes:
            val_policy = QLearningPolicy(q_table, config=cfg, epsilon=0.0)
            val_mean, val_std = run_validation(val_policy, VALIDATION_SEEDS, config=cfg)

            curves_records.append({
                "episode": ep,
                "epsilon": epsilon,
                "train_reward": ep_reward,
                "val_reward_mean": val_mean,
                "val_reward_std": val_std,
            })

            print(f"  Ep {ep:>5d}/{total_episodes} | eps={epsilon:.3f} | Train R: {ep_reward:>8.2f} | "
                  f"Val R: {val_mean:>8.2f} +/- {val_std:<6.2f}")

    curves_df = pd.DataFrame(curves_records)
    return q_table, visit_counts, curves_df


def analyze_learned_lighting(q_table: np.ndarray, config: EnvConfig) -> None:
    """Analyze and print learned light action for every state where activity is awake and dark (Point 5)."""
    from schedule import is_dark

    n_states = q_table.shape[0]
    dark_awake_states = []

    for s_idx in range(n_states):
        state = index_to_state(s_idx, config)
        time_block, activity, day_type, temp_bin, weather_regime = state[:5]
        mid_hour = (time_block + 0.5) * config.time_block_hours
        dark = is_dark(mid_hour, config)

        if activity == ACTIVITY_AWAKE and dark:
            greedy_act = int(np.argmax(q_table[s_idx]))
            cooling_mode, light_on = decode_action(greedy_act)
            dark_awake_states.append({
                "state_idx": s_idx,
                "time_block": time_block,
                "day_type": day_type,
                "temp_bin": temp_bin,
                "weather_regime": weather_regime,
                "action": greedy_act,
                "cooling": COOLING_MODES[cooling_mode],
                "light_on": light_on,
            })

    total_states = len(dark_awake_states)
    if total_states == 0:
        return

    lights_on = sum(1 for s in dark_awake_states if s["light_on"] == 1)
    share_on = (lights_on / total_states) * 100.0

    print("\n" + "=" * 95)
    print(" LEARNED LIGHTING ACTIONS IN DARK AWAKE STATES (Point 5)")
    print(f" Total Awake & Dark States: {total_states} | Light ON: {lights_on}/{total_states} ({share_on:.1f}%)")
    print("=" * 95)
    print(f"{'State Idx':<10} | {'Block':<6} | {'Day':<8} | {'Bin':<5} | {'Regime':<8} | {'Cooling Mode':<15} | {'Light Action':<12}")
    print("-" * 95)
    for s in dark_awake_states:
        day_str = "workday" if s["day_type"] == 0 else "weekend"
        reg_str = "dry" if s["weather_regime"] == 0 else "humid"
        light_str = "ON (1)" if s["light_on"] == 1 else "OFF (0)"
        print(f"{s['state_idx']:<10} | {s['time_block']:<6} | {day_str:<8} | {s['temp_bin']:<5} | {reg_str:<8} | {s['cooling']:<15} | {light_str:<12}")
    print("-" * 95)
    print(f" SUMMARY: Light is ON in {lights_on}/{total_states} ({share_on:.1f}%) of awake-and-dark states.")
    print("=" * 95 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Tabular Q-Learning Agent")
    parser.add_argument("--smoke", action="store_true", help="Run quick 500-episode smoke test")
    args = parser.parse_args()

    total_episodes = 500 if args.smoke else 20000
    val_freq = 500 if args.smoke else 500

    cfg = EnvConfig()
    try:
        q_table, visit_counts, curves_df = train(
            total_episodes=total_episodes,
            val_eval_freq=val_freq,
            config=cfg,
        )
    except NotImplementedError as e:
        print("\n" + "=" * 80)
        print(" [SCAFFOLDING READY] Algorithmic Stubs in qlearning.py:")
        print(f"   {e}")
        print(" Implement select_action() and q_update() in qlearning.py to run training.")
        print("=" * 80 + "\n")
        return

    # Save artifacts
    np.save("q_table.npy", q_table)
    print(" Saved: q_table.npy")

    curves_df.to_csv("curves.csv", index=False)
    print(" Saved: curves.csv")

    export_policy_table(q_table, visit_counts, cfg, output_path="policy.csv")

    plot_learning_curve("curves.csv", "learning_curve.png")
    print(" Saved: learning_curve.png")

    # Final evaluation on EVAL_SEEDS
    print("\n Evaluating final learned policy on held-out EVAL_SEEDS (10000..10199)...")
    eval_policy = QLearningPolicy(q_table, config=cfg, epsilon=0.0)
    df_eval = run_episodes(eval_policy, EVAL_SEEDS, config=cfg)
    summary = summarize_metrics(df_eval, "Q-Learning (Final)")
    print(" Final Evaluation Result:")
    for k, v in summary.items():
        if isinstance(v, float):
            print(f"   {k}: {v:.2f}")

    # Analyze learned lighting actions in dark awake states (Point 5)
    analyze_learned_lighting(q_table, cfg)


if __name__ == "__main__":
    main()
