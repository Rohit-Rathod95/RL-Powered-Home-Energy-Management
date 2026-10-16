from __future__ import annotations

import numpy as np
import pytest

from config import EnvConfig
from demo_logic import (
    PLACEHOLDER_BANNER,
    q_table_action_policy,
    q_update_preview,
    replay_episode,
    search_scenarios,
)
from evaluate import EVAL_SEEDS, TRAIN_SEEDS, VALIDATION_SEEDS, run_episodes
from qlearning import get_num_states
from train import QLearningPolicy
from baselines import AlwaysOffPolicy


def test_replay_matches_evaluation_total_reward() -> None:
    config = EnvConfig()
    q_table = np.random.default_rng(7).normal(size=(get_num_states(config), 10))
    replay_policy = q_table_action_policy(q_table, config)
    evaluated_policy = QLearningPolicy(q_table.copy(), config=config, epsilon=0.0,
                                       rng=np.random.default_rng(42))
    seeds = list(EVAL_SEEDS)[:3]
    replay_rewards = [replay_episode(replay_policy, seed, config).metrics["total_reward"] for seed in seeds]
    evaluated = run_episodes(evaluated_policy, seeds, config)
    np.testing.assert_allclose(replay_rewards, evaluated["total_reward"].to_numpy(), atol=1e-6)


def test_reward_breakdown_equals_environment_reward() -> None:
    result = replay_episode(AlwaysOffPolicy(), 10000, EnvConfig())
    for row in result.rows:
        info = row["info"]
        expected = (
            -info["cost"]
            - EnvConfig().lambda_t * info["D_T"]
            - EnvConfig().lambda_l * info["D_L"]
            - EnvConfig().lambda_s * info["switched"]
        )
        assert row["reward"] == pytest.approx(expected, abs=1e-12)


def test_q_update_preview_is_copy_and_bootstraps_truncation() -> None:
    table = np.zeros((2, 10), dtype=float)
    table[0, 3] = 2.0
    table[1, 4] = 8.0
    original = table.copy()
    preview = q_update_preview(table, 0, 3, -2.0, 1, False, 0.1, 0.9)
    assert preview["gamma_next_max"] == pytest.approx(7.2)
    assert preview["target"] == pytest.approx(5.2)
    assert preview["after"] == pytest.approx(2.32)
    np.testing.assert_array_equal(table, original)


def test_scenario_search_uses_only_eval_seeds() -> None:
    config = EnvConfig()
    q_table = np.zeros((get_num_states(config), 10))
    scenarios = search_scenarios(q_table, config, list(EVAL_SEEDS)[:12])
    assert all(item["seed"] in set(EVAL_SEEDS) for item in scenarios)
    assert not any(item["seed"] in set(TRAIN_SEEDS) | set(VALIDATION_SEEDS) for item in scenarios)
    assert PLACEHOLDER_BANNER.startswith("Simulator with uncalibrated")
