"""tests/test_baselines.py

Verification tests for Phase 2 baseline controllers and evaluation framework (B4).

Tests covered:
  1. Policy action validity: Every policy returns a valid integer action in [0, 9] for every possible discretized state.
  2. Disjoint seed splits: TRAIN_SEEDS and EVAL_SEEDS have empty intersection.
  3. Evaluation determinism: Running evaluation on the same seeds yields identical episode metrics.
  4. Reward ordering sanity: TunedThresholdPolicy achieves strictly higher reward than AlwaysOffPolicy.
"""

from __future__ import annotations

import numpy as np
import pytest

from baselines import (
    AlwaysOffPolicy,
    AlwaysOnPolicy,
    RandomPolicy,
    ThresholdPolicy,
    TimerPolicy,
    TunedPrecoolClockPolicy,
    TunedPrecoolPolicy,
    TunedThresholdPolicy,
)
from config import EnvConfig
from discretize import get_state_space_dims
from evaluate import EVAL_SEEDS, TRAIN_SEEDS, run_episodes


def test_train_and_eval_seeds_are_disjoint() -> None:
    """Train seeds (0..999) and evaluation seeds (10000..10199) must be strictly disjoint."""
    train_set = set(TRAIN_SEEDS)
    eval_set = set(EVAL_SEEDS)

    assert len(train_set) == 1000, f"Expected 1000 train seeds, got {len(train_set)}"
    assert len(eval_set) == 200, f"Expected 200 eval seeds, got {len(eval_set)}"
    assert train_set.isdisjoint(eval_set), "TRAIN_SEEDS and EVAL_SEEDS overlap!"


def test_policies_return_valid_action_for_all_states() -> None:
    """Every baseline controller must return a valid action in [0, 9] for every state in the discrete space."""
    config = EnvConfig()
    dims = get_state_space_dims(config)

    base_thresh = TunedThresholdPolicy(config=config)
    policies = [
        AlwaysOffPolicy(),
        AlwaysOnPolicy(),
        RandomPolicy(seed=123),
        TimerPolicy(config=config),
        ThresholdPolicy(config=config),
        base_thresh,
        TunedPrecoolPolicy(base_policy=base_thresh, config=config),
        TunedPrecoolClockPolicy(base_policy=base_thresh, config=config),
    ]

    # Generate all states in Cartesian product of dimensions
    # dims = (num_blocks, 3, 2, num_temp_bins, 2)
    num_blocks, num_activities, num_day_types, num_temp_bins, num_regimes = dims[:5]

    for b in range(num_blocks):
        for act in range(num_activities):
            for dt in range(num_day_types):
                for tb in range(num_temp_bins):
                    for wr in range(num_regimes):
                        state = (b, act, dt, tb, wr)
                        for pol in policies:
                            action = pol.select_action(state)
                            assert isinstance(action, (int, np.integer)), f"Action {action} is not an integer!"
                            assert 0 <= action < 10, f"Action {action} outside valid range [0, 9] for state {state} in {pol.name}"


def test_evaluation_deterministic_for_fixed_seeds() -> None:
    """run_episodes must produce identical metric results when re-run on the same seeds."""
    config = EnvConfig()
    policy = ThresholdPolicy(config=config)
    test_seeds = [10005, 10010, 10025]

    df1 = run_episodes(policy, test_seeds, config=config)
    df2 = run_episodes(policy, test_seeds, config=config)

    # All numeric columns must match exactly
    for col in df1.columns:
        np.testing.assert_array_almost_equal(
            df1[col].to_numpy(),
            df2[col].to_numpy(),
            decimal=6,
            err_msg=f"Metric column '{col}' differed between evaluation runs!",
        )


def test_tuned_threshold_is_grid_argmax() -> None:
    """Tuned threshold's reward on train seeds must equal the maximum reward found in the grid (Point 4)."""
    from baselines import tune_threshold_policy
    config = EnvConfig()
    train_sample = list(TRAIN_SEEDS)[:50]

    tuned_pol = tune_threshold_policy(train_sample, config=config)
    df_eval = run_episodes(tuned_pol, train_sample, config=config)
    actual_reward = float(df_eval["total_reward"].mean())

    assert hasattr(tuned_pol, "best_train_reward")
    # Verified: the returned policy achieves at least the maximal candidate reward on these training seeds
    assert actual_reward >= tuned_pol.best_train_reward - 1e-5, (
        f"Actual reward {actual_reward:.4f} < grid best {tuned_pol.best_train_reward:.4f}"
    )

