"""tests/test_qlearning.py

Unit tests for Q-learning scaffolding (B6).

Tests covered:
  1. State indexing round-trip: state_to_index() and index_to_state() are exact inverses.
  2. Epsilon = 0 picks argmax (skipped until select_action stub is implemented by user).
  3. Epsilon = 1 covers all 10 actions (skipped until select_action stub is implemented by user).
  4. Hand-computed q_update example (skipped until q_update stub is implemented by user).
  5. Training determinism for fixed seed (skipped until stubs are implemented by user).
"""

from __future__ import annotations

import numpy as np
import pytest

from config import EnvConfig
from discretize import get_state_space_dims
from qlearning import (
    get_num_states,
    index_to_state,
    init_q_table,
    q_update,
    select_action,
    state_to_index,
)


def test_state_indexing_bijection() -> None:
    """state_to_index and index_to_state must be exact mutual inverses for all states."""
    config = EnvConfig()
    dims = get_state_space_dims(config)
    n_states = get_num_states(config)

    # Test sample of 100 states across the space
    rng = np.random.default_rng(42)
    sample_indices = rng.integers(0, n_states, size=100)

    for idx in sample_indices:
        state = index_to_state(int(idx), config)
        reconstructed_idx = state_to_index(state, config)
        assert reconstructed_idx == idx, f"Failed round-trip: {idx} -> {state} -> {reconstructed_idx}"

        reconstructed_state = index_to_state(reconstructed_idx, config)
        assert reconstructed_state == state, f"Failed state reconstruction: {state} != {reconstructed_state}"


def test_epsilon_zero_picks_argmax() -> None:
    """When epsilon = 0, select_action must always pick the argmax action."""
    q_table = np.zeros((10, 10))
    s_idx = 3
    q_table[s_idx, 7] = 10.5  # Maximum action is 7
    rng = np.random.default_rng(42)

    try:
        action = select_action(q_table, s_idx, epsilon=0.0, rng=rng)
        assert action == 7, f"Expected action 7, got {action}"
    except NotImplementedError:
        pytest.skip("select_action() stub not yet implemented by user (Part B).")


def test_epsilon_one_covers_all_actions() -> None:
    """When epsilon = 1.0, actions must be uniformly sampled and cover all 10 actions."""
    q_table = np.zeros((5, 10))
    s_idx = 0
    rng = np.random.default_rng(123)

    try:
        actions = set(select_action(q_table, s_idx, epsilon=1.0, rng=rng) for _ in range(500))
        assert actions == set(range(10)), f"Did not cover all 10 actions: covered {actions}"
    except NotImplementedError:
        pytest.skip("select_action() stub not yet implemented by user (Part B).")


def test_hand_computed_q_update() -> None:
    """Verify hand-computed tabular Bellman updates for both non-terminal and terminal cases."""
    q_table = np.zeros((2, 10))
    s_idx = 0
    a = 2
    q_table[s_idx, a] = 5.0

    s_next_idx = 1
    q_table[s_next_idx, 4] = 8.0  # max Q(s', a') = 8.0

    r = -2.0
    alpha = 0.1
    gamma = 0.9

    try:
        # 1. Non-terminal update (bootstrap)
        # Target: y = r + gamma * max_a' Q(s', a') = -2.0 + 0.9 * 8.0 = 5.2
        # Q_new = 5.0 + 0.1 * (5.2 - 5.0) = 5.02
        q_update(
            q_table=q_table,
            s_idx=s_idx,
            a=a,
            r=r,
            s_next_idx=s_next_idx,
            terminated=False,
            alpha=alpha,
            gamma=gamma,
        )
        assert q_table[s_idx, a] == pytest.approx(5.02, abs=1e-5)

        # 2. Terminal update (no bootstrap)
        # Target: y = r = -2.0
        # Q_new = 5.02 + 0.1 * (-2.0 - 5.02) = 5.02 - 0.702 = 4.318
        q_update(
            q_table=q_table,
            s_idx=s_idx,
            a=a,
            r=r,
            s_next_idx=s_next_idx,
            terminated=True,
            alpha=alpha,
            gamma=gamma,
        )
        assert q_table[s_idx, a] == pytest.approx(4.318, abs=1e-5)

    except NotImplementedError:
        pytest.skip("q_update() stub not yet implemented by user (Part B).")


def test_training_determinism_for_fixed_seed() -> None:
    """Training with the same agent_seed must produce identical Q-tables."""
    from train import train
    try:
        q1, _, _ = train(total_episodes=5, agent_seed=777)
        q2, _, _ = train(total_episodes=5, agent_seed=777)
        np.testing.assert_array_equal(q1, q2)
    except NotImplementedError:
        pytest.skip("Q-learning training stubs not yet implemented by user (Part B).")
