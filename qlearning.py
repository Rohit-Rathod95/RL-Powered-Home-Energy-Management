"""qlearning.py

Tabular Q-learning state mapping, data structures, and algorithmic stubs.

Implements B1 & B2:
  - state_to_index(state, config): Mixed-radix bijection between discrete state tuple and flat integer index.
  - index_to_state(idx, config): Inverse mapping.
  - init_q_table(), init_visit_counts().
  - select_action(): Epsilon-greedy with random tie-breaking.
  - q_update(): In-place tabular Bellman update (bootstraps unless terminated).
"""

from __future__ import annotations

import numpy as np

from config import EnvConfig
from discretize import get_state_space_dims


def get_num_states(config: EnvConfig | None = None) -> int:
    """Return total number of states in the discrete state space."""
    cfg = config if config is not None else EnvConfig()
    dims = get_state_space_dims(cfg)
    return int(np.prod(dims))


def state_to_index(state: tuple[int, ...], config: EnvConfig | None = None) -> int:
    """Map a discrete state tuple to a unique flat index in [0, n_states - 1].

    Args:
        state: Discrete state tuple (e.g. (time_block, activity, day_type, temp_bin, weather_regime)).
        config: EnvConfig specifying state dimensions.

    Returns:
        int: Unique integer index in [0, n_states - 1].
    """
    cfg = config if config is not None else EnvConfig()
    dims = get_state_space_dims(cfg)
    return int(np.ravel_multi_index(state, dims))


def index_to_state(idx: int, config: EnvConfig | None = None) -> tuple[int, ...]:
    """Map a flat integer index back to its discrete state tuple.

    Args:
        idx: Integer index in [0, n_states - 1].
        config: EnvConfig specifying state dimensions.

    Returns:
        tuple[int, ...]: Discrete state tuple.
    """
    cfg = config if config is not None else EnvConfig()
    dims = get_state_space_dims(cfg)
    return tuple(int(x) for x in np.unravel_index(idx, dims))


def init_q_table(n_states: int, n_actions: int = 10) -> np.ndarray:
    """Initialize Q-table with zeros of shape (n_states, n_actions)."""
    return np.zeros((n_states, n_actions), dtype=np.float64)


def init_visit_counts(n_states: int, n_actions: int = 10) -> np.ndarray:
    """Initialize state-action visit count table N[s, a] with zeros."""
    return np.zeros((n_states, n_actions), dtype=np.int64)


# ============================================================================
# Epsilon-greedy selection and tabular Q-learning update (B2)
# ============================================================================
def select_action(
    q_table: np.ndarray,
    s_idx: int,
    epsilon: float,
    rng: np.random.Generator,
) -> int:
    """Select an action using an epsilon-greedy policy with random tie-breaking.

    Behavior:
      - With probability (1 - epsilon): Greedy action selection (argmax over actions).
        If multiple actions share the maximal Q-value, break ties uniformly at random
        among the maximal actions using `rng`.
      - With probability epsilon: Exploration action selected uniformly at random
        from all available actions [0, 9] using `rng`.

    Args:
        q_table: 2D numpy array of shape (num_states, 10) containing state-action values.
        s_idx: Integer flat state index in [0, num_states - 1].
        epsilon: Exploration probability in [0.0, 1.0].
        rng: np.random.Generator instance for reproducible randomness.

    Returns:
        int: Selected action index in [0, 9].
    """
    n_actions = q_table.shape[1]
    if rng.random() < epsilon:
        return int(rng.integers(n_actions))
    row = q_table[s_idx]
    best_actions = np.flatnonzero(row == row.max())
    return int(rng.choice(best_actions))


def q_update(
    q_table: np.ndarray,
    s_idx: int,
    a: int,
    r: float,
    s_next_idx: int,
    terminated: bool,
    alpha: float,
    gamma: float,
) -> None:
    """Perform in-place Tabular Q-learning Bellman update.

    Formula:
        Target y:
            y = r  if terminated is True
            y = r + gamma * max_a' Q(s_next_idx, a')  if terminated is False
        Update:
            Q(s_idx, a) <- Q(s_idx, a) + alpha * (y - Q(s_idx, a))

    Note on truncation:
        In our 48-step daily episodes, truncation (reaching step 48) is NOT true termination.
        At truncation, the episode continues into the next day via the wrapped state,
        so `terminated` is False and the TD update MUST bootstrap from max_a' Q(s_next_idx, a').

    Args:
        q_table: 2D numpy array of shape (num_states, 10) modified in-place.
        s_idx: Current flat state index in [0, num_states - 1].
        a: Action taken in [0, 9].
        r: Immediate scalar reward received from env.step(a).
        s_next_idx: Next flat state index in [0, num_states - 1].
        terminated: Boolean indicating whether the state is truly terminal.
        alpha: Learning rate parameter in (0, 1].
        gamma: Discount factor in [0, 1].
    """
    future = 0.0 if terminated else float(q_table[s_next_idx].max())
    target = r + gamma * future
    q_table[s_idx, a] += alpha * (target - q_table[s_idx, a])
