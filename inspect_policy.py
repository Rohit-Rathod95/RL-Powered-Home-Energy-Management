"""inspect_policy.py -- what did the trained agent actually learn? Needs q_table.npy, policy.csv."""
from collections import Counter

import numpy as np
import pandas as pd

from config import ACTIVITY_AWAY, COOLING_MODES, EnvConfig
from env import RoomEnv, decode_action
from evaluate import EVAL_SEEDS
from qlearning import state_to_index
from train import QLearningPolicy

MIN_VISITS = 20


def main() -> None:
    cfg = EnvConfig()
    q = np.load("q_table.npy")
    df = pd.read_csv("policy.csv")

    # 1. Behaviour while the house is still empty, just before the 18:00 workday return.
    block = int(16 / cfg.time_block_hours)
    sub = df[
        (df.time_block == block)
        & (df.activity == ACTIVITY_AWAY)
        & (df.day_type == 0)
        & (df.total_state_visits >= MIN_VISITS)
    ]
    print(f"\nWorkday, block {block} (16:00-18:00), household away, states visited >= {MIN_VISITS}x:")
    print(
        sub[
            ["temp_bin", "weather_regime", "action_name", "action_visits", "total_state_visits"]
        ].to_string(index=False)
    )

    # 2. How much of the evaluation does the agent spend in rarely-visited states?
    visits = df["total_state_visits"].to_numpy()
    env = RoomEnv(cfg)
    policy = QLearningPolicy(q, config=cfg, epsilon=0.0)
    low = total = 0
    modes: Counter[str] = Counter()
    for seed in EVAL_SEEDS:
        state, _ = env.reset(seed=seed)
        truncated = False
        while not truncated:
            s_idx = state_to_index(state, cfg)
            total += 1
            low += int(visits[s_idx] < MIN_VISITS)
            action = policy.select_action(state)
            modes[COOLING_MODES[decode_action(action)[0]]] += 1
            state, _, _, truncated, _ = env.step(action)

    print(
        f"\nEval steps in states visited < {MIN_VISITS}x during training: "
        f"{low}/{total} ({100 * low / total:.1f}%)"
    )
    print("Cooling mode usage on eval days:")
    for name, n in modes.most_common():
        print(f"   {name:<12} {100 * n / total:5.1f}%")


if __name__ == "__main__":
    main()
