"""multi_seed.py -- train N independent agents and evaluate each on the held-out days."""
import argparse
import time

import numpy as np
import pandas as pd

from config import EnvConfig
from evaluate import EVAL_SEEDS, run_episodes
from train import QLearningPolicy, train


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--agents", type=int, default=5)
    parser.add_argument("--episodes", type=int, default=20000)
    args = parser.parse_args()

    cfg = EnvConfig()
    summary = []
    for i in range(args.agents):
        agent_seed = 1000 + i
        t0 = time.time()
        q, visits, _ = train(
            total_episodes=args.episodes,
            val_eval_freq=2500,
            agent_seed=agent_seed,
            config=cfg,
        )
        np.save(f"q_table_agent{agent_seed}.npy", q)
        np.save(f"visits_agent{agent_seed}.npy", visits)
        df = run_episodes(QLearningPolicy(q, config=cfg, epsilon=0.0), EVAL_SEEDS, config=cfg)
        df.to_csv(f"eval_agent{agent_seed}.csv", index=False)
        row = {
            "agent_seed": agent_seed,
            "reward": df["total_reward"].mean(),
            "energy_kwh": df["energy_kwh"].mean(),
            "d_t": df["total_d_t"].mean(),
            "minutes": (time.time() - t0) / 60,
        }
        summary.append(row)
        print(
            f"\n>>> agent {agent_seed}: reward {row['reward']:.2f}, "
            f"energy {row['energy_kwh']:.2f} kWh, took {row['minutes']:.1f} min\n"
        )

    out = pd.DataFrame(summary)
    out.to_csv("multi_seed_summary.csv", index=False)
    print(out.to_string(index=False))
    print(
        f"\nMean reward over {args.agents} agents: {out['reward'].mean():.2f} "
        f"+/- {out['reward'].std(ddof=1):.2f} (std across agents)"
    )


if __name__ == "__main__":
    main()
