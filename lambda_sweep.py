"""lambda_sweep.py -- how does the learned policy change with the comfort weight lambda_T?"""
import argparse
from dataclasses import replace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from config import EnvConfig
from evaluate import EVAL_SEEDS, run_episodes
from train import QLearningPolicy, train

LAMBDAS = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--agents", type=int, default=3)
    parser.add_argument("--episodes", type=int, default=20000)
    args = parser.parse_args()

    base = EnvConfig()
    rows = []
    for lam in LAMBDAS:
        cfg = replace(base, lambda_t=lam)
        for k in range(args.agents):
            seed = 2000 + k
            q, _, _ = train(
                total_episodes=args.episodes,
                val_eval_freq=args.episodes,
                agent_seed=seed,
                config=cfg,
            )
            df = run_episodes(QLearningPolicy(q, config=cfg, epsilon=0.0), EVAL_SEEDS, config=cfg)
            rows.append({
                "lambda_t": lam,
                "agent_seed": seed,
                "energy_kwh": df["energy_kwh"].mean(),
                "cost": df["cost"].mean(),
                "d_t": df["total_d_t"].mean(),
                "deg_hours_above": df["degree_hours_above_comfort"].mean(),
                "away_kwh": df["energy_while_away"].mean(),
                "reward_own_lambda": df["total_reward"].mean(),
            })
            print(
                f"lambda_T={lam:<5} agent {seed}: energy {rows[-1]['energy_kwh']:.2f} kWh, "
                f"deg-h above band {rows[-1]['deg_hours_above']:.2f}"
            )

    raw = pd.DataFrame(rows)
    raw.to_csv("lambda_sweep_raw.csv", index=False)
    agg = raw.groupby("lambda_t").agg(["mean", "std"]).drop(columns="agent_seed")
    agg.to_csv("lambda_sweep_summary.csv")
    print("\n(reward_own_lambda is measured under each run's own lambda: do not compare across rows)")
    print(agg.round(2).to_string())

    g = raw.groupby("lambda_t")[["energy_kwh", "deg_hours_above"]].mean()
    plt.figure(figsize=(6, 4.5))
    plt.plot(g["energy_kwh"], g["deg_hours_above"], "o-")
    for lam, r in g.iterrows():
        plt.annotate(
            f"λ_T={lam}",
            (r["energy_kwh"], r["deg_hours_above"]),
            textcoords="offset points",
            xytext=(6, 6),
            fontsize=8,
        )
    plt.xlabel("Energy per day (kWh)")
    plt.ylabel("Degree-hours above comfort band per day")
    plt.title("Learned policies across comfort weights (3 agents each)")
    plt.tight_layout()
    plt.savefig("lambda_sweep.png", dpi=150)
    print("Saved lambda_sweep.png")


if __name__ == "__main__":
    main()
