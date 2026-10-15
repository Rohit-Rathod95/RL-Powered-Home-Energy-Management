"""plot_training.py

Plot validation learning curves with baseline reference lines.

Implements B5:
  - Plots validation reward vs training episode from curves.csv.
  - Adds horizontal reference lines for Always On, Tuned Threshold, and Tuned Precool.
  - Exports learning_curve.png.
"""

from __future__ import annotations

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def get_baseline_reward(controller_name: str, results_csv: str = "results.csv", default_val: float = -200.0) -> float:
    """Retrieve mean reward for a controller from results.csv if available."""
    if os.path.exists(results_csv):
        try:
            df = pd.read_csv(results_csv)
            row = df.loc[df["controller"] == controller_name]
            if not row.empty:
                return float(row["total_reward_mean"].iloc[0])
        except Exception:
            pass
    return default_val


def plot_learning_curve(
    curves_csv: str = "curves.csv",
    output_png: str = "learning_curve.png",
    results_csv: str = "results.csv",
) -> None:
    """Plot validation learning curve with baseline reference lines."""
    if not os.path.exists(curves_csv):
        print(f"[NOTE] {curves_csv} does not exist yet. Run train.py first.")
        return

    df = pd.read_csv(curves_csv)
    if df.empty or "episode" not in df.columns:
        print(f"[NOTE] {curves_csv} contains no training points.")
        return

    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

    episodes = df["episode"]
    val_mean = df["val_reward_mean"]
    val_std = df["val_reward_std"]

    # Shaded std area
    ax.fill_between(
        episodes,
        val_mean - val_std,
        val_mean + val_std,
        color="#3498db",
        alpha=0.2,
        label="Validation Reward ± 1 std",
    )

    # Main learning curve
    ax.plot(episodes, val_mean, color="#2980b9", linewidth=2.0, label="Q-Learning Greedy Policy", marker="o", markersize=4)

    # Baseline reference lines
    r_always_on = get_baseline_reward("Always On (AC + Light)", results_csv, default_val=-238.64)
    r_tuned_threshold = get_baseline_reward("Tuned Threshold", results_csv, default_val=-85.21)
    r_tuned_precool = get_baseline_reward("Tuned Precool (Block)", results_csv, default_val=-80.0)

    ax.axhline(
        r_always_on,
        color="#e74c3c",
        linestyle="--",
        linewidth=1.5,
        alpha=0.8,
        label=f"Always On ({r_always_on:.1f})",
    )
    ax.axhline(
        r_tuned_threshold,
        color="#2ecc71",
        linestyle="--",
        linewidth=1.5,
        alpha=0.8,
        label=f"Tuned Threshold ({r_tuned_threshold:.1f})",
    )
    ax.axhline(
        r_tuned_precool,
        color="#f39c12",
        linestyle="--",
        linewidth=1.5,
        alpha=0.8,
        label=f"Tuned Precool ({r_tuned_precool:.1f})",
    )

    ax.set_xlabel("Training Episodes", fontsize=11, fontweight="bold")
    ax.set_ylabel("Validation Reward", fontsize=11, fontweight="bold")
    ax.set_title("Tabular Q-Learning Progression vs. Baselines", fontsize=13, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="lower right", framealpha=0.9, fontsize=9.5)

    plt.tight_layout()
    plt.savefig(output_png, dpi=300, bbox_inches="tight")
    plt.close()
    print(f" Learning curve plot successfully saved to: {output_png}")


if __name__ == "__main__":
    plot_learning_curve()
