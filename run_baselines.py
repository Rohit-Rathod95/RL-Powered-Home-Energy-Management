"""run_baselines.py

Execute baseline evaluation, generate results table, export CSV, and plot Pareto front.

Implements B3 and A1-A4:
  - Tunes TunedThresholdPolicy on TRAIN_SEEDS (0..999).
  - Tunes TunedPrecoolPolicy and TunedPrecoolClockPolicy on TRAIN_SEEDS (0..999).
  - Evaluates all controllers on disjoint EVAL_SEEDS (10000..10199):
      1. Always Off
      2. Always On (AC + Light)
      3. Random (seeded)
      4. Timer
      5. Threshold Rule
      6. Tuned Threshold
      7. Tuned Precool (Block)
      8. Tuned Precool (Clock)
  - Prints formatted results table including away energy, light-needed-but-off, mode switches,
    and degree-hours above comfort.
  - Saves results.csv and pareto.png.
  - Fails loudly if always_off beats tuned_threshold.
  - Warns if tuned_precool does not beat tuned_threshold.
"""

from __future__ import annotations

import sys
import matplotlib
matplotlib.use("Agg")  # Non-interactive background backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from baselines import (
    AlwaysOffPolicy,
    AlwaysOnPolicy,
    RandomPolicy,
    ThresholdPolicy,
    TimerPolicy,
    tune_precool_clock_policy,
    tune_precool_policy,
    tune_threshold_policy,
)
from config import EnvConfig
from discretize import get_state_space_dims
from evaluate import EVAL_SEEDS, TRAIN_SEEDS, run_episodes, summarize_metrics


def print_state_space_info(config: EnvConfig) -> None:
    """Print time_block_hours, temp_bins edges, and total discrete state space size (A4)."""
    dims = get_state_space_dims(config)
    n_states = int(np.prod(dims))

    print("\n" + "=" * 80)
    print(" STATE SPACE CONFIGURATION (A4)")
    print("=" * 80)
    print(f"  Time Block Hours:        {config.time_block_hours} h  ({int(24.0 / config.time_block_hours)} blocks/day)")
    print(f"  Temperature Bin Edges:   {config.temp_bins}  ({len(config.temp_bins) + 1} bins)")
    print(f"  State Dimensions:        {dims}")
    print(f"  Total Discrete States:   {n_states}")
    print("=" * 80 + "\n")


def run_all_baselines(config: EnvConfig | None = None) -> tuple[pd.DataFrame, list[dict], dict]:
    """Tune policies on TRAIN_SEEDS and evaluate all controllers on EVAL_SEEDS.

    Returns:
        tuple[pd.DataFrame, list[dict], dict]: (summary_df, summaries, all_dfs)
            all_dfs maps controller name -> per-episode DataFrame for paired comparisons.
    """
    cfg = config if config is not None else EnvConfig()

    print_state_space_info(cfg)

    print("=" * 80)
    print(" 1. Tuning Controllers on TRAIN_SEEDS (0..999)...")
    tuned_threshold = tune_threshold_policy(TRAIN_SEEDS, config=cfg)
    print(f"    Tuned Threshold Parameters: bin_2={tuned_threshold.mode_bin_2}, "
          f"bin_3={tuned_threshold.mode_bin_3}, bin_4_dry={tuned_threshold.mode_bin_4_dry}, "
          f"bin_4_humid={tuned_threshold.mode_bin_4_humid}, "
          f"precool_mode={tuned_threshold.precool_mode}")
    if hasattr(tuned_threshold, "best_train_reward"):
        print(f"    Tuned Threshold Best Train Reward: {tuned_threshold.best_train_reward:.4f}")

    tuned_precool = tune_precool_policy(TRAIN_SEEDS, base_policy=tuned_threshold, config=cfg)
    print(f"    Tuned Precool (Block): lead_steps={tuned_precool.lead_steps} "
          f"(lead_blocks={tuned_precool.lead_blocks}), trigger_bin={tuned_precool.trigger_bin}, "
          f"mode={tuned_precool.precool_mode}")

    tuned_precool_clock = tune_precool_clock_policy(TRAIN_SEEDS, base_policy=tuned_threshold, config=cfg)
    print(f"    Tuned Precool (Clock): lead_steps={tuned_precool_clock.lead_steps}, "
          f"trigger_bin={tuned_precool_clock.trigger_bin}, mode={tuned_precool_clock.precool_mode}")
    print("=" * 80)

    controllers = [
        AlwaysOffPolicy(),
        AlwaysOnPolicy(),
        RandomPolicy(seed=42),
        TimerPolicy(config=cfg),
        ThresholdPolicy(config=cfg),
        tuned_threshold,
        tuned_precool,
        tuned_precool_clock,
    ]

    summaries: list[dict] = []
    all_dfs: dict[str, pd.DataFrame] = {}

    print(f"\n 2. Evaluating {len(controllers)} controllers on EVAL_SEEDS (10000..10199)...")
    for ctrl in controllers:
        name = ctrl.name
        print(f"    Running: {name}...")
        df_ep = run_episodes(ctrl, EVAL_SEEDS, config=cfg)
        all_dfs[name] = df_ep
        summary = summarize_metrics(df_ep, name)
        summaries.append(summary)

    summary_df = pd.DataFrame(summaries)
    return summary_df, summaries, all_dfs


def print_results_table(summary_df: pd.DataFrame) -> None:
    """Print results table with extended metrics (A2)."""
    print("\n" + "=" * 165)
    print(" PHASE 2 BASELINE EVALUATION RESULTS (200 Held-Out Days, Seeds 10000..10199)")
    print("=" * 165)
    header = (
        f"{'Controller':<23} | {'Energy (kWh)':<15} | {'Cost (INR)':<15} | {'Discomfort D_T':<18} | "
        f"{'Deg-h > Comf':<15} | {'Away E (kWh)':<14} | {'Light Off %':<12} | {'Switches':<12} | {'Total Reward':<18}"
    )
    print(header)
    print("-" * 165)

    for _, row in summary_df.iterrows():
        line = (
            f"{row['controller']:<23} | "
            f"{row['energy_kwh_mean']:>6.2f} +/- {row['energy_kwh_std']:<5.2f} | "
            f"{row['cost_mean']:>6.2f} +/- {row['cost_std']:<5.2f} | "
            f"{row['total_d_t_mean']:>8.2f} +/- {row['total_d_t_std']:<6.2f} | "
            f"{row['degree_hours_above_comfort_mean']:>6.2f} +/- {row['degree_hours_above_comfort_std']:<5.2f} | "
            f"{row['energy_while_away_mean']:>5.2f} +/- {row['energy_while_away_std']:<5.2f} | "
            f"{row['light_needed_but_off_share_mean']*100:>4.1f}% +/- {row['light_needed_but_off_share_std']*100:<3.1f}% | "
            f"{row['switches_per_day_mean']:>5.1f} +/- {row['switches_per_day_std']:<4.1f} | "
            f"{row['total_reward_mean']:>8.2f} +/- {row['total_reward_std']:<6.2f}"
        )
        print(line)
    print("=" * 165)


def plot_pareto_front(summary_df: pd.DataFrame, output_path: str = "pareto.png") -> None:
    """Generate and save Pareto trade-off curve (Daily Energy vs Thermal Discomfort)."""
    fig, ax = plt.subplots(figsize=(10, 6.5), dpi=300)

    colors = ["#e74c3c", "#3498db", "#9b59b6", "#e67e22", "#2ecc71", "#1abc9c", "#34495e", "#d35400"]
    markers = ["o", "s", "^", "D", "v", "P", "X", "*"]

    for i, (_, row) in enumerate(summary_df.iterrows()):
        x = row["energy_kwh_mean"]
        y = row["total_d_t_mean"]
        xerr = row["energy_kwh_std"]
        yerr = row["total_d_t_std"]
        name = row["controller"]

        ax.errorbar(
            x, y, xerr=xerr, yerr=yerr,
            fmt=markers[i % len(markers)],
            color=colors[i % len(colors)],
            ecolor=colors[i % len(colors)],
            elinewidth=1.2,
            capsize=3,
            markersize=9 if markers[i % len(markers)] != "*" else 12,
            label=name,
            alpha=0.9,
        )

        offset_y = 12.0 if y < 400 else -20.0
        ax.annotate(
            name,
            (x, y),
            textcoords="offset points",
            xytext=(4, offset_y),
            fontsize=8.5,
            fontweight="semibold",
            color=colors[i % len(colors)],
        )

    ax.set_xlabel("Daily Energy Consumption (kWh)", fontsize=11, fontweight="bold")
    ax.set_ylabel(r"Daily Thermal Discomfort $D_T$ ($^\circ\mathrm{C}^2$)", fontsize=11, fontweight="bold")
    ax.set_title("Energy vs. Thermal Discomfort Trade-off (Pareto Front)", fontsize=13, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="upper right", framealpha=0.9, fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f" Pareto plot successfully saved to: {output_path}")


def print_paired_comparisons(all_dfs: dict[str, pd.DataFrame]) -> None:
    """Print paired comparisons with mean difference, SE, and 95% CI (Point 3)."""
    from evaluate import compute_paired_differences

    pairs = [
        ("Tuned Precool (Block) vs Tuned Threshold", "Tuned Precool (Block)", "Tuned Threshold"),
        ("Tuned Precool (Clock) vs Tuned Precool (Block)", "Tuned Precool (Clock)", "Tuned Precool (Block)"),
        ("Always On (AC + Light) vs Tuned Precool (Block)", "Always On (AC + Light)", "Tuned Precool (Block)"),
    ]

    print("\n" + "=" * 105)
    print(" PAIRED REWARD COMPARISONS ACROSS IDENTICAL EVALUATION SEEDS (Point 3)")
    print("=" * 105)
    header = f"{'Comparison (A vs B)':<48} | {'Mean Diff':<12} | {'Std Error':<11} | {'95% CI Interval':<22}"
    print(header)
    print("-" * 105)

    for label, name_a, name_b in pairs:
        res = compute_paired_differences(all_dfs[name_a], all_dfs[name_b], metric="total_reward")
        mean_diff = res["mean_diff"]
        se_diff = res["se_diff"]
        ci_low = res["ci_95_lower"]
        ci_high = res["ci_95_upper"]

        line = (
            f"{label:<48} | "
            f"{mean_diff:>+9.2f}    | "
            f"{se_diff:>8.2f}    | "
            f"[{ci_low:>+8.2f}, {ci_high:>+8.2f}]"
        )
        print(line)
    print("=" * 105 + "\n")


def main() -> None:
    config = EnvConfig()
    summary_df, summaries, all_dfs = run_all_baselines(config)

    # Print results table
    print_results_table(summary_df)

    # Save results.csv
    csv_path = "results.csv"
    summary_df.to_csv(csv_path, index=False)
    print(f" Results successfully exported to: {csv_path}")

    # Save pareto.png
    plot_pareto_front(summary_df, output_path="pareto.png")

    # Print paired comparisons across identical eval seeds (Point 3)
    print_paired_comparisons(all_dfs)

    # Reward checks
    r_off = float(summary_df.loc[summary_df["controller"] == "Always Off", "total_reward_mean"].iloc[0])
    r_tuned = float(summary_df.loc[summary_df["controller"] == "Tuned Threshold", "total_reward_mean"].iloc[0])
    r_precool = float(summary_df.loc[summary_df["controller"] == "Tuned Precool (Block)", "total_reward_mean"].iloc[0])
    r_clock = float(summary_df.loc[summary_df["controller"] == "Tuned Precool (Clock)", "total_reward_mean"].iloc[0])

    print("=" * 80)
    print(" Reward Sanity Checks:")
    print(f"   Always Off Mean Reward:               {r_off:.2f}")
    print(f"   Tuned Threshold Mean Reward:          {r_tuned:.2f}")
    print(f"   Tuned Precool (Block) Mean Reward:    {r_precool:.2f}")
    print(f"   Tuned Precool (Clock) Mean Reward:    {r_clock:.2f}")
    print(f"   Delta (Tuned Threshold vs Off):       {r_tuned - r_off:+.2f}")
    print(f"   Delta (Tuned Precool vs Threshold):   {r_precool - r_tuned:+.2f}")
    print(f"   Delta (Clock Precool vs Threshold):   {r_clock - r_tuned:+.2f}")
    print("=" * 80)

    # 1. FAIL LOUDLY if always_off beats tuned_threshold
    if r_off >= r_tuned:
        sys.stderr.write(
            f"\n[FATAL ERROR] Broken reward weights: Always Off reward ({r_off:.2f}) "
            f">= Tuned Threshold reward ({r_tuned:.2f})!\n"
        )
        sys.exit(1)

    # 2. WARNING if tuned_precool does not beat tuned_threshold (A6)
    if r_precool <= r_tuned:
        print("\n[WARNING] tuned_precool mean reward does not beat tuned_threshold under evaluated distribution.")


if __name__ == "__main__":
    main()
