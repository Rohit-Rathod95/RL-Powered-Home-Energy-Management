"""dashboard.py -- Home Electricity Saver demo.   Run:  streamlit run dashboard.py"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from config import COOLING_MODES, EnvConfig
from env import RoomEnv, decode_action, encode_action
from evaluate import EVAL_SEEDS
from train import QLearningPolicy

CFG = EnvConfig()
ACTIVITY = {0: "away", 1: "awake", 2: "asleep"}
DAY = {0: "workday", 1: "weekend"}
REGIME = {0: "dry", 1: "humid"}

st.set_page_config(page_title="Home Electricity Saver", layout="wide")
st.title("Home Electricity Saver: tabular Q-learning for a single room")
st.warning(
    "All numbers come from a simulator with uncalibrated placeholder parameters. "
    "They show how the methods compare inside the simulation, not measured savings."
)


class FixedPolicy:
    def __init__(self, action: int) -> None:
        self.action = action

    def select_action(self, state):
        return self.action


def load_policies() -> dict:
    policies = {}
    if Path("q_table.npy").exists():
        policies["Q-learning (trained)"] = QLearningPolicy(
            np.load("q_table.npy"), config=CFG, epsilon=0.0
        )
    policies["Always off"] = FixedPolicy(encode_action(0, 0))
    policies["Always on (AC + light)"] = FixedPolicy(encode_action(4, 1))  # mode 4 = ac_fan_low
    return policies


@st.cache_data
def seeds_for(day_type: int, regime: int) -> list[int]:
    env = RoomEnv(CFG)
    out = []
    for s in EVAL_SEEDS:
        _, info = env.reset(seed=s)
        if info["day_type"] == day_type and info["weather_regime"] == regime:
            out.append(int(s))
    return out


def run_day(policy, seed: int) -> pd.DataFrame:
    env = RoomEnv(CFG)
    state, _ = env.reset(seed=seed)
    rows, truncated = [], False
    while not truncated:
        action = policy.select_action(state)
        state, reward, _, truncated, info = env.step(action)
        mode, light = decode_action(action)
        rows.append({
            "hour": (info["step"] - 1) * CFG.dt,
            "temp": info["indoor_temp"],
            "t_eff": info["effective_temp"],
            "activity": info["activity"],
            "mode_idx": mode,
            "mode": COOLING_MODES[mode],
            "light": light,
            "energy": info["energy_kwh"],
            "cost": info["cost"],
            "d_t": info["D_T"],
            "switched": bool(info["switched"]),
            "reward": reward,
        })
    return pd.DataFrame(rows)


def plot_day(df: pd.DataFrame, title: str):
    fig, (a1, a2, a3) = plt.subplots(
        3, 1, figsize=(7, 8), sharex=True, gridspec_kw={"height_ratios": [3, 2, 2]}
    )
    x = df["hour"] + CFG.dt
    a1.axhspan(CFG.t_comfort_lo, CFG.t_comfort_hi, color="green", alpha=0.12, label="comfort band")
    a1.plot(x, df["temp"], label="indoor temp")
    a1.plot(x, df["t_eff"], "--", label="felt (effective) temp")
    for h in df.loc[df["activity"] == 0, "hour"]:
        for a in (a1, a2, a3):
            a.axvspan(h, h + CFG.dt, color="grey", alpha=0.15, lw=0)
    a1.set_ylabel("deg C")
    a1.set_title(title + "  (grey = household away)")
    a1.legend(fontsize=7, loc="upper left")
    a2.step(df["hour"], df["mode_idx"], where="post")
    on = df[df["light"] == 1]
    a2.scatter(
        on["hour"] + CFG.dt / 2, [-0.8] * len(on), marker="s", s=14, color="orange", label="light on"
    )
    a2.set_yticks(range(len(COOLING_MODES)))
    a2.set_yticklabels(list(COOLING_MODES), fontsize=7)
    a2.legend(fontsize=7, loc="upper left")
    a2.set_ylabel("cooling mode")
    a3.plot(x, df["energy"].cumsum())
    a3.set_ylabel("cumulative kWh")
    a3.set_xlabel("hour of day")
    fig.tight_layout()
    return fig


tab_results, tab_replay, tab_policy = st.tabs(["Results", "Day replay", "Learned policy"])

with tab_results:
    st.subheader("Held-out evaluation (200 days)")
    if Path("results.csv").exists():
        st.caption("Baselines, from run_baselines.py")
        st.dataframe(pd.read_csv("results.csv").round(2), use_container_width=True)
    if Path("multi_seed_summary.csv").exists():
        st.caption("Q-learning, independent training runs, from multi_seed.py")
        st.dataframe(pd.read_csv("multi_seed_summary.csv").round(2), use_container_width=True)
    if Path("lambda_sweep_raw.csv").exists():
        st.caption("Effect of the comfort weight lambda_T (mean over agents), from lambda_sweep.py")
        sweep = pd.read_csv("lambda_sweep_raw.csv").groupby("lambda_t").mean(numeric_only=True)
        st.dataframe(sweep.drop(columns="agent_seed").round(2), use_container_width=True)
    cols = st.columns(3)
    for col, f in zip(cols, ["pareto.png", "learning_curve.png", "lambda_sweep.png"]):
        if Path(f).exists():
            col.image(f, caption=f, use_container_width=True)

with tab_replay:
    c1, c2, c3 = st.columns(3)
    day_type = c1.selectbox("Day type", list(DAY), format_func=DAY.get)
    regime = c2.selectbox("Weather", list(REGIME), format_func=REGIME.get)
    seeds = seeds_for(day_type, regime)
    idx = c3.slider("Evaluation day", 1, max(1, len(seeds)), 1)
    policies = load_policies()
    default = list(policies)[:2]
    chosen = st.multiselect("Compare policies", list(policies), default=default)
    if seeds and chosen:
        seed = seeds[idx - 1]
        st.caption(f"Evaluation seed {seed}. Same day for every policy.")
        for col, name in zip(st.columns(len(chosen)), chosen):
            df = run_day(policies[name], seed)
            with col:
                m = st.columns(4)
                m[0].metric("kWh", f"{df['energy'].sum():.2f}")
                m[1].metric("Cost (INR)", f"{df['cost'].sum():.1f}")
                m[2].metric("Discomfort", f"{df['d_t'].sum():.1f}")
                m[3].metric("Reward", f"{df['reward'].sum():.1f}")
                fig = plot_day(df, name)
                st.pyplot(fig)
                plt.close(fig)

with tab_policy:
    if not Path("policy.csv").exists():
        st.info("Run python train.py to create policy.csv.")
    else:
        pol = pd.read_csv("policy.csv")
        c1, c2, c3, c4 = st.columns(4)
        d = c1.selectbox("Day type ", list(DAY), format_func=DAY.get)
        r = c2.selectbox("Weather ", list(REGIME), format_func=REGIME.get)
        a = c3.selectbox("Activity", list(ACTIVITY), index=0, format_func=ACTIVITY.get)
        minv = c4.slider("Min. training visits", 0, 200, 20)
        sub = pol[(pol.day_type == d) & (pol.weather_regime == r) & (pol.activity == a)].copy()
        sub["cell"] = np.where(sub["total_state_visits"] >= minv, sub["action_name"], "-")
        table = sub.pivot(index="time_block", columns="temp_bin", values="cell")
        edges = (-np.inf,) + tuple(CFG.temp_bins) + (np.inf,)
        table.columns = [f"{edges[int(b)]:g} to {edges[int(b) + 1]:g} C" for b in table.columns]
        table.index = [f"{int(b * CFG.time_block_hours):02d}:00" for b in table.index]
        st.caption("Greedy action per (time of day, indoor temperature). '-' = rarely or never visited.")
        st.dataframe(table, use_container_width=True)
