"""Professor-friendly interactive Home Electricity Saver demonstration.

Run with: streamlit run demo_app.py
"""

from __future__ import annotations

from pathlib import Path
import html

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from baselines import AlwaysOffPolicy, AlwaysOnPolicy
from config import (
    ACTIVITIES,
    ACTIVITY_AWAY,
    ACTIVITY_AWAKE,
    COOLING_MODES,
    DAY_TYPES,
    EnvConfig,
    REGIMES,
)
from demo_logic import (
    PLACEHOLDER_BANNER,
    action_words,
    load_baseline_params,
    make_baseline_policies,
    one_step_previews,
    q_table_action_policy,
    q_update_preview,
    replay_episode,
    search_scenarios,
    state_words,
    train_with_progress,
)
from discretize import discretize_state
from dynamics import compute_effective_temp
from env import RoomEnv, decode_action
from evaluate import EVAL_SEEDS
from qlearning import state_to_index
from schedule import is_dark
from train import QLearningPolicy


CFG = EnvConfig()
st.set_page_config(page_title="Home Electricity Saver Demo", layout="wide")
st.markdown(
    """
    <style>
    html, body, [class*="css"] { font-size: 18px; }
    h1 { font-size: 2.2rem; }
    .story-box { border: 2px solid #334155; border-radius: 12px; padding: 18px;
                 min-height: 145px; background: #0f172a; color: #f8fafc; }
    .story-box h3 { font-size: 28px; margin: 0 0 10px 0; }
    .story-icon { font-size: 30px; margin-right: 8px; }
    .status { font-size: 20px; padding: 10px 0; }
    </style>
    """,
    unsafe_allow_html=True,
)
st.title("Home Electricity Saver")
st.caption("Simulation with placeholder numbers: it shows how methods compare inside the simulation, not real savings.")


def _load_q_table() -> np.ndarray | None:
    path = Path("q_table.npy")
    return np.load(path) if path.exists() else None


def _matching_eval_seeds(day_type: int, regime: int) -> list[int]:
    env = RoomEnv(CFG)
    matches = []
    for seed in EVAL_SEEDS:
        _, info = env.reset(seed=int(seed))
        if info["day_type"] == day_type and info["weather_regime"] == regime:
            matches.append(int(seed))
    return matches


def _find_workday_seed() -> int:
    env = RoomEnv(CFG)
    for seed in EVAL_SEEDS:
        _, info = env.reset(seed=int(seed))
        if info["day_type"] == 0:
            return int(seed)
    raise RuntimeError("No workday was found in EVAL_SEEDS.")


def _init_state(seed: int) -> None:
    env = RoomEnv(CFG)
    state, _ = env.reset(seed=seed)
    st.session_state.demo_env = env
    st.session_state.demo_state = state
    st.session_state.demo_seed = seed
    st.session_state.demo_rows = []
    st.session_state.demo_last = None
    st.session_state.playing = False


def _advance(policy: QLearningPolicy, q_table: np.ndarray | None) -> None:
    env: RoomEnv = st.session_state.demo_env
    state = st.session_state.demo_state
    if env.step_idx >= CFG.episode_steps:
        return
    s_idx = state_to_index(state, CFG)
    raw_temp = env.indoor_temp
    action = int(policy.select_action(state))
    next_state, reward, terminated, truncated, info = env.step(action)
    st.session_state.demo_rows.append({
        "step": env.step_idx - 1,
        "state": state,
        "action": action,
        "reward": float(reward),
        "terminated": terminated,
        "truncated": truncated,
        "info": info,
        "q_before": q_table[s_idx].copy() if q_table is not None else None,
        "next_state": next_state,
        "raw_indoor_temp": raw_temp,
    })
    st.session_state.demo_last = st.session_state.demo_rows[-1]
    st.session_state.demo_state = next_state


def _replay_to(seed: int, step_count: int, q_table: np.ndarray) -> None:
    _init_state(seed)
    policy = QLearningPolicy(q_table, CFG, epsilon=0.0, rng=np.random.default_rng(42))
    for _ in range(step_count):
        _advance(policy, q_table)


def _room_svg(env: RoomEnv, last: dict | None) -> str:
    mode, light = decode_action(last["action"]) if last else (0, 0)
    activity = int(env.activity_schedule[min(env.step_idx, CFG.episode_steps - 1)])
    hour = min(env.step_idx * CFG.dt, 24.0)
    dark = is_dark(hour % 24.0, CFG)
    person = "away" if activity == ACTIVITY_AWAY else "awake" if activity == ACTIVITY_AWAKE else "asleep"
    fan_speed = "off" if mode == 0 else "low" if mode == 1 else "high"
    bulb = "#fbbf24" if light else "#64748b"
    cooler = "ON" if mode == 3 else "OFF"
    ac = "ON" if mode == 4 else "OFF"
    return f"""
    <div class="room" style="background:#111827;border:2px solid #334155;border-radius:12px;
         padding:18px;color:#f8fafc;">
      <div style="display:flex;justify-content:space-between;font-size:20px;">
        <b>Room at {hour:04.1f} h</b><b style="font-size:30px">{'☾' if dark else '☀'}</b>
      </div>
      <div style="display:grid;grid-template-columns:repeat(5,1fr);gap:10px;text-align:center;margin-top:12px;">
        <div class="device">Person<br><b>{html.escape(person)}</b></div>
        <div class="device">Thermometer<br><b>{env.indoor_temp:.2f} °C</b><br>
          effective {compute_effective_temp(env.indoor_temp, mode, CFG):.2f} °C<br>
          comfort {CFG.t_comfort_lo:g}–{CFG.t_comfort_hi:g} °C</div>
        <div class="device">Fan<br><b>{fan_speed}</b></div>
        <div class="device">Cooler<br><b>{cooler}</b></div>
        <div class="device">AC <b>{ac}</b><br>Light <span style="color:{bulb}">●</span></div>
      </div>
    </div>
    """


def _reason(last: dict) -> str:
    info = last["info"]
    mode, _ = decode_action(last["action"])
    reasons = []
    if info["activity"] == ACTIVITY_AWAY and info["day_type"] == 0 and 16.0 <= last["step"] * CFG.dt < 18.0:
        reasons.append("the family returns within two hours")
    if info["effective_temp"] > CFG.t_comfort_hi:
        reasons.append(f"the effective temperature is above {CFG.t_comfort_hi:g} °C")
    if info["activity"] != ACTIVITY_AWAY and info["effective_temp"] <= CFG.t_comfort_hi:
        reasons.append("the room is already in the comfort band")
    if mode == 0 and info["activity"] == ACTIVITY_AWAY:
        reasons.append("nobody is home")
    return " ".join(reasons[:2])


def _story_boxes(last: dict, env: RoomEnv) -> None:
    info = last["info"]
    mode, light = decode_action(last["action"])
    home = "someone is home" if info["activity"] != ACTIVITY_AWAY else "the room is empty"
    weather = REGIMES[info["weather_regime"]]
    reason = _reason(last)
    decision = f"It chooses {COOLING_MODES[mode]} with the light {'on' if light else 'off'}."
    if reason:
        decision += f" {reason.capitalize()}."
    penalty = CFG.lambda_t * info["D_T"] + CFG.lambda_l * info["D_L"]
    if info["cost"] > 0:
        note = "The score is negative because electricity cost is charged."
    elif penalty > 0:
        note = "The score is negative because comfort or lighting was penalised."
    else:
        note = "No cost or comfort penalty was recorded in this step."
    result = (
        f"Cost: ₹{info['cost']:.3f}. Comfort penalty: {CFG.lambda_t * info['D_T']:.3f}. "
        f"Lighting penalty: {CFG.lambda_l * info['D_L']:.3f}. Step reward: {last['reward']:.3f}. {note}"
    )
    cards = [
        ("👁", "It sees", f"It is {last['step'] * CFG.dt:04.1f} hours into the day; {home}. "
         f"The room is {last['raw_indoor_temp']:.2f} °C and the weather is {weather}."),
        ("⚙", "It decides", decision),
        ("✓", "Result", result),
    ]
    columns = st.columns(3)
    for column, (icon, title, text) in zip(columns, cards):
        column.markdown(f'<div class="story-box"><h3><span class="story-icon">{icon}</span>{title}</h3>{text}</div>',
                        unsafe_allow_html=True)


def _day_metrics(policy, seed: int, steps: int) -> dict[str, float]:
    return replay_episode(policy, seed, CFG, max_steps=steps).metrics


def _comparison_table(q_table: np.ndarray, seed: int, steps: int, threshold) -> None:
    trained = QLearningPolicy(q_table, CFG, epsilon=0.0, rng=np.random.default_rng(42))
    if threshold is None:
        rows = [
            ("Always on", _day_metrics(AlwaysOnPolicy(), seed, steps)),
            ("Always off", _day_metrics(AlwaysOffPolicy(), seed, steps)),
        ]
    else:
        rows = [
            ("Trained agent", _day_metrics(trained, seed, steps)),
            ("Always on", _day_metrics(AlwaysOnPolicy(), seed, steps)),
        ]
        rows.append(("Simple rule", _day_metrics(threshold, seed, steps)))
    frame = pd.DataFrame([
        {
            "method": name,
            "energy (kWh)": values["energy_kwh"],
            "hours uncomfortable": values["comfort_violations_hours"],
            "cumulative score": values["total_reward"],
        }
        for name, values in rows
    ])
    st.dataframe(frame.round(3), hide_index=True, use_container_width=True)


q_saved = _load_q_table()
if q_saved is None:
    st.error("q_table.npy is missing. Run the project training command before starting this demo.")
    st.stop()
if "demo_q_table" not in st.session_state:
    st.session_state.demo_q_table = q_saved.copy()
q_table = st.session_state.demo_q_table

params = load_baseline_params()
threshold_policy = None
if params is not None:
    threshold_policy, _ = make_baseline_policies(params, CFG)

with st.sidebar:
    st.header("Choose the real day")
    day_type = st.selectbox("Day type", range(2), format_func=lambda value: DAY_TYPES[value])
    regime = st.selectbox("Weather", range(2), format_func=lambda value: REGIMES[value])
    matching = _matching_eval_seeds(day_type, regime)
    if not matching:
        st.error("No matching EVAL seed exists for this combination.")
        matching = [int(next(iter(EVAL_SEEDS)))]
    seed = st.selectbox("Evaluation seed", matching)
    with st.expander("Playback speed", expanded=False):
        speed = st.slider("Autoplay interval (seconds)", 0.2, 2.0, 0.8)

if "demo_env" not in st.session_state or st.session_state.get("demo_seed") != seed:
    _init_state(seed)
env = st.session_state.demo_env
policy = QLearningPolicy(q_table, CFG, epsilon=0.0, rng=np.random.default_rng(42))

control_columns = st.columns(4)
if control_columns[0].button("Step", use_container_width=True):
    _advance(policy, q_table)
if control_columns[1].button("Play / Pause", use_container_width=True):
    st.session_state.playing = not st.session_state.get("playing", False)
if control_columns[2].button("Reset", use_container_width=True):
    _init_state(seed)
if control_columns[3].button("Jump to 4 pm on a workday", use_container_width=True):
    _replay_to(_find_workday_seed(), 32, q_table)

if hasattr(st, "fragment"):
    @st.fragment(run_every=speed)
    def _autoplay() -> None:
        if st.session_state.get("playing") and env.step_idx < CFG.episode_steps:
            _advance(policy, q_table)
    _autoplay()

rows = st.session_state.demo_rows
last = st.session_state.demo_last
clock = min(env.step_idx * CFG.dt, 24.0)
score = sum(row["reward"] for row in rows)
st.markdown(
    f'<div class="status"><b>Day {st.session_state.demo_seed}</b> · '
    f'<b>{int(clock):02d}:{int((clock % 1) * 60):02d}</b> · '
    f'<b>Score so far: {score:.3f}</b> · '
    f'<span style="background:#2563eb;color:white;padding:5px 10px;border-radius:8px;">'
    "TRAINED AGENT (not learning during this demo)</span></div>",
    unsafe_allow_html=True,
)
st.markdown(_room_svg(env, last), unsafe_allow_html=True)

if last is not None:
    _story_boxes(last, env)

st.subheader("Today so far")
_comparison_table(q_table, st.session_state.demo_seed, len(rows), threshold_policy)
if threshold_policy is None:
    st.caption("Simple rule is unavailable. Run run_baselines.py to create baseline_params.json.")

st.markdown(
    '<div class="story-box"><h3>How it learns</h3>'
    "1. It tries actions and records the score it got.<br>"
    "2. It updates its table: new estimate moves toward (reward + best future estimate).<br>"
    "3. After 20,000 practice days the table is fixed. This demo only uses it."
    "</div>",
    unsafe_allow_html=True,
)

scenario_results = st.session_state.get("scenario_focus")
scenario_clicked = False
scenario_buttons = st.columns(2)
for column, scenario_name in zip(scenario_buttons, ("Pre-cooling", "Awake in the dark")):
    if column.button(scenario_name):
        scenario_clicked = True
        results = search_scenarios(q_table, CFG)
        wanted = "Empty room shortly before return on a workday" if scenario_name == "Pre-cooling" else "Awake in the dark"
        match = next((item for item in results if item["scenario"] == wanted), None)
        st.session_state.scenario_focus = match
        scenario_results = match
if scenario_results:
    if scenario_clicked:
        _replay_to(scenario_results["seed"], scenario_results["step"], q_table)
        st.rerun()
    contradiction = " The real action contradicts the intended story." if scenario_results["contradiction"] else ""
    st.info(
        f"{scenario_results['scenario']}: EVAL seed {scenario_results['seed']}, step {scenario_results['step']}. "
        f"The agent chose {scenario_results['action_label']}.{contradiction}"
    )

with st.expander("Advanced (for questions)", expanded=False):
    if last is not None:
        s_idx = state_to_index(last["state"], CFG)
        next_idx = state_to_index(last["next_state"], CFG)
        q_row = last["q_before"] if last["q_before"] is not None else q_table[s_idx]
        st.subheader("Q-values and update")
        fig, ax = plt.subplots(figsize=(8, 2.5))
        ax.bar(range(10), q_row, color=["#f59e0b" if i == last["action"] else "#2563eb" for i in range(10)])
        ax.set_xlabel("encoded action; selected action is orange")
        ax.set_ylabel("Q-value")
        st.pyplot(fig)
        plt.close(fig)
        preview = q_update_preview(q_table, s_idx, last["action"], last["reward"], next_idx,
                                   last["terminated"], 0.1, 0.97)
        st.dataframe(pd.DataFrame([{
            "Q before": preview["before"],
            "reward": preview["reward"],
            "gamma × best future": preview["gamma_next_max"],
            "target": preview["target"],
            "alpha": preview["alpha"],
            "Q after": preview["after"],
        }]).round(4), hide_index=True)
        st.caption("Illustrative only: the trained policy is frozen and is not changed during this demo.")

    st.subheader("What-if panel — state query, not a simulation")
    query_columns = st.columns(5)
    what_hour = query_columns[0].slider("Hour", 0.0, 23.5, 12.0, 0.5)
    what_activity = query_columns[1].selectbox("Activity", range(3), format_func=lambda value: ACTIVITIES[value], key="what_activity")
    what_day = query_columns[2].selectbox("Day", range(2), format_func=lambda value: DAY_TYPES[value], key="what_day")
    what_regime = query_columns[3].selectbox("Weather", range(2), format_func=lambda value: REGIMES[value], key="what_regime")
    what_temp = query_columns[4].slider("Indoor °C", 15.0, 45.0, 30.0)
    what_state = discretize_state(int(round(what_hour / CFG.dt)), what_activity, what_day, what_temp, what_regime, CFG)
    query_idx = state_to_index(what_state, CFG)
    query_action = int(np.argmax(q_table[query_idx]))
    st.write(f"State {what_state}; greedy action: {action_words(query_action)['label']}.")
    st.dataframe(pd.DataFrame(one_step_previews(what_state, what_temp, range(10), CFG)).round(3),
                 hide_index=True, use_container_width=True)
    st.caption("One step ahead only, with zero outdoor noise.")

    st.subheader("Visit counts")
    visits_path = Path("visit_counts.npy")
    if visits_path.exists():
        st.write(f"Current state visits: {int(np.load(visits_path)[query_idx].sum())}")
    else:
        st.caption("visit_counts.npy is unavailable.")

    st.subheader("Train a new agent")
    episodes = st.number_input("Practice days", 1, 20000, 3000, 100)
    if st.button("Train in session only"):
        progress = st.progress(0.0)
        chart = st.empty()
        curve = []
        def update(ep, total, val_mean, train_reward, new_table, visits):
            progress.progress(ep / total)
            curve.append({"episode": ep, "validation reward": val_mean})
            chart.line_chart(pd.DataFrame(curve).set_index("episode"))
        new_q, new_visits, new_curve = train_with_progress(
            int(episodes), CFG,
            validation_frequency=max(100, min(500, int(episodes) // 10 or 1)),
            callback=update,
        )
        st.session_state.demo_q_table = new_q
        st.session_state.demo_visits = new_visits
        st.session_state.train_curve = new_curve
        st.success("The new table stays in session state; q_table.npy and policy.csv were not changed.")

    st.subheader("Other real scenarios")
    if st.button("Search remaining scenarios"):
        st.session_state.all_scenarios = search_scenarios(q_table, CFG)
    if st.session_state.get("all_scenarios"):
        st.dataframe(pd.DataFrame(st.session_state.all_scenarios)[
            ["scenario", "seed", "step", "state", "action_label", "reward", "contradiction"]
        ], hide_index=True, use_container_width=True)
