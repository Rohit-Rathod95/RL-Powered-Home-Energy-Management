# Home Electricity Saver
Home Electricity Saver is a small tabular Q-learning project.
It controls cooling and a light in one simulated room.
All results are inside the simulator. They are **not real household energy savings**.
> **Status**
>
> The environment, Q-learning code, baselines, tests, dashboard, and professor demo work.
>
> The physical values are still placeholders. `CALIBRATED` is `False` in `config.py`.
>
> The saved baseline table and saved Q-learning tables are old experiment outputs. Run commands only when you are ready to regenerate them.
## Table of contents
1. [The project in one minute](#the-project-in-one-minute) 2. [The big picture](#the-big-picture) 3. [The pretend room](#the-pretend-room-the-simulator) 4. [What the agent sees](#what-the-agent-sees-state) 5. [What the agent can do](#what-the-agent-can-do-actions) 6. [How the agent is scored](#how-the-agent-is-scored-reward) 7. [How the agent learns](#how-the-agent-learns-q-learning) 8. [Which files do what](#which-files-do-what) 9. [How we tested fairly](#how-we-tested-fairly) 10. [The comparison](#the-comparison) 11. [What the agent learned](#what-the-agent-learned) 12. [The comfort-weight experiment](#the-comfort-weight-experiment) 13. [Known weaknesses](#known-weaknesses-say-these-before-the-professor-does) 14. [How to run everything](#how-to-run-everything) 15. [Glossary](#glossary) 16. [Questions a professor may ask](#questions-a-professor-may-ask) 17. [Documentation to fix](#documentation-to-fix) 18. [Verified-number and missing-artifact checklist](#verified-number-and-missing-artifact-checklist)
---
## The project in one minute
Imagine a student keeping a notebook.
The student writes:
```text
Situation:     hot room, family away, dry weather, late afternoon
Switch used:   cooler + light off
Score received: a negative score
```
After many days, the notebook contains a score for every switch in every kind of situation.
That notebook is the **Q-table**. It tells the agent which action looks best for the current situation.
> **Analogy:** The Q-table is a student’s revision notebook. It does not
> contain a perfect answer. It contains the best guesses learned from practice.
The simulator makes a room, a schedule, weather, and temperature. The agent sees a small, discretized description of that room. It chooses one of 10 actions every 30 minutes. The environment changes the temperature and returns a reward. Q-learning updates the notebook.
### A professor-ready summary
> **Say this in the viva:** “I built a small single-room simulator and used
> tabular Q-learning to choose one of 10 cooling-and-light actions every
> 30-minute step. The agent learns a table of action values for discretized
> states. I compare it with hand-written baselines on separate held-out
> simulated days. The parameters are placeholders, so the findings describe
> this simulator and not measured savings in a real home.”
---
## The big picture
```text
+-------------+     +-------------+     +----------------+
| Situation   | --> | Pick action | --> | Room reacts    |
| state       |     | from table  |     | temp changes   |
+-------------+     +-------------+     +----------------+
        ^                                      |
        |                                      v
        |                              +---------------+
        +------------------------------| Score/reward  |
                                       +---------------+
                                                |
                                                v
                                       +---------------+
                                       | Update table  |
                                       +---------------+
```
In code, the main loop is in `train.py`.
1. `RoomEnv.reset()` creates one day. 2. `state_to_index()` changes the state tuple into a table row number. 3. `select_action()` chooses an action. 4. `RoomEnv.step()` applies the action. 5. `q_update()` updates one Q-table cell. 6. The loop repeats until the day is truncated.
---
## The pretend room: the simulator
The room is simulated in steps of `0.5` hours, which is 30 minutes (`config.py`).
One day has `48` steps (`config.py`, `env.py`).
The environment creates these things at reset:
| Thing | What the code does | Source |
|---|---|---|
| Day type | Samples workday or weekend | `env.py` |
| Weather | Samples dry or humid and keeps it for the day | `env.py` |
| People | Creates asleep, awake, away, awake, asleep blocks | `schedule.py` |
| Schedule noise | Adds at most `1` step of jitter by default | `config.py`, `schedule.py` |
| Outdoor noise | Pre-generates Gaussian noise with standard deviation `0.5` | `config.py`, `env.py` |
| Indoor temperature | Starts from outdoor temperature plus an offset | `env.py` |
| Daylight | Uses the clock, from `06:00` to before `18:00` | `config.py`, `schedule.py` |
The activity schedule is **exogenous**. That means the agent does not control when people leave or return.
The schedule is also created before actions are taken. Two action sequences with the same seed get the same activity schedule and outdoor-noise sequence. This is checked in `tests/test_env.py`.
### People schedule
The default workday times are:
| Event | Time | Source |
|---|---:|---|
| Wake | `7.0` hours | `config.py` |
| Leave | `8.5` hours | `config.py` |
| Return | `18.0` hours | `config.py` |
| Sleep | `23.0` hours | `config.py` |
The default weekend times are:
| Event | Time | Source |
|---|---:|---|
| Wake | `8.5` hours | `config.py` |
| Leave | `12.0` hours | `config.py` |
| Return | `16.0` hours | `config.py` |
| Sleep | `23.5` hours | `config.py` |
These are household-routine assumptions. They are not measured household data.
### Temperature equation
The code uses:
```text
outdoor temperature =
    t_mean
    + t_amplitude * cos(2*pi*(hour - t_peak_hour)/24)
    + noise
```
The indoor update is:
```text
next indoor temperature =
    current temperature
    + passive heat transfer
    + occupant heat
    - active cooling
```
The exact implementation is in `dynamics.py`.
> **Common confusion:** A fan does not lower the simulated air temperature.
> It only lowers the effective, or felt, temperature. The cooler and AC change
> the air temperature.
### Cooling modes
There are five mutually exclusive cooling modes.
| Mode | What it does in the code |
|---|---|
| `off` | No cooling |
| `fan_low` | No air-temperature change; felt temperature is lower by `1.5` degrees |
| `fan_high` | No air-temperature change; felt temperature is lower by `2.5` degrees |
| `cooler_fan` | Pulls the air temperature toward `24.0` degrees; humidity weakens it |
| `ac_fan_low` | Pulls the air temperature toward `25.0` degrees and includes low-fan effect |
The values `1.5`, `2.5`, `24.0`, and `25.0` are from `config.py`.
### Placeholder values
Every row below is copied from `config.py`.
The last column is deliberately simple: these values are not read from a real appliance label or electricity bill.
| Parameter | Value | Meaning | From real label? |
|---|---:|---|---|
| `dt` | `0.5` hours | Length of one step | no, placeholder |
| `episode_steps` | `48` | Steps in one day | no, simulator choice |
| `time_block_hours` | `2.0` hours | Time-state block size | no, design choice |
| `tau` | `5.0` hours | Room thermal time constant | no, placeholder |
| `g_occ` | `0.12` degrees per step | Occupant heat gain | no, placeholder |
| `t_mean` | `32.0` degrees | Mean outdoor temperature | no, placeholder |
| `t_amplitude` | `6.0` degrees | Outdoor daily-wave amplitude | no, placeholder |
| `t_peak_hour` | `15.0` hours | Outdoor-temperature peak | no, placeholder |
| `outdoor_noise_std` | `0.5` degrees | Outdoor noise spread | no, placeholder |
| `t_init_offset_std` | `0.5` degrees | Initial indoor offset spread | no, placeholder |
| `c_cooler` | `0.18` | Cooler gap-closing fraction | no, placeholder |
| `cooler_temp_floor` | `24.0` degrees | Cooler target floor | no, placeholder |
| `humidity_factor_dry` | `0.20` | Dry-weather degradation factor | no, placeholder |
| `humidity_factor_humid` | `0.80` | Humid-weather degradation factor | no, placeholder |
| `c_ac` | `0.35` | AC gap-closing fraction | no, placeholder |
| `ac_setpoint` | `25.0` degrees | AC target | no, placeholder |
| `k_fan_low` | `1.5` degrees | Low-fan felt cooling | no, placeholder |
| `k_fan_high` | `2.5` degrees | High-fan felt cooling | no, placeholder |
| `p_fan_low` | `0.04` kW | Low-fan power | no, placeholder |
| `p_fan_high` | `0.07` kW | High-fan power | no, placeholder |
| `p_cooler_fan` | `0.18` kW | Cooler power | no, placeholder |
| `p_ac_fan_low` | `1.40` kW | AC power | no, placeholder |
| `p_light` | `0.02` kW | Light power | no, placeholder |
| `price_per_kwh` | `7.0` currency units/kWh | Flat tariff | no, placeholder |
| `t_comfort_lo` | `23.0` degrees | Lower comfort edge | no, placeholder |
| `t_comfort_hi` | `27.0` degrees | Upper comfort edge | no, placeholder |
| `daylight_start_hour` | `6.0` hours | Daylight begins | no, placeholder |
| `daylight_end_hour` | `18.0` hours | Darkness begins | no, placeholder |
| `lambda_t` | `2.0` | Temperature-penalty weight | no, placeholder |
| `lambda_l` | `1.5` | Lighting-penalty weight | no, placeholder |
| `lambda_s` | `0.0` | Switching-penalty weight by default | no, placeholder |
> **Why it matters:** A low score here does not mean a real family saved
> money. It means the chosen action received a low score under these assumed
> numbers.
---
## What the agent sees: state
The agent does not see every decimal temperature.
`discretize.py` turns the room into a small tuple:
```text
(time_block, activity, day_type, temperature_bin, weather_regime)
```
The default state dimensions are:
| State variable | Possible values in code | Why it is present |
|---|---|---|
| Time block | `12` blocks | The agent needs the time of day |
| Activity | `3`: away, awake, asleep | Cooling and light needs depend on people |
| Day type | `2`: workday, weekend | Return times differ |
| Temperature bin | `7` bins | The agent needs a rough temperature |
| Weather regime | `2`: dry, humid | Cooler performance differs |
The temperature edges are:
```text
23.0, 27.0, 28.5, 29.5, 32.0, 35.0
```
These are six edges, so `discretize.py` creates seven bins.
The real state count is:
```text
12 time blocks
* 3 activities
* 2 day types
* 7 temperature bins
* 2 weather regimes
= 1008 states
```
The saved `q_table.npy` has shape `(1008, 10)`, verified from the file.
The optional previous-mode state is supported by the code. If it is enabled, the extra dimension has `5` cooling modes. The default is `include_prev_mode=False`, so the running project uses `1008` states.
Daylight is **not** a state variable.
The code derives darkness from the time block and `schedule.is_dark()`.
> **Common confusion:** The state says `temperature_bin`, not exact
> temperature. Two different temperatures can produce the same state.
---
## What the agent can do: actions
An action combines one cooling mode with light off or light on.
That gives:
```text
5 cooling modes * 2 light settings = 10 actions
```
`env.encode_action()` stores the action as:
```text
action = cooling_mode * 2 + light_on
```
| Action number | Cooling mode | Light | Cooling power | Step energy without light | Step energy with light |
|---:|---|---|---:|---:|---:|
| `0` | off | off | `0.00` kW | `0.00` kWh | `0.01` kWh |
| `1` | off | on | `0.00` kW | `0.00` kWh | `0.01` kWh |
| `2` | fan_low | off | `0.04` kW | `0.02` kWh | `0.03` kWh |
| `3` | fan_low | on | `0.04` kW | `0.02` kWh | `0.03` kWh |
| `4` | fan_high | off | `0.07` kW | `0.04` kWh | `0.05` kWh |
| `5` | fan_high | on | `0.07` kW | `0.04` kWh | `0.05` kWh |
| `6` | cooler_fan | off | `0.18` kW | `0.09` kWh | `0.10` kWh |
| `7` | cooler_fan | on | `0.18` kW | `0.09` kWh | `0.10` kWh |
| `8` | ac_fan_low | off | `1.40` kW | `0.70` kWh | `0.71` kWh |
| `9` | ac_fan_low | on | `1.40` kW | `0.70` kWh | `0.71` kWh |
The energy values use `dt=0.5` hours and `p_light=0.02` kW from `config.py` and `dynamics.py`.
---
## How the agent is scored: reward
The code computes:
```text
cost = price_per_kwh * energy_kwh
D_T = 0 when nobody is home
      otherwise:
      max(0, effective_temp - 27.0)^2
      + max(0, 23.0 - effective_temp)^2
D_L = 1 when a person is awake, it is dark, and the light is off
      otherwise 0
reward = -cost - 2.0*D_T - 1.5*D_L - 0.0*switching_flag
```
The values `27.0`, `23.0`, `2.0`, `1.5`, and `0.0` come from `config.py`.
The environment logs `cost`, `D_T`, `D_L`, `switched`, and `energy_kwh` in the `info` dictionary.
### Worked example: one AC step
Use an AC action with the light off.
| Quantity | Calculation | Value |
|---|---|---:|
| AC power | `1.40` kW | `1.40` kW |
| Step length | `0.5` hours | `0.5` hours |
| Energy | `1.40 * 0.5` | `0.70` kWh |
| Price | `7.0` per kWh | `7.0` |
| Cost | `7.0 * 0.70` | `4.90` |
This is an arithmetic example from the configured values. It is not a result from one particular random day.
### Worked example: one discomfort value
Suppose the effective temperature is `30.0` degrees while someone is home.
```text
D_T = (30.0 - 27.0)^2
    = 3.0^2
    = 9.0
temperature penalty = 2.0 * 9.0
                   = 18.0
```
Therefore, if the action also used the AC example above and had no lighting or switching penalty:
```text
reward = -4.90 - 18.0
       = -22.90
```
The formula and values are from `dynamics.py` and `config.py`.
> **Analogy:** The reward is a bill plus a comfort complaint. Electricity
> creates a cost. A hot occupied room creates a comfort penalty. A missing
> light creates a lighting penalty.
> **Why it matters:** “Switch everything off” avoids energy cost, but the
> occupied room can become very uncomfortable. In the saved evaluation,
> Always Off has mean reward `-2205.27`, while Tuned Precool (Clock) has
> mean reward `-51.36` (`results.csv`).
---
## How the agent learns: Q-learning
The agent keeps one row for each state and one column for each action.
The saved Q-table shape is:
```text
1008 states * 10 actions = 10080 Q-values
```
This shape is verified from `q_table.npy`.
> **Analogy:** After every practice step, the student opens the notebook
> cell for the current situation and changes the guess for the switch used.
### Update rule
The project uses:
```text
future = 0                         if terminated
       = max Q(next_state, action) otherwise
target = reward + gamma * future
Q(current_state, action)
    = Q(current_state, action)
      + alpha * (target - Q(current_state, action))
```
The exact code is in `qlearning.py`.
| Symbol | Plain meaning | Project value or source |
|---|---|---|
| `Q` | Current guess for an action | Stored in the table |
| `reward` | Score from the environment | Returned by `env.step()` |
| `alpha` | How quickly the old guess moves | `0.1` in `train.py` |
| `gamma` | Importance of future value | `0.97` in `train.py` |
| `future` | Best next-state action value | Maximum next Q-row value |
| `target` | New value the current guess moves toward | Bellman target |
### Numeric update example
Use the requested simple example:
```text
old Q = 0
reward = -5
alpha = 0.1
gamma = 0.97
best future value = -20
target = -5 + 0.97 * (-20)
       = -24.4
new Q = 0 + 0.1 * (-24.4 - 0)
      = -2.44
```
The result `-2.44` follows the formula in `qlearning.py`.
The repository tests also verify the same update idea with:
```text
old Q = 5.0
reward = -2.0
gamma = 0.9
best future value = 8.0
alpha = 0.1
```
The test expects `5.02` after the non-terminal update and `4.318` after the terminal update (`tests/test_qlearning.py`).
### Exploration and exploitation
The agent uses epsilon-greedy action selection.
| Term | Meaning |
|---|---|
| Exploration | Try a random action |
| Exploitation | Use the best known action |
| `epsilon=1.0` | Fully random choice |
| `epsilon=0.0` | Greedy choice, with random tie-breaking |
`train.py` starts at epsilon `1.0` and ends at `0.05`.
It linearly decays epsilon over the first `70%` of the requested episodes.
The default training command requests `20000` episodes.
The saved `curves.csv` has checkpoints through episode `20000` and reaches epsilon `0.05` from episode `14000` onward (`train.py`, `curves.csv`).
### Why the last step of the day still uses the future value
The end of day is **truncation**, not true termination.
At step `48`, the environment returns:
```text
terminated = False
truncated = True
```
The next state wraps its time block to the start of the day while carrying the temperature and other relevant values.
Because `terminated` is `False`, `q_update()` bootstraps from the best next Q-value.
> **Why it matters:** Truncation means “we stopped this episode because the
> day boundary arrived.” Termination means “there is no future state.” This
> project treats the next day as a continuation for learning.
---
## Which files do what
| File | Role |
|---|---|
| `design.md` | Early design notes and intended experiment plan |
| `config.py` | Constants, schedules, placeholder physics, reward weights |
| `schedule.py` | Workday/weekend activity schedules and darkness |
| `dynamics.py` | Outdoor temperature, felt temperature, energy, room dynamics, reward |
| `env.py` | The room environment with `reset()` and `step()` |
| `discretize.py` | Converts raw values into state tuples |
| `qlearning.py` | State indexing, epsilon-greedy choice, Q-table update |
| `train.py` | Q-learning training, validation, exports, learned-policy wrapper |
| `evaluate.py` | Runs policies and computes daily metrics |
| `baselines.py` | Always-off, always-on, random, timer, threshold, and pre-cooling rules |
| `run_baselines.py` | Tunes baselines, evaluates them, writes `results.csv` |
| `multi_seed.py` | Trains several independent agents and writes multi-agent results |
| `lambda_sweep.py` | Trains agents with several temperature-penalty weights |
| `inspect_policy.py` | Prints what the saved policy does in selected states |
| `steady_state.py` | Hand-check and steady-state analysis helper |
| `dashboard.py` | Existing results, replay, and learned-policy Streamlit dashboard |
| `demo_logic.py` | Pure helper functions for the professor demo |
| `demo_app.py` | Simplified interactive professor demo |
| `plot_training.py` | Creates the training learning-curve plot |
| `tests/test_env.py` | Environment, schedule, dynamics, and no-NaN tests |
| `tests/test_baselines.py` | Baseline validity, seed separation, and determinism tests |
| `tests/test_qlearning.py` | Q-table indexing and Q-update tests |
| `tests/test_demo_logic.py` | Demo replay, reward, update-preview, and scenario tests |
| `results.csv` | Saved baseline means and standard deviations |
| `curves.csv` | Saved training and validation checkpoints |
| `policy.csv` | Saved greedy action and visit information per state |
| `multi_seed_summary.csv` | Saved results for independent Q-learning agents |
| `lambda_sweep_summary.csv` | Saved comfort-weight experiment summary |
| `q_table.npy` | Saved trained Q-table |
| `pareto.png` | Saved energy-versus-discomfort plot |
`visit_counts.npy` is not present in the current workspace.
The training code can create it.
---
## How we tested fairly
The project uses three separate groups of random seeds.
| Group | Seed range | Count | Used for |
|---|---|---:|---|
| Training | `0` through `999` | `1000` | Learning and baseline tuning |
| Validation | `20000` through `20049` | `50` | Checkpoints during training |
| Evaluation | `10000` through `10199` | `200` | Final held-out comparison |
These ranges are defined in `evaluate.py`.
The tests check that the three sets are pairwise disjoint (`tests/test_baselines.py`).
> **Why it matters:** If the same simulated days are used to learn and judge
> the policy, the policy may look good because it has seen those days before.
The baseline tuner uses the first `200` training seeds from the supplied training range (`baselines.py`).
The final baseline table uses `200` evaluation days (`run_baselines.py`, `results.csv`).
The Q-learning multi-agent experiment uses `15` saved agents in the current `multi_seed_summary.csv`.
`steady_state.py` is available for checking simple temperature behavior. The environment tests also check hand-computable dynamics, cooling behavior, reward decomposition, determinism, and no NaN or infinity values.
---
## The comparison
The table below uses the saved held-out evaluation results.
For most controllers, the source is `results.csv`.
For the trained Q-learning row, the source is the mean across the `15` independent agents in `multi_seed_summary.csv`.
The trained row is not directly the same single run as the baseline rows.
| Controller | What it does | Energy kWh mean ± spread | Thermal `D_T` mean ± spread | Reward mean ± spread | Source |
|---|---|---:|---:|---:|---|
| Always off | Cooling off and light off at every step | `0.00 ± 0.00` | `1094.99 ± 239.23` | `-2205.27 ± 479.24` | `results.csv` |
| Always on | AC on and light on at every step | `34.08 ± 0.00` | `0.04 ± 0.17` | `-238.64 ± 0.34` | `results.csv` |
| Random | Uniform random action from `0` through `9` | `8.17 ± 1.76` | `136.17 ± 103.06` | `-337.05 ± 200.29` | `results.csv` |
| Timer | Fixed time schedule: AC in evening, fan during sleep | `8.84 ± 0.00` | `269.58 ± 201.35` | `-601.03 ± 402.70` | `results.csv` |
| Threshold rule | Hand-written temperature and occupancy rules | `3.49 ± 2.22` | `92.83 ± 33.22` | `-210.09 ± 52.11` | `results.csv` |
| Tuned threshold | Grid-searched threshold parameters on training days | `5.02 ± 1.24` | `25.05 ± 21.56` | `-85.21 ± 46.94` | `results.csv` |
| Tuned pre-cool block | Threshold rule plus block-level pre-cooling | `5.69 ± 1.59` | `6.74 ± 4.53` | `-53.34 ± 19.89` | `results.csv` |
| Tuned pre-cool clock | Threshold rule plus step-level pre-cooling | `5.19 ± 1.54` | `7.52 ± 4.52` | `-51.36 ± 19.39` | `results.csv` |
| Trained Q-learning | Greedy policy from trained Q-tables | `5.21 ± 0.35` | `11.81 ± 14.58` | `-60.72 ± 27.80` | `multi_seed_summary.csv` |
Here, “spread” means standard deviation.
The trained agents do **not** beat the best hand-tuned rule by mean reward.
The best saved hand-tuned reward is Tuned Precool (Clock) at `-51.36`.
The saved trained-agent mean is `-60.72`.
So the honest wording is:
> **Say this in the viva:** “The trained agents are competitive with the
> tuned baselines in this saved experiment, but they do not beat the best
> hand-tuned pre-cooling rule. The result is a simulator result, not a real
> savings claim.”
### What this table does show
- It shows behavior on evaluation seeds that are separate from training seeds.
- It shows the trade-off between energy, comfort penalty, and total reward.
- It shows that Always Off avoids energy but receives a very large comfort cost.
- It shows that always running the AC costs much more energy.
- It shows that pre-cooling rules are strong in this saved experiment.
### What this table does not show
- It does not measure a real electricity bill.
- It does not prove that a real AC uses constant power.
- It does not prove transfer to another room or family.
- It does not prove that one training run is reliable.
- It does not include a value-iteration optimum. That controller is mentioned
in `design.md` but is not implemented in the current source.
### Multi-seed result
Across the `15` agents saved in `multi_seed_summary.csv`:
| Metric | Mean | Standard deviation | Source |
|---|---:|---:|---|
| Reward | `-60.72` | `27.80` | `multi_seed_summary.csv` |
| Energy | `5.21` kWh | `0.35` kWh | `multi_seed_summary.csv` |
| Thermal `D_T` | `11.81` | `14.58` | `multi_seed_summary.csv` |
The per-agent saved run time ranges from `0.486` to `0.583` minutes in that file. A fresh run can take a different time.
---
## What the agent learned
The saved policy was inspected with:
```powershell
python inspect_policy.py
```
The inspection checks workday block `8`, which is `16:00-18:00`, while the household is away and states have at least `20` training visits.
The output contains these real actions:
| Temperature bin | Dry action | Humid action |
|---:|---|---|
| `2` | `fan_high+light_on` | `fan_low+light_on` |
| `3` | `cooler_fan+light_on` | `ac_fan_low+light_on` |
| `4` | `ac_fan_low+light_off` | `ac_fan_low+light_off` |
| `5` | `ac_fan_low+light_off` | `ac_fan_low+light_on` |
| `6` | `ac_fan_low+light_on` | `ac_fan_low+light_on` |
The inspection also reports:
```text
Evaluation steps in states visited less than 20 times:
0/9600 (0.0%)
```
The cooling-mode usage on evaluation days is:
| Mode | Share |
|---|---:|
| `cooler_fan` | `29.5%` |
| `fan_high` | `24.1%` |
| `fan_low` | `22.4%` |
| `off` | `14.1%` |
| `ac_fan_low` | `9.8%` |
This is evidence that the saved policy has learned different actions for different temperatures and weather regimes.
It is not proof that every action is sensible. Some actions, such as a light being on while away, can still appear in rarely visited or imperfectly valued states.
### Why pre-cooling is interesting
The pre-cooling baseline explicitly uses the block before the nominal return time. The tuned block policy and tuned clock policy are defined in `baselines.py`.
The trained Q-table also contains different actions in the `16:00-18:00` workday-away block, as shown above.
> **Analogy:** A reactive rule waits until the person returns and then starts
> cooling. Pre-cooling starts while the room is empty so the room can be more
> comfortable at return time.
> **Why it matters:** The simulator’s temperature has inertia. An action now
> changes a later temperature. That is the reason an RL method can be useful
> even though a simple threshold rule is a strong competitor.
The current inspection output does not prove a causal real-home benefit.
It only shows the selected actions and the saved evaluation behavior.
---
## The comfort-weight experiment
`lambda_sweep.py` trains agents with these `lambda_T` values:
```text
0.25, 0.5, 1.0, 2.0, 4.0, 8.0
```
The saved `lambda_sweep_summary.csv` contains the following means and standard deviations across the experiment’s agents:
| `lambda_T` | Energy mean | Energy std | Cost mean | `D_T` mean | Degree-hours above mean | Away energy mean | Reward under own lambda mean |
|---:|---:|---:|---:|---:|---:|---:|---:|
| `0.25` | `3.25` | `0.33` | `22.76` | `70.40` | `14.56` | `1.01` | `-40.40` |
| `0.5` | `3.87` | `0.15` | `27.06` | `28.30` | `9.25` | `1.29` | `-42.32` |
| `1.0` | `4.63` | `0.17` | `32.44` | `51.27` | `8.78` | `1.65` | `-84.99` |
| `2.0` | `5.87` | `0.46` | `41.10` | `6.65` | `3.68` | `2.27` | `-55.18` |
| `4.0` | `6.13` | `0.31` | `42.92` | `3.62` | `2.55` | `2.19` | `-57.42` |
| `8.0` | `7.57` | `0.23` | `53.01` | `3.51` | `1.32` | `2.43` | `-81.60` |
The table has a warning from `lambda_sweep.py`:
> `reward_own_lambda` is measured under each run’s own lambda: do not compare
> across rows.
The broad pattern is that larger comfort weights generally use more energy and reduce degree-hours above the comfort band, although the `lambda_T=1.0` row is not monotonic in every metric.
This is another reason not to choose a reward weight from one result alone.
---
## Known weaknesses: say these before the professor does
### Placeholder physical values
`config.py` sets `CALIBRATED=False`.
The warning tells the user to calibrate parameters against appliance labels and bills.
The values in the simulator are assumptions, not measurements.
### The trained agent does not beat the best hand-tuned rule
The saved trained-agent mean reward is `-60.72`.
The saved Tuned Precool (Clock) mean reward is `-51.36`.
Therefore, say “does not beat,” not “beats.”
### One room only
`env.py` models one room.
There is no multi-room coordination.
### Binning hides detail
The state uses `7` temperature bins rather than the exact temperature.
Different raw temperatures can map to the same table row.
### Occupancy is unrealistically known
The activity state is supplied by the simulator.
The agent does not have to infer whether a person is home.
### Day wrap is an approximation
At the end of `48` steps, the next state wraps to the start time block while carrying some continuous and mode information.
This makes the daily boundary suitable for bootstrapping, but it is not a full multi-day household model.
### Fixed learning rate
`alpha=0.1` stays constant in `train.py`.
The training curve has unstable validation checkpoints. For example, validation reward is `-365.31` at episode `15000` and `-49.24` at episode `16000` in `curves.csv`.
This is evidence of noisy or unstable checkpoints, not proof that every training run is unstable.
### Rare-state action quality
The saved `policy.csv` has `1008` state rows, and many rows are marked `never_visited=True`.
A state with little experience can have an arbitrary-looking greedy action.
The current `inspect_policy.py` evaluation found `0/9600` evaluation steps below the `20`-visit threshold, but that does not mean every possible state was well learned.
### Simple rules are strong
Tuned pre-cooling is hand-designed and tuned on training seeds.
The trained agent must compete with that strong prior.
### Flat tariff
The code uses one flat price, `7.0` per kWh.
There is no time-of-use tariff.
### Simplified AC
The AC uses constant configured power when on.
The code does not model compressor cycling or inverter behavior.
### No real data
The project uses generated schedules, generated outdoor noise, and assumed physical values.
### No value-iteration oracle
`design.md` lists a value-iteration optimum as a planned baseline.
There is no value-iteration implementation in the current file list or `results.csv`.
---
## How to run everything
Run commands from the project folder.
### 1. Run the tests
```powershell
python -m pytest -q
```
This runs the environment, baseline, Q-learning, and demo-logic tests.
The current repository test suite has `33` passing tests and one expected calibration warning when the demo tests are run in the current workspace.
### 2. Train the main agent
```powershell
python train.py
```
The default in `train.py` is `20000` training episodes.
It prints the state-space size, epsilon schedule, training progress, and validation checkpoints.
It writes or overwrites:
```text
q_table.npy
visit_counts.npy
curves.csv
policy.csv
learning_curve.png
```
> **Common confusion:** Do not run `python train.py --smoke` before a demo
> if you need the saved full table. The smoke option requests `500` episodes
> and the script saves its result to `q_table.npy`, replacing the existing
> table.
The source code does not record a guaranteed wall-clock time for one normal training run. A fresh run time is therefore **not generated yet**.
### 3. Run the baseline experiment
```powershell
python run_baselines.py
```
This tunes the threshold and pre-cooling policies on training seeds, evaluates the controllers on evaluation seeds, prints the comparison table, and writes:
```text
results.csv
pareto.png
baseline_params.json
```
The current saved results came from `200` evaluation days.
### 4. Inspect the learned policy
```powershell
python inspect_policy.py
```
This reads `q_table.npy` and `policy.csv`.
It prints the workday `16:00-18:00` away-state actions, the share of evaluation steps from low-visit states, and cooling-mode usage.
It does not write a result file.
### 5. Run independent agents
```powershell
python multi_seed.py --agents 5 --episodes 20000
```
This trains the requested number of independent agents.
It writes one Q-table, one visits file, and one evaluation CSV per agent, then writes `multi_seed_summary.csv`.
The current `multi_seed_summary.csv` has `15` rows, so it was generated with a different agent count than the command above.
The current saved per-agent timings are between `0.486` and `0.583` minutes.
### 6. Run the comfort-weight sweep
```powershell
python lambda_sweep.py --agents 3 --episodes 20000
```
The code uses six lambda values:
```text
0.25, 0.5, 1.0, 2.0, 4.0, 8.0
```
It writes:
```text
lambda_sweep_raw.csv
lambda_sweep_summary.csv
lambda_sweep.png
```
The current saved sweep files already exist.
### 7. Open the results dashboard
```powershell
streamlit run dashboard.py
```
This opens the existing results, replay, and learned-policy dashboard.
It reads saved result files and `q_table.npy`.
### 8. Open the simplified professor demo
```powershell
streamlit run demo_app.py
```
This opens the smaller story-focused demo.
It uses the real environment, the saved Q-table, and the real reward function.
It does not claim real savings.
If `baseline_params.json` is missing, the demo cannot load the tuned simple rule. Run `python run_baselines.py` to create it.
### 9. Check a hand calculation
```powershell
python steady_state.py
```
The exact output depends on the helper’s current implementation.
Use it as a simulator sanity check, not as measured building data.
---
## Glossary
| Term | Plain meaning |
|---|---|
| State | The compact description the agent receives before acting |
| Action | One cooling-and-light choice |
| Reward | The score returned after an action |
| Episode | One simulated day |
| Step | One 30-minute decision point |
| Q-table | A notebook of values for state/action pairs |
| Q-value | The table’s guess for how good one action is in one state |
| State index | The single row number made from a state tuple |
| Epsilon | The chance of trying a random action |
| Exploration | Trying actions to learn about them |
| Exploitation | Using the best action currently known |
| Alpha | The learning-rate number controlling the update size |
| Gamma | The discount number controlling future importance |
| Reward target | The value the old Q-value moves toward |
| Baseline | A comparison controller that is not the learned agent |
| Held-out day | A simulated day not used for training |
| Training seed | A random seed used while learning |
| Validation seed | A separate seed used for checkpoint checking |
| Evaluation seed | A separate seed used for final comparison |
| Truncation | Stopping because the day boundary was reached |
| Termination | Stopping because no future state exists |
| Pre-cooling | Cooling before people return |
| Placeholder | An assumed value that still needs calibration |
| Effective temperature | Felt temperature after fan effects |
| Thermal discomfort | Squared distance outside the comfort band |
| Lighting discomfort | Penalty for an awake person in darkness with the light off |
| Exogenous | Generated by the environment, not controlled by the agent |
| Discretization | Turning continuous values into bins |
| Policy | The rule that maps a state to an action |
| Greedy action | The action with the largest current Q-value |
| Standard deviation | A simple measure of spread around a mean |
---
## Questions a professor may ask
### 1. Why use RL and not only rules?
Rules can react to the current temperature.
RL can learn that cooling now can affect a later temperature, such as before the household returns. The saved comparison also shows that the hand-tuned rule is still stronger than the saved trained agents, so RL is not claimed to be automatically better.
### 2. Why tabular Q-learning?
The state space is small enough for a table: `1008` states and `10` actions.
This makes the algorithm easy to inspect and explain. No deep-learning library is needed.
### 3. What is the reward?
It is negative energy cost plus negative comfort, lighting, and optional switching penalties.
The configured weights are `lambda_t=2.0`, `lambda_l=1.5`, and `lambda_s=0.0`.
### 4. What is exploration?
Exploration means trying a random action instead of always using the current best guess.
The code starts epsilon at `1.0`, decays it over `70%` of training, and ends at `0.05`.
### 5. What does the update do?
It moves the current Q-value toward the immediate reward plus the best future value.
With the project settings, `alpha=0.1` and `gamma=0.97`.
### 6. Why does the last step still use a future value?
The end of a day is truncation, not termination.
The environment returns `terminated=False`, so the update bootstraps from the wrapped next state.
### 7. Why does the agent tie or lose to the hand-tuned rule?
The best saved hand-tuned reward is `-51.36`.
The multi-seed trained-agent mean is `-60.72`. The tuned rule has direct, hand-designed pre-cooling behavior, while the learned table must discover it from finite experience.
### 8. How do you know it is not memorising the evaluation days?
Training uses seeds `0` through `999`.
Validation uses `20000` through `20049`, and evaluation uses `10000` through `10199`. The tests verify that these ranges do not overlap.
### 9. What does “placeholder parameter” mean?
It means the value was chosen for the simulator and was not measured from the real room, appliance labels, or electricity bill.
`CALIBRATED=False` makes this explicit.
### 10. What are the limits of the simulator?
It has one room, known occupancy, a flat tariff, simplified AC power, and discretized temperature.
It also uses generated weather noise and a simplified day wrap.
### 11. What did the agent learn about pre-cooling?
The inspected Q-table has different actions in the away, workday, `16:00-18:00` block for different temperature bins and weather regimes.
The saved pre-cooling baselines also receive the best mean rewards in the baseline table.
### 12. What would you do next?
Calibrate the physics and reward weights with real measurements.
Then add a value-iteration reference, more rooms, realistic appliance power, uncertain occupancy, and a time-of-use tariff.
---
## Documentation to fix
The code is the source of truth for these points.
### State-space size
`design.md` says `6 * 3 * 2 * 5 * 2 = 360` base states.
The code uses `12` time blocks and `7` temperature bins:
```text
12 * 3 * 2 * 7 * 2 = 1008
```
The README follows the code.
### Outdoor-temperature formula
`design.md` describes a sine wave.
`dynamics.py` implements a cosine wave.
### Time-block count
`design.md` describes `6` four-hour blocks.
`config.py` uses `2.0`-hour blocks, producing `12` blocks.
### Temperature-bin count
`design.md` describes `5` bins and example edges.
`config.py` contains six edges and `discretize.py` therefore creates `7` bins.
### Discount factor
`design.md` marks gamma as TBD and suggests trying values from `0.95` to `0.99`.
`train.py` uses `gamma=0.97` by default.
### End-of-day semantics
`design.md` originally says the day terminates and the terminal step should not bootstrap in one section.
`env.py`, `train.py`, `qlearning.py`, and the tests use truncation: `terminated=False`, `truncated=True`, with bootstrapping.
### Dynamics comments
The header comment in `dynamics.py` still calls `next_temperature()` and `compute_reward()` “Mode B stubs”.
Both functions are implemented.
### Baseline list
`design.md` mentions a value-iteration optimum.
The current implementation does not provide that controller.
### Baseline persistence
`run_baselines.py` now writes `baseline_params.json`.
That file is missing in the current workspace, so the simplified demo cannot load its tuned simple-rule parameters until the baseline command is run.
---
## Verified-number and missing-artifact checklist
### Numbers that could not be verified
The following requested facts were not available as measured or guaranteed values in the code or saved result files:
| Requested fact | What the README says instead |
|---|---|
| Real energy savings | Not claimed; all values are simulator outputs |
| Normal full-training wall-clock time | Not recorded; not generated yet |
| Real appliance power labels | Not available; values remain placeholders |
| Real electricity tariff | Not available; `7.0` is a placeholder |
| Value-iteration result | Not implemented; not generated yet |
| Causal real-home pre-cooling benefit | Not measured; not claimed |
### Places marked “not generated yet”
| Item | Why | Command that would create it |
|---|---|---|
| Normal full-training wall-clock measurement | The training script does not save timing | Run `python train.py` and time it externally |
| Value-iteration comparison | No value-iteration implementation exists | No current command; implementation is needed |
| Calibrated physical values | No real labels or bill data are included | Supply measurements and update configuration deliberately |
### Existing but currently missing or present
| Artifact | Current status |
|---|---|
| `q_table.npy` | Present and shape `(1008, 10)` |
| `policy.csv` | Present with `1008` state rows |
| `results.csv` | Present |
| `curves.csv` | Present with checkpoints through episode `20000` |
| `multi_seed_summary.csv` | Present with `15` agent rows |
| `lambda_sweep_summary.csv` | Present |
| `visit_counts.npy` | Not present in the current workspace |
| `baseline_params.json` | Not present in the current workspace |
Do not present simulator numbers as measured savings.
