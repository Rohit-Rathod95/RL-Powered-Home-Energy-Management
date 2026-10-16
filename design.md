# Home Electricity Saver: Phase 0 Design

Status: DRAFT. Everything marked **TBD** is yours to decide or calibrate. Nothing here is a measured fact. All numbers are placeholders until you set them from your own appliances and bill.

---

## 1. Problem and why RL

A simulated home room has a fan, an evaporative cooler, an AC and a light. An agent picks a cooling mode and a light setting every 30 minutes. It should keep the room comfortable while someone is home and avoid wasting electricity.

**Why RL and not rules:** the AC and cooler change the future room temperature, so an action now affects the cost and comfort later (for example, pre-cooling before the household returns). The best choice also depends on context (cooler works in dry weather, poorly in humid weather). The mapping from state to action is learned from the objective, not written by hand.

**What stays hand-designed (be honest in the viva):** the simulator, the reward weights, the comfort band, and every physical parameter.

---

## 2. Decisions log

| # | Decision | Choice |
|---|---|---|
| 1 | Scope | Single zone (one room), framed as a home |
| 2 | Activity model | Fixed schedule + small random noise; workday and weekend variants |
| 3 | Weather regime | Dry or humid, sampled at the start of each day |
| 4 | Tariff | Flat price (MVP). Two-period price is the first extension |
| 5 | Previous action in state | TBD, decide after seeing whether the agent chatters |
| 6 | Name | Home Electricity Saver |

---

## 3. Time and episode

- Step: 30 min. Episode: 1 day = 48 steps. Each episode starts at 00:00.
- Discount gamma: **TBD** (try 0.95 to 0.99 and compare).
- End of day is truncation, not termination: `terminated=False`, `truncated=True`. Q-learning still bootstraps from the wrapped next state.

---

## 4. State

| Variable | Values | Notes |
|---|---|---|
| Time block | 6 (4-hour blocks) | 00-04, 04-08, 08-12, 12-16, 16-20, 20-24 |
| Activity | 3: away / awake / asleep | Exogenous (follows the schedule, not the agent's action) |
| Day type | 2: workday / weekend | Needed because the schedule differs; without it the agent cannot tell when the household returns |
| Indoor temp bin | 5 | Bin edges **TBD** (e.g. around 25 / 28 / 31 / 34 degrees C) |
| Weather regime | 2: dry / humid | Sampled per day, fixed within the day |
| Previous cooling mode | optional | Decision 5 |

Base size: 6 x 3 x 2 x 5 x 2 = **360 states** (about 2,160 with previous mode, if added).

Daylight is **not** a state variable. It is derived from time of day.

---

## 5. Actions

Cooling mode (mutually exclusive) x light (on / off):

| Cooling mode | Effect |
|---|---|
| off | nothing running |
| fan_low | feels-cooler effect only |
| fan_high | stronger feels-cooler effect |
| cooler_fan | evaporative cooling, strength depends on weather regime |
| ac_fan_low | strong cooling toward a fixed setpoint |

5 x 2 = **10 actions**. AC setpoint choice is out of scope for the MVP.

---

## 6. Dynamics (simulator)

Indoor temperature, per step:

```
T[t+1] = T[t] + (dt/tau) * (T_out[t] - T[t]) + g_occ * occupied[t] - cooling(mode, regime, T[t])
```

- Outdoor temperature: `T_out[t] = T_mean + A * sin(2*pi*(hour - hour_peak)/24) + noise`
- `tau` (room thermal time constant): **TBD**, a few hours
- `g_occ`: heat from people, small, **TBD**
- Solar gain is folded into the outdoor temperature curve (fewer parameters).
- Cooling terms (all clipped so they never push the room past a physical floor):
  - fan: 0 (fans do not change air temperature)
  - cooler: `c_cooler * (1 - humidity_factor(regime)) * max(0, T - T_floor)`. `c_cooler`, humidity factors and `T_floor`: **TBD**
  - AC: `c_ac * max(0, T - T_set)`. `c_ac`, `T_set`: **TBD**

Effective (felt) temperature used for comfort:

```
T_eff = T - k_fan(mode)
```

`k_fan` values (degrees C of feels-cooler effect for fan_low, fan_high, and modes that include a fan): **TBD**.

---

## 7. Energy and cost

```
E[t]    = P(mode) * dt + P_light * light_on * dt     (kWh)
cost[t] = price * E[t]
```

| Parameter | Value |
|---|---|
| P_fan_low, P_fan_high | **TBD**, read your fan label |
| P_cooler_fan | **TBD**, read your cooler label |
| P_ac | **TBD**, read your AC label (inverter vs fixed-speed matters) |
| P_light | **TBD** |
| price (flat, per kWh) | **TBD**, read your bill |

Known simplification: AC power is treated as constant while on (no compressor cycling).

---

## 8. Comfort

Comfort is penalized only when someone is home (awake or asleep).

```
occupied   = (activity != away)
D_T[t]     = occupied * ( max(0, T_eff - T_hi)^2 + max(0, T_lo - T_eff)^2 )
D_L[t]     = 1 if (activity == awake and it is dark and light is off) else 0
```

Comfort band `[T_lo, T_hi]`: **TBD**. "Dark" is determined by time of day (daylight hours: **TBD**).

---

## 9. Reward

```
r[t] = - cost[t] - lambda_T * D_T[t] - lambda_L * D_L[t] - lambda_S * switched[t]
```

- `switched[t]` = 1 if the cooling mode changed from the previous step. Include only if decision 5 says so.
- All terms are in comparable units (think "rupee-equivalent"). `lambda_T`, `lambda_L`, `lambda_S`: **TBD**.
- There is deliberately **no** separate "empty room waste" penalty (the energy cost already covers it) and no hand-written "inappropriate lighting" rule (visual discomfort covers it).
- Log every component (cost, D_T, D_L, switches) separately, not just the sum.

---

## 10. Acceptance tests (before any learning)

1. Hand-compute 3 to 4 steps on paper and match the code.
2. Always-off is clearly worse than a sensible policy (comfort penalty outweighs energy saved).
3. Always-AC costs clearly more than a sensible policy.
4. The temperature curve under always-off looks like a lagged copy of the outdoor curve.
5. In the humid regime the cooler does visibly worse than in the dry regime.
6. No NaNs; per-step reward equals the sum of its logged components.

If any test fails, fix the environment or the weights before touching Q-learning.

---

## 11. Evaluation plan (summary)

- **Baselines:** always-on, always-off, random, timer schedule, simple threshold rule, tuned threshold rule (grid-searched), and the value-iteration optimum.
- **Metrics:** daily energy (kWh) and cost, discomfort-hours, % occupied time uncomfortable, light-needed-but-off %, energy used while away, switches per day.
- **Method:** at least 10 seeds, mean and std, evaluation on held-out simulated days, energy vs discomfort Pareto plot. Report what the experiments show.

---

## 12. Out of scope (for now)

Multi-room control, slab tariffs, AC setpoint choice, other appliances (geyser, washing machine), real data, deep RL, hardware.

Extensions, in rough priority: value-iteration oracle comparison, reward-weight sweep and Pareto front, occupant profile transfer, two-period tariff, policy heatmap in the dashboard.

---

## 13. Known weaknesses to state openly

- The agent learns this simulator, not a real home. The sim-to-real gap is unquantified.
- Reward weights are arbitrary, so different weights give different policies.
- Real occupancy is partially observable; here it is assumed known.
- Discretization hides distinctions inside a bin.
- Physical parameters are assumptions; run a sensitivity check.
