"""baselines.py

Baseline controllers for Home Electricity Saver (Phase 2).

All controllers operate strictly on the discretized state (fair comparison with RL agents):
    state = (time_block, activity, day_type, temp_bin, weather_regime, [prev_cooling_mode])

Controllers implemented:
  1. AlwaysOffPolicy: Cooling off, light off (action 0).
  2. AlwaysOnPolicy: AC on, light on (action 9).
  3. RandomPolicy: Uniform random action in [0, 9] (seeded).
  4. TimerPolicy: Fixed daily schedule based on time_block.
  5. ThresholdPolicy: Hand-written rules on discretized state.
  6. TunedThresholdPolicy: Grid-searched threshold policy tuned on TRAIN seeds only.
"""

from __future__ import annotations

from typing import Any, Protocol
import numpy as np

from config import (
    ACTIVITY_ASLEEP,
    ACTIVITY_AWAKE,
    ACTIVITY_AWAY,
    COOLING_MODES,
    DAY_WORKDAY,
    EnvConfig,
    LIGHT_OFF,
    LIGHT_ON,
    MODE_AC_FAN_LOW,
    MODE_COOLER_FAN,
    MODE_FAN_HIGH,
    MODE_FAN_LOW,
    MODE_OFF,
    REGIME_DRY,
    REGIME_HUMID,
)
from env import RoomEnv, encode_action
from schedule import is_dark


class Policy(Protocol):
    """Protocol for state-based policies."""

    def select_action(self, state: tuple[int, ...]) -> int:
        """Select an action in [0, 9] given the discretized state tuple."""
        ...


class AlwaysOffPolicy:
    """Always turns off all cooling and lighting (action 0)."""

    name: str = "Always Off"

    def select_action(self, state: tuple[int, ...]) -> int:
        return encode_action(MODE_OFF, LIGHT_OFF)


class AlwaysOnPolicy:
    """Always runs AC and keeps light on (action 9)."""

    name: str = "Always On (AC + Light)"

    def select_action(self, state: tuple[int, ...]) -> int:
        return encode_action(MODE_AC_FAN_LOW, LIGHT_ON)


class RandomPolicy:
    """Selects an action uniformly at random from [0, 9] using a seeded RNG."""

    name: str = "Random"

    def __init__(self, seed: int = 42) -> None:
        self.rng = np.random.default_rng(seed)

    def select_action(self, state: tuple[int, ...]) -> int:
        return int(self.rng.integers(0, 10))


def is_dark_block(time_block: int, config: EnvConfig) -> bool:
    """Determine if a time block falls predominantly during dark hours."""
    # Use midpoint of the time block
    mid_hour = (time_block + 0.5) * config.time_block_hours
    return is_dark(mid_hour, config)


class TimerPolicy:
    """Fixed-schedule timer based on time blocks.

    Schedule logic:
      - Evening return hours (18:00 to 22:00/23:00): AC on, light on if dark.
      - Sleep hours (23:00 to 07:00): Fan low, light off.
      - Away / daytime working hours: Cooling off, light off.
    """

    name: str = "Timer"

    def __init__(self, config: EnvConfig | None = None) -> None:
        self.config = config if config is not None else EnvConfig()

    def select_action(self, state: tuple[int, ...]) -> int:
        time_block = state[0]
        hour = time_block * self.config.time_block_hours
        dark = is_dark_block(time_block, self.config)

        # Evening return period: 18:00 to 23:00
        if 18.0 <= hour < 23.0:
            mode = MODE_AC_FAN_LOW
            light = LIGHT_ON if dark else LIGHT_OFF
        # Night sleeping period: 23:00 to 07:00
        elif hour >= 23.0 or hour < 7.0:
            mode = MODE_FAN_LOW
            light = LIGHT_OFF
        # Morning / afternoon away period: 07:00 to 18:00
        else:
            mode = MODE_OFF
            light = LIGHT_OFF

        return encode_action(mode, light)


class ThresholdPolicy:
    """Hand-crafted threshold rules on discretized state.

    Rules:
      1. Lighting: Turn on light only when awake and dark.
      2. Cooling:
         - If away: Cooling off.
         - If occupied:
           - bin <= 1 (<= 27°C, inside comfort band): Cooling off.
           - bin == 2 (27.0 - 28.5°C): Fan low.
           - bin == 3 (28.5 - 29.5°C): Fan high.
           - bin >= 4 (> 29.5°C):
               - If dry regime: Evaporative cooler.
               - If humid regime: AC.
    """

    name: str = "Threshold Rule"

    def __init__(self, config: EnvConfig | None = None) -> None:
        self.config = config if config is not None else EnvConfig()

    def select_action(self, state: tuple[int, ...]) -> int:
        time_block, activity, day_type, temp_bin, weather_regime = state[:5]
        dark = is_dark_block(time_block, self.config)

        # Light rule
        light = LIGHT_ON if (activity == ACTIVITY_AWAKE and dark) else LIGHT_OFF

        # Cooling rule
        if activity == ACTIVITY_AWAY:
            mode = MODE_OFF
        else:
            if temp_bin <= 1:
                mode = MODE_OFF
            elif temp_bin == 2:
                mode = MODE_FAN_LOW
            elif temp_bin == 3:
                mode = MODE_FAN_HIGH
            else:  # temp_bin >= 4
                if weather_regime == REGIME_DRY:
                    mode = MODE_COOLER_FAN
                else:
                    mode = MODE_AC_FAN_LOW

        return encode_action(mode, light)


class TunedThresholdPolicy:
    """Threshold policy with grid-searched parameters fitted on TRAIN seeds only.

    Parameters:
      - mode_bin_2: cooling mode for temp_bin == 2
      - mode_bin_3: cooling mode for temp_bin == 3
      - mode_bin_4_dry: cooling mode for temp_bin >= 4 in dry regime
      - mode_bin_4_humid: cooling mode for temp_bin >= 4 in humid regime
      - precool_mode: cooling mode in the time block right before return on workdays
    """

    name: str = "Tuned Threshold"

    def __init__(
        self,
        mode_bin_2: int = MODE_FAN_LOW,
        mode_bin_3: int = MODE_FAN_HIGH,
        mode_bin_4_dry: int = MODE_COOLER_FAN,
        mode_bin_4_humid: int = MODE_AC_FAN_LOW,
        precool_mode: int = MODE_OFF,
        config: EnvConfig | None = None,
    ) -> None:
        self.config = config if config is not None else EnvConfig()
        self.mode_bin_2 = mode_bin_2
        self.mode_bin_3 = mode_bin_3
        self.mode_bin_4_dry = mode_bin_4_dry
        self.mode_bin_4_humid = mode_bin_4_humid
        self.precool_mode = precool_mode

        # Precompute the return block on workdays (workday return is 18:00)
        # Block immediately preceding 18:00 (e.g. 16:00-18:00 or 17:00-18:00)
        return_block = int(self.config.workday_return // self.config.time_block_hours)
        self.precool_block = max(0, return_block - 1)

    def select_action(self, state: tuple[int, ...]) -> int:
        time_block, activity, day_type, temp_bin, weather_regime = state[:5]
        dark = is_dark_block(time_block, self.config)

        # Light rule: light on when awake and dark
        light = LIGHT_ON if (activity == ACTIVITY_AWAKE and dark) else LIGHT_OFF

        # Cooling rule
        if activity == ACTIVITY_AWAY:
            # Check pre-cooling condition on workdays in the block right before return
            if day_type == DAY_WORKDAY and time_block == self.precool_block and self.precool_mode != MODE_OFF:
                mode = self.precool_mode
            else:
                mode = MODE_OFF
        else:
            if temp_bin <= 1:
                mode = MODE_OFF
            elif temp_bin == 2:
                mode = self.mode_bin_2
            elif temp_bin == 3:
                mode = self.mode_bin_3
            else:  # temp_bin >= 4
                if weather_regime == REGIME_DRY:
                    mode = self.mode_bin_4_dry
                else:
                    mode = self.mode_bin_4_humid

        return encode_action(mode, light)


def tune_threshold_policy(
    train_seeds: list[int] | range,
    config: EnvConfig | None = None,
) -> TunedThresholdPolicy:
    """Grid-search candidate threshold parameters strictly on TRAIN seeds.

    Args:
        train_seeds: Iterable of integer seeds strictly within training split (e.g. 0..999).
        config: EnvConfig instance.

    Returns:
        TunedThresholdPolicy: Policy configured with the best-performing parameters.
            The returned policy also has a `best_train_reward` attribute equal to the
            highest average reward found in the grid (same seeds and env as tuning).
    """
    cfg = config if config is not None else EnvConfig()

    # ------------------------------------------------------------------
    # Candidate parameter grids — strict superset of the original 144-combo grid:
    #   Original: bin_2(3) × bin_3(4) × bin_4_dry(2) × bin_4_humid(2) × precool(3) = 144
    #   This:     bin_2(4) × bin_3(4) × bin_4_dry(3) × bin_4_humid(2) × precool(3) = 288
    #
    #   bin_2    : added MODE_FAN_HIGH (was [OFF, FAN_LOW, COOLER_FAN])
    #   bin_3    : unchanged [FAN_LOW, FAN_HIGH, COOLER_FAN, AC_FAN_LOW]
    #   bin_4_dry: added MODE_FAN_HIGH (was [COOLER_FAN, AC_FAN_LOW])
    #   bin_4_humid: unchanged [COOLER_FAN, AC_FAN_LOW]
    #   precool  : unchanged [OFF, AC_FAN_LOW, COOLER_FAN]
    # ------------------------------------------------------------------
    candidate_bin_2      = [MODE_OFF, MODE_FAN_LOW, MODE_FAN_HIGH, MODE_COOLER_FAN]
    candidate_bin_3      = [MODE_FAN_LOW, MODE_FAN_HIGH, MODE_COOLER_FAN, MODE_AC_FAN_LOW]
    candidate_bin_4_dry  = [MODE_FAN_HIGH, MODE_COOLER_FAN, MODE_AC_FAN_LOW]
    candidate_bin_4_humid = [MODE_COOLER_FAN, MODE_AC_FAN_LOW]
    candidate_precool    = [MODE_OFF, MODE_AC_FAN_LOW, MODE_COOLER_FAN]

    best_reward = -float("inf")
    best_params = (MODE_FAN_LOW, MODE_FAN_HIGH, MODE_COOLER_FAN, MODE_AC_FAN_LOW, MODE_OFF)

    # Use the first 200 train seeds (matches precool tuning; fast but representative)
    sample_seeds = list(train_seeds)[:200]

    env = RoomEnv(cfg)

    for b2 in candidate_bin_2:
        for b3 in candidate_bin_3:
            for b4_d in candidate_bin_4_dry:
                for b4_h in candidate_bin_4_humid:
                    for pc in candidate_precool:
                        candidate_policy = TunedThresholdPolicy(
                            mode_bin_2=b2,
                            mode_bin_3=b3,
                            mode_bin_4_dry=b4_d,
                            mode_bin_4_humid=b4_h,
                            precool_mode=pc,
                            config=cfg,
                        )

                        # Evaluate on train seeds
                        total_rewards = 0.0
                        for s in sample_seeds:
                            state, _ = env.reset(seed=s)
                            ep_reward = 0.0
                            truncated = False
                            while not truncated:
                                action = candidate_policy.select_action(state)
                                state, r, _, truncated, _ = env.step(action)
                                ep_reward += r
                            total_rewards += ep_reward

                        avg_reward = total_rewards / len(sample_seeds)
                        if avg_reward > best_reward:
                            best_reward = avg_reward
                            best_params = (b2, b3, b4_d, b4_h, pc)

    policy = TunedThresholdPolicy(
        mode_bin_2=best_params[0],
        mode_bin_3=best_params[1],
        mode_bin_4_dry=best_params[2],
        mode_bin_4_humid=best_params[3],
        precool_mode=best_params[4],
        config=cfg,
    )
    # Store the best average reward found in the grid (on the tuning seeds)
    # so callers can verify the returned policy is truly the argmax.
    policy.best_train_reward = best_reward
    return policy


# ============================================================================
# Pre-cooling Controllers (A1)
# ============================================================================
class TunedPrecoolPolicy:
    """Tuned threshold rule plus pre-cooling based on discretized state.

    While the household is away, within lead_steps steps before the NOMINAL return time
    (evaluated at time_block resolution) and temperature bin is at or above trigger_bin,
    runs precool_mode.

    Operates strictly on the discretized state tuple:
        state = (time_block, activity, day_type, temp_bin, weather_regime)
    """

    name: str = "Tuned Precool (Block)"

    def __init__(
        self,
        base_policy: TunedThresholdPolicy,
        lead_steps: int = 4,
        trigger_bin: int = 3,
        precool_mode: int = MODE_AC_FAN_LOW,
        config: EnvConfig | None = None,
    ) -> None:
        self.config = config if config is not None else EnvConfig()
        self.base_policy = base_policy
        self.lead_steps = lead_steps
        self.trigger_bin = trigger_bin
        self.precool_mode = precool_mode

        # Time block resolution: convert lead_steps to lead_blocks
        # 1 step = config.dt hours. 1 block = config.time_block_hours.
        steps_per_block = max(1, int(round(self.config.time_block_hours / self.config.dt)))
        self.lead_blocks = int(np.ceil(lead_steps / steps_per_block)) if lead_steps > 0 else 0

        # Nominal return blocks from config base schedule
        self.workday_return_block = int(self.config.workday_return // self.config.time_block_hours)
        self.weekend_return_block = int(self.config.weekend_return // self.config.time_block_hours)

    def select_action(self, state: tuple[int, ...]) -> int:
        time_block, activity, day_type, temp_bin, weather_regime = state[:5]

        # While household is away, strictly follow tuned pre-cooling rule
        if activity == ACTIVITY_AWAY:
            if self.lead_blocks > 0:
                nom_return_block = (
                    self.workday_return_block if day_type == DAY_WORKDAY else self.weekend_return_block
                )
                # Within lead_blocks before return
                if nom_return_block - self.lead_blocks <= time_block < nom_return_block:
                    if temp_bin >= self.trigger_bin:
                        return encode_action(self.precool_mode, LIGHT_OFF)
            return encode_action(MODE_OFF, LIGHT_OFF)

        # When occupied, fall back to base occupied threshold rules
        return self.base_policy.select_action(state)


class TunedPrecoolClockPolicy:
    """Tuned threshold rule plus pre-cooling with access to raw step index.

    # NOTE (A1): The clock variant has finer time information than a block-level
    # learner because it knows the exact 30-minute step index within the day,
    # rather than being quantized into 2-hour or multi-hour time blocks.

    While the household is away, within lead_steps steps before the NOMINAL return time
    (evaluated at step-level resolution) and temperature bin is at or above trigger_bin,
    runs precool_mode.
    """

    name: str = "Tuned Precool (Clock)"

    def __init__(
        self,
        base_policy: TunedThresholdPolicy,
        lead_steps: int = 3,
        trigger_bin: int = 3,
        precool_mode: int = MODE_AC_FAN_LOW,
        config: EnvConfig | None = None,
    ) -> None:
        self.config = config if config is not None else EnvConfig()
        self.base_policy = base_policy
        self.lead_steps = lead_steps
        self.trigger_bin = trigger_bin
        self.precool_mode = precool_mode
        self.current_step: int = 0

        # Nominal return steps from config base schedule
        self.workday_return_step = int(round(self.config.workday_return / self.config.dt))
        self.weekend_return_step = int(round(self.config.weekend_return / self.config.dt))

    def reset(self) -> None:
        """Reset internal step counter for new episode."""
        self.current_step = 0

    def select_action(self, state: tuple[int, ...], step: int | None = None) -> int:
        step_idx = step if step is not None else self.current_step
        self.current_step = (step_idx + 1) % self.config.episode_steps

        time_block, activity, day_type, temp_bin, weather_regime = state[:5]

        # While household is away, strictly follow tuned step-level pre-cooling rule
        if activity == ACTIVITY_AWAY:
            if self.lead_steps > 0:
                nom_return_step = (
                    self.workday_return_step if day_type == DAY_WORKDAY else self.weekend_return_step
                )
                # Within lead_steps before return
                if nom_return_step - self.lead_steps <= step_idx < nom_return_step:
                    if temp_bin >= self.trigger_bin:
                        return encode_action(self.precool_mode, LIGHT_OFF)
            return encode_action(MODE_OFF, LIGHT_OFF)

        # When occupied, fall back to base occupied threshold rules
        return self.base_policy.select_action(state)


def tune_precool_policy(
    train_seeds: list[int] | range,
    base_policy: TunedThresholdPolicy,
    config: EnvConfig | None = None,
) -> TunedPrecoolPolicy:
    """Grid-search pre-cooling parameters (lead_steps 0..8, trigger_bin, mode) on TRAIN seeds.

    Operates at time_block resolution.
    """
    cfg = config if config is not None else EnvConfig()
    sample_seeds = list(train_seeds)[:200]
    env = RoomEnv(cfg)

    candidate_lead_steps = [0, 2, 4, 6, 8]
    candidate_trigger_bins = [2, 3, 4]
    candidate_modes = [MODE_AC_FAN_LOW, MODE_COOLER_FAN]

    best_reward = -float("inf")
    best_params = (0, 3, MODE_AC_FAN_LOW)

    for ls in candidate_lead_steps:
        for tb in candidate_trigger_bins:
            for mode in candidate_modes:
                pol = TunedPrecoolPolicy(
                    base_policy=base_policy,
                    lead_steps=ls,
                    trigger_bin=tb,
                    precool_mode=mode,
                    config=cfg,
                )

                total_rewards = 0.0
                for s in sample_seeds:
                    state, _ = env.reset(seed=s)
                    ep_r = 0.0
                    truncated = False
                    while not truncated:
                        a = pol.select_action(state)
                        state, r, _, truncated, _ = env.step(a)
                        ep_r += r
                    total_rewards += ep_r

                avg_r = total_rewards / len(sample_seeds)
                if avg_r > best_reward:
                    best_reward = avg_r
                    best_params = (ls, tb, mode)

    return TunedPrecoolPolicy(
        base_policy=base_policy,
        lead_steps=best_params[0],
        trigger_bin=best_params[1],
        precool_mode=best_params[2],
        config=cfg,
    )


def tune_precool_clock_policy(
    train_seeds: list[int] | range,
    base_policy: TunedThresholdPolicy,
    config: EnvConfig | None = None,
) -> TunedPrecoolClockPolicy:
    """Grid-search pre-cooling clock parameters (lead_steps 0..8, trigger_bin, mode) on TRAIN seeds.

    Operates at raw step-level resolution.
    """
    cfg = config if config is not None else EnvConfig()
    sample_seeds = list(train_seeds)[:200]
    env = RoomEnv(cfg)

    candidate_lead_steps = list(range(0, 9))  # 0..8 steps
    candidate_trigger_bins = [2, 3, 4]
    candidate_modes = [MODE_AC_FAN_LOW, MODE_COOLER_FAN]

    best_reward = -float("inf")
    best_params = (0, 3, MODE_AC_FAN_LOW)

    for ls in candidate_lead_steps:
        for tb in candidate_trigger_bins:
            for mode in candidate_modes:
                pol = TunedPrecoolClockPolicy(
                    base_policy=base_policy,
                    lead_steps=ls,
                    trigger_bin=tb,
                    precool_mode=mode,
                    config=cfg,
                )

                total_rewards = 0.0
                for s in sample_seeds:
                    pol.reset()
                    state, _ = env.reset(seed=s)
                    ep_r = 0.0
                    truncated = False
                    while not truncated:
                        step_now = env.step_idx
                        a = pol.select_action(state, step=step_now)
                        state, r, _, truncated, _ = env.step(a)
                        ep_r += r
                    total_rewards += ep_r

                avg_r = total_rewards / len(sample_seeds)
                if avg_r > best_reward:
                    best_reward = avg_r
                    best_params = (ls, tb, mode)

    return TunedPrecoolClockPolicy(
        base_policy=base_policy,
        lead_steps=best_params[0],
        trigger_bin=best_params[1],
        precool_mode=best_params[2],
        config=cfg,
    )

