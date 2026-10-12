"""env.py

Hand-crafted Gym-style environment for Home Electricity Saver (Phase 1).

Interface:
    reset(seed=None) -> (state, info)
    step(action)     -> (state, reward, terminated, truncated, info)

Key Design Features:
- Exactly 48 steps of 30 minutes (00:00 to 24:00) per episode.
- Exogenous variables (activity schedule and outdoor noise) pre-generated at reset().
- Single np.random.Generator seeded in reset() for strict determinism.
- At step 48, returns terminated=False, truncated=True.
- Wrapped next state on truncation carries over indoor temperature, day type,
  weather regime, and prev_cooling_mode, with time block wrapped to 0 and
  activity from the start of the schedule.
- Initial prev_cooling_mode at reset is MODE_OFF (0).
"""

from __future__ import annotations

from typing import Any
import numpy as np

from config import (
    ACTIVITY_AWAY,
    COOLING_MODES,
    DAY_WORKDAY,
    DAY_WEEKEND,
    EnvConfig,
    LIGHT_OFF,
    LIGHT_ON,
    MODE_OFF,
    REGIME_DRY,
    REGIME_HUMID,
    check_calibration_warning,
)
from discretize import discretize_state
from dynamics import (
    compute_effective_temp,
    compute_energy,
    compute_reward,
    next_temperature,
)
from schedule import generate_daily_schedule, is_dark


def encode_action(cooling_mode: int, light_on: int) -> int:
    """Encode (cooling_mode, light_on) into a single integer action in [0, 9].

    Args:
        cooling_mode: Integer in [0, 4] (MODE_OFF..MODE_AC_FAN_LOW).
        light_on: Integer in [0, 1] (LIGHT_OFF, LIGHT_ON).

    Returns:
        int: Action index in [0, 9].
    """
    if not (0 <= cooling_mode < len(COOLING_MODES)):
        raise ValueError(f"Invalid cooling_mode {cooling_mode}; must be in [0, {len(COOLING_MODES)-1}].")
    if light_on not in (LIGHT_OFF, LIGHT_ON):
        raise ValueError(f"Invalid light_on {light_on}; must be 0 or 1.")
    return cooling_mode * 2 + light_on


def decode_action(action: int) -> tuple[int, int]:
    """Decode a single integer action in [0, 9] into (cooling_mode, light_on).

    Args:
        action: Action integer in [0, 9].

    Returns:
        tuple[int, int]: (cooling_mode, light_on).
    """
    if not (0 <= action < 10):
        raise ValueError(f"Invalid action {action}; must be in [0, 9].")
    return action // 2, action % 2


class RoomEnv:
    """Single-room home energy management environment (Phase 1)."""

    def __init__(self, config: EnvConfig | None = None) -> None:
        """Initialize RoomEnv and issue calibration warning if applicable."""
        check_calibration_warning()
        self.config: EnvConfig = config if config is not None else EnvConfig()

        self.rng: np.random.Generator = np.random.default_rng()
        self.day_type: int = DAY_WORKDAY
        self.weather_regime: int = REGIME_DRY
        self.activity_schedule: np.ndarray = np.zeros(self.config.episode_steps, dtype=np.int32)
        self.outdoor_noises: np.ndarray = np.zeros(self.config.episode_steps, dtype=np.float64)

        self.step_idx: int = 0
        self.indoor_temp: float = self.config.t_mean
        self.prev_cooling_mode: int = MODE_OFF

    def reset(self, seed: int | None = None) -> tuple[tuple[int, ...], dict[str, Any]]:
        """Reset environment to the beginning of a day (00:00).

        Strictly deterministic when seed is provided:
        1. Samples day type (workday / weekend) and weather regime (dry / humid).
        2. Pre-generates the entire 48-step exogenous activity schedule.
        3. Pre-generates the entire 48-step exogenous outdoor noise sequence.
        4. Sets initial prev_cooling_mode to MODE_OFF (0).
        5. Sets initial indoor room temperature to t_mean.

        Args:
            seed: Optional integer seed for the random number generator.

        Returns:
            tuple[tuple[int, ...], dict[str, Any]]: (initial_discrete_state, info)
        """
        self.rng = np.random.default_rng(seed)

        self.day_type = int(self.rng.choice([DAY_WORKDAY, DAY_WEEKEND]))
        self.weather_regime = int(self.rng.choice([REGIME_DRY, REGIME_HUMID]))

        # Pre-generate exogenous sequences upfront so they never depend on actions
        self.activity_schedule = generate_daily_schedule(self.day_type, self.rng, self.config)
        self.outdoor_noises = self.rng.normal(0.0, self.config.outdoor_noise_std, size=self.config.episode_steps)

        self.step_idx = 0
        self.indoor_temp = float(self.config.t_mean)
        # Requirement 7: Initial prev_cooling_mode at reset is MODE_OFF (0)
        self.prev_cooling_mode = MODE_OFF

        obs = discretize_state(
            step=0,
            activity=int(self.activity_schedule[0]),
            day_type=self.day_type,
            indoor_temp=self.indoor_temp,
            weather_regime=self.weather_regime,
            config=self.config,
            prev_cooling_mode=self.prev_cooling_mode,
        )

        info: dict[str, Any] = {
            "step": 0,
            "day_type": self.day_type,
            "weather_regime": self.weather_regime,
            "activity": int(self.activity_schedule[0]),
            "indoor_temp": self.indoor_temp,
            "prev_cooling_mode": self.prev_cooling_mode,
        }

        return obs, info

    def step(self, action: int) -> tuple[tuple[int, ...], float, bool, bool, dict[str, Any]]:
        """Execute one 30-minute time step.

        Args:
            action: Integer in [0, 9] encoding (cooling_mode, light_on).

        Returns:
            tuple containing:
                - next_state (tuple[int, ...]): Discretized state.
                - reward (float): Scalar reward.
                - terminated (bool): Always False in this daily episodic setting.
                - truncated (bool): True at step 48 (end of day).
                - info (dict): Detailed breakdown of reward components and physical states.
        """
        if self.step_idx >= self.config.episode_steps:
            raise RuntimeError(
                f"Episode has already ended (step {self.step_idx} >= {self.config.episode_steps}). "
                "Call reset() before stepping again."
            )

        cooling_mode, light_on = decode_action(action)
        switched = (cooling_mode != self.prev_cooling_mode)

        hour = self.step_idx * self.config.dt
        activity = int(self.activity_schedule[self.step_idx])
        outdoor_noise = float(self.outdoor_noises[self.step_idx])
        dark = is_dark(hour, self.config)
        occupied = (activity != ACTIVITY_AWAY)

        # 1. Electrical energy consumed during this step
        energy_kwh = compute_energy(cooling_mode, light_on, self.config)

        # 2. Perceived effective temperature for comfort evaluation
        t_eff = compute_effective_temp(self.indoor_temp, cooling_mode, self.config)

        # 3. Reward and component breakdown (Mode B stub)
        reward, components = compute_reward(
            effective_temp=t_eff,
            activity=activity,
            is_dark=dark,
            light_on=light_on,
            energy_kwh=energy_kwh,
            switched=switched,
            config=self.config,
        )

        # 4. Thermal transition: next indoor temperature T[t+1] (Mode B stub)
        next_temp = next_temperature(
            current_temp=self.indoor_temp,
            cooling_mode=cooling_mode,
            weather_regime=self.weather_regime,
            hour=hour,
            occupied=occupied,
            outdoor_noise=outdoor_noise,
            config=self.config,
        )

        # Update environment internal state
        self.indoor_temp = next_temp
        self.prev_cooling_mode = cooling_mode
        self.step_idx += 1

        # Check end of episode (step 48 = 24.0 hours)
        terminated = False
        if self.step_idx >= self.config.episode_steps:
            truncated = True
            # Wrapped next state approximation:
            # - time block wraps to 0 (00:00 to 00:00+time_block_hours)
            # - activity wraps to start of day (schedule[0], which is asleep)
            # - continuous indoor_temp, day_type, weather_regime, and prev_cooling_mode carry over
            next_state = discretize_state(
                step=0,
                activity=int(self.activity_schedule[0]),
                day_type=self.day_type,
                indoor_temp=self.indoor_temp,
                weather_regime=self.weather_regime,
                config=self.config,
                prev_cooling_mode=self.prev_cooling_mode,
            )
        else:
            truncated = False
            next_state = discretize_state(
                step=self.step_idx,
                activity=int(self.activity_schedule[self.step_idx]),
                day_type=self.day_type,
                indoor_temp=self.indoor_temp,
                weather_regime=self.weather_regime,
                config=self.config,
                prev_cooling_mode=self.prev_cooling_mode,
            )

        info: dict[str, Any] = {
            "step": self.step_idx,
            "cost": components["cost"],
            "D_T": components["D_T"],
            "D_L": components["D_L"],
            "switched": switched,
            "energy_kwh": energy_kwh,
            "indoor_temp": self.indoor_temp,
            "effective_temp": t_eff,
            "activity": activity,
            "day_type": self.day_type,
            "weather_regime": self.weather_regime,
            "prev_cooling_mode": self.prev_cooling_mode,
        }

        return next_state, reward, terminated, truncated, info
