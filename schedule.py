"""schedule.py

Exogenous activity schedule generator and lighting conditions for
Home Electricity Saver (Phase 1).

Generates occupancy activity (away, awake, asleep) per 30-minute step
for workday and weekend variants, including random jitter.
The schedule is strictly exogenous and independent of the agent's actions.
"""

from __future__ import annotations

import numpy as np

from config import (
    ACTIVITY_ASLEEP,
    ACTIVITY_AWAKE,
    ACTIVITY_AWAY,
    DAY_WORKDAY,
    EnvConfig,
)


def get_base_transitions(day_type: int, config: EnvConfig) -> tuple[int, int, int, int]:
    """Calculate base transition step indices (wake, leave, return, sleep) without jitter.

    Returns:
        tuple[int, int, int, int]: (wake_step, leave_step, return_step, sleep_step)
    """
    if day_type == DAY_WORKDAY:
        wake_h = config.workday_wake
        leave_h = config.workday_leave
        return_h = config.workday_return
        sleep_h = config.workday_sleep
    else:
        wake_h = config.weekend_wake
        leave_h = config.weekend_leave
        return_h = config.weekend_return
        sleep_h = config.weekend_sleep

    wake_step = int(round(wake_h / config.dt))
    leave_step = int(round(leave_h / config.dt))
    return_step = int(round(return_h / config.dt))
    sleep_step = int(round(sleep_h / config.dt))

    return wake_step, leave_step, return_step, sleep_step


def generate_daily_schedule(
    day_type: int,
    rng: np.random.Generator,
    config: EnvConfig | None = None,
) -> np.ndarray:
    """Generate a 48-step exogenous activity schedule for one day.

    Applies bounded random jitter to the transition times while enforcing
    monotonic ordering: 0 < wake < leave < return < sleep < episode_steps.

    Args:
        day_type: 0 for workday, 1 for weekend.
        rng: Seeded numpy random Generator instance.
        config: Optional EnvConfig instance.

    Returns:
        np.ndarray of shape (episode_steps,) with dtype int32 containing
        ACTIVITY_ASLEEP, ACTIVITY_AWAKE, or ACTIVITY_AWAY.
    """
    if config is None:
        config = EnvConfig()

    base_wake, base_leave, base_return, base_sleep = get_base_transitions(day_type, config)
    max_jitter = config.schedule_jitter_max_steps

    # Sample random integer jitters in [-max_jitter, +max_jitter]
    if max_jitter > 0:
        j_wake = int(rng.integers(-max_jitter, max_jitter + 1))
        j_leave = int(rng.integers(-max_jitter, max_jitter + 1))
        j_return = int(rng.integers(-max_jitter, max_jitter + 1))
        j_sleep = int(rng.integers(-max_jitter, max_jitter + 1))
    else:
        j_wake = j_leave = j_return = j_sleep = 0

    # Enforce strict monotonic order so transitions never cross or exceed bounds
    wake = max(1, base_wake + j_wake)
    leave = max(wake + 1, base_leave + j_leave)
    ret = max(leave + 1, base_return + j_return)
    sleep = min(config.episode_steps - 1, max(ret + 1, base_sleep + j_sleep))

    schedule = np.empty(config.episode_steps, dtype=np.int32)
    schedule[0:wake] = ACTIVITY_ASLEEP
    schedule[wake:leave] = ACTIVITY_AWAKE
    schedule[leave:ret] = ACTIVITY_AWAY
    schedule[ret:sleep] = ACTIVITY_AWAKE
    schedule[sleep:config.episode_steps] = ACTIVITY_ASLEEP

    return schedule


def is_dark(hour: float, config: EnvConfig) -> bool:
    """Determine whether it is dark outside based on time of day.

    Darkness is purely exogenous and derived from the 24-hour clock.

    Args:
        hour: Hour of day [0.0, 24.0).
        config: EnvConfig containing daylight start and end hours.

    Returns:
        True if the hour is before daylight_start_hour or after/equal daylight_end_hour.
    """
    return hour < config.daylight_start_hour or hour >= config.daylight_end_hour
