"""discretize.py

Discretization functions for mapping continuous environment observations
into the discrete state space specified in design.md Section 4.

Supports dynamic time_block_hours and arbitrary temperature bin edges.
Returns a 5-tuple by default, or a 6-tuple when include_prev_mode is True.
"""

from __future__ import annotations

import numpy as np

from config import EnvConfig


def get_time_block(step: int, config: EnvConfig) -> int:
    """Map a step index (0..48) to its discrete time block index.

    Uses config.time_block_hours to determine block intervals.

    Args:
        step: Step index within the day (0 to 48).
        config: EnvConfig specifying dt and time_block_hours.

    Returns:
        int: Block index in [0, num_blocks - 1], where num_blocks = 24 / time_block_hours.
    """
    num_blocks = int(round(24.0 / config.time_block_hours))
    hour = step * config.dt
    block = int(hour // config.time_block_hours)
    return min(max(0, block), num_blocks - 1)


def get_temp_bin(temp: float, bin_edges: tuple[float, ...]) -> int:
    """Map a continuous temperature to a discrete bin index.

    Number of bins = len(bin_edges) + 1.
    Bin 0: temp < bin_edges[0]
    Bin i: bin_edges[i-1] <= temp < bin_edges[i]
    Bin N: temp >= bin_edges[-1]

    Args:
        temp: Indoor air temperature in °C.
        bin_edges: Strictly ascending tuple of edge temperatures.

    Returns:
        int: Bin index in [0, len(bin_edges)].
    """
    return int(np.digitize(temp, bin_edges))


def discretize_state(
    step: int,
    activity: int,
    day_type: int,
    indoor_temp: float,
    weather_regime: int,
    config: EnvConfig,
    prev_cooling_mode: int = 0,
) -> tuple[int, ...]:
    """Map raw environment variables to a discrete state tuple.

    Args:
        step: Current step index in the episode (0..48).
        activity: Exogenous activity index (0=away, 1=awake, 2=asleep).
        day_type: Day type index (0=workday, 1=weekend).
        indoor_temp: Continuous indoor temperature in °C.
        weather_regime: Weather regime index (0=dry, 1=humid).
        config: EnvConfig instance.
        prev_cooling_mode: Previous cooling mode index (0..4).

    Returns:
        tuple[int, ...]:
            If config.include_prev_mode is False:
                (time_block, activity, day_type, temp_bin, weather_regime)  [5-tuple]
            If config.include_prev_mode is True:
                (time_block, activity, day_type, temp_bin, weather_regime, prev_cooling_mode) [6-tuple]
    """
    time_block = get_time_block(step, config)
    temp_bin = get_temp_bin(indoor_temp, config.temp_bins)

    if config.include_prev_mode:
        return (time_block, activity, day_type, temp_bin, weather_regime, prev_cooling_mode)
    return (time_block, activity, day_type, temp_bin, weather_regime)


def get_state_space_dims(config: EnvConfig) -> tuple[int, ...]:
    """Return the cardinality of each discrete state dimension.

    Returns:
        tuple[int, ...]: Dimensions of the discrete state space.
    """
    num_blocks = int(round(24.0 / config.time_block_hours))
    num_activities = 3
    num_day_types = 2
    num_temp_bins = len(config.temp_bins) + 1
    num_regimes = 2

    if config.include_prev_mode:
        num_modes = 5
        return (num_blocks, num_activities, num_day_types, num_temp_bins, num_regimes, num_modes)
    return (num_blocks, num_activities, num_day_types, num_temp_bins, num_regimes)
