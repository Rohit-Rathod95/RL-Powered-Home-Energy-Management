"""config.py

Configuration constants, dataclass, and placeholder parameters for
Home Electricity Saver (Phase 1).

Every parameter marked TBD in design.md is explicitly flagged with a # PLACEHOLDER comment.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import warnings

# ============================================================================
# Calibration Flag (Section 13)
# ============================================================================
CALIBRATED: bool = False
_WARNED_CALIBRATION: bool = False


def check_calibration_warning() -> None:
    """Issue a warning once per process if the environment runs uncalibrated."""
    global _WARNED_CALIBRATION
    if not CALIBRATED and not _WARNED_CALIBRATION:
        warnings.warn(
            "CALIBRATED is False: Environment is running with uncalibrated placeholder parameters. "
            "Calibrate parameters against real appliance labels and electricity bills before evaluation.",
            UserWarning,
            stacklevel=2,
        )
        _WARNED_CALIBRATION = True


# ============================================================================
# Discrete Enums and Named Constants
# ============================================================================
# Cooling modes (5 mutually exclusive modes)
MODE_OFF = 0
MODE_FAN_LOW = 1
MODE_FAN_HIGH = 2
MODE_COOLER_FAN = 3
MODE_AC_FAN_LOW = 4
COOLING_MODES = ("off", "fan_low", "fan_high", "cooler_fan", "ac_fan_low")

# Light settings (2 settings)
LIGHT_OFF = 0
LIGHT_ON = 1

# Exogenous activity states (3 states)
ACTIVITY_AWAY = 0
ACTIVITY_AWAKE = 1
ACTIVITY_ASLEEP = 2
ACTIVITIES = ("away", "awake", "asleep")

# Day types (2 types)
DAY_WORKDAY = 0
DAY_WEEKEND = 1
DAY_TYPES = ("workday", "weekend")

# Weather regimes (2 regimes, sampled per day)
REGIME_DRY = 0
REGIME_HUMID = 1
REGIMES = ("dry", "humid")


# ============================================================================
# Schedule Base Timings (Hours in 24h clock)
# ============================================================================
# PLACEHOLDER: Workday schedule base times (set from household routine)
WORKDAY_WAKE_HOUR: float = 7.0       # 07:00
WORKDAY_LEAVE_HOUR: float = 8.5      # 08:30
WORKDAY_RETURN_HOUR: float = 18.0    # 18:00
WORKDAY_SLEEP_HOUR: float = 23.0     # 23:00

# PLACEHOLDER: Weekend schedule base times (set from household routine)
WEEKEND_WAKE_HOUR: float = 8.5       # 08:30
WEEKEND_LEAVE_HOUR: float = 12.0     # 12:00
WEEKEND_RETURN_HOUR: float = 16.0    # 16:00
WEEKEND_SLEEP_HOUR: float = 23.5     # 23:30

# PLACEHOLDER: Max random schedule jitter in 30-min steps (+/- 1 step = +/- 30 min)
SCHEDULE_JITTER_MAX_STEPS: int = 1


# ============================================================================
# Environment Configuration Dataclass
# ============================================================================
@dataclass(frozen=True)
class EnvConfig:
    """Environment configuration container with physical and reward parameters."""

    # Time discretization
    dt: float = 0.5                      # hours per step (30 min)
    episode_steps: int = 48              # 48 steps = 24 hours
    time_block_hours: float = 2.0        # duration of each time block in hours (default 2.0h -> 12 blocks)

    # State space option (Decision 5)
    include_prev_mode: bool = False      # if True, state includes prev_cooling_mode (6-tuple)

    # Thermal dynamics parameters (Section 6)
    # PLACEHOLDER: Room thermal time constant in hours (read/measure room insulation)
    tau: float = 5.0
    # PLACEHOLDER: Heat gain from occupants in °C per 30-min step
    g_occ: float = 0.12
    # PLACEHOLDER: Mean daily outdoor temperature in °C
    t_mean: float = 32.0
    # PLACEHOLDER: Outdoor temperature sine wave amplitude in °C
    t_amplitude: float = 6.0
    # PLACEHOLDER: Hour of day when outdoor temperature peaks (24h clock)
    t_peak_hour: float = 15.0
    # PLACEHOLDER: Standard deviation of outdoor temperature noise in °C
    outdoor_noise_std: float = 0.5
    # PLACEHOLDER: Standard deviation of initial indoor temp offset relative to T_out(00:00) in °C
    t_init_offset_std: float = 0.5

    # Cooling model parameters (Section 6)
    # Fractional gap closed per step: strictly between 0 and 1
    # PLACEHOLDER: Cooler fraction of gap closed per step
    c_cooler: float = 0.18
    # PLACEHOLDER: Cooler physical floor temperature in °C
    cooler_temp_floor: float = 24.0
    # PLACEHOLDER: Fractional degradation in dry weather (0 = no degradation)
    humidity_factor_dry: float = 0.20
    # PLACEHOLDER: Fractional degradation in humid weather (1 = completely ineffective)
    humidity_factor_humid: float = 0.80

    # Fractional gap closed per step: strictly between 0 and 1
    # PLACEHOLDER: AC fraction of gap closed per step toward setpoint
    c_ac: float = 0.35
    # PLACEHOLDER: AC setpoint in °C (chosen so steady state effective temp is in comfort band)
    ac_setpoint: float = 25.0

    # Fan effective temperature wind-chill effect (Section 6)
    # PLACEHOLDER: Perceived cooling effect of fan_low in °C
    k_fan_low: float = 1.5
    # PLACEHOLDER: Perceived cooling effect of fan_high in °C
    k_fan_high: float = 2.5

    # Appliance power ratings in kW (Section 7)
    # PLACEHOLDER: Fan low power in kW (read from appliance label, e.g. 40W = 0.04kW)
    p_fan_low: float = 0.04
    # PLACEHOLDER: Fan high power in kW (read from appliance label, e.g. 70W = 0.07kW)
    p_fan_high: float = 0.07
    # PLACEHOLDER: Evaporative cooler + pump + fan power in kW (e.g. 180W = 0.18kW)
    p_cooler_fan: float = 0.18
    # PLACEHOLDER: AC + fan power in kW (e.g. 1400W = 1.4kW)
    p_ac_fan_low: float = 1.40
    # PLACEHOLDER: Room light power in kW (e.g. 20W LED = 0.02kW)
    p_light: float = 0.02

    # Tariff (Section 7)
    # PLACEHOLDER: Flat electricity price per kWh (read from bill, e.g. ₹7.0/kWh)
    price_per_kwh: float = 7.0

    # Comfort band parameters (Section 8)
    # PLACEHOLDER: Comfort band lower bound in °C
    t_comfort_lo: float = 23.0
    # PLACEHOLDER: Comfort band upper bound in °C
    t_comfort_hi: float = 27.0
    # PLACEHOLDER: Daylight start hour (24h clock)
    daylight_start_hour: float = 6.0
    # PLACEHOLDER: Daylight end hour (24h clock)
    daylight_end_hour: float = 18.0

    # Reward penalty weights (Section 9)
    # PLACEHOLDER: Temperature discomfort penalty weight lambda_T
    lambda_t: float = 2.0
    # PLACEHOLDER: Visual discomfort penalty weight lambda_L
    lambda_l: float = 1.5
    # PLACEHOLDER: Switching penalty weight lambda_S (must be 0.0 if include_prev_mode is False)
    lambda_s: float = 0.0

    # Schedule base timings
    workday_wake: float = WORKDAY_WAKE_HOUR
    workday_leave: float = WORKDAY_LEAVE_HOUR
    workday_return: float = WORKDAY_RETURN_HOUR
    workday_sleep: float = WORKDAY_SLEEP_HOUR

    weekend_wake: float = WEEKEND_WAKE_HOUR
    weekend_leave: float = WEEKEND_LEAVE_HOUR
    weekend_return: float = WEEKEND_RETURN_HOUR
    weekend_sleep: float = WEEKEND_SLEEP_HOUR
    schedule_jitter_max_steps: int = SCHEDULE_JITTER_MAX_STEPS

    # Temperature bin edges (Section 4)
    # Default edges: t_comfort_lo, t_comfort_hi, t_comfort_hi + k_fan_low,
    # t_comfort_hi + k_fan_high, plus two hotter placeholder edges.
    temp_bins: tuple[float, ...] = field(
        default_factory=lambda: (
            23.0,  # t_comfort_lo
            27.0,  # t_comfort_hi
            28.5,  # t_comfort_hi + k_fan_low (27.0 + 1.5)
            29.5,  # t_comfort_hi + k_fan_high (27.0 + 2.5)
            32.0,  # PLACEHOLDER: hotter edge 1
            35.0,  # PLACEHOLDER: hotter edge 2
        )
    )

    def __post_init__(self) -> None:
        """Validate configuration consistency."""
        if not self.include_prev_mode and self.lambda_s != 0.0:
            raise ValueError(
                f"lambda_s must be 0.0 when include_prev_mode is False (Decision 5), "
                f"got lambda_s={self.lambda_s}."
            )
        if not (0.0 < self.c_cooler < 1.0):
            raise ValueError(
                f"c_cooler must be strictly between 0 and 1 (fraction of gap closed), "
                f"got c_cooler={self.c_cooler}."
            )
        if not (0.0 < self.c_ac < 1.0):
            raise ValueError(
                f"c_ac must be strictly between 0 and 1 (fraction of gap closed), "
                f"got c_ac={self.c_ac}."
            )
        if self.time_block_hours <= 0.0 or 24.0 % self.time_block_hours != 0.0:
            raise ValueError(
                f"time_block_hours must be a positive divisor of 24, "
                f"got {self.time_block_hours}."
            )
        if not (0.0 <= self.humidity_factor_dry <= 1.0 and 0.0 <= self.humidity_factor_humid <= 1.0):
            raise ValueError("Humidity degradation factors must be within [0, 1].")
        if self.t_comfort_lo >= self.t_comfort_hi:
            raise ValueError("t_comfort_lo must be strictly less than t_comfort_hi.")
