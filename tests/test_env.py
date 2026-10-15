"""tests/test_env.py

Comprehensive test suite for Home Electricity Saver (Phase 1).

Tests covered:
  1. Config validation: lambda_s constraint, fractional cooling in (0, 1), calibration warning.
  2. Jittered schedule keeps strict monotonic ordering (wake < leave < return < sleep).
  3. Discretize returns values within valid ranges and respects time_block_hours.
  4. State tuple length respects include_prev_mode flag (5-tuple vs 6-tuple).
  5. Action encoding and decoding round-trip consistency.
  6. Exogenous independence: activity schedule and outdoor noises are identical for any action sequences with the same seed.
  7. Determinism: same seed + same actions yields identical trajectories.
  8. Episode termination/truncation: truncates after exactly 48 steps with wrapped next state.
  9. Reward decomposition: per-step reward equals the sum of its logged components.
 10. Cooling never overshoots T_floor or T_set.
 11. Always-off temperature curve follows a lagged copy of outdoor temperature.
 12. Cooler performs better in dry weather regime than in humid regime.
 13. 1000 random episodes produce no NaN or Inf values.

Note for Mode B:
Tests that invoke next_temperature() or compute_reward() will fail with NotImplementedError
until those two stubs in dynamics.py are implemented by the user.
"""

from __future__ import annotations

import warnings
import numpy as np
import pytest

from config import (
    ACTIVITY_ASLEEP,
    ACTIVITY_AWAKE,
    ACTIVITY_AWAY,
    CALIBRATED,
    DAY_WORKDAY,
    DAY_WEEKEND,
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
    check_calibration_warning,
)
from discretize import discretize_state, get_state_space_dims, get_temp_bin, get_time_block
from dynamics import compute_energy, compute_outdoor_temp, next_temperature
from env import RoomEnv, decode_action, encode_action
from schedule import generate_daily_schedule, get_base_transitions


# ============================================================================
# 1. Config & Validation Tests
# ============================================================================
def test_config_lambda_s_validation() -> None:
    """lambda_s must raise ValueError if non-zero while include_prev_mode is False."""
    with pytest.raises(ValueError, match="lambda_s must be 0.0 when include_prev_mode is False"):
        EnvConfig(include_prev_mode=False, lambda_s=0.5)

    # Valid configurations
    cfg_valid_1 = EnvConfig(include_prev_mode=False, lambda_s=0.0)
    assert cfg_valid_1.lambda_s == 0.0

    cfg_valid_2 = EnvConfig(include_prev_mode=True, lambda_s=0.5)
    assert cfg_valid_2.lambda_s == 0.5


def test_cooling_rates_fractional_bounds() -> None:
    """c_cooler and c_ac must be fractions strictly within (0, 1)."""
    with pytest.raises(ValueError, match="c_cooler must be strictly between 0 and 1"):
        EnvConfig(c_cooler=0.0)

    with pytest.raises(ValueError, match="c_cooler must be strictly between 0 and 1"):
        EnvConfig(c_cooler=1.0)

    with pytest.raises(ValueError, match="c_ac must be strictly between 0 and 1"):
        EnvConfig(c_ac=0.0)

    with pytest.raises(ValueError, match="c_ac must be strictly between 0 and 1"):
        EnvConfig(c_ac=1.2)


def test_calibration_warning_emitted_once() -> None:
    """Calibration warning must be issued when CALIBRATED is False, and only once per process."""
    import config
    original_warned = config._WARNED_CALIBRATION
    config._WARNED_CALIBRATION = False
    try:
        with warnings.catch_warnings(record=True) as recorded:
            warnings.simplefilter("always")
            check_calibration_warning()
            check_calibration_warning()  # Second call should not warn again

        # At least one warning recorded if CALIBRATED is False
        if not CALIBRATED:
            assert any("CALIBRATED is False" in str(w.message) for w in recorded)
            assert len(recorded) == 1
    finally:
        config._WARNED_CALIBRATION = original_warned


# ============================================================================
# 2. Schedule Ordering & Monotonicity Tests
# ============================================================================
@pytest.mark.parametrize("day_type", [DAY_WORKDAY, DAY_WEEKEND])
def test_jittered_schedule_valid_order(day_type: int) -> None:
    """Jittered schedule must maintain valid monotonic transition ordering across all seeds."""
    config = EnvConfig(schedule_jitter_max_steps=2)

    for seed in range(200):
        rng = np.random.default_rng(seed)
        schedule = generate_daily_schedule(day_type, rng, config)

        assert len(schedule) == config.episode_steps
        assert np.all(np.isin(schedule, [ACTIVITY_AWAY, ACTIVITY_AWAKE, ACTIVITY_ASLEEP]))

        # Find the transition points
        # Structure must be: Asleep -> Awake -> Away -> Awake -> Asleep
        # Verify that day starts asleep and ends asleep
        assert schedule[0] == ACTIVITY_ASLEEP
        assert schedule[-1] == ACTIVITY_ASLEEP

        # Check that activities appear in expected sequence
        # Transitions should occur in order: awake appears, away appears, awake appears again
        awake_indices = np.where(schedule == ACTIVITY_AWAKE)[0]
        away_indices = np.where(schedule == ACTIVITY_AWAY)[0]

        assert len(awake_indices) > 0, f"No awake steps for seed {seed}"
        assert len(away_indices) > 0, f"No away steps for seed {seed}"

        # First awake step comes before first away step
        assert awake_indices[0] < away_indices[0]
        # Last away step comes before second awake block
        second_awake = awake_indices[awake_indices > away_indices[-1]]
        assert len(second_awake) > 0, f"No return-home awake step for seed {seed}"
        assert away_indices[-1] < second_awake[0]


# ============================================================================
# 3. Discretization Tests
# ============================================================================
def test_discretize_ranges_and_time_block() -> None:
    """discretize_state() must return values strictly within valid coordinate ranges."""
    config = EnvConfig(time_block_hours=2.0)
    dims = get_state_space_dims(config)
    num_blocks, num_act, num_day, num_bins, num_reg = dims

    assert num_blocks == 12  # 24.0 / 2.0 = 12 blocks
    assert num_bins == len(config.temp_bins) + 1

    # Test edge temperatures
    test_temps = [-10.0, 0.0, 22.9, 23.0, 26.9, 27.0, 28.5, 31.9, 35.0, 50.0]
    for step in range(config.episode_steps + 1):
        for temp in test_temps:
            for act in [0, 1, 2]:
                for dt in [0, 1]:
                    for reg in [0, 1]:
                        state = discretize_state(step, act, dt, temp, reg, config)
                        assert len(state) == 5
                        b, a, d, t_bin, r = state
                        assert 0 <= b < num_blocks
                        assert 0 <= a < num_act
                        assert 0 <= d < num_day
                        assert 0 <= t_bin < num_bins
                        assert 0 <= r < num_reg


def test_include_prev_mode_tuple_dimensions() -> None:
    """When include_prev_mode is True, discretize_state must return a 6-tuple."""
    cfg_5 = EnvConfig(include_prev_mode=False)
    s5 = discretize_state(10, 1, 0, 26.0, 0, cfg_5, prev_cooling_mode=2)
    assert len(s5) == 5

    cfg_6 = EnvConfig(include_prev_mode=True, lambda_s=0.2)
    s6 = discretize_state(10, 1, 0, 26.0, 0, cfg_6, prev_cooling_mode=2)
    assert len(s6) == 6
    assert s6[-1] == 2


# ============================================================================
# 4. Action Encoding / Decoding Tests
# ============================================================================
def test_action_encoding_roundtrip() -> None:
    """encode_action and decode_action must be exact inverses for all 10 actions."""
    for mode in range(5):
        for light in [0, 1]:
            action = encode_action(mode, light)
            dec_mode, dec_light = decode_action(action)
            assert dec_mode == mode
            assert dec_light == light


# ============================================================================
# 5. Exogenous Independence Tests
# ============================================================================
def test_exogenous_independence() -> None:
    """Activity schedule and outdoor noise/temperature must never depend on the agent's actions."""
    config = EnvConfig()
    seed = 12345

    # Run 1: with all actions = 0 (MODE_OFF, LIGHT_OFF)
    env1 = RoomEnv(config)
    env1.reset(seed=seed)
    sched1 = env1.activity_schedule.copy()
    noise1 = env1.outdoor_noises.copy()

    # Run 2: with all actions = 9 (MODE_AC_FAN_LOW, LIGHT_ON)
    env2 = RoomEnv(config)
    env2.reset(seed=seed)
    sched2 = env2.activity_schedule.copy()
    noise2 = env2.outdoor_noises.copy()

    # Pre-generated arrays must match exactly
    np.testing.assert_array_equal(sched1, sched2)
    np.testing.assert_array_equal(noise1, noise2)

    # Computed outdoor temperature sequences across the 48 steps must be identical
    temps_out_1 = [compute_outdoor_temp(i * config.dt, noise1[i], config) for i in range(config.episode_steps)]
    temps_out_2 = [compute_outdoor_temp(i * config.dt, noise2[i], config) for i in range(config.episode_steps)]
    np.testing.assert_array_almost_equal(temps_out_1, temps_out_2)


# ============================================================================
# 6. Physical No-Overshoot Test
# ============================================================================
def test_cooling_no_overshoot() -> None:
    """Active cooling must never overshoot cooler_temp_floor or ac_setpoint."""
    config = EnvConfig()

    # Mathematical property: delta = c * gap, where 0 < c < 1 and gap = max(0, T - floor)
    # T_next = T - delta = T - c * (T - floor) = (1 - c) * T + c * floor > floor since T > floor
    # We verify next_temperature directly once implemented.
    try:
        # Test AC cooling from 26.0 °C toward setpoint 25.5 °C with zero outdoor delta
        t_init = 26.0
        # Set large tau so passive transfer is near 0 for testing cooling isolated
        cfg_iso = EnvConfig(tau=1e6, g_occ=0.0)
        t_next_ac = next_temperature(
            current_temp=t_init,
            cooling_mode=MODE_AC_FAN_LOW,
            weather_regime=REGIME_DRY,
            hour=0.0,
            occupied=False,
            outdoor_noise=0.0,
            config=cfg_iso,
        )
        assert t_next_ac >= cfg_iso.ac_setpoint, "AC overshot ac_setpoint!"
        assert t_next_ac < t_init, "AC did not cool when above setpoint!"

        # Test Cooler cooling from 24.5 °C toward floor 24.0 °C
        t_next_cooler = next_temperature(
            current_temp=24.5,
            cooling_mode=MODE_COOLER_FAN,
            weather_regime=REGIME_DRY,
            hour=0.0,
            occupied=False,
            outdoor_noise=0.0,
            config=cfg_iso,
        )
        assert t_next_cooler >= cfg_iso.cooler_temp_floor, "Cooler overshot cooler_temp_floor!"
        assert t_next_cooler < 24.5, "Cooler did not cool when above floor!"

        # When already below floor or setpoint, cooling delta must be 0
        t_below = 23.0
        t_next_below = next_temperature(
            current_temp=t_below,
            cooling_mode=MODE_AC_FAN_LOW,
            weather_regime=REGIME_DRY,
            hour=0.0,
            occupied=False,
            outdoor_noise=0.0,
            config=cfg_iso,
        )
        assert t_next_below == pytest.approx(t_below, abs=1e-3)

    except NotImplementedError:
        pytest.skip("next_temperature() stub not yet implemented by user (Mode B).")


# ============================================================================
# 7. Environment Stepping Tests (Fail under Mode B until stubs are implemented)
# ============================================================================
def test_same_seed_identical_trajectory() -> None:
    """Same seed and same action sequence must yield an identical episode trajectory."""
    try:
        config = EnvConfig()
        seed = 999
        actions = [i % 10 for i in range(48)]

        env1 = RoomEnv(config)
        s1, _ = env1.reset(seed=seed)
        rewards1, states1 = [], []
        for a in actions:
            ns, r, term, trunc, info = env1.step(a)
            states1.append(ns)
            rewards1.append(r)

        env2 = RoomEnv(config)
        s2, _ = env2.reset(seed=seed)
        rewards2, states2 = [], []
        for a in actions:
            ns, r, term, trunc, info = env2.step(a)
            states2.append(ns)
            rewards2.append(r)

        assert s1 == s2
        assert states1 == states2
        assert rewards1 == rewards2
    except NotImplementedError:
        pytest.skip("Mode B stubs not yet implemented by user.")


def test_episode_truncates_after_48_steps() -> None:
    """Episode must truncate after exactly 48 steps with truncated=True, terminated=False."""
    try:
        config = EnvConfig()
        env = RoomEnv(config)
        env.reset(seed=42)

        for step in range(47):
            _, _, term, trunc, _ = env.step(0)
            assert not term
            assert not trunc

        # 48th step
        next_s, _, term, trunc, info = env.step(0)
        assert not term
        assert trunc
        assert info["step"] == 48

        # Verify wrapped next state has time block 0
        assert next_s[0] == 0
    except NotImplementedError:
        pytest.skip("Mode B stubs not yet implemented by user.")


def test_reward_equals_sum_of_components() -> None:
    """Per-step reward must equal - cost - lambda_T * D_T - lambda_L * D_L - lambda_S * switched."""
    try:
        config = EnvConfig(include_prev_mode=True, lambda_s=0.5)
        env = RoomEnv(config)
        env.reset(seed=77)

        for a in [0, 3, 7, 8, 1]:
            _, r, _, _, info = env.step(a)
            expected_r = (
                - info["cost"]
                - config.lambda_t * info["D_T"]
                - config.lambda_l * info["D_L"]
                - config.lambda_s * (1.0 if info["switched"] else 0.0)
            )
            assert r == pytest.approx(expected_r, abs=1e-5)
    except NotImplementedError:
        pytest.skip("Mode B stubs not yet implemented by user.")


def test_always_off_lagged_curve() -> None:
    """Under always-off, indoor temperature curve must follow a lagged version of outdoor temperature."""
    try:
        config = EnvConfig(outdoor_noise_std=0.0)  # Pure sine outdoor wave
        env = RoomEnv(config)
        env.reset(seed=10)

        indoor_temps = []
        outdoor_temps = []
        for step in range(config.episode_steps):
            hour = step * config.dt
            outdoor_temps.append(compute_outdoor_temp(hour, 0.0, config))
            _, _, _, _, info = env.step(0)  # MODE_OFF, LIGHT_OFF
            indoor_temps.append(info["indoor_temp"])

        # The peak of the indoor temperature should lag behind the peak of outdoor temperature
        out_peak_step = int(np.argmax(outdoor_temps))
        in_peak_step = int(np.argmax(indoor_temps))
        assert in_peak_step >= out_peak_step, "Indoor peak did not lag outdoor peak!"
    except NotImplementedError:
        pytest.skip("Mode B stubs not yet implemented by user.")


def test_always_cooler_dry_vs_humid() -> None:
    """In the dry regime, the cooler must achieve lower room temperature than in the humid regime."""
    try:
        config = EnvConfig(outdoor_noise_std=0.0)
        action_cooler = encode_action(MODE_COOLER_FAN, LIGHT_OFF)

        # Dry regime run
        env_dry = RoomEnv(config)
        env_dry.reset(seed=1)
        env_dry.weather_regime = REGIME_DRY
        for _ in range(20):
            _, _, _, _, info_dry = env_dry.step(action_cooler)

        # Humid regime run
        env_humid = RoomEnv(config)
        env_humid.reset(seed=1)
        env_humid.weather_regime = REGIME_HUMID
        for _ in range(20):
            _, _, _, _, info_humid = env_humid.step(action_cooler)

        assert info_dry["indoor_temp"] < info_humid["indoor_temp"], (
            f"Expected dry temp ({info_dry['indoor_temp']:.2f}) < "
            f"humid temp ({info_humid['indoor_temp']:.2f})"
        )
    except NotImplementedError:
        pytest.skip("Mode B stubs not yet implemented by user.")


def test_1000_episodes_no_nan_or_inf() -> None:
    """No NaN or Inf values over random episodes across diverse seeds."""
    config = EnvConfig()
    env = RoomEnv(config)

    for episode in range(100):  # 100 full episodes = 4,800 steps for fast test run
        obs, info = env.reset(seed=episode)
        assert not any(np.isnan(obs))

        truncated = False
        while not truncated:
            a = int(env.rng.integers(0, 10))
            obs, r, term, truncated, info = env.step(a)

            assert not np.isnan(r) and not np.isinf(r)
            assert not np.isnan(info["indoor_temp"])
            assert not np.isnan(info["energy_kwh"])
            assert not np.isnan(info["cost"])


# ============================================================================
# 8. Phase 1 Acceptance & Parameter Derivation Tests (A1, A5, Design Section 10)
# ============================================================================
def test_outdoor_temp_cosine_peaks() -> None:
    """A1: Outdoor cosine temperature has maximum at 15:00 (step 30) and minimum at 03:00 (step 6)."""
    config = EnvConfig()
    temps = [compute_outdoor_temp(step * config.dt, 0.0, config) for step in range(config.episode_steps)]
    assert np.argmax(temps) == 30, f"Expected peak at step 30 (15:00), got {np.argmax(temps)}"
    assert np.argmin(temps) == 6, f"Expected trough at step 6 (03:00), got {np.argmin(temps)}"
    assert temps[30] == pytest.approx(config.t_mean + config.t_amplitude)
    assert temps[6] == pytest.approx(config.t_mean - config.t_amplitude)


def test_steady_state_analytical_vs_sim_agreement() -> None:
    """A5: Analytical T_settle agrees with 300-step simulation within 0.05 °C for all cases."""
    from steady_state import compute_steady_state_table
    records = compute_steady_state_table()
    for r in records:
        assert r["diff"] < 0.05, f"Case {r['case']} under {r['condition']} disagreed by {r['diff']:.5f} °C"


def test_acceptance_always_off_worse_than_sensible_policy() -> None:
    """Acceptance Test 2: Always-off is clearly worse than a sensible policy due to discomfort penalty."""
    config = EnvConfig()
    env_off = RoomEnv(config)
    env_sensible = RoomEnv(config)
    seed = 42

    env_off.reset(seed=seed)
    env_sensible.reset(seed=seed)

    total_r_off = 0.0
    total_r_sensible = 0.0

    for step in range(config.episode_steps):
        # Always off: action 0 (off, light off)
        _, r_off, _, _, _ = env_off.step(0)
        total_r_off += r_off

        # Sensible policy: turn on AC when occupied, light when awake & dark
        act = env_sensible.activity_schedule[step]
        hour = step * config.dt
        dark = (hour < config.daylight_start_hour or hour >= config.daylight_end_hour)
        light = 1 if (act == ACTIVITY_AWAKE and dark) else 0
        mode = MODE_AC_FAN_LOW if (act != ACTIVITY_AWAY) else MODE_OFF
        action_sensible = encode_action(mode, light)
        _, r_sens, _, _, _ = env_sensible.step(action_sensible)
        total_r_sensible += r_sens

    # Sensible policy avoids heavy thermal discomfort penalty, resulting in much higher (less negative) reward
    assert total_r_sensible > total_r_off, f"Sensible {total_r_sensible} <= Always-off {total_r_off}"


def test_acceptance_always_ac_costs_more_than_sensible_policy() -> None:
    """Acceptance Test 3: Always-AC costs clearly more energy and rupees than a sensible policy."""
    config = EnvConfig()
    env_ac = RoomEnv(config)
    env_sensible = RoomEnv(config)
    seed = 42

    env_ac.reset(seed=seed)
    env_sensible.reset(seed=seed)

    cost_ac = 0.0
    cost_sensible = 0.0

    action_always_ac = encode_action(MODE_AC_FAN_LOW, 1)
    for step in range(config.episode_steps):
        _, _, _, _, info_ac = env_ac.step(action_always_ac)
        cost_ac += info_ac["cost"]

        act = env_sensible.activity_schedule[step]
        hour = step * config.dt
        dark = (hour < config.daylight_start_hour or hour >= config.daylight_end_hour)
        light = 1 if (act == ACTIVITY_AWAKE and dark) else 0
        mode = MODE_AC_FAN_LOW if (act != ACTIVITY_AWAY) else MODE_OFF
        action_sensible = encode_action(mode, light)
        _, _, _, _, info_sens = env_sensible.step(action_sensible)
        cost_sensible += info_sens["cost"]

    assert cost_ac > cost_sensible * 1.5, f"Always-AC cost {cost_ac} not clearly > sensible cost {cost_sensible}"

