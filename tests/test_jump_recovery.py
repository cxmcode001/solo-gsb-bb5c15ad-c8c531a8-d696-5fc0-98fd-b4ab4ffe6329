"""Unit tests for the read-only single-channel jump recovery diagnostic.

Change rates use the samples' actual time difference; an expired TTL gap
cuts continuity (no jump or recovery is judged across it); the last sample
before the window may serve as a pre-jump baseline while still valid; an
episode is the span from the jump sample to the first within-tolerance
recovery sample inside the observation window, otherwise it stays not
recovered with a reason.
"""

import pytest

from app.jump_recovery import (
    REASON_OBSERVATION_EXPIRED,
    REASON_RANGE_END,
    REASON_TTL_GAP,
    diagnose_jump_recovery,
)

H = 1000.0  # global data endpoint (horizon)


def series(values, start=0.0, step=60.0):
    return [(start + i * step, v) for i, v in enumerate(values)]


# --------------------------------------------------------------------------- #
# jump detection
# --------------------------------------------------------------------------- #


def test_change_rate_uses_actual_time_difference():
    # 0 -> 10s: +1 over 10s = 0.1/s jump; the later +9 over 110s is only
    # 0.082/s and stays below the threshold (irregular sampling).
    samples = [(0, 10), (10, 11), (120, 20), (180, 10)]
    result = diagnose_jump_recovery(samples, H, 0, 200, 0.09, 1.0, 200)
    episodes = result.episodes
    assert len(episodes) == 1
    ep = episodes[0]
    assert (ep.baseline.ts, ep.baseline.value) == (0, 10)
    assert (ep.jump_sample.ts, ep.jump_sample.value) == (10, 11)
    assert ep.jump_rate == pytest.approx(0.1)


def test_threshold_is_strictly_greater():
    samples = series([10, 16, 16])  # exactly 0.1/s at 60s spacing
    assert diagnose_jump_recovery(samples, H, 0, 180, 0.1, 1.0, 120).episode_count == 0
    result = diagnose_jump_recovery(samples, H, 0, 180, 0.099, 1.0, 120)
    assert result.episode_count == 1


def test_negative_direction_is_also_a_jump():
    result = diagnose_jump_recovery(series([20, 10, 20]), H, 0, 180, 0.1, 1.0, 120)
    ep = result.episodes[0]
    assert (ep.jump_sample.ts, ep.jump_sample.value) == (60, 10)
    assert ep.max_deviation == pytest.approx(10)
    assert ep.recovered is True
    assert ep.recovery_sample.ts == 120


def test_equal_timestamps_last_value_wins():
    result = diagnose_jump_recovery(
        [(0, 10), (60, 10), (60, 20), (120, 10)], H, 0, 180, 0.1, 1.0, 120
    )
    assert result.sample_count == 3
    ep = result.episodes[0]
    assert (ep.baseline.ts, ep.baseline.value) == (0, 10)


def test_no_jump_when_values_flat():
    result = diagnose_jump_recovery(series([5, 5, 5, 5]), H, 0, 240, 0.01, 0.0, 60)
    assert result.episodes == []


# --------------------------------------------------------------------------- #
# TTL gaps cut continuity
# --------------------------------------------------------------------------- #


def test_ttl_gap_blocks_jump_across_it():
    # sample at 0 expires 30s later; the next sample at 120 starts a new
    # chain even though |20-10|/120 alone looks like a jump at 0.05/s... it
    # is simply not compared across the gap.
    samples = [(0, 10), (120, 20), (180, 20)]
    result = diagnose_jump_recovery(samples, H, 0, 240, 0.01, 1.0, 120, ttl=30)
    assert result.episodes == []


def test_touching_held_intervals_stay_continuous():
    # TTL exactly equal to the 60s spacing: intervals [t, t+60) touch and
    # count as continuous, so the jump is detected.
    result = diagnose_jump_recovery(
        series([10, 10, 20, 10]), H, 0, 240, 0.1, 1.0, 120, ttl=60
    )
    assert result.episodes[0].recovered is True


def test_ttl_gap_during_observation_is_not_recovered():
    # dense connected run-up, jump at 120; then the value would "recover" at
    # 240 but an expired gap (120+60=180 < 240) lies between: never recovered
    # across the gap -> ttl_gap.
    samples = [(0, 10), (60, 10), (120, 20), (240, 10)]
    result = diagnose_jump_recovery(samples, H, 0, 300, 0.1, 1.0, 300, ttl=60)
    ep = result.episodes[0]
    assert ep.recovered is False
    assert ep.unrecovered_reason == REASON_TTL_GAP
    assert (ep.end_sample.ts, ep.end_sample.value) == (120, 20)


def test_ttl_gap_covering_deadline_is_ttl_gap():
    # jump at 60 with samples spaced 20s apart (connected under TTL 30);
    # the last attributed sample at 90 expires at 120, well before the
    # observation deadline at 180, and the query range (600) does not clip it.
    samples = [(0, 10), (40, 10), (60, 20), (80, 20)]
    result = diagnose_jump_recovery(samples, H, 0, 600, 0.4, 1.0, 120, ttl=30)
    ep = result.episodes[0]
    assert ep.unrecovered_reason == REASON_TTL_GAP
    assert (ep.end_sample.ts, ep.end_sample.value) == (80, 20)


# --------------------------------------------------------------------------- #
# pre-window baseline
# --------------------------------------------------------------------------- #


def test_pre_window_sample_serves_as_baseline():
    samples = [(-60, 10), (60, 20), (120, 10)]
    result = diagnose_jump_recovery(samples, 200, 0, 180, 0.05, 1.0, 300)
    assert result.used_pre_window_baseline is True
    ep = result.episodes[0]
    assert (ep.baseline.ts, ep.baseline.value) == (-60, 10)
    assert (ep.jump_sample.ts, ep.jump_sample.value) == (60, 20)
    assert ep.recovery_sample.ts == 120


def test_expired_pre_window_sample_is_dropped():
    samples = [(-60, 10), (60, 20), (90, 10)]
    result = diagnose_jump_recovery(samples, 200, 0, 180, 0.3, 1.0, 300, ttl=30)
    assert result.used_pre_window_baseline is False
    # first in-window sample starts the chain; jump at 90 (20 -> 10 over 30s)
    ep = result.episodes[0]
    assert (ep.baseline.ts, ep.baseline.value) == (60, 20)
    assert (ep.jump_sample.ts, ep.jump_sample.value) == (90, 10)


# --------------------------------------------------------------------------- #
# recovery / episodes
# --------------------------------------------------------------------------- #


def test_recovery_is_first_sample_back_within_tolerance():
    samples = [(0, 10), (60, 30), (120, 40), (180, 35), (240, 11)]
    result = diagnose_jump_recovery(samples, 300, 0, 300, 0.1, 1.0, 300)
    ep = result.episodes[0]
    assert ep.recovered is True
    assert (ep.recovery_sample.ts, ep.recovery_sample.value) == (240, 11)
    assert ep.recovery_seconds == pytest.approx(180)
    # max deviation over post-jump attributed samples (recovery excluded)
    assert ep.max_deviation == pytest.approx(30)
    assert (ep.max_deviation_at.ts, ep.max_deviation_at.value) == (120, 40)
    assert (ep.end_sample.ts, ep.end_sample.value) == (240, 11)


def test_tolerance_boundary_is_inclusive():
    # |11 - 10| == tolerance 1 counts as recovered
    result = diagnose_jump_recovery(
        [(0, 10), (60, 20), (120, 11)], H, 0, 180, 0.1, 1.0, 120
    )
    ep = result.episodes[0]
    assert ep.recovered is True and ep.recovery_sample.ts == 120


def test_recovery_past_observation_window_is_observation_expired():
    # jump at 120, observation 60s -> deadline 180; recovery only at 300
    samples = series([10, 10, 20, 20, 20, 10, 10])
    result = diagnose_jump_recovery(samples, 420, 0, 420, 0.1, 1.0, 60)
    ep = result.episodes[0]
    assert ep.recovered is False
    assert ep.unrecovered_reason == REASON_OBSERVATION_EXPIRED
    assert ep.observation_deadline == pytest.approx(180)
    # sample exactly on the deadline is still within the observation window
    assert (ep.end_sample.ts, ep.end_sample.value) == (180, 20)


def test_range_end_clipping_is_range_end_reason():
    samples = series([10, 10, 20, 20])
    result = diagnose_jump_recovery(samples, H, 0, 150, 0.1, 1.0, 300)
    ep = result.episodes[0]
    assert ep.recovered is False
    assert ep.unrecovered_reason == REASON_RANGE_END
    assert (ep.end_sample.ts, ep.end_sample.value) == (120, 20)


def test_attributed_samples_do_not_reopen_an_episode():
    # jump at 60 (10->20), 15 within tolerance 5 at 120 => recovered;
    # next jump starts from the recovery sample (120/15): 180/10 is 0.083/s,
    # then 240/20 is another jump, recovered at 300 (|10-20| vs baseline 15).
    samples = series([10, 20, 15, 10, 20, 10])
    result = diagnose_jump_recovery(samples, 300, 0, 300, 0.05, 5.0, 300)
    episodes = result.episodes
    assert [(e.jump_sample.ts, e.recovery_sample.ts) for e in episodes] == [
        (60, 120),
        (180, 240),
    ]
    # post-jump samples 120 (recovery) anchors, and samples are never double-counted
    all_attributed = set()
    for e in episodes:
        assert e.jump_sample.ts not in all_attributed
        all_attributed.add(e.jump_sample.ts)


def test_recovery_resumes_scan_without_second_episode_inside():
    # after recovery at 180 the value stays flat, so no more episodes
    samples = series([10, 20, 20, 10, 10])
    result = diagnose_jump_recovery(samples, 300, 0, 300, 0.1, 1.0, 300)
    assert len(result.episodes) == 1
    assert result.episodes[0].recovery_sample.ts == 180


# --------------------------------------------------------------------------- #
# window / payload / argument validation
# --------------------------------------------------------------------------- #


def test_half_open_end_excludes_sample_on_end():
    result = diagnose_jump_recovery(
        [(0, 10), (60, 20)], 60, 0, 60, 0.1, 1.0, 60
    )
    assert result.sample_count == 1
    assert result.episodes == []


def test_empty_and_single_sample_inputs():
    assert diagnose_jump_recovery([], H, 0, 60, 0.1, 1.0, 60).episodes == []
    assert diagnose_jump_recovery([(30, 5)], H, 0, 60, 0.1, 1.0, 60).episodes == []


def test_payload_shape():
    result = diagnose_jump_recovery(
        series([10, 20, 10]), H, 0, 180, 0.1, 1.0, 120
    )
    payload = result.to_dict()
    assert payload["start"] == 0 and payload["end"] == 180
    assert payload["jump_rate_threshold"] == 0.1
    assert payload["regression_tolerance"] == 1.0
    assert payload["observation_seconds"] == 120
    assert payload["horizon"] == H
    assert payload["ttl_seconds"] is None
    assert payload["sample_count"] == 3
    assert payload["episode_count"] == 1
    assert payload["recovered_count"] == 1 and payload["not_recovered_count"] == 0
    ep = payload["episodes"][0]
    assert ep["status"] == "recovered" and ep["recovered"] is True
    assert ep["baseline"] == {"ts": 0, "value": 10}
    assert ep["start_sample"] == {"ts": 60, "value": 20}
    assert ep["end_sample"] == {"ts": 120, "value": 10}
    assert ep["jump_at"] == 60
    assert ep["recovery_seconds"] == pytest.approx(60)
    assert ep["unrecovered_reason"] is None


@pytest.mark.parametrize(
    "kwargs",
    [
        {"start": 100, "end": 100},
        {"start": 200, "end": 100},
        {"jump_rate_threshold": 0},
        {"jump_rate_threshold": -1.0},
        {"jump_rate_threshold": float("nan")},
        {"jump_rate_threshold": float("inf")},
        {"regression_tolerance": -0.1},
        {"regression_tolerance": float("nan")},
        {"observation_seconds": 0},
        {"observation_seconds": -5.0},
        {"ttl": 0},
        {"ttl": -10.0},
    ],
)
def test_invalid_arguments_rejected(kwargs):
    base = dict(
        samples=[(0, 1), (60, 2)],
        horizon=H,
        start=0,
        end=120,
        jump_rate_threshold=0.01,
        regression_tolerance=1.0,
        observation_seconds=60,
    )
    base.update(kwargs)
    with pytest.raises(ValueError):
        diagnose_jump_recovery(**base)
