"""Single-channel abrupt-change (jump) recovery diagnostic (read-only).

Given one existing channel's samples over a half-open UTC query window
``[start, end)``, the diagnostic walks the channel's samples in time order
and finds abrupt **jumps** — adjacent *valid* samples whose change rate
``|vᵢ − vᵢ₋₁| / (tᵢ − tᵢ₋₁)`` (per second, over the samples' **actual time
difference**) is **strictly greater** than the positive
``jump_rate_threshold``:

* continuity follows the channel's configured **sample validity period
  (TTL)**: a sample holds on ``[tᵢ, min(tᵢ₊₁, tᵢ + ttl, horizon))``; when a
  sample expires before the next one arrives, the invalid gap **cuts
  continuity** — a jump is never detected across a gap, and after a gap the
  next sample starts a fresh chain;
* the **last sample before the window start** may act as the baseline for
  the first in-window jump, but only while it is still valid at ``start``
  (an expired pre-start sample is dropped, never extrapolated);
* when a jump is found, the **pre-jump baseline sample** is recorded and the
  diagnostic looks **forward**, up to ``jump_ts + observation_seconds``
  (clipped to the window end), for the **first** later sample whose value is
  back within ``|v − baseline| ≤ regression_tolerance``:
    - found  → the episode is ``recovered`` and its recovery time is
      ``recovery_sample.ts − jump_ts``;
    - absent → the episode is ``not_recovered`` with one of the reasons
      ``observation_expired`` (the observation window elapsed inside the
      query range), ``range_end`` (the query range ended first) or
      ``ttl_gap`` (an expired gap cut the observation; recovery is never
      judged across the gap).

Samples already attributed to an episode never open another one: post-jump
samples are consumed by the open episode, and scanning resumes only after
the episode ends (at the recovery sample, at the observation deadline, or at
the gap).  Every episode reports its start/end samples, the **maximum
deviation** from the baseline (and the sample reaching it), the recovery
time (when recovered) and the not-recovered reason.

Everything here is a pure function of the supplied samples: nothing is read
from or written to any table, and rule evaluation is never triggered.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

_EPS = 1e-7  # seconds of slack for boundary comparisons (float timestamps)

# Not-recovered reasons (stable machine codes; the UI maps them to labels).
REASON_OBSERVATION_EXPIRED = "observation_expired"
REASON_RANGE_END = "range_end"
REASON_TTL_GAP = "ttl_gap"


@dataclass(frozen=True)
class Sample:
    """One channel sample (``ts`` epoch seconds, numeric ``value``)."""

    ts: float
    value: float

    def to_dict(self) -> dict:
        return {"ts": self.ts, "value": self.value}


@dataclass(frozen=True)
class JumpEpisode:
    """One maximal jump episode from the jump sample up to recovery/end.

    ``baseline`` is the pre-jump sample; ``jump_sample`` is the first sample
    whose change rate exceeds the threshold (the episode start); ``end_sample``
    is the first within-tolerance recovery sample when recovered, otherwise
    the last post-jump sample attributed before the observation deadline /
    range end / TTL gap.
    """

    baseline: Sample
    jump_sample: Sample
    jump_rate: float
    end_sample: Sample
    max_deviation: float
    max_deviation_at: Sample
    observation_deadline: float
    recovery_sample: Sample | None
    unrecovered_reason: str | None

    @property
    def recovered(self) -> bool:
        return self.recovery_sample is not None

    @property
    def status(self) -> str:
        return "recovered" if self.recovered else "not_recovered"

    @property
    def recovery_seconds(self) -> float | None:
        if self.recovery_sample is None:
            return None
        return self.recovery_sample.ts - self.jump_sample.ts

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "recovered": self.recovered,
            "baseline": self.baseline.to_dict(),
            "start_sample": self.jump_sample.to_dict(),
            "end_sample": self.end_sample.to_dict(),
            "jump_at": self.jump_sample.ts,
            "jump_rate": self.jump_rate,
            "max_deviation": self.max_deviation,
            "max_deviation_at": self.max_deviation_at.to_dict(),
            "observation_deadline": self.observation_deadline,
            "recovery_sample": (
                self.recovery_sample.to_dict() if self.recovery_sample is not None else None
            ),
            "recovery_seconds": self.recovery_seconds,
            "unrecovered_reason": self.unrecovered_reason,
        }


@dataclass(frozen=True)
class JumpRecoveryResult:
    start: float
    end: float
    jump_rate_threshold: float
    regression_tolerance: float
    observation_seconds: float
    horizon: float
    ttl_seconds: float | None
    samples: list[Sample] = field(default_factory=list)
    used_pre_window_baseline: bool = False
    episodes: list[JumpEpisode] = field(default_factory=list)

    @property
    def sample_count(self) -> int:
        return len(self.samples)

    @property
    def episode_count(self) -> int:
        return len(self.episodes)

    @property
    def recovered_count(self) -> int:
        return sum(1 for e in self.episodes if e.recovered)

    @property
    def not_recovered_count(self) -> int:
        return sum(1 for e in self.episodes if not e.recovered)

    def to_dict(self) -> dict:
        return {
            "start": self.start,
            "end": self.end,
            "jump_rate_threshold": self.jump_rate_threshold,
            "regression_tolerance": self.regression_tolerance,
            "observation_seconds": self.observation_seconds,
            "horizon": self.horizon,
            "ttl_seconds": self.ttl_seconds,
            "sample_count": self.sample_count,
            "used_pre_window_baseline": self.used_pre_window_baseline,
            "episode_count": self.episode_count,
            "recovered_count": self.recovered_count,
            "not_recovered_count": self.not_recovered_count,
            "episodes": [e.to_dict() for e in self.episodes],
        }


def _ordered_samples(samples) -> list[Sample]:
    """Sort ``(ts, value)`` pairs; equal timestamps keep the last value."""
    ordered: list[Sample] = []
    for ts, value in sorted(((float(t), float(v)) for t, v in samples), key=lambda p: p[0]):
        if ordered and ordered[-1].ts == ts:
            ordered[-1] = Sample(ts, value)  # last write at identical timestamp
        else:
            ordered.append(Sample(ts, value))
    return ordered


def _connected(earlier: Sample, later: Sample, ttl: float | None) -> bool:
    """Whether ``earlier`` stays valid until ``later`` arrives (no TTL gap).

    Without a TTL a sample holds until the next one, so two consecutive
    samples are always connected; intervals touching end-to-start count as
    continuous.  ``horizon`` can never interpose between two real samples.
    """
    if ttl is None:
        return True
    return earlier.ts + ttl >= later.ts - _EPS


def diagnose_jump_recovery(
    samples,
    horizon: float,
    start: float,
    end: float,
    jump_rate_threshold: float,
    regression_tolerance: float,
    observation_seconds: float,
    ttl: float | None = None,
) -> JumpRecoveryResult:
    """Diagnose abrupt jumps and their recovery over ``[start, end)``.

    ``samples`` is an iterable of ``(ts, value)`` pairs covering the window
    plus optionally the last sample before it; ``horizon`` is the global data
    endpoint and ``ttl`` the optional per-channel sample validity period.
    See the module docstring for the jump/episode/recovery rules.

    Raises ``ValueError`` for a non-finite/inverted window, a non-positive
    threshold/observation, a negative/non-finite tolerance, or a
    non-positive TTL.
    """
    start = float(start)
    end = float(end)
    threshold = float(jump_rate_threshold)
    tolerance = float(regression_tolerance)
    observation = float(observation_seconds)
    horizon = float(horizon)
    if not math.isfinite(start) or not math.isfinite(end) or end <= start:
        raise ValueError("diagnostic end must be greater than start (half-open [start, end))")
    if not math.isfinite(threshold) or threshold <= 0:
        raise ValueError("jump_rate_threshold must be a positive finite number per second")
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("regression_tolerance must be a finite non-negative number")
    if not math.isfinite(observation) or observation <= 0:
        raise ValueError("observation_seconds must be a positive finite number of seconds")
    if not math.isfinite(horizon):
        raise ValueError("horizon must be a finite number")
    if ttl is not None:
        ttl = float(ttl)
        if not math.isfinite(ttl) or ttl <= 0:
            raise ValueError("sample validity (ttl) must be a positive number of seconds")

    ordered = _ordered_samples(samples)
    in_range = [s for s in ordered if start <= s.ts < end]

    # The most recent pre-window sample is a valid baseline only while its
    # held interval still reaches the window start (touching counts, an
    # expired sample is dropped rather than extrapolated).
    baseline: Sample | None = None
    used_pre_window = False
    pre = None
    for s in reversed(ordered):
        if s.ts < start:
            pre = s
            break
    if pre is not None and in_range:
        held_end = min(in_range[0].ts, horizon)
        if ttl is not None:
            held_end = min(held_end, pre.ts + ttl)
        if held_end >= start - _EPS:
            baseline = pre
            used_pre_window = True

    episodes: list[JumpEpisode] = []
    i = 0
    n = len(in_range)
    while i < n:
        cur = in_range[i]
        if baseline is None:
            # First chain member: nothing to compute a rate against yet.
            baseline = cur
            i += 1
            continue
        if not _connected(baseline, cur, ttl):
            # Expired gap: continuity is cut, never detect a jump across it;
            # the sample after the gap starts a fresh chain.
            baseline = cur
            i += 1
            continue
        dt = cur.ts - baseline.ts
        rate = abs(cur.value - baseline.value) / dt
        if rate <= threshold:
            baseline = cur
            i += 1
            continue

        # Abrupt jump: open an episode at `cur` and look forward for the
        # first within-tolerance sample, never re-opening on attributed
        # samples.  Recovery is never judged across an expired TTL gap.
        deadline = min(cur.ts + observation, end)
        max_deviation = abs(cur.value - baseline.value)
        max_at = cur
        j = i + 1
        recovery: Sample | None = None
        reason: str | None = None
        while j < n and in_range[j].ts <= deadline + _EPS:
            nxt = in_range[j]
            if not _connected(in_range[j - 1], nxt, ttl):
                reason = REASON_TTL_GAP
                break
            if abs(nxt.value - baseline.value) <= tolerance + _EPS:
                recovery = nxt
                break
            deviation = abs(nxt.value - baseline.value)
            if deviation > max_deviation:
                max_deviation = deviation
                max_at = nxt
            j += 1

        jump_baseline = baseline
        if recovery is not None:
            episodes.append(
                JumpEpisode(
                    baseline=jump_baseline,
                    jump_sample=cur,
                    jump_rate=rate,
                    end_sample=recovery,
                    max_deviation=max_deviation,
                    max_deviation_at=max_at,
                    observation_deadline=deadline,
                    recovery_sample=recovery,
                    unrecovered_reason=None,
                )
            )
            # The recovery sample anchors the next chain.
            baseline = recovery
            i = j + 1
        else:
            end_sample = in_range[j - 1]  # always >= cur (last attributed)
            if reason is None:
                # Which bound stopped the forward search?  When the range end
                # clips the observation window the range is responsible; when
                # the full observation window elapsed, a last attributed
                # sample expiring before the deadline means a TTL gap covered
                # the deadline (no cross-gap recovery either).
                if deadline >= end - _EPS:
                    reason = REASON_RANGE_END
                elif ttl is not None and end_sample.ts + ttl < deadline - _EPS:
                    reason = REASON_TTL_GAP
                else:
                    reason = REASON_OBSERVATION_EXPIRED
            episodes.append(
                JumpEpisode(
                    baseline=jump_baseline,
                    jump_sample=cur,
                    jump_rate=rate,
                    end_sample=end_sample,
                    max_deviation=max_deviation,
                    max_deviation_at=max_at,
                    observation_deadline=deadline,
                    recovery_sample=None,
                    unrecovered_reason=reason,
                )
            )
            # Resume after the attributed span: from the end sample onward
            # (a TTL gap to the next sample is handled by the main loop).
            baseline = end_sample
            i = j

    return JumpRecoveryResult(
        start=start,
        end=end,
        jump_rate_threshold=threshold,
        regression_tolerance=tolerance,
        observation_seconds=observation,
        horizon=horizon,
        ttl_seconds=ttl,
        samples=in_range,
        used_pre_window_baseline=used_pre_window,
        episodes=episodes,
    )
