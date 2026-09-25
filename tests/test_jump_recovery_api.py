"""End-to-end tests for the read-only single-channel jump recovery API.

The diagnostic reads the current events and the channel's sample validity
period on the fly: it never inserts/updates/deletes events, never writes
fault results and never triggers rule re-evaluation.
"""

import json
import os
import tempfile

import pytest

from app import db, services
from app.engine import to_epoch

BASE = to_epoch("2026-09-23T00:00:00Z")
P = 60.0


@pytest.fixture()
def scratch_db():
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    db.DB_PATH = tmp.name
    db._conn = None
    db.init_db()
    yield tmp.name
    for suffix in ("", "-wal", "-shm"):
        try:
            os.unlink(tmp.name + suffix)
        except OSError:
            pass


@pytest.fixture()
def client(scratch_db):
    from fastapi.testclient import TestClient
    from app.main import app

    return TestClient(app)


def _load(rows, device="pump-1", metric="temp"):
    payload = [{"device": device, "metric": metric, "ts": ts, "value": v} for ts, v in rows]
    services.import_events("\n".join(json.dumps(r) for r in payload), reevaluate=False)


def _series(values, step=P):
    return [(BASE + i * step, v) for i, v in enumerate(values)]


def _params(**over):
    params = {
        "device": "pump-1", "metric": "temp",
        "start": "2026-09-23T00:00:00Z", "end": "2026-09-23T00:06:00Z",
        "jump_rate_threshold": 0.1,
        "regression_tolerance": 1,
        "observation_seconds": 300,
    }
    params.update(over)
    return params


# --------------------------------------------------------------------------- #
# service
# --------------------------------------------------------------------------- #


def test_service_recovered_episode(scratch_db):
    _load(_series([10, 10, 20, 20, 10, 10]))
    result = services.query_jump_recovery(
        "pump-1", "temp", BASE, BASE + 6 * P, 0.1, 1.0, 300
    )
    assert result["ttl_seconds"] is None
    assert result["sample_count"] == 6
    assert result["used_pre_window_baseline"] is False
    assert result["episode_count"] == 1
    ep = result["episodes"][0]
    assert ep["status"] == "recovered"
    assert ep["baseline"] == {"ts": BASE + P, "value": 10}
    assert ep["start_sample"] == {"ts": BASE + 2 * P, "value": 20}
    assert ep["jump_rate"] == pytest.approx(10 / P)
    assert ep["max_deviation"] == pytest.approx(10)
    assert ep["end_sample"] == {"ts": BASE + 4 * P, "value": 10}
    assert ep["recovery_sample"]["ts"] == BASE + 4 * P
    assert ep["recovery_seconds"] == pytest.approx(2 * P)
    assert ep["unrecovered_reason"] is None
    assert result["channel"]["event_count"] == 6


def test_service_observation_expired(scratch_db):
    # jump at 00:02, observation 60s ends at 00:03 (the 00:03 sample is still
    # offset and inside the window), recovery only at 00:05
    _load(_series([10, 10, 20, 20, 20, 10]))
    result = services.query_jump_recovery(
        "pump-1", "temp", BASE, BASE + 6 * P, 0.1, 1.0, P
    )
    ep = result["episodes"][0]
    assert ep["status"] == "not_recovered"
    assert ep["unrecovered_reason"] == "observation_expired"
    assert ep["recovery_seconds"] is None
    assert ep["end_sample"]["ts"] == BASE + 3 * P
    assert ep["observation_deadline"] == pytest.approx(BASE + 3 * P)


def test_service_range_end_reason(scratch_db):
    _load(_series([10, 10, 20]))
    result = services.query_jump_recovery(
        "pump-1", "temp", BASE, BASE + 2.5 * P, 0.1, 1.0, 600
    )
    ep = result["episodes"][0]
    assert ep["status"] == "not_recovered"
    assert ep["unrecovered_reason"] == "range_end"
    assert ep["end_sample"]["ts"] == BASE + 2 * P


def test_service_ttl_cuts_continuity(scratch_db):
    # 60s spacing, 30s TTL: every adjacency has an expired gap -> no jumps
    _load(_series([10, 20, 10, 20]))
    db.upsert_channel_ttl("pump-1", "temp", 30.0)
    result = services.query_jump_recovery(
        "pump-1", "temp", BASE, BASE + 4 * P, 0.01, 1.0, 300
    )
    assert result["ttl_seconds"] == 30.0
    assert result["episodes"] == []


def test_service_ttl_gap_during_observation(scratch_db):
    # TTL 60 with 60s spacing: intervals touch end-to-start, so the chain is
    # continuous and the jump recovers normally.
    _load(_series([10, 10, 20, 10]))
    db.upsert_channel_ttl("pump-1", "temp", 60.0)
    result = services.query_jump_recovery(
        "pump-1", "temp", BASE, BASE + 4 * P, 0.1, 1.0, 180
    )
    assert result["episodes"][0]["status"] == "recovered"

    # shortening the TTL below the spacing makes every adjacency an expired
    # gap, including the one spanning the jump: no episode opens at all.
    db.upsert_channel_ttl("pump-1", "temp", 30.0)
    result = services.query_jump_recovery(
        "pump-1", "temp", BASE, BASE + 4 * P, 0.05, 1.0, 300
    )
    assert result["episodes"] == []


def test_service_ttl_gap_after_jump_is_not_recovered(scratch_db):
    # dense connected run-up (20s apart, TTL 30): jump at 00:00:40,
    # then the next sample only arrives at 00:01:40 — the jump sample
    # expired at +30s, so an invalid gap blocks recovery (never judged
    # across it) and the episode is ttl_gap.
    _load([
        (BASE, 10), (BASE + 20, 10), (BASE + 40, 20), (BASE + 100, 10)
    ])
    db.upsert_channel_ttl("pump-1", "temp", 30.0)
    result = services.query_jump_recovery(
        "pump-1", "temp", BASE, BASE + 200, 0.4, 1.0, 300
    )
    ep = result["episodes"][0]
    assert ep["status"] == "not_recovered"
    assert ep["unrecovered_reason"] == "ttl_gap"
    assert ep["start_sample"] == {"ts": BASE + 40, "value": 20}
    assert ep["end_sample"]["ts"] == BASE + 40


def test_service_pre_window_baseline(scratch_db):
    _load([(BASE - 60, 10), (BASE + 60, 20), (BASE + 120, 10)])
    result = services.query_jump_recovery(
        "pump-1", "temp", BASE, BASE + 180, 0.05, 1.0, 300
    )
    assert result["used_pre_window_baseline"] is True
    ep = result["episodes"][0]
    assert ep["baseline"] == {"ts": BASE - 60, "value": 10}
    assert ep["start_sample"]["ts"] == BASE + 60
    assert ep["recovery_sample"]["ts"] == BASE + 120


def test_service_unknown_channel_and_bad_args(scratch_db):
    _load(_series([1, 2]))
    with pytest.raises(KeyError):
        services.query_jump_recovery("ghost", "temp", BASE, BASE + P, 0.1, 1.0, 60)
    with pytest.raises(ValueError):
        services.query_jump_recovery("pump-1", "temp", BASE + P, BASE, 0.1, 1.0, 60)
    with pytest.raises(ValueError):
        services.query_jump_recovery("pump-1", "temp", BASE, BASE + P, 0, 1.0, 60)
    with pytest.raises(ValueError):
        services.query_jump_recovery("pump-1", "temp", BASE, BASE + P, 0.1, -1, 60)
    with pytest.raises(ValueError):
        services.query_jump_recovery("pump-1", "temp", BASE, BASE + P, 0.1, 1.0, 0)


def test_service_is_read_only(scratch_db):
    _load(_series([10, 20, 10]))
    db.execute(
        "INSERT INTO rules (name, logic, duration, created_at) VALUES (?,?,?,?)",
        ("r", "AND", 0, 0),
    )
    db.execute(
        "INSERT INTO fault_intervals (rule_id, start_ts, end_ts, duration_required,"
        " confirmed_at, evidence_json) VALUES (?,?,?,?,?,?)",
        (1, BASE, BASE + P, 0, BASE, "[]"),
    )
    before_events = db.query_one("SELECT COUNT(*) AS c FROM events")["c"]
    before_faults = db.query_one("SELECT COUNT(*) AS c FROM fault_intervals")["c"]
    services.query_jump_recovery("pump-1", "temp", BASE, BASE + 3 * P, 0.1, 1.0, 300)
    assert db.query_one("SELECT COUNT(*) AS c FROM events")["c"] == before_events
    assert db.query_one("SELECT COUNT(*) AS c FROM fault_intervals")["c"] == before_faults
    row = db.query_one("SELECT start_ts, end_ts FROM fault_intervals WHERE rule_id = 1")
    assert (row["start_ts"], row["end_ts"]) == (BASE, BASE + P)


# --------------------------------------------------------------------------- #
# HTTP
# --------------------------------------------------------------------------- #


def test_api_diagnostic_ok(scratch_db, client):
    _load(_series([10, 10, 20, 20, 10, 10]))
    resp = client.get("/api/jump-recovery", params=_params())
    assert resp.status_code == 200
    body = resp.json()
    assert body["device"] == "pump-1" and body["metric"] == "temp"
    assert body["jump_rate_threshold"] == 0.1
    assert body["regression_tolerance"] == 1
    assert body["observation_seconds"] == 300
    assert body["channel"] == {
        "device": "pump-1", "metric": "temp", "ttl_seconds": None,
        "event_count": 6, "min_ts": BASE, "max_ts": BASE + 5 * P,
    }
    assert body["episode_count"] == 1
    assert body["recovered_count"] == 1 and body["not_recovered_count"] == 0
    ep = body["episodes"][0]
    assert ep["status"] == "recovered"
    assert ep["start_sample"] == {"ts": BASE + 2 * P, "value": 20}
    assert ep["recovery_seconds"] == pytest.approx(120)


def test_api_epoch_range_and_half_open_end(scratch_db, client):
    # the jump sample lands exactly on the end bound: outside the window
    _load(_series([10, 20, 20]))
    resp = client.get("/api/jump-recovery", params=_params(
        start=BASE, end=BASE + P, observation_seconds=60,
    ))
    assert resp.status_code == 200
    assert resp.json()["episodes"] == []


def test_api_unknown_channel_404(scratch_db, client):
    _load(_series([1]))
    resp = client.get("/api/jump-recovery", params=_params(device="ghost"))
    assert resp.status_code == 404


def test_api_inverted_window_400(scratch_db, client):
    _load(_series([1]))
    resp = client.get("/api/jump-recovery", params=_params(
        start="2026-09-23T00:06:00Z", end="2026-09-23T00:00:00Z",
    ))
    assert resp.status_code == 400


def test_api_bad_thresholds_rejected(scratch_db, client):
    _load(_series([1]))
    resp = client.get("/api/jump-recovery", params=_params(jump_rate_threshold=0))
    assert resp.status_code in (400, 422)
    resp = client.get("/api/jump-recovery", params=_params(jump_rate_threshold=-1))
    assert resp.status_code == 422  # Query(..., gt=0)
    resp = client.get("/api/jump-recovery", params=_params(regression_tolerance=-1))
    assert resp.status_code == 422  # Query(..., ge=0)
    resp = client.get("/api/jump-recovery", params=_params(observation_seconds=0))
    assert resp.status_code == 422  # Query(..., gt=0)
    resp = client.get("/api/jump-recovery", params=_params(jump_rate_threshold="inf"))
    assert resp.status_code == 400  # non-finite rejected explicitly
    resp = client.get("/api/jump-recovery", params=_params(observation_seconds="nan"))
    assert resp.status_code == 422  # NaN fails the gt=0 query validation


def test_api_missing_params_422(client):
    assert client.get("/api/jump-recovery").status_code == 422
    resp = client.get("/api/jump-recovery", params={
        "device": "pump-1", "metric": "temp",
        "start": "2026-09-23T00:00:00Z", "end": "2026-09-23T00:06:00Z",
        "jump_rate_threshold": 0.1, "regression_tolerance": 1,
    })
    assert resp.status_code == 422  # observation_seconds missing


def test_api_does_not_touch_events_or_faults(scratch_db, client):
    _load(_series([10, 20, 10]))
    db.execute(
        "INSERT INTO rules (name, logic, duration, created_at) VALUES (?,?,?,?)",
        ("r", "AND", 0, 0),
    )
    db.execute(
        "INSERT INTO fault_intervals (rule_id, start_ts, end_ts, duration_required,"
        " confirmed_at, evidence_json) VALUES (?,?,?,?,?,?)",
        (1, BASE, BASE + P, 0, BASE, "[]"),
    )
    before_events = db.query_one("SELECT COUNT(*) AS c FROM events")["c"]
    before_faults = db.query_one("SELECT COUNT(*) AS c FROM fault_intervals")["c"]
    resp = client.get("/api/jump-recovery", params=_params(end="2026-09-23T00:03:00Z"))
    assert resp.status_code == 200
    assert db.query_one("SELECT COUNT(*) AS c FROM events")["c"] == before_events
    assert db.query_one("SELECT COUNT(*) AS c FROM fault_intervals")["c"] == before_faults
