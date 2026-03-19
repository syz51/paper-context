from __future__ import annotations

import json
import logging
import sys
from collections.abc import Mapping
from datetime import UTC, date, datetime
from pathlib import Path
from typing import cast
from uuid import UUID

import pytest

from paper_context.logging import JsonLogFormatter, _json_default
from paper_context.observability import (
    MetricsRegistry,
    TimingMetric,
    get_metrics,
    get_metrics_registry,
    metrics_snapshot,
    observe_operation,
    reset_metrics,
    track_timing,
)

pytestmark = pytest.mark.unit


def test_metrics_registry_records_and_orders_timing_snapshots() -> None:
    registry = MetricsRegistry()
    registry.observe("ingest.parse", 0.0155)
    registry.observe("ingest.parse", 0.0200)
    registry.observe("retrieval.rerank", 0.0070)

    snapshots = registry.timing_snapshots()
    snapshot_map = {snapshot.operation: snapshot for snapshot in snapshots}

    assert snapshot_map["ingest.parse"].count == 2
    assert snapshot_map["ingest.parse"].total_ms == 35.5
    assert snapshot_map["ingest.parse"].avg_ms == 17.75
    assert snapshot_map["ingest.parse"].max_ms == 20.0
    assert snapshot_map["retrieval.rerank"].count == 1


def test_observe_operation_records_completion_metric(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO)
    logger = logging.getLogger("paper_context.tests.observability")

    with observe_operation("queue.lease_renewal", logger=logger, fields={"message_id": 123}):
        pass

    assert any(
        getattr(record, "structured_data", {}).get("event") == "queue_lease_renewal"
        for record in caplog.records
    )


def test_json_log_formatter_outputs_structured_payload() -> None:
    formatter = JsonLogFormatter()
    record = logging.LogRecord(
        name="paper_context.tests",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="worker started",
        args=(),
        exc_info=None,
    )
    record.structured_data = {
        "event": "worker.started",
        "timestamp_source": datetime(2026, 3, 19, 8, 0, tzinfo=UTC),
    }

    payload = json.loads(formatter.format(record))

    assert payload["level"] == "INFO"
    assert payload["logger"] == "paper_context.tests"
    assert payload["message"] == "worker started"
    assert payload["event"] == "worker.started"
    assert payload["timestamp_source"] == "2026-03-19T08:00:00+00:00"


def test_observability_wrappers_snapshot_and_reset_cover_remaining_paths() -> None:
    reset_metrics()
    registry = get_metrics_registry()

    assert get_metrics() is registry

    registry.increment("queue.lease_renewal", 2)
    registry.observe("ingest.parse", 0.125)
    snapshot = metrics_snapshot()
    assert snapshot["counters"] == {"queue.lease_renewal": 2}
    timings = cast(Mapping[str, Mapping[str, object]], snapshot["timings"])
    assert timings["ingest.parse"]["count"] == 1
    assert timings["ingest.parse"]["max_seconds"] == 0.125

    reset_metrics()
    assert metrics_snapshot() == {"counters": {}, "timings": {}}


def test_metrics_registry_timing_snapshots_handle_missing_last_timestamp() -> None:
    registry = MetricsRegistry()
    registry._timings["orphaned"] = TimingMetric(  # type: ignore[attr-defined]
        count=1,
        total_seconds=1.5,
        last_seconds=None,
        max_seconds=None,
        last_recorded_at=None,
    )

    snapshots = registry.timing_snapshots(limit=1)

    assert len(snapshots) == 1
    assert snapshots[0].operation == "orphaned"
    assert snapshots[0].last_ms is None
    assert snapshots[0].max_ms == 0.0


def test_track_timing_logs_duration_and_counter_without_extra_fields(
    caplog: pytest.LogCaptureFixture,
) -> None:
    reset_metrics()
    caplog.set_level(logging.INFO)
    logger = logging.getLogger("paper_context.tests.track_timing")

    with track_timing(
        "retrieval.embedding",
        logger=logger,
        event="custom_event",
        counter_name="retrieval.calls",
    ) as holder:
        pass

    assert holder["duration_seconds"] >= 0
    snapshot = metrics_snapshot()
    assert snapshot["counters"] == {"retrieval.calls": 1}
    assert any(
        getattr(record, "structured_data", {}).get("event") == "custom_event"
        for record in caplog.records
    )


def test_json_log_formatter_handles_exceptions_and_json_default_types() -> None:
    formatter = JsonLogFormatter()
    try:
        raise ValueError("boom")
    except ValueError:
        exc_info = sys.exc_info()

    record = logging.LogRecord(
        name="paper_context.tests",
        level=logging.ERROR,
        pathname=__file__,
        lineno=1,
        msg="worker failed",
        args=(),
        exc_info=exc_info,
    )
    record.structured_data = "ignore-me"

    payload = json.loads(formatter.format(record))

    assert payload["message"] == "worker failed"
    assert "exception" in payload
    assert "event" not in payload
    assert _json_default(datetime(2026, 3, 19, 8, 0, tzinfo=UTC)) == "2026-03-19T08:00:00+00:00"
    assert _json_default(date(2026, 3, 19)) == "2026-03-19"
    assert _json_default(Path("/tmp/example")) == "/tmp/example"
    assert _json_default(UUID("11111111-1111-1111-1111-111111111111")) == (
        "11111111-1111-1111-1111-111111111111"
    )
    fallback_repr = cast(str, _json_default(object()))
    assert fallback_repr.startswith("<object object at ")
