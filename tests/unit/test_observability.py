from __future__ import annotations

import json
import logging
from datetime import UTC, datetime

import pytest

from paper_context.logging import JsonLogFormatter
from paper_context.observability import MetricsRegistry, observe_operation

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
