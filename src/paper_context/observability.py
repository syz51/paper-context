from __future__ import annotations

import logging
import threading
import time
from collections.abc import Iterator, Mapping
from contextlib import AbstractContextManager, contextmanager
from dataclasses import asdict, dataclass
from datetime import UTC, datetime


@dataclass
class TimingMetric:
    count: int = 0
    total_seconds: float = 0.0
    last_seconds: float | None = None
    max_seconds: float | None = None
    last_recorded_at: datetime | None = None

    def observe(self, duration_seconds: float) -> None:
        self.count += 1
        self.total_seconds += duration_seconds
        self.last_seconds = duration_seconds
        self.last_recorded_at = datetime.now(UTC)
        self.max_seconds = (
            duration_seconds
            if self.max_seconds is None
            else max(self.max_seconds, duration_seconds)
        )


@dataclass
class MetricsSnapshot:
    counters: dict[str, int]
    timings: dict[str, TimingMetric]


@dataclass(frozen=True)
class OperationTimingSnapshot:
    operation: str
    count: int
    total_ms: float
    avg_ms: float
    max_ms: float
    last_ms: float | None
    last_recorded_at: datetime | None


class MetricsRegistry:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counters: dict[str, int] = {}
        self._timings: dict[str, TimingMetric] = {}

    def increment(self, name: str, amount: int = 1) -> None:
        with self._lock:
            self._counters[name] = self._counters.get(name, 0) + amount

    def observe(self, name: str, duration_seconds: float) -> None:
        with self._lock:
            metric = self._timings.setdefault(name, TimingMetric())
            metric.observe(duration_seconds)

    def snapshot(self) -> MetricsSnapshot:
        with self._lock:
            return MetricsSnapshot(
                counters=dict(self._counters),
                timings={
                    name: TimingMetric(**asdict(metric)) for name, metric in self._timings.items()
                },
            )

    def reset(self) -> None:
        with self._lock:
            self._counters.clear()
            self._timings.clear()

    def timing_snapshots(self, *, limit: int = 20) -> list[OperationTimingSnapshot]:
        with self._lock:
            ordered = sorted(
                self._timings.items(),
                key=lambda item: (
                    item[1].last_recorded_at or datetime.fromtimestamp(0, tz=UTC),
                    item[0],
                ),
                reverse=True,
            )
            return [
                OperationTimingSnapshot(
                    operation=name,
                    count=metric.count,
                    total_ms=round(metric.total_seconds * 1000, 3),
                    avg_ms=round((metric.total_seconds / metric.count) * 1000, 3),
                    max_ms=round((metric.max_seconds or 0.0) * 1000, 3),
                    last_ms=None
                    if metric.last_seconds is None
                    else round(metric.last_seconds * 1000, 3),
                    last_recorded_at=metric.last_recorded_at,
                )
                for name, metric in ordered[:limit]
            ]


_METRICS = MetricsRegistry()


def get_metrics() -> MetricsRegistry:
    return _METRICS


def get_metrics_registry() -> MetricsRegistry:
    return _METRICS


def reset_metrics() -> None:
    _METRICS.reset()


def metrics_snapshot() -> dict[str, object]:
    snapshot = _METRICS.snapshot()
    return {
        "counters": snapshot.counters,
        "timings": {
            name: {
                "count": metric.count,
                "total_seconds": round(metric.total_seconds, 6),
                "last_seconds": None
                if metric.last_seconds is None
                else round(metric.last_seconds, 6),
                "max_seconds": None if metric.max_seconds is None else round(metric.max_seconds, 6),
            }
            for name, metric in snapshot.timings.items()
        },
    }


def log_event(
    logger: logging.Logger,
    level: int,
    event: str,
    /,
    **fields: object,
) -> None:
    logger.log(
        level,
        event.replace("_", " "),
        extra={"structured_data": {"event": event, **fields}},
    )


@contextmanager
def track_timing(
    metric_name: str,
    *,
    logger: logging.Logger | None = None,
    event: str | None = None,
    level: int = logging.INFO,
    counter_name: str | None = None,
    fields: Mapping[str, object] | None = None,
) -> Iterator[dict[str, float]]:
    started_at = time.perf_counter()
    if counter_name is not None:
        _METRICS.increment(counter_name)
    holder: dict[str, float] = {}
    try:
        yield holder
    finally:
        duration_seconds = time.perf_counter() - started_at
        _METRICS.observe(metric_name, duration_seconds)
        holder["duration_seconds"] = duration_seconds
        if logger is not None:
            payload: dict[str, object] = {"duration_seconds": round(duration_seconds, 6)}
            if fields:
                payload.update(fields)
            log_event(logger, level, event or metric_name.replace(".", "_"), **payload)


def observe_operation(
    metric_name: str,
    *,
    logger: logging.Logger | None = None,
    event: str | None = None,
    level: int = logging.INFO,
    counter_name: str | None = None,
    fields: Mapping[str, object] | None = None,
) -> AbstractContextManager[dict[str, float]]:
    return track_timing(
        metric_name,
        logger=logger,
        event=event,
        level=level,
        counter_name=counter_name,
        fields=fields,
    )
