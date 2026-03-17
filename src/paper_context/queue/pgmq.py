from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from sqlalchemy import text
from sqlalchemy.engine import Connection


@dataclass(frozen=True)
class PgmqMessage:
    msg_id: int
    read_ct: int
    enqueued_at: datetime
    vt: datetime
    message: dict[str, Any]


@dataclass(frozen=True)
class QueueMetrics:
    queue_name: str
    queue_length: int
    queue_visible_length: int
    newest_msg_age_sec: int | None
    oldest_msg_age_sec: int | None
    total_messages: int
    scrape_time: datetime


class PgmqAdapter:
    def __init__(self, queue_name: str) -> None:
        self.queue_name = queue_name

    def send(self, connection: Connection, payload: Any, delay_seconds: int = 0) -> int:
        stmt = text(
            """
            SELECT pgmq.send(:queue_name, CAST(:payload AS jsonb), :delay_seconds) AS message_id
            """
        )
        result = connection.execute(
            stmt,
            {
                "queue_name": self.queue_name,
                "payload": json.dumps(payload),
                "delay_seconds": delay_seconds,
            },
        )
        return int(result.scalar_one())

    def read_with_poll(
        self,
        connection: Connection,
        vt_seconds: int,
        max_poll_seconds: int,
        poll_interval_ms: int,
        qty: int = 1,
    ) -> list[PgmqMessage]:
        stmt = text(
            """
            SELECT msg_id, read_ct, enqueued_at, vt, message
            FROM pgmq.read_with_poll(
                :queue_name,
                :vt_seconds,
                :qty,
                :max_poll_seconds,
                :poll_interval_ms
            )
            """
        )
        result = connection.execute(
            stmt,
            {
                "queue_name": self.queue_name,
                "vt_seconds": vt_seconds,
                "qty": qty,
                "max_poll_seconds": max_poll_seconds,
                "poll_interval_ms": poll_interval_ms,
            },
        )
        return [self._row_to_message(dict(row)) for row in result.mappings().all()]

    def set_vt(self, connection: Connection, msg_id: int, vt_seconds: int) -> PgmqMessage | None:
        stmt = text(
            """
            SELECT msg_id, read_ct, enqueued_at, vt, message
            FROM pgmq.set_vt(:queue_name, :msg_id, :vt_seconds)
            """
        )
        result = connection.execute(
            stmt,
            {
                "queue_name": self.queue_name,
                "msg_id": msg_id,
                "vt_seconds": vt_seconds,
            },
        )
        row = result.mappings().one_or_none()
        if row is None:
            return None
        return self._row_to_message(dict(row))

    def archive_message(self, connection: Connection, msg_id: int) -> bool:
        stmt = text("SELECT pgmq.archive(:queue_name, :msg_id)")
        result = connection.execute(
            stmt,
            {"queue_name": self.queue_name, "msg_id": msg_id},
        )
        return bool(result.scalar_one())

    def delete_message(self, connection: Connection, msg_id: int) -> bool:
        stmt = text("SELECT pgmq.delete(:queue_name, :msg_id)")
        result = connection.execute(
            stmt,
            {"queue_name": self.queue_name, "msg_id": msg_id},
        )
        return bool(result.scalar_one())

    def metrics(self, connection: Connection) -> QueueMetrics:
        stmt = text(
            """
            SELECT
                queue_name,
                queue_length,
                queue_visible_length,
                newest_msg_age_sec,
                oldest_msg_age_sec,
                total_messages,
                scrape_time
            FROM pgmq.metrics(:queue_name)
            """
        )
        result = connection.execute(stmt, {"queue_name": self.queue_name})
        row = result.mappings().one()
        return QueueMetrics(**row)

    @staticmethod
    def _row_to_message(row: Mapping[str, Any]) -> PgmqMessage:
        return PgmqMessage(
            msg_id=int(row["msg_id"]),
            read_ct=int(row["read_ct"]),
            enqueued_at=row["enqueued_at"],
            vt=row["vt"],
            message=row["message"]
            if isinstance(row["message"], dict)
            else json.loads(row["message"]),
        )
