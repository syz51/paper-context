from __future__ import annotations

import base64
from collections.abc import Callable
from typing import Any, cast
from unittest.mock import MagicMock
from uuid import uuid4

import pytest
from fastapi import HTTPException
from starlette.requests import Request

from paper_context.api.routes import documents as documents_route_module
from paper_context.ingestion.api import (
    DocumentNotFoundError,
    InvalidCursorError,
    UploadTooLargeError,
)
from paper_context.pagination import CursorError, decode_cursor, encode_cursor, fingerprint_payload

pytestmark = pytest.mark.unit


def test_decode_cursor_round_trips_and_validates_kind() -> None:
    payload = {"kind": "documents:list", "position": 3, "fingerprint": "abc"}
    cursor = encode_cursor(payload)

    assert decode_cursor(cursor, expected_kind="documents:list") == payload
    assert fingerprint_payload(payload) == fingerprint_payload(
        {"fingerprint": "abc", "kind": "documents:list", "position": 3}
    )


@pytest.mark.parametrize(
    ("cursor", "match"),
    [
        (base64.urlsafe_b64encode(b"not-json").decode("ascii").rstrip("="), "invalid cursor"),
        (encode_cursor(cast(Any, ["not", "a", "dict"])), "invalid cursor"),
        (encode_cursor({"kind": "documents:list"}), "cursor kind mismatch"),
    ],
)
def test_decode_cursor_rejects_invalid_payloads(cursor: str, match: str) -> None:
    expected_kind = "documents:search" if match == "cursor kind mismatch" else None

    with pytest.raises(CursorError, match=match):
        decode_cursor(cursor, expected_kind=expected_kind)


def test_trace_headers_extracts_known_headers_and_prefixes() -> None:
    request = Request(
        {
            "type": "http",
            "headers": [
                (b"TraceParent", b"00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-00"),
                (b"X-Request-Id", b"req-123"),
                (b"X-B3-TraceId", b"trace-456"),
                (b"content-type", b"application/pdf"),
            ],
        }
    )

    assert documents_route_module._trace_headers(request) == {
        "traceparent": "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-00",
        "x-request-id": "req-123",
        "x-b3-traceid": "trace-456",
    }


@pytest.mark.parametrize(
    ("error", "status_code", "detail"),
    [
        (DocumentNotFoundError("missing"), 404, "document not found"),
        (UploadTooLargeError("too large"), 413, "too large"),
        (InvalidCursorError("bad cursor"), 400, "bad cursor"),
        (ValueError("bad request"), 400, "bad request"),
    ],
)
def test_translate_document_error_maps_known_exceptions(
    error: Exception, status_code: int, detail: str
) -> None:
    translated = documents_route_module._translate_document_error(error)

    assert isinstance(translated, HTTPException)
    assert translated.status_code == status_code
    assert translated.detail == detail


def test_translate_document_error_reraises_unknown_exception() -> None:
    with pytest.raises(RuntimeError, match="boom"):
        documents_route_module._translate_document_error(RuntimeError("boom"))


@pytest.mark.parametrize(
    ("route_fn", "service_method", "expected_detail"),
    [
        (documents_route_module.get_document, "get_document", "document not found"),
        (documents_route_module.get_document_outline, "get_document_outline", "document not found"),
        (documents_route_module.get_document_tables, "get_document_tables", "document not found"),
    ],
)
def test_document_routes_return_404_when_service_returns_none(
    route_fn: Callable[..., object],
    service_method: str,
    expected_detail: str,
) -> None:
    service = MagicMock()
    getattr(service, service_method).return_value = None

    with pytest.raises(HTTPException, match=expected_detail) as exc_info:
        route_fn(uuid4(), service=service)

    assert exc_info.value.status_code == 404


def test_get_ingest_job_returns_404_when_service_returns_none() -> None:
    service = MagicMock()
    service.get_ingest_job.return_value = None

    with pytest.raises(HTTPException, match="ingest job not found") as exc_info:
        documents_route_module.get_ingest_job(uuid4(), service=service)

    assert exc_info.value.status_code == 404
