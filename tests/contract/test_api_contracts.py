from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from paper_context.api import app as api_app_module
from paper_context.api.app import create_app
from paper_context.api.routes import health as health_module
from paper_context.schemas import api as api_schemas
from paper_context.schemas import mcp as mcp_schemas
from paper_context.schemas.common import QueueMetricsResponse

pytestmark = pytest.mark.contract

GOLDEN_DIR = Path(__file__).with_name("golden")


class _FakeMcpApp:
    async def __call__(self, scope, receive, send) -> None:  # pragma: no cover - test helper only
        await send(
            {
                "type": "http.response.start",
                "status": 200,
                "headers": [(b"content-type", b"text/plain; charset=utf-8")],
            }
        )
        await send({"type": "http.response.body", "body": b"ok"})

    def lifespan(self, app):  # pragma: no cover - test helper only
        from contextlib import asynccontextmanager

        @asynccontextmanager
        async def _lifespan(_app):
            yield

        return _lifespan(app)


def _golden_payload(name: str) -> dict[str, object]:
    return json.loads((GOLDEN_DIR / name).read_text())


def _patch_contract_app(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, *, database_ready: bool
) -> None:
    storage_root = tmp_path / "artifacts"
    storage_root.mkdir(parents=True, exist_ok=True)
    settings = SimpleNamespace(
        log_level="INFO",
        storage=SimpleNamespace(root_path=storage_root),
        queue=SimpleNamespace(name="document_ingest"),
        providers=SimpleNamespace(
            voyage_model="voyage-4-large",
            reranker_model="zerank-2",
            index_version="mvp-v1",
        ),
    )
    monkeypatch.setattr(api_app_module, "create_http_app", _FakeMcpApp)
    monkeypatch.setattr(api_app_module, "get_settings", lambda: settings)
    monkeypatch.setattr(health_module, "get_settings", lambda: settings)
    monkeypatch.setattr(health_module, "database_is_ready", lambda: database_ready)
    monkeypatch.setattr(
        health_module,
        "get_metrics_registry",
        lambda: SimpleNamespace(timing_snapshots=lambda limit=20: []),
    )
    monkeypatch.setattr(
        health_module,
        "_queue_metrics",
        lambda queue_name: (
            QueueMetricsResponse(
                queue_name=queue_name,
                queue_length=3,
                queue_visible_length=2,
                newest_msg_age_sec=5,
                oldest_msg_age_sec=12,
                total_messages=3,
                scrape_time=datetime(2026, 3, 19, 8, 0, tzinfo=UTC),
            )
            if database_ready
            else None
        ),
    )
    monkeypatch.setattr(
        api_app_module,
        "LocalFilesystemStorage",
        lambda root_path: SimpleNamespace(ensure_root=lambda: None),
    )
    monkeypatch.setattr(api_app_module, "configure_logging", lambda level: None)
    monkeypatch.setattr(api_app_module, "dispose_engine", lambda: None)


def test_openapi_contract_exposes_health_and_readiness_models(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _patch_contract_app(monkeypatch, tmp_path, database_ready=True)

    with TestClient(create_app()) as client:
        schema = client.get("/openapi.json").json()

    health_schema = schema["paths"]["/healthz"]["get"]["responses"]["200"]["content"][
        "application/json"
    ]["schema"]
    readiness_schema = schema["paths"]["/readyz"]["get"]["responses"]["200"]["content"][
        "application/json"
    ]["schema"]

    assert health_schema == {"$ref": "#/components/schemas/HealthResponse"}
    assert readiness_schema == {"$ref": "#/components/schemas/ReadinessResponse"}


def test_health_response_matches_golden_contract(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _patch_contract_app(monkeypatch, tmp_path, database_ready=True)

    with TestClient(create_app()) as client:
        payload = client.get("/healthz").json()

    assert payload == _golden_payload("healthz.json")
    assert api_schemas.HealthResponse.model_validate(payload).model_dump(mode="json") == payload
    assert mcp_schemas.HealthResponse.model_validate(payload).model_dump(mode="json") == payload


@pytest.mark.parametrize(
    ("database_ready", "golden_name"),
    [(True, "readyz-ready.json"), (False, "readyz-degraded.json")],
)
def test_readiness_response_matches_golden_contract(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    database_ready: bool,
    golden_name: str,
) -> None:
    _patch_contract_app(monkeypatch, tmp_path, database_ready=database_ready)

    with TestClient(create_app()) as client:
        payload = client.get("/readyz").json()

    expected = _golden_payload(golden_name)
    expected["storage_root"] = str(tmp_path / "artifacts")
    assert payload == expected
    assert api_schemas.ReadinessResponse.model_validate(payload).model_dump(mode="json") == payload
    assert mcp_schemas.ReadinessResponse.model_validate(payload).model_dump(mode="json") == payload


def test_api_and_mcp_schema_exports_stay_aligned() -> None:
    assert api_schemas.HealthResponse is mcp_schemas.HealthResponse
    assert api_schemas.ReadinessResponse is mcp_schemas.ReadinessResponse
    assert api_schemas.DocumentListResponse is mcp_schemas.DocumentListResponse
    assert api_schemas.DocumentOutlineResponse is mcp_schemas.DocumentOutlineResponse
    assert api_schemas.DocumentResult is mcp_schemas.DocumentResult
    assert api_schemas.PassageSearchResponse is mcp_schemas.PassageSearchResponse
    assert api_schemas.TableSearchResponse is mcp_schemas.TableSearchResponse
    assert api_schemas.PassageContextResponse is mcp_schemas.PassageContextResponse
    assert api_schemas.TableDetailResponse is mcp_schemas.TableDetailResponse
    assert api_schemas.ContextPackResponse is mcp_schemas.ContextPackResponse
