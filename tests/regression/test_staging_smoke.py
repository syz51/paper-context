from __future__ import annotations

import os

import httpx
import pytest

pytestmark = [
    pytest.mark.regression,
    pytest.mark.staging_only,
    pytest.mark.slow,
]


def _base_url() -> str:
    base_url = os.environ.get("PAPER_CONTEXT_STAGING_BASE_URL")
    if not base_url:
        pytest.skip("set PAPER_CONTEXT_STAGING_BASE_URL to run staging smoke tests")
    return base_url.rstrip("/")


def test_staging_health_readiness_and_mcp_smoke() -> None:
    with httpx.Client(base_url=_base_url(), follow_redirects=True, timeout=10.0) as client:
        health = client.get("/healthz")
        readiness = client.get("/readyz")
        mcp = client.get("/mcp", headers={"accept": "application/json"})

    assert health.status_code == 200
    assert health.json()["status"] == "ok"
    assert readiness.status_code == 200
    assert readiness.json()["service"] == "app"
    assert mcp.status_code == 406
