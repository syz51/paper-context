from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = pytest.mark.regression


def _service_block(text: str, service: str) -> str:
    lines = text.splitlines()
    start = None
    for index, line in enumerate(lines):
        if line.startswith(f"  {service}:"):
            start = index + 1
            break
    if start is None:
        raise AssertionError(f"service {service!r} not found")

    block = []
    for line in lines[start:]:
        if line.startswith("  ") and not line.startswith("    "):
            break
        block.append(line)
    return "\n".join(block)


def _compose_text() -> str:
    return Path("docker-compose.yml").read_text()


def _compose_prod_text() -> str:
    return Path("docker-compose.prod.yml").read_text()


def test_compose_defines_only_the_app_service() -> None:
    text = _compose_text()
    assert "\n  api:" not in text
    assert "\n  mcp:" not in text
    assert "\n  app:" in text


def test_no_standalone_mcp_port_exposed() -> None:
    assert "8001:8001" not in _compose_text()


def test_migrate_service_isan_isolated_runner() -> None:
    block = _service_block(_compose_text(), "migrate")
    assert "alembic" in block
    assert "upgrade" in block
    assert "head" in block
    assert "profiles:" in block
    assert "- migrate" in block


def test_worker_depends_on_app_service() -> None:
    block = _service_block(_compose_text(), "worker")
    assert "depends_on" in block
    assert "app:" in block


def test_production_compose_keeps_single_hosted_app_shape() -> None:
    text = _compose_prod_text()
    assert "\n  db:" not in text
    assert "\n  api:" not in text
    assert "\n  mcp:" not in text
    assert "\n  app:" in text
    assert "\n  migrate:" in text
    assert "dokploy-network" in text
    assert "8001:8001" not in text
