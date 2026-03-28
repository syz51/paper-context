from __future__ import annotations

from typing import cast

import pytest

from paper_context.retrieval import clients
from paper_context.retrieval.types import EmbeddingBatch, RerankItem, RetrievalError

pytestmark = pytest.mark.unit


class _FakeResponse:
    def __init__(self, payload: bytes, *, status: int = 200) -> None:
        self._payload = payload
        self.status = status

    def read(self) -> bytes:
        return self._payload


def test_post_json_builds_https_request(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    class _FakeConnection:
        def __init__(self, *, host: str, port: int | None, timeout: int) -> None:
            captured["host"] = host
            captured["port"] = port
            captured["timeout"] = timeout

        def request(
            self,
            method: str,
            url: str,
            body: bytes,
            headers: dict[str, str],
        ) -> None:
            captured["method"] = method
            captured["url"] = url
            captured["body"] = body
            captured["headers"] = headers

        def getresponse(self) -> _FakeResponse:
            return _FakeResponse(b'{"ok": true}')

        def close(self) -> None:
            captured["closed"] = True

    monkeypatch.setattr(clients.http_client, "HTTPSConnection", _FakeConnection)

    result = clients._post_json(
        url="https://example.com/api",
        api_key="secret",
        payload={"query": "paper"},
    )

    assert captured["timeout"] == 30
    assert captured["host"] == "example.com"
    assert captured["port"] is None
    assert captured["method"] == "POST"
    assert captured["url"] == "/api"
    assert captured["body"] == b'{"query": "paper"}'
    headers = {
        key.lower(): value for key, value in cast(dict[str, str], captured["headers"]).items()
    }
    assert headers["authorization"] == "Bearer secret"
    assert headers["content-type"] == "application/json"
    assert captured["closed"] is True
    assert result == {"ok": True}


def test_post_json_rejects_non_https_url() -> None:
    with pytest.raises(RetrievalError, match="unsupported provider endpoint scheme"):
        clients._post_json(url="http://example.com/api", api_key="secret", payload={})

    with pytest.raises(RetrievalError, match="missing a hostname"):
        clients._post_json(url="https:///api", api_key="secret", payload={})


def test_post_json_preserves_query_string_in_request_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class _FakeConnection:
        def __init__(self, *, host: str, port: int | None, timeout: int) -> None:
            captured["host"] = host
            captured["port"] = port
            captured["timeout"] = timeout

        def request(
            self,
            method: str,
            url: str,
            body: bytes,
            headers: dict[str, str],
        ) -> None:
            captured["method"] = method
            captured["url"] = url
            captured["body"] = body
            captured["headers"] = headers

        def getresponse(self) -> _FakeResponse:
            return _FakeResponse(b'{"ok": true}')

        def close(self) -> None:
            captured["closed"] = True

    monkeypatch.setattr(clients.http_client, "HTTPSConnection", _FakeConnection)

    clients._post_json(
        url="https://example.com/api?version=1",
        api_key="secret",
        payload={"query": "paper"},
    )

    assert captured["host"] == "example.com"
    assert captured["port"] is None
    assert captured["timeout"] == 30
    assert captured["method"] == "POST"
    assert captured["url"] == "/api?version=1"
    assert captured["body"] == b'{"query": "paper"}'
    headers = {
        key.lower(): value for key, value in cast(dict[str, str], captured["headers"]).items()
    }
    assert headers["authorization"] == "Bearer secret"
    assert headers["content-type"] == "application/json"
    assert captured["closed"] is True


def test_post_json_wraps_http_and_url_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    class _ErrorResponseConnection:
        def request(
            self,
            method: str,
            url: str,
            body: bytes,
            headers: dict[str, str],
        ) -> None:
            assert method == "POST"
            assert url == "/api"
            assert body == b"{}"
            assert headers["Accept"] == "application/json"

        def getresponse(self) -> _FakeResponse:
            return _FakeResponse(b"slow down", status=429)

        def close(self) -> None:
            return None

    monkeypatch.setattr(
        clients.http_client, "HTTPSConnection", lambda **kwargs: _ErrorResponseConnection()
    )

    with pytest.raises(RetrievalError, match="provider request failed with 429: slow down"):
        clients._post_json(url="https://example.com/api", api_key="secret", payload={})

    class _OfflineConnection:
        def request(
            self,
            method: str,
            url: str,
            body: bytes,
            headers: dict[str, str],
        ) -> None:
            assert method == "POST"
            assert url == "/api"
            assert body == b"{}"
            assert headers["Accept"] == "application/json"
            raise OSError("offline")

        def close(self) -> None:
            return None

    monkeypatch.setattr(
        clients.http_client, "HTTPSConnection", lambda **kwargs: _OfflineConnection()
    )

    with pytest.raises(RetrievalError, match="provider request failed: offline"):
        clients._post_json(url="https://example.com/api", api_key="secret", payload={})


def test_normalize_embeddings_and_failures() -> None:
    assert clients._normalize_embeddings([]) == ()
    assert clients._normalize_embeddings([[1, 2.5], [3, 4]]) == ((1.0, 2.5), (3.0, 4.0))

    with pytest.raises(RetrievalError, match="non-list embedding payload"):
        clients._normalize_embeddings([{"embedding": [1, 2]}])

    with pytest.raises(RetrievalError, match="empty embedding vector"):
        clients._normalize_embeddings([[]])

    with pytest.raises(RetrievalError, match="inconsistent dimensions"):
        clients._normalize_embeddings([[1, 2], [3]])


def test_voyage_embedding_client_embed_uses_data_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def fake_post_json(*, url: str, api_key: str, payload: dict[str, object]) -> dict[str, object]:
        captured["url"] = url
        captured["api_key"] = api_key
        captured["payload"] = payload
        return {"data": [{"embedding": [0.1, 0.2]}, {"embedding": [0.3, 0.4]}]}

    monkeypatch.setattr(clients, "_post_json", fake_post_json)

    client = clients.VoyageEmbeddingClient(
        api_key="secret",
        model="voyage-test",
        endpoint="https://example.com/embeddings",
    )
    batch = client.embed(["alpha", "beta"], input_type="document")

    assert captured == {
        "url": "https://example.com/embeddings",
        "api_key": "secret",
        "payload": {"input": ["alpha", "beta"], "model": "voyage-test", "input_type": "document"},
    }
    assert batch == EmbeddingBatch(
        provider="voyage",
        model="voyage-test",
        dimensions=2,
        embeddings=((0.1, 0.2), (0.3, 0.4)),
    )

    empty_batch = client.embed([], input_type="document")
    assert empty_batch == EmbeddingBatch(
        provider="voyage",
        model="voyage-test",
        dimensions=0,
        embeddings=(),
    )


def test_voyage_embedding_client_embed_uses_embeddings_payload_and_rejects_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        clients,
        "_post_json",
        lambda **kwargs: {"embeddings": [[1, 2], [3, 4]]},
    )

    client = clients.VoyageEmbeddingClient(api_key="secret", model="voyage-test")
    batch = client.embed(["alpha", "beta"], input_type="query")
    assert batch.dimensions == 2
    assert batch.embeddings == ((1.0, 2.0), (3.0, 4.0))

    monkeypatch.setattr(clients, "_post_json", lambda **kwargs: {"embeddings": [[1, 2]]})
    with pytest.raises(RetrievalError, match="different number of embeddings"):
        client.embed(["alpha", "beta"], input_type="query")

    monkeypatch.setattr(clients, "_post_json", lambda **kwargs: {"unexpected": []})
    with pytest.raises(RetrievalError, match="did not contain embeddings"):
        client.embed(["alpha"], input_type="query")


def test_zero_entropy_reranker_client_rerank_handles_payload_variants(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_post_json(*, url: str, api_key: str, payload: dict[str, object]) -> dict[str, object]:
        captured["url"] = url
        captured["api_key"] = api_key
        captured["payload"] = payload
        return {"results": [{"index": 1, "score": 2.5}, {"index": 0, "relevance_score": 1.5}]}

    monkeypatch.setattr(clients, "_post_json", fake_post_json)

    client = clients.ZeroEntropyRerankerClient(
        api_key="secret",
        model="rerank-test",
        endpoint="https://example.com/rerank",
    )
    results = client.rerank(query="paper", documents=["alpha", "beta"], top_n=1)

    assert captured == {
        "url": "https://example.com/rerank",
        "api_key": "secret",
        "payload": {
            "query": "paper",
            "documents": ["alpha", "beta"],
            "model": "rerank-test",
            "top_n": 1,
        },
    }
    assert results == [RerankItem(index=1, score=2.5), RerankItem(index=0, score=1.5)]

    monkeypatch.setattr(
        clients,
        "_post_json",
        lambda **kwargs: {"data": [{"index": 0, "relevance": 3}, "skip me"]},
    )
    assert client.rerank(query="paper", documents=["alpha"], top_n=None) == [
        RerankItem(index=0, score=3.0)
    ]


def test_zero_entropy_reranker_client_rerank_failure_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    client = clients.ZeroEntropyRerankerClient(api_key="secret", model="rerank-test")

    assert client.rerank(query="paper", documents=[]) == []

    monkeypatch.setattr(clients, "_post_json", lambda **kwargs: {"results": "bad"})
    with pytest.raises(RetrievalError, match="did not contain rerank results"):
        client.rerank(query="paper", documents=["alpha"])

    monkeypatch.setattr(
        clients,
        "_post_json",
        lambda **kwargs: {"results": [{"index": "bad", "score": 1.0}]},
    )
    with pytest.raises(RetrievalError, match="non-integer rerank index"):
        client.rerank(query="paper", documents=["alpha"])

    monkeypatch.setattr(
        clients,
        "_post_json",
        lambda **kwargs: {"results": [{"index": 0}]},
    )
    with pytest.raises(RetrievalError, match="without a score"):
        client.rerank(query="paper", documents=["alpha"])


def test_heuristic_reranker_client_rerank_orders_and_truncates() -> None:
    client = clients.HeuristicRerankerClient(model="heuristic-test")

    results = client.rerank(
        query="Alpha beta",
        documents=[
            "beta only",
            "alpha beta beta",
            "gamma",
        ],
        top_n=2,
    )

    assert results == [RerankItem(index=1, score=5.0), RerankItem(index=0, score=2.0)]
    assert client.rerank(query="Alpha beta", documents=["!!!", "beta"], top_n=None) == [
        RerankItem(index=1, score=2.0),
        RerankItem(index=0, score=0.0),
    ]


def test_normalize_terms_and_deterministic_embedding_client() -> None:
    assert clients._normalize_terms('Alpha, beta!  "Gamma"') == ["alpha", "beta", "gamma"]
    assert clients._normalize_terms("   ") == []

    client = clients.DeterministicEmbeddingClient(model="det", dimensions=4)
    batch = client.embed(["Alpha beta", ""], input_type="query")

    assert batch.provider == "deterministic"
    assert batch.model == "det"
    assert batch.dimensions == 4
    assert len(batch.embeddings) == 2
    assert batch.embeddings[1] == (0.0, 0.0, 0.0, 0.0)
    assert sum(value * value for value in batch.embeddings[0]) == pytest.approx(1.0)


def test_deterministic_embedding_client_returns_zero_vector_when_norm_cancels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Digest:
        def __init__(self, token: bytes) -> None:
            self._token = token

        def digest(self) -> bytes:
            if self._token == b"alpha":
                return bytes([0, 0, 0, 0, 0, 0]) + bytes(10)
            return bytes([0, 0, 0, 0, 1, 0]) + bytes(10)

    monkeypatch.setattr(clients, "blake2b", lambda token, digest_size: _Digest(token))
    client = clients.DeterministicEmbeddingClient(model="det", dimensions=1)

    batch = client.embed(["alpha beta"], input_type="document")

    assert batch.embeddings == ((0.0,),)
