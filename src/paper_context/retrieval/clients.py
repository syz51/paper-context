from __future__ import annotations

import json
import math
from dataclasses import dataclass
from hashlib import blake2b
from http import client as http_client
from urllib.parse import urlsplit

from .types import EmbeddingBatch, EmbeddingInputType, RerankItem, RetrievalError


def _post_json(
    *,
    url: str,
    api_key: str,
    payload: dict[str, object],
) -> dict[str, object]:
    parsed_url = urlsplit(url)
    if parsed_url.scheme != "https":
        raise RetrievalError(f"unsupported provider endpoint scheme: {parsed_url.scheme!r}")
    if parsed_url.hostname is None:
        raise RetrievalError("provider endpoint is missing a hostname")
    body = json.dumps(payload).encode("utf-8")
    request_path = parsed_url.path or "/"
    if parsed_url.query:
        request_path = f"{request_path}?{parsed_url.query}"
    connection = http_client.HTTPSConnection(
        host=parsed_url.hostname,
        port=parsed_url.port,
        timeout=30,
    )
    try:
        connection.request(
            "POST",
            request_path,
            body=body,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
        )
        response = connection.getresponse()
        response_payload = response.read().decode("utf-8", errors="replace")
    except OSError as exc:  # pragma: no cover - exercised only with live providers
        raise RetrievalError(f"provider request failed: {exc}") from exc
    except (
        http_client.HTTPException
    ) as exc:  # pragma: no cover - exercised only with live providers
        raise RetrievalError(f"provider request failed: {exc}") from exc
    finally:
        connection.close()

    if response.status >= 400:
        raise RetrievalError(f"provider request failed with {response.status}: {response_payload}")
    return json.loads(response_payload)


def _normalize_embeddings(raw_embeddings: list[object]) -> tuple[tuple[float, ...], ...]:
    embeddings: list[tuple[float, ...]] = []
    for embedding in raw_embeddings:
        if not isinstance(embedding, list):
            raise RetrievalError("provider returned a non-list embedding payload")
        embeddings.append(tuple(float(value) for value in embedding))
    if not embeddings:
        return ()
    expected_dimensions = len(embeddings[0])
    if expected_dimensions == 0:
        raise RetrievalError("provider returned an empty embedding vector")
    for embedding in embeddings:
        if len(embedding) != expected_dimensions:
            raise RetrievalError("provider returned embeddings with inconsistent dimensions")
    return tuple(embeddings)


class VoyageEmbeddingClient:
    provider = "voyage"

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        endpoint: str = "https://api.voyageai.com/v1/embeddings",
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.endpoint = endpoint

    def embed(
        self,
        texts: list[str],
        *,
        input_type: EmbeddingInputType,
    ) -> EmbeddingBatch:
        if not texts:
            return EmbeddingBatch(
                provider=self.provider,
                model=self.model,
                dimensions=0,
                embeddings=(),
            )
        payload = {
            "input": texts,
            "model": self.model,
            "input_type": input_type,
        }
        response = _post_json(url=self.endpoint, api_key=self.api_key, payload=payload)
        embeddings_payload: list[object]
        data_payload = response.get("data")
        embeddings_field = response.get("embeddings")
        if isinstance(data_payload, list):
            embeddings_payload = [
                item.get("embedding")
                for item in data_payload
                if isinstance(item, dict) and "embedding" in item
            ]
        elif isinstance(embeddings_field, list):
            embeddings_payload = list(embeddings_field)
        else:
            raise RetrievalError("Voyage response did not contain embeddings")

        embeddings = _normalize_embeddings(embeddings_payload)
        if len(embeddings) != len(texts):
            raise RetrievalError("Voyage returned a different number of embeddings than requested")
        dimensions = len(embeddings[0]) if embeddings else 0
        return EmbeddingBatch(
            provider=self.provider,
            model=self.model,
            dimensions=dimensions,
            embeddings=embeddings,
        )


class DeterministicEmbeddingClient:
    provider = "deterministic"

    def __init__(self, *, model: str = "deterministic-hash", dimensions: int = 1024) -> None:
        self.model = model
        self.dimensions = dimensions

    def embed(
        self,
        texts: list[str],
        *,
        input_type: EmbeddingInputType,
    ) -> EmbeddingBatch:
        del input_type
        embeddings = tuple(self._embed_text(text) for text in texts)
        return EmbeddingBatch(
            provider=self.provider,
            model=self.model,
            dimensions=self.dimensions,
            embeddings=embeddings,
        )

    def _embed_text(self, text: str) -> tuple[float, ...]:
        buckets = [0.0] * self.dimensions
        tokens = [token for token in text.lower().split() if token]
        if not tokens:
            return tuple(buckets)
        for token in tokens:
            digest = blake2b(token.encode("utf-8"), digest_size=16).digest()
            index = int.from_bytes(digest[:4], byteorder="big") % self.dimensions
            sign = -1.0 if digest[4] % 2 else 1.0
            magnitude = 1.0 + (digest[5] / 255.0)
            buckets[index] += sign * magnitude
        norm = math.sqrt(sum(value * value for value in buckets))
        if norm == 0:
            return tuple(buckets)
        return tuple(value / norm for value in buckets)


class ZeroEntropyRerankerClient:
    provider = "zero_entropy"

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        endpoint: str = "https://api.zeroentropy.dev/v1/models/rerank",
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.endpoint = endpoint

    def rerank(
        self,
        *,
        query: str,
        documents: list[str],
        top_n: int | None = None,
    ) -> list[RerankItem]:
        if not documents:
            return []
        payload: dict[str, object] = {
            "query": query,
            "documents": documents,
            "model": self.model,
        }
        if top_n is not None:
            payload["top_n"] = top_n

        response = _post_json(url=self.endpoint, api_key=self.api_key, payload=payload)
        raw_results = response.get("results")
        if not isinstance(raw_results, list):
            raw_results = response.get("data")
        if not isinstance(raw_results, list):
            raise RetrievalError("Zero Entropy response did not contain rerank results")

        results: list[RerankItem] = []
        for default_index, item in enumerate(raw_results):
            if not isinstance(item, dict):
                continue
            index = item.get("index", default_index)
            score = (
                item.get("score")
                if item.get("score") is not None
                else item.get("relevance_score", item.get("relevance"))
            )
            if not isinstance(index, int):
                raise RetrievalError("Zero Entropy returned a non-integer rerank index")
            if score is None:
                raise RetrievalError("Zero Entropy returned a rerank item without a score")
            results.append(RerankItem(index=index, score=float(score)))
        return results


@dataclass(frozen=True)
class _ScoredText:
    index: int
    score: float


class HeuristicRerankerClient:
    provider = "deterministic"

    def __init__(self, *, model: str = "heuristic-overlap") -> None:
        self.model = model

    def rerank(
        self,
        *,
        query: str,
        documents: list[str],
        top_n: int | None = None,
    ) -> list[RerankItem]:
        query_terms = set(_normalize_terms(query))
        scored = [
            _ScoredText(index=index, score=self._score_document(query_terms, document))
            for index, document in enumerate(documents)
        ]
        scored.sort(key=lambda item: (-item.score, item.index))
        if top_n is not None:
            scored = scored[:top_n]
        return [RerankItem(index=item.index, score=item.score) for item in scored]

    def _score_document(self, query_terms: set[str], document: str) -> float:
        document_terms = _normalize_terms(document)
        if not document_terms:
            return 0.0
        overlap = sum(1 for term in document_terms if term in query_terms)
        distinct_overlap = len(set(document_terms) & query_terms)
        return float(overlap + distinct_overlap)


def _normalize_terms(text: str) -> list[str]:
    return [
        token.strip(".,:;!?()[]{}\"'").lower()
        for token in text.split()
        if token.strip(".,:;!?()[]{}\"'")
    ]
