from __future__ import annotations

import uuid

_ARTIFACT_NAMESPACE = uuid.UUID("2f4f87fd-c8ae-42e7-9590-0b7f654a33c8")
_RETRIEVAL_NAMESPACE = uuid.UUID("68b1ce8d-1a1e-49f5-b1db-47052d7ece6c")


def artifact_id(*, ingest_job_id: uuid.UUID, artifact_type: str, parser: str) -> uuid.UUID:
    return uuid.uuid5(
        _ARTIFACT_NAMESPACE,
        f"{ingest_job_id}:{artifact_type}:{parser}",
    )


def retrieval_index_run_id(*, ingest_job_id: uuid.UUID) -> uuid.UUID:
    return uuid.uuid5(_RETRIEVAL_NAMESPACE, str(ingest_job_id))
