from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Protocol


@dataclass(frozen=True)
class StorageArtifact:
    storage_ref: str
    checksum: str
    size_bytes: int


class StorageLimitExceededError(ValueError):
    """Raised when a streamed write exceeds the configured size limit."""


class StorageInterface(Protocol):
    def ensure_root(self) -> None:
        pass

    def store_bytes(self, relative_path: str, content: bytes) -> StorageArtifact: ...

    def store_file(
        self,
        relative_path: str,
        fileobj: BinaryIO,
        *,
        max_size_bytes: int | None = None,
        chunk_size: int = 1024 * 1024,
    ) -> StorageArtifact: ...

    def resolve(self, storage_ref: str) -> Path: ...

    def delete(self, storage_ref: str) -> None: ...
