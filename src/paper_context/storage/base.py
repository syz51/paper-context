from __future__ import annotations

from abc import abstractmethod
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
    @abstractmethod
    def ensure_root(self) -> None:  # pragma: no cover - protocol stub
        return None

    @abstractmethod
    def store_bytes(
        self, relative_path: str, content: bytes
    ) -> StorageArtifact:  # pragma: no cover - protocol stub
        raise NotImplementedError

    @abstractmethod
    def store_file(
        self,
        relative_path: str,
        fileobj: BinaryIO,
        *,
        max_size_bytes: int | None = None,
        chunk_size: int = 1024 * 1024,
    ) -> StorageArtifact:  # pragma: no cover - protocol stub
        raise NotImplementedError

    @abstractmethod
    def resolve(self, storage_ref: str) -> Path:  # pragma: no cover - protocol stub
        raise NotImplementedError

    @abstractmethod
    def delete(self, storage_ref: str) -> None:  # pragma: no cover - protocol stub
        raise NotImplementedError
