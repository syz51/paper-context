from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True)
class StorageArtifact:
    storage_ref: str
    checksum: str
    size_bytes: int


class StorageInterface(Protocol):
    def ensure_root(self) -> None:
        pass

    def store_bytes(self, relative_path: str, content: bytes) -> StorageArtifact: ...

    def resolve(self, storage_ref: str) -> Path: ...
