from __future__ import annotations

import hashlib
from pathlib import Path

from .base import StorageArtifact, StorageInterface


class LocalFilesystemStorage(StorageInterface):
    def __init__(self, root_path: Path) -> None:
        self._root_path = root_path

    def ensure_root(self) -> None:
        self._root_path.mkdir(parents=True, exist_ok=True)

    def store_bytes(self, relative_path: str, content: bytes) -> StorageArtifact:
        self.ensure_root()
        target = self._root_path / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
        checksum = hashlib.sha256(content).hexdigest()
        return StorageArtifact(
            storage_ref=str(target.relative_to(self._root_path)),
            checksum=checksum,
            size_bytes=len(content),
        )

    def resolve(self, storage_ref: str) -> Path:
        return self._root_path / storage_ref
