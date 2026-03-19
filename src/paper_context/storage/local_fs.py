from __future__ import annotations

import hashlib
import uuid
from io import BytesIO
from pathlib import Path
from typing import BinaryIO

from .base import StorageArtifact, StorageInterface, StorageLimitExceededError


class LocalFilesystemStorage(StorageInterface):
    def __init__(self, root_path: Path) -> None:
        self._root_path = root_path

    def _resolve_storage_path(self, storage_ref: str) -> Path:
        if Path(storage_ref).is_absolute():
            raise ValueError(f"storage ref {storage_ref!r} escapes storage root")
        root = self._root_path.resolve(strict=False)
        resolved_target = (root / Path(storage_ref)).resolve(strict=False)
        try:
            resolved_target.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"storage ref {storage_ref!r} escapes storage root") from exc
        return resolved_target

    def ensure_root(self) -> None:
        self._root_path.mkdir(parents=True, exist_ok=True)

    def store_bytes(self, relative_path: str, content: bytes) -> StorageArtifact:
        return self.store_file(relative_path, BytesIO(content))

    def store_file(
        self,
        relative_path: str,
        fileobj: BinaryIO,
        *,
        max_size_bytes: int | None = None,
        chunk_size: int = 1024 * 1024,
    ) -> StorageArtifact:
        target = self._resolve_storage_path(relative_path)
        self.ensure_root()
        target.parent.mkdir(parents=True, exist_ok=True)
        temp_target = target.parent / f".{target.name}.{uuid.uuid4().hex}.tmp"
        checksum = hashlib.sha256()
        size_bytes = 0

        if hasattr(fileobj, "seek"):
            fileobj.seek(0)

        try:
            with temp_target.open("wb") as handle:
                while True:
                    chunk = fileobj.read(chunk_size)
                    if not chunk:
                        break
                    if not isinstance(chunk, (bytes, bytearray)):
                        raise TypeError("storage streams must yield bytes")
                    size_bytes += len(chunk)
                    if max_size_bytes is not None and size_bytes > max_size_bytes:
                        raise StorageLimitExceededError(
                            f"stream exceeded the {max_size_bytes}-byte limit"
                        )
                    handle.write(chunk)
                    checksum.update(chunk)

            temp_target.replace(target)
        except Exception:
            temp_target.unlink(missing_ok=True)
            raise

        return StorageArtifact(
            storage_ref=str(target.relative_to(self._root_path.resolve(strict=False))),
            checksum=checksum.hexdigest(),
            size_bytes=size_bytes,
        )

    def resolve(self, storage_ref: str) -> Path:
        return self._resolve_storage_path(storage_ref)

    def delete(self, storage_ref: str) -> None:
        target = self._resolve_storage_path(storage_ref)
        try:
            target.unlink()
        except FileNotFoundError:
            return

        parent = target.parent
        root = self._root_path.resolve(strict=False)
        while parent != root and parent.exists():
            try:
                parent.rmdir()
            except OSError:
                break
            parent = parent.parent
