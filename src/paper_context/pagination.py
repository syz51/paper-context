from __future__ import annotations

import base64
import binascii
import hashlib
import json
from collections.abc import Mapping
from typing import Any


class CursorError(ValueError):
    """Raised when an opaque pagination cursor is invalid."""


def encode_cursor(payload: Mapping[str, Any]) -> str:
    raw = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def decode_cursor(cursor: str, *, expected_kind: str | None = None) -> dict[str, Any]:
    padding = "=" * (-len(cursor) % 4)
    try:
        decoded = base64.urlsafe_b64decode((cursor + padding).encode("ascii"))
        payload = json.loads(decoded.decode("utf-8"))
    except (ValueError, UnicodeDecodeError, json.JSONDecodeError, binascii.Error) as exc:
        raise CursorError("invalid cursor") from exc
    if not isinstance(payload, dict):
        raise CursorError("invalid cursor")
    if expected_kind is not None and payload.get("kind") != expected_kind:
        raise CursorError("cursor kind mismatch")
    return payload


def fingerprint_payload(payload: object) -> str:
    raw = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return hashlib.blake2b(raw, digest_size=12).hexdigest()
