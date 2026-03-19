from __future__ import annotations

import subprocess
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from typing import Any, BinaryIO, cast
from unittest.mock import MagicMock

import orjson
import pytest

from paper_context.db.session import connection_scope
from paper_context.ingestion import parser_isolation, parser_worker
from paper_context.ingestion.types import (
    ParsedDocument,
    ParsedParagraph,
    ParsedSection,
    ParserArtifact,
    ParserResult,
)
from paper_context.models.retrieval import PgVector
from paper_context.storage.base import StorageInterface, StorageLimitExceededError
from paper_context.storage.local_fs import LocalFilesystemStorage

pytestmark = pytest.mark.unit


class _StringStream:
    def __init__(self) -> None:
        self._calls = 0

    def seek(self, offset: int) -> None:
        assert offset == 0

    def read(self, chunk_size: int) -> str:
        del chunk_size
        self._calls += 1
        return "not-bytes" if self._calls == 1 else ""


class _BytesNoSeekStream:
    def __init__(self, payload: bytes) -> None:
        self._payload = payload
        self._read = False

    def read(self, chunk_size: int) -> bytes:
        del chunk_size
        if self._read:
            return b""
        self._read = True
        return self._payload


class _Buffer:
    def __init__(self, read_value: bytes = b"") -> None:
        self._read_value = read_value
        self.writes: list[bytes] = []

    def read(self) -> bytes:
        return self._read_value

    def write(self, payload: bytes) -> None:
        self.writes.append(payload)


def test_local_filesystem_storage_enforces_size_limit(tmp_path: Path) -> None:
    storage = LocalFilesystemStorage(tmp_path / "artifacts")

    with pytest.raises(StorageLimitExceededError, match="3-byte limit"):
        storage.store_file("nested/file.bin", BytesIO(b"abcd"), max_size_bytes=3, chunk_size=2)

    assert not (tmp_path / "artifacts" / "nested" / "file.bin").exists()
    assert list((tmp_path / "artifacts" / "nested").glob("*.tmp")) == []


def test_local_filesystem_storage_rejects_non_bytes_stream(tmp_path: Path) -> None:
    storage = LocalFilesystemStorage(tmp_path / "artifacts")

    with pytest.raises(TypeError, match="must yield bytes"):
        storage.store_file("nested/file.bin", cast(BinaryIO, _StringStream()))

    assert not (tmp_path / "artifacts" / "nested" / "file.bin").exists()


def test_local_filesystem_storage_accepts_stream_without_seek(tmp_path: Path) -> None:
    storage = LocalFilesystemStorage(tmp_path / "artifacts")

    artifact = storage.store_file("nested/file.bin", cast(BinaryIO, _BytesNoSeekStream(b"abc")))

    assert artifact.size_bytes == 3
    assert (tmp_path / "artifacts" / "nested" / "file.bin").read_bytes() == b"abc"


def test_local_filesystem_storage_delete_prunes_empty_directories(tmp_path: Path) -> None:
    root = tmp_path / "artifacts"
    storage = LocalFilesystemStorage(root)
    storage.store_bytes("deep/path/file.bin", b"payload")

    storage.delete("deep/path/file.bin")

    assert not (root / "deep" / "path" / "file.bin").exists()
    assert not (root / "deep" / "path").exists()
    assert not (root / "deep").exists()
    storage.delete("deep/path/file.bin")


def test_local_filesystem_storage_delete_stops_when_parent_not_empty(tmp_path: Path) -> None:
    root = tmp_path / "artifacts"
    storage = LocalFilesystemStorage(root)
    storage.store_bytes("deep/path/file.bin", b"payload")
    storage.store_bytes("deep/keep.bin", b"keep")

    storage.delete("deep/path/file.bin")

    assert not (root / "deep" / "path").exists()
    assert (root / "deep" / "keep.bin").read_bytes() == b"keep"


@pytest.mark.parametrize("method_name", ["store_bytes", "resolve", "delete"])
def test_local_filesystem_storage_rejects_paths_outside_root(
    tmp_path: Path, method_name: str
) -> None:
    root = tmp_path / "artifacts"
    storage = LocalFilesystemStorage(root)

    method = getattr(storage, method_name)
    args: tuple[object, ...]
    if method_name == "store_bytes":
        args = ("../escape.bin", b"payload")
    else:
        args = ("../escape.bin",)

    with pytest.raises(ValueError, match="escapes storage root"):
        method(*args)

    assert not (tmp_path / "escape.bin").exists()


def test_pgvector_processors_cover_scalar_and_sequence_inputs() -> None:
    unbounded = PgVector()
    bounded = PgVector(3)

    assert unbounded.get_col_spec() == "vector"
    assert bounded.get_col_spec() == "vector(3)"

    bind = bounded.bind_processor(cast(Any, None))
    assert bind(None) is None
    assert bind("[1,2,3]") == "[1,2,3]"
    assert bind((1.0, 2.0, 3.0)) == "[1.0,2.0,3.0]"

    result = bounded.result_processor(cast(Any, None), cast(Any, None))
    assert result(None) is None
    assert result([1.0, 2.0]) == [1.0, 2.0]
    assert result((1.0, 2.0)) == [1.0, 2.0]
    assert result("[1,2.5,3]") == [1.0, 2.5, 3.0]
    assert result("[]") == []


def test_connection_scope_commits_and_closes_on_success() -> None:
    connection = MagicMock()
    connection.in_transaction.return_value = True
    engine = MagicMock(connect=MagicMock(return_value=connection))

    with connection_scope(engine) as yielded:
        assert yielded is connection

    connection.commit.assert_called_once_with()
    connection.rollback.assert_not_called()
    connection.close.assert_called_once_with()


def test_connection_scope_rolls_back_on_error() -> None:
    connection = MagicMock()
    connection.in_transaction.return_value = True
    engine = MagicMock(connect=MagicMock(return_value=connection))

    with pytest.raises(RuntimeError, match="boom"):
        with connection_scope(engine):
            raise RuntimeError("boom")

    connection.rollback.assert_called_once_with()
    connection.commit.assert_not_called()
    connection.close.assert_called_once_with()


def test_connection_scope_can_skip_transaction_management() -> None:
    connection = MagicMock()
    connection.in_transaction.return_value = True
    engine = MagicMock(connect=MagicMock(return_value=connection))

    with connection_scope(engine, transactional=False):
        pass

    connection.commit.assert_not_called()
    connection.rollback.assert_not_called()
    connection.close.assert_called_once_with()


def test_parser_worker_main_validates_argv() -> None:
    with pytest.raises(SystemExit, match="usage: python -m paper_context.ingestion.parser_worker"):
        parser_worker.main(["docling"])


def test_parser_worker_main_reads_from_stdin(monkeypatch: pytest.MonkeyPatch) -> None:
    stdout_buffer = _Buffer()
    stdin_buffer = _Buffer(read_value=b"stdin-bytes")
    captured: dict[str, object] = {}

    def fake_run_parser_worker(
        parser_name: str,
        filename: str,
        output_root: Path,
        content: bytes | None = None,
        *,
        source_path: Path | None = None,
    ) -> bytes:
        captured.update(
            parser_name=parser_name,
            filename=filename,
            output_root=output_root,
            content=content,
            source_path=source_path,
        )
        return b"payload"

    monkeypatch.setattr(parser_worker, "run_parser_worker", fake_run_parser_worker)
    monkeypatch.setattr(parser_worker.sys, "stdin", SimpleNamespace(buffer=stdin_buffer))
    monkeypatch.setattr(parser_worker.sys, "stdout", SimpleNamespace(buffer=stdout_buffer))

    assert parser_worker.main(["docling", "paper.pdf", "/tmp/out"]) == 0
    assert stdout_buffer.writes == [b"payload"]
    assert captured == {
        "parser_name": "docling",
        "filename": "paper.pdf",
        "output_root": Path("/tmp/out"),
        "content": b"stdin-bytes",
        "source_path": None,
    }


def test_parser_worker_main_uses_source_path_when_provided(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stdout_buffer = _Buffer()
    captured: dict[str, object] = {}

    def fake_run_parser_worker(
        parser_name: str,
        filename: str,
        output_root: Path,
        content: bytes | None = None,
        *,
        source_path: Path | None = None,
    ) -> bytes:
        captured.update(
            parser_name=parser_name,
            filename=filename,
            output_root=output_root,
            content=content,
            source_path=source_path,
        )
        return b"payload"

    monkeypatch.setattr(parser_worker, "run_parser_worker", fake_run_parser_worker)
    monkeypatch.setattr(parser_worker.sys, "stdout", SimpleNamespace(buffer=stdout_buffer))

    assert parser_worker.main(["pdfplumber", "paper.pdf", "/tmp/out", "/tmp/source.pdf"]) == 0
    assert stdout_buffer.writes == [b"payload"]
    assert captured == {
        "parser_name": "pdfplumber",
        "filename": "paper.pdf",
        "output_root": Path("/tmp/out"),
        "content": None,
        "source_path": Path("/tmp/source.pdf"),
    }


def test_storage_interface_default_ensure_root_is_a_noop() -> None:
    noop_storage = cast(StorageInterface, SimpleNamespace())
    assert StorageInterface.ensure_root(noop_storage) is None


def test_parser_isolation_subprocess_parser_covers_failure_branches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parser = parser_isolation.SubprocessPdfParser("docling")

    no_input = parser.parse("paper.pdf")
    assert no_input.gate_status == "fail"
    assert no_input.failure_code == "docling_subprocess_protocol_failed"

    def _nonzero_exit(*args, **kwargs):
        del kwargs
        return subprocess.CompletedProcess(args[0], 1, stdout=b"oops", stderr=b"err")

    monkeypatch.setattr(parser_isolation.subprocess, "run", _nonzero_exit)
    nonzero = parser.parse("paper.pdf", b"%PDF-1.4")
    assert nonzero.gate_status == "fail"
    assert nonzero.failure_code == "docling_subprocess_failed"

    def _bad_json(*args, **kwargs):
        del kwargs
        return subprocess.CompletedProcess(args[0], 0, stdout=b"not-json", stderr=b"")

    monkeypatch.setattr(parser_isolation.subprocess, "run", _bad_json)
    bad_json = parser.parse("paper.pdf", b"%PDF-1.4")
    assert bad_json.gate_status == "fail"
    assert bad_json.failure_code == "docling_subprocess_protocol_failed"


def test_parser_isolation_payload_helpers_cover_validation_and_cleanup(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source"
    output_root = tmp_path / "payload-output"
    artifact_file = source_root / "artifact.bin"
    artifact_file.parent.mkdir(parents=True, exist_ok=True)
    artifact_file.write_bytes(b"artifact")

    parsed_document = ParsedDocument(
        title="Paper",
        authors=["Ada"],
        abstract="Abstract",
        publication_year=2024,
        metadata_confidence=0.9,
        sections=[
            ParsedSection(
                key="s1",
                heading="Intro",
                heading_path=["Intro"],
                level=1,
                page_start=1,
                page_end=1,
                paragraphs=[ParsedParagraph(text="body", page_start=1, page_end=1)],
            )
        ],
        tables=[],
        references=[],
    )
    result = ParserResult(
        gate_status="pass",
        parsed_document=parsed_document,
        artifact=ParserArtifact(
            artifact_type="docling_parse",
            parser="docling",
            filename="docling.json",
            content_path=artifact_file,
            cleanup_root=source_root,
        ),
    )

    payload = parser_isolation._write_parser_result_payload(result, output_root=output_root)
    restored = parser_isolation._payload_to_parser_result(
        parser_name="docling",
        payload=payload,
        output_root=output_root,
    )

    assert restored.gate_status == "pass"
    assert restored.parsed_document is None
    assert restored.load_parsed_document() == parsed_document

    bad_payloads = [
        b"not-json",
        orjson.dumps({"artifact": []}),
        orjson.dumps({"artifact": {"content_path": 1}}),
        orjson.dumps(
            {
                "artifact": {"content_path": "missing.bin"},
                "parsed_document_path": "missing.json",
            }
        ),
    ]
    for index, payload in enumerate(bad_payloads, start=1):
        failure = parser_isolation._payload_to_parser_result(
            parser_name=f"parser-{index}",
            payload=payload,
            output_root=tmp_path / f"bad-{index}",
        )
        assert failure.gate_status == "fail"
        assert failure.failure_code == f"parser-{index}_subprocess_protocol_failed"


def test_parser_isolation_helper_functions_cover_misc_paths(tmp_path: Path) -> None:
    created = parser_isolation.build_pdf_parser("docling", isolated=True)
    assert isinstance(created, parser_isolation.SubprocessPdfParser)
    assert parser_isolation.build_pdf_parser("pdfplumber", isolated=False)

    assert parser_isolation._optional_str(1) == "1"
    assert parser_isolation._optional_int("2") == 2
    assert parser_isolation._optional_float("3.5") == 3.5
    assert parser_isolation._decode_output(b"hello") == "hello"

    resolved = parser_isolation._resolve_worker_output_path(tmp_path, "nested/result.json")
    assert resolved == (tmp_path / "nested" / "result.json").resolve()
    assert parser_isolation._resolve_worker_output_path(tmp_path, "../escape") is None

    failure = parser_isolation._subprocess_failure_result(
        parser_name="docling",
        failure_code="docling_failed",
        failure_message="boom",
    )
    assert failure.gate_status == "fail"
    assert failure.failure_code == "docling_failed"

    cleanup_root = tmp_path / "cleanup"
    cleanup_root.mkdir()
    parser_isolation._cleanup_output_root(cleanup_root)
    assert not cleanup_root.exists()
