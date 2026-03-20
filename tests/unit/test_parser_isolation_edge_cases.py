from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import orjson
import pytest

from paper_context.ingestion import parser_isolation
from paper_context.ingestion.types import ParserArtifact

pytestmark = pytest.mark.unit

_REAL_POPEN = subprocess.Popen


def _launch_real_process(code: str, **kwargs):
    child_kwargs = {key: kwargs[key] for key in ("stdout", "stderr", "preexec_fn") if key in kwargs}
    return _REAL_POPEN([sys.executable, "-c", code], **child_kwargs)


def test_subprocess_parser_requires_content_or_source_path() -> None:
    result = parser_isolation.SubprocessPdfParser("docling").parse("paper.pdf")

    assert result.gate_status == "fail"
    assert result.failure_code == "docling_subprocess_protocol_failed"
    assert result.failure_message == "parser requires content bytes or a source path"


def test_subprocess_parser_handles_os_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def raise_os_error(*args, **kwargs):
        raise OSError("no exec")

    monkeypatch.setattr(parser_isolation.subprocess, "Popen", raise_os_error)

    result = parser_isolation.SubprocessPdfParser("docling").parse("paper.pdf", b"%PDF-1.4")

    assert result.gate_status == "fail"
    assert result.failure_code == "docling_subprocess_launch_failed"
    assert result.failure_message == "no exec"


def test_subprocess_parser_uses_source_path_and_handles_nonzero_exit(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, object] = {}

    def fake_popen(command, **kwargs):
        captured["command"] = command
        return _launch_real_process(
            "import sys; "
            "sys.stdout.write('bad stdout'); "
            "sys.stderr.write('bad stderr'); "
            "sys.stdout.flush(); "
            "sys.stderr.flush(); "
            "sys.exit(9)",
            **kwargs,
        )

    monkeypatch.setattr(parser_isolation.subprocess, "Popen", fake_popen)

    result = parser_isolation.SubprocessPdfParser("docling").parse(
        "paper.pdf",
        source_path=tmp_path / "paper.pdf",
    )

    assert str(tmp_path / "paper.pdf") == cast(list[str], captured["command"])[-1]
    assert result.gate_status == "fail"
    assert result.failure_code == "docling_subprocess_failed"
    assert "exited with code 9" in (result.failure_message or "")


def test_subprocess_parser_stops_after_exceeding_output_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_popen(command, **kwargs):
        return _launch_real_process(
            "import sys; sys.stdout.write('x' * 1024); sys.stdout.flush()",
            **kwargs,
        )

    monkeypatch.setattr(parser_isolation.subprocess, "Popen", fake_popen)

    result = parser_isolation.SubprocessPdfParser(
        "docling",
        config=parser_isolation.ParserIsolationConfig(output_limit_mb=0),
    ).parse("paper.pdf", b"%PDF-1.4")

    assert result.gate_status == "fail"
    assert result.failure_code == "docling_subprocess_output_limit_exceeded"
    assert "output limit" in (result.failure_message or "")


def test_run_parser_subprocess_bounds_flooded_stdout_and_stderr() -> None:
    code = (
        "import os\n"
        "stdout_chunk = b'o' * 4096\n"
        "stderr_chunk = b'e' * 4096\n"
        "for _ in range(2048):\n"
        "    os.write(1, stdout_chunk)\n"
        "    os.write(2, stderr_chunk)\n"
    )

    result = parser_isolation._run_parser_subprocess(
        [sys.executable, "-c", code],
        timeout_seconds=10,
        output_limit_bytes=128 * 1024,
        preexec_fn=None,
    )

    assert result.timed_out is False
    assert result.output_limit_exceeded is True
    assert len(result.stdout) + len(result.stderr) == 128 * 1024
    assert result.stdout
    assert result.stderr


def test_build_pdf_parser_and_create_inprocess_parser_branches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sentinel = object()
    original = parser_isolation._create_inprocess_parser
    monkeypatch.setattr(parser_isolation, "_create_inprocess_parser", lambda name: sentinel)

    assert parser_isolation.build_pdf_parser("docling", isolated=False) is sentinel
    assert isinstance(
        parser_isolation.build_pdf_parser("docling", isolated=True),
        parser_isolation.SubprocessPdfParser,
    )

    monkeypatch.setattr(parser_isolation, "_create_inprocess_parser", original)
    monkeypatch.setattr(parser_isolation, "DoclingPdfParser", lambda: "docling-parser")
    monkeypatch.setattr(parser_isolation, "PdfPlumberPdfParser", lambda: "pdfplumber-parser")
    assert parser_isolation._create_inprocess_parser("docling") == "docling-parser"
    assert parser_isolation._create_inprocess_parser("pdfplumber") == "pdfplumber-parser"
    with pytest.raises(ValueError, match="unsupported parser 'other'"):
        parser_isolation._create_inprocess_parser("other")


def test_copy_artifact_content_paths_and_missing_sources(tmp_path: Path) -> None:
    source = tmp_path / "artifact.bin"
    source.write_bytes(b"payload")
    destination = tmp_path / "copied.bin"

    parser_isolation._copy_artifact_content(
        ParserArtifact(
            artifact_type="docling_parse",
            parser="docling",
            filename="artifact.bin",
            content_path=source,
        ),
        destination,
    )
    assert destination.read_bytes() == b"payload"

    with pytest.raises(ValueError, match="missing both content and content_path"):
        parser_isolation._copy_artifact_content(
            cast(ParserArtifact, SimpleNamespace(content=None, content_path=None)),
            destination,
        )


def test_payload_to_parser_result_protocol_failures(tmp_path: Path) -> None:
    output_root = tmp_path / "output"
    output_root.mkdir()

    invalid_json = parser_isolation._payload_to_parser_result(
        parser_name="docling",
        payload=b"{not-json",
        output_root=output_root,
    )
    assert invalid_json.failure_code == "docling_subprocess_protocol_failed"

    output_root.mkdir()
    invalid_artifact = parser_isolation._payload_to_parser_result(
        parser_name="docling",
        payload=orjson.dumps({"artifact": "bad"}),
        output_root=output_root,
    )
    assert invalid_artifact.failure_message == "docling parser returned an invalid artifact payload"

    output_root.mkdir()
    invalid_content_path = parser_isolation._payload_to_parser_result(
        parser_name="docling",
        payload=orjson.dumps({"artifact": {"content_path": 7}}),
        output_root=output_root,
    )
    assert invalid_content_path.failure_message == (
        "docling parser returned an invalid artifact content path"
    )

    output_root.mkdir()
    missing_artifact = parser_isolation._payload_to_parser_result(
        parser_name="docling",
        payload=orjson.dumps({"artifact": {"content_path": "artifact.bin"}}),
        output_root=output_root,
    )
    assert missing_artifact.failure_message == "docling parser did not produce an artifact file"


def test_payload_to_parser_result_handles_parsed_document_loader_paths(tmp_path: Path) -> None:
    output_root = tmp_path / "output"
    output_root.mkdir()
    artifact_path = output_root / "artifact.bin"
    artifact_path.write_bytes(b"artifact")

    missing_document = parser_isolation._payload_to_parser_result(
        parser_name="docling",
        payload=orjson.dumps(
            {
                "gate_status": "pass",
                "parsed_document_path": "parsed_document.json",
                "artifact": {"content_path": "artifact.bin"},
            }
        ),
        output_root=output_root,
    )
    assert missing_document.failure_message == (
        "docling parser did not produce a parsed document file"
    )

    output_root.mkdir()
    artifact_path = output_root / "artifact.bin"
    artifact_path.write_bytes(b"artifact")
    parsed_document_path = output_root / "parsed_document.json"
    parsed_document_path.write_bytes(orjson.dumps(["not-a-dict"]))

    result = parser_isolation._payload_to_parser_result(
        parser_name="docling",
        payload=orjson.dumps(
            {
                "gate_status": "pass",
                "parsed_document_path": "parsed_document.json",
                "warnings": "not-a-list",
                "artifact": {
                    "artifact_type": "docling_parse",
                    "parser": "docling",
                    "filename": "docling.json",
                    "content_path": "artifact.bin",
                },
            }
        ),
        output_root=output_root,
    )

    assert result.warnings == []
    with pytest.raises(TypeError, match="must be a JSON object"):
        result.load_parsed_document()


def test_path_resolution_and_coercion_helpers(tmp_path: Path) -> None:
    output_root = tmp_path / "output"
    output_root.mkdir()

    assert (
        parser_isolation._resolve_worker_output_path(output_root, "artifact.bin")
        == (output_root / "artifact.bin").resolve()
    )
    assert parser_isolation._resolve_worker_output_path(output_root, "../escape.bin") is None

    assert parser_isolation._optional_str(5) == "5"
    assert parser_isolation._optional_int(None) is None
    assert parser_isolation._optional_int(True) == 1
    assert parser_isolation._optional_int("7") == 7
    with pytest.raises(TypeError, match="cannot coerce object to int"):
        parser_isolation._optional_int(object())

    assert parser_isolation._optional_float(None) is None
    assert parser_isolation._optional_float(False) == 0.0
    assert parser_isolation._optional_float("7.5") == 7.5
    with pytest.raises(TypeError, match="cannot coerce object to float"):
        parser_isolation._optional_float(object())

    parsed_table = parser_isolation._parsed_table_from_dict(
        {
            "section_key": "results",
            "caption": "cap",
            "headers": ["A", 2],
            "rows": [["1", None], "skip"],
            "page_start": 1,
            "page_end": 2,
        }
    )
    assert parsed_table.headers == ["A", "2"]
    assert parsed_table.rows == [["1", "None"]]

    parsed_reference = parser_isolation._parsed_reference_from_dict(
        {
            "raw_citation": "Ada 2024",
            "normalized_title": "Title",
            "authors": "Ada",
            "publication_year": 2024,
            "doi": "10.1/example",
            "source_confidence": 0.7,
        }
    )
    assert parsed_reference.authors is None
    assert parsed_reference.normalized_title == "Title"


def test_decode_output_cleanup_and_failure_result(tmp_path: Path) -> None:
    cleanup_root = tmp_path / "cleanup"
    cleanup_root.mkdir()
    (cleanup_root / "file.txt").write_text("payload", encoding="utf-8")

    assert parser_isolation._decode_output(None) is None
    assert parser_isolation._decode_output("text") == "text"
    assert parser_isolation._decode_output(b"bytes") == "bytes"
    parser_isolation._cleanup_output_root(cleanup_root)
    parser_isolation._cleanup_output_root(None)
    assert not cleanup_root.exists()

    result = parser_isolation._subprocess_failure_result(
        parser_name="docling",
        failure_code="docling_failed",
        failure_message="boom",
        details={"stdout": "trace"},
    )
    assert result.gate_status == "fail"
    assert result.artifact.content is not None
    assert orjson.loads(result.artifact.content) == {
        "error": "boom",
        "details": {"stdout": "trace"},
    }


def test_set_resource_limit_failure_paths() -> None:
    class GetLimitErrorResource:
        RLIM_INFINITY = 2**63 - 1

        @staticmethod
        def getrlimit(limit_kind: int):
            del limit_kind
            raise OSError("boom")

    parser_isolation._set_resource_limit(GetLimitErrorResource, 1, 10)

    class SetLimitErrorResource:
        RLIM_INFINITY = 2**63 - 1

        @staticmethod
        def getrlimit(limit_kind: int):
            del limit_kind
            return (0, 5)

        @staticmethod
        def setrlimit(limit_kind: int, limits: tuple[int, int]) -> None:
            del limit_kind, limits
            raise ValueError("boom")

    parser_isolation._set_resource_limit(SetLimitErrorResource, 1, 10)

    class FinalizeLimitErrorResource:
        RLIM_INFINITY = 2**63 - 1
        calls: list[tuple[int, int]] = []

        @staticmethod
        def getrlimit(limit_kind: int):
            del limit_kind
            return (0, FinalizeLimitErrorResource.RLIM_INFINITY)

        @staticmethod
        def setrlimit(limit_kind: int, limits: tuple[int, int]) -> None:
            del limit_kind
            FinalizeLimitErrorResource.calls.append(limits)
            if len(FinalizeLimitErrorResource.calls) == 2:
                raise ValueError("boom")

    parser_isolation._set_resource_limit(FinalizeLimitErrorResource, 1, 10)
    assert FinalizeLimitErrorResource.calls == [
        (10, FinalizeLimitErrorResource.RLIM_INFINITY),
        (10, 10),
    ]

    class NoFinalizeResource:
        RLIM_INFINITY = 2**63 - 1
        calls: list[tuple[int, int]] = []

        @staticmethod
        def getrlimit(limit_kind: int):
            del limit_kind
            return (0, 20)

        @staticmethod
        def setrlimit(limit_kind: int, limits: tuple[int, int]) -> None:
            del limit_kind
            NoFinalizeResource.calls.append(limits)

    parser_isolation._set_resource_limit(NoFinalizeResource, 1, 10)
    assert NoFinalizeResource.calls == [(10, 20)]


def test_parser_resource_limits_import_error(monkeypatch: pytest.MonkeyPatch) -> None:
    real_import = __import__

    def fake_import(name, *args, **kwargs):
        if name == "resource":
            raise ImportError("missing")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", fake_import)

    assert (
        parser_isolation._parser_resource_limits(parser_isolation.ParserIsolationConfig()) is None
    )


def test_payload_to_parser_result_preserves_failure_fields(tmp_path: Path) -> None:
    output_root = tmp_path / "output"
    output_root.mkdir()
    artifact_path = output_root / "artifact.bin"
    artifact_path.write_bytes(b"artifact")

    result = parser_isolation._payload_to_parser_result(
        parser_name="docling",
        payload=orjson.dumps(
            {
                "gate_status": "degraded",
                "artifact": {
                    "artifact_type": "docling_parse",
                    "parser": "docling",
                    "filename": "docling.json",
                    "content_path": "artifact.bin",
                },
                "warnings": ["warn"],
                "failure_code": "warned",
                "failure_message": "degraded",
            }
        ),
        output_root=output_root,
    )

    assert result.gate_status == "degraded"
    assert result.failure_code == "warned"
    assert result.failure_message == "degraded"
    assert result.artifact.content_path == artifact_path
