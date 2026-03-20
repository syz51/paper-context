from __future__ import annotations

import os
import selectors
import shutil
import subprocess  # nosec B404
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import orjson

from .parsers import DoclingPdfParser, PdfParser, PdfPlumberPdfParser
from .types import (
    GateStatus,
    ParsedDocument,
    ParsedParagraph,
    ParsedReference,
    ParsedSection,
    ParsedTable,
    ParserArtifact,
    ParserResult,
)

_DEFAULT_TIMEOUT_SECONDS = 120
_DEFAULT_MEMORY_LIMIT_MB = 2_048
_DEFAULT_OUTPUT_LIMIT_MB = 32
_ARTIFACT_FILENAME = "artifact.bin"
_PARSED_DOCUMENT_FILENAME = "parsed_document.json"
_SUBPROCESS_CAPTURE_CHUNK_SIZE = 64 * 1024


@dataclass(frozen=True)
class ParserIsolationConfig:
    timeout_seconds: int = _DEFAULT_TIMEOUT_SECONDS
    memory_limit_mb: int = _DEFAULT_MEMORY_LIMIT_MB
    output_limit_mb: int = _DEFAULT_OUTPUT_LIMIT_MB


@dataclass(frozen=True)
class _BoundedSubprocessResult:
    returncode: int | None
    stdout: bytes
    stderr: bytes
    timed_out: bool = False
    output_limit_exceeded: bool = False


class SubprocessPdfParser:
    def __init__(
        self,
        parser_name: str,
        *,
        config: ParserIsolationConfig | None = None,
    ) -> None:
        self.name = parser_name
        self._config = config or ParserIsolationConfig()

    def parse(
        self,
        filename: str,
        content: bytes | None = None,
        *,
        source_path: Path | None = None,
    ) -> ParserResult:
        output_root = Path(tempfile.mkdtemp(prefix=f"{self.name}-parser-"))
        staged_input_path: Path | None = None
        command = [
            sys.executable,
            "-m",
            "paper_context.ingestion.parser_worker",
            self.name,
            filename,
            str(output_root),
        ]
        try:
            if source_path is not None:
                command.append(str(source_path))
            elif content is not None:
                staged_input_path = _stage_parser_input(
                    content,
                    parser_name=self.name,
                    filename=filename,
                )
                command.append(str(staged_input_path))
            else:
                _cleanup_output_root(output_root)
                return _subprocess_failure_result(
                    parser_name=self.name,
                    failure_code=f"{self.name}_subprocess_protocol_failed",
                    failure_message="parser requires content bytes or a source path",
                )
            completed = _run_parser_subprocess(
                command,
                timeout_seconds=self._config.timeout_seconds,
                output_limit_bytes=max(0, self._config.output_limit_mb) * 1024 * 1024,
                preexec_fn=_parser_resource_limits(self._config),
            )
        except OSError as exc:
            _cleanup_output_root(output_root)
            return _subprocess_failure_result(
                parser_name=self.name,
                failure_code=f"{self.name}_subprocess_launch_failed",
                failure_message=str(exc),
            )
        except subprocess.SubprocessError as exc:
            _cleanup_output_root(output_root)
            return _subprocess_failure_result(
                parser_name=self.name,
                failure_code=f"{self.name}_subprocess_launch_failed",
                failure_message=str(exc),
            )
        finally:
            _cleanup_staged_input(staged_input_path)

        if completed.timed_out:
            _cleanup_output_root(output_root)
            return _subprocess_failure_result(
                parser_name=self.name,
                failure_code=f"{self.name}_timeout",
                failure_message=(
                    f"{self.name} parser exceeded the "
                    f"{self._config.timeout_seconds}-second isolation timeout"
                ),
                details={
                    "stdout": _decode_output(completed.stdout),
                    "stderr": _decode_output(completed.stderr),
                },
            )
        if completed.output_limit_exceeded:
            _cleanup_output_root(output_root)
            return _subprocess_failure_result(
                parser_name=self.name,
                failure_code=f"{self.name}_subprocess_output_limit_exceeded",
                failure_message=(
                    f"{self.name} parser exceeded the "
                    f"{self._config.output_limit_mb}-MB isolation output limit"
                ),
                details={
                    "stdout": _decode_output(completed.stdout),
                    "stderr": _decode_output(completed.stderr),
                },
            )
        if completed.returncode != 0:
            _cleanup_output_root(output_root)
            return _subprocess_failure_result(
                parser_name=self.name,
                failure_code=f"{self.name}_subprocess_failed",
                failure_message=(f"{self.name} parser exited with code {completed.returncode}"),
                details={
                    "stdout": _decode_output(completed.stdout),
                    "stderr": _decode_output(completed.stderr),
                },
            )

        return _payload_to_parser_result(
            parser_name=self.name,
            payload=completed.stdout,
            output_root=output_root,
        )


def build_pdf_parser(
    parser_name: str,
    *,
    isolated: bool,
    config: ParserIsolationConfig | None = None,
) -> PdfParser:
    if isolated:
        return SubprocessPdfParser(parser_name, config=config)
    return _create_inprocess_parser(parser_name)


def _create_inprocess_parser(parser_name: str) -> PdfParser:
    if parser_name == "docling":
        return DoclingPdfParser()
    if parser_name == "pdfplumber":
        return PdfPlumberPdfParser()
    raise ValueError(f"unsupported parser {parser_name!r}")


def run_parser_worker(
    parser_name: str,
    filename: str,
    output_root: Path,
    content: bytes | None = None,
    *,
    source_path: Path | None = None,
) -> bytes:
    parser = _create_inprocess_parser(parser_name)
    result = parser.parse(filename=filename, content=content, source_path=source_path)
    return _write_parser_result_payload(result, output_root=output_root)


def _write_parser_result_payload(result: ParserResult, *, output_root: Path) -> bytes:
    output_root.mkdir(parents=True, exist_ok=True)
    artifact_path = output_root / _ARTIFACT_FILENAME
    try:
        _copy_artifact_content(result.artifact, artifact_path)
        parsed_document_path: str | None = None
        if result.gate_status == "pass" and result.parsed_document is not None:
            parsed_document_file = output_root / _PARSED_DOCUMENT_FILENAME
            parsed_document_file.write_bytes(orjson.dumps(result.parsed_document))
            parsed_document_path = _PARSED_DOCUMENT_FILENAME
        return orjson.dumps(
            {
                "gate_status": result.gate_status,
                "parsed_document_path": parsed_document_path,
                "artifact": {
                    "artifact_type": result.artifact.artifact_type,
                    "parser": result.artifact.parser,
                    "filename": result.artifact.filename,
                    "content_path": _ARTIFACT_FILENAME,
                },
                "warnings": result.warnings,
                "failure_code": result.failure_code,
                "failure_message": result.failure_message,
            }
        )
    finally:
        result.artifact.cleanup_local_copy()


def _copy_artifact_content(artifact: ParserArtifact, destination: Path) -> None:
    if artifact.content is not None:
        destination.write_bytes(artifact.content)
        return
    if artifact.content_path is None:
        raise ValueError("parser artifact is missing both content and content_path")
    with artifact.content_path.open("rb") as source, destination.open("wb") as target:
        shutil.copyfileobj(source, target, length=1024 * 1024)


def _stage_parser_input(content: bytes, *, parser_name: str, filename: str) -> Path:
    suffix = Path(filename).suffix or ".bin"
    staged_input_fd, staged_input_name = tempfile.mkstemp(
        prefix=f"{parser_name}-parser-input-",
        suffix=suffix,
    )
    try:
        with os.fdopen(staged_input_fd, "wb") as staged_input:
            staged_input.write(content)
        return Path(staged_input_name)
    except Exception:
        _cleanup_file(Path(staged_input_name))
        raise


def _cleanup_staged_input(staged_input_path: Path | None) -> None:
    if staged_input_path is None:
        return
    _cleanup_file(staged_input_path)


def _cleanup_file(path: Path | None) -> None:
    if path is None:
        return
    try:
        path.unlink()
    except FileNotFoundError:
        return


def _run_parser_subprocess(
    command: list[str],
    *,
    timeout_seconds: int,
    output_limit_bytes: int,
    preexec_fn: Any | None,
) -> _BoundedSubprocessResult:
    process = subprocess.Popen(  # nosec B603
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        preexec_fn=preexec_fn,
    )
    stdout_stream = process.stdout
    stderr_stream = process.stderr
    if stdout_stream is None or stderr_stream is None:
        process.kill()
        process.wait()
        raise subprocess.SubprocessError("parser subprocess did not expose output pipes")

    stdout_buffer = bytearray()
    stderr_buffer = bytearray()
    captured_bytes = 0
    timed_out = False
    output_limit_exceeded = False
    selector = selectors.DefaultSelector()
    selector.register(stdout_stream, selectors.EVENT_READ, "stdout")
    selector.register(stderr_stream, selectors.EVENT_READ, "stderr")
    deadline = time.monotonic() + timeout_seconds
    try:
        while selector.get_map():
            remaining_seconds = deadline - time.monotonic()
            if remaining_seconds <= 0:
                timed_out = True
                _kill_parser_process(process)
                break
            for key, _ in selector.select(timeout=remaining_seconds):
                stream = cast(Any, key.fileobj)
                chunk = stream.read1(_SUBPROCESS_CAPTURE_CHUNK_SIZE)
                if not chunk:
                    selector.unregister(stream)
                    continue
                remaining_output_bytes = output_limit_bytes - captured_bytes
                if remaining_output_bytes <= 0:
                    output_limit_exceeded = True
                    _kill_parser_process(process)
                    break
                captured_chunk = chunk[:remaining_output_bytes]
                if key.data == "stdout":
                    stdout_buffer.extend(captured_chunk)
                else:
                    stderr_buffer.extend(captured_chunk)
                captured_bytes += len(captured_chunk)
                if len(chunk) > len(captured_chunk):
                    output_limit_exceeded = True
                    _kill_parser_process(process)
                    break
            if timed_out or output_limit_exceeded:
                break
        returncode = _wait_for_parser_process(process)
        return _BoundedSubprocessResult(
            returncode=returncode,
            stdout=bytes(stdout_buffer),
            stderr=bytes(stderr_buffer),
            timed_out=timed_out,
            output_limit_exceeded=output_limit_exceeded,
        )
    finally:
        selector.close()
        stdout_stream.close()
        stderr_stream.close()


def _kill_parser_process(process: subprocess.Popen[bytes]) -> None:
    try:
        process.kill()
    except OSError:
        return


def _wait_for_parser_process(process: subprocess.Popen[bytes]) -> int:
    try:
        return process.wait(timeout=1)
    except subprocess.TimeoutExpired:
        _kill_parser_process(process)
        return process.wait()


def _payload_to_parser_result(
    *,
    parser_name: str,
    payload: bytes,
    output_root: Path,
) -> ParserResult:
    try:
        decoded = orjson.loads(payload)
    except orjson.JSONDecodeError as exc:
        _cleanup_output_root(output_root)
        return _subprocess_failure_result(
            parser_name=parser_name,
            failure_code=f"{parser_name}_subprocess_protocol_failed",
            failure_message=f"{parser_name} parser returned invalid JSON: {exc}",
            details={"stdout": _decode_output(payload)},
        )
    artifact_payload = decoded.get("artifact")
    if not isinstance(artifact_payload, dict):
        _cleanup_output_root(output_root)
        return _subprocess_failure_result(
            parser_name=parser_name,
            failure_code=f"{parser_name}_subprocess_protocol_failed",
            failure_message=f"{parser_name} parser returned an invalid artifact payload",
            details={"stdout": decoded},
        )
    parsed_document_path_value = decoded.get("parsed_document_path")
    warnings_payload = decoded.get("warnings")
    warnings = (
        [str(value) for value in warnings_payload] if isinstance(warnings_payload, list) else []
    )
    artifact_relative_path = artifact_payload.get("content_path")
    if not isinstance(artifact_relative_path, str):
        _cleanup_output_root(output_root)
        return _subprocess_failure_result(
            parser_name=parser_name,
            failure_code=f"{parser_name}_subprocess_protocol_failed",
            failure_message=f"{parser_name} parser returned an invalid artifact content path",
            details={"stdout": decoded},
        )
    artifact_path = _resolve_worker_output_path(output_root, artifact_relative_path)
    if artifact_path is None or not artifact_path.is_file():
        _cleanup_output_root(output_root)
        return _subprocess_failure_result(
            parser_name=parser_name,
            failure_code=f"{parser_name}_subprocess_protocol_failed",
            failure_message=f"{parser_name} parser did not produce an artifact file",
            details={"stdout": decoded},
        )
    parsed_document: ParsedDocument | None = None
    if isinstance(parsed_document_path_value, str):
        parsed_document_path = _resolve_worker_output_path(output_root, parsed_document_path_value)
        if parsed_document_path is None or not parsed_document_path.is_file():
            _cleanup_output_root(output_root)
            return _subprocess_failure_result(
                parser_name=parser_name,
                failure_code=f"{parser_name}_subprocess_protocol_failed",
                failure_message=f"{parser_name} parser did not produce a parsed document file",
                details={"stdout": decoded},
            )
        parsed_document_loader = _build_parsed_document_loader(parsed_document_path)
    else:
        parsed_document_loader = None
    return ParserResult(
        gate_status=cast(GateStatus, decoded.get("gate_status", "fail")),
        parsed_document=parsed_document,
        artifact=ParserArtifact(
            artifact_type=str(artifact_payload.get("artifact_type", f"{parser_name}_parse")),
            parser=str(artifact_payload.get("parser", parser_name)),
            filename=str(artifact_payload.get("filename", f"{parser_name}.json")),
            content_path=artifact_path,
            cleanup_root=output_root,
        ),
        warnings=warnings,
        failure_code=(
            str(decoded["failure_code"]) if decoded.get("failure_code") is not None else None
        ),
        failure_message=(
            str(decoded["failure_message"]) if decoded.get("failure_message") is not None else None
        ),
        parsed_document_loader=parsed_document_loader,
    )


def _resolve_worker_output_path(output_root: Path, relative_path: str) -> Path | None:
    candidate = (output_root / relative_path).resolve()
    try:
        candidate.relative_to(output_root.resolve())
    except ValueError:
        return None
    return candidate


def _parsed_document_from_dict(payload: dict[str, Any]) -> ParsedDocument:
    return ParsedDocument(
        title=_optional_str(payload.get("title")),
        authors=[str(author) for author in payload.get("authors", [])],
        abstract=_optional_str(payload.get("abstract")),
        publication_year=_optional_int(payload.get("publication_year")),
        metadata_confidence=_optional_float(payload.get("metadata_confidence")),
        sections=[_parsed_section_from_dict(section) for section in payload.get("sections", [])],
        tables=[_parsed_table_from_dict(table) for table in payload.get("tables", [])],
        references=[
            _parsed_reference_from_dict(reference) for reference in payload.get("references", [])
        ],
    )


def _build_parsed_document_loader(parsed_document_path: Path):
    def _load() -> ParsedDocument:
        parsed_document_payload = orjson.loads(parsed_document_path.read_bytes())
        if not isinstance(parsed_document_payload, dict):
            raise TypeError("parsed document payload must be a JSON object")
        return _parsed_document_from_dict(parsed_document_payload)

    return _load


def _parsed_section_from_dict(payload: dict[str, Any]) -> ParsedSection:
    return ParsedSection(
        key=str(payload["key"]),
        heading=_optional_str(payload.get("heading")),
        heading_path=[str(value) for value in payload.get("heading_path", [])],
        level=int(payload["level"]),
        page_start=_optional_int(payload.get("page_start")),
        page_end=_optional_int(payload.get("page_end")),
        parent_key=_optional_str(payload.get("parent_key")),
        paragraphs=[
            _parsed_paragraph_from_dict(paragraph) for paragraph in payload.get("paragraphs", [])
        ],
    )


def _parsed_paragraph_from_dict(payload: dict[str, Any]) -> ParsedParagraph:
    provenance = payload.get("provenance_offsets")
    return ParsedParagraph(
        text=str(payload["text"]),
        page_start=_optional_int(payload.get("page_start")),
        page_end=_optional_int(payload.get("page_end")),
        provenance_offsets=provenance if isinstance(provenance, dict) else None,
    )


def _parsed_table_from_dict(payload: dict[str, Any]) -> ParsedTable:
    return ParsedTable(
        section_key=str(payload["section_key"]),
        caption=_optional_str(payload.get("caption")),
        headers=[str(value) for value in payload.get("headers", [])],
        rows=[
            [str(cell) for cell in row] for row in payload.get("rows", []) if isinstance(row, list)
        ],
        page_start=_optional_int(payload.get("page_start")),
        page_end=_optional_int(payload.get("page_end")),
    )


def _parsed_reference_from_dict(payload: dict[str, Any]) -> ParsedReference:
    authors_payload = payload.get("authors")
    authors = (
        [str(value) for value in authors_payload] if isinstance(authors_payload, list) else None
    )
    return ParsedReference(
        raw_citation=str(payload["raw_citation"]),
        normalized_title=_optional_str(payload.get("normalized_title")),
        authors=authors,
        publication_year=_optional_int(payload.get("publication_year")),
        doi=_optional_str(payload.get("doi")),
        source_confidence=_optional_float(payload.get("source_confidence")),
    )


def _optional_str(value: object) -> str | None:
    return str(value) if value is not None else None


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float, str)):
        return int(value)
    raise TypeError(f"cannot coerce {type(value).__name__} to int")


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float, str)):
        return float(value)
    raise TypeError(f"cannot coerce {type(value).__name__} to float")


def _subprocess_failure_result(
    *,
    parser_name: str,
    failure_code: str,
    failure_message: str,
    details: dict[str, Any] | None = None,
) -> ParserResult:
    detail_payload: dict[str, object] = {"error": failure_message}
    if details:
        detail_payload["details"] = details
    return ParserResult(
        gate_status="fail",
        parsed_document=None,
        artifact=ParserArtifact(
            artifact_type=f"{parser_name}_parse",
            parser=parser_name,
            filename=f"{parser_name}-isolation-error.json",
            content=orjson.dumps(detail_payload),
        ),
        failure_code=failure_code,
        failure_message=failure_message,
    )


def _decode_output(payload: bytes | str | None) -> str | None:
    if not payload:
        return None
    if isinstance(payload, str):
        return payload
    return payload.decode("utf-8", errors="replace")


def _cleanup_output_root(output_root: Path | None) -> None:
    if output_root is None:
        return
    shutil.rmtree(output_root, ignore_errors=True)


def _parser_resource_limits(config: ParserIsolationConfig):
    try:
        import resource
    except ImportError:  # pragma: no cover - platform specific
        return None

    def _apply() -> None:
        memory_bytes = config.memory_limit_mb * 1024 * 1024
        _set_resource_limit(resource, resource.RLIMIT_AS, memory_bytes)
        cpu_limit = max(1, config.timeout_seconds)
        _set_resource_limit(resource, resource.RLIMIT_CPU, cpu_limit)
        output_limit = config.output_limit_mb * 1024 * 1024
        _set_resource_limit(resource, resource.RLIMIT_FSIZE, output_limit)

    return _apply


def _set_resource_limit(resource: Any, limit_kind: int, value: int) -> None:
    try:
        _, hard_limit = resource.getrlimit(limit_kind)
    except OSError, ValueError:
        return

    target_soft = value
    if hard_limit != resource.RLIM_INFINITY:
        target_soft = min(value, hard_limit)

    try:
        resource.setrlimit(limit_kind, (target_soft, hard_limit))
    except OSError, ValueError:
        return

    if hard_limit in (resource.RLIM_INFINITY, target_soft):
        try:
            resource.setrlimit(limit_kind, (target_soft, target_soft))
        except OSError, ValueError:
            return
