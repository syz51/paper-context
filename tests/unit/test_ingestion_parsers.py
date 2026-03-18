from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest

from paper_context.ingestion import parser_isolation, parsers
from paper_context.ingestion.types import (
    ParsedDocument,
    ParsedParagraph,
    ParsedSection,
    ParserArtifact,
    ParserResult,
)

pytestmark = pytest.mark.unit


@dataclass(frozen=True)
class _FakeProv:
    page_no: int
    charspan: tuple[int, int]


class _FakeLabel:
    def __init__(self, value: str) -> None:
        self.value = value


class _FakeTitleItem:
    def __init__(self, text: str, prov: list[_FakeProv] | None = None) -> None:
        self.text = text
        self.prov = prov


class _FakeSectionHeaderItem:
    def __init__(self, text: str, level: int, prov: list[_FakeProv] | None = None) -> None:
        self.text = text
        self.level = level
        self.prov = prov


class _FakeTextItem:
    def __init__(self, text: str, label: str, prov: list[_FakeProv] | None = None) -> None:
        self.text = text
        self.label = _FakeLabel(label)
        self.prov = prov


class _FakeCaptionRef:
    def __init__(self, text: str) -> None:
        self._text = text

    def resolve(self, document: object) -> object:
        return SimpleNamespace(text=self._text)


class _FakeDataFrame:
    def __init__(self, columns: list[object], rows: list[tuple[object, ...]]) -> None:
        self.columns = columns
        self._rows = rows

    def itertuples(self, index: bool = False, name: str | None = None):
        return iter(self._rows)


class _FakeTableItem:
    def __init__(
        self,
        dataframe: _FakeDataFrame,
        *,
        prov: list[_FakeProv] | None = None,
        captions: list[_FakeCaptionRef] | None = None,
    ) -> None:
        self._dataframe = dataframe
        self.prov = prov
        self.captions = captions or []

    def export_to_dataframe(self, document: object) -> _FakeDataFrame:
        return self._dataframe


class _FakeDocument:
    def __init__(self, items: list[object]) -> None:
        self._items = items

    def iterate_items(self):
        for item in self._items:
            yield item, None

    def export_to_dict(self) -> dict[str, object]:
        return {"items": len(self._items)}

    def save_as_json(self, filename: str | Path, *args, **kwargs) -> None:
        Path(filename).write_text(f'{{"items": {len(self._items)}}}', encoding="utf-8")


@dataclass(frozen=True)
class _FakeConversion:
    status: object
    document: _FakeDocument


class _FakeDoclingConverter:
    def __init__(
        self,
        conversion: _FakeConversion | None = None,
        error: Exception | None = None,
    ) -> None:
        self.conversion = conversion
        self.error = error
        self.calls: list[tuple[str, bool]] = []
        self.format_options: dict[object, object] | None = None

    def convert(self, document_stream: object, raises_on_error: bool = False) -> _FakeConversion:
        self.calls.append((getattr(document_stream, "name", ""), raises_on_error))
        if self.error is not None:
            raise self.error
        if self.conversion is None:
            raise AssertionError("missing fake conversion")
        return self.conversion


class _FakePdfPipelineOptions:
    def __init__(self) -> None:
        self.do_ocr = None
        self.do_table_structure = None
        self.force_backend_text = None
        self.generate_page_images = None


class _FakePdfFormatOption:
    def __init__(self, *, pipeline_options: _FakePdfPipelineOptions) -> None:
        self.pipeline_options = pipeline_options


class _FakeInputFormat:
    PDF = "pdf"


class _FakePdfPage:
    def __init__(self, text: str | None, tables: list[list[list[object]]] | None = None) -> None:
        self._text = text
        self._tables = tables or []
        self.close_calls = 0

    def extract_text(self) -> str | None:
        return self._text

    def extract_tables(self) -> list[list[list[object]]]:
        return self._tables

    def close(self) -> None:
        self.close_calls += 1


class _FakePdfContext:
    def __init__(self, pages: list[_FakePdfPage]) -> None:
        self.pages = pages

    def __enter__(self) -> _FakePdfContext:
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False


def _install_docling_fakes(
    monkeypatch: pytest.MonkeyPatch,
    *,
    conversion: _FakeConversion | None = None,
    error: Exception | None = None,
) -> _FakeDoclingConverter:
    fake_converter = _FakeDoclingConverter(conversion=conversion, error=error)

    def _factory(format_options: dict[object, object]) -> _FakeDoclingConverter:
        fake_converter.format_options = format_options
        return fake_converter

    monkeypatch.setattr(parsers, "PdfPipelineOptions", _FakePdfPipelineOptions)
    monkeypatch.setattr(parsers, "PdfFormatOption", _FakePdfFormatOption)
    monkeypatch.setattr(parsers, "InputFormat", _FakeInputFormat)
    monkeypatch.setattr(parsers, "DocumentConverter", _factory)
    monkeypatch.setattr(parsers, "TitleItem", _FakeTitleItem)
    monkeypatch.setattr(parsers, "SectionHeaderItem", _FakeSectionHeaderItem)
    monkeypatch.setattr(parsers, "TextItem", _FakeTextItem)
    monkeypatch.setattr(parsers, "TableItem", _FakeTableItem)
    return fake_converter


def test_parser_helpers_cover_basic_paths() -> None:
    no_prov = SimpleNamespace(prov=None)
    prov = [
        _FakeProv(page_no=3, charspan=(12, 18)),
        _FakeProv(page_no=1, charspan=(0, 4)),
    ]

    assert parsers._page_bounds(no_prov) == (None, None)
    assert parsers._page_bounds(SimpleNamespace(prov=prov)) == (1, 3)
    assert parsers._provenance_offsets(no_prov) is None
    assert parsers._provenance_offsets(SimpleNamespace(prov=prov)) == {
        "charspans": [[12, 18], [0, 4]],
        "pages": [3, 1],
    }
    assert parsers._coerce_text(None) == ""
    assert parsers._coerce_text("  trimmed  ") == "trimmed"
    assert (
        parsers._estimate_metadata_confidence(
            title=None,
            authors=[],
            abstract=None,
            publication_year=None,
            has_headings=False,
        )
        == 0.2
    )
    assert (
        parsers._estimate_metadata_confidence(
            title="Title",
            authors=["Ada"],
            abstract="Abstract",
            publication_year=2024,
            has_headings=True,
        )
        == 0.95
    )
    assert (
        parsers._classify_docling_result(
            parsers.ParsedDocument(
                title=None,
                authors=[],
                abstract=None,
                publication_year=None,
                metadata_confidence=None,
                sections=[
                    parsers.ParsedSection(
                        key="root",
                        heading=None,
                        heading_path=[],
                        level=0,
                        page_start=None,
                        page_end=None,
                    )
                ],
                tables=[],
                references=[],
            )
        )
        == "fail"
    )
    assert (
        parsers._classify_docling_result(
            parsers.ParsedDocument(
                title=None,
                authors=[],
                abstract=None,
                publication_year=None,
                metadata_confidence=None,
                sections=[
                    parsers.ParsedSection(
                        key="root",
                        heading=None,
                        heading_path=[],
                        level=0,
                        page_start=None,
                        page_end=None,
                        paragraphs=[ParsedParagraph("body", None, None)],
                    )
                ],
                tables=[],
                references=[],
            )
        )
        == "degraded"
    )
    assert (
        parsers._classify_docling_result(
            parsers.ParsedDocument(
                title=None,
                authors=[],
                abstract=None,
                publication_year=None,
                metadata_confidence=None,
                sections=[
                    parsers.ParsedSection(
                        key="intro",
                        heading="Intro",
                        heading_path=["Intro"],
                        level=1,
                        page_start=1,
                        page_end=1,
                        paragraphs=[ParsedParagraph("body", 1, 1)],
                    )
                ],
                tables=[],
                references=[],
            )
        )
        == "pass"
    )
    assert parsers._heading_level_from_text("1.2.3 Methods") == 3
    assert parsers._heading_level_from_text("Methods") == 1
    assert parsers._looks_like_heading("References") is True
    assert parsers._looks_like_heading("1.2 Methods") is True
    assert parsers._looks_like_heading("ALL CAPS") is True
    assert parsers._looks_like_heading("Deep Learning Methods") is True
    assert parsers._looks_like_heading("This is a sentence.") is False
    assert (
        parsers._looks_like_heading(
            "one two three four five six seven eight nine ten eleven twelve thirteen"
        )
        is False
    )
    assert parsers._looks_like_heading("") is False
    assert parsers._infer_publication_year("Published in 2024.") == 2024
    assert parsers._infer_publication_year("No year here") is None
    assert parsers._build_heading_path([(1, "s1", "Intro"), (2, "s2", "Methods")]) == [
        "Intro",
        "Methods",
    ]


def test_docling_parser_initializes_pipeline_options(monkeypatch: pytest.MonkeyPatch) -> None:
    converter = _install_docling_fakes(monkeypatch)

    parser = parsers.DoclingPdfParser()

    assert isinstance(parser._converter, _FakeDoclingConverter)
    assert converter.format_options is not None
    option = converter.format_options[_FakeInputFormat.PDF]
    assert isinstance(option, _FakePdfFormatOption)
    assert option.pipeline_options.do_ocr is False


def test_parser_worker_round_trips_parser_results(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    expected = ParserResult(
        gate_status="pass",
        parsed_document=ParsedDocument(
            title="Isolated parser paper",
            authors=["Ada Lovelace"],
            abstract="A parser isolation fixture.",
            publication_year=2024,
            metadata_confidence=0.9,
            sections=[
                ParsedSection(
                    key="intro",
                    heading="Introduction",
                    heading_path=["Introduction"],
                    level=1,
                    page_start=1,
                    page_end=1,
                    paragraphs=[ParsedParagraph(text="hello", page_start=1, page_end=1)],
                )
            ],
            tables=[],
            references=[],
        ),
        artifact=ParserArtifact(
            artifact_type="docling_parse",
            parser="docling",
            filename="docling.json",
            content=b"{}",
        ),
    )
    fake_parser = SimpleNamespace(parse=lambda filename, content=None, source_path=None: expected)
    monkeypatch.setattr(
        parser_isolation,
        "_create_inprocess_parser",
        lambda parser_name: fake_parser,
    )

    output_root = tmp_path / "parser-output"
    payload = parser_isolation.run_parser_worker(
        "docling",
        "paper.pdf",
        output_root,
        b"%PDF-1.4",
    )
    restored = parser_isolation._payload_to_parser_result(
        parser_name="docling",
        payload=payload,
        output_root=output_root,
    )

    assert restored.gate_status == expected.gate_status
    assert restored.parsed_document is None
    assert restored.load_parsed_document() == expected.parsed_document
    assert restored.parsed_document == expected.parsed_document
    assert restored.artifact.artifact_type == expected.artifact.artifact_type
    assert restored.artifact.parser == expected.artifact.parser
    assert restored.artifact.filename == expected.artifact.filename
    assert restored.artifact.content is None
    assert restored.artifact.content_path is not None
    assert restored.artifact.content_path.read_bytes() == b"{}"
    assert restored.artifact.cleanup_root == output_root
    parser_isolation._cleanup_output_root(restored.artifact.cleanup_root)


def test_subprocess_parser_returns_machine_readable_timeout_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd=args[0], timeout=5, stderr=b"timeout")

    monkeypatch.setattr(parser_isolation.subprocess, "run", _timeout)

    result = parser_isolation.SubprocessPdfParser("docling").parse("paper.pdf", b"%PDF-1.4")

    assert result.gate_status == "fail"
    assert result.failure_code == "docling_timeout"
    assert "timeout" in (result.failure_message or "")


def test_subprocess_parser_returns_machine_readable_launch_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _launch_failure(*args, **kwargs):
        raise subprocess.SubprocessError("resource limits unavailable")

    monkeypatch.setattr(parser_isolation.subprocess, "run", _launch_failure)

    result = parser_isolation.SubprocessPdfParser("docling").parse("paper.pdf", b"%PDF-1.4")

    assert result.gate_status == "fail"
    assert result.failure_code == "docling_subprocess_launch_failed"
    assert result.failure_message == "resource limits unavailable"


def test_parser_resource_limits_lower_soft_limit_before_hard_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rlim_infinity = 2**63 - 1
    state = {
        1: (rlim_infinity, rlim_infinity),
        2: (rlim_infinity, rlim_infinity),
        3: (rlim_infinity, rlim_infinity),
    }
    calls: list[tuple[int, tuple[int, int]]] = []

    class _FakeResource:
        RLIMIT_AS = 1
        RLIMIT_CPU = 2
        RLIMIT_FSIZE = 3
        RLIM_INFINITY = rlim_infinity

        @staticmethod
        def getrlimit(limit_kind: int) -> tuple[int, int]:
            return state[limit_kind]

        @staticmethod
        def setrlimit(limit_kind: int, limits: tuple[int, int]) -> None:
            calls.append((limit_kind, limits))
            _, current_hard = state[limit_kind]
            next_soft, next_hard = limits
            if next_hard != rlim_infinity and state[limit_kind][0] > next_hard:
                raise ValueError("current limit exceeds maximum limit")
            if current_hard != rlim_infinity and next_hard > current_hard:
                raise ValueError("cannot raise hard limit")
            state[limit_kind] = limits

    monkeypatch.setitem(sys.modules, "resource", _FakeResource)

    apply_limits = parser_isolation._parser_resource_limits(
        parser_isolation.ParserIsolationConfig(
            timeout_seconds=120,
            memory_limit_mb=2_048,
            output_limit_mb=32,
        )
    )

    assert apply_limits is not None
    apply_limits()

    assert calls[:2] == [
        (_FakeResource.RLIMIT_AS, (2_048 * 1024 * 1024, rlim_infinity)),
        (_FakeResource.RLIMIT_AS, (2_048 * 1024 * 1024, 2_048 * 1024 * 1024)),
    ]
    assert state[_FakeResource.RLIMIT_CPU] == (120, 120)
    assert state[_FakeResource.RLIMIT_FSIZE] == (32 * 1024 * 1024, 32 * 1024 * 1024)


def test_docling_parser_handles_success_and_structure(monkeypatch: pytest.MonkeyPatch) -> None:
    doc = _FakeDocument(
        [
            _FakeTitleItem(""),
            _FakeTitleItem("paper title"),
            _FakeSectionHeaderItem("Introduction", level=1, prov=[_FakeProv(1, (0, 11))]),
            _FakeTextItem(
                " ".join(["intro"] * 45),
                "body",
                prov=[_FakeProv(1, (0, 120)), _FakeProv(2, (0, 20))],
            ),
            _FakeTextItem("skip me", "page_header", prov=[_FakeProv(1, (0, 8))]),
            _FakeTableItem(
                _FakeDataFrame(
                    columns=[" A ", 2],
                    rows=[(" first ", None), (None, 3)],
                ),
                prov=[_FakeProv(1, (10, 20)), _FakeProv(2, (4, 7))],
                captions=[_FakeCaptionRef("  Table caption  ")],
            ),
            _FakeSectionHeaderItem("Methods", level=2, prov=[_FakeProv(2, (0, 7))]),
            _FakeTextItem("explicit citation 2024", "reference", prov=[_FakeProv(2, (0, 20))]),
            _FakeSectionHeaderItem("References", level=1, prov=[_FakeProv(3, (0, 10))]),
            _FakeTextItem("Smith 2023", "body", prov=[_FakeProv(3, (0, 10))]),
            _FakeSectionHeaderItem("Conclusion", level=1, prov=[_FakeProv(4, (0, 10))]),
            _FakeTextItem("closing paragraph text", "body", prov=[_FakeProv(4, (0, 20))]),
        ]
    )
    conversion = _FakeConversion(status=parsers.ConversionStatus.SUCCESS, document=doc)
    converter = _install_docling_fakes(monkeypatch, conversion=conversion)

    result = parsers.DoclingPdfParser().parse("paper.pdf", b"%PDF-1.4")

    assert converter.calls == [("paper.pdf", False)]
    assert result.gate_status == "pass"
    assert result.parsed_document is not None
    parsed = result.parsed_document
    assert parsed.title == "paper title"
    assert parsed.abstract == " ".join(["intro"] * 45)
    assert [section.heading for section in parsed.sections] == [
        "Introduction",
        "Methods",
        "References",
        "Conclusion",
    ]
    assert parsed.sections[1].parent_key == parsed.sections[0].key
    assert parsed.sections[1].heading_path == ["Introduction", "Methods"]
    assert len(parsed.tables) == 1
    assert parsed.tables[0].caption == "Table caption"
    assert parsed.tables[0].headers == ["A", "2"]
    assert parsed.tables[0].rows == [["first", ""], ["", "3"]]
    assert len(parsed.references) == 2
    assert parsed.references[0].raw_citation == "explicit citation 2024"
    assert parsed.references[0].publication_year == 2024
    assert parsed.references[1].raw_citation == "Smith 2023"
    assert parsed.references[1].publication_year == 2023
    assert result.warnings == []
    assert result.failure_code is None
    assert result.failure_message is None


def test_docling_parser_accepts_source_path(monkeypatch: pytest.MonkeyPatch) -> None:
    conversion = _FakeConversion(
        status=parsers.ConversionStatus.SUCCESS,
        document=_FakeDocument(
            [
                _FakeTitleItem("paper title"),
                _FakeSectionHeaderItem("Introduction", level=1, prov=[_FakeProv(1, (0, 11))]),
                _FakeTextItem("body text", "body", prov=[_FakeProv(1, (0, 9))]),
            ]
        ),
    )
    converter = _install_docling_fakes(monkeypatch, conversion=conversion)

    result = parsers.DoclingPdfParser().parse("paper.pdf", source_path=Path("/tmp/paper.pdf"))

    assert converter.calls == [("paper.pdf", False)]
    assert result.gate_status == "pass"
    assert result.parsed_document is not None


def test_docling_parser_marks_degraded_documents_without_headings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    doc = _FakeDocument(
        [
            _FakeTitleItem("paper title"),
            _FakeTextItem("short intro", "body", prov=[_FakeProv(1, (0, 10))]),
            _FakeTextItem(
                " ".join(["abstract"] * 45),
                "body",
                prov=[_FakeProv(1, (11, 140))],
            ),
        ]
    )
    conversion = _FakeConversion(status=parsers.ConversionStatus.PARTIAL_SUCCESS, document=doc)
    _install_docling_fakes(monkeypatch, conversion=conversion)

    result = parsers.DoclingPdfParser().parse("degraded.pdf", b"%PDF-1.4")

    assert result.gate_status == "degraded"
    assert result.parsed_document is not None
    assert result.parsed_document.sections[0].heading is None
    assert result.parsed_document.sections[0].paragraphs[0].text == "short intro"
    assert result.warnings == ["reduced_structure_confidence"]
    assert result.failure_code is None


def test_parser_worker_skips_parsed_document_handoff_for_degraded_results(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    degraded = ParserResult(
        gate_status="degraded",
        parsed_document=ParsedDocument(
            title="Degraded parser paper",
            authors=[],
            abstract=None,
            publication_year=2024,
            metadata_confidence=0.4,
            sections=[
                ParsedSection(
                    key="root",
                    heading=None,
                    heading_path=[],
                    level=0,
                    page_start=1,
                    page_end=1,
                    paragraphs=[ParsedParagraph(text="body", page_start=1, page_end=1)],
                )
            ],
            tables=[],
            references=[],
        ),
        artifact=ParserArtifact(
            artifact_type="docling_parse",
            parser="docling",
            filename="docling.json",
            content=b"{}",
        ),
        warnings=["reduced_structure_confidence"],
    )
    fake_parser = SimpleNamespace(parse=lambda filename, content=None, source_path=None: degraded)
    monkeypatch.setattr(
        parser_isolation,
        "_create_inprocess_parser",
        lambda parser_name: fake_parser,
    )

    output_root = tmp_path / "parser-output"
    payload = parser_isolation.run_parser_worker(
        "docling",
        "paper.pdf",
        output_root,
        b"%PDF-1.4",
    )
    restored = parser_isolation._payload_to_parser_result(
        parser_name="docling",
        payload=payload,
        output_root=output_root,
    )

    assert restored.gate_status == "degraded"
    assert restored.parsed_document is None
    assert restored.load_parsed_document() is None
    assert not (output_root / "parsed_document.json").exists()
    parser_isolation._cleanup_output_root(restored.artifact.cleanup_root)


def test_docling_parser_returns_fail_for_bad_conversion_status(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    doc = _FakeDocument([_FakeTitleItem("paper title")])
    conversion = _FakeConversion(status="unsupported", document=doc)
    _install_docling_fakes(monkeypatch, conversion=conversion)

    result = parsers.DoclingPdfParser().parse("broken.pdf", b"%PDF-1.4")

    assert result.gate_status == "fail"
    assert result.parsed_document is None
    assert result.failure_code == "docling_conversion_failed"
    assert result.failure_message == "Docling could not parse the PDF into a stable document."


def test_docling_parser_returns_fail_for_conversion_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_docling_fakes(monkeypatch, error=RuntimeError("boom"))

    result = parsers.DoclingPdfParser().parse("broken.pdf", b"%PDF-1.4")

    assert result.gate_status == "fail"
    assert result.parsed_document is None
    assert result.failure_code == "docling_conversion_failed"
    assert result.failure_message == "boom"
    assert result.artifact.filename == "docling-error.json"


def test_pdfplumber_parser_handles_structured_documents(monkeypatch: pytest.MonkeyPatch) -> None:
    pages = [
        _FakePdfPage(
            "\n".join(
                [
                    "paper title",
                    "Introduction",
                    " ".join(["intro"] * 45),
                    "1.1 Methods",
                    "methods text is described here.",
                    "REFERENCES",
                    "Smith 2024.",
                ]
            ),
            tables=[
                [
                    [" a ", 2],
                    [" first ", None],
                    [None, 3],
                ]
            ],
        ),
        _FakePdfPage(
            "\n".join(
                [
                    "2 Conclusion",
                    "closing paragraph text.",
                ]
            )
        ),
    ]
    monkeypatch.setattr(parsers.pdfplumber, "open", lambda content: _FakePdfContext(pages))

    result = parsers.PdfPlumberPdfParser().parse("paper.pdf", b"%PDF-1.4")

    assert result.gate_status == "pass"
    assert result.parsed_document is not None
    parsed = result.parsed_document
    assert parsed.title == "paper title"
    assert parsed.abstract == " ".join(["intro"] * 45)
    assert [section.heading for section in parsed.sections] == [
        None,
        "Introduction",
        "1.1 Methods",
        "REFERENCES",
        "2 Conclusion",
    ]
    assert parsed.sections[2].parent_key == parsed.sections[1].key
    assert parsed.tables[0].caption == "Table 1"
    assert parsed.tables[0].headers == ["a", "2"]
    assert parsed.tables[0].rows == [["first", ""], ["", "3"]]
    assert [reference.raw_citation for reference in parsed.references] == ["Smith 2024."]
    assert parsed.references[0].publication_year == 2024
    assert result.warnings == []
    assert result.failure_code is None
    assert [page.close_calls for page in pages] == [1, 1]


def test_pdfplumber_parser_accepts_source_path(monkeypatch: pytest.MonkeyPatch) -> None:
    pages = [_FakePdfPage("paper title\nIntroduction\nbody text")]
    monkeypatch.setattr(parsers.pdfplumber, "open", lambda content: _FakePdfContext(pages))

    result = parsers.PdfPlumberPdfParser().parse(
        "paper.pdf",
        source_path=Path("/tmp/paper.pdf"),
    )

    assert result.gate_status == "pass"
    assert result.parsed_document is not None
    assert pages[0].close_calls == 1


def test_pdfplumber_parser_marks_degraded_documents_without_headings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pages = [
        _FakePdfPage(
            "\n".join(
                [
                    "paper title",
                    "short intro",
                    " ".join(["body"] * 45),
                ]
            )
        )
    ]
    monkeypatch.setattr(parsers.pdfplumber, "open", lambda content: _FakePdfContext(pages))

    result = parsers.PdfPlumberPdfParser().parse("plain.pdf", b"%PDF-1.4")

    assert result.gate_status == "pass"
    assert result.parsed_document is not None
    assert result.parsed_document.sections[0].heading is None
    assert result.warnings == ["reduced_structure_confidence"]


def test_pdfplumber_parser_returns_fail_for_empty_structured_document(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pages = [_FakePdfPage("\n".join(["Introduction", "Methods"]))]
    monkeypatch.setattr(parsers.pdfplumber, "open", lambda content: _FakePdfContext(pages))

    result = parsers.PdfPlumberPdfParser().parse("empty.pdf", b"%PDF-1.4")

    assert result.gate_status == "fail"
    assert result.parsed_document is None
    assert result.failure_code == "pdfplumber_structure_failed"
    assert result.failure_message == "pdfplumber could not recover stable passages with provenance."


def test_pdfplumber_parser_returns_fail_for_open_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _raise(_: object) -> _FakePdfContext:
        raise RuntimeError("boom")

    monkeypatch.setattr(parsers.pdfplumber, "open", _raise)

    result = parsers.PdfPlumberPdfParser().parse("broken.pdf", b"%PDF-1.4")

    assert result.gate_status == "fail"
    assert result.parsed_document is None
    assert result.failure_code == "pdfplumber_conversion_failed"
    assert result.failure_message == "boom"
    assert result.artifact.filename == "pdfplumber-error.json"
