from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from paper_context.ingestion import parsers
from paper_context.ingestion.types import ParsedParagraph

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

    def extract_text(self) -> str | None:
        return self._text

    def extract_tables(self) -> list[list[list[object]]]:
        return self._tables


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
    assert option.pipeline_options.do_table_structure is True
    assert option.pipeline_options.force_backend_text is True
    assert option.pipeline_options.generate_page_images is True


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
