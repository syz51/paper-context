from __future__ import annotations

from typing import Protocol

from .types import EnrichmentResult, ParsedDocument


class MetadataEnricher(Protocol):
    def enrich(self, parsed_document: ParsedDocument) -> EnrichmentResult:
        """Return any metadata improvements and warnings for the parsed document."""
        ...


class NullMetadataEnricher:
    def enrich(self, parsed_document: ParsedDocument) -> EnrichmentResult:
        return EnrichmentResult()
