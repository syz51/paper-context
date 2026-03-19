from __future__ import annotations

from abc import abstractmethod
from typing import Protocol

from .types import EnrichmentResult, ParsedDocument


class MetadataEnricher(Protocol):
    @abstractmethod
    def enrich(self, parsed_document: ParsedDocument) -> EnrichmentResult:
        """Return any metadata improvements and warnings for the parsed document."""
        raise NotImplementedError


class NullMetadataEnricher:
    """Phase-1 default enricher: explicitly no-op until enrichment ships beyond sign-off."""

    def enrich(self, parsed_document: ParsedDocument) -> EnrichmentResult:
        return EnrichmentResult()
