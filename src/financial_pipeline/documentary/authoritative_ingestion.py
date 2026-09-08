"""Ingest authoritative fund documents into the existing document pipeline."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import PurePosixPath
from urllib.parse import urlparse

import boto3
import httpx

from financial_pipeline.config import settings
from financial_pipeline.storage.document_repo import DocumentRepository


@dataclass(frozen=True)
class AuthoritativeFundDocument:
    scheme_family_key: str
    scheme_code: str | None
    provider: str
    source: str
    source_url: str
    authoritative_domain: str
    document_type: str
    file_name: str
    title: str | None = None
    publication_date: str | None = None


class AuthoritativeFundDocumentIngestor:
    """Download trusted scheme documents, persist raw bytes, and register identity."""

    _ALLOWED_TYPES = {
        "fund_prospectus",
        "scheme_information_document",
        "key_information_memorandum",
        "fund_fact_sheet",
        "fund_strategy_document",
        "portfolio_disclosure",
        "regulatory_filing",
        "annual_report",
    }

    def __init__(self, repository: DocumentRepository) -> None:
        self._repository = repository
        self._s3 = boto3.client("s3", region_name=settings.aws_region)

    def ingest(self, document: AuthoritativeFundDocument) -> dict:
        self._validate(document)

        response = httpx.get(
            document.source_url,
            timeout=settings.request_timeout,
            follow_redirects=True,
        )
        response.raise_for_status()
        body = response.content
        if not body:
            raise ValueError("authoritative document download returned an empty body")

        file_hash = hashlib.sha256(body).hexdigest()
        extension = PurePosixPath(document.file_name).suffix.lstrip(".").lower() or "pdf"
        s3_key = (
            "bronze/fund_documents/"
            f"{document.scheme_family_key.replace(' ', '_')}/"
            f"{document.document_type}/{file_hash[:16]}-{document.file_name}"
        )
        if not settings.s3_bucket:
            raise RuntimeError("S3_BUCKET is required for authoritative document ingestion")

        self._s3.put_object(
            Bucket=settings.s3_bucket,
            Key=s3_key,
            Body=body,
            ContentType=response.headers.get("content-type", "application/octet-stream"),
            Metadata={
                "scheme_family_key": document.scheme_family_key,
                "document_type": document.document_type,
                "provider": document.provider,
            },
        )

        document_id, action = self._repository.upsert_metadata(
            source=document.source,
            provider=document.provider,
            document_type=document.document_type,
            s3_raw_key=s3_key,
            original_url=document.source_url,
            file_name=document.file_name,
            file_size_bytes=len(body),
            file_hash=file_hash,
            title=document.title,
            file_type=extension,
        )
        self._repository.bind_document_identity(
            document_id=document_id,
            scheme_family_key=document.scheme_family_key,
            scheme_code=document.scheme_code,
            resolution_source="authoritative_ingestion",
            resolution_confidence=1.0,
        )
        return {
            "document_id": document_id,
            "action": action,
            "scheme_family_key": document.scheme_family_key,
            "document_type": document.document_type,
            "s3_raw_key": s3_key,
        }

    def _validate(self, document: AuthoritativeFundDocument) -> None:
        if document.document_type not in self._ALLOWED_TYPES:
            raise ValueError(f"unsupported authoritative document type: {document.document_type}")
        parsed = urlparse(document.source_url)
        if parsed.scheme != "https" or not parsed.netloc:
            raise ValueError("authoritative source URL must use HTTPS")
        allowed = document.authoritative_domain.lower().strip()
        hostname = (parsed.hostname or "").lower()
        if not allowed or not (
            hostname == allowed or hostname.endswith("." + allowed)
        ):
            raise ValueError(
                f"source URL host {hostname!r} does not match authoritative domain {allowed!r}"
            )
        if not document.scheme_family_key.strip():
            raise ValueError("scheme_family_key is required")
        if not document.provider.strip():
            raise ValueError("provider is required")
