from __future__ import annotations

import io
from typing import TYPE_CHECKING

import boto3
import pandas as pd
import structlog
from botocore.exceptions import ClientError

from financial_pipeline.storage.base import BaseStorage

if TYPE_CHECKING:
    from mypy_boto3_s3 import S3Client

log = structlog.get_logger()


class S3Storage(BaseStorage):
    """S3-backed storage — reads/writes Parquet objects under a key prefix."""

    def __init__(
        self,
        bucket: str,
        prefix: str = "",
        region: str = "ap-south-1",
        aws_access_key_id: str | None = None,
        aws_secret_access_key: str | None = None,
    ) -> None:
        self._bucket = bucket
        self._prefix = prefix.rstrip("/")
        self._log = log.bind(backend="s3", bucket=bucket)
        self._client: S3Client = boto3.client(
            "s3",
            region_name=region,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
        )

    # ------------------------------------------------------------------
    # BaseStorage interface
    # ------------------------------------------------------------------

    def save(self, df: pd.DataFrame, table: str) -> int:
        key = self._key(f"{table}.parquet")
        buf = io.BytesIO()
        df.to_parquet(buf, index=False, engine="pyarrow")
        buf.seek(0)
        self._client.put_object(Bucket=self._bucket, Key=key, Body=buf.getvalue())
        self._log.info("s3.saved", key=key, rows=len(df))
        return len(df)

    def load(self, table: str) -> pd.DataFrame:
        key = self._key(f"{table}.parquet")
        try:
            obj = self._client.get_object(Bucket=self._bucket, Key=key)
        except ClientError as exc:
            if exc.response["Error"]["Code"] == "NoSuchKey":
                raise FileNotFoundError(f"s3://{self._bucket}/{key} not found") from exc
            raise
        df = pd.read_parquet(io.BytesIO(obj["Body"].read()), engine="pyarrow")
        self._log.info("s3.loaded", key=key, rows=len(df))
        return df

    # ------------------------------------------------------------------
    # Extra helpers used by the AMFI fetch script
    # ------------------------------------------------------------------

    def exists(self, key_suffix: str) -> bool:
        """Return True if the key already exists in S3."""
        try:
            self._client.head_object(Bucket=self._bucket, Key=self._key(key_suffix))
            return True
        except ClientError as exc:
            if exc.response["Error"]["Code"] in ("404", "NoSuchKey"):
                return False
            raise

    def put_raw(self, key_suffix: str, data: bytes, content_type: str = "text/plain") -> str:
        """Upload raw bytes. Returns the full S3 key."""
        key = self._key(key_suffix)
        self._client.put_object(
            Bucket=self._bucket,
            Key=key,
            Body=data,
            ContentType=content_type,
        )
        self._log.info("s3.put_raw", key=key, bytes=len(data))
        return key

    def put_csv(self, key_suffix: str, df: pd.DataFrame) -> str:
        """Upload a DataFrame as CSV. Returns the full S3 key."""
        key = self._key(key_suffix)
        buf = io.StringIO()
        df.to_csv(buf, index=False)
        self._client.put_object(
            Bucket=self._bucket,
            Key=key,
            Body=buf.getvalue().encode(),
            ContentType="text/csv",
        )
        self._log.info("s3.put_csv", key=key, rows=len(df))
        return key

    # ------------------------------------------------------------------

    def _key(self, suffix: str) -> str:
        return f"{self._prefix}/{suffix}" if self._prefix else suffix
