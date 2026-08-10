"""Reading NGED's telemetry JSON from S3 and writing it to Delta Lake."""

from .storage import upsert_metadata

__all__ = ["upsert_metadata"]
