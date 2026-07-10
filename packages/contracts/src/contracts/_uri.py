"""Small URI helpers for paths that may be local filesystem paths *or* remote URIs.

Settings data-location fields are plain ``str`` so they can hold either a local path
(``/home/.../data/NWP``) or a remote URI (``s3://bucket/NWP``). ``pathlib.Path`` mangles
the latter (``Path("s3://b/a") / "c"`` drops the scheme), so joins route through here.

The existence/parent helpers below give the asset IO layer a single local-or-remote-aware
call for the two things it does around every Delta/parquet write: make sure the parent
directory exists (a no-op on object stores, which have no directories) and check whether a
table/object is already there. Remote calls go through delta-rs / obstore with the caller's
``storage_options`` so the same code path serves both a local ``data_path`` and an ``s3://`` one.
"""

import posixpath
from pathlib import Path
from typing import Final, TypedDict
from urllib.parse import urlparse

import obstore
from deltalake import DeltaTable

from contracts.typing_utils import typeddict_to_dict

_SCHEME_SEP: Final[str] = "://"


class ObjectStoreOptions(TypedDict, total=False):
    """object_store options for the managed data tables — the shared ``aws_*`` aliases understood
    by delta-rs, Polars, and obstore alike, so one value feeds every IO site.

    Authored as a ``TypedDict`` (rather than a bare ``dict[str, str]``) so ``ty`` checks every key
    where it is written — see ``Settings.storage_options``. Widen it to the plain ``dict`` the IO
    libraries expect with ``typeddict_to_dict`` at each call boundary. Empty on AWS (object_store
    auto-discovers the IAM-role credentials and region) and for a local ``data_path``.
    """

    aws_endpoint_url: str
    aws_allow_http: str
    aws_access_key_id: str
    aws_secret_access_key: str
    aws_region: str


def is_remote_uri(uri: str) -> bool:
    """Return whether ``uri`` carries a URI scheme (e.g. ``s3://``) rather than a local path."""
    return _SCHEME_SEP in uri


def uri_join(base: str, *parts: str) -> str:
    """Join ``parts`` onto ``base`` with local- or remote-aware semantics.

    Local bases join through ``pathlib`` (yielding an absolute path string); remote,
    scheme-bearing bases join posix-style so ``"s3://bucket/a"`` + ``"b"`` stays
    ``"s3://bucket/a/b"`` instead of being mangled by ``Path.__truediv__``.
    """
    if is_remote_uri(base):
        return posixpath.join(base.rstrip("/"), *parts)
    return str(Path(base).joinpath(*parts))


def ensure_local_parent(uri: str) -> None:
    """Create the parent directory of a *local* ``uri``; a no-op for a remote URI.

    Local filesystems need a table/file's parent directory to exist before a write; object
    stores (``s3://``) have no directories — a write creates the key's prefix implicitly — so
    there is nothing to do and this returns immediately.
    """
    if is_remote_uri(uri):
        return
    Path(uri).parent.mkdir(parents=True, exist_ok=True)


def delta_table_exists(uri: str, storage_options: ObjectStoreOptions | None = None) -> bool:
    """Return whether a Delta table already exists at ``uri`` (local path or remote URI).

    Wraps ``DeltaTable.is_deltatable``, which inspects the ``_delta_log`` through delta-rs'
    object_store and so works identically for a local path and an ``s3://`` URI given the
    matching ``storage_options``. Replaces ``Path(uri).exists()`` at the write-guard sites,
    which would raise on a remote URI.
    """
    return DeltaTable.is_deltatable(uri, storage_options=typeddict_to_dict(storage_options) or {})


def object_exists(uri: str, storage_options: ObjectStoreOptions | None = None) -> bool:
    """Return whether a single object/file at ``uri`` exists (local file or remote object).

    For a local ``uri`` this is ``Path.exists()``; for a remote URI it issues an object-store
    ``head`` via obstore, so it works for a plain file (e.g. a ``.parquet``) that is not a Delta
    table. Use ``delta_table_exists`` for Delta tables.
    """
    if not is_remote_uri(uri):
        return Path(uri).exists()
    parsed = urlparse(uri)
    store = obstore.store.S3Store(parsed.netloc, config=typeddict_to_dict(storage_options) or {})
    try:
        obstore.head(store, parsed.path.lstrip("/"))
    except FileNotFoundError:
        return False
    return True
