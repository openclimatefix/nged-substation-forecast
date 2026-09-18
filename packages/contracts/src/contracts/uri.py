"""Small URI helpers for paths that may be local filesystem paths *or* remote URIs.

Settings data-location fields are plain ``str`` so they can hold either a local path
(``/home/.../data/NWP``) or a remote URI (``s3://bucket/NWP``). ``pathlib.Path`` mangles a remote
URI (``Path("s3://b/a") / "c"`` collapses the double slash after the scheme and yields
``"s3:/b/a/c"``), so joins route through here.

The asset IO layer, the code that writes our Delta tables and parquet files, takes two steps
around every Delta/parquet write: it makes sure the parent directory exists, and it checks
whether a table or object is already there. The existence/parent helpers below give that layer a
single local-or-remote-aware call for those two steps. Making the parent directory is a no-op on
object stores, which have no directories. Remote calls go through delta-rs / obstore with the
caller's ``storage_options`` so the same code path serves both a local data-path root and an
``s3://`` one.
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
    """object_store options for the managed data tables.

    These are the shared ``aws_*`` aliases understood by delta-rs, Polars, and obstore alike, so
    one value feeds every IO site.

    Authored as a ``TypedDict`` (rather than a bare ``dict[str, str]``) so ``ty`` checks every
    key where it is written — see ``Settings.storage_options``. Widen it to the plain ``dict``
    the IO libraries expect with ``typeddict_to_dict`` at each call boundary. The mapping is
    empty on AWS, where object_store auto-discovers the identity-and-access-management (IAM)
    role's credentials and region. The mapping is empty for a local data-path root too.
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

    A local base joins through ``pathlib``, which yields an absolute path string. A remote,
    scheme-bearing base joins posix-style, so ``"s3://bucket/a"`` + ``"b"`` stays
    ``"s3://bucket/a/b"`` rather than being mangled by ``Path.__truediv__``.
    """
    if is_remote_uri(base):
        return posixpath.join(base.rstrip("/"), *parts)
    return str(Path(base).joinpath(*parts))


def if_local_path_then_make_parent_dir(uri: str) -> None:
    """Create the parent directory of a *local* ``uri``; a no-op for a remote URI.

    Local filesystems need a table or file's parent directory to exist before a write. Object
    stores (``s3://``) have no directories, and a write creates the key's prefix implicitly.
    There is therefore nothing to do for a remote URI, and this function returns immediately.
    """
    if is_remote_uri(uri):
        return
    Path(uri).parent.mkdir(parents=True, exist_ok=True)


def delta_table_exists(uri: str, storage_options: ObjectStoreOptions | None = None) -> bool:
    """Return whether a Delta table already exists at ``uri`` (local path or remote URI).

    Wraps ``DeltaTable.is_deltatable``, which inspects the ``_delta_log`` through delta-rs'
    object_store and so works identically for a local path and an ``s3://`` URI given the
    matching ``storage_options``. Use this function instead of ``Path(uri).exists()`` at the
    write-guard sites. There, ``pathlib`` reads ``s3://bucket/key`` as a relative local path that
    happens not to exist. ``Path.exists()`` therefore returns ``False`` without raising, and the
    guard concludes the table is absent.
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
