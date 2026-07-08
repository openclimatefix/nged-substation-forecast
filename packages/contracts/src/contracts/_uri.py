"""Small URI helpers for paths that may be local filesystem paths *or* remote URIs.

Settings data-location fields are plain ``str`` so they can hold either a local path
(``/home/.../data/NWP``) or a remote URI (``s3://bucket/NWP``). ``pathlib.Path`` mangles
the latter (``Path("s3://b/a") / "c"`` drops the scheme), so joins route through here.
"""

import posixpath
from pathlib import Path
from typing import Final

_SCHEME_SEP: Final[str] = "://"


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
