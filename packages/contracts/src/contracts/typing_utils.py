"""Reusable typing shims.

The shims below are domain-agnostic helpers that reconcile the expressive in-code types we prefer
with the plainer types third-party APIs annotate.

The shims are kept generic and separate from any one domain so they can be shared across the
workspace (and promoted to their own package if they grow). The first inhabitant bridges the gap
between a ``TypedDict`` and the ``dict[str, str]`` that libraries such as delta-rs, Polars, and
obstore expect. We use a ``TypedDict`` so that ``ty``, the type checker this repo runs, checks
every key at the point it is written. A ``TypedDict`` is deliberately *not* assignable to any
``dict[..]`` type, because a ``dict`` permits destructive operations like ``clear()``. The
``TypedDict`` therefore has to be widened explicitly at the boundary.
"""

from collections.abc import Mapping
from typing import cast


def typeddict_to_dict(td: Mapping[str, object] | None) -> dict[str, str] | None:
    """Widen a string-valued ``TypedDict`` to the plain ``dict[str, str]`` libraries expect.

    Author options as a ``TypedDict``, so that ``ty`` catches a mistyped key where the key is
    written. Then call this function at the last moment, at the boundary to a third-party API
    annotated ``dict[str, str]`` (or ``dict[str, Any]``). ``None`` passes straight through so a
    call site can forward an optional value unchanged.

    The widening is a zero-copy ``cast``: the caller is responsible for the ``TypedDict``'s
    values actually being ``str`` (they are, for every current caller).
    """
    return None if td is None else cast("dict[str, str]", td)
