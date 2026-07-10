"""Reusable typing shims — domain-agnostic helpers that reconcile expressive in-code types
with the plainer types third-party APIs annotate.

Kept generic and separate from any one domain so they can be shared across the workspace (and
promoted to their own package if they grow). The first inhabitant bridges the gap between a
``TypedDict`` — which we use so ``ty`` checks every key at the point it is written — and the
``dict[str, str]`` that libraries such as delta-rs, Polars, and obstore expect: a ``TypedDict`` is
deliberately *not* assignable to any ``dict[..]`` type (a ``dict`` permits destructive operations
like ``clear()``), so it has to be widened explicitly at the boundary.
"""

from collections.abc import Mapping
from typing import cast


def typeddict_to_dict(td: Mapping[str, object] | None) -> dict[str, str] | None:
    """Widen a string-valued ``TypedDict`` to the plain ``dict[str, str]`` libraries expect.

    Author options as a ``TypedDict`` — so ``ty`` catches a mistyped key where it is written —
    then call this at the last moment, at the boundary to a third-party API annotated
    ``dict[str, str]`` (or ``dict[str, Any]``). ``None`` passes straight through so a call site
    can forward an optional value unchanged.

    The widening is a zero-copy ``cast``: the caller is responsible for the ``TypedDict``'s values
    actually being ``str`` (they are, for every current caller).
    """
    return None if td is None else cast("dict[str, str]", td)
