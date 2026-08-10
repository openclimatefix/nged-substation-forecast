---
name: ty-workarounds
description: >-
  Known upstream `ty` bugs where the code is correct and the checker is wrong, with the exact
  workaround for each: Altair losing the chart type after a `mark_*()` call
  (`unresolved-attribute: ... has no attribute 'encode'`), and numpy's `.view(np.uint32)`
  inferring `type[...]` instead of `dtype[...]` (`unsupported-operator`). Load when `ty check`
  flags Altair or numpy `.view()` code, or before adding any `# ty: ignore`.
---

# `ty` workarounds for known upstream bugs

In both cases **the code is correct** — pyright infers the right type — so the fix is a narrow
workaround, not a rewrite. Each entry records the signal that says it can be deleted.

## Altair: `ty` loses the chart type after a `mark_*()` call

Altair decorates every `mark_*` method with `@use_signature`, whose return type is expressed
through a hand-written generic `TypeAliasType` over `Concatenate`. Since ty 0.0.64, ty resolves
that alias but never solves its type variable, so `alt.Chart(df).mark_line()` infers as the bare
`T@__call__` and the next call in the chain fails with
`unresolved-attribute: Object of type 'T@__call__' has no attribute 'encode'`. This is upstream ty
bug [astral-sh/ty#2520](https://github.com/astral-sh/ty/issues/2520).

**How to apply:** put `# ty: ignore[unresolved-attribute]` on the `.encode(` line of each chart
chain. Restructuring does not help: annotating an intermediate variable as `alt.Chart` instead
raises `invalid-assignment`, and calling `.encode()` before `.mark_*()` just moves the unsolved
type variable to the function's return. When ty fixes the bug, every suppression turns into an
`unused-ignore-comment` warning, which is the signal to delete them all.

## numpy: `ty` mis-types `.view(np.uint32)` — pass `np.dtype(np.uint32)` instead

Since ty 0.0.67, `arr.view(np.uint32)` infers as
`ndarray[_AnyShape, type[unsignedinteger[_32Bit]]]` instead of
`ndarray[_AnyShape, dtype[unsignedinteger[_32Bit]]]`, so every subsequent operation on that array
fails — a bit-mask check reports `unsupported-operator: Unsupported & operation`. `ndarray.view`
is overloaded, and the inferred type looks like the overload taking
`DTypeT | _HasDType[DTypeT]` (with `DTypeT` bound to `np.dtype`) matched with `DTypeT` solved as
`type[np.uint32]`, in violation of that bound. The code is correct at runtime — pyright infers
`dtype[unsignedinteger[_32Bit]]` — and this is upstream ty bug
[astral-sh/ty#4208](https://github.com/astral-sh/ty/issues/4208).

**How to apply:** pass a real dtype object — `arr.view(np.dtype(np.uint32))` — which is the same
call at runtime and which ty resolves to the correct `ndarray[..., dtype[uint32]]`. Prefer this
over a `# ty: ignore` comment: the suppression would have to sit on the line that *uses* the
array, which can be several lines away from the `.view()` call that actually causes it.
Annotating the intermediate as `npt.NDArray[np.uint32]` does not work — it raises
`invalid-assignment` instead. The significand-rounding tests in `packages/delta_store/tests/`
are the worked examples. Nothing warns when the upstream bug is fixed, so the signal to delete
this section is astral-sh/ty#4208 closing; the `np.dtype(...)` calls themselves can stay, because
they are correct either way.
