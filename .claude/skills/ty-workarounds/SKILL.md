---
name: ty-workarounds
description: >-
  Workaround for the upstream `ty` bug where Altair loses the chart type after a `mark_*()`
  call (`unresolved-attribute: ... has no attribute 'encode'`). Load when `ty check` flags
  Altair code, or before adding any `# ty: ignore`.
---

# `ty` workarounds for known upstream bugs

## Altair: `ty` loses the chart type after a `mark_*()` call

Altair decorates every `mark_*` method with `@use_signature`, whose return type is expressed through
a hand-written generic `TypeAliasType` over `Concatenate`. Since ty 0.0.64, ty resolves that alias
but never solves its type variable, so `alt.Chart(df).mark_line()` infers as the bare `T@__call__`
and the next call in the chain fails with `unresolved-attribute: Object of type 'T@__call__' has no
attribute 'encode'`. The code is correct — pyright infers `Chart` — and this is upstream ty bug
[astral-sh/ty#2520](https://github.com/astral-sh/ty/issues/2520).

**How to apply:** put `# ty: ignore[unresolved-attribute]` on the `.encode(` line of each chart
chain. Restructuring does not help: annotating an intermediate variable as `alt.Chart` instead
raises `invalid-assignment`, and calling `.encode()` before `.mark_*()` just moves the unsolved type
variable to the function's return. When ty fixes the bug, every suppression turns into an
`unused-ignore-comment` warning, which is the signal to delete them all.
