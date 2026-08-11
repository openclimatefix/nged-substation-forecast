---
name: code-style
description: >-
  This repo's code conventions, which Jack cares about and expects to be followed: Python 3.14+,
  Polars only (pandas banned), Patito schemas, ruff configuration and its traps, naming, how
  expressive a type hint has to be, `Final` on constants, calling functions with keyword arguments,
  comment and doc-link rules, and Polars style. Load before writing or editing any Python in this repo — a new module, a new function, a
  refactor, a test — and before reviewing Python for style, or answering a question about what the
  house style is.
---

# Code style

**The rules live in one file: `docs/architecture/code-style.md`. Read it now, in full, before
writing or editing Python.** It is deliberately the only copy — this skill exists to make sure it
gets read, not to restate it, because two copies of a style guide drift apart and then nobody
knows which one is binding.

What is in there, so you know what you are getting:

| Section | Covers |
|---|---|
| General Principles | Python 3.14+, modularity, small functions (and when extra parameters are fine), re-using existing tools, tests |
| Formatting & Linting (Ruff) | line length, quotes, docstring convention, import rules, how `select`/`ignore` are maintained, two `per-file-ignores` traps |
| Type hints and signatures | how expressive a signature must be — `Literal` aliases, `TypedDict`, named aliases — and `Final` on every constant |
| Calling functions | keyword arguments at every call site the callee allows, and the three positional exceptions |
| Comments, docstrings and links | current-state-only rule, which docs code may link to, MkDocs-compatible constant docstrings |
| Data Handling | Polars/Patito/Xarray choices, lazy-evaluation contract, Patito friction budget, the 2³² row-count rule |
| Polars style | `.cast` vs `.with_columns`, keyword-argument column naming, `Type`-suffixed `Literal` aliases |
| Gotchas that fail silently | which sibling skill to load for which trap |
| Error Handling | exception style, Sentry, validation at boundaries |

Two rules there are enforced by nothing and fail silently, so they are the ones to be deliberate
about: **never row-count a table that can exceed 2³² rows with Polars** (counts wrap with no
error), and **comments must describe only how the code works now** — never a previous iteration or
a deleted file.

If you are about to write Polars or Patito code, load **`polars-patito-gotchas`** as well; the
style rules and the traps are separate documents on purpose.

## When you change the rules

Edit `docs/architecture/code-style.md` — never copy a rule into `CLAUDE.md` or into this skill.
The page is published at
<https://openclimatefix.github.io/nged-substation-forecast/architecture/code-style/> and is linked
from `docs/index.md`, `docs/design-philosophy/index.md` and `design-principles.md`, so it is the
version humans read too.
