---
name: marimo-notebooks
description: >-
  Three authoring rules for this repo's Marimo notebooks (`packages/dashboard/*.py`,
  `packages/notebooks/*.py`), each reversing a normal Python habit: a leading underscore makes a
  name cell-local, so cross-cell helpers must be public; every import belongs in `with app.setup:`
  or ruff stops seeing it; and `ruff check --fix` must never be run over a notebook, because an
  autofix can insert an import outside `app.setup` and break it while reporting success. Load
  before creating or editing a Marimo notebook, or when one fails with a `NameError` on a helper
  or an import that looks present.
---

# Authoring Marimo notebooks

Marimo notebooks (`packages/dashboard/*.py`, `packages/notebooks/*.py`) are reactive: each
`@app.cell` function is a separate cell, and the `with app.setup:` block holds names shared by
every cell. The rules below follow from how Marimo scopes names and how ruff sees them.

## Never give a leading underscore to anything you want to reuse across cells

Marimo treats any name starting with `_` (a variable *or* a function) as *cell-local* — it is not
exported to other cells, so a `_helper()` defined in one cell (or in `app.setup`) is invisible
everywhere else and the call fails at runtime. A helper that multiple cells call must have a
public name (no leading underscore). This is the opposite of the usual `_private` convention that
CLAUDE.md's Code Style section asks for elsewhere in the repo, so it is easy to get wrong; the
leading-underscore-means-private habit does not apply inside a Marimo notebook.

## Put every import in the `with app.setup:` block

Never at module top level, and never let Marimo thread them through cell signatures. When imports
live in `app.setup`, they are real `import` statements that ruff analyses, so ruff flags a missing
or unused import. If you instead scatter imports into individual cells, Marimo passes the imported
names into the cells that use them as function parameters (`def _(pl, mo): ...`), and ruff treats
a parameter as always defined — so a genuinely missing import is invisible to the linter and only
blows up at runtime. Keeping all imports in `app.setup` keeps them statically checkable and
available to every cell.

## Never run `ruff check --fix` over a Marimo notebook

When an autofix needs a name the file does not import yet, ruff inserts the import into the file's
*top-level* import block — outside `app.setup`, where no cell can see it. The notebook then dies
with a `NameError` the next time it is opened, while `ruff check` reports success, because what
ruff produced is valid Python. This is a whole class, not one rule: any fix that adds an import
does it, and ruff has no per-file fixability setting to prevent it (`unfixable` is global, and
`per-file-ignores` would silence the check as well as the fix).

The pre-commit hook is split so notebooks are checked but never auto-fixed; a bare `uv run ruff
check . --fix` typed by hand is *not* covered, so after running one, check `git diff` for an
import that landed above `import marimo` and move it into `app.setup`.
