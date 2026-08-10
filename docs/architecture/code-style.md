# Code Style Guidelines

(This is mostly written for LLM coding agents!)

This page is the **single source of truth for code style in this repo**. `CLAUDE.md` does not
repeat any of it — it points here. Read this page before writing or editing Python.

## General Principles

- **Python Version**: Use Python 3.14+.
- **Type Hints**: All function signatures **must** use expressive type hints for all arguments and
  return types, including the return type. Use `typing` and `collections.abc` as needed. See
  [Type hints and signatures](#type-hints-and-signatures) for how expressive.
- **Modularity**: Keep logic in small, focused packages under `packages/`. The main app in `src/`
  should primarily handle orchestration.
- **Small functions**: Prefer small function bodies that do one, well-defined thing. Extract
  private helpers (`_name`) rather than letting a function body grow long, even if that means more
  parameters — a well-named helper with a clear docstring beats a long inline block. Eight
  parameters is acceptable when each is distinct and the division of labour is clear.
- **Minimalism**: Re-use existing tools (Polars, Xarray, Dagster) instead of reinventing logic.
- **Tests**: Unit tests should each be a short, simple function. For each function in the main
  code, there should be at least one test function that tests the "happy path", and one test
  function for each of the main "unhappy" paths. Never relax an existing test just to get it to
  pass! See the [Testing](testing.md) page for where tests live, how they are wired, mocking, and
  the assertion house style.

## Formatting & Linting (Ruff)

- **Line Length**: 100 characters, enforced by `ruff check` (`E501`) as well as by the formatter.
  The formatter breaks code but not comments, docstrings, or long string literals, so those have
  to be wrapped by hand.
- **Quotes**: Use **double quotes** (`"`) for strings.
- **Docstrings**: Use **Google convention**, enforced by the `D` rules. Every public module,
  package, class, and function needs one; tests and marimo notebooks are exempt.
- **Type annotations**: Enforced by the `ANN` rules on every signature. `typing.Any` is allowed
  where no honest narrower type exists.
- **Imports**: Sorted automatically by `ruff` (isort rules). `import pandas` is banned outright
  (`TID251`), not merely discouraged.
- **Rule selection**: `[tool.ruff.lint] select` in `pyproject.toml` names the enabled families
  explicitly rather than inheriting ruff's defaults, so a `uv lock` refresh cannot quietly change
  which rules the repo enforces. `select` names whole families; `ignore` names each family member
  we decline, with the reason on the line above it. **When a rule fires somewhere it should not,
  add an `ignore` entry (or a `per-file-ignores` entry) with its justification — do not drop the
  whole family.** Ruff's defaults are not a superset of the old `E4`/`E7`/`E9`/`F` gate: of
  pycodestyle-E they keep only `E722` and `E902`, which is why `E4`/`E7`/`E9` are listed
  explicitly.
- **Two traps when editing `[tool.ruff.lint.per-file-ignores]`**:

    - **`*` crosses `/`.** A key of `packages/dashboard/*.py` also silences
      `packages/dashboard/src/dashboard/*.py`. Name individual files when you mean "just the ones
      in this directory"; `**/tests/**` is the right way to say "every tests directory, root
      included".
    - **A `# noqa` cannot live inside a docstring**, because it would become part of the string.
      An over-long docstring line has to be reworded or re-wrapped, never suppressed.

- **Naming**:
    - Variables/Functions: `snake_case`
    - Classes: `PascalCase`
    - Constants: `UPPER_SNAKE_CASE`

## Type hints and signatures

**Prefer self-documenting type hints over bare containers — a signature is documentation.** Jack
strongly prefers expressive signatures and is happy to spend a few extra lines of code to get
them, as long as complexity stays low. Whenever you would write `dict[str, str]` (or a bare `str`
for a value from a fixed set, or a tuple of positional values), stop and ask whether a more
self-documenting type is practical. Reach for:

- a `Type`-suffixed `Literal` alias for a closed set of string values
  (`StageType = Literal["register", "train", "predict", "metrics"]`);
- a named alias for a recurring shape (`MlflowTags = dict[str, str]`) so the intent is stated once
  and reused;
- a `TypedDict` for a structured mapping with known keys (e.g. `ObjectStoreOptions`) — taking the
  `TypedDict` in the signature and widening to a plain dict at the call boundary.

Constraining `dict` *keys* to a `Literal` alias (`dict[TableNameType, str]`) is worthwhile for a
closed vocabulary and works with bidirectional inference when callers pass dict literals.
`packages/ml_core/src/ml_core/_repro.py` is the worked example. Don't force it where no honest
stricter type exists — a genuinely heterogeneous or open-ended dict stays `dict[str, str]`.

**All constants must be marked with the maximally "constant" type**, e.g.
`CONST_SEQ: Final[tuple[str, ...]] = ("a", "b")` or `FOO: Final[str] = "bar"`.

## Comments, docstrings and links

- **Do not remove existing comments** unless they are misleading or out of date. Only add new
  comments if you're doing something that isn't obvious from the code. Write self-documenting
  code, and assume the reader is fluent in Python.
- **Comments and docs must reflect current state only** — never reference previous iterations of
  the code or deleted files. This is the same rule as "Write about the present, not the past" in
  `CLAUDE.md`, applied to code.
- **Code links only to durable docs** — `docs/design-philosophy/`, `docs/background/`,
  `docs/techniques/`, `docs/architecture/`, `docs/ml_experimentation/`, `docs/live_service/`.
  Never link from code *or* docs to `plans/` files, and never from code to `docs/roadmap/` pages
  or to any "Implementation details (deleted when this ships)" section — all of those are deleted
  when the work lands, so the reference rots. (Docs-to-docs links into `docs/roadmap/` are fine;
  retargeting them is part of ship-time triage.) Linking from a docstring to a durable page — e.g.
  `docs/architecture/` — is encouraged.
- **MkDocs-compatible constant docs** — document module-level constants with a string literal
  immediately after the assignment, not with Sphinx-style `#:` comments. This is correct:

    ```python
    MY_CONST: Final[str] = "value"
    """One-line summary.

    Optional further detail.
    """
    ```

## Data Handling

- **Tabular Data**: Use **Polars** (`import polars as pl`) for dataframes. Pandas is strictly
  forbidden. Use Polars for all tabular data.
- **Lazy evaluation**: Use `pl.LazyFrame` throughout the pipeline. **Do not call `.collect()`
  before the model boundary.** See [Lazy evaluation strategy](performance.md#lazy-evaluation-strategy)
  for the full contract.
- **Gridded/NWP Data**: Use **Xarray** and **Zarr**.
- **Data Contracts**: Use **Patito** for defining and validating data schemas. Use Patito type
  annotations (`pt.DataFrame[MySchema]`, `pt.LazyFrame[MySchema]`) whenever a function consumes or
  returns data that conforms to an existing schema — whether the function is public or private.
  Don't invent a new schema just to annotate a private helper; if no existing schema fits, use
  plain `pl.DataFrame` / `pl.LazyFrame`.
- **Patito friction budget**: the `polars-patito-gotchas` skill documents five Patito gotchas
  (cross-model LazyFrame joins, dict-`.cast` on model-bearing frames, `ge`/`le` silently ignored
  on a datetime field, `.filter()` dropping the Patito subclass, and Delta dictionary-encoded
  columns). Five workarounds is an acceptable price for schema validation — but if a sixth becomes
  necessary, revisit the approach: either validate only at I/O boundaries (typed annotations
  everywhere, `.validate()` only at persistence edges) or evaluate an alternative such as
  `dataframely`.
- **Never row-count a table that can exceed 2³² rows with Polars.** Default Polars builds use a
  32-bit row index, so past ~4.29 billion rows `pl.len()` and `group_by(...).agg(pl.len())` wrap
  modulo 2³² with **no error**. Use the Delta log instead — `DeltaTable(path).count()`, or sum
  `num_records` over `get_add_actions(flatten=True)` — both metadata-only and exact. Value
  aggregations (`sum`, `min`/`max`, quantiles) are unaffected, and so are filtered queries whose
  *result* stays under the cap. NWP (~5.9B rows) is past it today; `power_forecasts` will pass it
  at V2 scale. Full analysis:
  [The other hard ceiling](performance.md#the-other-hard-ceiling-polars-32-bit-row-index).
- **Persistence**: Prefer partitioned Parquet files for tabular data.

## Polars style

These rules are all about making Polars code easy to read.

- When casting, prefer using the `cast` method like this: `df.cast({"foo": pl.Int8})`, in favour of
  using `df.with_columns(pl.col("foo").cast(pl.Int8))`. **Caveat:** this is only safe on a plain
  Polars frame — passing a `{column: dtype}` mapping to a *model-bearing* Patito frame silently
  does the wrong thing. See the `polars-patito-gotchas` skill.
- When using `.with_columns`, prefer specifying the destination column name as a key word argument
  like this: `df.with_columns(bar=pl.col("foo").expression())` instead of using `alias` like this:
  `df.with_columns(pl.col("foo").expression().alias("bar"))`
- **`Literal` type aliases — use a `Type` suffix** to distinguish them from the runtime tuples
  that drive Polars `Enum` declarations. Example:

    ```python
    EVALUATION_SCOPES: Final[tuple[str, ...]] = ("leaderboard", "production_monitoring", "ad_hoc")
    """Runtime tuple — used as pl.Enum(EVALUATION_SCOPES)."""

    EvalScopeType = Literal["leaderboard", "ad_hoc"]
    """Type annotation — currently-implemented subset; update when adding a new scope."""
    ```

    The `Type`-suffixed alias is what goes in function signatures; the `UPPER_SNAKE_CASE` tuple is
    what goes into `pl.Enum(...)`. They serve different purposes and should both exist.

## Gotchas that fail silently

Three groups of trap in this codebase produce **no error at the point of the mistake**, so each
lives in a skill you are expected to load *before* writing the code rather than after the
confusing failure:

- **`polars-patito-gotchas`** — Patito's model machinery colliding with Polars and delta-rs: a
  cross-model `.join()` that has to have its right-hand operand stripped, a `{column: dtype}`
  `.cast` swallowed on a model-bearing frame, `ge`/`le` doing nothing on a datetime field,
  `.filter()` dropping the Patito subclass, and a dictionary-encoded column blocking Delta
  predicate pushdown so a partition-filtered query reads the whole table.
- **`marimo-notebooks`** — leading underscores are cell-local, imports belong in `app.setup`, and
  `ruff check --fix` must never be run over a notebook.
- **`ty-workarounds`** — known upstream `ty` bugs on Altair and numpy, where the code is correct
  and the checker is not.

## Machine Learning (PyTorch)

- Use **PyTorch** for differentiable physics models.
- Use **MLFlow** for tracking experiments.
- Follow the "test-harness" pattern: separate research logic from production orchestration but
  ensure they use the same data contracts.

## Error Handling

- Use specific exceptions.
- Leverage Sentry for observability in production-like code.
- Validate data at boundaries using data contracts.

Production code is bound by a stronger rule about *when* to raise at all, summarised in
`CLAUDE.md` and set out in full on the
[Inherent stability](../design-philosophy/inherent-stability.md) page.

## Testing

Test wiring, fixtures, mocking, the network-test gate, and the Patito assertion house style now live
on their own page: **[Testing](testing.md)**.
