# Testing

How the test suite is wired up, the house style for writing tests, and the notable test suites
that guard tricky invariants. One testing gotcha lives elsewhere because it is not really about
tests: Polars row counts wrapping past 2³² rows, in
[Performance and Scale](performance.md#the-other-hard-ceiling-polars-32-bit-row-index).

## Where tests and their dependencies live

- **Test tooling is declared once, at the workspace root.** `pytest`, `moto`, and `numpy` live in
  the root `pyproject.toml` `[dependency-groups] dev`, and every workspace package inherits them.
  A package that gains a `tests/` directory does **not** re-declare `pytest` in its own
  `pyproject.toml` — we run the whole suite from the repo root with `uv run pytest`, against the
  root environment. (`packages/geo` declares its own `pytest`/`pytest-cov`; treat that as a
  historical exception, not the pattern to copy.)
- **Discovery is automatic.** The only pytest configuration is the root
  `[tool.pytest.ini_options]` block; there is no `testpaths` setting, so pytest collects both the
  top-level `tests/` directory and every `packages/*/tests/` directory. A brand-new
  `packages/<pkg>/tests/` directory is picked up with no configuration change — provided the
  package is installed in the root environment, which is automatic only when something already
  depends on it. A **leaf** package that nothing depends on (for example `dashboard`, a marimo app)
  is *not* in the default environment, so `uv run pytest` cannot import its tests; add it to the
  root `[dependency-groups] dev` list (and give it a `[tool.uv.sources]` workspace entry) so a plain
  `uv sync` installs it.
- **Run the whole suite with plain `uv run pytest`, never `--all-packages`.** `uv run pytest`
  executes against the root environment, which holds exactly the packages reachable from the root's
  dependencies and dev group — i.e. every package that has tests, by the rule above. `--all-packages`
  additionally installs workspace members that have no tests (`notebooks`) and each member's own
  dev-groups, so it is heavier for no benefit here. It is the right tool for the pre-commit `ty` hook
  (`uv run --all-packages ty check`) for a different reason: `ty` type-checks the *source* of every
  workspace member, including leaf packages that are never installed as a dependency of anything, and
  that source must be present for the check. Type-checking needs the source; running tests needs the
  package installed — so the two commands legitimately differ.
- **`--import-mode=importlib` is set deliberately** so that identically-named test modules in
  different packages (for example, two `test_storage.py` files) do not collide during collection.
  Because of this, test directories do not need `__init__.py` files.
- **Test data files go in a `tests/data/` subdirectory** and are loaded relative to the test
  module with `Path(__file__).parent / "data" / filename`. `packages/nged_data/tests/` is the
  canonical example (it also keeps a small script documenting how the fixtures were trimmed down).

## Fixtures and mocking

- **Define fixtures inline in the test module by default.** When a fixture — or a fixture factory
  — is shared across more than one test module *within a single package*, put it in a
  package-level `tests/conftest.py`. `packages/dynamical_data/tests/conftest.py` is the example:
  it builds synthetic Xarray datasets that two test modules share. The only repo-root `conftest.py`
  holds cross-package pytest plumbing, not fixtures — currently the network-test gate below.
- **A factory shared *across* packages goes in the root `tests/` directory, not in any one
  package's `tests/`.** The root `pyproject.toml` sets `pythonpath = ["tests"]` for the whole
  `uv run pytest` session, so every module placed at the top level of `tests/` is importable by
  bare name from any test suite in the repo — `packages/delta_store/tests`,
  `packages/ml_core/tests`, and the root `tests/` alike — the same mechanism
  `tests/_nwp_test_data.py` already relies on for the root integration tests. Putting a
  cross-package factory inside one specific package's `tests/` and importing it from another
  package's suite would work by accident of that package being installed, but it reads as a
  dependency of the *package under test* on another package's test code, which is backwards; the
  root `tests/` directory carries no such implication because it is not itself a workspace member.
  A factory production code needs (not just tests) still belongs in `contracts` or another
  library package, never here.
- **Mock with pytest's `monkeypatch` fixture, not `unittest.mock`.** Patch environment variables
  (`monkeypatch.setenv`), object attributes, and module-level functions
  (`monkeypatch.setattr(some_module, "open", fake_open)`) through the built-in fixture. For S3,
  drive the in-process `moto` server instead of mocking — `tests/test_s3_data_paths.py` is the
  canonical pattern.
- **Reset the moto S3 backend per test.** The in-process `moto` server keeps its bucket contents
  in a **process-global backend that outlives the `ThreadedMotoServer` object**, so a
  module-scoped server does not hand each test a clean slate. A test whose write path runs twice
  against that server — a re-run, or state left behind by an earlier test — reads stale data: an
  appended Delta table returns double the rows, and an `object_exists` precondition sees a
  leftover parquet. Keep the *server* module-scoped for speed, but give each test a
  **function-scoped** fixture that `POST`s to `/moto-api/reset` and recreates the bucket before
  the test body runs, so every test starts pristine and independent of execution order.
- **Take the `dagster_instance` fixture, or enter the instance as a context manager — never leave
  a `DagsterInstance.ephemeral()` unowned.** The fixture (in `tests/conftest.py`) enters the
  instance for you, so `dispose()` runs when the test ends. A module-level test helper that a
  fixture cannot reach — `_run_live_check` in `tests/test_checks.py` — takes the second form
  instead, and disposes on the way out of the helper.

    `DagsterInstance` has no finaliser, so an undisposed instance defers two cleanups to whenever
    the garbage collector reaches it. Both then surface as failures owned by no test:

    - Its run storage and event-log storage each hold one SQLAlchemy connection to an in-memory
      SQLite database. At interpreter shutdown the connection-pool finaliser can run *after*
      SQLite has closed the database, printing a bare `Exception during reset or similar`
      traceback after pytest's summary line.
    - `TemporaryLocalArtifactStorage` defers its `tempfile.TemporaryDirectory()` cleanup to
      `dispose()` as well, leaving it to a `weakref.finalize` callback that can fire at any point,
      not just at shutdown. The callback emits `ResourceWarning: Implicitly cleaning up
      <TemporaryDirectory ...>`, which the [warnings-are-errors](#warnings-are-errors) policy
      *raises* — and
      an exception inside a weakref callback is unraisable, so pytest's `sys.unraisablehook` turns
      it into a hard `PytestUnraisableExceptionWarning` attributed to *whatever unrelated test was
      running* when the collector fired.

    Neither is rare or environmental: both are what always happens when a used ephemeral instance
    is collected rather than disposed. Only *which* test gets blamed varies.

- **In a script, use the context manager directly** — `with DagsterInstance.ephemeral() as
  instance:`. Letting the local go out of scope is *not* enough: once the instance has run a job,
  Dagster's own caches retain it (a `RunDomain`, and the partition-loading contexts holding it as
  `dynamic_partitions_store`), so it reaches interpreter shutdown with both connections open even
  on a completely successful run. The context manager also covers the failure path, where an
  unhandled exception's traceback pins the raising frame. Worked example:
  `scripts/run_baseline_experiment.py`. Measured with an `atexit` probe after one real
  `materialize`: 2 connections still open with a bare call, 0 with the context manager, on both
  paths.
- **`build_asset_context()` (and the other `build_*_context()` helpers) needs the same treatment
  when called without an explicit `instance=`.** It defaults to `DagsterInstance.ephemeral()`,
  owned by an `ExitStack` that closes only on `__exit__` or `__del__`. Passed straight into an
  asset (`some_asset(build_asset_context(...))`) nothing ever calls `__exit__`, and inside a
  `pytest.raises(...) as exc_info:` block the captured traceback keeps the context referenced past
  the end of that block, so `__del__` waits on a GC pass. Enter it as a context manager instead:
  `with build_asset_context(...) as context, pytest.raises(...) as exc_info:`. Worked example:
  `tests/test_assets.py::test_ecmwf_ens_retries_when_run_not_yet_available` — probing right after
  that test's teardown with **no forced `gc.collect()`** found 2 open connections before the fix,
  0 after, every time.
- **`build_asset_check_context()` is the exception: give it `instance=`, because it cannot be
  entered.** `DirectAssetCheckExecutionContext` defines no `__enter__`, so the context-manager
  form above is a `TypeError`, and the `with` block has to hold the instance instead. Dagster
  builds one of these for a directly-invoked check that declares *no* context parameter too, so
  `power_data_is_fresh()` leaks an instance on a call that mentions Dagster nowhere; pass it a
  context anyway, which Dagster accepts and uses in place of the one it would build. Worked
  examples: `_run_freshness_check` and `_run_live_check` in `tests/test_checks.py`.
- **`materialize()` and `JobDefinition.execute_in_process()` do *not* need this.** Both wrap their
  default instance in their own internal `with ephemeral_instance_if_missing(instance):`, entered
  and exited inside the call, so disposal is already deterministic however the caller uses the
  return value. The distinguishing question for any Dagster helper that can default-construct an
  instance is whether *the helper itself* closes the `with` block, or hands you an object that
  expects *you* to.
- **An autouse fixture fails whichever test leaks one.**
  `_fail_on_an_undisposed_dagster_instance` in `tests/conftest.py` wraps `DagsterInstance.ephemeral`
  and `DagsterInstance.dispose` for the duration of each test, then asserts at teardown that every
  instance created was disposed — so the fault is reported by name instead of against an unrelated
  test. It catches a leak, not an omission: an unreferenced context is freed as soon as the helper
  that built it returns, so a missing `instance=` can still pass green until something pins the
  frame holding it.

## Running the suite in parallel

A plain `uv run pytest` — no file or node id, no `-n`/`-k`/`-m` of its own — runs under
`pytest-xdist` with one worker process per physical CPU core (`-n auto`; `pytest-xdist` counts
physical, not logical, cores when `psutil` is installed). The flag is added by the
`tests/_pytest_autoinject.py` plugin, loaded via `-p _pytest_autoinject` in `addopts`, rather than
a static `addopts` entry naming `-n auto` directly: the plugin can see the invocation before adding
the flag, so a targeted run (`uv run pytest path/to/test_foo.py::test_bar`) stays serial instead of
paying worker start-up cost for one test. An invocation that already passes its own `-n` is left
alone. The auto-injection can't be a `pytest_load_initial_conftests` hook in the root `conftest.py`
itself — that hook fires as part of loading the root `conftest.py`, so a hookimpl defined inside it
is never registered in time to run for that same call.

The root `conftest.py` caps `OMP_NUM_THREADS` and `POLARS_MAX_THREADS` at 4 for the whole session.
XGBoost and Polars each default to a thread pool sized to the machine's core count, so without the
cap, `-n auto`'s one-process-per-core plus each process's own full-width thread pool oversubscribes
the machine by core-count² and every worker's threads fight the others for the same cores —
measured on a 32-thread workstation, removing the caps made the full suite run in 95s with 4755
threads and a load average of 201, versus 21-24s capped. The cap is 4 rather than 1 because it costs
nothing in wall-clock on that workstation while matching production more closely, where a single job
runs with no thread cap at all.

## Warnings are errors

`[tool.pytest.ini_options] filterwarnings` in `pyproject.toml` starts with `error`, so **any**
warning fails the test that raised it. A deprecation introduced by our own code is therefore caught
by the PR that introduces it, instead of accumulating in the warnings summary.

The exceptions listed after `error` are third-party deprecations we cannot fix from this repo. Each
one is pinned to the exact warning message *and* the exact upstream module that raises it — the
module anchor is what stops an entry from ever masking the same deprecation appearing in our own
code — and carries a comment naming the package, the version, and the condition for deleting it.

When a dependency upgrade introduces a new upstream warning, the suite fails loudly. Add a new
entry in that same form rather than widening an existing one; if the warning comes from our code,
fix the code. Later entries take precedence, so `error` stays first.

## Network-gated tests

Most tests run fully offline, mocking any network call (for example, patching
`dynamical_catalog.open` to return a synthetic `xr.Dataset`). A handful of tests are worth running
against a **real** external service — chiefly to catch the *shared-convention blind spot*, where a
synthetic fixture and the code under test share the same wrong assumption about the real data's
shape (dimension order, latitude orientation, longitude range, dtypes, units) and both pass.

Mark such a test `@pytest.mark.network`. The root `conftest.py` skips every `network`-marked test
unless the caller passes `--run-network`, so a plain `uv run pytest` (local dev and the per-PR CI)
never touches the network. Run them explicitly — the nightly CI job (see
[Continuous integration](#continuous-integration) below) or on demand — with:

```bash
uv run pytest --run-network              # whole suite, network tests included
uv run pytest --run-network -m network   # only the network tests
```

`packages/dynamical_data/tests/test_ecmwf_ens_network.py` is the canonical example: it drives the
real `open → download → convert` pipeline against the Dynamical.org ECMWF ENS catalog and asserts
the conventions the offline fixtures merely assume.

The gate is a collection hook, **not** an `addopts = "... -m 'not network'"`. pytest keeps only the
*last* `-m` it is given, so any caller-supplied marker expression (e.g. `-m "not integration"`) would
silently replace an `addopts` `-m "not network"` and re-include the network tests. A skip applied
during collection cannot be overridden that way — the gate holds whatever `-m` the caller passes, and
even `-m network` alone stays skipped until `--run-network` is added.

## Continuous integration

Two GitHub workflows in `.github/workflows/` run the checks described on this page:

- **`ci.yml` — the per-PR quality gate.** Runs on every pull request and every push to `main`:
  `ruff check`, `ruff format --check`, `ty check`, the `pymarkdown scan` command from CLAUDE.md,
  and the offline test suite (plain `uv run pytest` — the network gate above keeps CI off the
  network). The job installs with `uv sync --locked --all-packages`: `--all-packages` because
  `ty` type-checks the source of every workspace member, including leaf packages that a plain
  sync would omit, and `--locked` so the build fails loudly when `uv.lock` is stale. Every
  subsequent step passes `uv run --no-sync`, because a bare `uv run` re-syncs to the root
  environment and would silently uninstall those extra workspace members. The job also sets
  dummy values for the three required `NGED_S3_*` `Settings` fields: most tests monkeypatch
  them, but a few construct `Settings()` directly and locally rely on the developer's `.env`,
  which CI doesn't have. The `ci` job is a required status check on `main` (configured in the
  GitHub branch-protection settings, not in the workflow file).
- **`nightly_network_tests.yml` — the nightly network job.** Runs *only* the network-gated
  tests (`uv run pytest --run-network -m network`) on a daily schedule, plus
  `workflow_dispatch` for on-demand runs. This is the only CI that touches the real
  Dynamical.org catalog, and it needs no secrets — the catalog is public. A failure notifies
  (GitHub emails the workflow author when a scheduled run fails) but deliberately does not
  block PRs: a red nightly run signals drift in the upstream catalog's conventions, not a
  defect in whatever PR happens to be open.

### Why a bespoke workflow rather than OCF's template

OCF's organisation template ([`openclimatefix/.github` →
`workflow-templates/branch_ci.yml`](https://github.com/openclimatefix/.github/blob/main/workflow-templates/branch_ci.yml))
is a thin caller of the org-wide reusable workflow
[`branch_ci.yml`](https://github.com/openclimatefix/.github/blob/main/.github/workflows/branch_ci.yml).
We deliberately don't use it, because it is built for OCF's standard single-package service
repos and fits this repo poorly:

- **Single-package assumptions.** The reusable workflow expects one `pyproject.toml` and one
  test folder (its `tests_folder` input defaults to `src/tests`). This repo is a uv
  *workspace* monorepo: the suite must run from the root so pytest collects `tests/` plus
  every `packages/*/tests/`, and type-checking needs an `--all-packages` install (see above).
- **Floating tool versions.** It lints and type-checks with `uvx ruff` / `uvx ty`, i.e.
  whatever version is newest on the day the job runs — so org CI can disagree with the locked
  dev-group versions used locally and by pre-commit, and a new upstream release can break CI
  with no change in this repo. We run the locked versions via `uv run`.
- **Missing checks, and no offline/network split.** It has no equivalent of
  `ruff format --check` (formatting is opt-in and separate) or the `pymarkdown scan` command,
  and no concept of this repo's network-gated nightly job.
- **Container build baggage.** Roughly half the reusable workflow builds and publishes Docker
  images to ghcr.io; this repo doesn't produce a container image.
- **Trigger and pinning mismatch.** The template triggers on pushes to non-default branches,
  whereas a required status check wants `pull_request` (+ push to `main`); and it pins the
  reusable workflow `@main`, so upstream edits to the org workflow change this repo's CI
  without any PR here.

We did keep the template's good conventions: a concurrency group with `cancel-in-progress`, a
cached `astral-sh/setup-uv`, and locked installs (`UV_LOCKED=1` there, `uv sync --locked`
here). If OCF's reusable workflow ever grows first-class uv-workspace support, revisiting it
would shrink `ci.yml` to a few lines — but until then the bespoke workflow is smaller than the
configuration the template would need.

## NWP grid → H3 orientation coverage

The NWP-grid-to-H3 mapping is the classic place for a silent orientation bug — a vertically or
horizontally flipped weather grid, a transpose (`np.meshgrid` `indexing="ij"` vs `"xy"`), or a
lat/lon swap. Three tests guard it in layers, from cheap-and-synthetic to real-and-networked. Each
was checked by *mutation*: introducing the bug into the production code and confirming the test goes
red. The table records which mutation each layer catches (✓ = the test fails when that bug is
present):

| Mutation in production code | synthetic `convert` test¹ | cached real-slice test² | geo landmark test³ |
| --- | :---: | :---: | :---: |
| `np.meshgrid` `indexing="ij"` → `"xy"` (transpose) | ✓ | ✓ | n/a |
| lat/lon swap in the value-join keys | ✓ | ✓ | ✓ |
| reversed latitude ravel (vertical flip) | ✓ | ✓ | n/a |
| swap `cell_to_lat`/`cell_to_lng` when snapping | n/a | n/a | ✓ |

- ¹ `dynamical_data/tests/test_convert_to_polars.py::test_convert_maps_each_grid_point_to_its_own_lat_lon`
  — a 2×2 synthetic grid with a distinct value at every corner. Guards the ravel-alignment step
  *inside* `convert`; the value-join itself is positional-agnostic, so which hexagon owns a given
  (lat, lon) is delegated to the upstream `h3_grid_weights` asset (the geo test below).
- ² `dynamical_data/tests/test_ecmwf_ens_cached.py::test_cached_real_slice_conventions_and_orientation`
  — the same orientation check on a committed real ECMWF ENS slice, so the conventions the synthetic
  fixture only *assumes* (descending latitude, °C, dimension order) are exercised on genuine bytes.
- ³ `geo/tests/test_h3.py::test_grid_weights_preserve_geographic_orientation` — proves
  `compute_h3_grid_weights` labels each H3 cell with grid points at the cell's own (lat, lon), using
  two well-separated GB landmarks. This is what fixes the hexagon↔(lat, lon) geography the `convert`
  tests delegate.

`test_ecmwf_ens_network.py` (network-gated, above) runs the full open → download → convert pipeline
against the live catalog, but only re-checks *orientation and bounds*: descending latitude, longitude
in [-180, 180], the slice landing on the requested box, the expected variable names, and a
physical-range sanity check on temperature. It does not re-check the value↔(lat, lon) mapping —
neither the ravel-alignment mutation the synthetic `convert` test guards nor the hexagon↔(lat, lon)
geography the geo landmark test guards, since both are value-agnostic (index alignment, not a value
comparison) and so are already fully proven offline. What only this test can catch is *future
upstream drift* — a change in Dynamical.org's own conventions that the committed slice, frozen at
capture time, cannot.

## Marimo notebooks bind every name their cells reference

Marimo rebuilds a notebook from its `with app.setup:` block plus its `@app.cell` functions, and
never runs the module-level statements in between. A name bound at module level is therefore
invisible to every cell: the notebook raises `NameError` the next time it is opened, while ruff, ty
and pytest all pass, because the file they were handed is valid Python. Two tools produce exactly
that shape from a working notebook — `ruff check --fix`, which writes an import an autofix needs
into the top-level import block, and `marimo check --fix`, which deletes such an import and
rewrites the cell that used the name as `def _(name)`, leaving a cell input nothing defines.

`scripts/check_marimo_notebooks.py` reads each cell's `refs` and `defs` and reports any name a cell
references that no cell binds. It runs as a pre-commit hook over changed notebooks, and
`tests/test_marimo_notebooks.py` runs it over every notebook in `packages/notebooks/` and
`packages/dashboard/`. Three properties are worth knowing:

- **It is static.** Nothing executes, so the check needs none of the notebooks' runtime
  dependencies — only marimo itself, which the root environment has via the `dashboard` dev
  dependency. It cannot catch a notebook that binds every name and still fails inside a Polars or
  Altair call; executing the notebooks is not an option, because they read real Delta tables and
  S3.
- **It rides on private marimo API.** `Cell.refs` and `Cell.defs` are documented, but loading a
  notebook without running it is not. So the checker raises rather than reporting "no findings"
  whenever a file does not parse into at least one cell, and the tests keep a positive control —
  a deliberately broken notebook, held as a string so ruff never sees it — that fails if a marimo
  release stops the check detecting a real breakage.
- **Every `.py` file directly inside those two directories must be a notebook**, and a file that
  is not one is a finding. The ruff pre-commit hooks share that assumption: they use it to decide
  which files must never be auto-fixed.

Testing what a notebook's cells actually *do* is a separate job, and
`packages/notebooks/plot_missing_NWP_data.py` is the worked example. Its chart-building helper is
an `@app.function` — marimo's form for a top-level reusable function — so an ordinary `test_*`
function in the same notebook can exercise it on a synthetic frame. Naming the notebook in
`python_files` is what makes a plain `uv run pytest` collect it. The authoring rules for writing
one are in the `marimo-notebooks` skill.

## Assertion style for Patito frames

Build a frame, attach the model, cast, and validate for the happy path:

```python
df = pt.DataFrame({...}).set_model(MySchema).cast()
df.validate()  # happy path: raises nothing
MySchema.validate(existing_df)  # or validate a frame produced elsewhere
```

For the unhappy path, assert that validation raises:

```python
from patito.exceptions import DataFrameValidationError

with pytest.raises(DataFrameValidationError):
    bad_df.cast().validate()
```

`packages/contracts/tests/test_geo_schemas.py` is the shortest end-to-end example.
