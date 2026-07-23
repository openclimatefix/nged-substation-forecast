# Testing

How the test suite is wired up, the house style for writing tests, and the notable test suites
that guard tricky invariants. Two testing *gotchas* are documented alongside the other gotchas in
CLAUDE.md rather than here: the moto S3 backend being process-global (reset it per test), and Polars
row counts wrapping past 2³² rows.

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
- **Mock with pytest's `monkeypatch` fixture, not `unittest.mock`.** Patch environment variables
  (`monkeypatch.setenv`), object attributes, and module-level functions
  (`monkeypatch.setattr(some_module, "open", fake_open)`) through the built-in fixture. For S3,
  drive the in-process `moto` server instead of mocking — `tests/test_s3_data_paths.py` is the
  canonical pattern.

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

`test_ecmwf_ens_network.py` (network-gated, above) composes all three against the live catalog and is
the only layer that can catch *future upstream drift* — a change in Dynamical.org's own conventions
that the committed slice, frozen at capture time, cannot.

## Assertion style for Patito frames

Build a frame, attach the model, cast, and validate for the happy path:

```python
df = pt.DataFrame({...}).set_model(MySchema).cast()
df.validate()                       # happy path: raises nothing
MySchema.validate(existing_df)      # or validate a frame produced elsewhere
```

For the unhappy path, assert that validation raises:

```python
from patito.exceptions import DataFrameValidationError

with pytest.raises(DataFrameValidationError):
    bad_df.cast().validate()
```

`packages/contracts/tests/test_geo_schemas.py` is the shortest end-to-end example.
