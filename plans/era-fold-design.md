# Era-boundary fold design for the studies (issues #868 and #892)

**The problem.** Every study that cuts folds within each UKV era (`weather_products.with_eras`, `assign_folds(by=("site", "era"))`) reuses fold numbers 0 to 4 in both eras, so a fold can hold the same calendar month out of both eras at once and the fitted model then never sees that season (#868). The ENS-horizons page also trains across ECMWF's IFS Cycle 49r1 change (live 2024-11-12) with no era cut, although the past-wind study found a step in ENS's and HRES's 10 m wind at that date (#892).

**The plan.** Measure both effects first, in scratch, on one wind study and one solar study. Then add opt-in options to `packages/studies` (a covering era-fold design, and an IFS 49r1 boundary constant), pin today's default behaviour with tests, and stop with a recommendation. Whether to re-run published pages is the maintainer's decision after the measurement, so this plan writes no study re-run.

## Verdict, size and departures

- **Verdict:** worth doing. The past-wind study (PR #885) already solved the problem for its own row set with `cut_eras` and `ENS_HRES_WIND_ERA_FOLD_OFFSETS`, so the machinery exists and the remaining work is generalising it and measuring the effect on the other pages.
- **Size: complex.** The five triggers:
    - *What gets stored:* fires. It adds options to `studies.cross_validation`, and a published page's numbers can change if the maintainer chooses to re-run.
    - *The production serving path:* does not fire. Only `packages/studies` and `studies/` change.
    - *A degradation rule:* does not fire. Study code fails fast by design.
    - *More than one defensible design:* fires. Offset era folds, whole-record folds, and a row-set restriction each answer part of the problem.
    - *Callers you could not name without searching:* fires. `assign_folds` and `with_eras` have about 20 callers across `studies/beam_diffuse_split/`.
- **Reviews:** both plan reviews, then both diff reviews (correctness, then mutation, since `packages/studies` changes).
- **Departure from the issues:** #868 suggests re-fitting one study; the maintainer asked for a measurement on one wind and one solar study before any published re-run, and for #892 to be planned with it.

## Measurement (scratch only, nothing under `data/studies/`)

Scripts and outputs live in `.claude/worktrees/scratch/era-fold/`. Design D0 is today's folds, D1 offsets each era's fold numbers so every calendar month is covered (`search_fold_offsets`), and D2 is whole-record contiguous folds with no era cut. Part A counts scored hours in uncovered (site, fold, calendar month) cells. Part B refits the planned-contrast arms of one wind and one solar study under D0, D1 and D2, and reports each contrast with its 95% interval and its change from D0. Part C refits the ENS-horizons wind and solar arms with a 49r1 era cut, and with rows from 2024-12-01, and compares the page's claims, including ENS day-0 minus ERA5 wind (+0.170 [+0.021, +0.322]).

Results: MEASUREMENT RESULTS TO BE PASTED IN BEFORE THE FIRST PLAN REVIEW.

## What changes, file by file

- `packages/studies/src/studies/cross_validation.py`: add `cut_eras_covering(frame, first_months)`, which returns `cut_eras` with the first offsets `search_fold_offsets` finds, and raises `ValueError` if no offsets cover every calendar month unless `allow_uncovered=True`. Add `IFS_CYCLE_49R1_FIRST_MONTH = "2024-12"`. `assign_folds`, `rotate_folds`, `cut_eras`, `search_fold_offsets` and `ENS_HRES_WIND_ERA_FOLD_OFFSETS` keep today's behaviour.
- `packages/studies/tests/test_cross_validation.py`: pin today's behaviour (a synthetic two-era frame where `with_eras`-style cutting holds a calendar month out of both eras, so `uncovered_months` is non-empty), and test `cut_eras_covering` (result covers, the smallest-rotation design is chosen, the error path, `allow_uncovered`).
- `studies/beam_diffuse_split/weather_products.py`: `with_eras` gains a keyword-only `covering: bool = False`. The default keeps every existing caller's folds unchanged, and a test compares fold assignments with and without the keyword on a fixed frame.
- `studies/beam_diffuse_split/ens_forecast_horizons.py`: `_complete` gains an option to add an IFS 49r1 era boundary, off by default. No published figure moves unless the maintainer chooses a re-run.
- `docs/studies/`: no page changes in this PR. The measurement report is summarised in the PR body and the maintainer's decision is recorded in issues #868 and #892.

## Design-philosophy check

This is R&D code, so it fails fast: `cut_eras_covering` raises when no design covers, as the wind study's coverage check does. No production path, asset or asset check is touched.

## Tests

Each new test states the assertion that fails on `main` today:

- `cut_eras_covering` returns a design with `uncovered_months(...)` empty on a frame where `cut_eras` with zero offsets leaves cells uncovered (`cut_eras_covering` does not exist on `main`).
- The default `with_eras` fold assignment equals `assign_folds(by=("site", "era"))` on a fixed frame, so a change of default is caught.
- `IFS_CYCLE_49R1_FIRST_MONTH` is the first whole month after 2024-11-12.

## Verification

`uv run ruff check .`, `uv run ruff format --check .`, `uv run ty check`, `pydoclint` (pinned 0.9.1), `uv run pymarkdown scan -r docs README.md CLAUDE.md packages/*/README.md`, `uv run pytest`, `uv run mkdocs build --strict`, and `scripts/lint/check_docs_links.py`. A mutation pass over the `packages/studies` change.

## Risks and open questions

- **Which design to recommend depends on the measurement.** Whole-record folds (D2) leave post-upgrade rows scored by models that trained on no post-upgrade rows, which is the failure era-cutting exists to prevent, so D1 is the working recommendation.
- **A covering offset may not exist for a row set.** The offsets found for the past-wind row set do not transfer, so each study must search on its own rows.
- **Published pages.** A re-run of every page that uses `with_eras` is large. The recommendation goes to the maintainer with the effect sizes.
