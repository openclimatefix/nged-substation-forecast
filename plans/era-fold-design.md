# Era-boundary fold design for the studies (issues #868 and #892)

**The problem.** Every study that cuts folds within each UKV era (`weather_products.with_eras`,
`assign_folds(by=("site", "era"))`) reuses fold numbers 0 to 4 in both eras, so a fold can hold the
same calendar month out of both eras at once and the fitted model then never sees that season
(#868). The ENS-horizons page also trains across ECMWF's IFS Cycle 49r1 change (live 2024-11-12)
with no era cut, although the past-wind study found a step in ENS's and HRES's 10 m wind at that
date (#892).

**The plan.** Measure both effects first, in scratch, on one wind study and one solar study, then on
the two further panels the second plan review found to be the most exposed. Publish the measurement
on #868 with a recommendation and no library change. Rewrite the ENS-horizons page's wind claims
(#892) from a re-run on rows from 2024-12-01 with a pinned covering fold rotation, written to a new
results folder. The re-run needs the coordinator's go-ahead, because it writes new outputs under
`data/studies/`.

## Verdict, size and departures

- **Verdict:** worth doing. The past-wind study (PR #885) already solved the problem for its own row
  set with `cut_eras` and `ENS_HRES_WIND_ERA_FOLD_OFFSETS`, so the machinery exists and the
  remaining work is measuring the effect on the other pages and applying the same pinned-offsets
  pattern where a study is re-run.
- **Size: complex.** The five triggers:
    - *What gets stored:* fires. The ENS-horizons re-run writes a new results folder under
      `data/studies/` and changes the numbers on a published page.
    - *The production serving path:* does not fire. Only `packages/studies` and `studies/` change.
    - *A degradation rule:* does not fire. Study code fails fast by design.
    - *More than one defensible design:* fires. Offset era folds, whole-record folds, and a row-set
      restriction each answer part of the problem.
    - *Callers you could not name without searching:* fires. `assign_folds` and `with_eras` have
      about 20 callers across `studies/beam_diffuse_split/`.
- **Reviews:** both plan reviews, then the first diff review and, for the rewritten page, the
  `study` skill's two science reviews. No mutation pass, because nothing changes in
  `packages/studies`.
- **Departure from the issues:** #868 suggests re-fitting one study; the maintainer asked for a
  measurement on one wind and one solar study before any published re-run, and for #892 to be
  planned with it.

## Measurement (scratch only, nothing under `data/studies/`)

Scripts and outputs live in `.claude/worktrees/scratch/era-fold/`. Design D0 is today's folds, D1
offsets each era's fold numbers so every calendar month is covered (`search_fold_offsets`), and D2
is whole-record contiguous folds with no era cut. Part A counts scored hours in uncovered (site,
fold, calendar month) cells. Part B refits the planned-contrast arms of one wind and one solar study
under D0, D1 and D2, and reports each contrast with its 95% interval and its change from D0. Part C
refits the ENS-horizons wind and solar arms with a 49r1 era cut, and with rows from 2024-12-01, and
compares the page's claims, including ENS day-0 minus ERA5 wind (+0.170 [+0.021, +0.322]).

### Results

The full report, with every table, is `.claude/worktrees/scratch/era-fold/out/report.md` (not
committed; it holds farm-level labels only as W1 to W3 and A to F). Every contrast below is in
percentage points of capacity, with a 95% interval that resamples whole calendar months and a seed.
Each refit at design D0 reproduces the published per-row losses bit for bit (wind, solar and the ENS
wind fits) or to 1.5e-7 of capacity (the ENS solar fits), at both hyperparameter settings.

**Coverage under today's folds (design D0).** In the wind study (50,734 farm-hours), 8,326 hours
(16.4%) sit in cells where the held-out calendar month, June or July, has no training row, at each
of the three farms. In the solar `long` panel (76,727 farm-hours), 2,129 hours (2.8%) do, at one
farm, in May, June and July. `cut_eras` with zero offsets reproduces `with_eras` fold for fold in
both studies. Rotating the post-upgrade era's fold numbers by 2, 3 or 4 covers every calendar month
in both. Whole-record folds (D2) cover every month too, but 13,218 of 15,480 solar post-upgrade
hours (85%) are then scored by a model that trained on no post-upgrade row.

**Planned contrasts of the two published studies.** At the primary setting no planned contrast
changes sign under D1 (rotation 2) or D2. At the second setting one does: wind ICON-EU minus UKV goes
from +0.082 [−0.022, +0.178] under D0 to −0.001 [−0.107, +0.091] under D2, with an interval that
includes zero on both sides. Under D1 the change from D0 is at most 0.033 points (wind) and 0.043
(solar, across rotations 2 to 4) at the primary setting, and at most 0.052 at the second setting.
Its interval excludes zero for two solar contrasts at rotation 2 (CAMS minus ICON-D2, −0.030
[−0.059, −0.003]; ICON-EU minus ICON-D2, −0.033 [−0.068, −0.004]); at rotations 3 and 4 only CAMS
minus ICON-D2 remains. At the second setting the wind ICON-EU minus ERA5 and UKV minus ERA5
contrasts move by +0.05 under D1, with intervals that exclude zero, and stay significant. Under D2
only wind ICON-EU minus UKV loses its significance at the primary setting (+0.126 [+0.009, +0.232]
to +0.040 [−0.060, +0.133]), and the largest change is solar ICON-EU minus UKV, −0.176 [−0.289,
−0.068]. Every fitted arm's absolute error moves by 0.00 to 0.19 points. A single fold assignment
carries a spread of 0.02 to 0.04 points across the three covering rotations, which no interval
includes.

**Coverage of the published pages the measurement did not fit.** The second plan review computed
uncovered hours from the folds saved with each published loss file. `past_weather_v2/solar_all`, the
past-solar leaderboard's row set with three planned contrasts, has 14,853 of 40,243 hours (36.9%) in
uncovered cells, in April to June at every farm. `wind_icon_dream` has 12,570 of 50,041 (25.1%), the
blending page 9,520 of 128,350 (7.4%), the wind page 8,603 of 52,996 (16.2%), `ens_past_solar`
1,194 of 54,447 (2.2%), and the `long` panel 2,129 of 76,727 (2.8%). `ens_hres_past_wind`, the
station arms and the NWP-comparison studies have none. The `all` and `wind_icon_dream` panels carry
9 to 13 times the `long` panel's share, and the measurement so far says nothing about how far
their planned contrasts move; a D1 refit of both is queued in the measurement.

**The ENS-horizons page (#892).** ENS day-0 mean minus ERA5 for wind, published as +0.170 [+0.021,
+0.322], is already +0.083 [−0.064, +0.239] under today's folds at the second setting. It becomes
+0.136 [−0.012, +0.291] once the 19 days straddling the 49r1 change are dropped (a row trim, not an
era cut), and +0.001 [−0.119, +0.132] with a three-era cut. Re-cutting today's two UKV-era folds on
rows from 2024-12-01 without a rotation (design II as first measured) leaves 37.7% of wind hours and
41.7% of solar hours uncovered, so the first design-II numbers (−0.071 for this contrast, +1.404 at
day 14, −1.778 for solar day 7) come from a fold design worse than the one being fixed and are not
used. The second plan review refitted design II with covering rotations 2 and 3 at the primary
setting. Under rotation 2 the wind contrast is −0.031 [−0.160, +0.118], the wind day-14 mean minus
climatology +1.115 [+0.517, +1.741], the solar day-7 mean minus climatology (planned) −0.905
[−1.639, −0.231], the solar day-14 mean minus climatology +0.178 [−0.678, +0.917], and the wind
day-0 control member minus ERA5 +0.140 [+0.018, +0.285]. Rotation 3 differs by at most 0.07 points.
So the headline wind claim does not survive design I or covered design II, most of the move of the
solar day-7 result to −1.778 was the fold defect and not the shorter record, and the solar day-14
claim that climatology beats the mean stops being significant. On design I minus D0trim, the wind
planned contrasts day-7 mean minus day 0 and day-7 mean minus climatology move by +0.161 and +0.194
(intervals include zero), and the exploratory control day 0 minus ERA5 and day-0 mean minus UKV with
ICON-EU by −0.199 [−0.346, −0.071] and −0.164 [−0.281, −0.054]. The at-most-0.10-point statement
holds for solar only. The measurement's final covered design II (IIr) figures, at both settings, are in
`studies/era_fold_design/report.md`; they are wind day 0 minus ERA5 −0.026 [−0.151, +0.114], wind
day-14 mean minus climatology +1.207 [+0.514, +1.915], solar day-7 mean minus climatology −0.868
[−1.603, −0.188], and solar day-14 +0.186 [−0.667, +0.931]. Where they differ from the review
refits above, the report's figures govern.

**What the measurement did not cover.** The three-era design tried one of six covering wind offset
sets (two for solar). The upsampling choice was not re-run per design. The member-by-member arm was
refit at day 1 only. The second-setting ENS fits reproduce the published wind losses bit for bit;
the solar second setting has not been reproduced.

### Recommendation

Keep today's folds (design D0) as the library default and change no library code for #868. Publish
the measurement on #868 as the answer to its "measure the effect first" option, with the coverage
table above. Where a study is re-run for another reason, pin a covering offset map for that study's
own row set, in the way `ens_hres_past_wind.py` does (`rotate_folds` on top of `with_eras`, with a
pinned offsets constant, and `raise_on_uncovered_months` before the first fit), so a change of row
set fails loudly instead of moving folds silently. Reject whole-record contiguous folds (D2),
because they score most post-upgrade solar hours with a model that never saw the new UKV, which is
the failure era-cutting exists to prevent.

The "limitation sentence and no re-run" conclusion is scoped to what was measured: the planned
contrasts of the wind page and of the past-solar `long` panel, where D1 moves no planned contrast by
more than 0.033 points (wind) or 0.043 (solar), or by more than 0.052 at the second setting, and
changes no sign at the primary setting. Each page's limitation sentence quotes its own uncovered
share. For the `all` and `wind_icon_dream` panels the recommendation waits for the queued D1 refit,
and the plan recommends a re-run where that refit moves a planned contrast by more than the
covering-rotation spread (0.04 points) or changes its significance.

For #892, rewrite the ENS-horizons page's wind claims around rows from 2024-12-01 with the pinned
rotation `{0: 0, 1: 2}` (covered design II), because its headline wind contrast, +0.170 [+0.021,
+0.322], does not survive an IFS 49r1 era cut in design I or in covered design II, and was already
not significant under today's folds at the second setting. Design II is the simplest design to write
into the study and shares the past-wind study's start date. The cost is 3.5 months of wind rows. The
page reports whatever the covered re-run prints for solar day 7 and day 14 against climatology, and
does not attribute any change to the shorter record without separating it from the fold effect.

## Departures from the first draft of this plan

The first simplicity review found that the first draft's library API had no caller, since the plan
re-runs nothing and keeps every default. The draft is replaced as follows:

- **Dropped:** `cut_eras_covering`, `allow_uncovered`, `with_eras(covering=)`, the `_complete`
  option, and `IFS_CYCLE_49R1_FIRST_MONTH`. Existing calls already express the covering design
  (`rotate_folds` over `with_eras`, as at `ens_hres_past_wind.py:2340`), and
  `IFS_CYCLE_49R1_CUT_MONTH` already exists in that script. If a second study needs it, move that
  constant into `studies.cross_validation` then.
- **Dropped:** the planned package tests, which repeated existing tests of `cut_eras`,
  `rotate_folds` and `search_fold_offsets`.
- **Kept, changed:** the ENS-horizons fix becomes a row filter, not an era option, because the
  script cuts folds a second time for its native-step-width arms (`ens_forecast_horizons.py`, the
  `assign_folds` call near line 1338), and a flag on `_complete` would not reach that cut, nor drop
  the straddle rows from 2024-11-12 to 2024-11-30.
- **Not adopted:** a run-time `search_fold_offsets`. A search moves folds silently when the row set
  changes, and a pinned constant fails loudly.

## What changes, file by file

- **#868:** a comment on issue #868 with the measurement tables and the recommendation above. No
  code changes.
- `studies/beam_diffuse_split/ens_forecast_horizons.py`:
    - `_complete` drops rows before `2024-12-01`, named `ROW_SET_FIRST_HOUR`, with a docstring saying
      why. Rows valid from that date at leads up to day 14 come from runs from 17 November onwards,
      all on Cycle 49r1. Whether solar is cut too is an open question below.
    - A pinned constant `ROW_SET_FOLD_OFFSETS: Final[Mapping[int, int]]` (`{0: 0, 1: 2}`), with a
      docstring naming the row set it was searched on. It does not reuse
      `ens_hres_past_wind.HORIZONS_ROTATED_FOLD_OFFSETS`, which belongs to a different row set.
    - Folds are cut with `rotate_folds(with_eras(...), ROW_SET_FOLD_OFFSETS)`, and
      `raise_on_uncovered_months` runs before any fit, for both domains. The native-arm cut near line
      1338 no longer calls `assign_folds` again: the native rows take the main frame's fold by a join
      on (site, month), so the two cuts cannot drift apart.
    - `RESULTS_DIR = OUTPUT_DIR / "from_2024_12"` receives every result file (`*_losses`,
      `*_predictions`, `leaderboard`, `intervals`, `report.md`). The input extract stays in
      `OUTPUT_DIR`. A fresh run raises if `RESULTS_DIR` already holds a `*_losses.parquet`, which
      makes the folder write-once. The run must be `--refit both`, since a partial refit reads the
      other domain's outputs from the empty folder.
    - `_dropped_lines` restricts its base frame to `ROW_SET_FIRST_HOUR` before the anti-join, so the
      deliberate filter is not reported as dropped hours.
    - The report prints the rows per domain, the pinned offsets and the uncovered-cell count (zero)
      for each fold cut.
- `studies/beam_diffuse_split/ens_forecast_charts.py` imports `RESULTS_DIR`. `ens_hres_past_wind.py`
  keeps reading the old folder (`ENS_INPUTS_PATH` and the horizons report for +0.170), because the
  past-wind page's cross-reference quotes the published figure; the past-wind page's two mentions
  (lines 121 to 122 and 992 to 1010) get a sentence saying the horizons page was re-run on rows from
  2024-12-01, in the same PR.
- `docs/studies/` (the ENS-horizons page, at its new path if PR #917 has landed): rewrite every
  fit-dependent claim from the covered re-run, at both hyperparameter settings for the planned and
  deciding contrasts. The second plan review listed them: the two lead paragraphs, Figures 1 to 11,
  all Key findings, the folds section, every Results section, all four "What to use" bullets, and
  the Limitations (about 21 months, the 49r1 and fold-pairing bullets, member spread, dropped hours).
  The upsampling combination is re-chosen by `choose_method` on the new rows, so the upsampling
  figures may change too.
- **No change** to `packages/studies`, `weather_products.with_eras`, or any other study.

**The ENS-horizons re-run writes new outputs under `data/studies/`, so it needs the coordinator's
go-ahead first.** It writes only to `RESULTS_DIR`, moves nothing a merged page quotes until the page
is rewritten, and runs every fit of a contrast on one device.

## Design-philosophy check

This is R&D code, so it fails fast: the coverage check raises before the first fit, as the
past-wind study's does. No production path, asset or asset check is touched.

## Tests

Study scripts have no unit tests, so the script's own output is its check. The re-run prints the
row count, the coverage result (zero uncovered cells) and each arm's columns into the report, and
the page's numbers are checked against that report by the number guard. Nothing changes in
`packages/studies`, so no mutation pass is planned; the PR body says so.

## Verification

`uv run ruff check .`, `uv run ruff format --check .`, `uv run ty check`, `pydoclint` (pinned
0.9.1), `uv run pymarkdown scan -r docs README.md CLAUDE.md packages/*/README.md`,
`uv run pytest`, `uv run mkdocs build --strict` with a read of the rendered page, and
`scripts/lint/check_docs_links.py`. The number guard is run once per section:
`uv run python studies/beam_diffuse_split/check_page_numbers.py --section <heading>` for every
Results heading, Key findings, What to use and Limitations, and `--bullet` for the lead paragraphs
if it supports them. Because the guard passes a bare number shifted by 0.01 about 70% of the time,
the review also greps the page and the past-wind page for superseded values (`0.170` outside the
deliberate cross-reference, `50,268`, `50,643`, `April 2024`, `August 2024`, `8 of the 30`,
`8 of the 26`, `4,148`, `54,791`) and checks that the report prints the rows per domain and zero
uncovered cells for both fold cuts.

## Risks and open questions

- **Design II versus the three-era cut (I).** Both make the headline wind contrast
  non-significant. Covered design II is simpler and shares the past-wind study's start date; the
  three-era cut keeps 3.5 more months of rows and duplicates `long_row_designs`' cut design in
  `ens_hres_past_wind.py`, which reports independent results for the same offsets. The recommendation
  is II; the coordinator can choose I.
- **Should solar be cut at 2024-12-01?** The row filter is shared by both domains and cuts solar
  from 50,643 to 37,166 hours (27%), with no measured evidence of a 49r1 step in solar (design I
  moves solar contrasts by at most 0.10 points). Recommendation: make the start date wind-only,
  keep solar's rows, and pin a covering rotation for solar's own row set. If solar is cut for
  comparability, the page must separate the row-set effect from the fold effect.
- **Rotation 2 is an arbitrary covering choice.** `search_fold_offsets` returns it first, and
  rotations 3 and 4 also cover. The page says so and quotes the spread across rotations; no
  interval includes the fold-assignment spread of 0.02 to 0.07 points.
- **Second-setting and panel refits are done.** Their results are in `studies/era_fold_design/report.md`
  and replace the provisional covered design II values above.
- **Whether the `all` and `wind_icon_dream` pages need a re-run.** The D1 refits moved no sign or
  significance, so the recommendation is no re-run. The coordinator decided.
- **Size and role wording.** "Coordinator" and "maintainer" mean the same decision-maker here.
