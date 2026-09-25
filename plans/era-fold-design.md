# Era-boundary fold design for the studies (issues #868 and #892)

**The problem.** Every study that cuts folds within each UKV era (`weather_products.with_eras`,
`assign_folds(by=("site", "era"))`) reuses fold numbers 0 to 4 in both eras, so a fold can hold the
same calendar month out of both eras at once and the fitted model then never sees that season
(#868). The ENS-horizons page also trains across ECMWF's IFS Cycle 49r1 change (live 2024-11-12)
with no era cut, although the past-wind study found a step in ENS's and HRES's 10 m wind at that
date (#892).

**The plan.** Measure both effects first, in scratch, on one wind study and one solar study. Then
publish the measurement on #868 with a recommendation and no library change, and rewrite the
ENS-horizons page's wind claims (#892) from a re-run on rows from 2024-12-01. The re-run needs the
maintainer's go-ahead, because it writes new outputs under `data/studies/`.

## Verdict, size and departures

- **Verdict:** worth doing. The past-wind study (PR #885) already solved the problem for its own row
  set with `cut_eras` and `ENS_HRES_WIND_ERA_FOLD_OFFSETS`, so the machinery exists and the
  remaining work is measuring the effect on the other pages and applying the same pinned-offsets
  pattern where a study is re-run.
- **Size: complex.** The five triggers:
    - *What gets stored:* fires. It adds options to `studies.cross_validation`, and a published
      page's numbers can change if the maintainer chooses to re-run.
    - *The production serving path:* does not fire. Only `packages/studies` and `studies/` change.
    - *A degradation rule:* does not fire. Study code fails fast by design.
    - *More than one defensible design:* fires. Offset era folds, whole-record folds, and a row-set
      restriction each answer part of the problem.
    - *Callers you could not name without searching:* fires. `assign_folds` and `with_eras` have
      about 20 callers across `studies/beam_diffuse_split/`.
- **Reviews:** both plan reviews, then both diff reviews (correctness, then mutation, since
  `packages/studies` changes).
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

**Planned contrasts of the two published studies.** No planned contrast changes sign under D1
(rotation 2) or D2. Under D1 the change from D0 is at most 0.03 points at the primary setting, and
its interval excludes zero for two solar contrasts (CAMS minus ICON-D2, −0.030 [−0.059, −0.003];
ICON-EU minus ICON-D2, −0.033 [−0.068, −0.004]). At the second setting the wind ICON-EU minus ERA5
and UKV minus ERA5 contrasts move by +0.05 under D1, with intervals that exclude zero, and stay
significant. Under D2 only wind ICON-EU minus UKV loses its significance (+0.126 [+0.009, +0.232] to
+0.040 [−0.060, +0.133]), and the largest change is solar ICON-EU minus UKV, −0.176 [−0.289,
−0.068]. Every fitted arm's absolute error moves by 0.00 to 0.19 points.

**The ENS-horizons page (#892).** ENS day-0 mean minus ERA5 for wind, published as +0.170 [+0.021,
+0.322], becomes +0.136 [−0.012, +0.291] once the 19 days straddling the 49r1 change are dropped,
+0.001 [−0.119, +0.132] with a three-era cut, and −0.071 [−0.218, +0.084] on rows from 2024-12-01.
At the second setting it is already +0.083 [−0.064, +0.239] under today's folds. The wind claim that
climatology beats the day-14 ensemble mean by 1.17 points is not significant under the three-era cut
(+0.618 [−0.494, +1.602]) and is significant on rows from 2024-12-01 (+1.404 [+0.817, +2.005]). The
solar day-7 planned contrast against climatology is −0.157 [−0.581, +0.238] under today's folds and
−1.778 [−2.719, −0.890] on rows from 2024-12-01, because climatology gets worse on the shorter
record. The other planned and solar contrasts change by at most 0.10 points under the three-era cut,
with paired intervals that include zero, apart from wind's member-by-member minus control.

**What the measurement did not cover.** The three-era design tried one covering offset set. The
upsampling choice was not re-run per design. The member-by-member arm was refit at day 1 only. The
second setting was fitted for the ENS wind fits under D0 and the trimmed rows only, and is not yet
done for the three-era design or the solar fits (about 1.7 hours at 16 cores).

### Recommendation

Keep today's folds (design D0) as the default and change no library code for #868. Publish the
measurement on #868 as the answer to its "measure the effect first" option. Where a study is
re-run for another reason, pin a covering offset map for that study's own row set, in the way
`ens_hres_past_wind.py` does (`rotate_folds` on top of `with_eras`, with a pinned offsets constant,
and `raise_on_uncovered_months` before the first fit), so a change of row set fails loudly instead
of moving folds silently. Reject whole-record folds (D2), because they score most post-upgrade
solar hours with a model that never saw the new UKV, which is the failure era-cutting exists to
prevent. Under D1 no planned contrast of the past-wind or past-solar study moves by more than 0.03
points (0.05 at the second setting) or changes sign, so those pages need a limitation sentence and
no re-run.

For #892, rewrite the ENS-horizons page's wind claims around rows from 2024-12-01 (design II),
because its headline wind contrast, +0.170 [+0.021, +0.322], does not survive an IFS 49r1 era cut
under any of the three designs tried. Design II is the simplest design to write into the study,
and it gives the page the same row set as the past-wind study, so the two pages compare directly.
The cost is 3.5 months of rows, and the shorter record weakens climatology, so the page must say
that its solar day-7 result against climatology moves from −0.157 to −1.778 for that reason.

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
- `studies/beam_diffuse_split/ens_forecast_horizons.py`: `_complete` drops rows before
  `2024-12-01`, a named constant `ROW_SET_FIRST_HOUR` states that date and why, and a coverage check
  (`calendar_month_coverage`, `raise_on_uncovered_months`) runs on the new row set before any fit.
  If the check finds uncovered cells, the script pins an offsets map found once with
  `search_fold_offsets` and applies it with `rotate_folds` to both fold cuts (the main one and the
  native-arm one). The script's default-run behaviour is otherwise unchanged.
- `docs/studies/` (the ENS-horizons page, at its new path if PR #917 has landed): rewrite the wind
  claims and the affected figures from a re-run under design II, at both hyperparameter settings for
  the planned and deciding contrasts. Re-check every solar claim. Say that the record starts on
  2024-12-01 and why, and report the day-7 climatology change.
- **No change** to `packages/studies`, `weather_products.with_eras`, or any other study.

**The ENS-horizons re-run writes new outputs under `data/studies/`, so it needs the coordinator's
go-ahead first.** It uses a new write-once output folder, moves nothing a merged page quotes until
the
page is rewritten, and runs on one device (the measurement used the CPU).

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
`scripts/lint/check_docs_links.py`.

## Risks and open questions

- **Design II versus the three-era cut (I).** Both make the headline wind contrast
  non-significant. Design II is simpler and comparable with the past-wind study, and the three-era
  cut keeps 3.5 more months of rows. The maintainer can choose the other; the recommendation is II.
- **Second-setting fits are incomplete for the three-era cut and for solar.** They run now, and
  their results go into the report before the re-run is designed.
- **Whether the past-weather and past-solar pages need a re-run.** The recommendation is a
  limitation sentence only. The maintainer decides.
- **Coverage on the shorter record.** With 22 calendar months, the covering offsets may differ from
  the long record's. The script must search once and pin the result.

