# Plan: score autonomous research with the reviewed scorer, from `main`

**The problem.** Autonomous studies score their forecasts with study-local code on study-local folds, so a study number cannot be compared with a leaderboard number. Nothing stops an autonomous session from editing the scoring code or from reading the validation actuals it is scored against. The docs also disagree on which tier owns a leaderboard candidate: design principle 3 says research runs on the production pipeline, while `studies/README.md` says studies never touch it.

**The plan.** Keep the scorer where it is (`ml_core.metrics`, `ml_core.cv_helpers`, `conf/cv/default.yaml`, and the Dagster `metrics` asset). Protect it with three cheap mechanisms. First, an import rule keeps the two scoring modules free of Dagster, MLflow, and study code. Second, an autonomous study's leaderboard number comes only from running the `metrics` asset from the `main` checkout, which a session working in its own worktree cannot edit. Third, the `metrics` asset gains two refusals: a forecast that omits rows of the fold, and a window past a final-test date. Tail metrics (#254) go straight into `compute_metrics`. The study-code refactor (#861, #829) becomes the last, mechanical phase.

## Verdict, size and departures

**Verdict: worth doing, and Phases 1 and 2 are worth doing before v0.3.** Every leaderboard row written before these rules exist is a row whose definition can drift.

**Size: complex.** One line per trigger:

- **What gets stored:** no new table and no contract change. The `forecast_metrics` table gains rows whose `experiment_name` starts with `study/`.
- **Production serving path:** no. The `metrics` asset is R&D, and the live service does not import it.
- **A degradation rule:** no.
- **More than one defensible design:** yes. The boundary could sit at a shared training framework, a separate repository, a new package, or the existing scorer.
- **Callers not nameable without searching:** yes. `cv_assets.py` is 1,067 lines, and the change edits the `metrics` asset in it.

The asset trigger (an edit to an asset) also fires, so the size is complex. Reviews: the simplicity review of an earlier, larger draft is done and its findings are applied below. The correctness review of this plan, and both diff reviews, are still to run.

**Departures from the earlier discussion, each with its reason:**

- **No new `evaluation` package.** `ml_core.metrics` and `ml_core.cv_helpers` already import only `polars`, `patito`, `contracts`, and `pydantic`, so they are already pure. A new package would buy nothing but a move.
- **No `score()` entry point and no registry.** The `metrics` asset already loads the actuals, refuses nothing it should not, and writes `forecast_metrics`, which carries the fold, experiment, and model name. MLflow run tags already record the git commit.
- **No kernel hash, stale-row filter, or CI path-guard script.** Running the scorer from `main` prevents tampering, where a hash only detects it. Branch protection does not exist yet, so a failing check would block nothing.
- **No epoch column on `Metrics`,** and no Benjamini–Hochberg control, holdout-ladder rounding, or job queue. Each waits until the volume of studies justifies it.
- **Published studies are not re-scored,** and their study-local scoring code is not deleted. Their scoring runs per site over five month-block folds on past-weather reanalysis, so they cannot be scored on the leaderboard rows. Deleting it would break reproduction of published pages.
- **No quantile-to-member adapter.** A study that wants a leaderboard number submits ensemble members. A second, approximate scoring definition inside the scorer would defeat the purpose.
- **The Fractions Skill Score stays out of the scorer.** It is a spatial weather metric, not a power score. Only its duplicate copy is removed, in Phase 4.

## What changes, in phases

**All phases sit under the v0.3 epic (#6).** Phases 1 and 2 land before v0.3 releases. Phase 3 is #254 and #226. Phase 4 can follow.

### Phase 1: protect the scorer (about half a day)

- **Add an import-linter contract to `pyproject.toml` and CI.** `ml_core.metrics` and `ml_core.cv_helpers` may not import `dagster`, `mlflow`, or `studies`. No package outside `packages/studies` and `studies/` may import `studies`. This turns the paragraph in `studies/README.md` into a check.
- **Set up the separate Unix user** described under "Cheating", and optionally add `.claude/settings.json`, which does not exist today, denying `Edit` and `Write` for `packages/ml_core/src/ml_core/metrics.py`, `packages/ml_core/src/ml_core/cv_helpers.py`, `conf/cv/**`, and `packages/contracts/**`. Bash can bypass this rule, so it is a guard against accidents and casual edits, not against a determined session.
- **Optionally add a CODEOWNERS file and turn on branch protection** for the same paths, so a change to them needs the maintainer's review. Whether to turn it on is the maintainer's call.
- **Amend principle 3** in `docs/design-philosophy/design-principles.md`, and `studies/README.md`. For reviewed research, principle 3 stands. For autonomous studies the rule becomes: one scoring path for all research, and one execution path from reviewed research to production. A study finding reaches production only as a written spec and a reviewed re-implementation, never as merged study code.

### Phase 2: make the `metrics` asset the only scorer (about two days)

- **Add the row-set refusal to the `metrics` asset.** If a forecast omits any `(time_series_id, power_fcst_init_time, valid_time)` row that the fold's eligible series should cover, the asset raises, so an agent cannot raise its score by abstaining on hard rows.
- **Add `FINAL_TEST_START`** to the fold configuration as a single date. The `metrics` asset refuses any window reaching past it unless `NGED_FINAL_TEST=1` is set, which only the maintainer's shell does. The study power reader in `packages/studies` truncates at the same date. This is #226 in its smallest form. It does not seal the data physically, because studies could read the Delta table directly.
- **Write a small script, `scripts/score_study.py`,** that materialises the `metrics` asset for a study's predictions from the `main` checkout, with `evaluation_scope="leaderboard"` and an `experiment_name` prefixed `study/`. The autonomous session runs the script but cannot edit what it runs, because the deny rule covers the scorer files and the `main` checkout is not the session's worktree.
- **Add a `study` skill rule and a reviewer-checklist line:** a study's leaderboard number comes only from `scripts/score_study.py`, and every number on a study page traces to a `forecast_metrics` row.
- **Label rows** through the `study/` prefix on `experiment_name`, so the leaderboard chart can show or hide them and no promotion path reads them.

### Phase 3: tail metrics and the final-test window (about two days, after v0.3)

- **Land #254 (twCRPS, exceedance rate, Brier score) inside `compute_metrics`.** Note that `effective_capacity` in `cv_assets.py` is a full-history P99 of the absolute power, which includes the validation window. If the exceedance thresholds follow that convention, they see validation outcomes. Decide in #254 whether to compute them from the training window instead.
- **Report the number of exceedance days beside each tail metric,** and refuse a tail contrast below a minimum event count.
- **Finish #226:** the maintainer scores the final-test window rarely and deliberately, with `NGED_FINAL_TEST=1`.

### Phase 4: the study-code refactor (about a week, last, deferrable)

- **Supersedes #861 and #829.** Close #829 as a duplicate of #861, and keep #861 as this phase.
- **The scoring change makes one item of #861 obsolete:** moving scoring helpers into `packages/studies`. Everything else stays: move the shared source readers, the dataset build, and the fit helpers into `packages/studies` with tests, give each study its own folder, and update every script path quoted in `docs/`.
- **Do it as a mechanical `git mv` plus a path rewrite,** verified by an import smoke test over every script and by the docs link check. Do not re-run every study's headline. That is a week of compute, and the worktrees share one data folder, so parallel re-runs overwrite each other's evidence. Re-run only a script whose code, not just its path, changed.
- **Remove the second copy of the Fractions Skill Score** and move #902's `calendar_balanced_difference` into `packages/studies/bootstrap.py` with tests. #902 stays its own small issue.

## Cheating

**The design follows Karpathy's autoresearch: the agent may edit only the model, and the evaluator is a file it cannot change.** Here there are three layers, in order of strength.

1. **Autonomous sessions run as a separate Unix user.** The maintainer chose a Unix user over Docker. The agent user can read the training data, and read and write its own worktrees and `data/studies/`. It cannot read the validation and final-test actuals, and cannot write to the `main` checkout or the scorer files. It has its own Claude, CDS, and AWS credentials, so it cannot read the maintainer's. Setup needs the user, `setfacl` on the data folders, setgid directories with umask 002, its own `uv` cache, and tightening `/mnt/data` (world-writable today, mode 2777). Claude Code's built-in Bash sandbox (needs `socat` and an AppArmor profile for `bwrap`, with `failIfUnavailable` set) can sit inside this user as a network filter.
2. **The scorer runs from `main`, as the maintainer's user.** The agent hands over a predictions parquet file, through one narrow `sudo` rule that runs only `scripts/score_study.py`. The script must resolve the path, refuse symlinks, and copy the file into a directory the agent cannot then alter. A session editing scoring code in its own worktree changes nothing that the script executes.
3. **The `metrics` asset owns the actuals and the row set,** so a study cannot pass its own targets, drop hard rows, or score past the final-test date without the maintainer's environment variable.

**Residual risk.** A missed ACL on a new data folder exposes actuals, a shared `/tmp` is visible to both users, and the `sudo` rule is a small attack surface. The real defences are the reviewed re-implementation of any finding, and the reviewers. The `.claude/settings.json` deny rule and branch protection become optional extras, because the operating system now enforces what they would only ask for.

## Design-philosophy check

- **Inherent stability:** the change is R&D-only and fails fast. It touches nothing in `defs/production_assets.py` or the serving path.
- **Principle 3 is amended,** and the price is stated: study code is rewritten on its way to production. What is bought is that the autonomous side moves without the maintainer reading its code.
- **Principle 4 holds:** studies extend reviewed code only through a reviewed pull request.
- **Principle 8 is strengthened:** identical scoring is enforced by where the scorer runs, not by discipline.
- **Hypothesis H2** counts experiments one person registers. With autonomous studies the meaningful rate becomes confirmed promotions per month. Update the wording in Phase 3, appending a label and never renumbering.

## Tests

- **The `metrics` asset raises on a forecast missing rows of an eligible series.** On `main` it silently scores what it is given.
- **The `metrics` asset raises on a window past `FINAL_TEST_START` without the environment variable, and accepts it with the variable set.**
- **The study power reader returns no rows after `FINAL_TEST_START`.**
- **`lint-imports` fails if `ml_core.metrics` gains an import of `dagster`, `mlflow`, or `studies`.** This is run in CI, not in pytest.
- **Phase 3:** the twCRPS, exceedance, and Brier tests named in #254.
- **Phase 4:** an import smoke test over every study script.

## Docs to update

- `docs/design-philosophy/design-principles.md` (principle 3) and `docs/design-philosophy/engineering-hypotheses.md` (H2 wording).
- `docs/roadmap/metrics-and-leaderboard.md`: the `study/` prefix, the final-test date, and the row-set refusal.
- `docs/ml_experimentation/index.md`: the autonomous-study route and the promotion path.
- `studies/README.md` and the `study` skill: `scripts/score_study.py`, the reviewer-checklist line, the deny rule.
- `CLAUDE.md`: the `study` row of the skills table stays in step.
- The two Stage A / Stage B ideas go in `docs/roadmap/xgboost-improvements.md` in a separate PR.

## Issues absorbed

- **Superseded:** #829 (closed into #861), and #861 becomes Phase 4.
- **Folded in:** #226 (Phases 2 and 3) and #902 (Phase 4).
- **Land in `compute_metrics`, otherwise unchanged:** #254 and #225.
- **Consume the `study/`-labelled rows:** #4 (visual leaderboard) and #808 (interval on a score difference), which needs only `forecast_metrics` rows.
- **Interact but stay separate:** #350 (four daily inits changes the row set), #438 (failure-scenario scoring), and #147 (baselines).
- **Not checked beyond the title:** #229, #715, #362.

## Verification commands

```bash
uv run ruff check . && uv run ruff format --check . && uv run ty check
uv run pytest
uv run lint-imports
uv run pymarkdown scan -r docs README.md CLAUDE.md packages/*/README.md
uv run mkdocs build --strict
```

Plus the pydoclint and docs-link checks that CI runs. Phase 4 also needs the mutation pass the `study` skill requires for any change to `packages/studies`.

## Risks and open questions

- **How does a study write `PowerForecast` rows into the store the `metrics` asset reads?** Recommendation: `scripts/score_study.py` takes a parquet file, validates it against the contract, writes it under the `study/` experiment name, and materialises `metrics`.
- **Should `NGED_FINAL_TEST` gate on more than an environment variable?** Recommendation: no. One user, one workstation.
- **Will reviewed re-implementation keep pace with the specs?** Recommendation: decide now that study code is never merged into reviewed packages.
- **Should Phase 4 wait?** Recommendation: yes. Outcomes 1 and 2 do not depend on it, so it can follow v0.3.

## Longer ENS history and the retraining schedule (added after discussion; to be decided)

**A staged ECMWF ENS archive may replace the interim "research on CEDA UKV, confirm on ENS" route.** Dynamical.org stages raw MARS GRIB1 files at `s3://us-west-2.opendata.source.coop/dynamical/ecmwf-ifs-grib/ecmwf-ifs-ens/{date}/`, readable anonymously, not requester-pays, 00Z only. Complete dates run from 2020-11-12 (patchy to 2021-03-20) to 2024-03-31. Late 2019 to autumn 2020 has no surface fields. A second reviewer's findings, still to be verified:

- **Which members to fetch depends on the use.** Training uses only the control member today, but inference and scoring use all 51 members, and we plan to experiment with training on all of them. Every validation month in a rolling-origin evaluation therefore needs all 51 members, or the continuous ranked probability score and interval coverage cannot be computed. The control member alone is about 1,020 range requests and 0.5 GB per date, so about 4–5 hours and 530 GB for 2021-03-21 to 2024-03-31. All 51 members is about 52,020 requests and 25 GB per date, so about 64 million requests and 30 TB in total. That is about 6–14 days on the workstation, or 4–8 hours on a cloud machine in `us-west-2`, at an estimated $10–30. Both figures are extrapolated from small measurements.
- **A staged fetch limits the cost.** Fetch the control member for the whole period first, then all members only for the months that will be scored, and for any period used to test training on members. The first year of a rolling-origin evaluation is training-only, so it can stay control-only.
- **A cloud machine is also kinder to the source.** The workstation route sends tens of millions of requests through a small provider's proxy, which reset connections at high concurrency in testing. A machine in `us-west-2` reads the bucket directly.
- **Ask Dynamical.org first** about their Zarr backfill ETA, and whether the staging bucket is permanent. A Zarr backfill would need no new code.
- **The hand-written GRIB decode must match eccodes bit for bit** on `10u`, `2t`, and `ssrd` at step 360, and each range must start at byte 0 so the decimal scale factor is read. De-accumulation must match Dynamical.org's clipping rule.
- **The archive contains seams:** the ENS resolution change (reported as 2023-06-27, IFS cycle 48r1), the 2024-04-01 product change, and 2024-11-13 (cycle 49r1). Each is an era boundary under the `study` skill.
- **A pilot decides it:** the control member for 2024-03-31 and 2023-06-26/28, a bit-exact eccodes comparison, `Nwp.validate`, and lead-0 error against ERA5 across the 2024-04-01 seam.

**Retraining schedule options.** Production will retrain much more often than once a year, and the cadence is undecided, so a single train window followed by one validation year is unrealistic. The options, all scored with a paired moving-block bootstrap over calendar months of the concatenated out-of-sample predictions (blocks of 2–3 months, because adjacent test months share a training set):

1. **One origin per validation year.** Cheapest, and stale by up to 12 months, unlike production.
2. **Rolling origin at a fixed cadence** (for example quarterly), expanding window, refitting at every origin. Predictions from all origins are concatenated and scored once. It mimics production, and costs one refit per origin.
3. **Option 2 plus a staleness report.** Also report skill by months since the last refit. It costs nothing extra and provides the data to choose the cadence.
4. **Sparse origins, fresh slice only.** Refit every 6 or 12 months and score only the first 3 months after each. Cheaper than option 2, and covers only some months.
5. **Discovery lane.** Whole-month-block cross-validation, era-aware, which trains on the future and so is not causal. It ranks ideas cheaply, and rolling origin confirms them.
6. **Cheap updates.** Continued boosting monthly and a full retrain quarterly, if production will do the same.

**Recommendation for discussion:** option 3 for the leaderboard and option 5 for autonomous discovery, with the last 12 months sealed as the final test. Whole calendar months are the resampling unit, never series, because neighbouring generators are correlated. Bootstrap intervals are conditional on the fitted models, so also report seed-to-seed variation for a few arms. Each origin trains only on rows with a target time before the origin, and lag features stay nulled by `_nullify_leaky_lags`. Compute cost scales with the number of origins times the number of series, so 2,500 series will need a cheaper cadence than 32.

## Appendix: related research ideas from the same discussion

**A two-stage weather-to-power model,** for the roadmap. Stage A learns weather to power from best-estimate weather only (CAMS, ERA5, analyses), then is frozen. Stage B takes Stage A's output on forecast weather, plus raw forecast weather, lead time, and lagged power, and learns to calibrate uncertainty and correct with lags. Reviewer conclusions:

- It is the classical "perfect prog" method followed by output statistics. A model trained on forecast weather learns the response smoothed by forecast error, so a frozen Stage A will be over-confident at long lead.
- Stage B needs raw forecast weather too, so it becomes stacking.
- The plausible gain is calibration and diagnosis, not skill.
- Test it in a short study against strong single-stage baselines (lead-time feature, ensemble statistics, training on members, warm start on ENS), scored on identical rows by lead, with a kill criterion. The out-of-sample requirements are already written in `docs/roadmap/switching-events.md`.

**Longer UKV history** stays a lane only for questions ENS cannot answer: 2 km resolution at short leads, and weather from late 2019 to March 2021. CEDA UKV is a different product from live UKV, so no result trained on one and applied to the other is trusted.
