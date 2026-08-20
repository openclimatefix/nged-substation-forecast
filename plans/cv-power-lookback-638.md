# CV power-history window truncates lag/rolling features near every fold's start (#638)

**The problem.** `load_engineering_inputs` loads observed power filtered to exactly
`[window_start, window_end]`. `trained_cv_model` and `cv_power_forecasts` (`cv_assets.py`) pass
`window_start=train_start` / `window_start=val_start` straight through, with no look-back margin.
`_apply_power_lag` joins each row to that same power frame on `target_time = valid_time -
lag_hours`, so for any row whose lag target falls before `window_start`, the join misses and the
lag is silently null — even though the history exists on disk (`eligible_time_series` already
requires `min_training_months` of prior coverage). Every trained fold and every leaderboard
comparison has therefore been computed with power lag/rolling features nulled near each fold's
start. `live_forecasts` (`production_assets.py`) already avoids this by subtracting a fixed
`LIVE_POWER_HISTORY` (15 days) from `window_start` before calling the same loader.

**The fix.** Add a `power_lookback: timedelta` parameter to `load_engineering_inputs` that widens
*only* the power scan's lower bound (`window_start − power_lookback`), leaving the NWP `valid_time`
and `init_time` bounds untouched. `trained_cv_model` and `cv_power_forecasts` derive it
automatically per experiment from `ParsedFeatures(config.selected_features).max_power_lag_hours()`,
rather than a single hard-coded constant sized to the worst case across every experiment that could
ever exist. This is not a novel design: it is the "Power-lag lookback at feature-engineering load
time" item already designed in `docs/roadmap/metrics-and-leaderboard.md`'s PR 2 — this plan
implements that one item now, ahead of the other two items bundled with it there
(`uses_nwp_ensemble`, `ensemble_member` docs), which are out of scope for #638.

## Verdict, size and departures

**Verdict: worth fixing, roughly as described, but not by the mechanism the issue suggests.** The
issue's premise is factually correct — verified against `cv_assets.py:334-341` (`trained_cv_model`)
and `:486-494` (`cv_power_forecasts`): neither passes any look-back margin, unlike
`live_forecasts` (`production_assets.py:299-307`).

**Size: Medium.** Not Simple — there is real design work (a new parameter, deriving it from
`selected_features`, and a non-obvious correctness argument for *why* it must be a separate
parameter rather than a wider `window_start`, see below). Not Complex — it changes no Patito
model, Delta table or asset, touches no production serving path (`live_forecasts` is explicitly
left alone), and the design space has exactly one defensible answer once the NWP-centric bulk-mode
join is understood, not several to choose between. **Both plan reviews are worth running** (a new
parameter + a new `ParsedFeatures` method is exactly the "adds an abstraction" trigger for the
simplicity pass, and the account of current behaviour — the NWP-centric join, why widening
`window_start` itself would be wrong — took real digging to pin down, which is exactly the
correctness pass's trigger). Diff-review counts are `implement-issue`'s call, not this plan's.

**Departure from the issue.** The issue suggests "give `trained_cv_model` and `cv_power_forecasts`
the same look-back margin production uses, subtracted from `window_start` before calling
`load_engineering_inputs`" — i.e., mirror `live_forecasts`'s `window_start=power_fcst_init_time -
LIVE_POWER_HISTORY` pattern. **That mechanism would be a correctness bug, not just a style
choice, if applied to the CV callers:**

- `load_engineering_inputs` uses `window_start` for *three* things: the power scan's lower bound,
  the NWP `valid_time` lower bound, and (when `init_time_start` is not given explicitly) the
  default `init_time_start = window_start - MAX_NWP_LEAD`. Widening `window_start` itself would
  widen all three.
- `live_forecasts` gets away with this because it always passes an explicit
  `init_time_start=init_time_end=nwp_init` (a single run), which overrides the NWP init-partition
  default, and because it filters out the resulting early rows explicitly afterwards
  (`production_assets.py:326-329`, "History rows... are join artefacts, not genuine forecasts").
- `trained_cv_model` and `cv_power_forecasts` do neither. `trained_cv_model` does not override
  `init_time_start`, so widening `window_start` would also pull in up to two extra weeks of NWP
  partitions never needed. Worse: in bulk mode, `_join_nwp_bulk_mode`
  (`packages/ml_core/src/ml_core/features/_nwp.py:21-49`) builds the spine by joining power *onto*
  NWP rows (`nwp_with_init.join(power_lf, on=["time_series_id", "valid_time"], how="left")`), so
  widening the NWP `valid_time` filter would manufacture spurious extra spine/label rows before
  `train_start`/`val_start` — with no equivalent post-hoc filter to drop them, unlike
  `live_forecasts`.
- A `power_lookback` parameter dedicated to the power scan avoids all three problems: `_apply_power_lag`
  (`_lags.py:14-47`) is a left join keyed on `target_time`, so extra early power rows only fill in
  lag *values* for rows already in the (unchanged) spine — they add no new spine rows. Verified by
  reading `_join_nwp_bulk_mode`'s join direction and `_apply_power_lag`'s join keys.

- Sizing the margin from a single global constant (as `LIVE_POWER_HISTORY` does) is also rejected:
  the issue itself asks for something "sized to the longest lag/rolling feature any experiment
  config might use, not just the current champion's" — but the roadmap's planned
  `nged_incumbent` baseline uses 49–55-week lags (`docs/roadmap/metrics-and-leaderboard.md:200`),
  so a global constant sized for that would force every experiment's CV load to scan a year of
  power history it does not use. Deriving it per experiment from `config.selected_features` costs
  nothing extra for `conf/model/xgboost.yaml` (whose longest lag today is 336h) and scales
  correctly to whatever the next experiment declares.

"Rolling" in the issue title is not separately affected: `RollingFeature` on `power` is already
forbidden (`_parsed_features.py`'s `RollingFeature` docstring), and weather lag/rolling features
read from the NWP frame, not the power frame (`_apply_weather_lag`, `_apply_rolling_mean_feature`),
so they are bounded by NWP's own `init_time`/`MAX_NWP_LEAD` window, which this plan does not touch.
Only power *lag* features are affected by the bug, and only power lags are the target of this fix.

## What changes, file by file

**`src/nged_substation_forecast/defs/_engineering_inputs.py`**
- Add `power_lookback: timedelta = timedelta(0)` to `load_engineering_inputs`'s signature (default
  preserves today's behaviour for every existing caller unless it opts in).
- Change the power scan's lower-bound predicate from `pl.col("time") >= window_start` to
  `pl.col("time") >= window_start - power_lookback`. The NWP scan's `valid_time` and `init_time`
  predicates are untouched.
- Extend the docstring: document `power_lookback` in the Args section, and add one sentence to the
  "Memory" preamble or the `window_start` entry clarifying that `power_lookback` widens the power
  scan only, so the reader does not have to re-derive the NWP-safety argument above.

**`packages/ml_core/src/ml_core/features/_parsed_features.py`**
- Add `ParsedFeatures.max_power_lag_hours() -> int`, alongside the existing `get_leaky_features`:
  `max((lag.hours for lag in self.lags if lag.base_col == "power"), default=0)`.

**`src/nged_substation_forecast/defs/cv_assets.py`**
- Import `ParsedFeatures` from `ml_core.features._parsed_features` (matching
  `production_helpers.py`'s existing import).
- `trained_cv_model`: after `forecaster_cls, config = load_experiment_forecaster(experiment_name)`,
  compute `power_lookback = timedelta(hours=ParsedFeatures.from_strings(config.selected_features)
  .max_power_lag_hours())` and pass `power_lookback=power_lookback` into the existing
  `load_engineering_inputs` call.
- `cv_power_forecasts`: same derivation, passed into the `load_engineering_inputs` call inside the
  `while chunk_start <= val_end` loop. Add a one-line comment noting the (already-lazy) power scan
  is cheaply re-issued per `init_time` chunk — this is existing behaviour (the scan is already
  re-issued each iteration today), not a new cost `power_lookback` introduces, but worth naming so
  a future profiler does not misattribute it.

**`src/nged_substation_forecast/defs/production_assets.py`**
- No behaviour change. Add one sentence to `LIVE_POWER_HISTORY`'s docstring cross-referencing the
  new `power_lookback` parameter, so a future long-lag live model is not silently starved by a
  constant nobody remembered to revisit (this cross-reference is explicitly called for in the
  roadmap text this plan implements).

**`docs/roadmap/metrics-and-leaderboard.md`**
- PR 2's bullet list currently reads "CV predict-path framework: `uses_nwp_ensemble`, power-lag
  lookback, `ensemble_member` docs" and includes the "Power-lag lookback at feature-engineering
  load time" bullet plus its dedicated test sub-bullet. Remove that bullet and its test sub-bullet
  (this plan implements it) and retitle the PR to name only the two items still outstanding
  (`uses_nwp_ensemble`, `ensemble_member` docs). Leave the "After PRs 1 + 2 land back-to-back, run
  one `trained_cv_model++` backfill..." paragraph in place — it still describes a real future
  trigger for the remaining items, and is independent of whether this fix's own retrain happens
  now or later (see Risks, below).

## Plan review 1: simplicity (ran)

A fresh sub-agent, briefed only with the issue and this plan (not this reasoning), independently
verified the join-direction argument above by reading `_join_nwp_bulk_mode` and `_apply_power_lag`
itself, and searched for a simpler design. **No simplification survived.** It confirmed: the naive
"widen `window_start`" mechanism the issue suggests is a real correctness bug for the CV callers,
not a style choice; mimicking `live_forecasts`'s full pattern (explicit `init_time_start` +
post-hoc filter) would cost more code than the dedicated `power_lookback` parameter; per-experiment
derivation via `ParsedFeatures` is justified specifically because CV runs many concurrent
experiments (H2) where a single global constant sized for the worst case would multiply wasted
power-history reads across every one of them, unlike `live_forecasts`'s single promoted champion;
and pushing the fix down into `_apply_power_lag` would just duplicate the scan/loading logic that
already lives centrally in `load_engineering_inputs`.

One additional candidate the reviewer considered and rejected on its own: dropping the power scan's
lower bound entirely (always load full history, no new parameter at all). Rejected because
`load_engineering_inputs`'s docstring already states a "prune the scan at source" memory discipline,
and unbounded historical power reads multiplied across "a hundred experiments per person in a peak
month" (H2) is an avoidable I/O cost the targeted parameter avoids by design — simpler code, but
trading away a documented discipline for no clear need.

The one open call the plan itself flagged — `max_power_lag_hours()` as a `ParsedFeatures` method
vs. an inline expression at the two `cv_assets.py` call sites — the reviewer called a genuine,
low-stakes toss-up and did not settle it either way. Left as-is (method) per Risk 3 above; not
worth a second review pass on its own.

## Plan review 2: correctness and testability (ran)

A second fresh sub-agent, briefed only with the issue and the post-review-1 plan, independently
verified every current-behaviour claim (the loader's filter predicates and `init_time_start`
default, the CV callers' exact `window_start` values, `_apply_power_lag`'s join), confirmed all
three proposed tests would genuinely fail on `main` for the stated reason, traced the
row-count-unchanged claim algebraically through `_join_nwp_bulk_mode` (it holds unconditionally,
not just for the plan's example fixture, because the NWP `valid_time` floor stays pinned at
`window_start` regardless of `power_lookback`, so no power row pulled in by the lookback can ever
match an NWP row and manufacture a spine row), confirmed `_nullify_leaky_lags` is genuinely
orthogonal to where a lag's source data came from and runs unconditionally over the whole frame, and
re-verified the "naive widen would corrupt CV" claim directly by reading the NWP-centric join and
confirming the bulk-mode hindcast filter (`valid_time > power_fcst_init_time`, keyed off each row's
own derived `power_fcst_init_time`) would **not** catch the spurious early rows a naive widen would
introduce — it checks a per-row derived value, not `window_start`.

**One real defect found and fixed above**: test 3 (the end-to-end lag test) was originally placed
in `packages/ml_core/tests/test_features.py`, driving through the private `_engineer_features`. The
reviewer found `load_engineering_inputs` lives in the root Dagster app
(`src/nged_substation_forecast/`), that no file under `packages/` imports from the root app
anywhere in the repo, and that CLAUDE.md's architecture section states packages have no dependency
on it — so a package-level test reaching for the loader would be the repo's first reverse
dependency of that kind. Verified independently (`grep` for any `from nged_substation_forecast`
import anywhere under `packages/`: none). Fixed: moved to `tests/test_trained_cv_model.py`, driven
through the public `TabularFeatureEngineer().engineer()` entry point (see "Tests" above).

One minor point raised and correctly judged not to need a plan change: `eligible_time_series_ids`
anchors its coverage check to `val_start`, not `train_start`, so "eligible ⇒ the lookback data is
on disk" is typically true but not a structural guarantee for every marginally-eligible series.
Doesn't change the fix's correctness — where the data genuinely doesn't reach back far enough, the
widened join still correctly returns null (real missing data, not a bug).

No other defects found; every other claim in the plan — the current-behaviour account, the
join-direction safety argument, the nullification-is-unaffected argument, the caller/doc
inventory — was independently verified against the code.

## Design-philosophy check

Both touched assets (`trained_cv_model`, `cv_power_forecasts`) carry the `research` layer tag —
this change stays inside `docs/design-philosophy/inherent-stability.md`'s "R&D fails the other
way" posture (fail-fast, never silently degrade) and does not touch it: the fix makes the loaded
window *correct* rather than making a missing-data path degrade more gracefully. No asset check is
added or changed, so the `WARN`/`blocking=False` rule does not apply here. No Patito contract
changes. `live_forecasts` (production) is explicitly untouched, so principle 1 ("never stop") is
not engaged either way.

The one hypothesis this serves: [H2 — a hundred experiments per person in a peak
month](../design-philosophy/engineering-hypotheses.md#h2-a-hundred-experiments-per-person-in-a-peak-month)
and [principle 8 — every experiment is scored
identically](../design-philosophy/design-principles.md#8-every-experiment-is-scored-identically):
a leaderboard number that is silently biased by which fold-boundary rows happened to null out is
exactly the "score that cannot be set against the scores already on the board" failure that
principle exists to prevent.

## Tests

Test 1 goes in `packages/ml_core/tests/test_features.py` (where `ParsedFeatures` is already
tested). Tests 2 and 3 go in `tests/test_trained_cv_model.py`, which already holds
`load_engineering_inputs`-level tests — not `packages/ml_core/tests/`, since
`load_engineering_inputs` lives in the root Dagster app and packages have no dependency on it (see
test 3 below and Plan review 2).

1. **`ParsedFeatures.max_power_lag_hours()`** (`test_features.py`, near
   `test_parsed_features_from_selected_features`): `ParsedFeatures.from_strings({"power_lag_336h",
   "power_lag_24h", "temperature_2m_lag_48h"}).max_power_lag_hours() == 336` (mixes in a
   non-power lag to prove the filter, not just the max), and
   `ParsedFeatures.from_strings(set()).max_power_lag_hours() == 0`. **Fails on `main`** — the method
   does not exist (`AttributeError`).

2. **`load_engineering_inputs` widens only the power scan** (`tests/test_trained_cv_model.py`, next
   to `test_load_engineering_inputs_prunes_nwp_to_requested_cells_and_init_window`): build a power
   fixture with rows both inside `[window_start, window_end]` and a few days before `window_start`,
   and an NWP fixture with `valid_time` rows in the same before-window span. Call
   `load_engineering_inputs(..., window_start=..., window_end=..., power_lookback=timedelta(days=3))`
   and assert the collected `power_ts` includes the before-window rows while the collected `nwp_lf`
   still excludes `valid_time` rows before `window_start` — proving the widening is power-only.
   Also assert the default (`power_lookback` omitted) excludes the before-window power rows,
   matching today's behaviour. **Fails on `main`** — `power_lookback` is not a recognised keyword
   argument (`TypeError`).

3. **End-to-end: a lag near the window start is no longer null** (`tests/test_trained_cv_model.py`,
   not `packages/ml_core/tests/test_features.py` — `load_engineering_inputs` lives in the root
   Dagster app (`src/nged_substation_forecast/`), and no file under `packages/` imports from it
   anywhere in the repo today; `packages/ml_core` reusable code has no dependency on the root app
   (CLAUDE.md's Architecture section), so a package-level test reaching for it would be the first
   reverse dependency of that kind. Drive it through the public `TabularFeatureEngineer().engineer()`
   entry point, not the private `_engineer_features`, matching how every other root-level test
   exercises feature engineering): with observed power extending well before `train_start` and
   `selected_features={"power_lag_336h"}`, assert the `power_lag_336h` value for a row a few hours
   after `train_start` is a real (non-null) number equal to the corresponding historical observation
   when `load_engineering_inputs` is called with the derived `power_lookback`, and is `None` when
   called with `power_lookback=timedelta(0)` (today's behaviour) — same fixture, two assertions, so
   the test demonstrates the fix rather than merely the new code path. Also assert the row count of
   the engineered frame is identical between the two calls, confirming the widened power window
   adds no spine rows (the NWP-centric-join argument above, made concrete). **Fails on `main`** for
   the same reason as (2) — `power_lookback` does not exist yet, so the "non-null with lookback"
   branch cannot be expressed.

No new leakage/nullification test is needed: `power_lookback` only ever adds power rows *before*
`window_start`, which is always in the past relative to any `valid_time` in
`[window_start, window_end]`, and `_nullify_leaky_lags` (unmodified by this change) already covers
the boundary cases (`test_nullify_leaky_lags`,
`test_engineer_features_power_lag_nullification_end_to_end`).

## Docs to update

- `docs/roadmap/metrics-and-leaderboard.md` — see "What changes" above.
- `_engineering_inputs.py` and `production_assets.py` docstrings — see "What changes" above.
- No page in `docs/ml_experimentation/` or `docs/architecture/` describes the current (buggy)
  window-truncation behaviour explicitly (checked `dagster-workflow.md`, `ml-orchestration.md`,
  `model-configuration.md`, `cross-validation-folds.md`), so nothing else needs a "now fixed" edit.

## Verification commands

```bash
uv run ruff check .
uv run ruff format . --check
uv run ty check
uv run pytest packages/ml_core/tests/test_features.py tests/test_trained_cv_model.py tests/test_cv_power_forecasts.py
uv run pytest   # full suite before push
uv run pymarkdown scan -r docs README.md CLAUDE.md packages/*/README.md
```

## Risks and open questions

1. **Should this PR also trigger a retrain of every existing CV fold and experiment?** This is the
   question issue #638 says is the real point. Recommendation: **no, not as part of this PR.**
   Reasoning: (a) landing the loader fix costs nothing extra now, and every future retrain —
   whenever next triggered, including the `uses_nwp_ensemble`/PR-1 backfill the roadmap already
   plans — picks up the corrected loader for free; (b) *not* retraining immediately does not make
   anything worse than `main` today — CV metrics stay exactly as biased as they already are until
   the next retrain, whenever that happens; (c) per this repo's current stage ("we haven't trained
   any 'serious' ML models yet" — `CLAUDE.md`), the cost of a stale leaderboard right now is low;
   (d) a retrain is explicitly flagged in this session's instructions as something to *not* trigger
   without asking, partly because other sessions may be running concurrently and a full backtest is
   memory-heavy. If the human reviewer wants the retrain to happen as part of shipping this fix
   rather than deferred, say so and it becomes a follow-up step (`trained_cv_model++` backfill,
   per the mechanism `docs/roadmap/metrics-and-leaderboard.md` already describes), run only after
   this PR merges and by explicit request — never automatically.
2. **Should the `power_lookback` derivation also cover a future power-*rolling* feature, given
   `RollingFeature` on `power` is only "currently forbidden" (docstring), not structurally
   impossible?** Recommendation: no — out of scope for #638, and premature: nothing today can
   request a power rolling feature (`ParsedFeatures.from_strings` never produces one), so there is
   no caller to size for. If that restriction is later lifted, `max_power_lag_hours()` is the
   obvious place to extend (rename/broaden it) — flagging here rather than speculatively building
   it now.
3. **Does the per-experiment `power_lookback` derivation belong on `ParsedFeatures` itself, or as a
   free function in `cv_assets.py`?** Went with a `ParsedFeatures` method because the equivalent
   concept (`get_leaky_features`) already lives there, and both `trained_cv_model` and
   `cv_power_forecasts` need it — a named, tested method avoids duplicating the same filter+max
   logic twice. Flagging for the simplicity reviewer in case a plainer free function is preferred.
