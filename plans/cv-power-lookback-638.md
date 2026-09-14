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

## Status — resume from here

**As of 2026-08-20: planning is complete, both adversarial plan reviews have run and their
findings are incorporated below. No implementation code has been written. No PR against `main`
exists yet — only this checkpoint PR, opened as a draft purely to make this plan visible without
needing the Claude Code conversation that produced it.** The session that wrote this plan was
interrupted by a workstation shutdown before the human review this skill stops for could happen.

**To resume in a fresh Claude Code session:**

1. Check out the branch (worktree already exists on this machine at
   `.claude/worktrees/cv-power-lookback-638`; on a different machine, `git fetch origin
   cv-power-lookback-638 && git worktree add .claude/worktrees/cv-power-lookback-638
   cv-power-lookback-638`).
2. Re-read this whole file — it is self-contained and supersedes any summary in the checkpoint PR
   body or in an old conversation transcript.
3. **Get the human's answer to "Risks and open questions" item 1 below (the retrain decision)
   before doing anything else** — that is what this plan stopped on.
4. Once the plan itself is approved (with or without changes from item 1's answer), hand off to
   the `implement-issue` skill, **resuming at its step 2** (step 1 — worktree and branch — is
   already done): implement per "What changes, file by file" below, run the green-before-push
   verification set, open the real implementation PR (closing #638, with labels and the
   `JackKelly` assignee), then run whichever of the two diff reviews this **Medium**-sized issue
   calls for (`implement-issue` step 3's rules — both plan reviews already ran here, so err toward
   running at least one diff review too unless the diff comes out exactly as this plan describes).
5. Close or convert this checkpoint PR once the real implementation PR exists, so there is only
   one open PR for #638 at a time.

Nothing about the design below is provisional or needs re-deriving — both reviews verified the
correctness-critical claims (the NWP-centric join, why widening `window_start` naively would be a
bug, the nullification argument) directly against the code, not just against each other's
reasoning. A fresh session can trust this plan and start implementing once a human has signed off
on it and on the retrain question.

## Verdict, size and departures

**Verdict: worth fixing, roughly as described, but not by the mechanism the issue suggests.** The
issue's premise is factually correct — verified against `cv_assets.py:347-354` (`trained_cv_model`)
and `:500-508` (`cv_power_forecasts`): neither passes any look-back margin, unlike
`live_forecasts` (`production_assets.py:308-316`).

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
  (`production_assets.py:333-338`, "History rows... are join artefacts, not genuine forecasts").
- `trained_cv_model` and `cv_power_forecasts` do neither. `trained_cv_model` does not override
  `init_time_start`, so widening `window_start` would also pull in up to two extra weeks of NWP
  partitions never needed. Worse: in bulk mode, `_join_nwp_bulk_mode`
  (`packages/ml_core/src/ml_core/features/_nwp.py:21-49`) builds the spine by joining power *onto*
  NWP rows (`nwp_with_init.join(power_lf, on=["time_series_id", "valid_time"], how="left")`), so
  widening the NWP `valid_time` filter would manufacture spurious extra spine/label rows before
  `train_start`/`val_start` — with no equivalent post-hoc filter to drop them, unlike
  `live_forecasts`.
- A `power_lookback` parameter dedicated to the power scan avoids all three problems: `_apply_power_lag`
  (`_lags.py:14-53`) is a left join keyed on `target_time`, so extra early power rows only fill in
  lag *values* for rows already in the (unchanged) spine — they add no new spine rows. Verified by
  reading `_join_nwp_bulk_mode`'s join direction and `_apply_power_lag`'s join keys.

- Sizing the margin from a single global constant (as `LIVE_POWER_HISTORY` does) is also rejected:
  the issue itself asks for something "sized to the longest lag/rolling feature any experiment
  config might use, not just the current champion's" — but the roadmap's planned
  `manual_heuristic` baseline uses 49–55-week lags
  (`docs/roadmap/metrics-and-leaderboard.md:364`), which PR 2 already commits to supporting, so a
  global constant sized for the current champion alone would leave that baseline's annual lags
  all-null and would have to be widened again the moment it lands; sizing it for the worst case
  instead would force every smaller experiment's CV load to scan a year of power history it does
  not use. (The NWP Delta table dominates CV's I/O at ~86 GB —
  `docs/architecture/performance.md:66` — so the power scan's own cost is secondary; the real
  reason for per-experiment sizing is that no single constant serves both a 336 h and a 55-week
  experiment well.) Deriving it per experiment from `config.selected_features` costs nothing extra
  for `conf/model/xgboost.yaml` (whose longest lag today is 336h) and scales correctly to whatever
  the next experiment declares.

"Rolling" in the issue title is not separately affected: `RollingFeature` on `power` is
structurally impossible, not just documented as forbidden — `RollingFeature.base_col` is typed
`WeatherFeature` (`_parsed_features.py:44`), without the `| Literal["power"]` widening `LagFeature`
adds at line 77, so pydantic rejects a `"power"` rolling feature at parse time. Weather lag/rolling
features
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
  scan only, so the reader does not have to re-derive the NWP-safety argument above. Update the
  summary sentence ("Both are filtered to the inclusive `[window_start, window_end]` window...",
  currently `:42-43`) to say power is filtered to `[window_start - power_lookback, window_end]`
  while NWP stays bounded to `[window_start, window_end]`. State that a caller needing power history
  before `window_start` must either widen `window_start` itself (as `live_forecasts` does) or pass
  `power_lookback` — the default does neither. Note that `power_lookback == max_power_lag_hours()`
  is exactly sufficient with no margin needed (the scan predicate is inclusive and the earliest lag
  target is exactly `window_start - max_lag`), unlike `LIVE_POWER_HISTORY`'s deliberate margin over
  its longest lag — so a reader does not "fix" an apparent off-by-one later.

**`packages/ml_core/src/ml_core/features/_parsed_features.py`**
- Add `ParsedFeatures.max_power_lag()  -> timedelta`, alongside the existing `get_leaky_features`:
  `timedelta(hours=max((lag.hours for lag in self.lags if lag.base_col == "power"), default=0))`.
  Returning `timedelta` (not `int` hours) puts the unit conversion in one place instead of repeating
  `timedelta(hours=...)` at both `cv_assets.py` call sites.

**`src/nged_substation_forecast/defs/cv_assets.py`**
- Import `ParsedFeatures` from `ml_core.features._parsed_features` (matching
  `production_helpers.py`'s existing import).
- `trained_cv_model`: after `forecaster_cls, config = load_experiment_forecaster(experiment_name)`,
  compute `power_lookback = ParsedFeatures.from_strings(config.selected_features).max_power_lag()`
  and pass `power_lookback=power_lookback` into the existing `load_engineering_inputs` call.
- `cv_power_forecasts`: same derivation, passed into the `load_engineering_inputs` call inside the
  `while chunk_start <= val_end` loop. Add a one-line comment stating the power scan is lazy and
  re-issued per `init_time` chunk, and is small next to the NWP scan — present-tense, describing
  the code as it now stands rather than narrating what `power_lookback` did or didn't add (CLAUDE.md,
  "Write about the present, not the past").

**`src/nged_substation_forecast/defs/production_assets.py`**
- No behaviour change. Add one sentence to `LIVE_POWER_HISTORY`'s docstring cross-referencing the
  new `power_lookback` parameter, so a future long-lag live model is not silently starved by a
  constant nobody remembered to revisit (this cross-reference is explicitly called for in the
  roadmap text this plan implements).

**`docs/roadmap/metrics-and-leaderboard.md`**
- PR 2's bullet list currently reads "CV predict-path framework: `uses_nwp_ensemble`, power-lag
  lookback, `ensemble_member` docs". Remove the "Power-lag lookback at feature-engineering load
  time" bullet (`:361-377`, this plan implements it) and retitle the PR to name only the two items
  still outstanding (`uses_nwp_ensemble`, `ensemble_member` docs). The `Tests:` bullet (`:381-384`)
  is one bullet covering all three PR-2 items — there is no separate lookback test sub-bullet to
  remove; instead, edit that bullet to drop only its lookback clause ("a feature-engineer test that
  a 336 h lag at the window edge is non-null *with* lookback while the spine row count is
  unchanged"), keeping the `uses_nwp_ensemble` and leak-test clauses. `:435`, under the
  `manual_heuristic` baseline, reads "Depends on PR 2's lookback (the 55-week annual lags are
  all-null without it)" — repoint that to say the lookback already landed (issue #638), not that it
  depends on PR 2. Leave the "After PRs 1 + 2 land back-to-back, run one `trained_cv_model++`
  backfill..." paragraph in place — it still describes a real future trigger for the remaining
  items, and is independent of whether this fix's own retrain happens now or later (see Risks,
  below).

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

## Plan review 3: Opus adversarial review (ran)

A third fresh sub-agent (Opus 5), run after both plan reviews above and after the `origin/main`
merge, independently re-derived the core correctness argument (the NWP-centric join, why naively
widening `window_start` is a bug for both CV callers, not one) directly against the code, and
re-ran the simplicity and correctness attacks from scratch. **Verdict: the design is right; no
simplification survived.** Six findings were accepted and are folded into the sections above:

- **One real defect**: `docs/ml_experimentation/cross-validation-folds.md:148-152` states power is
  bounded to `[window_start, window_end]`, which this fix makes false — added to "Docs to update".
- The roadmap edit instructions were imprecise: no dedicated test sub-bullet exists to remove, and
  the `manual_heuristic` baseline's "Depends on PR 2's lookback" note needed repointing — both fixed
  in "What changes, file by file".
- The `_engineering_inputs.py` docstring edits missed the summary sentence that goes stale, and
  should state the "no margin needed" property and that a caller must opt in explicitly — added.
- Test 3 has a fixture trap: the target row's NWP `init_time` must be early enough that its lead
  time stays under the lag length even with the lookback applied, or the assertion fails for an
  unrelated reason — named explicitly against the existing `_EARLY_INIT_TIME` fixture.
- The prescribed `cv_power_forecasts` comment narrated what the diff *didn't* change, which CLAUDE.md's
  "write about the present" rule forbids for comments — reworded.
- `max_power_lag_hours() -> int` was changed to `max_power_lag() -> timedelta`, so the unit
  conversion lives in one place instead of being repeated at both `cv_assets.py` call sites.

Findings verified and **not** acted on: the join-direction/no-lookahead argument (re-confirmed,
already correct); an alternative of making `load_engineering_inputs` feature-aware by passing
`selected_features` directly (rejected — makes a purely geometric loader feature-aware for a
smaller call-site saving, and removes callers' ability to request a lookback for other reasons);
folding `LIVE_POWER_HISTORY` into `power_lookback` as a future follow-up (noted as not cheap —
`LIVE_POWER_HISTORY` also sizes `build_live_power_frame`'s dense spine, so unifying would mean
reworking that too — recorded so it isn't proposed again at diff-review time); the "sized to the
worst case" justification was reframed (see the `power_lookback` sizing paragraph above) rather than
dropped, since the reviewer's point was about which argument is strongest, not whether per-experiment
sizing is right; the issue's "power lag **and rolling**" premise being wrong for rolling — already
correctly overruled by this plan and independently re-confirmed, now with the structural (pydantic)
reason cited rather than only the docstring.

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

1. **`ParsedFeatures.max_power_lag()`** (`test_features.py`, near
   `test_parsed_features_from_selected_features`): `ParsedFeatures.from_strings({"power_lag_336h",
   "power_lag_24h", "temperature_2m_lag_48h"}).max_power_lag() == timedelta(hours=336)` (mixes in a
   non-power lag to prove the filter, not just the max), and
   `ParsedFeatures.from_strings(set()).max_power_lag() == timedelta(0)`. **Fails on `main`** — the
   method does not exist (`AttributeError`).

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

   **Fixture trap:** `_nullify_leaky_lags` nulls a lag whenever the row's lead time is `>=` the lag
   length, regardless of where the lag value came from, so the target row's NWP `init_time` must be
   early enough that its lead time at `power_lag_336h` stays below 336 h even with the lookback
   applied — otherwise the assertion fails for a reason unrelated to this fix. Reuse the existing
   `_EARLY_INIT_TIME` fixture (`train_start - timedelta(days=10)`): a row at `train_start + 3h` has
   a 234 h lead there, safely under 336 h.

No new leakage/nullification test is needed: `power_lookback` only ever adds power rows *before*
`window_start`, which is always in the past relative to any `valid_time` in
`[window_start, window_end]`, and `_nullify_leaky_lags` (unmodified by this change) already covers
the boundary cases (`test_nullify_leaky_lags`,
`test_engineer_features_power_lag_nullification_end_to_end`).

## Docs to update

- `docs/roadmap/metrics-and-leaderboard.md` — see "What changes" above.
- `_engineering_inputs.py` and `production_assets.py` docstrings — see "What changes" above.
- `docs/ml_experimentation/cross-validation-folds.md:148-152` ("Overlapping forecasts do not
  contaminate the test set") states that `load_engineering_inputs` bounds *both* power and NWP to
  `[window_start, window_end]` — that becomes false for power once this fix lands. Restate the
  argument: the fold boundary is enforced by the NWP `valid_time` filter, which bounds the spine
  and therefore the labels; power lag lookups deliberately reach backwards past `window_start`
  (exactly as the live service already does), and remain leak-free because `_nullify_leaky_lags`
  nulls any lag shorter than the lead time regardless of where the value came from. The conclusion
  ("no observation appears on both sides of a fold boundary") stays true; only the mechanism
  changes.
- `dagster-workflow.md`, `ml-orchestration.md` and `model-configuration.md` do not describe the
  window-truncation behaviour and need no edit (checked).

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
   question issue #638 says is the real point. **Sizing the current bias**, for the leaderboard's
   single fold (`conf/cv/default.yaml`: train 2024-04-01→2025-06-30, val 2025-07-01→2026-06-30) and
   the champion's longest lag (336 h, `conf/model/xgboost.yaml`): the affected span is the first 14
   days of each window, roughly 3% of training rows and 3.8% of scored validation rows, with the
   feature null rather than wrong (XGBoost handles nulls natively) — a real but small bias.
   Recommendation: **no, not as part of this PR.**
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
2. **Should the `power_lookback` derivation also cover a future power-*rolling* feature?**
   Recommendation: no — out of scope for #638, and premature: `RollingFeature` on `power` is
   structurally impossible today (see above), so there is no caller to size for. If that pydantic
   restriction is ever lifted, `max_power_lag()` is the obvious place to extend (rename/broaden it)
   — flagging here rather than speculatively building it now.
3. **Does the per-experiment `power_lookback` derivation belong on `ParsedFeatures` itself, or as a
   free function in `cv_assets.py`?** Went with a `ParsedFeatures` method because the equivalent
   concept (`get_leaky_features`) already lives there, and both `trained_cv_model` and
   `cv_power_forecasts` need it — a named, tested method avoids duplicating the same filter+max
   logic twice. Flagging for the simplicity reviewer in case a plainer free function is preferred.
