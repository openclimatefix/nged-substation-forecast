# Issue #652 — should `"live"` availability mode apply `NWP_PUBLICATION_DELAY_HOURS`?

**The problem.** `live_forecasts` picks its one NWP run in `"live"` availability mode, which
applies no modelled publication delay — `select_nwp_init_time` accepts any run genuinely present in
the Delta table. The single-run analysis-proxy join inside `_engineer_features` then gates that same
run with a 9-hour `available_at` cut, so when the selected run is closer than 9 hours to
`power_fcst_init_time` the cut throws the run away and every weather-lag feature for the slot comes
back null. The two pieces of the serving path disagree about what "available" means.

**The planned solution.** In single-run mode, filter the NWP frame to runs at or before the run the
caller already selected, then call `select_analysis_proxy` with no availability parameters at all —
and delete `available_at` and `publication_delay` from `select_analysis_proxy`, which after this
change has no other caller for them. The selected run is by construction the freshest run that was
available, so a ceiling at that run reproduces the availability set exactly in both `"live"` and
`"replay"` mode, with no `availability_mode` parameter threaded into feature engineering. Replay and
the derived-`nwp_init_time` backfill path come out unchanged; only the live case changes, and it
changes to match what bulk-mode training already does.

## Verdict, size and departures

**Verdict: worth doing, and the answer to the issue's question is that `"live"` mode should not
apply the delay.** Three reasons, in order of weight:

**The project has already decided this, and `_engineer_features` never got the memo.** The
asymmetry is documented as deliberate in three places: the `AvailabilityModeType` docstring in
`production_helpers.py`, the `NWP_PUBLICATION_DELAY_HOURS` docstring in `analysis_proxy.py` ("Of
`select_nwp_init_time`'s two modes, only `"replay"` needs the delay"), and
[Resolve NWP availability asymmetrically](https://openclimatefix.github.io/nged-substation-forecast/architecture/production-deployment/#resolve-nwp-availability-asymmetrically-live-vs-replay)
in `docs/architecture/production-deployment.md`. So #652 is not an open semantic question. It is one
code path contradicting a decision the rest of the serving path already records. The reasoning
behind that decision applies unchanged here: a modelled delay reconstructs availability when
availability cannot be observed, and in live mode it can be observed, because the Delta table
contains only runs that genuinely landed.

**Today's behaviour is training/serving skew.** Bulk mode applies no `available_at` cut at all, and
the comment in the bulk branch argues the effective ceiling is the row's own `nwp_init_time` — an
NWP run's earliest `valid_time` is its own `init_time`, and at one run a day no run falls between a
row's own run and its derived `power_fcst_init_time`. So in training, a weather lag sees runs up to
and including the row's own run. Under this plan live mode sees exactly the same set. Under today's
behaviour it sometimes sees a strictly smaller set and the feature goes null. A feature that trains
on a populated column and serves nulls at one slot in four is the failure the issue is worried
about.

**Applying the delay in live mode is not conservative, it is unpredictable in effect.** The cut does
not widen uncertainty or record a degradation — it silently empties a feature, which reads
downstream as "no weather" rather than "degraded weather". That is the fail-open-becoming-fail-closed
shape `docs/design-philosophy/inherent-stability.md` warns about.

### Reachability

**The bug needs no code change, no schedule change, and no manual backfill to reach — an operator
re-running a missed slot is enough.** `availability_mode` defaults to `"live"`
(`production_assets.py:204`), and live mode's cutoff is `power_fcst_init_time` itself. So
re-materialising a failed or missed 06:00 partition any time after 08:30 UTC the same day finds
today's 00Z run in the table, accepts it (`00:00 <= 06:00`), and lands a 6-hour gap — under the
9-hour cut. The operator opted into nothing. Three further routes exist, all needing more: a manual
`ecmwf_ens` backfill landing a run earlier in the day, `ecmwf_ens_schedule` moving earlier, or
ingesting more than one ECMWF run a day.

**Blast radius today is nevertheless zero, and that bounds how much apparatus the fix deserves.**
`conf/model/xgboost.yaml` selects `temperature_2m`, `dew_point_temperature_2m`, and
`wind_speed_10m` as same-run weather features and no `*_lag_*h` weather lag, so no current champion
feature takes this path. On the normal schedule, too, the freshest run present at each slot is 12,
18, 24, or 30 hours old, all past the cut. Fix it, but keep the test and docs apparatus proportionate
— which is what the review below cut it down to.

**Size: complex.** The change is on the production serving path, which `plan-issue` step 3 makes
complex on its own. It gets this plan, both plan reviews, and both diff reviews in
`implement-issue`.

**Departures from the issue body.** The issue frames the choice as "apply the delay in live mode
too, or declare the asymmetry intentional", and expects the fix to plumb the availability mode into
feature engineering. This plan takes neither branch literally. It resolves the question the second
way — the asymmetry is intentional — but concludes that the single-run cut should therefore stop
modelling availability at all, rather than learning which mode it is in. Cutting at the already
selected `nwp_init_time` is mode-independent and equivalent to the replay cut, so
`ml_core.features` needs no knowledge of `AvailabilityModeType` and no new parameter.

## What changes, file by file

### `packages/ml_core/src/ml_core/features/tabular_feature_engineer.py`

**This is the whole behaviour change.** In `_engineer_features`' single-run branch, filter
`processed_nwp` to `pl.col("nwp_init_time") <= _resolve_nwp_init_time(nwp_init_time,
power_fcst_init_time, nwp_publication_delay_hours)` before handing it to `select_analysis_proxy`,
and drop the `available_at`/`publication_delay` arguments from the call. Both branches then call
`select_analysis_proxy` with the same three arguments, so the two call sites may collapse into one
with the filter applied conditionally — the implementer's choice, under one constraint: **both
explanatory comments must survive**. The bulk branch's comment carries the run-cadence invariant
that makes bulk mode leak-free, and losing it to a tidier `if` would be a bad trade.

`_resolve_nwp_init_time` already exists in `ml_core.features._nwp` and is already imported into this
module by `_check_or_warn_on_missing_control_member`, so no new helper is needed and the three
places that decide "which run is the selected run" stay on one definition.

Rewrite the single-run comment. It currently justifies the cut by a wide-multi-run-frame hazard no
caller in the repo produces. The new comment should say that the selected run is the ceiling because
it is by construction the freshest run that was available, that this reproduces the replay cut and
matches bulk mode's effective ceiling, and that a run later than the selected one is excluded
whether the frame holds one run or many.

Rewrite the `nwp_publication_delay_hours` argument docstring, whose last clause says the delay
"gates the single-run analysis-proxy's `available_at` cut". After the change the delay reaches the
analysis proxy only indirectly, through the derived `nwp_init_time` fallback when the caller omits
`nwp_init_time`.

`timedelta` is imported at line 32 and used only at line 278, so the import goes with the change —
check this at implementation time rather than trusting the plan, since an unrelated edit may have
added a use.

### `packages/weather_utils/src/weather_utils/analysis_proxy.py`

**Delete `available_at` and `publication_delay` from `select_analysis_proxy`**, along with the
`if available_at is not None:` filter and the two argument docstring entries. After the change above
there is no non-test caller of either; the dashboard passes only `max_lead`
(`view_forecasts.py:411-421`).

Filtering at the caller instead is the pattern the repo already uses: the dashboard filters
`init_time` on the Delta scan *before* calling `select_analysis_proxy`, for the pushdown reason the
function's own docstring asks callers to check. The single-run filter runs one node earlier than
`available_at` did — before the member filter rather than after — which is no worse for pushdown.

Rewrite the `NWP_PUBLICATION_DELAY_HOURS` docstring's third paragraph, which lists
`select_analysis_proxy`'s `available_at` cut as one of the constant's three consumers. After this
change the consumers are bulk mode's per-row `power_fcst_init_time` derivation, single-run mode's
`nwp_init_time` fallback, and `select_nwp_init_time`'s replay cutoff.

### `src/nged_substation_forecast/defs/production_assets.py`

Rewrite the `Note:` paragraph in `live_forecasts`' docstring. The paragraph currently describes the
too-fresh null as real behaviour and names the 06:00 slot. After the change the only remaining cause
of an all-null weather lag in a live slot is a run whose control member (`ensemble_member == 0`) is
wholly absent — a partial or malformed ECMWF ENS download — which still degrades with a logged
warning rather than failing the slot. Keep the closing sentence's point that no current champion
feature is a weather lag, and keep the cross-reference to the test under its new name.

This docstring renders in the Dagster UI as operator documentation, so it stays full-bodied rather
than being trimmed to a line.

## Design-philosophy check

**This code path runs in production, and the change moves it further towards degrading rather than
failing.** `_engineer_features`' single-run branch is reached by `live_forecasts` on every scheduled
slot. Nothing in the change adds a raise: the filter can leave an empty frame exactly as the old cut
could, and the downstream weather-lag join treats an empty proxy as a join miss and produces nulls,
which is the existing degradation path. The `ValueError` in `_engineer_features` on `nwp_init_time`
without `power_fcst_init_time` is unchanged, and remains a caller bug rather than an outside-world
condition.

**No asset check is added or edited**, so the `WARN`/`blocking=False` rule is untouched. The
`_check_or_warn_on_missing_control_member` warning path is read but not changed, and it still cannot
raise in single-run mode.

**Inherent-stability rule 1 — liberal about missing inputs, strict about malformed ones — is what
this change serves.** A run that is genuinely present and genuinely usable is not a missing input,
and today's cut treats it as one. The malformed case, a run with no control member, keeps its
existing treatment.

**No principle in `design-principles.md` is traded away.** The change removes two parameters and a
concept rather than adding any.

**Hypotheses.** This change is a correctness fix on the serving path rather than a delivery of a
numbered hypothesis, so it cites none. Confirm against
`docs/design-philosophy/engineering-hypotheses.md` at implementation time and cite a label in the PR
body only if one genuinely fits, rather than stretching one to fit.

## Tests

Net effect: one Dagster materialisation removed, three `weather_utils` tests removed, one existing
test's contrivance removed, one small test added.

### `tests/test_live_forecasts.py` — rewrite the pinning test, two cases not three

`test_live_weather_lag_nulls_only_when_the_selected_run_is_too_fresh` pins the behaviour this change
removes, so it is rewritten rather than extended. Rename it to
`test_live_weather_lag_survives_a_run_fresher_than_the_publication_delay`, and reduce the
parametrised cases from three to two:

| Case | `gap_hours` | `members` | `expect_null` today | `expect_null` after |
|---|---|---|---|---|
| Below the delay | 6 | `(0,)` | `True` | **`False`** |
| No control member | 9 | `(1,)` | `True` | `True` |

**The assertion that fails on `main` today is the first row**: with a 6-hour gap between the
selected run and `power_fcst_init_time`, `temperature_2m_lag_6h` is entirely null on `main` and
populated after the change. The dropped case is the old `gap_hours=9, members=(0,)` row: it existed
to sit on the 9-hour boundary, and after the change the delay plays no part in this path, so nothing
distinguishes a 9-hour gap from a 6-hour one. Each case costs a Dagster `materialize` plus an
XGBoost train, so dropping it is worth real wall-clock time. Rewrite the test's docstring to describe
the new rule and to keep the second case's separate explanation (a partial download degrades the
same way, and `materialize` succeeding at all is part of what the case checks).

### `packages/ml_core/tests/test_features.py` — remove a contrivance from the leakage test

`test_engineer_features_weather_lag_leakage_prevention` sets `power_fcst_init_time = nwp_init_time +
NWP_PUBLICATION_DELAY_HOURS` under a two-line comment saying the value "must clear
`select_analysis_proxy`'s default `publication_delay`". That line exists only because of the wart
being removed. **Change it to `nwp_init_time + timedelta(hours=2)` and delete the comment.**

Traced: `nwp_init_time` is `2026-06-10 00:00`, so `power_fcst_init_time` becomes `02:00`. On `main`
the cut is `init + 9h <= 02:00`, excluding both runs, so the proxy is empty and the
`temperature_2m_lag_36h == 8.0` assertion goes null — **the test fails on `main` today**. After the
change the ceiling is `nwp_init_time`, run 1 is kept, run 2 (`06-10 12:00`, later than the selected
run) is still excluded, and both assertions pass. `temperature_2m_lag_2h == 10.0` is unaffected: its
target time is after `power_fcst_init_time`, so it takes the same-run branch.

### `packages/ml_core/tests/test_features.py` — the lookahead guard, unchanged assertion

`test_engineer_features_single_run_freshest_run_excludes_unpublished_nwp_run` builds three runs: T0
at 2026-06-10 00:00, T1 (the selected `nwp_init_time`) at 2026-06-11 00:00, and T2 at 2026-06-11
03:00 carrying a decoy value. Under the run ceiling, T2 is excluded because it is later than the
selected run and T0 answers, so the test's `== 8.0` assertion holds unchanged. Update the docstring,
which explains the exclusion in terms of `available_at`/`publication_delay`, to explain it in terms
of the run ceiling. **This is the test that proves the change does not reopen lookahead bias**, and
it must keep passing without loosening.

### `packages/ml_core/tests/test_features.py` — one new test

Add a single-run test with an explicit `nwp_init_time` and a `nwp_publication_delay_hours` large
enough that the selected run would fail the old cut, asserting the weather lag is populated. **On
`main` this asserts a populated value where the old cut returns null, so it fails today.**

The first plan review argued this is redundant once the leakage test above carries a failing-on-`main`
assertion, and that is nearly right — the two overlap. It is kept anyway, for one reason: the whole
issue is that this rule was invisible, and a rule pinned only as a side effect of a test named for
something else is invisible again. A test whose name states the rule is what stops the next refactor
re-introducing a modelled delay here. Keep it small; it costs no Dagster run.

### `packages/ml_core/tests/test_cross_mode_equivalence.py` — delete a now-dead argument

`test_bulk_and_single_run_features_are_identical` passes `nwp_publication_delay_hours=_DELAY_HOURS`
twice: once to the bulk call, and once to the single-run call inside the replay loop under a
seven-line comment about `select_analysis_proxy`'s `available_at` cut. **Delete the single-run
argument and its comment outright**, keeping the bulk call's, which bulk mode's
`power_fcst_init_time` derivation genuinely needs. In single-run mode after this change the delay is
consumed only by `_resolve_nwp_init_time`'s `None` fallback and by the degradation warning's run
naming, and the test passes `nwp_init_time=run` explicitly, so neither fires. The argument becoming
dead is a cleaner demonstration of the change than trimming its comment would be. The test's
assertions are unchanged, and it passing both before and after is the point — cross-mode equivalence
must survive.

### `packages/weather_utils/tests/test_analysis_proxy.py` — delete three tests

Three tests exercise `available_at` (around lines 112, 133, and 193). The parameter is gone, so the
tests go with it. Nothing is left unpinned: the leakage guard they approximate is exercised through
the real caller by
`test_engineer_features_single_run_freshest_run_excludes_unpublished_nwp_run`.

## Docs to update

- **`docs/architecture/production-deployment.md`**, the "Resolve NWP availability asymmetrically:
  `live` vs `replay`" section. The section describes the asymmetry at `select_nwp_init_time` and is
  silent on the analysis-proxy cut, which is how the two drifted apart. **Append one sentence to the
  existing `"live"` bullet** saying the weather-lag analysis proxy cuts at the selected run in both
  modes — not a new paragraph, since the rule is symmetric and the heading's claim is asymmetry.
  Present tense, describing how the code works now.
- **`docs/roadmap/switching-events.md`** (around lines 290–299) says the natural home for the
  residual pipeline's availability cut is `select_analysis_proxy`, "which ... already applies an
  `available_at` cut". That clause goes stale. Rewrite the sentence to point at the caller-side
  ceiling the single-run branch now applies. The surrounding claim — that the bulk freshest-run join
  is leak-free only as a side effect of daily run cadence — stays true and stays.
- **No roadmap ship-time triage.** #652 is a spike under the v0.2.1 follow-ups epic (#642) and
  completes no roadmap item, so there is no "Implementation details" section to delete and no status
  banner to move.
- No heading is renamed, so no anchor slug changes. Confirm that at implementation time —
  `analysis_proxy.py`'s `NWP_PUBLICATION_DELAY_HOURS` docstring links to the production-deployment
  anchor.

## Verification commands

```bash
uv run ruff check .
uv run ruff format .
uv run ty check
uv run pytest
uv run pymarkdown scan -r docs README.md CLAUDE.md packages/*/README.md
uv run mkdocs build --strict
```

`mkdocs build --strict` is in the set because the change edits two docs pages, one of which carries
an anchor other prose links to. The NWP-convention network tests (`uv run pytest --run-network -m
network`) are **not** needed: nothing here touches NWP download, unit conversion, or the sign
convention.

Run the four directly affected test modules first for a fast loop, then the whole suite before
pushing:

```bash
uv run pytest tests/test_live_forecasts.py packages/ml_core/tests/test_features.py \
  packages/ml_core/tests/test_cross_mode_equivalence.py \
  packages/weather_utils/tests/test_analysis_proxy.py
```

## Risks and open questions

**Does a ceiling at `nwp_init_time` really reproduce the replay cut?** Worth a reviewer's own check
rather than taking the plan's word. The argument: `select_nwp_init_time` in replay mode returns
`max{init in table : init + delay <= power_fcst_init_time}`. For any run in the table, `init <=
nwp_init_time` implies `init + delay <= nwp_init_time + delay <= power_fcst_init_time`; and `init +
delay <= power_fcst_init_time` puts that run in the qualifying set, so `init <= max =
nwp_init_time`. The two conditions select the same runs. When the caller omits `nwp_init_time`
entirely, `_resolve_nwp_init_time` derives `power_fcst_init_time - delay`, so the ceiling is
literally the old cut. **If this equivalence is wrong, the plan is wrong**, so it is the first thing
to attack.

**The equivalence holds for every caller that exists, not for every caller imaginable.** A caller
passing an explicit `nwp_init_time` *older* than the freshest published run, alongside a multi-run
frame, gets a strictly smaller proxy set than today's cut gives. No caller does this —
`live_forecasts` always passes what `select_nwp_init_time` returned — and the new behaviour is the
more self-consistent of the two, since a caller naming a run is saying that is the run it had. Noted
so the claim is not read as stronger than it is.

**Should `nwp_init_time` become required in single-run mode?** No non-test caller reaches the
derived fallback: `live_forecasts` passes `nwp_init_time` explicitly in both availability modes, so
the documented "Backfilling / None" path is test-only. Requiring it would delete
`_resolve_nwp_init_time`, one branch of the `_engineer_features` docstring, and the last single-run
use of `nwp_publication_delay_hours` — leaving the delay a pure bulk-mode parameter and making the
single-run/bulk split easier to explain. It costs an argument at eight test call sites.
**Recommendation: its own issue afterwards, not this one.** It is a pure API change with no behaviour
at stake, and folding it in would roughly double this diff.

**Should the remaining all-null weather-lag case record a degradation rather than log a warning?**
After this change the only cause left is a run with no control member, which nulls every weather lag
behind nothing more than a `logger.warning`, while
`docs/design-philosophy/inherent-stability.md` asks for degradation recorded on the row and named in
telemetry an alert rule can route on. **Recommendation: check whether #655
(`degrade-missing-control-member-655`) already covers it; if not, raise a separate issue.** Out of
scope here either way.

## Review record

### Review 1 — simplicity (fresh sub-agent, no access to the planning reasoning)

**Accepted, and the plan above is the revised version:**

- **Delete `available_at`/`publication_delay` rather than renaming them to `max_init_time`.** The
  first draft replaced the pair with a single `max_init_time` parameter. The reviewer showed the
  only production single-run caller (`live_forecasts`) loads exactly one NWP run
  (`load_engineering_inputs(..., init_time_start=nwp_init, init_time_end=nwp_init)`), so the shared
  function does not need an availability concept at all — one `.filter()` at the caller does it, and
  the dashboard already sets that precedent. Verified both claims against the source.
- **Cut the 9-hour live-test case** — after the change it exercises the same path as the 6-hour case,
  and each case costs a Dagster materialisation plus an XGBoost train.
- **Fix the existing leakage test's contrivance** rather than leaving it — its
  `power_fcst_init_time = nwp_init_time + NWP_PUBLICATION_DELAY_HOURS` line exists only because of
  the wart being removed, and changing it makes that test fail on `main` too.
- **Delete the cross-mode test's single-run `nwp_publication_delay_hours` argument outright**, not
  just its comment. The reviewer correctly separated the bulk call (which still needs it) from the
  single-run call (which does not).
- **Add `docs/roadmap/switching-events.md` to the docs list.** It names the `available_at` cut and
  would have gone stale unnoticed.
- **Strengthen the reachability paragraph.** The cheapest route is an operator re-materialising a
  missed 06:00 slot with the default config — no code change, no schedule change, no manual backfill.
  Verified `availability_mode` defaults to `"live"`.
- **Shrink the architecture docs change** from a new paragraph to one sentence on the existing
  `"live"` bullet, and scope the "replay is byte-identical" claim to the callers it holds for.

**Rejected:**

- **"Cut the new `test_features.py` test; the fixed leakage test covers it."** The two do overlap.
  Kept anyway because the rule this issue exists to fix was invisible, and pinning it only as a side
  effect of a test named for leakage prevention leaves it invisible.
- **"Make `nwp_init_time` required in single-run mode" as part of this change.** Genuinely good, and
  the reviewer itself recommended deferring; moved to Risks as its own issue.
- **The two-line `publication_delay=timedelta(0)` minimal alternative.** Offered as the smallest
  user-indistinguishable change; the reviewer did not recommend it either. An argument whose only
  job is neutralising the argument beside it is unreadable six months on.

### Review 2 — correctness and testability

Pending.
