# Degrade instead of raise on missing NWP control member (#655)

**Problem.** `_engineer_features` (in `tabular_feature_engineer.py`) raises whenever weather-lag
features are requested and the NWP frame it was handed has no `ensemble_member == 0` (control
member) rows. `live_forecasts` reaches this on every 6-hourly slot through single-run mode, so a
partial or malformed ECMWF ENS download that drops the control member aborts the whole slot for
every series and every ensemble member — a hard failure for what is, per
[inherent-stability rule 1](https://openclimatefix.github.io/nged-substation-forecast/design-philosophy/inherent-stability/#the-rules),
the outside world misbehaving, not our bug.

**Solution.** Gate the raise on `power_fcst_init_time is None` — i.e. keep it in bulk
training/backtesting mode (the only mode any current caller uses to get there: both `cv_assets.py`
call sites), and drop it in single-run mode (production inference and replay backfills). With the
raise gone, a control-member-absent run flows into the weather-lag pipeline's own existing
join-miss path: `select_analysis_proxy` filters to `ensemble_member == 0`, finds nothing, and
`historical_weather` comes back empty, so every past-target-time weather lag left-joins to null —
structurally the same path already exercised, and tested, when the selected run is too fresh to
pass the publication-delay cut (`test_live_weather_lag_nulls_only_when_the_selected_run_is_too_fresh`).
No new nulling logic is needed.

## Verdict, size, departures

**Worth implementing, as described.** The issue's premise checks out against the code
(`tabular_feature_engineer.py:205-213`, `production_assets.py:316`), and the fix it asks for —
delete/scope the raise, confirm the existing null-join path covers it, keep fail-fast for
CV/training — is the right shape.

**Size: complex**, as the issue states. It changes production error-handling behaviour on a path
`live_forecasts` hits every slot. Full plan, all four adversarial reviews.

**Departures from the issue body:**

- The issue asks to "confirm degradation actually reaches all three of the required channels —
  in-band, the warning table, and Sentry — not just doesn't raise." I'm not adding new machinery
  for any of the three, and I think that's correct rather than a shortfall:
  - **In-band (wider uncertainty bands):** doesn't exist anywhere in the codebase yet. Today's
    model is a point forecast (`XGBoostConfig.objective` defaults to `reg:squarederror`);
    band-widening depends on quantile output ([#263](https://github.com/openclimatefix/nged-substation-forecast/issues/263))
    plus regime-conditional conformal calibration, both un-shipped project-wide. The best
    available substitute — XGBoost's default-direction routing over a nulled feature — already
    applies here with no new code, and it's the same substitute every other "feature goes null"
    case in this pipeline relies on today (e.g. the lead-0 window).
  - **Warning table:** `power_forecast_warnings` is not yet in code anywhere
    (`docs/roadmap/delivery-tables.md`: "🚧 Planned ... not yet in code"). There is nothing for
    this change to hook into, and building that table is out of scope for a bug fix in
    `_engineer_features`.
  - **Sentry:** a missing control member is already caught and reported one layer upstream, at
    ingest. `ecmwf_ens` (`assets.py:438`) runs `assess_nwp_run_completeness` on every downloaded
    run and its `_nwp_completeness_check_result` (`assets.py:519`) turns a missing member — control
    member included — into a WARN `AssetCheckResult` naming `missing_ensemble_members` in Dagster's
    Checks view. That's the same "Checks view only, no proactive Sentry push on a WARN pass" level
    every other non-erroring WARN check in this codebase is at today (see the inherent-stability
    failure-modes table: most rows read "nobody is told 🚧" for exactly this reason, tracked under
    [#501](https://github.com/openclimatefix/nged-substation-forecast/issues/501)). Wiring
    WARN-passing checks to Sentry proactively is a cross-cutting gap affecting every check in
    `checks.py` and `assets.py`, not something specific to this failure mode, so I'm not adding a
    one-off Sentry call here that the rest of the codebase doesn't have either.

  So "reaches all three channels" is true only in the weakened sense the rest of the codebase is
  currently held to: in-band via existing null-routing (no quantile bands yet, nowhere), warning
  surfaced in Dagster Checks at the point the fault actually originates (ingest, not inference), and
  Sentry only on an evaluation *error*, not on a clean WARN. If a human reviewer wants this issue
  to also close the general WARN→Sentry gap or build a first slice of `power_forecast_warnings`,
  that's materially bigger than this bug fix and I'd want it as its own issue — flagged in Risks
  below rather than folded in silently.

## What changes, file by file

**`packages/ml_core/src/ml_core/features/tabular_feature_engineer.py`**

- `_engineer_features`: wrap the existing `if nwp_lf is not None and weather_lags and
  nwp_lf.filter(...).limit(1).collect().is_empty(): raise ValueError(...)` block in an additional
  `power_fcst_init_time is None` condition, so it only fires in bulk mode. Single-run mode (used by
  both live inference and replay backfills) skips the check entirely and falls through to the
  existing `historical_weather` construction and `_apply_weather_lag` join, which already produces
  nulls when `historical_weather` is empty.
- Update the `ValueError` message to make clear it's a bulk-mode/training guard (e.g. "... to build
  historical weather during bulk training or backtesting, but no such rows were found ..."), so a
  future reader hitting it in a traceback isn't confused about why single-run mode never sees it.
- Module docstring (lines 12-18, "Lazy Evaluation"): rewrite the "NWP control-member check" bullet
  to say the eager `.collect()` guard runs only in bulk mode (training/backtesting fail-fast); in
  single-run mode (production, replay) a missing control member is absent input, not a contract
  violation, and is left to degrade weather lags to null through the ordinary join-miss path.

**`src/nged_substation_forecast/defs/production_assets.py`**

- `live_forecasts` docstring (the existing paragraph starting "Note: only one NWP run is loaded
  here..."): extend it to also cover this case — a control-member-absent run nulls every weather
  lag feature for that slot, the same way a too-fresh run already does, rather than failing the
  slot. Point at the new test (see below) alongside the existing
  `test_live_weather_lag_nulls_only_when_the_selected_run_is_too_fresh` reference.

**`docs/architecture/performance.md`**

- Line 58 ("The one exception is a `limit(1).collect()` guard...fails loudly instead of silently
  returning an empty frame"): qualify to say this applies in bulk/training mode only, so the
  "always eager, always fails loudly" framing doesn't contradict the new single-run behaviour.

## Design-philosophy check

This is squarely the production-degradation path inherent-stability.md describes. A missing
control member is the outside world misbehaving (a partial/malformed ECMWF ENS download), not our
bug, so per rule 1 it must degrade rather than raise. The fix keeps the CV/training raise (rule 9:
R&D fails fast, because a silently-degraded training run poisons every comparison built on it,
while production fails forward). No asset check changes, so rule 6/7 (WARN, non-blocking,
guarded-body) aren't touched — `assess_nwp_run_completeness`'s existing check already satisfies
them. Delivers toward
[H1](https://openclimatefix.github.io/nged-substation-forecast/design-philosophy/engineering-hypotheses/#h1-a-service-that-mostly-runs-itself):
one fewer way for `live_forecasts` to need a human to intervene on a 6-hourly cadence.

## Tests

**`packages/ml_core/tests/test_features.py`**

- Change `test_engineer_features_raises_when_no_control_member_for_weather_lag` (currently calls
  `_engineer_features` with `power_fcst_init_time=datetime(...)`, i.e. single-run mode) to assert
  the *new* behaviour: no raise, and the weather-lag column comes back entirely null. This is the
  test that would fail on `main` today (it currently asserts a raise) and passes once the raise is
  scoped to bulk mode.
- Add `test_engineer_features_raises_in_bulk_mode_when_no_control_member_for_weather_lag`: same
  fixture shape (NWP with only `ensemble_member=1`, a weather-lag feature requested), called with
  `power_fcst_init_time=None` (bulk mode). Asserts the `ValueError` still raises. This is new
  coverage — nothing today distinguishes bulk-mode from single-run-mode for this check, so this
  test is what would catch a future change accidentally dropping the raise everywhere.

**`tests/test_live_forecasts.py`**

- Add an integration test alongside `test_live_weather_lag_nulls_only_when_the_selected_run_is_too_fresh`,
  reusing its `_save_model_trained_on_weather_lag` / spy-on-`predict` scaffolding: write NWP records
  for the selected run with only non-control members (e.g. `ensemble_member=1`, no `0`), materialise
  `live_forecasts`, and assert (a) the run succeeds — `result.success` — where today it would raise,
  and (b) the captured pre-predict frame's weather-lag column is entirely null. This is the test
  that would fail on `main` today (the materialisation currently raises) and is the direct
  regression test for the issue's failure mode.

## Docs to update

- `packages/ml_core/src/ml_core/features/tabular_feature_engineer.py` module docstring (as above).
- `docs/architecture/performance.md` (as above).
- `src/nged_substation_forecast/defs/production_assets.py`'s `live_forecasts` docstring (as above).
- No roadmap "Implementation details" section covers this issue and no milestone status banner
  changes — #655 isn't a roadmap item, it's a bug fix.

## Verification commands

```bash
uv run ruff check .
uv run ruff format . --check
uv run ty check
uv run pytest packages/ml_core/tests/test_features.py
uv run pytest tests/test_live_forecasts.py
uv run pytest                      # full suite, green-before-push
uv run pymarkdown scan -r docs README.md CLAUDE.md packages/*/README.md
```

## Risks and open questions

1. **Should this issue also add a proactive Sentry event for a WARN-passing completeness check?**
   I've argued no (see Departures above) — it's a cross-cutting gap ([#501](https://github.com/openclimatefix/nged-substation-forecast/issues/501))
   affecting every WARN check in the codebase, not specific to this failure mode, and bolting one
   Sentry call onto `assess_nwp_run_completeness` alone would be an inconsistent, one-off exception
   to how every other check in `checks.py`/`assets.py` behaves today. My recommendation is to leave
   it to #501 and note the connection there. A human reviewer who wants #655 to close that gap for
   this one check specifically should say so.
2. **Does "single-run mode" ever legitimately want the old fail-fast behaviour?** Single-run mode
   also covers replay backfills (`select_nwp_init_time`'s `"replay"` availability mode). A backfill
   of a slot with a genuinely broken NWP run would now silently null its weather lags rather than
   raising. I think that's correct — a backfill re-processing the past should behave like the live
   slot it's replaying, not diverge — but flagging it since the issue text focuses on the *live*
   call site specifically.
3. **Naming:** the issue calls the guard "the NWP control-member check" throughout; I've kept that
   name in the updated docstring/error message rather than inventing new vocabulary, since it's
   already used in `docs/architecture/performance.md` and the tests.
