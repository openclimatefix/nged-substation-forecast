# Issue #652 — should `"live"` availability mode apply `NWP_PUBLICATION_DELAY_HOURS`?

**The problem.** `live_forecasts` picks its one NWP run in `"live"` availability mode, which
applies no modelled publication delay — `select_nwp_init_time` accepts any run genuinely present in
the Delta table. The single-run analysis-proxy join inside `_engineer_features` then gates that same
run with a 9-hour `available_at` cut, so when the selected run is closer than 9 hours to
`power_fcst_init_time` the cut throws the run away and every weather-lag feature for the slot comes
back null. The two pieces of the serving path disagree about what "available" means.

**The planned solution.** In single-run mode, cut the analysis proxy at the NWP run the caller
already selected — keep runs whose `init_time` is at or before `nwp_init_time` — instead of
re-deriving availability from a modelled delay. The selected run is by construction the freshest run
that was available, so a ceiling at that run reproduces the availability set exactly in both
`"live"` and `"replay"` mode, with no `availability_mode` parameter threaded into feature
engineering. Replay behaviour and the derived-`nwp_init_time` backfill path come out byte-identical;
only the live case changes, and it changes to match what bulk-mode training already does.

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

**Applying the delay in live mode is not conservative, it is wrong in the safe direction and
unpredictable in effect.** The cut does not widen uncertainty or record a degradation — it silently
empties a feature, which reads downstream as "no weather" rather than "degraded weather". That is
the fail-open-becoming-fail-closed shape `docs/design-philosophy/inherent-stability.md` warns
about.

**Reachability.** The issue's blast radius today is zero, and that is worth stating plainly: the
champion config selects no weather-lag features, and on the current ingest schedule no live slot
reaches the too-fresh case. With one 00Z run a day downloaded at 08:30 UTC and slots at
00/06/12/18 UTC, the freshest run present is 12, 18, 24, or 30 hours old, all past the 9-hour cut.
The case becomes reachable the moment any of three things happen: a manual `ecmwf_ens` backfill
lands a run earlier in the day than the schedule would, `ecmwf_ens_schedule` moves earlier, or we
ingest more than one ECMWF run a day. None is exotic, and the first needs no code change at all.

**Size: complex.** The change is on the production serving path, which `plan-issue` step 3 makes
complex on its own. It gets this plan, both plan reviews, and both diff reviews in
`implement-issue`.

**Departures from the issue body.** The issue frames the choice as "apply the delay in live mode
too, or declare the asymmetry intentional", and expects the fix to plumb the availability mode into
feature engineering. This plan takes neither branch literally. It resolves the question the second
way — the asymmetry is intentional — but concludes that the single-run cut should therefore stop
modelling availability at all, rather than learning which mode it is in. Cutting at the already
selected `nwp_init_time` is mode-independent and exactly equivalent to the replay cut, so
`ml_core.features` needs no knowledge of `AvailabilityModeType` and no new parameter.

## What changes, file by file

### `packages/weather_utils/src/weather_utils/analysis_proxy.py`

Replace `select_analysis_proxy`'s `available_at: datetime | None` and `publication_delay:
timedelta` parameter pair with a single `max_init_time: datetime | None`, filtering
`pl.col(init_time_col) <= max_init_time`. After the change in `tabular_feature_engineer.py` below,
`available_at` has no non-test caller left, and the replacement says what both surviving uses
actually need: a ceiling on the run, not a modelled publication time. The function stops importing
the publication-delay concept entirely — `NWP_PUBLICATION_DELAY_HOURS` stays in the module for its
other three consumers, but `select_analysis_proxy` no longer refers to it.

Two smaller consequences to carry through:

- The docstring's `available_at` argument entry is rewritten for `max_init_time`, and the
  parenthetical about replay-mode availability moves with it — the caller now owns that reasoning.
- The filter becomes a plain column-against-literal comparison rather than an arithmetic
  expression, which is no worse for the Delta predicate pushdown the docstring asks callers to
  check with `.explain()`.

Rewrite the `NWP_PUBLICATION_DELAY_HOURS` docstring's third paragraph, which currently lists
`select_analysis_proxy`'s `available_at` cut as one of the constant's three consumers. After this
change the consumers are bulk mode's per-row `power_fcst_init_time` derivation, single-run mode's
`nwp_init_time` fallback, and `select_nwp_init_time`'s replay cutoff.

### `packages/ml_core/src/ml_core/features/tabular_feature_engineer.py`

In `_engineer_features`' single-run branch, call `select_analysis_proxy` with
`max_init_time=_resolve_nwp_init_time(nwp_init_time, power_fcst_init_time,
nwp_publication_delay_hours)` in place of the `available_at`/`publication_delay` pair.
`_resolve_nwp_init_time` already exists in `ml_core.features._nwp` and is already imported into this
module by `_check_or_warn_on_missing_control_member`, so no new helper is needed and the three
places that decide "which run is the selected run" stay on one definition.

Rewrite the comment above that call. It currently justifies the cut by the wide-multi-run-frame
hazard; the new comment should say that the selected run is the ceiling because it is by
construction the freshest run that was available, that this reproduces the replay cut exactly and
matches bulk mode's effective ceiling, and that a run later than the selected one is excluded
whether the frame holds one run or many.

Rewrite the `nwp_publication_delay_hours` argument docstring, whose last clause says the delay
"gates the single-run analysis-proxy's `available_at` cut". After the change the delay reaches the
analysis proxy only indirectly, through the derived `nwp_init_time` fallback when the caller omits
`nwp_init_time`.

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
slot. Nothing in the change adds a raise: `select_analysis_proxy` with `max_init_time` can return an
empty frame exactly as it can today, and the downstream weather-lag join treats an empty proxy as a
join miss and produces nulls, which is the existing degradation path. The `ValueError` in
`_engineer_features` on `nwp_init_time` without `power_fcst_init_time` is unchanged, and remains a
caller bug rather than an outside-world condition.

**No asset check is added or edited**, so the `WARN`/`blocking=False` rule is untouched. The
`_check_or_warn_on_missing_control_member` warning path is read but not changed, and it still cannot
raise in single-run mode.

**Inherent-stability rule 1 — liberal about missing inputs, strict about malformed ones — is what
this change serves.** A run that is genuinely present and genuinely usable is not a missing input,
and today's cut treats it as one. The malformed case, a run with no control member, keeps its
existing treatment.

**No principle in `design-principles.md` is traded away.** The change removes a parameter and a
concept rather than adding either.

**Hypotheses.** This change is a correctness fix on the serving path rather than a delivery of a
numbered hypothesis, so it cites none. The nearest relevant claim is the inherent-stability family
about production degrading rather than failing; confirm the exact label against
`docs/design-philosophy/engineering-hypotheses.md` at implementation time and cite it in the PR body
only if one fits, rather than stretching one to fit.

## Tests

### `tests/test_live_forecasts.py` — rewrite the pinning test

`test_live_weather_lag_nulls_only_when_the_selected_run_is_too_fresh` pins the behaviour this change
removes, so it is rewritten rather than extended. Rename it to something that states the new rule —
`test_live_weather_lag_survives_a_run_fresher_than_the_publication_delay` — and change the
parametrised cases:

| Case | `gap_hours` | `members` | `expect_null` today | `expect_null` after |
|---|---|---|---|---|
| Below the delay | 6 | `(0,)` | `True` | **`False`** |
| At the delay | 9 | `(0,)` | `False` | `False` |
| No control member | 9 | `(1,)` | `True` | `True` |

**The assertion that fails on `main` today is the first row**: with a 6-hour gap between the
selected run and `power_fcst_init_time`, `temperature_2m_lag_6h` is entirely null on `main` and
populated after the change. Rewrite the test's docstring to describe the new rule and to keep the
third case's separate explanation (a partial download degrades the same way, and `materialize`
succeeding at all is part of what the case checks).

### `packages/ml_core/tests/test_features.py` — the lookahead guard, unchanged assertion

`test_engineer_features_single_run_freshest_run_excludes_unpublished_nwp_run` builds three runs: T0
at 2026-06-10 00:00, T1 (the selected `nwp_init_time`) at 2026-06-11 00:00, and T2 at 2026-06-11
03:00 carrying a decoy value. Under `max_init_time=T1`, T2 is excluded because it is later than the
selected run and T0 answers, so the test's `== 8.0` assertion holds unchanged. Update the docstring,
which explains the exclusion in terms of `available_at`/`publication_delay`, to explain it in terms
of the run ceiling. **This is the test that proves the change does not reopen lookahead bias**, and
it must keep passing without loosening.

### `packages/ml_core/tests/test_features.py` — one new test

Add a single-run test with an explicit `nwp_init_time` and a `nwp_publication_delay_hours` large
enough that the selected run would fail the old 9-hour cut, asserting the weather lag is populated.
**On `main` this asserts a populated value where the old cut returns null, so it fails today.** The
existing live test above covers the same rule end to end through Dagster; this one pins it at the
`_engineer_features` boundary, where a future refactor of the single-run branch would trip over it
without a Dagster materialisation in the loop.

### `packages/ml_core/tests/test_cross_mode_equivalence.py` — trim an obsolete comment

`test_bulk_and_single_run_features_are_identical` passes `nwp_publication_delay_hours=_DELAY_HOURS`
under a seven-line comment warning that leaving it on the default 9 hours would make the row's own
run look unpublished by the test's own `power_fcst_init_time`. That footgun is exactly what this
change removes, and the comment having been needed at all is corroborating evidence that the cut was
a wart. The argument still has to pass, because bulk mode's `power_fcst_init_time` derivation needs
it; cut the comment down to that reason alone. The test's assertions are unchanged, and it passing
both before and after is the point — cross-mode equivalence must survive the change.

### `packages/weather_utils/tests/test_analysis_proxy.py` — follow the signature

Three tests pass `available_at` (around lines 126, 138, and 205). Rewrite each to pass
`max_init_time` with the equivalent ceiling — where a test today passes `available_at = base + 24h`
against a 9-hour delay, it passes `max_init_time = base + 15h`. Keep each test's assertion and its
intent; only the parameter and the arithmetic in the fixture move. These tests pass before and after
by construction, so they are not tests of this change; they are the signature change's cost.

## Docs to update

- **`docs/architecture/production-deployment.md`**, the "Resolve NWP availability asymmetrically:
  `live` vs `replay`" section. The section describes the asymmetry at
  `select_nwp_init_time` and is silent on the analysis-proxy cut, which is how the two drifted
  apart. Add a short paragraph stating that the weather-lag analysis proxy cuts at the selected NWP
  run in both modes, so the two modes' availability rules meet in exactly one place. Written in the
  present tense, describing how the code works now, with no account of what it used to do.
- **No roadmap ship-time triage.** #652 is a spike under the v0.2.1 follow-ups epic (#642) and
  completes no roadmap item, so there is no "Implementation details" section to delete and no
  status banner to move.
- Check for inbound links to the production-deployment anchor before touching any heading. This
  plan adds a paragraph rather than renaming a heading, so no slug changes, but confirm that at
  implementation time — `analysis_proxy.py`'s `NWP_PUBLICATION_DELAY_HOURS` docstring links to that
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

`mkdocs build --strict` is in the set because the change edits a docs page that carries an anchor
other prose links to. The NWP-convention network tests (`uv run pytest --run-network -m network`)
are **not** needed: nothing here touches NWP download, unit conversion, or the sign convention.

Run the three directly affected test modules first for a fast loop, then the whole suite before
pushing:

```bash
uv run pytest tests/test_live_forecasts.py packages/ml_core/tests/test_features.py \
  packages/ml_core/tests/test_cross_mode_equivalence.py \
  packages/weather_utils/tests/test_analysis_proxy.py
```

## Risks and open questions

**Should `select_analysis_proxy` lose `available_at`/`publication_delay`, or keep them?** The plan
replaces the pair with `max_init_time`. The alternative is a smaller diff: keep `available_at` and
have the single-run branch pass `available_at=<selected run>, publication_delay=timedelta(0)`. That
touches one file instead of three and leaves `test_analysis_proxy.py` alone. **Recommendation: make
the replacement.** A `publication_delay=timedelta(0)` argument whose purpose is to neutralise the
parameter next to it is the kind of call nobody can read six months later, and after this change
`available_at` has no non-test caller to justify keeping it. The issue itself names
`analysis_proxy.py` as a file it expects to change.

**Does a ceiling at `nwp_init_time` really reproduce the replay cut?** Worth a reviewer's own check
rather than taking the plan's word. The argument: `select_nwp_init_time` in replay mode returns
`max{init in table : init + delay <= power_fcst_init_time}`. For any run in the table, `init <=
nwp_init_time` implies `init + delay <= nwp_init_time + delay <= power_fcst_init_time`; and `init +
delay <= power_fcst_init_time` puts that run in the qualifying set, so `init <= max =
nwp_init_time`. The two conditions select the same runs. When the caller omits `nwp_init_time`
entirely, `_resolve_nwp_init_time` derives `power_fcst_init_time - delay`, so the ceiling is
literally the old cut. **If this equivalence is wrong, the plan is wrong**, so it is the first thing
to attack.

**Should the too-fresh case have recorded a degradation rather than silently nulling?** Out of scope
here, because after this change the case no longer arises. But the sibling case — a run with no
control member — still nulls every weather lag behind nothing more than a `logger.warning`, and
`docs/design-philosophy/inherent-stability.md` asks for degradation recorded on the row and named in
telemetry an alert rule can route on. **Recommendation: raise a separate issue** rather than widen
this one, and note that #655 (`degrade-missing-control-member-655`) may already cover it — check
before filing.

**Is the new `test_features.py` test worth its weight, given the live test covers the same rule?**
The two exercise the same behaviour at different altitudes, and the plan keeps both.
**Recommendation: keep both**, because the Dagster test is slow and coarse and the unit test is what
a future refactor of the single-run branch will actually run. A reviewer who disagrees should cut
the unit test rather than the live one, since the live one is what the issue asked to be updated.

## Review record

Filled in as the plan reviews run — each finding taken into the plan, and each finding rejected with
its one-line reason.
