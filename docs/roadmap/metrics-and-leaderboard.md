# Metrics & leaderboard

How OCF measures the skill of its forecasts and compares forecasting approaches.

> **Status legend** — ✅ Implemented · 🚧 Planned · 🔬 Research. The `Metrics` schema, the
> `metrics` Dagster asset, the deterministic metrics (MAE, NMAE, RMSE, MBE), and the
> probabilistic metrics (CRPS, spread-skill ratio, pinball loss, PICP, interval width — see the
> [evaluation-metrics reference](../techniques/evaluation-metrics.md)) are ✅ implemented. The
> interactive leaderboard visualisation is 🚧 planned.
> The implemented [cross-validation protocol](../ml_experimentation/cross-validation-folds.md) has
> moved out of the roadmap. See the [roadmap index](index.md) for status conventions.
> The 🚧 items are tracked under the v0.3 epic
> [#6](https://github.com/openclimatefix/nged-substation-forecast/issues/6):
> baseline forecasters [#147](https://github.com/openclimatefix/nged-substation-forecast/issues/147) ·
> probabilistic evaluation [#225](https://github.com/openclimatefix/nged-substation-forecast/issues/225) ·
> peak-events filter [#254](https://github.com/openclimatefix/nged-substation-forecast/issues/254) ·
> tricky-days filter [#255](https://github.com/openclimatefix/nged-substation-forecast/issues/255) ·
> fold hygiene [#226](https://github.com/openclimatefix/nged-substation-forecast/issues/226).

---

## The leaderboard concept 🚧

Issue: [#4](https://github.com/openclimatefix/nged-substation-forecast/issues/4)

A key deliverable is a **leaderboard** comparing many forecasting approaches. We plan **one
leaderboard per `time_series_type`**, e.g. primary substations, GSPs, BSPs, solar PV sites, wind
farms, BESS, etc.

Each leaderboard will have tens (maybe hundreds) of rows. Each row is one **ML experiment**: a
particular model, trained with a particular set of features, processed a particular way. Entrants
must be compared apples-to-apples — same test dataset, same metrics, same assumptions.

Per-experiment configuration, trained weights, and metrics are stored in the project's **MLflow**
database. The leaderboard will be displayed as an interactive table showing multiple metrics at a
glance, inspired by the [WeirdML leaderboard](https://htihle.github.io/weirdml.html):

![WeirdML leaderboard](assets/WeirdML_leaderboard.png)

---

## Baseline forecasters 🚧

Issue: [#147](https://github.com/openclimatefix/nged-substation-forecast/issues/147)

No naive baseline exists anywhere in the codebase (only docstring mentions, e.g.
`contracts/power_schemas.py:242`). Until the leaderboard carries naive rows, XGBoost's NMAE
numbers aren't interpretable — and, more to the point, we can't answer the question this project
exists to answer: **do we beat what NGED does today?**

### The headline baseline — `nged_incumbent`

`nged_incumbent` is a faithful reproduction of
[NGED's incumbent forecast](../background/nged-incumbent-forecast.md) — the analogue-ensemble method
they use today, with no weather model and no ML. In brief (full description and the operator's-eye
view are in the background page): for each target half-hour it takes the observed power at the
**same weekday & time-of-day** from the **last 6 weeks** and from **49–55 weeks back** — **13
analogues** — which NGED plot and read by eye (taking the 95th percentile if they need a single
number). Reproducing it matters because it is *the bar we have to clear to justify the project* —
"XGBoost beats persistence" is table stakes; "XGBoost beats the incumbent" is the deliverable. It is
the first baseline we implement; if we implement only one, it is this one.

It slots into our machinery beautifully, because every one of its 13 members is just a **power
lag**:

- Weekly group (last 6 weeks, same weekday & time): `power_lag_168h, 336h, 504h, 672h, 840h, 1008h`
- Annual group (49–55 weeks ago, same weekday & time): `power_lag_8232h, 8400h, 8568h, 8736h,
  8904h, 9072h, 9240h`

So it rides the same audited, no-lookahead pipeline as `PersistenceForecaster` (below) with zero
new time-series logic. `_nullify_leaky_lags` already sheds the shortest members as lead time grows
(past 7 days the 168 h member nullifies, past 14 days the 336 h, and so on), leaving the annual
members to carry the full 14-day horizon. Because the shortest member is a week old, the incumbent
has *no* short-horizon skill from recent power — realistic, since that is exactly what NGED do
today, and a reason to keep the pure `PersistenceForecaster` as a contrast rather than to sneak a
recent-power member in.

**It is also our first _probabilistic_ baseline — and this is the faithful representation, not a
bonus.** The plotted spread *is* the incumbent's output — an operator reads it by eye. We emit the
13 analogues as 13 `ensemble_member` rows and let the [probabilistic
metrics](#phase-b-probabilistic-metrics-from-the-existing-ensemble) score them for free — scoring
the spread is the closest automatable proxy for the plot a human actually reads. Two consequences
worth stating plainly:

- **`ensemble_member` is overloaded here.** For NWP models that column indexes an NWP ensemble
  member; for `nged_incumbent` it indexes a *historical analogue*. Same column, different meaning.
  We document this on the `PowerForecast` / `AllFeatures` schema so nobody assumes
  `ensemble_member ⇒ NWP`. The incumbent *synthesises* its ensemble inside `predict()` (by
  unpivoting its analogue-lag columns into member rows) rather than consuming an NWP ensemble; it
  runs with `weather_source: "none"`.
- **Deterministic collapse is a property of the metrics layer, not the incumbent.** The incumbent
  emits its 13 members and nothing else; the [metric-matched collapse
  decision](#which-ensemble-collapse-defines-the-deterministic-point-forecast) then scores its MAE
  on the members' median (apples-to-apples with every other model's central forecast) and reports
  NGED's *actual* operating point — the **95th percentile** — as a labelled secondary number
  (`mae`/`mbe` at `metric_param="p95"`). Being deliberately conservative, the P95 carries a large
  *positive* MBE **by design** (a peak-safety choice, not a forecasting error), so it belongs
  *beside* the central metric, not in place of it. Either way NGED weight the analogues equally ("no
  further processing at all"), so equiprobable members — and the probabilistic metrics (CRPS etc.)
  computed over them — are faithful, not an approximation.

### A faithful replica and a "cheap upgrades" variant

We implement two closely-related incumbent baselines, and the *pair* carries a message that is
itself a valuable project outcome — **most of the benefit may come from a few simple upgrades to
what NGED already do, not from heavy ML**:

- `nged_incumbent` — the faithful replica above. No holiday handling; warts and all. Pure lag
  features.
- `nged_incumbent_holiday_aligned` — the same skeleton, but analogue *selection* becomes
  calendar-aware: a bank-holiday target draws from prior bank holidays / the matching day-type
  (a bank-holiday Monday behaves like a Sunday), and moveable feasts align holiday-to-holiday
  (Easter→Easter) rather than by fixed week offset. This no longer rides the pure lag machinery —
  the analogue offset is conditional on the calendar — so it needs a bespoke picker plus a GB
  bank-holiday calendar (the pure-Python `holidays` package), and ships as an immediate follow-up
  PR, not part of the first one.

### Persistence and climatology — diagnostic bookends

The incumbent is really a *hybrid* — its weekly group is persistence-like recency, its annual
group is climatology-like seasonality — so the two pure forms are still worth having: they isolate
short-horizon vs long-horizon naive skill (at 0–6 h persistence is famously hard to beat; at day
8–14 seasonal climatology often beats everything). A model could "win" the leaderboard while
adding no skill over either, and without these rows nobody would know.

Side benefit: several more `BaseForecaster` implementations pressure-test the abstraction (the
docs promise the interface is model-agnostic; today only `XGBoostForecaster` exercises it).

### Implementation details — baselines (deleted when they ship)

Five PRs, in order. PRs 1–2 are shared-framework groundwork (no baseline yet); PRs 3–5 add one
baseline each. The `nged_incumbent_holiday_aligned` variant (described under [A faithful replica
and a "cheap upgrades" variant](#a-faithful-replica-and-a-cheap-upgrades-variant)) is a later
sixth PR, out of scope for this arc but given its own tracked issue so it is not lost when #147
closes.

**Guiding principle — no special path.** New workspace package `packages/baseline_forecasters/`
mirroring the `xgboost_forecaster` layout (`pyproject.toml`, `src/baseline_forecasters/`,
`tests/`), added to the root `pyproject.toml` `[tool.uv.sources]` and dependencies. Every baseline
subclasses `BaseForecaster` and rides the **identical** asset chain
(`register_experiment_job` → `trained_cv_model` → `cv_power_forecasts` → `metrics`) that
`XGBoostForecaster` uses — a `train()` that only records `trained_time_series_ids` and a
`meta.json`-only `save()` cost nothing, and the uniform provenance is exactly what makes re-running
routine. Re-running CV for any algo is then a native Dagster operation: select **`trained_cv_model++`**
in the asset graph (two `+` — `metrics` is two hops downstream, so a single `+` would silently stop
at `cv_power_forecasts` and skip scoring), choose the experiment's partitions, launch a backfill;
the unpartitioned `metrics` asset materialises once afterwards with its default config (no filter,
`leaderboard` scope). Verify this drill end-to-end on a smoke-test fold before documenting it — the
Dagster version supports mixed partitioned/unpartitioned backfill selections, but confirm the UI
behaviour rather than assuming it. To re-score a *single* experiment without touching the rest, use
the `metrics` asset's `PopulationFilter` config instead.

**PR 1 — deterministic-collapse rework in `compute_metrics`.** Implements the [metric-matched
collapse decision](#which-ensemble-collapse-defines-the-deterministic-point-forecast). Forecasters
emit **members only**; every collapse lives in the metrics layer, so there is no per-experiment
collapse config and no designated point-forecast columns on `PowerForecast`. In
`packages/ml_core/src/ml_core/metrics.py`:

- In the per-run aggregation, emit both `power_fcst_mean` (`.mean()`) and `power_fcst_median` (reuse
  the already-computed `q_p50` empirical-quantile column — Polars `.quantile(0.5, "linear")` is
  exactly `.median()`), and keep `q_p95`.
- Score `mae`/`nmae` on the median error; `rmse`/`mbe` on the mean error; the spread-skill
  denominator stays the mean-error RMSE (its Fortin "1.0 = calibrated" target is defined against the
  mean). Add extra labelled rows: `mae`/`mbe` at `metric_param="p95"` (NGED's operating point) and
  `mbe` at `metric_param="p50"` (bias of the delivered median). `METRIC_PARAMS` already contains
  `"p95"` and `"p50"` (both are in `DELIVERY_QUANTILES`), so **no `Metrics` schema change** — and
  the primary key includes `metric_param`, so the new rows do not collide with the `metric_param="all"`
  headline rows.
- Extend the MLflow allowlist `_MLFLOW_LOGGED_PARAMETRIC` with `("mbe", "p95")` and `("mbe", "p50")`
  so the operating-point bias and the median's bias appear on the leaderboard (aggregate keys come
  out distinct, e.g. `mbe_p95__all` vs `mbe__all`).
- Update the docstrings that currently say deterministic metrics are "scored on the per-run ensemble
  mean" (`METRIC_NAMES` in `contracts/ml_schemas.py`), the `Metrics.metric_param` field description
  (no longer pinball-only), and the `_MLFLOW_LOGGED_PARAMETRIC` key-count claims. The promoted
  [evaluation-metrics reference](../techniques/evaluation-metrics.md) section must state explicitly
  that MAE/NMAE and RMSE/MBE score *different point forecasts* and why (otherwise the first person to
  recompute RMSE from the stored median members files a bug), and note the identity
  `mae ≡ 2 × pinball_loss@p50` as a deliberate internal consistency check.
- Tests: member sets where mean ≠ median ≠ p95, with hand-computed expected values per metric; the
  single-member ensemble (all collapses coincide; CRPS still reduces to MAE); the pinball-p50
  identity. Recompute existing expected values — never relax a test to absorb the shift.
- No standalone re-score needed: the one backfill after PR 2 (below) covers it.

**PR 2 — CV predict-path framework: `uses_nwp_ensemble`, power-lag lookback, `ensemble_member`
docs.** Three changes to the shared rails, none baseline-specific.

- **`uses_nwp_ensemble: ClassVar[bool] = True` on `BaseForecaster`.** `cv_power_forecasts` passes
  `ensemble_members=[0]` to `_load_engineering_inputs` when a class sets it `False`. Semantics to
  document: `True` → the forecaster consumes the NWP member axis (output members = NWP members);
  `False` → the forecaster does **not fan out across NWP members** — it either passes member 0
  through (persistence) or synthesises its own member axis (`nged_incumbent`: 13 analogues;
  `climatology`: quantile-derived members). This is model-family identity, like `MODEL_NAME`, so a
  `ClassVar` is correct — and the class is resolved from the experiment's `forecaster_target` MLflow
  tag *before* inputs are loaded, so the flag is available in time. Document alongside it that
  `weather_source: "none"` does **not** mean "no NWP input": in bulk mode the control-member NWP scan
  defines the shared `(init_time, valid_time)` forecast-run grid, which is what keeps every
  leaderboard row — baseline or ML — scored on the identical grid.
- **Power-lag lookback at feature-engineering load time.** Today `_load_engineering_inputs` filters
  power to `[window_start, window_end]`, and `_apply_power_lag` reads lags from that same frame — so
  any lag longer than the elapsed window is null. This is a real existing gap: XGBoost's 336 h lag is
  null for the first fortnight of every validation window, and the incumbent's 49–55-week lags would
  be null for all but roughly the last three weeks of the fold (lag targets only re-enter the window
  from `val_start + 49 weeks`). Fix: a `power_lookback: timedelta` parameter that widens **only the
  power scan's** lower bound (`window_start − power_lookback`); callers derive it from the
  experiment's `selected_features` via `ParsedFeatures` (max power-lag hours), so it is automatic
  per-experiment (~2 weeks for XGBoost, ~55 weeks for the incumbent). Safe because the bulk-mode
  spine is NWP-centric (`_join_nwp_bulk_mode` left-joins power *onto* NWP rows), so the extra early
  power rows feed only the lag lookup and add no spine rows and no label rows (labels live on spine
  rows, bounded by the NWP `valid_time` filter); and `_nullify_leaky_lags` (which nulls any lag with
  `lead ≥ lag_hours`) still guards leakage, so a longer lookback cannot leak. Apply to both
  `trained_cv_model` and `cv_power_forecasts`. `LIVE_POWER_HISTORY` in `production_assets.py` is a
  hard-coded 15-day equivalent for the live path — leave it, but cross-reference it from the new
  parameter so a future long-lag live model is not silently starved. Add a comment where
  `cv_power_forecasts` re-scans the widened power window per `init_time` chunk (cheap next to NWP,
  but worth flagging so nobody blames the wrong thing when profiling).
- **Document the `ensemble_member` overload** on `PowerForecast` and `AllFeatures`: an NWP-member
  index for NWP-consuming models, a historical-analogue index for `nged_incumbent`, a
  quantile-sample index for `climatology`. Nobody may assume `ensemble_member ⇒ NWP`.
- Tests: a dummy `uses_nwp_ensemble = False` forecaster exercising the member-0 path; a feature-
  engineer test that a 336 h lag at the window edge is non-null *with* lookback while the spine row
  count is unchanged; the leak test unchanged; the single-run (live) path unaffected by
  `power_lookback`.
- **After PRs 1 + 2 land back-to-back, run one `trained_cv_model++` backfill over every existing
  experiment partition** — retrain, re-predict, and re-score everything under the fixed lag lookback
  and the new collapse. This is deliberately the exact "re-run everything after a pipeline fix"
  drill, and it doubles as the empirical verification of the backfill mechanics before the recipe is
  written into `docs/ml_experimentation/dagster-workflow.md`. Treat both PRs as a single leaderboard
  epoch event, since each shifts existing numbers.

**PR 3 — package skeleton + `PersistenceForecaster` (seasonal-naive; the cheapest end-to-end
probe).** Ships the package and proves the PR-2 framework on the simplest model.

- Shared helper `_meta_io.py`: the `meta.json` save/load round-trip (config dump,
  `trained_time_series_ids`, the fully-qualified `model_class`). No `StatelessForecaster` base class
  until a third stateless model exists.
- `MODEL_NAME = "persistence"`, `MODEL_VERSION = 1`, `uses_nwp_ensemble = False`. Config default
  `selected_features = {"power_lag_24h", "power_lag_48h", "power_lag_168h", "power_lag_336h"}`.
- `train()` records the sorted `trained_time_series_ids` (the requested ids present in the data,
  mirroring XGBoost's "no usable rows → not trained" semantics). `predict()` = `pl.coalesce` of the
  lag columns in ascending-lag order — `_nullify_leaky_lags` nulls any lag with `lead ≥ lag_hours`
  (so the 24 h lag survives all intraday leads), and coalesce then selects the shortest *non-leaky*
  lag per row: same-time-yesterday for day-1, last-week for day 2–7, two-weeks-ago beyond. Zero
  lookahead risk because it rides the audited pipeline. Output keeps the spine's `ensemble_member`
  (= 0 only, given member-0 inputs), cast UInt8 → Int8 via an *expression* cast, following
  `XGBoostForecaster._build_part` (a dict-`.cast({...})` on the concat-of-group-by frames would hit
  the Patito cast trap). Rows where all lags are null are dropped, the count logged, and the
  per-series dropped/coverage counts recorded in asset metadata.
- `conf/model/persistence.yaml`: `weather_source: "none"`, `training_strategy: "none"`.
- Tests: unit (shortest-non-leaky selection on a hand-built `AllFeatures` fixture; all-null-row
  dropping; `save`/`load` round-trip freezing ids) plus an integration smoke fold via the
  `tests/test_trained_cv_model.py` fixture pattern.
- Register (`smoke_test` → `full_cv`), materialise the chain, sanity-check: NMAE worse than XGBoost
  overall but plausibly competitive intraday; single-member CRPS = MAE; spread-skill 0; PICP /
  interval width degenerate (expected and ignorable for a deterministic baseline).
- **Coverage caveat:** persistence's longest lag is 336 h, so leads in `[336 h, ~360 h]` drop out and
  its `all` / `extended_range` aggregates cover a shorter lead population than other models'. Fair
  CRPS is size-comparable so nothing is *wrong*, but state the caveat where the sanity numbers are
  read, backed by the per-series coverage counts in asset metadata.

**PR 4 — `NGEDIncumbentForecaster` (`nged_incumbent`; the deliverable).** The faithful replica,
landing on a now-proven rail.

- `MODEL_NAME = "nged_incumbent"`, `MODEL_VERSION = 1`, `uses_nwp_ensemble = False`,
  `weather_source: "none"`. Config `n_weekly_analogues = 6` and `annual_week_span = (49, 55)` drive
  the 13 analogue lags (weekly `168h × {1..6}`; annual `168h × {49..55}` = 8232…9240 h — all within
  the feature parser's 17 520 h cap), with `selected_features` derived from them so variants stay one
  override cheap.
- `predict()` unpivots the 13 analogue-lag columns into `ensemble_member` rows (member index =
  analogue index; Int8 holds 0–12). Members nulled by `_nullify_leaky_lags` (as lead time grows, the
  short weekly members shed first) or by insufficient history are dropped; rows where *all* members
  are null are dropped with the count logged and per-series surviving-member counts recorded in asset
  metadata. No point forecast is emitted — PR 1's metrics layer produces the median headline and the
  p95 / p50 labelled rows.
- Depends on PR 2's lookback (the 55-week annual lags are all-null without it).
- **Data check before interpreting results:** `val_start − 55 weeks` ≈ mid-2024. Confirm which
  eligible series actually have observations that far back — eligibility requires only
  `min_training_months` of history, so a series can qualify yet have too little for the annual
  analogues, degrading silently to a weekly-only (≤6-member) ensemble. Nothing crashes and fair CRPS
  stays size-comparable, but the leaderboard means then average differently-shaped ensembles across
  series, so surface the per-series member counts rather than discovering it later in a dashboard.
- Tests: hand-computed unpivot to the expected members; the median, p95, and p50 collapses match
  hand-computed values through `compute_metrics`; leaky-lag shedding as lead grows; `save`/`load`
  round-trip; an integration smoke fold with a synthetic power history spanning the annual lags.
  Sanity-check: the median is a roughly unbiased central estimate while `mbe@p95` shows a clear
  positive bias (the conservative operating point — **not** a bug); no short-horizon skill (the
  shortest member is a week old, which is realistic).
- Ship-time triage: delete this item's details (summary → PR body); cross-link the [NGED's incumbent
  forecast](../background/nged-incumbent-forecast.md) background page.

**PR 5 — `ClimatologyForecaster` (`climatology`; the pure probabilistic reference).** The calendar-
only skill floor the NWP ensemble must clear at long horizons.

- `MODEL_NAME = "climatology"`, `MODEL_VERSION = 1`, `uses_nwp_ensemble = False`.
- `train()`: from the features frame take `(time_series_id, valid_time, power)` and **dedupe on
  `(time_series_id, valid_time)` first** — bulk-mode `AllFeatures` repeats each target row once per
  covering `(nwp_init_time, ensemble_member)` (~15× at a 15-day horizon), so without the dedupe the
  per-cell samples would be weighted by NWP-run coverage rather than by calendar. Then, per
  `(time_series_id, month, half-hour-of-day, is_weekend)` cell, store the empirical quantiles of that
  cell's power samples. Cell keys derive from **local** (Europe/London) time computed inside the
  forecaster from `valid_time`, aligning with the demand rhythm (and matching the `local_*` time
  features). `save()` writes the lookup as one parquet + `meta.json`.
- **Member emission — deliberate deviation from an earlier draft.** Emit members at *equiprobable*
  quantile levels `(i − 0.5)/m`, **not** at the tail-heavy `DELIVERY_QUANTILES` levels. Fair CRPS and
  the per-run empirical delivery quantiles the metrics layer derives from members treat members as an
  equiprobable sample; feeding 13 members at the delivery levels in with equal weight would put 7.7 %
  of the mass at `q(0.01)` and `q(0.02)`, i.e. an ensemble materially wider-tailed than the
  climatology it represents, corrupting CRPS, PICP, and the derived delivery quantiles. The delivery
  quantiles are still produced — by `compute_metrics`' per-run quantile aggregation, same as every
  other model. Keep `m = 13`: over the ~14-month training window a weekend cell holds only ~9–17
  samples (weekday ~21–43), so quantile levels below ~0.06 are already min-sample extrapolation and
  51 members would be false precision plus ~4× more `power_forecasts` rows; only raise it with
  empirical justification.
- `predict()`: join the lookup onto the prediction rows by the cell keys (rows in an unseen cell
  dropped with the count logged), then unpivot the quantile columns into `ensemble_member` rows.
- **Why a distribution, not a mean:** a deterministic climatology forecast collapses CRPS to MAE, so
  it could only be compared against the ML ensemble on point accuracy — the axis a squared-error
  XGBoost is built to win. The claim we need to test is distributional: does the NWP-driven ensemble
  know more about day 8–14 power than the plain seasonal/time-of-day distribution, or is it dressing
  up climatology in weather-shaped clothing? (See [Probabilistic forecasting from NWP
  ensembles](../techniques/probabilistic-forecasting.md#long-horizons-are-not-automatically-safe).)
  Only a quantile/ensemble climatology answers that, via `crps__all__extended_range` head-to-head.
  Caveat to record where that comparison is read: a *deterministic-quantile* ensemble slightly
  out-scores an i.i.d. ensemble of the same size under fair CRPS (a known property of Ferro's
  correction), so climatology carries a small structural edge in exactly this comparison — read a
  near-tie as "the NWP ensemble is worth little out here", not as climatology winning outright.
- Known limitation, documented not solved: small per-cell samples make the tail quantiles noisy;
  pooling adjacent months is a possible refinement if the numbers look ragged.
- Tests: hand-computed per-cell quantiles; the dedupe (a duplicated spine must not change the
  quantiles); `save`/`load` round-trip; an integration smoke fold; CRPS flows over the members.
- Ship-time triage: unblocks
  [#354](https://github.com/openclimatefix/nged-substation-forecast/issues/354) (the dashboard
  climatology reference band). As the last 🚧 baseline item, delete the whole "Implementation
  details — baselines" section (summary → PR body), close #147, and update the status banner plus the
  milestone section in [`docs/roadmap/index.md`](index.md) if the arc changed.

**Recipe confirmed by NGED (July 2026).** No open questions remain. Full write-up in
[NGED's incumbent forecast](../background/nged-incumbent-forecast.md); the implementation spec:

- **Weekly analogues:** the last **6** weeks, same weekday & time-of-day.
- **Annual analogues:** the **seven** weeks spanning **49–55 weeks back**, same weekday & time.
- **Deterministic value:** NGED's own operating point is the **95th percentile** of all 13 analogue
  values ("more of a vibe") — reported alongside the metric-matched **median** headline (PR 1).
- **No further processing:** no weighting, no holiday handling, no anomaly rejection, no
  load-growth scaling. (This is precisely why the holiday-aligned variant is a genuine, un-done
  upgrade — not a reimplementation of something NGED already do.)

**Cross-cutting.** (1) **Issue hygiene:** create one tracked sub-issue per PR under epic
[#6](https://github.com/openclimatefix/nged-substation-forecast/issues/6) / #147 following the
CLAUDE.md issue-creation rules (labels, Type, OCF project fields, sub-issue ordering), *including*
one for `nged_incumbent_holiday_aligned` so it survives #147 closing. (2) **Re-run recipe:** add a
short "Re-running CV for an experiment" subsection to `docs/ml_experimentation/dagster-workflow.md`
describing the `trained_cv_model++` backfill, written only after the drill is verified end-to-end,
and mentioning `PopulationFilter` for single-experiment scoring. (3) **MLflow re-logging** on a
re-run is idempotent in effect (latest value wins; history accumulates harmlessly).

---

## Cross-fold validation

The cross-validation protocol is **implemented**, so it has moved to its permanent home:
[ML Experimentation → Cross-validation folds](../ml_experimentation/cross-validation-folds.md).
That page covers the expanding-window protocol, the current single fold (and why the available
weather data constrains us to it), the target multiple-yearly-fold protocol, and the fold-design
alternatives we considered.

### Fold hygiene: selection bias and a final-test window 🚧

Issue: [#226](https://github.com/openclimatefix/nged-substation-forecast/issues/226)

The single leaderboard fold (`mid_2025_to_mid_2026` in `conf/cv/default.yaml`: train 2024-04 →
2025-06, validate 2025-07 → 2026-06) serves as **both** the model-selection set and the
reported skill number. Every hyperparameter choice, feature ablation, and model comparison is
adjudicated on the same 12 months that the leaderboard reports. With hundreds of planned
experiments (the roadmap mentions LLM-driven auto-experimentation in v0.5), the winner's
reported skill will be optimistically biased — classic leaderboard overfitting. The epoch
mechanism handles *data* changes but not *adaptive selection* on a fixed fold.

Until the structural fix lands: leaderboard metrics are selection metrics; differences smaller
than fold-level noise should not drive decisions; and the number of experiments per epoch is
itself a relevant statistic (visible as the MLflow experiment count).

#### Implementation details — final-test window (deleted when it ships)

**1. Document the caveat (immediately).** A short "Selection bias" subsection in
`docs/ml_experimentation/cross-validation-folds.md` restating the paragraph above.

**2. Reserve a final-test window (next leaderboard epoch).** Found a new epoch in
`conf/cv/default.yaml` (the epoch mechanism exists for exactly this):

- Shrink the leaderboard fold's validation window to `2025-07-01 → 2026-03-31`.
- Add a `final_test` fold `2026-04-01 → 2026-06-30` with a new per-fold flag
  `final_test: true` (extend `CvConfig` / the fold schema in
  `packages/contracts/src/contracts/hydra_schemas.py`; it is neither a leaderboard fold nor a
  dev fold — `_fold_ids_for_run_mode` in `defs/jobs.py:96` must *not* include it in any
  run mode, so no experiment trains or scores on it in the normal flow).
- Scoring against the final-test window is a deliberate, rare act — only for champion
  candidates immediately before promotion — via the `metrics` asset with
  `evaluation_scope="ad_hoc"` and the window's `valid_time` bounds in the existing
  `PopulationFilter`. No new asset needed; the discipline is procedural. Note: the model
  trained for the leaderboard fold is reused as-is (train window unchanged), so final-test
  scoring needs a `cv_power_forecasts` run over the reserved window — check whether the
  existing asset can forecast a window disjoint from the fold's `val_start/val_end`, and add a
  window override to its config if not.
- Rule, documented alongside: final-test results are never used to *choose between*
  candidates (that re-creates the problem); they exist to report honest skill for the chosen
  champion and to detect gross overfitting (final-test NMAE ≫ validation NMAE).

**A multi-fold gotcha to handle when folds proliferate.** The parent-MLflow-run aggregation in
the `metrics` asset averages each metric key over *only the folds in which that key appears*
(`exp_metrics.setdefault(key, []).append(value)` then `sum/len` in `defs/cv_assets.py`). Today
every fold emits the same key set, so this is invisible — but folds with *different horizon
coverage* (one fold's forecasts stop at 36 h, another's reach day 14) would emit different
per-horizon-slice keys, and a key like `rmse__all__extended_range` would then silently average
over a different fold subset than `rmse__all`, with nothing marking the smaller denominator.
Per-`time_series_type` keys have the same property if fold populations differ. When adding
folds (the multi-fold epoch below, or the `final_test` fold), either guarantee every
leaderboard fold emits an identical key set, or make the parent-run aggregation record its
per-key denominator.

**3. Trade-off (decide at implementation time).** This costs 3 of the 12 validation months, on
a dataset that is already short. The alternative — accepting documented bias until
Dynamical.org backfills enable multiple yearly folds — is defensible; if the backfill is
expected within a couple of months, do part 1 now and fold part 2 into the multi-fold epoch
instead of spending a separate epoch on it. Decide based on the backfill outlook.

**Verification.** (1) `register_experiment_job` in all three run modes never creates a
partition for the `final_test` fold (extend `tests/test_register_experiment_job.py`).
(2) Eligibility for the leaderboard fold is unchanged by the shrunk validation window (the
eligibility rule keys off `val_start`/`val_end` — re-materialise `eligible_time_series` and
diff). (3) End-to-end: score one existing experiment against the reserved window via the
`ad_hoc` metrics path and confirm rows land in `forecast_metrics.delta` with the window label,
and nothing is logged to the leaderboard MLflow runs.

---

## Evaluation metrics

| Metric | Type | Status | Purpose |
|---|---|---|---|
| Mean absolute error (MAE) | Deterministic | ✅ | Typical error magnitude (MW). |
| Normalised MAE (NMAE) | Deterministic | ✅ | MAE normalised by the series' [effective capacity](#normalising-nmae-by-effective_capacity) (full-history P99) — comparable across substations of different sizes. |
| Root mean squared error (RMSE) | Deterministic | ✅ | Heavily penalises large misses (one 100 MW error costs more than two 50 MW errors). |
| Mean bias error (MBE) | Deterministic | ✅ | Systematic over/under-prediction. |
| Histogram of errors | Deterministic | 🚧 | Visual check that errors are ~Normal. |
| [Pinball loss (quantile loss)](../techniques/evaluation-metrics.md#pinball-loss) | Quantile | ✅ | Penalises asymmetrically by target quantile, at the 13 NGED delivery quantiles. Averaged across quantiles for a single quantile-skill score. |
| [PICP (Prediction Interval Coverage Probability)](../techniques/evaluation-metrics.md#picp-prediction-interval-coverage-probability) | Quantile | ✅ | Coverage of six symmetric bands (p1–p99 … p35–p65). Judge against the finite-ensemble calibrated reference (≈ 0.769 for p10–p90 at 51 members), not the nominal rate. |
| [Interval width](../techniques/evaluation-metrics.md#interval-width) | Quantile | ✅ | Mean band width (MW) — the sharpness companion that stops PICP being gamed by over-widening. |
| [CRPS (Continuous Ranked Probability Score)](../techniques/evaluation-metrics.md#crps-continuous-ranked-probability-score) | Ensemble | ✅ | Probabilistic equivalent of MAE; rewards both accuracy and sharpness. Fair (finite-ensemble-unbiased) form — the one metric comparable across ensemble sizes. |
| [Spread-Skill Ratio](../techniques/evaluation-metrics.md#spread-skill-ratio) | Ensemble | ✅ | Fortin-corrected RMS ensemble spread ÷ RMSE of the ensemble mean. 1.0 = well-calibrated; < 1 under-dispersed (overconfident); > 1 over-dispersed (underconfident). |

> The `Metrics` schema (`contracts.ml_schemas.Metrics`) stores results as
> `(time_series_id, power_fcst_model_name, fold_id, horizon_slice, metric_name, metric_param,
> metric_value)`. `metric_param` carries, e.g., the quantile for Pinball Loss (`p10`) or the band
> for PICP (`p10_p90`). The `metrics` Dagster asset computes every ✅ metric above and writes
> per-series rows to `forecast_metrics` Delta (partitioned by `experiment_name, fold_id`), with
> per-fold and mean-across-folds aggregates logged to MLflow — see
> [Running an ML experiment end-to-end](../ml_experimentation/dagster-workflow.md#step-8-materialise-metrics).

### Which ensemble collapse defines the deterministic point forecast? 🚧

**Decided: metric-matched collapse, uniform across every model.** Implemented as part of the
baseline work ([PR 1 in the baseline implementation
details](#implementation-details-baselines-deleted-when-they-ship)); the reasoning below is the
durable design rationale, promoted to
[the evaluation-metrics reference](../techniques/evaluation-metrics.md) when that PR ships.

#### The problem

The deterministic metrics (MAE, NMAE, RMSE, MBE) score a **single point forecast**, but every model
on the leaderboard is really an *ensemble* (51 NWP members for the ML models; 13 historical analogues
for `nged_incumbent`; a quantile sample for `climatology`). Something has to collapse each ensemble
to one number. `compute_metrics` today collapses every ensemble to its **mean**
(`packages/ml_core/src/ml_core/metrics.py`), and the risk we were guarding against was that
different models would be scored on *different* collapses — e.g. the ML models on their mean and
`nged_incumbent` on the median NGED effectively reads off its analogue spread. Mean and median
diverge for skewed or underdispersed ensembles, so scoring some models on one and some on the other
is **not apples-to-apples** — a silent trap that quietly mis-ranks models.

#### Mean versus median — the trade-off

The instinct is to "pick one central statistic and apply it everywhere". But mean and median are not
interchangeable, and the reason is the crux of the decision:

- The **mean** is the point forecast that minimises **squared error** — so RMSE (and MBE, which is
  a mean-of-errors and inherits the mean's clean energy-balance / expectations-aggregate reading)
  is *consistent* with the mean. Our squared-error XGBoost models literally learn a conditional
  mean, and the ensemble mean of 51 such members is a coherent estimate of `E[power]`. Scoring the
  mean on RMSE rewards a model for doing the statistically correct thing; the mean also averages out
  member noise, so it is the more stable statistic on a small ensemble, and it matches standard NWP
  verification practice.
- The **median** is the point forecast that minimises **absolute error** — so MAE and NMAE are
  *consistent* with the median. It is robust to the skew that is real in this problem (holiday
  weeks, solar clipping) and to the ensemble underdispersion the [probabilistic
  section](#delivering-the-probabilistic-metrics) documents, it is the faithful reading of NGED's
  equally-weighted analogue spread, and it is coherent with the quantile columns already on the
  leaderboard: median MAE is exactly `2 × pinball_loss@p50`, so a median headline makes the
  deterministic and probabilistic columns tell one story.

The key realisation is that **MAE and RMSE elicit *different* functionals, so no single collapse is
fair on both columns.** A uniform median is inconsistent for RMSE (it penalises squared-error models
for forecasting their honest mean); a uniform mean is inconsistent for MAE (a model could improve its
MAE ranking by warping its forecast away from its honest mean — Gneiting 2011, *Making and Evaluating
Point Forecasts*). The apples-to-apples requirement is only that the collapse be uniform *across
models*, not *across metrics*.

#### The decision

Score each deterministic metric on the ensemble collapse it is consistent with, **the same way for
every model**:

- **MAE / NMAE ← ensemble median** (NMAE is the headline cross-series metric, so the headline is
  effectively median-based).
- **RMSE / MBE ← ensemble mean.**
- **Spread-skill ratio ← ensemble mean, internally, unchanged** — its Fortin calibration target
  (RMSE of the ensemble mean = `√((m+1)/m) ×` RMS spread, so "1.0 = calibrated") is *defined*
  against the mean; switching its internal collapse would silently break that reading.

Everything else is an **extra, labelled** row, never a headline: NGED's **P95** operating point
(`mae`/`mbe` at `metric_param="p95"` — conservative by design, so a large positive MBE that belongs
*beside* the central number, not in place of it), and the **median's own bias** (`mbe` at
`metric_param="p50"`, so the delivered central forecast has an honest bias number distinct from the
mean's energy-balance bias). No model is structurally disadvantaged on any column — which a single
uniform statistic cannot achieve — and the trap is closed because the collapse is uniform across
models. Because the collapse lives entirely *downstream* of the stored forecasts, switching it
re-scores the whole leaderboard from a single `metrics` re-materialisation with no retraining or
re-prediction — and doing it now, before the leaderboard adjudicates anything, is the cheapest moment
to shift every existing deterministic number.

### Normalising NMAE by `effective_capacity`

NMAE is MAE divided by a per-series **effective capacity**, not by the mean or a per-fold P99. A
capacity-like denominator is what makes NMAE comparable across asset types: intermittent generators
(PV, wind) spend much of their time near zero output, so normalising by the *mean* would inflate
their NMAE relative to a demand substation of similar peak size. Computing the denominator over each
series' **full history** (rather than within the validation window) also keeps it stable across
folds — an unusually calm year for a wind farm would otherwise give a low in-window P99 and an
inflated NMAE.

Normalisation is also why NMAE is the **headline cross-series metric**: the aggregate `mae__all` /
`rmse__all` values logged to MLflow are unweighted means across series whose scales span roughly
two orders of magnitude, so the GSPs dominate them. They are useful for tracking a single model
over time, not for comparing skill across the population.

The denominator comes from the [`effective_capacity`](delivery-tables.md#table-4-effective_capacity)
Delta table (schema `contracts.power_schemas.EffectiveCapacity`), consumed by `compute_metrics`
(`ml_core.metrics`).

**v0.1 representation: one scalar row per series.** The `effective_capacity` asset writes one
row per `time_series_id` — `effective_capacity_mw` = P99 of `|power|` over the whole observation
history, `time` = the latest observed timestep. `compute_metrics` joins it onto the per-series
metrics **on `time_series_id` alone** and divides.

**Why v0.1 is a single row per series, not the value repeated at every half-hour.** The
v0.7 upgrade below *will* store one row per `(time_series_id, time)` half-hour — but with a
genuinely *time-varying* value. In v0.1 the value is a single constant per series, so repeating it
across every half-hour would just be a denormalised encoding of one number: at V2 scale (~2,500
series × ~4 years × 17,520 half-hours/yr ≈ 175M rows) that is hundreds of millions of rows to
express ~2,500 scalars, for zero extra information. It would also *not* buy forward-compatibility,
because the real v0.1→v0.7 interface change is not the data shape but **the join** (below). The
`EffectiveCapacity` schema — `(time_series_id, time, effective_capacity_mw)` — already accommodates
both the one-row-per-series v0.1 shape and the one-row-per-half-hour v0.7 shape; that is the
forward-compatibility we want. (The v0.7 upgrade does widen the *columns* — the value becomes a
mean + std pair,
[#247](https://github.com/openclimatefix/nged-substation-forecast/issues/247) — but the row shape
and the join are unaffected by that.)

**v0.7 upgrade: time-varying, and the join changes.** The
[differentiable-physics](capacity-estimation.md) capacity model produces a value that changes over
time (panel degradation, inverter trips, seasonal derating). At that point two things change, and
nothing else:

- the `effective_capacity` asset body emits one row per `(time_series_id, time)`; and
- `compute_metrics` changes its capacity join from `time_series_id`-only to a **temporal as-of join**
  on `(time_series_id, valid_time)` — matching each forecast's `valid_time` to the capacity in effect
  at that time.

The `Metrics` schema and the rest of the metrics pipeline are untouched. Note the table is
**backward-looking only** (it holds no future `valid_time`s): fine for historical CV folds (whose
validation windows lie inside the observed history), but live-forecast scoring
([production monitoring](live-service.md#production-monitoring)) must
choose which reference time's capacity to apply rather than expecting a row at a future `valid_time`.

One related distinction to keep straight: the *metric denominator* may use the full-history
**smoothed** capacity estimate, but any capacity used to normalise model inputs at forecast init
time (the two-pass training scheme) must be the **causal** estimate available at that init time, or
backtests gain lookahead — see
[Capacity estimation](capacity-estimation.md#causal-vs-smoothed-capacity-a-lookahead-trap-in-the-two-pass-scheme).

### Peak events — the metric filter that matters most for flexibility

Issue: [#254](https://github.com/openclimatefix/nged-substation-forecast/issues/254)

Because NGED's goal is **flexibility procurement** (entirely about peak management and congestion),
overall RMSE only tells half the story. We add a **"Peak Events"** filter:

- **Peak RMSE / Peak Pinball Loss**: score models *only* on the top 5% highest-demand half-hours
  (or hours where solar generation unexpectedly drops during peak demand).
- **Hand-picked "hard examples"**: if NGED supplies a list of historically tricky times, we compute
  performance on those alone.

### Tricky days — a calendar-deterministic metric filter 🚧

Issue: [#255](https://github.com/openclimatefix/nged-substation-forecast/issues/255)

Alongside Peak Events, we add a **"Tricky days"** filter: score models separately on the handful of
calendar dates whose demand shape departs sharply from the usual weekly rhythm. We scope it
deliberately to the **calendar-deterministic** set — fixed and moveable public holidays (Christmas,
Easter, and the rest of the GB bank holidays) plus the two annual daylight-saving transitions —
because these are exactly the days our weekday/seasonal analogues are *built* to mishandle. Days
that are hard for *data-dependent* reasons stay in their own already-planned filters:
[switching-event days](#measuring-performance-during-switching-events) and NGED's hand-picked
"hard examples" (above). One shared filter *mechanism*, several named filters — folding genuinely
different failure modes into one bucket would make the number impossible to act on ("bad on tricky
days" — is that Christmas, or a switching event?).

Mechanically it is another population filter (the same mechanism the planned Peak Events filter
uses): a boolean flag per timestep, derived from `valid_time` alone. Because it is purely
calendar-driven it **shares its calendar module with `nged_incumbent_holiday_aligned`** — the same
GB bank-holiday calendar (the pure-Python `holidays` package) plus the two DST dates feed both the
holiday-aligned baseline and this metric filter. And the two reinforce each other: `nged_incumbent`
(no holiday logic) should be *visibly* worst on tricky days, and `nged_incumbent_holiday_aligned`
should recover most of the gap — turning "we added holiday alignment" into a *measurable* number,
exactly the cheap-upgrades story we want to show NGED.

**Flag the day _and_ its analogue-relevant neighbours, not just the day itself.** The disruption
spills onto surrounding timesteps:

- **DST transitions**: the hard part is not only the 23/25-hour day but that lag and analogue
  features are misaligned by an hour on the days either side.
- **The Christmas run-up**: demand in the days *before* Christmas is already atypical, so the window
  must cover the run-up, not just the 25th.

So the flag covers a small **window** around each date rather than a single day; the exact per-event
widths are an implementation-time choice.

**A subtlety to document now but _not_ model yet.** The shape of the Christmas run-up depends not
just on the number of days before Christmas but also on **which weekday Christmas falls on** — the
run-up demand pattern shifts year to year with that day-of-week alignment. We record it here as a
known effect; the v0.3 tricky-days *flag* ignores it (it simply marks the window), and we defer any
explicit day-of-week-aware modelling of the run-up until there is evidence it moves the leaderboard.

#### Implementation details — tricky days (deleted when this ships)

- A small calendar module (shared with baseline 2, `nged_incumbent_holiday_aligned`) answers, for
  any `valid_time`, whether it falls inside a tricky-days window. Back it with the `holidays` GB
  calendar plus the two annual DST dates; expose the per-event window widths as config.
- Represent the tricky-days slice the same way the Peak Events filter is represented — one more
  named population filter, resolved by the same mechanism, **not** a new schema axis — so the
  leaderboard gains a **Tricky days** column with no `Metrics` schema change.
- Verification: unit-test the flag on known dates (a Christmas week, an Easter, both DST
  switchovers, and a plain week that must be *excluded*); on a smoke-test fold, confirm
  `nged_incumbent` scores worse on the tricky-days slice than overall.

---

## Time-slices for performance evaluation

We compute every metric separately per horizon slice, because the driver of model skill changes
with lead time:

| Horizon slice | Industry term | Primary driver of model skill |
|---|---|---|
| 0–6 hours | Intraday / Nowcasting | Lagged power & persistence. NWP is often too coarse to beat simple autoregressive features here. |
| 6–36 hours | Day-Ahead | Deterministic NWP. Covers the critical day-ahead market gate; relies on the diurnal cycle + high-res weather. |
| Day 2–7 | Short/Medium Range | Synoptic weather. Skill driven by mapping large weather fronts to power; ensemble spread starts to matter. |
| Day 8–14 | Extended Range | Ensemble probabilities. Deterministic weather is essentially noise; skill comes from processing ensemble uncertainty. |

### Measuring performance during switching events 🚧

We will flag each timestep for whether it contains a switching event, and compute metrics separately
for periods with switching events in the model inputs (or in the forecast's `valid_time`). This
distinguishes models that perform well *only* on clean periods from models that handle switching
events in their inputs. In v1 the flags can come straight from NGED's logged switching events
for the trial area; a fleet-scale version would need the discrete detector described in
[Switching events & latent demand](switching-events.md), whose fate is an open question — see
[the decision point](switching-events.md#the-decision-point-a-feature-based-mainline-vs-the-staged-detector).

---

## Delivering the probabilistic metrics 🚧

Issue: [#225](https://github.com/openclimatefix/nged-substation-forecast/issues/225)

The 51-member ensemble is very likely **underdispersed**: XGBoost trained with squared error on
the control member (`cv_assets.py:373`) learns a conditional mean, so pushing 51 members through
it yields spread from *weather uncertainty only* — no model or observation uncertainty. Such
ensembles are systematically overconfident, worst at short horizons where members haven't
diverged. Flexibility procurement is a tails problem (P90+ peaks), so this hits the use case
directly. Phases A and B below (both shipped) built the measurement machinery — every metric in
the [evaluation-metrics reference](../techniques/evaluation-metrics.md), per horizon slice; the
remaining phases act on what those numbers show.

The theory behind this diagnosis — the three-term uncertainty decomposition, why a
deterministic model driven by an NWP ensemble captures only the weather term, and what the
principled fix looks like — is the durable explainer
[Probabilistic forecasting from NWP ensembles](../techniques/probabilistic-forecasting.md).
This section is the *plan* that applies it.

Fix in four phases, each an independent PR (Phase D is itself several PRs). Phases A and B are
pure evaluation (no model changes) and should land before any further MAE-driven
experimentation.

### Phase A — horizon-sliced metrics ✅

Shipped. `compute_metrics` now scores every metric per `HORIZON_SLICES` band (derived from
`valid_time − power_fcst_init_time`, with the ensemble collapsed per forecast run), and
`build_mlflow_aggregate_metrics` logs overall per-slice keys like `nmae__all__day_ahead`.

### Phase B — probabilistic metrics from the existing ensemble ✅

Shipped. `compute_metrics` now computes fair CRPS, the Fortin-corrected spread-skill ratio,
pinball loss at the thirteen NGED delivery quantiles (plus their mean), and PICP and interval
width for the six symmetric delivery bands — all member-aware, computed before the
ensemble-mean collapse, per horizon slice. Definitions, equations, and the design decisions
(fair CRPS divisor, RMS spread, delivery quantiles, MLflow allowlist) live in the
[evaluation-metrics reference](../techniques/evaluation-metrics.md).

### Phase C — cheap calibration (after B proves the diagnosis)

Decide based on Phase B's spread-skill numbers — and on the **rank (Talagrand) histogram** of
the observations among the 51 members, computed ad hoc per horizon slice: a U-shape confirms
plain underdispersion (a single multiplicative inflation can fix it), whereas a sloped or
asymmetric histogram means bias or shape error, which a symmetric inflation cannot repair and
which would push toward a rank-dependent correction instead. **Post-hoc spread inflation**
(EMOS-lite): per
horizon slice, fit a scalar `s` on the *training* window so that inflating members around the
ensemble mean (`mean + s·(member − mean)`) makes spread match error. Zero schema change, zero
new model — implementable as an optional step in `predict` or as a wrapper forecaster. Fit on
train, apply on validation (no tuning on the fold being scored).

Spread inflation widens the fan but cannot reshape it (the inflated ensemble is still 51 point
forecasts, just pushed apart). It is the stopgap the full fix below must beat to earn its
build cost.

### Phase D — ensemble of quantile forecasts (Representation 3 → pooled Representation 2)

The full fix from the
[probabilistic-forecasting explainer](../techniques/probabilistic-forecasting.md): have the
model emit a **conditional distribution per ensemble member** (per-member quantiles —
[Representation 3](delivery-tables.md#representation-3-ensemble-of-percentile-forecasts) in the
delivery tables), then recombine the 51 members with the **linear-pool mixture** into one set
of delivered percentiles
([Representation 2](delivery-tables.md#representation-2-percentiles)). Three steps, one PR
each:

1. **Percentile representations in `PowerForecast`**
   ([#262](https://github.com/openclimatefix/nged-substation-forecast/issues/262)) — extend the
   contract (and the Delta write/read paths) with the Rep 2 and Rep 3 percentile columns,
   alongside the existing deterministic-ensemble representation.
2. **Quantile XGBoost model family**
   ([#263](https://github.com/openclimatefix/nged-substation-forecast/issues/263)) —
   `objective: reg:quantileerror` with several
   `quantile_alpha`s, as a separate experiment/model family emitting Rep 3. Sort each member's
   quantiles at predict time (monotonic rearrangement fixes quantile crossing). The lead-time
   feature and training on multiple members
   ([xgboost-improvements](xgboost-improvements.md) items 1 and 16) are the double-counting
   mitigations discussed in the explainer — land them first or measure without them
   consciously.
3. **Linear-pool combining step**
   ([#264](https://github.com/openclimatefix/nged-substation-forecast/issues/264)) — pool the
   per-member quantiles into delivered Rep 2
   percentiles (the pseudo-sample recipe in the explainer), with a per-horizon affine
   recalibration hook (fit on train) applied only if the pooled spread-skill/PICP numbers
   demand it. Scored with pinball/PICP/CRPS head-to-head against the Phase-C inflated
   deterministic champion.

---

## Grouping the results

Each ML experiment is tagged with metadata so we can group experiments and compute average
performance per group (e.g. "does lagged power *always* help, regardless of model sophistication?",
or "how robust is each model to weather-forecast uncertainty — CERRA reanalysis vs. operational
NWP?"). Example tags:

| Tag | Example values |
|---|---|
| `time_series_type` | PV, Wind, disaggregated demand (primaries) |
| `model_family` | nged_incumbent, baseline_persistence, xgboost, pytorch_mlp, pytorch_graph_dp |
| `weather_source` | none, ecmwf_control, full_ecmwf_ensemble, cerra |
| `input_features` | datetime, power_lag_24h, power_lag_7d, temperature |
| `training_strategy` | direct_multistep, horizon_as_feature, end_to_end |
| `generator_capacity_estimation` | none, simple_p99, convex_envelope, differentiable_physics |
| `switching_event_detection` | none, simple_statistical |
| `pre_training` | none, CERRA |

> Estimating **cost savings (£)** attributable to each forecasting approach, per leaderboard row, is
> a 🔬 v2 stretch goal.
