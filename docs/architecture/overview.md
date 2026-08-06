# Architecture Overview

The architecture prioritises developer velocity, idempotent re-runs, and strict **Training-Serving Symmetry**. The [design principles](#design-principles) below are the short version; the rest of this page, and the rest of `architecture/`, is the detail behind them.

The primary aim is to develop novel, ambitious, state-of-the-art ML approaches to forecasting. We are simultaneously building a "test-harness" production service so that ML research runs in a production-like environment from day one.

The aim is to manage the *entire* data pipeline in Dagster: download data, validate data, train ML models, run inference, perform backtests. MLflow tracks every experiment. Re-running a backtest should be as easy as clicking a button in Dagster. Swapping a new model into production should require minimal friction.

The system is designed as a modular monorepo using [uv workspaces](https://docs.astral.sh/uv/concepts/projects/workspaces/), with [Dagster](https://dagster.io/) orchestrating the data pipeline and [MLflow](https://mlflow.org/) tracking experiments.

## Design principles

Eleven principles govern the decisions on the rest of this page and across `architecture/`. They are
deliberately short here and link out to the page that argues each one in full — this is an index,
not the argument.

**Where they come from.** A stated aim of this project is to gather **industry best practice into a
single codebase for energy forecasting** — and not only best practice from the energy-forecasting
industry. Several of the principles below are borrowed from disciplines that have been solving the
same shape of problem for longer than data engineering has existed: *inherent stability* comes from
vehicle dynamics, *fail-operational* from avionics autoland and ISO 26262, *blast radius* and *error
budgets* from site reliability engineering, *poka-yoke* (mistake-proofing) from manufacturing.
Borrowing across disciplines is the point, not a flourish: a discipline that has been shipping
safety-critical systems for fifty years has usually already made the mistake we are about to make.
Where a borrowed term does not fit cleanly, we say so and disown it — Postel's law is named on the
[Inherent Stability](inherent-stability.md#not-postels-law) page precisely so that nobody mistakes it
for what we do.

**How these relate to the [engineering hypotheses](../engineering-hypotheses.md).** Principles are
constraints on *decisions*; hypotheses are claims about *outcomes*. A hypothesis can be falsified by
measurement, whereas a principle can only be overridden by a measurement or found not to be
load-bearing. The two are connected in one direction: the principles are the bets we are making in
order to achieve the hypotheses. That gives a test for admission to this list — **name the hypothesis
it serves and a decision it actually decided**. A principle serving no hypothesis is either
decoration or a sign of a missing hypothesis; a hypothesis with no principle behind it is a claim we
are merely hoping comes true.

1. **The forecast never stops. It gets less certain instead — and says so in the answer itself.**
   When inputs degrade, the system's normal path already moves toward a sensible, wider, still-true
   answer, rather than raising. Raising is reserved for states that are our own bug.
   *Decided:* every asset check in the repo is non-blocking `WARN`; there is deliberately no
   `ERROR`-severity check anywhere. *Serves:*
   [H1](../engineering-hypotheses.md#h1-a-service-that-mostly-runs-itself). *Detail:*
   [Inherent Stability](inherent-stability.md), whose [ten rules](inherent-stability.md#the-rules)
   are the fine-grained form of this principle.

2. **Complexity belongs offline, not in the serving path.** When a capability could be built into
   the training loop or into the production service, build it into the training loop: training runs
   in front of a human who can read the traceback, whereas the service runs unattended on the day
   the inputs are strangest. Note that this *relocates* a branch rather than removing it — a model
   that "handles anything" holds the same branch internally as a learned default direction — so it
   is only safe to lean on once a failure-scenario suite exists to measure the result.
   *Decided:* `promoted_model` copies the champion to local disk, so production inference makes no
   MLflow call at all. *Serves:*
   [H1](../engineering-hypotheses.md#h1-a-service-that-mostly-runs-itself),
   [H3](../engineering-hypotheses.md#h3-one-click-promotion-and-one-click-rollback). *Detail:*
   [Where complexity should live](inherent-stability.md#where-complexity-should-live),
   [Bake the model into the image](production-deployment.md#bake-the-model-into-the-image-at-build-time).

3. **One execution path from research to production.** The artifact that won the experiment *is* the
   artifact we deploy — not a re-implementation of it. There is no "now rewrite the research code for
   production" step, because every experiment already runs on the production pipeline.
   *Decided:* this was the deciding argument for Dagster over Airflow, and it is why splitting the
   live service onto a second orchestrator remains rejected. *Serves:*
   [H2](../engineering-hypotheses.md#h2-a-hundred-experiments-per-person-in-a-peak-month),
   [H3](../engineering-hypotheses.md#h3-one-click-promotion-and-one-click-rollback). *Detail:*
   [Nothing gets rewritten on the way to production](../ml_experimentation/mlops-approach.md#nothing-gets-rewritten-on-the-way-to-production),
   [Why Dagster, not Airflow?](why-dagster-not-airflow.md).

4. **Strict contracts at every boundary — liberal about missing inputs, strict about malformed
   ones.** Every tabular boundary is a Patito schema, validated rather than assumed. This is the
   deliberate *opposite* of Postel's law, and it is what stops principle 1 from decaying into
   "accept anything and hope".
   *Decided:* `AllFeatures`, `PowerForecast` and the rest are validated schemas rather than
   conventions, and every `PowerForecast` row is self-describing. *Serves:*
   [H1](../engineering-hypotheses.md#h1-a-service-that-mostly-runs-itself),
   [H6](../engineering-hypotheses.md#h6-scale-without-redesign). *Detail:*
   [Strict data contracts](forecast-delivery.md#strict-data-contracts-machine-verifiable),
   [Not Postel's law](inherent-stability.md#not-postels-law).

5. **Reject structurally-broken data; tolerate locally-corrupt data a model can absorb.** The
   granularity of rejection matters as much as the strictness. Throwing away an otherwise-good NWP
   run because a few percent of its pixels are null would convert a tolerable problem into an
   outage; failing to reject a wholly-broken slice would let silent corruption through.
   *Decided:* `Nwp.validate` fails a run only on a *whole-slice* null in a de-accumulated variable,
   and merely warns on scattered per-pixel nulls — which usefully converts fine-grained catastrophe
   into a coarse-grained missed run, the form principle 1 already handles. *Serves:*
   [H1](../engineering-hypotheses.md#h1-a-service-that-mostly-runs-itself). *Detail:*
   [The guiding principle](ecmwf-ens-known-issues.md#the-guiding-principle).

6. **Every comparison must differ in exactly one thing.** A leaderboard is only worth having if two
   numbers on it are genuinely comparable, which requires the population, the folds, the metric
   definitions and the pipeline to be held constant by construction rather than by discipline.
   *Decided:* fold eligibility is derived from data coverage alone and **never** from the model or
   config; a fold enters the leaderboard only once its validation window is complete; a new data
   source is assessed by a controlled ablation before it may enter the leaderboard at all. *Serves:*
   [H2](../engineering-hypotheses.md#h2-a-hundred-experiments-per-person-in-a-peak-month). *Detail:*
   [Eligibility](../ml_experimentation/cross-validation-folds.md#eligibility),
   [Complete validation windows only](ml-orchestration.md#complete-validation-windows-only),
   [Evaluating new data sources](../ml_experimentation/evaluating-new-data-sources.md).

7. **Provenance travels with the data.** Every row carries enough to say where it came from, so a
   forecast can be explained, reproduced or invalidated without an external lookup.
   *Decided:* `PowerForecast` carries `nwp_init_time`, model name and version, experiment name and
   MLflow experiment id; every MLflow run is stamped with the git SHA and the Delta table versions
   it read. *Serves:*
   [H2](../engineering-hypotheses.md#h2-a-hundred-experiments-per-person-in-a-peak-month),
   [H6](../engineering-hypotheses.md#h6-scale-without-redesign). *Detail:*
   [Two metric stores](ml-orchestration.md#two-metric-stores-one-division-of-labour),
   [The Universal Model Interface](#the-universal-model-interface).

8. **Every write is idempotent, and every failure is confined to one partition.** Re-running
   anything must be safe, and a failure must not be able to spread beyond the partition it happened
   in. Note that this is a *blast-radius* property — how much fails — which is a different axis from
   principle 1, which is about which *way* a thing fails.
   *Decided:* re-materialising a fold overwrites its `(experiment_name, fold_id)` partition rather
   than appending, so a retry cannot silently double-count; parallel experiments write to disjoint
   partition directories and never touch each other. *Serves:*
   [H1](../engineering-hypotheses.md#h1-a-service-that-mostly-runs-itself),
   [H6](../engineering-hypotheses.md#h6-scale-without-redesign). *Detail:*
   [Idempotent writes and concurrency](ml-orchestration.md#idempotent-writes-and-concurrency),
   [Serve only the trained population](production-deployment.md#serve-only-the-trained-population).

9. **Push the work down to the engine; materialise once, as late as possible.** No code between
   storage and the model boundary may call `.collect()`, so the query engine sees the whole plan and
   prunes the scan before any data crosses the wire. At this data scale the alternative is not slow,
   it is impossible.
   *Decided:* input pruning plus `init_time` chunking keeps a full 51-member validation prediction
   (~321M rows) under 9 GB on a laptop. *Serves:*
   [H6](../engineering-hypotheses.md#h6-scale-without-redesign). *Detail:*
   [Lazy Evaluation Strategy](#lazy-evaluation-strategy).

10. **Measure; do not assume.** Performance, size and cost claims are benchmarked on real data,
    through the real code path, before they are believed — and the measurement is written down next
    to the decision it justified, so a later reader can tell which numbers are still true.
    *Decided:* `BYTE_STREAM_SPLIT` is used for `power_fcst` but deliberately *not* for NWP, because
    it measured *worse* there; the NWP scan-pruning rules were each verified with
    `LazyFrame.explain()` rather than reasoned about. *Serves:*
    [H4](../engineering-hypotheses.md#h4-it-runs-for-pocket-money),
    [H6](../engineering-hypotheses.md#h6-scale-without-redesign). *Detail:*
    [Core Components](#core-components) (the measured storage-format numbers),
    [Bounding feature-engineering memory](#bounding-feature-engineering-memory-prune-the-inputs-not-the-output).

11. **A new technology must earn its place against one we already operate.** This is a burden of
    proof, not a ban: where something we already run does the job, use it; where it genuinely does
    not, adopt the new thing deliberately and write down what it bought. The reason for the asymmetry
    is that every additional service is one more thing to deploy, monitor, secure, upgrade, document
    and — if the service is one day handed over to NGED — teach to a new operator: a cost that is
    paid forever and is easy to overlook at the moment of adoption.
    *Decided:* delivery to NGED reuses the Delta-on-S3 stack we already operate rather than adding a
    REST API — and the REST API is not rejected forever, it has a documented set of conditions under
    which it would earn its keep. We *did* adopt Delta Lake, Dagster, MLflow, Marimo and Sentry, each
    for a stated reason recorded at the time. *Serves:*
    [H4](../engineering-hypotheses.md#h4-it-runs-for-pocket-money),
    [H5](../engineering-hypotheses.md#h5-operable-by-a-non-expert). *Detail:*
    [An established industry pattern](forecast-delivery.md#an-established-industry-pattern),
    [When would a REST API earn its keep?](forecast-delivery.md#when-would-a-rest-api-earn-its-keep),
    [Considered but rejected designs](production-deployment.md#considered-but-rejected-designs).

**Deliberately absent.** We have **no availability service-level objective (SLO) and no error
budget** — the pair of SRE practices that would normally quantify "how much downtime is acceptable"
and then spend that allowance deliberately. We have consciously declined both: the requirement is
recovery "next business day, via runbook", and the reasoning is in
[Uptime: lenient by design](../background/requirements.md#uptime-lenient-by-design). Recording the
rejection is the point — a practice we considered and declined is more useful to a reader than one we
never considered.

**How to use this list.** A proposed change should be checkable against these eleven: if it violates
one, that is not a veto, but it does require saying which principle is being traded away and what is
bought in return. And a principle that stops deciding anything should be deleted rather than left as
decoration — the same discipline the hypotheses page applies to its thresholds.

For the finer-grained rules that sit underneath these — how to write the code rather than how to
shape the system — see [Code Style](code-style.md) and [Testing](testing.md).

## Core Components

* **Environment & Modularity**: `uv` workspace (Monorepo). Python 3.14. Individual components must be pip-installable with expressive type hints.
* **Data Processing**: **Polars**. Chosen for extreme speed and its native `join_asof` functionality to guarantee no future-data leakage during feature engineering.
    * **Centralized Data Preparation**: All data entering ML models passes through a centralized preparation step to enforce strict data contracts, handle missing entities, and ensure consistency between training and inference.
* **Storage**: **Delta Lake** on cloud object storage for both power data and NWP weather data. Delta Lake provides ACID transactions, time-travel, and efficient partitioning. The same technology is also how forecasts are [delivered to NGED](forecast-delivery.md). Each table's physical layout is owned by the `delta_store` package (as `contracts` owns its logical shape), and both big tables use the same precision lever: values are rounded to a 13-bit significand in pure Polars arithmetic — Veltkamp splitting, rigorously documented on `delta_store.precision.round_to_significand_bits` (max relative error 2⁻¹³ ≈ 1.2×10⁻⁴) — which zeroes the low mantissa bits that are otherwise incompressible entropy. Beyond that shared lever, the two tables need *different* writer properties (measured, not assumed): **NWP** (`delta_store.nwp`) stores physical-unit `Float32` with plain ZSTD-3 and parquet's default dictionary+RLE encodings — `BYTE_STREAM_SPLIT` measures *worse* here because rounding collapses many cells/members onto repeated values — sorted member-early so single-member reads can skip row groups (~40–41 GB per year for the full ECMWF ENS dataset for Great Britain; a single day's NWP data takes about one minute to download and convert). The **`power_forecasts`** table (`delta_store.power_forecasts`) sorts ensemble members of the same target adjacent, uses `DELTA_BINARY_PACKED` timestamps and `BYTE_STREAM_SPLIT` for `power_fcst` (near-continuous ML output has no repeats for a dictionary to exploit), shrinking the 403.6M-row development table from 6.33 GB to 0.73 GB (8.7×) — details in [PR #268](https://github.com/openclimatefix/nged-substation-forecast/pull/268); the NWP format's measured numbers are in [PR #271](https://github.com/openclimatefix/nged-substation-forecast/pull/271).
* **Orchestration**: **Dagster**. Manages the pipeline via Software-Defined Assets (SDAs). Partitioned by NWP init time, not substation, allowing models to train globally across all substations (if they want to). Dagster is responsible for detecting bad data.
* **Configuration Management**: **Hydra** combined with `pydantic-settings`. Dagster passes string overrides (e.g., model=xgboost_global) to trigger Hydra's Config Groups, swapping massive architectural parameter trees. The resolved YAML is logged to MLflow.
* **Experiment Tracking**: **MLflow**.
* **Visualisation**: Altair for plotting, Marimo for interactive data exploration and web apps.

## Lazy Evaluation Strategy

**Rule: everything stays lazy until the model boundary.**

The entire feature-engineering pipeline operates on Polars `LazyFrame`s. No code between reading from storage and calling `BaseForecaster.train()` or `predict()` is allowed to call `.collect()`. The pipeline is:

```text
Nwp.scan_delta(path)                     # lazy scan, already Float32 physical units
  → FeatureEngineer.engineer()           # spatial join + feature pipeline;
                                         #   returns pt.LazyFrame[AllFeatures]; no collect
  → BaseForecaster.train/predict()       # subclass materialises the data here, at the boundary
```

**Why it matters:**

* **Memory**: Polars only materialises the data at the last moment, at the model boundary. Intermediate representations (individual join outputs, lag columns before filtering) are never written to RAM.
* **Query optimisation**: Polars can see the full plan end-to-end and push down filters (e.g. date ranges, `time_series_id` subsets) into the Delta Lake scan before any data crosses the wire.
* **Clean boundaries**: The materialisation site is the model boundary — the one place where a third-party library (XGBoost, PyTorch, …) needs a concrete in-memory array. Keeping it there makes the boundary explicit and easy to find.

**The one exception** is a `limit(1).collect()` guard in `_build_historical_weather` (inside `ml_core`) that cheaply checks for the NWP control member before building the lazy plan, so the pipeline fails loudly instead of silently returning an empty frame.

**Contract for `BaseForecaster` subclasses**: the lazy plan is materialised *at the model boundary*, as late as possible, and never by callers — never ask a caller to collect before passing data in. A subclass typically does a single `.collect()`; keeping that bounded is the **caller's** job, by pruning the *inputs* before feature engineering (not by filtering the engineered output — see below).

### Bounding feature-engineering memory: prune the inputs, not the output

The feature-engineering plan is dominated by the **NWP scan**. The NWP Delta is large — for the V1 trial it is **~86 GB**: 810 daily `init_time` partitions, each up to ~7.24M rows = **1671 H3 cells × 51 ensemble members × 85 native steps** (control member alone is 142k rows/partition). The 30-min upsample and the multi-run bulk join inflate that further. So the whole memory question is: *how little of that NWP do we touch?*

**You cannot prune the scan by filtering the engineered output.** `data.filter(time_series_id == x)` runs the cell-attach join, the 30-min upsample (`group_by` + `explode` + `sort` + `interpolate`) and the bulk join *first*, then drops rows. And NWP is keyed by `h3_index`, not `time_series_id`, so a `time_series_id` predicate can never reach the NWP scan at all. Pruning must be applied to the **raw inputs**, in `_load_engineering_inputs`, directly on the `Nwp.scan_delta` scan.

What actually prunes the NWP scan — verified with `LazyFrame.explain()`:

| Predicate on the **raw NWP scan** | Effect |
|---|---|
| `init_time ∈ [start − 16d, end]` | **Partition prune** — NWP is partitioned by `init_time`, so only those partition directories are opened. A `valid_time` filter *alone* does **not** prune partitions (it scanned ~887 files). |
| `ensemble_member ∈ {…}` on the raw scan | Only the requested members are decoded (≈50× less for control-member training) — and because rows are sorted member-early (`delta_store.nwp.NWP_SORT_COLS`), row-group min/max stats let the reader skip most of each partition outright for a single-member predicate. |
| `h3_index ∈ {cells}` | Restricts to the cells the requested series sit in. |
| `time_series_id == x` on the **output** | Prunes the power scan + metadata join, but **not** the NWP scan (no `time_series_id` there), and only after the upsample. Doesn't help. |
| `slice(n)` / `head(n)` (row count) on the output | **Nothing** — a row slice can't push through a join; the whole join runs first. |

**Pruning the partitions is necessary but not sufficient — you must also stream the collect.** `init_time` prunes whole partition *files*, but `ensemble_member` and `h3_index` are not partition keys, so they act *within* each file. The `ensemble_member` predicate is cheap because of the on-disk sort: rows are ordered `init → member → valid → h3` (`delta_store.nwp.NWP_SORT_COLS`), so each ~1M-row row group spans only a handful of members and a single-member read skips most row groups via min/max stats — measured on a real 29-day, 9-cell, control-member collect at **~5× faster and ~5× less peak memory** than the previous `valid_time`-first sort. `h3_index` is *not* a sort-early column, though: cell filtering is still decode-then-filter within the surviving row groups, so an in-memory collect would still materialise every surviving row before dropping 1662/1671 of the cells. The **streaming engine** (`collect(engine="streaming")`) applies the predicates per morsel, so peak memory is set by the morsel/row-group size, *not* the window length: measured pre-migration, a *full 15-month* control-member collect stayed at ~2.5 GB (the member-early sort only lowers this). `XGBoostForecaster.train`/`predict` therefore collect with `engine="streaming"`.

**Many-to-one `h3_index` ↔ `time_series_id`:** one NWP cell covers several series (the 32 V1 series live in just **9 cells**, one holding 12), so `h3_index` pruning is keyed on the (few) cells the requested series occupy, and the feature engineer's spatial join replicates each cell's weather across its series.

**Resulting design** (`_load_engineering_inputs` applies all three NWP predicates — `init_time`, `ensemble_member`, and `h3_index` = the requested series' cells — to the raw scan, and every collect streams):

| | control member (train) | full 51-member ensemble (predict) |
|---|---|---|
| All eligible series at once | ~5 GB → `train` collects once, groups in memory | ~25 GB ✘ (OOMs) |
| One `init_time` chunk at a time | — | ~9 GB → `cv_power_forecasts` chunks by `init_time`, appends to Delta |

Prediction is bounded by chunking on **`init_time`** (`_PREDICT_INIT_CHUNK`, 14 days), *not* by cell. `init_time` is both the partition key and the axis that fans the output out across runs, so a chunk's forecast frame stays small while each partition is read exactly once and all series/cells/members are processed together (a per-*cell* loop instead OOMs on the busiest cell — 10 series × 51 members × the 10-month window ≈ 116M rows ≈ 25 GB). Measured end-to-end on the `mid_2025_to_mid_2026` fold: training peaks ~5 GB and the full 51-member validation prediction (~321M forecast rows) peaks ~9 GB — both well under a 24 GB laptop.

### The other hard ceiling: Polars' 32-bit row index

RAM is not the only bound a Polars query must respect. Default Polars builds use a 32-bit row
index (`IdxSize`), so any single materialised frame, row count, or row index is capped at
**2³² ≈ 4.29 billion rows** — and crossing the cap raises **no error**. Row counts wrap modulo
2³²: `pl.scan_delta` over the ~5.9-billion-row NWP dev table reports
1,652,180,189 rows via `.select(pl.len())` (= 5,947,147,485 mod 2³², exactly), and
`group_by(...).agg(pl.len())` wraps identically for any single group past the cap, streaming
engine included. This was investigated and empirically pinned down in
[issue #293](https://github.com/openclimatefix/nged-substation-forecast/issues/293): the
wraparound reproduces on plain Parquet scans (it is not a Delta Lake bug), and the
`polars-u64-idx` build (64-bit index, lags mainline releases) returns correct counts on the same
queries.

What is and isn't affected — each verified empirically on 5-billion-row scans:

| Operation on a >2³²-row scan | Outcome |
|---|---|
| `pl.len()` / `group_by(...).agg(pl.len())` where a count exceeds 2³² | **Silently wraps mod 2³²** |
| Materialising a single frame of ≥2³² rows | Unsupported (in practice RAM is exhausted first, which at least fails loudly) |
| Filtered / partition-pruned query whose *result* is < 2³² rows | Correct |
| Value aggregations (`sum`, `min`/`max`, quantiles) over > 2³² rows | Correct — values aren't indices |

The rules that follow:

* **Never row-count a table that can exceed 2³² rows with Polars.** Whole-table counts and
  data-completeness checks must come from the Delta transaction log — `DeltaTable(path).count()`,
  or summing `num_records` over `get_add_actions(flatten=True)` — which is metadata-only, exact,
  and reads no data files.
* The [lazy evaluation strategy](#lazy-evaluation-strategy) above already keeps every production
  collect far below the cap (input pruning + `init_time` chunking), so pipeline reads are
  unaffected. The cap is a second, independent reason that rule exists: an unbounded collect of a
  V2-scale table wouldn't just OOM, its row accounting would be wrong.
* Tables past the cap today: **NWP** (~5.9B rows). **`power_forecasts`** will pass it at V2 scale
  (~1 trillion rows — see [forecast delivery](forecast-delivery.md#how-big-is-flexpectations-power-forecast-data)).
  The one code path that must be re-chunked before then is the `metrics` asset, which currently
  collects a whole `(experiment_name, fold_id)` group of `power_forecasts` at once and discovers
  groups via a full-table `unique()` — fine at V1 (≤ ~414M rows/fold), but a V2 fold is tens of
  billions of rows, so scoring will need to chunk within a fold (e.g. by `valid_time` window or
  `time_series_id` batch) for RAM reasons anyway; the index cap makes it correctness-critical too.

## The Universal Model Interface

All forecasting models subclass `BaseForecaster` (defined in `ml_core`), which provides a common `train` / `predict` / `save` / `load` interface. The model wrapper encapsulates the model weights and all translation logic, keeping Dagster assets completely agnostic to the underlying implementation. Each subclass of `BaseForecaster` is responsible for defining:

* _Feature engineering_: Each subclass carries a `feature_engineer: ClassVar[FeatureEngineer]` strategy (composition, not inheritance) that owns the full preparation pipeline — from raw inputs (observed power, gridded NWP, time-series metadata) to an `AllFeatures` frame. The default `TabularFeatureEngineer` does the nearest-cell NWP spatial join then runs the tabular feature pipeline. A future model that needs a different data view (e.g. a CNN wanting a spatial NWP crop per time series) overrides `feature_engineer` with a different `FeatureEngineer` subclass without touching `BaseForecaster` or any other model. `FeatureEngineer` and `TabularFeatureEngineer` live in `packages/ml_core/src/ml_core/features/`.
* _Input translation_: Transforms the canonical `AllFeatures` Polars LazyFrame into the required model shape.
* _Output translation_: Converts native model outputs into the strict `PowerForecast` schema.
* _Persistence_: Each subclass owns its own save/load format. `XGBoostForecaster` writes one `.ubj` file per `time_series_id` plus a `meta.json` containing the full serialised `XGBoostConfig`. (This may change later. We may switch to saving models using native MLflow flavors (e.g., `mlflow.xgboost.log_model`), which serialize the raw model object directly.)
* _Identity_: Model name, version, and optional MLflow experiment ID travel with the config, so every `PowerForecast` row is self-describing.
