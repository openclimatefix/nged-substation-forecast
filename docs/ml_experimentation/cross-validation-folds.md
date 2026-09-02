# Cross-validation folds

How we split data into training and validation windows to score and compare forecasting
experiments on the leaderboard.

> **Status legend:**
>
> - ✅ Implemented
> - 🚧 Planned
> - 🔬 Research

The fold definitions live in `conf/cv/default.yaml` and are read by every experiment, so all
models are scored on the **same** folds (apples-to-apples). The fold windowing in
`ml_core.cv_helpers` is generic over arbitrary `train_start/train_end/val_start/val_end` dates, so
changing the fold set is a config edit, not a code change.

---

## Why expanding-window cross-validation

We use **expanding-window** CV: the training period grows each fold while the validation window
stays a fixed length and lies strictly *after* training. This mimics production (we never train on
the future), and validating across a whole year per fold gives balanced **seasonal** coverage.

We chose expanding over sliding windows to maximise data for data-hungry models (neural nets). The
trade-off is that it confounds "algorithmic improvement" with "more data", so we never compare one
fold against another directly — we aggregate folds into a single leaderboard figure (mean across
folds).

---

## Current state: a single fold ✅

Honest forecast-skill validation needs **real forecast NWP (ECMWF ENS) for both training and
validation**. Our ECMWF ENS archive only reaches back to **2024-04-01** (Dynamical.org are
back-filling earlier years, but slowly), so the entire usable window is ~2024-04 to mid-2026 —
only enough for roughly **one** seasonally-complete fold. We therefore run a single fold:

| `fold_id` | Train | Validate | Weather source |
|---|---|---|---|
| `mid_2025_to_mid_2026` | 2024-04-01 → 2025-06-30 (15 months) | 2025-07-01 → 2026-06-30 (12 months) | ECMWF ENS only |

The training window is stretched to 15 months to use all the honest data available before the
validation window. This code is not expected to train a model until **after 2026-06-30**, by which
point the validation window has closed and validates on complete data.

A single fold gives no across-fold variance estimate, but it is still ample for the leaderboard's
main job: each fold scores ~22–32 time series × 51 ensemble members at half-hourly resolution —
millions of prediction points, more than enough statistical power to **rank** experiments against
each other.

### Eligibility

A time series is eligible for a fold when its observed-power coverage has at least
`min_training_months` (default **6**) of history *before* `val_start` **and** reaches `val_end`.
Eligibility is derived from data coverage alone — never from the model or config — so every
experiment evaluates the fold on the identical population. The eligible set is computed and frozen
per leaderboard epoch by the `eligible_time_series` asset.

---

## Target: multiple yearly folds 🚧

**Once Dynamical.org has backfilled ECMWF ENS to the earlier years**, we will move to the original
target protocol: an expanding training window with one **complete-year** validation fold per year
(2022, 2023, 2024, 2025, …), validated on real forecast NWP throughout. Adding those folds starts
a **new leaderboard epoch** (every experiment is re-scored against the new fold set), and is a
`conf/cv/default.yaml` edit with no schema change. That back-fill is not expected until
**~November 2027** — after v1.0 — and covers 00Z initialisations only
([reformatters#446](https://github.com/dynamical-org/reformatters/issues/446)).

Because of that timescale, the plan for using the long power histories some assets have back to
2020 is to **pre-train** on ERA5 reanalysis and fine-tune on ECMWF ENS. Pre-training is a
training-time technique, distinct from the validation folds described here; the design is in
[Extending the training history](../roadmap/training-history.md).

---

## Evaluating a data source whose history is shorter than the folds

**A new input data source whose archive starts after the canonical folds cannot enter those folds
at all, so its evaluation lives entirely in the `metrics` asset's `evaluation_scope="ad_hoc"` and
never feeds the leaderboard.** The motivating case is adding ICON-EU NWP (from Dynamical.org),
whose archive starts later than the leaderboard folds, leaving no overlapping history to score on.

**For a new *weather* source, check the ceiling first.** Before ingesting one at all, measure
how much forecast skill near-perfect weather would add — see [the perfect-weather
ceiling](../roadmap/metrics-and-leaderboard.md#the-perfect-weather-ceiling-what-it-gates). If a
model trained and scored on reanalysis barely beats the ENS-scored champion, there is little
forecast-error headroom for a further source to recover, and the patterns below are not yet
worth running — unless the candidate's case rests on *resolution*, which that ceiling does not
bound.

Three patterns answer three different questions.

### Controlled ablation — "does the source add skill?"

The principled comparison: hold *everything* constant except the source under test. Because the
new source only exists from (say) 2026, the shared window must live within its availability:

- Pick an evaluation window bounded by the new source's history; split it into train/validation
  within that window.
- **Baseline experiment:** existing features only (e.g. `weather_source = "ecmwf"`).
- **Treatment experiment:** existing + new-source features (e.g. `weather_source = "ecmwf_icon"`).
- Both train on the **identical rows** and are scored on the **identical rows** — same
  `time_series_id` population, same `power_fcst_init_time` grid — differing *only* in the
  feature set. Score both with `evaluation_scope="ad_hoc"` over the same `PopulationFilter`.

To inherit the leaderboard's same-population guarantee for this off-leaderboard window,
materialise a **frozen ad-hoc eligibility set** for the window that both experiments read,
rather than letting each pick its own population. (`trained_time_series_ids` forces
train == predict *per model*, but does not by itself force the *two* experiments to share a
population — the frozen set does.)

### The confounded comparison, which must not be read as the ablation

The tempting shortcut — take the canonical leaderboard champion, run it on the new source's
window, and compare against a new-source model on that window — is **statistically confounded**
and must not be read as evidence about the source. The two models differ in **two** variables
at once: the feature set *and* the training window (the champion trained on the full archive;
the new-source model is forced onto the short sliver). A win or loss cannot be attributed to
the source rather than to the training data.

The confounded comparison is legitimate only as a **deployment** question — "which forecast is
better to ship *today*?" — where the confound is irrelevant because we only care which is better
now, not why.

### The epoch path — the eventual leaderboard-quality answer

Once the new source has accumulated enough history (roughly 1–2 complete years), promote it via
a **new leaderboard epoch**: a fold set over source-era complete years in which the new source
is canonically available, with every experiment re-scored against that fold set for
apples-to-apples comparison. The ad-hoc ablation is the **interim** signal obtained before enough
history exists to do this properly; it should never be presented with leaderboard rigour.

These three patterns concern only *evaluation*. Actually **ingesting** a second NWP source (a
second downloader, NWP contract changes, source-aware weather-feature parsing, a dual-source
join in feature engineering) is separate engineering work — see the
[roadmap](../roadmap/index.md).

---

## Overlapping forecasts: what the fold boundary does and does not protect

A 14-day forecast reissued every 6 hours covers each target half-hour 56 times. It is
therefore worth saying precisely which problems that creates and which it does not.

**It does not contaminate the test set.** `load_engineering_inputs` bounds both power and NWP by
*target* time — power on `time` and NWP on `valid_time`, each within `[window_start, window_end]` —
not by forecast origin. A training row whose `power_fcst_init_time` falls in June 2025 therefore
cannot carry a target in the validation window, because the target itself is filtered out. The fold
boundary is a boundary on observations, so no observation appears on both sides of it. Power lag
features are separately protected by `_nullify_leaky_lags`, which nulls any lag shorter than the
lead time.

**It does inflate the apparent weight of evidence.** Within one horizon slice the same target
half-hour is still scored many times: `extended_range` spans 168 h and beyond. With 6-hourly
initialisations, roughly 28 forecasts therefore land in that band for a single target. Those are
not 28 independent measurements of skill — they share the weather, the recent load and most of the
model state. The leaderboard's per-slice counts therefore overstate how much independent evidence
separates two experiments. A naive significance test on them would report more confidence than the
data supports.

We have not fixed this. The [energy-forecasting literature
review](../background/energy-forecasting-review.md) found no paper offering a method to copy. The
closest is [Browell and Fasiolo (2021)](https://arxiv.org/abs/2103.10335), who build the
consistency intervals on their calibration diagrams "considering the temporal correlation of
net-load, as the usual assumption of independence between samples does not hold" — the right
instinct, applied to day-ahead forecasts that are not reissued. Until we adopt a better method,
the same rule already in force for [fold
hygiene](../roadmap/metrics-and-leaderboard.md#fold-hygiene-selection-bias-and-a-final-test-window):
differences smaller than fold-level noise must not drive decisions, and the per-slice point count
is not a sample size.

---

## Alternatives considered

We weighed three other ways to slice the limited honest data — monthly expanding CV, quarterly
non-overlapping walk-forward, and yearly folds backed by ERA5 reanalysis — before settling on the
single fold above. The reasoning for rejecting or deferring each is recorded in
[ML Experiment Orchestration — Design Decisions](../architecture/ml-orchestration.md#fold-design-alternatives-considered).
