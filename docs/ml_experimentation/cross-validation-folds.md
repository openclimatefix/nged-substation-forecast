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
`ml_core._cv_helpers` is generic over arbitrary `train_start/train_end/val_start/val_end` dates, so
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
`conf/cv/default.yaml` edit with no schema change.

Separately, and later, we plan to **pre-train** on weather reanalysis (CERRA) so models can use the
long power histories that some assets have back to 2020, then fine-tune on ECMWF ENS. Pre-training
is a training-time technique, distinct from the validation folds described here.

---

## Alternatives considered

We weighed three other ways to slice the limited honest data — monthly expanding CV, quarterly
non-overlapping walk-forward, and yearly folds backed by CERRA reanalysis — before settling on the
single fold above. The reasoning for rejecting or deferring each is recorded in
[ML Experiment Orchestration — Design Decisions](../architecture/ml-orchestration.md#fold-design-alternatives-considered).
