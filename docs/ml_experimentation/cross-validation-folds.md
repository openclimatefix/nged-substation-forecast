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

We considered three other ways to slice the limited honest data and chose the single fold above.
These are recorded so the decision is auditable and so we can revisit them as more data lands.

### Monthly expanding CV — rejected (redundant folds)

Slide a 12-month validation window forward by one month per fold, expanding training by one month
each time (fold 1 validates 2025-04→2026-03, fold 2 validates 2025-05→2026-04, …). This "buys"
several folds from today's data, but consecutive folds share **11/12 of the validation window** and
>90% of the training data, so their metrics are correlated ~0.9+. The effective number of
independent folds is barely more than one; the small spread across them **understates** true
sampling variance (false confidence in stability), at N× the compute for almost-duplicate
information. One month is too small a change to make the folds meaningfully different.

### Quarterly non-overlapping walk-forward — deferred (the sound multi-fold option)

Expanding training, but validate on the **next 3 months, non-overlapping**:

| Fold | Train | Validate (3 mo, non-overlapping) |
|---|---|---|
| 2025-Q2 | 2024-04-01 → 2025-03-31 | 2025-04 → 2025-06 |
| 2025-Q3 | 2024-04-01 → 2025-06-30 | 2025-07 → 2025-09 |
| 2025-Q4 | 2024-04-01 → 2025-09-30 | 2025-10 → 2025-12 |
| 2026-Q1 | 2024-04-01 → 2025-12-31 | 2026-01 → 2026-03 |

Because the validation windows do not overlap, the folds are **genuinely independent**
measurements, and the set covers all four seasons (so you see seasonal skill variation, then report
per-season and the mean). This is the statistically sound version of what monthly CV reaches for.
We deferred it to keep the initial CV setup minimal; it is the recommended next step if we want
multiple folds
*before* the ECMWF back-fill enables the full yearly protocol.

### Yearly folds backed by CERRA — rejected for validation

Keep the yearly 2022–2025 folds now by training on **CERRA reanalysis** for the pre-2024-04-01
years. Rejected: CERRA is *reanalysis* (it ingests future observations), so validating on it
measures the model's response to near-perfect weather, not **forecast** skill — systematically
misleading — and a leaderboard mixing CERRA folds and ECMWF folds is apples-to-oranges. It would
also pull a large CERRA-ingestion effort forward. CERRA is valuable, but for **pre-training**
(above), not for validation.
