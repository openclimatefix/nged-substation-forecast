# Evaluation metrics

> **Status: durable metric reference.** This page defines every metric computed by
> `compute_metrics` (`packages/ml_core/src/ml_core/metrics.py`): a plain-language summary of
> each, the equation, how to interpret the number, and — where the definition involved a real
> choice — why we chose what we chose. The theory of *why* our ensembles need probabilistic
> evaluation at all is the companion explainer
> [Probabilistic forecasting from NWP ensembles](probabilistic-forecasting.md); the plan that
> delivered these metrics (and the calibration work that builds on them) is
> [Metrics & leaderboard](../roadmap/metrics-and-leaderboard.md#delivering-the-probabilistic-metrics).

## How scoring works, in brief

Every forecast in this project is an **ensemble**: for each substation (`time_series_id`) and
each target half-hour (`valid_time`), one forecast *run* (`power_fcst_init_time`) produces $m$
member forecasts $x_1, \dots, x_m$ (51 members for the NWP-driven ML models; a deterministic
baseline is simply $m = 1$). Scoring proceeds in two steps:

1. **Per timestamp** — each forecast run's members are collapsed into per-timestamp quantities:
   the ensemble mean $\bar{x}_t$ (which the deterministic metrics score), plus the member-aware
   quantities (CRPS, ensemble variance, and the empirical quantiles). Runs that cover the same
   `valid_time` at different lead times are scored **independently**, exactly as a production
   consumer would experience each of them — members are never pooled across runs into a
   lagged-ensemble blend.
2. **Per group** — the per-timestamp values are averaged within each
   `(time_series_id, fold_id, power_fcst_model_name, horizon_slice)` group, where
   `horizon_slice` buckets rows by lead time (`intraday` < 6 h ≤ `day_ahead` < 36 h ≤
   `short_medium_range` < 168 h ≤ `extended_range`), plus an `"all"` group over every lead
   time. Dispersion errors are strongly horizon-dependent, so the per-slice view is usually
   the one to read.

Results land in two places: the `forecast_metrics` Delta table holds the **full** detail (one
row per `(time_series_id, fold_id, horizon_slice, metric_name, metric_param)`), and MLflow
receives per-experiment aggregates under a
[restricted headline subset](#where-the-numbers-live-delta-vs-mlflow).

Notation below: $y_t$ is the observed power at valid time $t$; $x_{t,1}, \dots, x_{t,m}$ are
the ensemble members of the run being scored; $\bar{x}_t$ is their mean; $T$ is the number of
timestamps in the group; $\hat{q}_t(\tau)$ is the empirical quantile of the members at level
$\tau$ (linear interpolation between order statistics).

## Deterministic metrics

These score the **ensemble mean** — a single number per timestamp — and say nothing about
whether the forecast's claimed uncertainty was honest.

### Mean absolute error (MAE)

**MW; smaller is better.** the typical size of the miss. An MAE of 3 MW means the forecast is off by
about 3 MW on an average half-hour, in one direction or the other.

$$
\mathrm{MAE} = \frac{1}{T} \sum_t \left| \bar{x}_t - y_t \right|
$$

Every MW of error costs the same, so MAE is not distorted by a few bad hours — but for the same
reason it cannot tell you whether errors are steady or spiky (that is RMSE's job), nor whether
they lean high or low (MBE's job).

### Normalised MAE (NMAE)

**Dimensionless; smaller is better.** MAE expressed as a fraction of the substation's size, so a 200 MW BSP and a
5 MW primary can be compared on one axis. NMAE ≈ 0.03 means typical errors are about 3% of the
series' effective capacity. This is the **headline cross-series metric**.

$$
\mathrm{NMAE} = \frac{\mathrm{MAE}}{\text{effective capacity}}
$$

**Why a capacity denominator, not the mean:** intermittent generators (PV, wind) spend much of
their time near zero output, so normalising by *mean* power would inflate their NMAE relative
to a demand substation of similar peak size. The denominator is the series' full-history
effective capacity (P99 of |power|), computed over the full history so that it stays stable
across CV folds — see
[Normalising NMAE by effective_capacity](../roadmap/metrics-and-leaderboard.md#normalising-nmae-by-effective_capacity).

### Root mean squared error (RMSE)

**MW; smaller is better.** like MAE, but big misses count disproportionately — one 100 MW error hurts
more than two 50 MW errors. If RMSE is much larger than MAE, the model's errors are spiky: it
is usually fine and occasionally very wrong.

$$
\mathrm{RMSE} = \sqrt{ \frac{1}{T} \sum_t \left( \bar{x}_t - y_t \right)^2 }
$$

### Mean bias error (MBE)

**MW; zero is best.** does the forecast lean high or low? Positive MBE means systematic
over-prediction. A model can have MBE ≈ 0 and still be terrible (large errors that cancel), so
MBE is a diagnosis of *direction*, never of *size*.

$$
\mathrm{MBE} = \frac{1}{T} \sum_t \left( \bar{x}_t - y_t \right)
$$

## Probabilistic metrics

These score the ensemble **before** the mean collapse, and are the reason we pay 51× inference
cost. A recurring caveat: of everything below, **only CRPS is fair across ensemble sizes** —
PICP, pinball loss, interval width, and (residually) the spread-skill ratio all shift with the
member count $m$, because empirical quantiles from few members are systematically too narrow.
When comparing models with different ensemble sizes (e.g. the 51-member ML models against
`nged_incumbent`'s 13 historical analogues), lean on CRPS.

### CRPS (continuous ranked probability score)

**MW; smaller is better.** MAE for a whole probability forecast. CRPS asks "how far was the forecast
*distribution* from what happened?" — it rewards being right, and being *confidently* right,
in one number. For a deterministic forecast ($m = 1$) CRPS literally equals MAE, so CRPS values
sit on the same MW scale as MAE and the two are directly comparable: an ensemble whose CRPS
beats its own MAE is extracting real value from its spread.

We compute the **fair** (finite-ensemble-unbiased) form of Ferro (2014), per timestamp:

$$
\mathrm{CRPS}_t = \frac{1}{m} \sum_{i} \left| x_{t,i} - y_t \right|
\;-\; \frac{1}{m(m-1)} \sum_{i<j} \left| x_{t,i} - x_{t,j} \right|
$$

with the second (spread) term defined as 0 when $m = 1$, and the group value being the mean of
$\mathrm{CRPS}_t$ over the group's timestamps. The first term is accuracy (how far members sit
from reality); the subtracted term is a reward for spread — hedging across the members' honest
disagreement is scored better than betting everything on their mean.

**Why the fair form:** the textbook ensemble estimator divides the spread term by $m^2$ rather
than $m(m-1)$, which under-credits small ensembles — a *perfectly calibrated* 2-member ensemble
scores ~50% worse than its true CRPS, shrinking to ~3% at 13 members (we verified this by
Monte Carlo). Since we will compare ensembles of different sizes, we use the unbiased divisor;
the $m = 1$ convention then makes "CRPS = MAE for deterministic forecasts" exact.

**How it is computed:** the naive pairwise sum is $O(m^2)$ per timestamp and, at fold scale,
would require a members self-join materialising billions of pair rows. We instead use the
sorted-member identity
$\sum_{i<j} |x_i - x_j| = \sum_{k=1}^{m} (2k - m - 1)\, x_{(k)}$ (for ascending order
statistics $x_{(k)}$), which is a single $O(m \log m)$ aggregation expression. The sum is
accumulated in Float64: its terms are as large as $\pm m \cdot |x|$ while the result is only
spread-sized, and that cancellation loses percent-level accuracy in Float32 when members are
near-identical (precisely the underdispersed-intraday case we most care about).

### Spread-skill ratio

**Dimensionless; 1.0 is best.** is the fan the right width? The ratio compares how uncertain the forecast
*claims* to be (the spread of its members) against how wrong it *actually* is (the RMSE of its
mean). A ratio of 1.0 means the claimed uncertainty matches the realised error — the fan can be
trusted. Below 1 the model is overconfident (**underdispersed**: the fan is too thin, and
reality keeps landing outside it); above 1 it is underconfident. This is the headline
diagnostic for the
[underdispersion this project expects](probabilistic-forecasting.md#where-forecast-uncertainty-actually-comes-from).

$$
\mathrm{SSR} = \frac{ \sqrt{ \dfrac{1}{T} \sum_t \dfrac{m+1}{m}\, s_t^2 } }{ \mathrm{RMSE} },
\qquad
s_t^2 = \frac{1}{m-1} \sum_i \left( x_{t,i} - \bar{x}_t \right)^2
$$

Two definitional choices matter here, both made so that "1.0 = well-calibrated" is *literally
true* rather than approximately true:

- **RMS spread, not mean-of-stddev.** Averaging per-timestamp standard deviations and dividing
  by RMSE looks natural but is biased low by Jensen's inequality whenever spread varies across
  timestamps — and power-forecast spread varies strongly (diurnally and synoptically). In a
  simulation of a *perfectly calibrated* heteroscedastic 51-member ensemble, the mean-of-stddev
  form read 0.80 while the RMS form read 0.99. We therefore aggregate variances and take one
  square root ("RMS spread"), so a calibrated model cannot be misdiagnosed as underdispersed.
- **The Fortin $(m+1)/m$ factor.** Even with RMS spread, a calibrated $m$-member ensemble
  satisfies $\mathrm{RMSE}^2 = \frac{m+1}{m} \times$ (mean ensemble variance) — the truth
  behaves like one extra draw from the same distribution (Fortin et al. 2014). Folding
  $(m+1)/m$ into the numerator makes the calibrated target exactly 1.0 at *any* ensemble size,
  instead of ~0.99 at 51 members and ~0.96 at 13 — where the uncorrected form would read as
  spurious underdispersion.

The variance uses `ddof=1` (the sample variance), and a single-member forecast scores spread 0
— zero spread is the honest description of a deterministic forecast, and 0/RMSE keeps the
ratio's meaning (maximally overconfident) rather than becoming null.

### Pinball loss

**MW; smaller is better.** the score for a single quantile forecast, with a deliberately lopsided
penalty that matches what the quantile *claims*. A p90 forecast claims "only a 10% chance power
exceeds this", so when power does exceed it the loss is steep (weight 0.9), and when power
falls below it the loss is shallow (weight 0.1). A forecaster minimises expected pinball loss
at level $\tau$ by reporting the true $\tau$-quantile — bluffing wide or narrow both cost — so
pinball loss is the honest scorecard for each individual quantile.

For the empirical member quantile $\hat{q}_t(\tau)$:

$$
L_\tau
= \frac{1}{T} \sum_t
\begin{cases}
\tau \, \left( y_t - \hat{q}_t(\tau) \right) & \text{if } y_t \ge \hat{q}_t(\tau) \\[2pt]
(1 - \tau) \, \left( \hat{q}_t(\tau) - y_t \right) & \text{otherwise.}
\end{cases}
$$

**Which quantiles — the NGED delivery thirteen.** Pinball loss is computed at exactly the
levels agreed with NGED for the [delivery tables](../roadmap/delivery-tables.md) (p1, p2, p5,
p10, p20, p35, p50, p65, p80, p90, p95, p98, p99; the single source of truth is
`contracts.common.DELIVERY_QUANTILES`), one `metric_param` row per level. Three reasons:
evaluation should measure what we sell; the tails *are* the product (NGED is far more
interested in the tails than the shoulders); and the
[Phase D quantile pipeline](../roadmap/metrics-and-leaderboard.md#phase-d-ensemble-of-quantile-forecasts-representation-3-pooled-representation-2)
will be scored head-to-head at these same levels, so today's numbers are directly the baseline
it must beat.

**Finite-ensemble tail caveat:** from 51 members, $\hat{q}(0.01)$ interpolates between the two
lowest members — the raw ensemble simply cannot represent its own far tails well, so its tail
pinball losses are partly a representational limit rather than pure skill. That is not a reason
to skip them (the product needs tails scored); it is the quantitative motivation for Phase D's
per-member quantile models.

### Mean pinball loss

**MW; smaller is better.** the thirteen pinball losses averaged into one quantile-skill scalar, for
leaderboard ranking. Because six of the thirteen delivery levels sit at or beyond p10/p90, this
average is deliberately **tail-weighted** — a model that nails the median but botches the tails
scores poorly, which matches NGED's priorities.

$$
\overline{L} = \frac{1}{13} \sum_{\tau \in Q} L_\tau ,
\qquad Q = \{0.01, 0.02, 0.05, 0.10, 0.20, 0.35, 0.50, 0.65, 0.80, 0.90, 0.95, 0.98, 0.99\}
$$

### PICP (prediction interval coverage probability)

**Dimensionless; aim for the calibrated reference below.** how often reality actually lands inside the claimed band. For the p10–p90
band, roughly 80% of observations should fall inside; markedly less means the band is too
narrow (overconfident), markedly more means too wide. PICP is scored for every symmetric pair
of delivery quantiles — six bands from p35–p65 (30%) out to p1–p99 (98%) — so together the six
values trace a **coverage curve** across the whole distribution.

$$
\mathrm{PICP}_{[\tau_{\mathrm{lo}}, \tau_{\mathrm{hi}}]}
= \frac{1}{T} \sum_t
\mathbf{1}\!\left[ \hat{q}_t(\tau_{\mathrm{lo}}) \le y_t \le \hat{q}_t(\tau_{\mathrm{hi}}) \right]
$$

**The calibrated reference is *below* nominal coverage.** Empirical quantiles from a finite
ensemble sit inside the true ones, so even a perfectly calibrated $m$-member ensemble covers
less than the nominal rate — approximately
$\frac{m-1}{m+1}(\tau_{\mathrm{hi}} - \tau_{\mathrm{lo}})$, which we confirmed by simulation.
Judge PICP against these references, not the nominal rates:

| Band | Nominal coverage | Calibrated reference at $m = 51$ |
|---|---|---|
| p1–p99 | 0.98 | ≈ 0.942 |
| p2–p98 | 0.96 | ≈ 0.923 |
| p5–p95 | 0.90 | ≈ 0.865 |
| p10–p90 | 0.80 | ≈ 0.769 |
| p20–p80 | 0.60 | ≈ 0.577 |
| p35–p65 | 0.30 | ≈ 0.288 |

A p10–p90 PICP of 0.72 at 51 members is therefore mild overconfidence (reference 0.769), not
the 8-point shortfall a naive comparison with 0.8 would suggest. The gap grows as $m$ shrinks —
one more reason PICP is not comparable across ensemble sizes.

PICP alone is also **gameable**: an absurdly wide band hits any coverage target for free. It
must always be read alongside a sharpness measure — which is what interval width is for (and
the proper scores, CRPS and pinball, punish over-widening automatically).

### Interval width

**MW; narrower is better *given* coverage.** how wide the claimed band is, on average — the sharpness companion to PICP.
On its own, narrower is not better (a zero-width band is maximally sharp and maximally
useless); the pair to optimise is *coverage at the reference, at the smallest width*. Reading
PICP and interval width together makes the coverage-vs-sharpness trade-off legible — in
particular, a calibration step that "fixes" coverage by inflating spread will visibly pay for
it here.

$$
\mathrm{Width}_{[\tau_{\mathrm{lo}}, \tau_{\mathrm{hi}}]}
= \frac{1}{T} \sum_t \left( \hat{q}_t(\tau_{\mathrm{hi}}) - \hat{q}_t(\tau_{\mathrm{lo}}) \right)
$$

It is computed for the same six bands as PICP.

## Where the numbers live: Delta vs MLflow

The `forecast_metrics` Delta table always holds the complete detail: all thirteen pinball
levels, all six bands, every horizon slice, per series. MLflow — the leaderboard — receives
per-experiment aggregates (unweighted means across series) under keys built from a token that
is `{metric_name}` for scalar metrics and `{metric_name}_{metric_param}` for parametric ones,
in three families: `{token}__all` (overall), `{token}__{type_slug}` (per
`time_series_type`), and `{token}__all__{horizon_slice}` (per lead-time band).

To keep the leaderboard legible (~130 keys rather than ~340), parametric metrics are
restricted in MLflow to a **headline subset**: `pinball_loss` at p10/p50/p90 and `picp` /
`interval_width` at p10_p90 (the allowlist is `_MLFLOW_LOGGED_PARAMETRIC` in
`ml_core.metrics`). Anything not in MLflow is still one Polars filter away in Delta.

## What is deliberately *not* here

- **Rank (Talagrand) histograms** — the most direct picture of *how* an ensemble is
  miscalibrated (U-shaped = underdispersed, domed = overdispersed). A histogram is not a scalar,
  so it does not fit the tall metrics schema; it is computed in ad-hoc analyses (e.g. when
  choosing the
  [Phase C calibration](../roadmap/metrics-and-leaderboard.md#phase-c-cheap-calibration-after-b-proves-the-diagnosis)
  approach) rather than stored per experiment.
- **Histogram of errors** — same reason; planned as a visual check on the
  [roadmap](../roadmap/metrics-and-leaderboard.md#evaluation-metrics).
