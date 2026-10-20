---
id: "00122"
slug: differentiable-physics-three-winter-backtest-gam-k
part: n/a
theme: differentiable-physics-demand-calendar-and-weather
also_relevant_to:
  - differentiable-physics-forecast-engine
  - differentiable-physics-long-history-and-drift
status: planned
size: >-
  simple. Stored: private files under
  data/ml-experiments/00122-differentiable-physics-three-winter-backtest-gam-k/ only (a scoring
  table); no fit, no new prediction set. Serving path: not touched. Degradation rule: not touched.
  Designs: one defensible design, because the arms, cuts and scoring are those of experiments 107,
  109 and 114, and the maintainer fixed the contrasts. Callers: this folder's scripts only; no
  change to packages/.
hypothesis: >-
  On past weather, the convex model with the Kalman filter (GAM-K) forecasts Primary demand one day
  ahead better than both the plain convex model (GAM) and lagless XGBoost, in each of the three
  winters 2023-24, 2024-25, and 2025-26, by at least 1 point of mean absolute error.
prediction: see "Predictions"
decision_rule: see "Decision rule"
benchmark: >-
  The plain convex model (B0 of experiment 114) and lagless XGBoost over 2 years of history (x_2y of
  experiment 109), both stored; GAM-K is B0-K of experiment 114
row_set_id: >-
  dp_mw_v1 (experiment 103) at D5, D10, D11, D14, and D16, rows outside the hand-filtered switching
  events; winters November to March of 2023-24, 2024-25, and 2025-26
metric: >-
  Mean absolute error of the median forecast, in points of effective capacity, at the day-ahead lead
  (day D+1 from a 09:00 UTC initialisation on day D, leads 15.5 to 39 hours)
verdict: <supported, partly supported, small gain, refuted, or inconclusive>
production_status: not-for-production
take_home_message: <written after the results>
date: <date>
pr: <pr>
advice: <written after the results>
code: ml-experiments/00122-differentiable-physics-three-winter-backtest-gam-k/
---

# Differentiable physics: GAM-K against the plain GAM and lagless XGBoost over three winters

**Planned, not yet run.** We ask whether the convex differentiable-physics demand model with a
Kalman filter of recent errors (GAM-K, the series' candidate) beats both the same model without the
filter (GAM) and lagless XGBoost in each of three winters, using observed past temperature and
irradiance, not weather forecasts. An earlier read-only analysis of stored forecasts found GAM-K
ahead in 2024-25 and 2025-26 but had only two winters. A third winter, 2023-24, is also stored, in
the first-year files of [experiment
114](../00114-differentiable-physics-n3-n4-interactions-and-heating-threshold/result.md) and
[experiment 109](../00109-differentiable-physics-n0-partially-pooled-convex-model/result.md). We
score the three winters side by side, with one rule for the cold-snap days and one for the evening
peak.

**Past-weather scoring ranks model structure and does not give production skill.** The inputs are
observed temperature and irradiance, so no weather-forecast error enters. A gap measured here is a
ceiling for the gap under live forecasts.

## Size

The five `plan-issue` triggers, each answered:

1. **What gets stored:** one private scoring table; nothing in `packages/` or any contract.
2. **Serving path:** not touched.
3. **Degradation rule:** not touched.
4. **More than one defensible design:** no. The arms, cuts, row set, and scoring follow experiments
   107, 109, and 114, and the maintainer fixed the contrasts.
5. **Code whose callers could not be named without searching:** none. The script reads three stored
   prediction sets and imports `studies.experiment_store` only.

Size: simple, a rescoring of stored forecasts with existing settings. The reviews are one Sonnet code
review (leakage, and each fit stopping at the cut), one Sonnet write-up review, and the prose
review. No Opus review runs at this stage. Audit triggers are listed under "Checks" in the
write-up.

## Terms

- **Points:** percentage points of a substation's effective capacity.
- **GAM, GAM-K:** experiment 114's B0 (24-month window, no drift, PV capacity pinned, ridge 0.1)
  without and with the Kalman filter of recent errors. [Experiment
  111](../00111-differentiable-physics-6-ens-forecast-engine-against-bg2/result.md) chose the
  filtered model as the series' candidate.
- **Lagless XGBoost:** experiment 109's `x_2y`: experiment 87's recipe with the three power lags
  removed, trained on the 2 years before each cut, on the same ERA5 and CAMS inputs.
- **Winter:** 1 November to 31 March, by the local (Europe/London) calendar day of the valid time.
- **Cold-snap days:** in each winter, the coldest 10% of covered days by daily mean ERA5 2 m
  temperature, averaged over the five Primaries (at least 3 days).
- **Evening peak:** valid times from 16:00 to 20:00 local time.

## Design

**No fit runs.** Experiments 109 and 114 already fitted every arm at the cuts 2022-07-01 (the
replica of 2023-24's spread), 2023-07-01, 2024-07-01, and 2025-07-01, one frozen fit per cut and
site, and stored the day-ahead predictions for the year after each cut. Refitting would reproduce
the same numbers, so this experiment reads them. The three winters come from the fits at 2023-07-01,
2024-07-01, and 2025-07-01, each frozen for its year, so no forecast uses data after its cut.

**Five Primaries in every winter.** D5, D10, D11, D14, and D16 only, because D9 has no history at
the 2023-07-01 cut and GAM-K lacks it in 2024-25. The planned contrasts use these five in all three
winters, so the winters are comparable. D9 is added to 2025-26 as a labelled sensitivity.

**Rows.** An inner join of the three arms on series, forecast origin, and valid time; rows inside
any arm's switching mask are dropped, as in experiments 107, 109, and 114. No valid time on or after
2026-07-01 is read.

**Statistic.** For each arm and each winter, the mean absolute error of the median forecast is
averaged over the rows of each local day (all five Primaries pooled), then over the days. The same
daily means, restricted to the evening peak and to the cold-snap days, give the other two columns.

**Interval.** The unit of resampling is a block of 7 consecutive calendar days, paired across arms,
2,000 draws of 95% intervals with a fixed seed. Consecutive days share weather, so a day-level
bootstrap would be too narrow, and a month-level bootstrap leaves only 5 blocks in a winter. A week
spans a typical weather spell. A 14-day block is the sensitivity. The interval covers within-winter
weather and not the difference between years or substations, and there are only three winters.

**How many independent winters.** Three. They are separate weather samples, but consecutive winters'
fits share part of their 24-month training window, and the two later winters have already been seen
in the read-only analysis (GAM-K minus GAM −1.83 and −1.64, GAM-K minus lagless XGBoost −2.00 and
−1.82 on six Primaries). Only 2023-24 is a new test of the hypothesis.

**One bias to state.** GAM-K's filter settings (four numbers per substation) were fitted by marginal
likelihood on the 2023-24 residuals in experiment 114, so 2023-24 is not out of sample for those
four numbers. The bias favours GAM-K in that winter. The 2024-25 and 2025-26 filter settings are the
same numbers, fitted a year or two earlier. The write-up shows GAM-K minus GAM in each winter so a
reader can see whether 2023-24 stands out. Experiment 109's choice of `x_2y` over `x_2yamq` was
also made on the 2023-24 year, which favours XGBoost there if anything.

Experiment 114's fit script is the only place a fit would live. Because nothing is refitted, the
Sonnet code review reads that script and experiment 109's to confirm each fit stops at its cut.

## Planned contrasts

All six are planned: GAM-K minus GAM, and GAM-K minus lagless XGBoost, each in 2023-24, 2024-25, and
2025-26, on all winter days. Negative means GAM-K is better. Everything else (cold-snap days,
evening peaks, D9, the 14-day blocks, the quantile score) is exploratory.

## Decision rule

A contrast **meets the bar** if its estimate is at most −1.0 points and its 95% interval lies wholly
below zero. The 1-point bar is the maintainer's, and it is set by the cost of the filter: GAM-K adds
a pipeline step over GAM and a different model family over XGBoost, so a gain must be large to pay
for it. One point is about a fifth of the winter error (4 to 6 points).

- **Supported:** all six contrasts meet the bar.
- **Partly supported:** 3 to 5 contrasts meet the bar and no contrast has an interval wholly above
  zero.
- **Small gain:** at most 2 contrasts meet the bar, all six intervals lie wholly below zero.
- **Refuted:** any contrast has an interval wholly above zero.
- **Inconclusive:** anything else.

The rule uses 95% intervals without a multiplicity correction, because "supported" needs all six to
pass, which is harder than any one passing.

## Predictions

GAM-K minus GAM is −1.0 to −1.8 points and GAM-K minus lagless XGBoost is −1.0 to −2.0 in 2023-24,
about the size seen in the later winters. The risk is 2023-24: in experiment 107 the best unfiltered
GAM beat lagless XGBoost by only 0.69 points there (two other winters, 1.55 and 0.73), and the filter
must add the rest. We put the chance that all six contrasts meet the bar at about 50%, and the
chance that all six are wholly below zero at about 85%. Cold-snap days: GAM-K ahead in the point
estimate everywhere but with intervals that cross zero in at least one winter, because a winter holds
12 to 15 cold-snap days in a few spells. Evening peaks: GAM-K ahead in every winter, as in the
earlier analysis.

## Checks

To be filled from the printed report: the error of the median forecast beside the quantile-score
contrasts (trigger 3), leave-one-site-out for each contrast (trigger 4), and which audit triggers
fired. Substation cards are not needed: no new fitted properties.

## Reproducing

    uv run python ml-experiments/00122-differentiable-physics-three-winter-backtest-gam-k/e00122_report.py
