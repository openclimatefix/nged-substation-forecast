---
id: "00104"
slug: differentiable-physics-2-simple-structural-model
part: 2
theme: differentiable-physics-demand-calendar-and-weather
also_relevant_to: [differentiable-physics-long-history-and-drift, retraining-and-training-history, solar-disaggregation]
status: done
size: >-
  complex for the framework module (new tested code in packages/studies, reused by Parts 3 to 6);
  medium for the experiment scripts. Stored: private files under
  data/ml-experiments/00104-differentiable-physics-2-simple-structural-model/ and a feature cache
  beside them; no contract or production table. Serving path: not touched. Degradation rule: not
  touched. Designs: more than one is defensible for the module (loss, spread model, drift form,
  operator interface), and this plan pre-registers one. Callers: the module's callers are this
  part's scripts and the later parts of the series only.
hypothesis: >-
  A convex demand-minus-solar model, trained on every usable year with a slowly drifting level per
  hour band, forecasts Primary demand better than the same model trained on the last 12 months.
prediction: see "Predictions"
decision_rule: see "Decision rule"
benchmark: >-
  Lagless XGBoost on the same ERA5 and CAMS inputs, with and without experiment 64's solar offset
  (am), each with a matching zero-start arm; climatology; manual_heuristic
row_set_id: dp_mw_v1 (experiment 103), six MW long-history sites, scored 2025-07-01 to 2026-06-30
metric: >-
  Quantile score (twice the mean pinball loss over experiment 87's 11 quantiles) in points of
  effective capacity, on rows outside the hand-filtered switching events at the scored site
verdict: inconclusive
production_status: not-for-production
adopted_in: none
date: 2026-10-19
pr: 1349
take_home_message: >-
  A simple model of demand minus solar, fitted as one convex optimisation per substation, forecast
  better when we trained it on every usable year than on the last 12 months only (a lower score is
  better: 0.40 points lower, a gain at 4 of the 5 decision substations). Adding a slowly changing
  demand level ("drift") fitted to the old data halved that gain, so the planned comparison, all
  years with drift against 12 months, is inconclusive: 0.20 points better, with an interval from
  0.68 better to 0.22 worse, and 3 of 5 substations better. The gain depends on one substation (D14)
  and does not show that the model beats XGBoost; against the best lagless XGBoost the all-years
  model without drift is 0.34 points better (post hoc), and XGBoost given recent power readings is
  still better than it.
advice:
  - >-
    Give a convex demand model more years of history before adding a drift term, since all years
    without drift lowered the score by 0.40 points against 12 months (G-all minus G1).
  - >-
    Start each XGBoost quantile from that level's own training quantile, not from a constant, since
    a zero start left the 95% and 99% quantiles covering only 66% and 73% and scored 0.30 points
    worse (X-12z minus X-12).
  - >-
    Do not add a solar offset to XGBoost that already sees CAMS irradiance: it changed the score by
    under 0.03 points, and it helped by 0.28 points only without the irradiance input (X-12am-ng
    minus X-12z-ng).
  - >-
    Fit forecast spread on a replica of the frozen forecast, a fit 12 months before the cut, because
    staleness runs from 0 to 12 months and quarterly fits cover only 0 to 3 (design reason; this
    experiment did not compare the two spreads).
  - >-
    Use the GPU for the pooled model in later parts: the pooled proxy ran 6.6 times faster there
    (33.5 s against 221 s).
code: ml-experiments/00104-differentiable-physics-2-simple-structural-model/
---

# Differentiable physics, part 2: a simple structural model of demand and solar

**A simple model of demand minus solar forecasts better with more years of history, but a drift
term held constant after the end of the training data halves the gain.** This is part 2 of a
six-part series on models built from physical terms (a daily profile, holidays, heating, and
solar output) and fitted by gradient-based optimisation; part 3 pools the model across substations.
Trained on all usable years with no drift, the model scored 0.40 points better than the same model
trained on 12 months (a lower score is better). The planned comparison, all years with drift
against 12 months, gave 0.20 points better with an interval from 0.68 better to 0.22 worse, and 3
of the 5 decision substations improved. The verdict is `inconclusive`, because several
pre-registered positive controls failed. On the comparison alone, the merged rule for the first
scoring pass would give `partly supported`.

The comparison rests on one substation. D14's winter of 2025-26 is under-forecast by every arm, and
without D14 the mean of the other four substations is 0.13 points worse, not better (D14 -1.52,
D10 +0.54, D16 +0.16, D5 -0.14, D11 -0.04). The result narrows the long-history question to "more
years help a convex model without drift, mainly through D14; a drift fitted to the end of the
training data does not help".

We fitted the model, one convex optimisation (a problem with a single best answer) per Primary
substation, on data to 30 June 2025, held it fixed, and scored it on every half-hour from 1 July
2025 to 30 June 2026. The score is twice the mean pinball loss (an error measure for quantile
forecasts) over 11 quantile levels, in points of effective capacity (1% of the substation's
effective capacity, the scale every experiment divides power by). The five decision substations
are D5, D10, D11, D14, and D16; D9 has 9.0 months of history at 2024-07-01 and is reported beside
them. ERA5 (a reanalysis) and CAMS (a satellite-based irradiance product) supply temperature and
irradiance; both are estimates of past weather.

## Figures

![Net power and three forecasts in a summer and a winter week at six substations](figure_01_raw_weeks.svg)

![Planned and exploratory contrasts, with intervals](figure_02_headline.svg)

![Quantile score of every arm at every site](figure_03_absolute.svg)

![Fitted drift paths per hour band](figure_04_drift.svg)

![Fitted capacity per season-year against experiment 45](figure_05_capacity.svg)

![Tail calibration of every arm](figure_06_tails.svg)

![Score and upper-tail coverage by month of the scored year](figure_07_staleness.svg)

![CPU against GPU for the framework module](figure_08_benchmark.svg)

## Results

- **Arms.** Convex model (G): `G1` (12 months, no drift), `G-all` (all years, no drift),
  `G-all+D1` (all years, a drift level per hour band), `G-all+D1+steps` (D1 plus sparse steps).
  XGBoost (X) without power lags, same weather inputs, named by window (`12` months or `all` years)
  and suffix: no suffix is XGBoost's default start, `z` a zero start, `am` a zero start plus
  experiment 64's solar offset, `amq` (post hoc) each quantile starting at its training quantile
  plus the offset, and `ng` (post hoc) no irradiance input. References: climatology and
  `manual_heuristic`. Penalties were tuned on 2024-07 to 2025-06 by mean absolute error (ridge
  0.1, drift step 3,000, step penalty 300). Every fit is frozen at 30 June 2025 with no refits, and
  uses one seed (the screening tier of the series).
- **Absolute scores (points, five decision sites; figure 3 has every site).** G1 3.13, G-all 2.73,
  G-all+D1 2.92, G-all+D1+steps 2.99; X-12 3.07, X-all 3.15, X-12amq 3.06; climatology 4.07,
  `manual_heuristic` 3.49. At D9 the drift arms score 6.39 and 15.67; D9 holds 21 months at the
  final cut, but its replica fit for the forecast spread has only 9 months, which is the likelier
  cause.
- **C1 (the verdict contrast): G-all+D1 minus G1 is -0.20** (98.33% interval, which corrects for
  three planned contrasts by the Bonferroni method: -0.68 to +0.22; 3 of 5 better). Dropping D14
  gives +0.13.
- **C2 and C3 (labels under the merged rule, not a claim of a DP gain).** C2, G1 minus X-12am: -0.24
  (-0.80 to +0.24; 3 of 5 better; `partly supported`). C3, G-all+D1 minus X-allam: -0.29 (-0.67 to
  +0.03; 4 of 5 better; `supported` on the point rule). Both compare against zero-start arms, which
  failed their own control and are the weaker XGBoost benchmark.
- **Against XGBoost (post hoc, 95%, decision sites).** G-all minus X-12, the best lagless arm:
  -0.34 (-0.66 to -0.07; 4 of 5 better, D5 worse). G-all minus X-all: -0.43 (-0.56 to -0.29; 3 of 5
  better). G1 minus X-12 +0.06 (-0.25 to +0.37). X-all is the weaker benchmark: it is 0.78 worse
  than X-12 at D16.
- **Exploratory (95%).** G-all minus G1 -0.40 (-0.66 to -0.15; 4 of 5 better; without D14, about
  -0.22 with 3 of 4 better). G-all+D1 minus G-all +0.20 (-0.05 to +0.43; 1 of 5 better). Steps minus
  D1 +0.06 (+0.02 to +0.11). X-12z minus X-12 +0.30 (+0.16 to +0.47). X-12am minus X-12z +0.01
  (-0.01 to +0.02): with irradiance as an input the offset adds nothing and the fitted weights `w`
  are -0.02 to -0.34, not -1. Without irradiance X-12am-ng minus X-12z-ng is -0.28 (-0.48 to -0.08),
  with `w` from -0.53 to -0.71. Coastal D5 and D9: C1 is -0.14 at D5 and +4.14 at D9.
- **Drift by month since the end of the training data.** G-all+D1 minus G-all, July 2025 onward:
  +0.18, +0.68, +0.66, +0.94, +0.06 (months 0 to 4), then -0.29, -0.30, -0.50 (December to
  February), then +0.21, +0.33, +0.14, 0.00. The drift arm loses most when its held value is
  freshest, so the loss does not come from the value going stale. A better explanation is that the
  last drift knot follows the final months of training, which rose by 4 to 6 points at D10; the
  drift step also sits at the edge of its grid (3,000).
- **Lagged reference (descriptive).** XGBoost with three power lags minus the same model without:
  -0.58 (-0.97 to -0.28) at leads of 24.5 to 48 hours and -0.32 (-0.56 to -0.15) at 144.5 to 168
  hours, 5 of 5 sites better in both. The lagged model scores 2.49 and 2.75 against 3.07, and so
  beats G-all (2.73) at the shorter lead.
- **Tails (coverage at the 1%, 5%, 95%, 99% quantiles, five sites).** G1 2.7/9.2/89.5/95.0%;
  G-all 1.5/4.5/88.4/95.5%; G-all+D1 4.2/12.0/93.6/97.8%; X-12 1.4/9.7/79.3/97.7%; X-12z 0.2/7.6/
  66.2/73.1%; `manual_heuristic` 4.2/6.1/90.6/93.3%. Pinball loss at the same levels: G-all 0.176,
  0.624, 0.795, 0.259; X-12 0.261, 0.716, 1.078, 0.338. The 95% quantile is too low for most arms
  from December to February (figure 7), though not for G-all+D1 in December. No curtailment-cost
  saving is reported: that metric needs a node with generators, and this experiment scores Primary
  demand only.
- **Predictions that held:** the framework tests pass; steps within 0.1 points of D1 (+0.06); run
  time (every arm under 10 minutes).
- **Predictions that failed:** old years without drift hurt (they helped by 0.40); solar offset with
  irradiance (no gain); zero start within -0.10 to +0.02 (+0.30); C1 in 0.0 to +0.4 (-0.20); C2 in
  0.1 to 0.5 (-0.24); C3 in 0.0 to 0.5 (-0.29); tails within 1 and 2 points of nominal (no arm);
  baselines worse than every arm at every site (about 20 arm-site pairs fail, listed in
  `report.txt`). The post-hoc no-irradiance offset gain is in range, but `w` is outside -1.3 to
  -0.6 at three sites. Capacities within 20% of the vintages: held at D10, mixed at D11 and D16,
  and D14 is not comparable.
- **Capacity.** The real-time `frozen` vintage (the weekly capacity estimate of experiment 45 made
  with data to each origin date) is not its hindsight season-year constant; the two differ in every
  shared week. See [data processing](../data-processing.md). No training row needed a growth-prior
  scaling, and the cold block has no real-time capacity before 2023-08, so it has no `am` arm.
- **Cold block (November 2022 to February 2023, a cold winter scored by fits trained to 31 October
  2022; descriptive).** Mean absolute error of the central forecast, five sites: G-all+D1 4.15,
  G-all 6.41, G1 7.66, X-12 7.31 points. The drift term helps most here, probably because it
  follows the 2022 fall in demand.
- **Scarce-data rows (no interval).** Bank holidays (1,728 rows): G-all+D1 2.50, G-all 2.62,
  `manual_heuristic` 6.05. Days below 0 °C (576 rows): G-all+D1 2.97, G-all 4.34. Christmas run
  (1,009 rows): X-allam 2.46, G-all 2.72.
- **GPU benchmark.** The pooled proxy ran 6.6 times faster on the GPU (221 s to 33.5 s); 20 stacked
  bootstrap fits 3.1 times (50.8 s to 16.4 s); one fit on six sites 2.1 times (3.3 s to 1.6 s).
  CPU and GPU predictions agree to 2e-9 points, and peak GPU memory was 0.87 GB. The two convex
  workloads run in float64, because the normal equations are too badly conditioned for float32.
  CPU timings ran while other jobs used the machine.
- **Handed on.** Part 4's default drift by the plan's rule is D1 plus steps, because steps lowered
  the tuning year's error by 0.22 points (over 0.1); in the scored year steps were 0.06 worse than
  D1, and on the tuning year G-all with no drift (5.03) already beat every drift step (5.08 at
  best). The spread model is the replica residual quantiles. The strongest XGBoost benchmark is the
  default start (`X-12`, `X-all`), and `X-12amq` is the matching solar-offset arm.

## Caveats

- ERA5 and CAMS are estimates of past weather and flatter the solar term; a production forecast
  uses forecast weather.
- The hand-filtered switching masks are hindsight labels shared by every arm. All-rows scores
  (`report.txt`) rank the arms the same way.
- Each fit is held fixed for 12 months, so it is staler than a production model, which is refitted
  weekly. The spread is fitted on the year before the end of the training data, a year tuning also
  used, so it is slightly optimistic.
- The decision set is five substations and 12 months, so intervals are wide.
- The spread model changed after a first look at the results and an Opus code review (a replica
  fit instead of quarter-by-quarter fits), so C1 to C3 here use a different spread from the one
  pre-registered, and this experiment did not compute them under the pre-registered spread. The
  post-hoc arms (`amq`, `ng`, `X-all`, `X-12amq`) were also added after that look. The verdict rule
  is unchanged.
- Penalties were tuned on one year; the drift step sits at the edge of its grid.
- Effective capacity is the project-wide constant per series used by every experiment.

## What the next experiment should answer

Part 3 pools the model across substations on the GPU. A drift that re-estimates its level from
recent data, for example by assimilating the latest observed demand, is one option for a later part,
and the drift-by-month table above says the existing drift should be examined before it is
extended: it lost most when its held value was freshest.
