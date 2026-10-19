---
id: "00104"
slug: differentiable-physics-2-simple-structural-model
part: 2
theme: differentiable-physics-demand-calendar-and-weather
also_relevant_to: [retraining-and-training-history, solar-disaggregation]
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
  A convex model of demand minus solar, trained on every usable year with no drift term, scored
  0.40 points lower than the same model trained on 12 months (95% interval 0.15 to 0.66 lower, 4 of
  5 sites better), and 0.43 points lower than XGBoost trained on the same years. Adding a slowly
  drifting level made the forecast worse again (0.20 points higher, interval spanning zero),
  because a frozen fit holds the last drift value for a year. The pre-registered contrast, all
  years with drift against 12 months, is therefore inconclusive (0.20 lower, interval 0.68 lower to
  0.22 higher, 3 of 5 sites better), and the next step is a drift that can adapt after the cut.
advice:
  - >-
    Give more years to a convex demand model without a frozen drift term before adding drift, since
    all years without drift lowered the score by 0.40 points against 12 months (G-all minus G1).
  - >-
    Start each XGBoost quantile from that level's own training quantile, not from a constant, since
    a zero start left the 95% and 99% quantiles covering only 66% and 73% and scored 0.30 points
    worse (X-12z minus X-12).
  - >-
    Do not add a solar offset to XGBoost that already sees CAMS irradiance: it changed the score by
    under 0.03 points, and it helped by 0.28 points only without the irradiance input (X-12am-ng
    minus X-12z-ng).
  - >-
    Fit forecast spread on a replica of the frozen forecast, a fit 12 months before the cut, not on
    quarter-by-quarter cross-fits, because staleness runs from 0 to 12 months and quarterly fits
    cover only 0 to 3.
  - >-
    Run the pooled and batched DP workloads on the GPU from Part 3, since the pooled proxy ran 6.6
    times faster there (33.5 s against 221 s).
code: ml-experiments/00104-differentiable-physics-2-simple-structural-model/
---

# Differentiable physics, part 2: a simple structural model of demand and solar

**A convex model of demand minus solar gains from more years of history, but its drift term, frozen
at the cut, takes the gain away again.** Trained on all usable years with no drift, the model scored
0.40 points of capacity better than the same model trained on 12 months, and 0.43 points better
than XGBoost trained on the same years. The pre-registered verdict contrast, all years with drift
against 12 months, was inconclusive: 0.20 points better, with an interval from 0.68 better to 0.22
worse, and only 3 of the 5 decision substations improving. Several pre-registered positive controls
failed, so the result is inconclusive by the plan's own rule. It narrows the long-history question
to "more years help a convex model; a held drift does not".

We asked whether a model built from physics terms can use old years of data. The model is a sum of
a daily profile, holiday and heating terms, and a solar term, fitted by one convex optimisation per
Primary substation. We fitted it on data to 30 June 2025, froze it, and scored it on every
half-hour from 1 July 2025 to 30 June 2026 at six substations. The score is twice the mean pinball
loss over 11 quantile levels, in points of the substation's effective capacity (smaller is better).
The five decision substations are D5, D10, D11, D14, and D16. D9 has only 9.0 months of history at
2024-07-01 and is reported beside them.

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
  `G-all+D1` (all years, drift level per hour band), `G-all+D1+steps` (D1 plus sparse level steps).
  XGBoost (X) without power lags, same weather inputs: `X-12`, `X-all` (default start), `X-12z`,
  `X-allz` (zero start), `X-12am`, `X-allam` (zero start plus experiment 64's solar offset). Post
  hoc: `X-12amq`, `X-allamq` (each quantile starts from its training quantile plus the offset) and
  `X-12z-ng`, `X-12am-ng` (no irradiance input). References: climatology and `manual_heuristic`.
  Penalties were tuned on 2024-07 to 2025-06 (ridge 0.1, drift step 3,000, step penalty 300). Every
  fit is frozen at 30 June 2025 with no refits, one seed (the screening tier).
- **Absolute scores, five decision sites (D9 in brackets).** G1 3.13 (2.25); G-all 2.73 (2.19);
  G-all+D1 2.92 (6.39); G-all+D1+steps 2.99 (15.67); X-12 3.07 (2.12); X-all 3.15 (2.02); X-12z 3.36;
  X-12am 3.37; X-allz 3.24; X-allam 3.22; X-12amq 3.06; X-allamq 3.13; climatology 4.07;
  `manual_heuristic` 3.49. D9's drift arms are poor because 9 months of history cannot hold a drift.
- **C1, years (the verdict contrast): G-all+D1 minus G1 is -0.20** (98.33% interval -0.68 to +0.22;
  3 of 5 sites better; D14 -1.52, D10 +0.54). Verdict by the Phase A rule: `inconclusive`.
- **C2, G1 minus X-12am: -0.24** (-0.80 to +0.24; 3 of 5 better). **C3, G-all+D1 minus X-allam:
  -0.29** (-0.67 to +0.03; 4 of 5 better). Both compare against zero-start arms, which turned out
  to be the weaker XGBoost benchmark (below). Against the default-start arms (post hoc, 95%): G1
  minus X-12 +0.06 (-0.25 to +0.37); G-all minus X-all -0.43 (-0.56 to -0.29); G-all+D1 minus X-all
  -0.23 (-0.45 to -0.01).
- **Exploratory (95%).** G-all minus G1 -0.40 (-0.66 to -0.15; 4 of 5 better). G-all+D1 minus
  G-all +0.20 (-0.05 to +0.43; 1 of 5 better). Steps minus D1 +0.06 (+0.02 to +0.11). X-12z minus
  X-12 +0.30 (+0.16 to +0.47). X-12am minus X-12z +0.01 (-0.01 to +0.02): with CAMS irradiance as an
  input the solar offset adds nothing, and the fitted weights `w` are -0.02 to -0.34, not -1. Without
  the irradiance input it helps: X-12am-ng minus X-12z-ng -0.28 (-0.48 to -0.08; `w` about -0.6 to
  -0.7). Coastal D5 and D9: C1 is -0.14 at D5 and +4.14 at D9.
- **Lagged reference (descriptive).** XGBoost with the three power lags minus the same model
  without: -0.58 (-0.97 to -0.28) at leads of 24.5 to 48 hours, -0.32 (-0.56 to -0.15) at 144.5 to
  168 hours; 5 of 5 sites better in both bands. The lagged model scores 2.49 and 2.75 against 3.07.
- **Tails (five sites; nominal 1%, 5%, 95%, 99%).** Coverage, G1 2.7/9.2/89.5/95.0%; G-all
  1.5/4.5/88.4/95.5%; G-all+D1 4.2/12.0/93.6/97.8%; X-12 1.4/9.7/79.3/97.7%; X-12z 0.2/7.6/66.2/
  73.1%; X-all 2.0/8.2/83.8/98.5%; `manual_heuristic` 4.2/6.1/90.6/93.3%. Pinball loss at q01, q05,
  q95, q99: G-all 0.176, 0.624, 0.795, 0.259; X-12 0.261, 0.716, 1.078, 0.338. Every arm's upper
  tail is too low in months 5 to 7 of the scored year (figure 7). No curtailment-cost saving is
  reported: the metric needs a node with generators, and this experiment scores Primary demand only.
- **Predictions.** Held: the framework tests pass; steps within 0.1 points of D1 (+0.06); the
  solar offset without the irradiance input. Failed: old years without drift hurt (they helped);
  the solar offset with irradiance (no gain); the zero start within -0.10 to +0.02 (+0.30); C1 in
  0.0 to +0.4 (-0.20); C2 in 0.1 to 0.5 (-0.24); C3 in 0.0 to 0.5 (-0.29); tails within 1 and 2
  points of nominal (not met by any arm); baselines worse than every arm at every site (G1,
  X-12, and the zero-start arms at D14, and the drift arms at D9, were not); capacities within 20%
  of the vintages (held at D10, mixed at D11, failed at D16, D14 not comparable); run time (all arms
  under 10 minutes; the fits took under 15 minutes of CPU in total).
- **Capacity.** The real-time `frozen` vintage of experiment 45 is not the hindsight season-year
  constant: over the weeks the two files share, the ratio runs from 0.30 to 1.48 across sites and
  no week agrees to 0.1%. The vintage lists every week back to 2019-09 from each origin, but the
  first origin is 2023-08, so the cold block has no real-time capacity and no `am` arm. No training
  row needed a growth-prior scaling.
- **Cold block (November 2022 to February 2023, descriptive).** Mean absolute error of the central
  forecast, five sites pooled: G-all+D1 4.15, G-all 6.41, G1 7.66, X-12 7.31 points. The drift term
  helps most here, probably because it follows the 2022 fall in demand.
- **Scarce-data rows (no interval).** Bank holidays (1,728 rows): G-all+D1 2.50, G-all 2.62,
  `manual_heuristic` 6.05. Days below 0 °C (576 rows): G-all+D1 2.97, G-all 4.34. Christmas run
  (1,009 rows): X-allam 2.46, G-all 2.72.
- **GPU benchmark.** The pooled proxy ran 6.6 times faster on the GPU (221 s to 33.5 s), so the
  series uses the GPU from Part 3 on; 20 stacked bootstrap fits 3.1 times (50.8 s to 16.4 s); one
  fit on six sites 2.1 times (3.3 s to 1.6 s). CPU and GPU predictions agree to 2e-9 points. Peak
  GPU memory was 0.87 GB. The two convex workloads run in float64, because the normal equations
  are too badly conditioned for float32. CPU timings ran while the machine carried other jobs.

## Caveats

- ERA5 and CAMS are estimates of past weather and flatter the solar term; a production forecast
  uses forecast weather.
- The hand-filtered switching masks are hindsight labels shared by every arm. All-rows scores
  (reported in `report.txt`) rank the arms the same way.
- Each frozen fit is staler than any production model. The spread is fitted on the year before the
  cut, a year the penalties were also tuned on, so it is slightly optimistic.
- The decision set is five substations and 12 months, so intervals are wide.
- The ridge, drift, and step penalties were tuned on one year; the drift step sits at the edge of
  its grid.
- Post-hoc arms (`amq`, `ng`, `X-all`) and the replica spread were added after a first look at the
  results and a code review; the pre-registered contrasts and verdict are unchanged.
- Effective capacity is the project-wide constant per series used by every experiment.

## What the next experiment should answer

Part 3 pools the model across substations on the GPU. Part 4 should test a drift that adapts after
the cut, such as the recent-level assimilation of the series brief, because a held drift gave back
the benefit of long history. The strongest XGBoost benchmark to carry forward is the default start
(`X-12`, `X-all`), not the zero start.
