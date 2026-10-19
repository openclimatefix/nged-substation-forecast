---
id: "00111"
slug: differentiable-physics-6-ens-forecast-engine-against-bg2
title: "Differentiable physics, part 6: the DP as a forecast engine on ENS, against BG2"
part: 6
theme: differentiable-physics-forecast-engine
also_relevant_to: [production-reference-and-best-guess, lagged-power-and-error-correction,
weather-products-and-blending]
status: planned
size: >-
  complex for the module changes (an ENS weather-input layer with a seasonal, per-lead-band
  calibration, the DP run per ensemble member, the filter on ENS-driven residuals, forward-chained
  base margins, all in packages/studies); medium for the experiment scripts. See "Size triggers".
hypothesis: >-
  Driven by ECMWF ENS forecasts, the convex DP demand model forecasts Primary demand up to 7 days
  ahead better than lagless XGBoost given the same recent-level correction, and, used as the base
  margin of the BG2 recipe's lagged XGBoost, makes the production recipe better like for like.
prediction: see "Predictions"
decision_rule: see "Decision rule"
benchmark: >-
  Lagless XGBoost on ENS (SIMP1's no-lag monthly recipe, rebuilt; with and without experiment 64's
  solar offset; with the same Kalman filter); the lagged BG2 member with its default start, a zero
  start, experiment 87's solar-offset start, and a placebo margin; BG2 rebuilt (experiment 87,
  5.260 over 16 Primaries as stored); climatology; manual_heuristic
row_set_id: >-
  fold_a_daily_gap_v1 (the 292 daily ENS runs of experiment 87) on the MW sites of dp_mw_v1;
  replicate on the even months of 2024-07 to 2025-06 (51-member frames built for this part)
metric: >-
  Quantile score (twice the mean pinball loss over experiment 87's 11 quantiles) in points of
  effective capacity, on rows outside the hand-filtered switching events at the scored site, over
  leads 0.5 to 168 hours from the 09:00 UTC power_fcst_init_time
production_status: not-for-production
code: ml-experiments/00111-differentiable-physics-6-ens-forecast-engine-against-bg2/
---

**Status: planned. No fit has run.** This page is the plan, committed before any result exists.

**The default base model is the convex GAM with pinned capacity and Part 4's Kalman filter (GAM-K,
the main arm), with GAM-margin second.** In Part 3's scored year the pooled model P lost to the
per-site model S and to lagless XGBoost (C1 +0.385; the failed gate caps the verdict at
`inconclusive`), while the convex G-all scored 2.726. Part 4's audited headline has the filter
lowering the convex model's day-ahead score from 2.90 to 2.29, and at matched leads scoring 2.349
against lagged XGBoost's 2.487. The pooled DP survives only as branch B1. Choices that still depend
on Parts 3 and 4 are branches, settled before any scored fit and recorded in `decisions.json`.

## What Part 6 asks, and why BG2 is compared like for like only once

**Part 6 asks whether the DP is worth having as a forecast engine on real weather forecasts**,
on its own with a recent-level correction (GAM-K) or as the starting point of the production
recipe's lagged XGBoost (GAM-margin), driven by the ECMWF ensemble (ENS) as BG2 is, at the same 292
runs, leads up to 7 days. Parts 1 to 5 used ERA5 and CAMS, which flatter the solar and heating.

**BG2 is the bar, but BG2 differs from GAM-K in four recipe parts**: three power lags, a
five-member blend with weights re-chosen monthly, weekly refits, and experiment 18's spread factor
1.15. The scored year's 73 every5d runs chose BG2's blend and offset. A DP gain is claimed in
two places only:

- **Against lagless XGBoost**, with and without experiment 64's solar offset, and with the same
  filter bolted on, so the filter is not credited to the DP.
- **Against BG2 like for like** in one contrast (C5, "Confirmation tier"). Every other BG2 row is
  labelled "production recipe, not like for like".

**A GAM gain over XGBoost mixes structure, history length, and training weather**: the GAM trains
on years of ERA5 and CAMS, the XGBoost bars on 3 to 27 months of ENS control rows from 2024-04-01.
C1 is named for that mixture, and GAM-ens (the GAM on ENS alone) stands beside GAM-0.

**Three earlier results shape the plan, beside the rule that one-year contrasts need a replicate:**

- **Recent information is worth about 0.9 points.** SIMP1 found that removing the three lags makes
  the Primary forecast 0.90 points worse (95% interval 0.52 to 1.44). A lagless DP against BG2 is a
  known loss, not a test, which is why GAM-K, not GAM-0, is the main arm.
- **Part 4's filter gain will not carry over unchanged.** On ENS the transient state also absorbs
  weather forecast error, and live residuals cannot use ERA5 (about 5 days late), so the decision
  arms use ENS-driven residuals.
- **Experiment 87's solar offset gained 0.010 points against a predicted 0.10 to 0.15** (experiment
  64 had found 0.152), so every margin and offset arm has a matching zero-start arm, and the lagged
  bar includes the offset start.

## Branches on Parts 3 and 4

**Each branch is settled from the merged write-ups before any Part 6 scored fit; none is guessed.**

| Branch | Read from | Rule |
|---|---|---|
| B1 pooled | Part 3's C2 and C3 | The GAM arms always run. The pooled arms (P-0, P-K, P-margin, exploratory) run only if Part 3's onboarding (C2) or short-history (C3) contrast is `supported`; otherwise they are recorded as not run. |
| B2 correction | Part 4's C3 rule | GAM-K (Kalman) is the decision arm by the maintainer's ruling. GAM-E (EWMA) runs beside it in seconds; if Part 4 handed on the EWMA, C2 is also reported for GAM-E. |
| B3 drift | Part 4's C4 rule | If C4 passed in both years, GAM-K has no drift (Part 4's A2 form); otherwise D1 (A3 form). B7 already says drift adds nothing once the filter runs. GAM-0 carries D1 if Part 4's W has it. |
| B4 window | Part 4's C1 and its replicate | All years if C1 is `supported` with the replicate's sign agreeing; otherwise Part 4's W (24 months with D1, tuned on 2023-24). |
| B5 capacity | Part 4; Part 5 | Pinned (experiment 45's `frozen` vintage, `studies.dp_pinned_capacity`) in every arm; Part 5's result enters only through GAM-K-fc. |
| B6 Part 4 late | Part 4's status | The module PR, the 51-member frames, calibration, gate tests 1, 2, and 5, and the XGBoost rebuilds start now; scored DP fits wait. |
| B7 Part 4 audit | Part 4's headline audit | Done: the filter's gain holds. At matched leads A3 scores 2.349 against lagged XGBoost's 2.487 (−0.138, 95% interval −0.37 to +0.06); the DP with the filter beats XGBoost with the filter by 0.21 and 0.15 in Part 4's two years; drift adds nothing once the filter runs. Only these audited numbers are quoted. |

**Part 4's tuned settings** (drift step, final knot, filter prior) are inherited as they stand.

## Module changes (tests under 5 minutes each)

**Size triggers:** private files only; no serving path or degradation rule; several defensible
designs, one pre-registered each; Part 6's scripts are the only callers. The module PR goes off
`main` with an Opus code review, leakage first (ENS availability, the filter's cut, the margin).

- **51-member frames at the 7 MW sites for the 456 runs of 2024-04-01 to 2025-06-30**, from
  `data/NWP/nwp_model_id=ECMWF_ENS_0_25_degree/` as experiment 48 did (about 1 CPU-hour).
- **ENS weather-input layer (`studies.dp_features`, filling the `WeatherInputLayer` hook).** For one
  daily run (00:00 UTC `nwp_init_time`, usable from its 09:00 UTC `power_fcst_init_time`) and one
  member, build the DP's weather columns at each target half-hour:
    - **Temperature:** the member's value after the initialisation; the trailing 6, 24, 72, and
      168-hour means stitch it onto the analysis proxy before the initialisation.
    - **The analysis proxy, with its lead defined from `nwp_init_time`:** for each past half-hour,
      the control member of the latest run whose `power_fcst_init_time` is at or before τ, at NWP
      lead 9 to 33 hours. Experiment 48's frames start at 09:30 on the run day, so 00:00 to 09:00
      half-hours take the previous run's values (NWP lead 24 to 33 hours). Never ERA5.
    - **Irradiance:** the member's shortwave as experiment 87's frames hold it, then the Erbs split
      and `dp_features.FLEET`'s unit PV shape, derated by the member's temperature.
- **Calibration to the training weather, per season and lead band.** The DP is trained on ERA5 and
  CAMS, so ENS is mapped onto them before it enters the model. Temperature takes an additive shift,
  and irradiance a multiplicative ratio clipped to [0, clear sky], per site, season (December to
  February, March to May, June to August, September to November), lead band (0–24, 24–48, 48–144,
  144–168 hours), and hour band. The calibration is fitted on the odd months' ensemble mean against
  ERA5 and CAMS and applied equally to every member. Every season holds at least one odd month, so
  the even months and the scored year use the same seasonal values. Then the forecast solar anomaly
  is shrunk towards CAMS's month-by-half-hour climatology by a coefficient per lead band (experiment
  51: 0.42 to 0.69; experiment 46: ENS has anomaly skill to about 2 days).
- **The DP run per member** (respecting the heating hinge's nonlinearity). Each member's central
  forecast gets the arm's model-error quantiles (`dp_gam`'s replica spread, or Part 4's post-filter
  spread), and the 51 members' quantiles are pooled by experiment 87's `mixture_quantiles`.
- **Filter on ENS-driven residuals:** observation minus the DP driven by the analysis proxy, under
  the 3.5-hour cut (BG2's telemetry delay too); an option uses ERA5 and CAMS for GAM-K-era5.
- **Forward-chained base margin.** The GAM is refitted at every monthly cut from 2024-04-01 (about
  27 cuts × 7 sites, minutes of convex fits). An XGBoost training row whose run is in month m takes
  the margin from the GAM of the latest monthly cut before that run, driven by the same ENS control
  member the row sees; at forecast time the run takes the GAM of the latest cut before it. Each row
  therefore gets exactly the margin a live service would have had.
- **Tests:** (1) a plumbing test: with every member set to the control, the mixture equals the
  control-member forecast exactly; (2) feeding ERA5 and CAMS through the ENS path with zero shift,
  ratio 1, and shrink 1 reproduces Part 4's stored predictions to 1e-6; (3) changing any
  observation, ENS run, or analysis-proxy value that a run at τ could not have seen leaves the
  forecast at τ unchanged exactly, for GAM-K, GAM-margin, and X-bar+K, and every proxy value comes
  from a run whose `power_fcst_init_time` is at or before τ; (4) no margin comes from a GAM whose
  cut lies after the row's run; (5) leakage asserts as in Parts 2 to 4, and nothing scores a valid
  time on or after 2026-07-01.

## Years, runs, sites, and rows

**Three periods, each with a role no other period plays.** The GAM arms refit monthly in all three,
matching SIMP1's monthly no-lag arm; there are no frozen cuts in Part 6.

| Period | Weather | Role |
|---|---|---|
| Odd months of 2024-07 to 2025-06 | ENS, 51 members | tuning: calibration, solar shrink, the filter's ENS-residual hyperparameters, the spread factor, and the bars' start choice |
| Even months of 2024-07 to 2025-06 | ENS, 51 members | replicate: the sign of C1 to C3, required only in February, April, and June 2025 |
| 2025-07-01 to 2026-06-30, the 292 runs of `fold_a_daily_gap_v1` | ENS, 51 members | scored: the verdict, and the only period with BG2 |

**The replicate's real weakness is XGBoost's short training history.** An XGBoost refit at
2024-07-01 sees 3 months of ENS rows, against 12 to 27 in the scored year, while the GAM has years
of ERA5, so the early months favour the DP. The replicate sign is required only in February, April,
and June 2025 (10 or more months of rows), the bars' start choice is read on January, March, and
May 2025, and the write-up states the bias.

**No longer ENS history is on local disk:** the Source Cooperative backfill (2021-03 to 2024-03)
exists only as the pilot's sample of control-member dates (`data/studies/ens_backfill_pilot/`).
Part 6 fetches nothing; a full backfill needs the maintainer's approval and is a future option.

- **Sites.** Decision sites D5, D10, D11, D14, and D16 (Phase A needs 4 of 5). D9 is scored beside
  them, and D18 is descriptive. MVA Primaries stay out, with at most one descriptive row if Part 5's
  observation layer has merged; BG2's other Primaries are outside the DP pool.
- **Scored rows.** Every present row of the 292 runs at leads 0.5 to 168 hours outside the
  switching masks at the scored site (headline), and every present row beside it. BG2, SIMP1's
  no-lag arm, climatology, and `manual_heuristic` are re-scored on exactly these rows. Experiment 87
  and `dp_features` both divide by `effective_capacity_mw`, and the module PR asserts that the two
  `actual` columns agree to 1e-6 on shared rows.
- **Lead bands** for every contrast: 0–24, 24–48, 48–144, 144–168, and 12–36 hours (curtailment).
- **Predictions written:** the keys, all 11 levels `q01` to `q99`, `actual`, `lead_hours`, `arm`,
  `seed`, plus `level` and `anomaly` (filter arms) and `margin` (margin arms).
- **Robustness rows, for every contrast:** November to March against April to October; without
  D16; the all-rows score; leaving one decision site out.

## Arms

**Every arm is screening tier unless it says otherwise.** GAM fits are convex and run on the CPU.
**DP arms (the convex GAM with pinned capacity is the default; B1 adds the P rows):**

| Arm | Model | Correction | Notes |
|---|---|---|---|
| GAM-0 | convex GAM, window and drift by B3 and B4 | none | the lagless DP |
| GAM-K | as GAM-0 | Kalman on ENS-driven residuals | the maintainer's GAM-K; the main arm |
| GAM-E | as GAM-0 | EWMA, 14-day half-life | B2 |
| GAM-K-era5 | as GAM-K | Kalman on ERA5-driven residuals | the filter on estimates of past weather, not available live |
| GAM-ens | convex GAM on ENS control rows | none | see below |
| GAM-K-w | as GAM-K | Kalman | weekly refits (52 cuts), exploratory |
| GAM-K-fc | as GAM-K, capacity fitted | Kalman | B5, exploratory |
| P-0, P-K | pooled DP, pinned capacity | none, Kalman | only under B1, exploratory |

**GAM-ens** is the GAM trained on ENS control rows at all leads 0.5 to 168 hours, as XGBoost
trains, each valid time weighted by one over the runs covering it, fitted only at 2025-07-01 (15
months), so neither tuned nor replicated. It answers the train-on-its-own-history rule; the series
brief's ERA5-trained design stands, and the write-up names the departure.

**Lagless XGBoost bars** use SIMP1's no-lag recipe at SIMP1's own settings: BG2's features without
the three lags, ENS control-member training rows on the whole history, 500 rounds, eta 0.05, a
half-life of 0.5 years, monthly refits, the 51-member mixture, and 11 quantiles.

| Arm | Start | Notes |
|---|---|---|
| X-nl | XGBoost's estimated base score | reproduces SIMP1's `p_nl_monthly` on the 292 runs within 0.01 points before use |
| X-nl-z | zero | matching arm for X-nl-o |
| X-nl-o | zero plus experiment 64's solar offset | offset as in `e00087_offset.py` at the decision sites; each site's weight w reported |
| X-nl+K | as X-nl | the same Kalman filter on X-nl's residuals, so a filter gain is not credited to the DP |

**The lagless bar (X-bar)** is the tuning-month winner of X-nl and X-nl-o; X-bar+K adds the filter.

**Margin arms: the BG2 recipe's weekly lagged member (`n1h` with the three lags) at weekly refits,
150 rounds, on the whole history.** The ml-experiment skill reserves P-roll for contrasts of 0.05 or
more, and
C3's predicted size is smaller, so C3 does not use P-roll.

| Arm | Base margin | Notes |
|---|---|---|
| GAM-margin | GAM-0's forecast, forward-chained | the maintainer's GAM-margin; the second arm |
| Xlag-d | none (XGBoost's base score) | the member's start at the sites without an offset |
| Xlag-o | median of the training labels plus experiment 87's solar offset, at D9, D10, D11, D14, and D16 | the member as BG2 fits it at the offset sites |
| Xlag-z | zero | the matching zero-start arm |
| Xlag-pl | GAM-0's forecast for the same calendar day one year earlier | placebo margin |
| P-margin | P-0's forecast | only under B1 |

**The lagged bar (Xlag-bar)** is the tuning-month winner of Xlag-d, Xlag-o, and Xlag-z, so C3 does
not credit the GAM's PV term with what experiment 87's offset gives on its own.

**Confirmation tier, run only if C3's scored estimate is below zero (approved: 1.5
GPU-hours, capped at 2.5 under experiment 87's stop rule).** C5 compares two rebuilt blends that
differ in one member's file:

- **BG2-r** is BG2 rebuilt (SIMP1's `p_b_bg2_rebuilt` recipe), its weekly member refitted at
  production settings (500 rounds, eta 0.05, half-life 0.5 years); it must reproduce `p_s6_bg2`
  within 0.005 points on the 7 sites before C5 is read.
- **BG2-margin** is BG2-r with that weekly member given the GAM margin. At the offset sites the GAM
  margin replaces the offset margin, because BG2's member already starts there. The GAM refits at
  monthly cuts, and the weekly XGBoost refits take forward-chained margins from the latest cut.
- **Both** reuse the other four members' stored files, re-choose the blend weights by BG2's own
  rolling monthly rule on the CPU (freezing BG2's weights would penalise the new member), and keep
  experiment 18's spread factor 1.15.

**References as stored:** `p_s6_bg2`, `p_s4_weekly`, `p_nl_monthly`, climatology, manual_heuristic.

## Tuning (odd months of 2024-25, by quantile score, before the gate)

- The calibration (fitted, not chosen), and the solar shrink coefficient per lead band on
  {0, 0.25, 0.42, 0.55, 0.69, 1.0}.
- The filter's ENS-residual hyperparameters by marginal likelihood, with Part 4's fitted prior as
  the starting prior; one set per base arm (GAM-0's for GAM-K, the winner's for X-bar+K).
- The spread factor per lead band on {0.9, 1.0, 1.15, 1.3}, with every arm in a contrast given the
  same grid on the same months. C5's arms keep 1.15, and the BG2 rows also show the DP arms at 1.15.
- X-bar and Xlag-bar, on January, March, and May 2025. D9 is left out of the tuning score.

## Known-answer gate (blocking, after tuning, before any replicate-month or scored-year fit)

**A failed blocking test stops scoring until it is fixed; if it cannot be fixed inside the part,
the verdict is `inconclusive`.** Each test runs on the tuned settings.

1. **The calibration recovers a known bias.** A fake forecast from ERA5 and CAMS with known biases
   (+1.5 K at day 1 to +3 K at day 7; irradiance ×0.8) and ENS-sized noise. Pass if the shift and
   ratio recover each bias within 15%, and GAM-0 on the calibrated fake at noise zero and shrink 1
   scores within 0.05 points of GAM-0 on ERA5 and CAMS.
2. **ENS irradiance has the right timing.** On clear-sky days of the tuning months, the hour-of-day
   profile of calibrated ENS irradiance peaks within one half-hour of CAMS's at every site, and the
   midday ratio lies in [0.85, 1.15]. A failure means a label convention is wrong (CAMS is hour
   ending; ENS shortwave is an accumulation over its step).
3. **The filter on ENS residuals tracks transfers.** Part 4's gate test 3 (transfers from
   `e00043_inject.py` at D10, D11, D14, and D16 in the tuning months) rerun with ENS-driven
   residuals: ℓ + z carries 50% of the moved MW by the end of day 2, 0.6 to 1.3 times the moved MW
   from day 2 to the end, and under 25% within 3 days of the end.
4. **The margin does not leak.** GAM-margin's training-row margin error is within 20% of its
   forecast-time error on the tuning months. Reported, not blocking: Xlag-pl minus Xlag-z, since a
   year-old forecast keeps the seasonal shape and a start alone can help (experiments 76, 78, 80).
5. **Module tests 1 to 5** pass.

## Planned contrasts and their margins

**Five planned contrasts.** C1 to C4 are screening contrasts, Bonferroni-corrected (98.75%
intervals from `studies.bootstrap`, resampling whole calendar months of the 292 runs, paired across
arms). C5 is read at 95%. Each comes with site counts, every arm's absolute score, the
mean-forecast error of the same contrast, the replicate months' estimate and count, the lead bands,
and the robustness rows.

- **C1, the ERA5-trained GAM on calibrated ENS against ENS-trained lagless XGBoost:** GAM-0 minus
  X-bar.
- **C2, the direct path, same filter on both:** GAM-K minus X-bar+K.
- **C3, the margin path:** GAM-margin minus Xlag-bar, with GAM-margin minus Xlag-pl beside it (the
  information beyond shape).
- **C4, does the filter still help on ENS:** GAM-K minus GAM-0.
- **C5, against BG2 like for like:** BG2-margin minus BG2-r.

**The noise floor and the margins.** Measured floors: 0.001 points for an XGBoost repeat fit
(experiment 87), zero for the convex GAM, and under B1 one second-seed P-0 fit. The binding floor is
the paired interval: about ±0.3 to ±0.4 for DP against XGBoost over one year (Parts 3 and 4), about
±0.05 for two XGBoost recipes differing in one part (experiment 87's blend step, 0.038 to 0.135).

- **C1 and C2** use the series' Phase A rule with zero as the threshold, plus the replicate sign
  check. Non-inferiority within 0.10 is not decidable at ±0.3, so an estimate within ±0.10 whose
  interval spans zero reads "no difference detected at this sample size", never "as good as".
- **C3 and C5** compare two XGBoost recipes that differ in one part, so each is decidable at about
  ±0.05. `supported` needs the interval wholly below zero, 4 of 5 decision sites better, and an
  estimate of at least 0.10 points; `small gain` if the interval lies below zero but the estimate
  is smaller than 0.10.
- **C4** is read by sign and lead band (both arms share the GAM, so its interval is narrow).

**Exploratory at 95%,** grouped:

- **Against BG2, labelled "production recipe, not like for like":** GAM-K minus BG2; GAM-margin
  minus BG2's weekly member; X-nl minus BG2 (SIMP1's lag gap, reproduced); GAM-K minus Xlag-bar.
- **Solar offset:** X-nl-o minus X-nl-z, X-nl-o minus X-nl, and Xlag-o minus Xlag-z, by month and
  on April-to-September daylight rows (experiment 87's untested advice).
- **Filter:** GAM-K-era5 minus GAM-K; GAM-E minus GAM-K; ℓ against z per day of lead; GAM-K-w minus
  GAM-K; coverage at 0–24 hours before and after the spread factor, because the post-filter spread
  and the member spread both hold short-lead weather error.
- **Inputs and pooling:** GAM-ens minus GAM-0; the shrink curve; GAM-K-fc minus GAM-K; under B1,
  P-0, P-K, and P-margin minus their GAM twins.
- **Descriptive:** D9; D18; coastal D5 and D9; bank holidays, the Christmas run, days below 0 °C.

## Predictions

Written before any fit; each line can fail, and none of them overrides the decision rule.

- **C1:** GAM-0 minus X-bar is −0.05 to −0.40 (Part 2's −0.55 on past weather shrinks once both use
  ENS), with the gain concentrated in 0–48 hours and under a third of it left at 144–168 hours.
- **C2:** GAM-K minus X-bar+K is −0.05 to −0.25 (Part 4's 0.21 and 0.15 on past weather), the same
  sign in the replicate months.
- **C3:** GAM-margin minus Xlag-bar is −0.02 to −0.15, larger beyond 48 hours, where the lags carry
  least; Xlag-pl lies within ±0.03 of Xlag-z.
- **C4:** GAM-K minus GAM-0 is −0.3 to −0.6 at 0–24 hours (Part 4's −0.61 on past weather, cut by
  ENS error) and under −0.1 beyond 144 hours.
- **C5:** BG2-margin minus BG2-r is −0.01 to −0.08 (`small gain`).
- **Against BG2:** GAM-K is 0.0 to 0.4 points worse; X-nl 0.5 to 1.4 worse (SIMP1's 0.90).
- **Past-weather filter and offset:** GAM-K-era5 beats GAM-K by 0.05 to 0.20 at 0–24 hours; X-nl-o
  minus X-nl-z is −0.05 to −0.20 on April-to-September daylight rows, under 0.03 over all rows.
- **Tails:** GAM-K's q05 and q95 coverage within 3 points of nominal, q01 and q99 within 1.5
  points, at every lead band; the member mixture under-disperses at 0–24 hours before the spread
  factor.
- **Run time:** each arm under 10 minutes (CPU) or 15 (GPU); screening under 4 hours; GPU <10 GB.

## Decision rule

**The verdict answers the series' Part 6 question, "does the DP, directly or as XGBoost's base
margin, beat BG2 on ENS?", and only C5 can say "beats BG2".**

- **`supported`:** C5 is `supported`, C3 was below zero in the replicate months, and the gate
  passed.
- **`small gain`:** C5 is `small gain` under the same conditions.
- **`partly supported`:** C5 is neither (or was not run), but C3 is `supported` or `small gain`, or
  C2 is `supported` by the Phase A rule while GAM-K minus Xlag-bar is not wholly above zero at 95%,
  each with the replicate months' sign agreeing.
- **`refuted`:** the intervals of C2 and C3 both lie wholly above zero.
- **`inconclusive`:** anything else, or a gate test that cannot be fixed inside the part.

**What each outcome hands on.** A `supported` or `small gain` C5 makes BG2-margin a production
candidate, which fires audit trigger 7; whether to run a second validation year is the maintainer's
call. A `partly supported` result through C2 makes GAM-K the DP's candidate for v2's 1,300
substations, where one XGBoost model per series is costly to maintain. If C4 is not below zero at
0–24 hours, Part 4's filter gain is reported as a past-weather result only.

## Run order, compute, and time

**Order:** the module PR and frames (about 6 agent-hours, startable now under B6); B1 to B7; the
X-nl reproduction; tuning; gate; replicate months; scored year; exploratory arms; BG2-r, BG2-margin.

**Compute: about 3 to 4 hours of screening wall clock.** GAM fits: about 27 monthly cuts × 7 sites,
minutes on the CPU (8 cores). The per-member GAM prediction covers about 820 runs × 51 members × 7
sites × 336 half-hours across the three periods, about 10 minutes per arm for about 7 arms. The
filter and EWMA take seconds. XGBoost on the GPU: the four lagless arms at 24 monthly refits from
2024-07 and the five margin arms at about 104 weekly refits, each predicting 51 members, about 1.5
to 2 hours in all, under 10 GB. The confirmation run adds 1.5 to 2.5 GPU-hours. Report, figures,
cards, and write-up take about 4 hours plus the reviews, any audit, and the prose sweep.

**If the budget is at risk, cut in this order:** the pooled arms, GAM-K-w, GAM-K-fc, then X-nl-z
(recorded as deferred). The decision arms, their bars, GAM-ens, and the placebo are never cut.

## Data paths

- **ENS at the demand sites:** the 292 scored runs, 51 members, in experiment 48's
  `work/frames_demand/init=<date>.parquet`; the 73 every5d runs in
  `data/studies/per_study/stage1_pilot/demand/ens_every5d/val/`; the 456 runs from 2024-04-01 in
  `data/NWP/nwp_model_id=ECMWF_ENS_0_25_degree/` (51 members; `.../ens_every5d/train.parquet` holds
  the control member only). Experiment 87's `e00087_common.py` reads the frames (copy it).
- **BG2:** `00087-*/predictions/` (`p_s6_bg2`, `p_s4_weekly`, `ref_climatology`,
  `ref_manual_heuristic`), code in `e00087_fit_demand.py`, `e00087_offset.py`, and
  `e00087_assemble.py` (the blend weights). **SIMP1:** `00102-*/work/arms/` (`p_nl_monthly`,
  `p_b_bg2_rebuilt`), code on its branch until it merges.
- **The DP:** Parts 1 to 4's outputs, Part 4's `work/tuned.json`, and the `studies.dp_*` modules.

## Checks: the seven audit-before-merge triggers

The write-up's "Checks" bullets give the median minus mean forecast per site and arm, the
mean-forecast error beside each quantile-score contrast, and, per trigger:

1. **A control fails:** a blocking gate test; X-nl's reproduction of SIMP1; BG2-r's of BG2.
2. **A result contradicts a prediction or an earlier finding:** chiefly C4 and C2 against Part 4,
   X-nl minus BG2 against SIMP1's 0.90, and the offset rows against experiments 64 and 87.
3. **Quantile score and mean-forecast error disagree in sign** on any of C1 to C5.
4. **Leaving one decision site out flips a sign or a label** of C1 to C5.
5. **A nested arm beats its container, or a null control moves:** GAM-0 beating GAM-K, X-nl beating
   X-nl+K; the placebo margin moving the score by more than its paired half-width.
6. **An interval more than twice or under half as wide as its sister's:** C1 and C2 against Part 4's
   C6, C3 and C5 against experiment 87's nested steps, X-nl minus BG2 against SIMP1's.
7. **A later design depends on an unreplicated result:** fires whenever C5 makes BG2-margin a
   production candidate, since the scored year is the only year with BG2.

## What the write-up must contain (beyond the series brief's standing rules)

Title "Differentiable physics, part 6: the DP as a forecast engine on ENS, against BG2"; front
matter as above (`part: 6`, theme `differentiable-physics-forecast-engine`). The take-home says in
plain words whether the DP forecasts better than XGBoost when both see the same weather forecast
and the same recent observations, whether giving the production recipe the DP's forecast as a
starting point improves it, and how much of Part 4's filter gain survives real weather forecasts.
The intro says which branches were taken, that C1 mixes structure, history length, and training
weather, and why BG2 is compared like for like only in C5.

- **Figures:** raw data first (a winter and a summer week per decision site: actual, GAM-K,
  GAM-margin, X-bar, BG2, at day 1 and day 6); the headline chart of C1 to C5 with intervals, site
  counts, the replicate months, and GAM-ens beside GAM-0; score against lead per main arm with BG2
  and SIMP1's no-lag arm; scores per site; C4 by lead as ℓ and z; the calibration; the gate; the
  tails (coverage and pinball at q01, q05, q95, q99, every arm, by lead band); the offset by month.
- **Tails table:** coverage and pinball at the four tail levels and threshold-weighted CRPS for
  every arm. Curtailment cost against `manual_heuristic` is reported with its event census; at the
  MW Primaries experiment 87 found 0 events at f = 0.9, so the metric cannot rank these forecasts.
- **Substation cards** (D-labels only, labelled "Provisional: fitted to past weather, forecast on
  ENS, screening run", from GAM-K's fit at the 2025-07-01 cut): the series brief's panels, with
  capacity as the convex fit and the pinned vintage (and the pooled fit under B1, captioned as not
  identified by the data); the filter's ℓ and z over the scored year on ENS-driven residuals beside
  GAM-K-era5's; each site's calibration by lead band; and how far GAM-margin's XGBoost moves the
  GAM's forecast, by lead. One cross-substation figure compares capacity, heating slope, filter
  step, and calibration.
- **Caveats, Part 6's own:** XGBoost's 3 to 27 months of ENS history against the GAM's years of
  ERA5; replicate months interleaved with tuning months; the 73 every5d runs (interleaved with the
  292) chose BG2's blend and offset, and BG2's 1.15 is experiment 18's; two mild shared leaks (the
  2024-25 margins use a calibration fitted on the same odd months, and the GAM's last training rows
  use ERA5 a live refit would not yet have); the analysis proxy; hindsight switching masks; 7 MW
  sites, not BG2's 16 Primaries.

**Privacy: D-labels only, with dates;** G-a and G-b are not plotted, injected transfers appear as
ratios only, and D15's generator is never named or guessed. **Departures from the series brief, each
deliberate:** the brief's hybrid is GAM-margin, on the convex model first; the ENS calibration is
per season, lead band, and hour band, with a ratio for irradiance; BG2 enters a decision contrast
only as C5, where "matched corrections and refits" holds by construction; interleaved months replace
a separate replicate year.
