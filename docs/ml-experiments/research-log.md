# Research log of the ML experiments

**This page lists every finished ml-experiment in the order it was started, what it set out to find,
and what it found.** A reader who wants the headline of each experiment starts here and follows the
link to the experiment's `result.md` for the numbers. The experiments that are planned or proposed
are on the [research plan](research-plan.md). Each experiment carries a production status. The
maintainer reviews the candidates every one to two weeks and ports the best into a new model
release; the [release log](../live_service/model-releases.md) records which experiments each release
adopted.

## Candidates for the next release

**These experiments are flagged `candidate` and are not yet adopted.** The flags are proposals for
the maintainer to confirm.

| Experiment | What would change in production | Note |
|---|---|---|
| [00001 - ERA5-trained stage 1, solar and wind, part 1](00001-era5-trained-stage1a-vs-ens-trained/result.md) | Train the weather-to-power model on ERA5 and apply it to ENS members | Needs the staged design, and helps only up to day 6 |
| [00002 - Stage 2 calibrator for solar and wind](00002-stage2-calibrator-and-mixture-vs-ens-quantile/result.md) | Pool the ERA5-trained quantiles over the ENS members | Stage 2 itself did not help |
| [00006 - Stage 1a demand inputs](00006-stage1a-demand-input-feature-sweep/result.md) | Use the clock, a COVID indicator, and recency weights as the stage 1a demand inputs | The Christmas column needs its 14-day window, a 2023 start needs a fresh year, and windchill can go |
| [00014 - CAMS irradiance for demand, part 3](00014-cams-trained-stage1-with-stage0-ens-correction-demand/result.md) | Map the ENS shortwave onto CAMS (a stage 0 quantile mapping) before a CAMS-trained stage 1a reads it | The year's mean is corrected but individual months are up to 18.6% off, and the member spread widens by 27% |
| [00017 - Retraining and error correction, part 1](00017-training-window-and-retraining-rule-drift-demand/result.md) | Train the stage 1a demand model on the last 12 months or with a 0.5-year half-life, retrain monthly, and add a 14-day error correction | The correction needs live power 2 days old, and every gain is on ERA5 weather |
| [00019 - Retraining and error correction, part 2](00019-primary-demand-stack-vs-manual-heuristic/result.md) | Add a 14-day error correction and a spread factor of 1.15 to the single-stage ENS-trained quantile model N1 for Primary demand | The correction supplies nearly all of the gain (-0.99); it is tested on forecasts 5 days apart and on a model that is never retrained |
| [00016 - Mixture calibration, solar and wind](00016-pooled-mixture-calibration-solar-wind/result.md) | Scale the solar mixture's quantiles about the median by a factor near 0.95, refitted monthly | Fixes 94% to 95% coverage at no change in CRPS; wind needs a spread factor on two lead groups, a post hoc arm |
| [00020 - Solar and wind stack](00020-solar-wind-stack-cams-stage0-spread-margin/result.md) | Add a power-curve base margin to the wind stage 1a quantile model | Lowers the wind CRPS by 0.055 points (0.007 to 0.110); the stack of margin and spread factor gives 0.071; nothing detectably helped solar |
| [00030 - Lagged power in demand, part 3](00030-lagged-power-feature-sets-primary-demand/result.md) | Add the mean and median of the 6 weekly power analogues to N1 for Primary demand, with the 14-day correction | Post hoc arm; 0.05 points below N1 with the correction (-0.08 to -0.02), 0.03 below the 00019 stack with an interval touching zero; the annual analogues hurt |
| [00024 - Pooled demand models, part 1](00024-pooling-groupings-demand-xgboost/result.md) | Train one XGBoost model per substation type (or per size band), with the series identifier as a categorical column, instead of one per series | Costs 0.15 points on Primary demand after the error correction and saves 16 of 20 models; needs the series identifier, and the test covered 20 series |
| [00033 - Best guess against production](00033-best-guess-architecture-vs-production/result.md) | Replace the production recipe with the best guess: N1 with the 14-day correction and spread factor 1.15 for Primary demand, N1 for BSP and GSP demand, ERA5-trained mixtures for solar and wind | The comparison that ranks the candidates against production; monthly retraining without the correction scored 5.12 on one seed and needs a three-seed test |
| [00035 - Quantile blend of stored models](00035-lead-weighted-quantile-blend-of-stored-models/result.md) | Serve a fixed-weight average of the quantiles of the monthly-retrained N1, `a7_cs`, and the stack for Primary demand | Lowers the quantile score by 0.10 points (-0.19 to -0.03) against the monthly-retrained N1; needs three models, one of them refitted monthly; solar and wind gains are unresolved |
| [00034 - Retraining cadence](00034-retraining-cadence-and-cheap-proxies-demand/result.md) | Retrain the Primary demand model monthly (or weekly) and drop the 14-day correction | Monthly scores 0.20 points below the stack (95% interval -0.38 to -0.01, unresolved at 99.375%); weekly 0.07 lower again |

## The experiments so far

| Experiment | Question | Verdict | Production status | Merged |
|---|---|---|---|---|
| [00001 - ERA5-trained stage 1, solar and wind, part 1](00001-era5-trained-stage1a-vs-ens-trained/result.md) | Does a weather-to-power model trained on ERA5 beat the same model trained on ENS weather? | Supported up to day 6 | Candidate | 2026-10-14 |
| [00002 - Stage 2 calibrator for solar and wind](00002-stage2-calibrator-and-mixture-vs-ens-quantile/result.md) | Does a stage 2 calibrator beat the pooled stage 1a mixture? | Partly supported | Candidate | 2026-10-14 |
| [00003 - ERA5-trained stage 1, demand, part 2](00003-era5-trained-stage1a-vs-ens-trained-demand/result.md) | Does the ERA5-trained model beat the ENS-trained model on demand series? | Not supported | Not for production | 2026-10-14 |
| [00005 - Lagged power in demand, part 1](00005-lag-placement-stage1b-vs-stage2-demand/result.md) | Where should lagged power enter a two-stage demand forecast: stage 1b, stage 2, or both? | Refuted: stage 2 is better | Not for production | 2026-10-14 |
| [00006 - Stage 1a demand inputs](00006-stage1a-demand-input-feature-sweep/result.md) | Which non-lag inputs are best for the ERA5-trained demand model (stage 1a)? | Partly supported | Candidate | 2026-10-14 |
| [00007 - CAMS irradiance for demand, part 1](00007-cams-static-correction-stage1a-residual/result.md) | Does a static CAMS correction fitted on one year remove most of the next year's daytime stage 1a residual? | Inconclusive | Not for production | 2026-10-14 |
| [00008 - Lagged power in demand, part 2](00008-lag-summaries-in-stage2-calibrator-demand/result.md) | Do summaries of recent power (minimum, mean, and maximum of the last 24 hours and of the same weekday) help the stage 2 calibrator? | Inconclusive: no gain on top of raw lags | Not for production | 2026-10-14 |
| [00009 - CAMS irradiance for demand, part 2](00009-cams-trained-stage1a-vs-era5-trained-demand/result.md) | Does a stage 1a trained on CAMS irradiance beat one trained on ERA5 shortwave, when both are applied to ENS members? | Refuted: a gain of 0.035 points, not 0.2 | Not for production | 2026-10-14 |
| [00010 - Price features for demand, part 1](00010-elexon-price-lags-stage1a-demand/result.md) | Does the lagged electricity price lower the stage 1a demand error at substations with batteries? | Inconclusive | Not for production | 2026-10-14 |
| [00011 - School-term columns for demand](00011-school-term-and-return-to-work-columns-stage1a-demand/result.md) | Do columns for the return to work after New Year and for Lincolnshire's school terms lower the stage 1a demand error after Christmas, at Easter, and in half-term holidays? | Refuted by the rule as written (inconclusive as intended): no gain in any window | Not for production | 2026-10-14 |
| [00012 - Solar and wind inputs](00012-solar-wind-stage1a-inputs-cams-and-training-start/result.md) | Which inputs and which training start work best for the solar and wind stage 1a model? | Inconclusive: no change reached the 0.15-point gain | Not for production | 2026-10-14 |
| [00013 - Primary demand baselines](00013-primary-demand-against-stored-baselines/result.md) | Where does Primary substation demand stand: does an upgraded ENS-trained quantile model (N1) beat `manual_heuristic`, and why does the production recipe lose to it? | Inconclusive: N1 is level with `manual_heuristic`; the production recipe trails it by 1.69 points | Not for production | 2026-10-14 |
| [00014 - CAMS irradiance for demand, part 3](00014-cams-trained-stage1-with-stage0-ens-correction-demand/result.md) | Does a stage 0 that maps ENS irradiance onto CAMS help the CAMS-trained demand model? | Inconclusive: a gain of 0.067 points, but monthly mismatches remain | Candidate | 2026-10-14 |
| [00017 - Retraining and error correction, part 1](00017-training-window-and-retraining-rule-drift-demand/result.md) | Do a recent training window, recency weights, monthly retraining, or a 14-day error correction lower the demand error? | Supported | Candidate | 2026-10-14 |
| [00016 - Mixture calibration, solar and wind](00016-pooled-mixture-calibration-solar-wind/result.md) | Does a correction bring the 5% to 95% band of the pooled mixture to 90% coverage for solar and wind? | Solar supported, wind refuted (a post hoc arm for wind reaches 90%) | Candidate | 2026-10-14 |
| [00015 - Price features for demand, part 2](00015-same-time-price-stage1-demand/result.md) | Does the same-time electricity price help a stage 1 demand model explain power across all fitted series? | Inconclusive | Not for production | 2026-10-14 |
| [00018 - Staged chain against N1](00018-staged-chain-vs-n1-serving-latency-lag-anomalies-demand/result.md) | Does the staged chain, with lag anomalies from the weather sources serving has, beat N1 and `manual_heuristic` on Primary substation demand? | Inconclusive: the chain is level with N1 (+0.08) and trails `manual_heuristic` (+0.52) | Not for production | 2026-10-14 |
| [00019 - Retraining and error correction, part 2](00019-primary-demand-stack-vs-manual-heuristic/result.md) | Does stacking a 14-day error correction, a 0.5-year half-life, and a spread factor of 1.15 on N1 beat `manual_heuristic` for Primary substation demand? | Supported: the stack beats `manual_heuristic` (-0.56) and N1 (-1.00); the correction supplies nearly all of it | Candidate | 2026-10-15 |
| [00020 - Solar and wind stack](00020-solar-wind-stack-cams-stage0-spread-margin/result.md) | Do the candidate changes for solar and wind (CAMS-trained stage 1a, a stage 0 quantile mapping, a power-curve base margin, a spread factor) add up to a lower CRPS than the pooled ERA5-trained mixture? | Partly supported: wind gains 0.071 points, solar does not | Candidate (wind) | 2026-10-14 |
| [00021 - Retraining and error correction, part 3](00021-error-correction-variants-bsp-gsp-demand/result.md) | Does the recent-error correction help BSP and GSP demand, and which variant works best? | Inconclusive for BSP and GSP; the 14-day additive form is the best variant | Not for production | 2026-10-15 |
| [00022 - Perfect-weather ceiling, demand](00022-era5-weather-ceiling-primary-demand/result.md) | How much would Primary demand gain from ERA5 weather in place of ENS weather at serving time, as a ceiling for better weather products? | Refuted: a gain of 0.10 points, not 0.3 | Not for production | 2026-10-15 |
| [00023 - Retraining and error correction, part 4](00023-lagged-power-and-error-correction-solar-wind/result.md) | Does recent observed power in the stage 2 calibrator, or a 14-day error correction, lower the CRPS of solar and wind forecasts? | Inconclusive: no sign of a benefit | Not for production | 2026-10-15 |
| [00026 - Capacity features, solar and wind](00026-capacity-features-solar-wind/result.md) | Does a feature stating each solar or wind farm's recent effective capacity lower the CRPS of the pooled mixture? | Inconclusive: no detectable gain | Not for production | 2026-10-15 |
| [00027 - PV_Live for GSP demand, part 1](00027-pvlive-lags-in-stage2-gsp-demand/result.md) | Does Sheffield Solar's PV_Live estimate at the lag times lower the stage 2 quantile score of the 2 GSP demand series? | Refuted: gains of 0.3 points or more are excluded | Not for production | 2026-10-15 |
| [00030 - Lagged power in demand, part 3](00030-lagged-power-feature-sets-primary-demand/result.md) | Which lagged power columns, up to all 13 lags of `manual_heuristic`, lower the quantile score of Primary demand in N1? | Inconclusive: the weekly analogues help alone and the annual ones hurt; with the correction the best arm gains 0.05 | Candidate (post hoc arm) | 2026-10-15 |
| [00028 - PV_Live for GSP demand, part 2](00028-pvlive-forecast-in-stage2-gsp-demand/result.md) | Does a forecast of the GSP's PV_Live solar generation, fed into the stage 2 calibrator, lower the quantile score of the two GSP demand series? | Inconclusive: -0.14 points, interval -0.34 to +0.04 | Not for production | 2026-10-15 |
| [00029 - Physics stage 1 for solar](00029-physics-stage1-solar-pytorch/result.md) | Does a differentiable PyTorch physics stage 1, fed into the XGBoost stage 2, beat the XGBoost stage 1 for the 6 solar farms? | Refuted: the physics stage 1 scores 0.079 points worse | Not for production | 2026-10-15 |
| [00025 - Price features for demand, part 3](00025-price-change-features-stage1-demand/result.md) | Does the change in the intra-day price, with its level, lower the stage 1 demand model's error for Primary substations beyond the level and an other-week price? | Small gain: 0.04 to 0.05 points, under the 0.1 threshold | Not for production | 2026-10-15 |
| [00024 - Pooled demand models, part 1](00024-pooling-groupings-demand-xgboost/result.md) | Which groupings of substations can share one XGBoost demand model without losing skill against one model per series? | Supported: 3 or 4 pooled models cost 0.15 points; one model for all series costs 0.29 | Candidate | 2026-10-15 |
| [00031 - Embedded PV features for demand](00031-embedded-pv-features-primary-demand/result.md) | Do trailing estimates of embedded solar capacity, as tree columns, a linear offset, or a sunshine-shaped correction, lower the Primary demand quantile score? | Inconclusive | Not for production | 2026-10-15 |
| [00032 - Pooled demand models, part 2](00032-substation-fingerprints-pooled-demand/result.md) | Do numeric fingerprints of each substation improve a by-type pooled demand model, and carry a substation held out of the pool? | Inconclusive: no fingerprint helped | Not for production | 2026-10-15 |
| [00033 - Best guess against production](00033-best-guess-architecture-vs-production/result.md) | Does the best-guess architecture beat production, N1, and `manual_heuristic` on the quantile score, twCRPS, and exceedance metrics, and how much would perfect weather add? | Supported: 4.89, 3.81, and 6.27 points below production | Candidate | 2026-10-15 |
| [00035 - Quantile blend of stored models](00035-lead-weighted-quantile-blend-of-stored-models/result.md) | Does a lead-weighted average of the quantiles of already-scored models beat the best single model for Primary demand, solar, and wind? | Supported for Primary demand (-0.10 points); null for solar and wind | Candidate (Primary demand) | 2026-10-15 |
| [00036 - BSP and GSP hierarchy](00036-hierarchy-aware-bsp-gsp-forecasts/result.md) | Do forecasts of the series beneath a BSP or GSP, and an ERA5-trained N1 applied to ENS members, lower the BSP and GSP quantile score? | Inconclusive: child forecasts and wind and solar columns gave no gain; the ERA5-trained N1 gained 0.36 to 0.40 points at the GSPs with intervals including zero | Not for production | 2026-10-16 |
| [00037 - Switching event detector, part 1](00037-switching-first-look/result.md) | Does a neighbour-balance detector find logged switching in the telemetry at a lower false-alarm rate than a single-series detector? | Inconclusive: size-weighted recall 0.326 (0.093 to 0.579) against the 0.35 bar | Not for production | 2026-10-15 |
| [00038 - Switching event detector, part 2](00038-switching-mask-value/result.md) | If the half-hours of detected switching events are left out of the stack's error correction, does the forecast improve on half-hours outside logged switching? | Refuted: +57.1% (+0.4% to +133.5%) threshold-weighted error against the 7-day correction | Not for production | 2026-10-16 |
| [00039 - PV disaggregation, part 1](00039-pv-disaggregation-cams-injection/result.md) | Can a CAMS-driven fit separate embedded solar from Primary substation demand, recovering injected solar-farm output within 25% and passing four checks on the real series? | Partly supported: recovery passes (0.854, 0.815 to 0.866), real checks pass at 6 of 8 Primaries | Not for production | 2026-10-16 |
| [00040 - PV disaggregation, part 2](00040-pv-capacity-error-bars/result.md) | Can a cross-arm misfit test detect a cloud-driven demand load that biases the capacity fitted to embedded solar, and how many directional Primaries are free of one in 2025/26? | Partly supported: calibrated (median capacity over truth 0.992, 0.951 to 1.032), but a day-long cloud load is flagged in only 0.35 of draws at 25% of capacity, so the verdict is `underpowered` | Not for production | 2026-10-16 |
| [00034 - Retraining cadence](00034-retraining-cadence-and-cheap-proxies-demand/result.md) | How does the Primary demand score change from frozen to daily retraining, with and without the 14-day correction, and which cheap proxy ranks arms like full monthly retraining? | Inconclusive at the planned level: monthly minus the stack -0.20 points (95% interval -0.38 to -0.01); weekly -0.07 lower again | Candidate | 2026-10-16 |
| [00041 - Switching-event labeller and gate](00041-sw2-0a-labeller-gate/result.md) | Can a labeller built on log-timed features that use no forecasting model pass a known-answer gate on injected transfers and placebo windows? | Partly supported: placebo 0 of 120 visible, recall 0.74 (0.57 to 0.85) on large, long transfers against a bar of 0.80; the receivers-named bar was waived post hoc | Not for production | 2026-10-16 |
| [00043 - Switching detector scoreboard](00043-sw2-1-detector-scoreboard/result.md) | How well do experiment 37's detectors find injected transfers when both start and end must be right? | Refuted at the 24-hour window: D1 minus D0 is -0.081 (-0.130 to -0.037); the 4-hour window reverses (+0.087) | Not for production | 2026-10-16 |
| [00044 - Switching events and the best guess's score](00044-sw2-2-switching-share-of-best-guess-score/result.md) | What share of experiment 33's best guess's quantile score and tail score falls in labelled switching half-hours, and in half-hours whose lagged power falls inside one? | Refuted: a perfect switching detector would lower the quantile score by 0.42% (95% interval -0.47% to 1.96%), under the 3% threshold, counting only transfers of 25% or more of the receiving load lasting over 24 hours | Not for production | 2026-10-16 |
| [00046 - Embedded solar share of Primary forecast error, PVDIS part 3](00046-embedded-solar-share-of-primary-forecast-error/result.md) | Does a solar term built from the ENS forecast of sunshine lower the error of a Primary demand forecast without radiation inputs by at least 0.35 points of capacity? | Refuted: -0.056 (-0.191 to +0.064) at 97.5%; the oracle built from the true sunshine gains -0.966 (-1.431 to -0.595) | Not for production | Pending |
| [00050 - A second weather forecast for the solar anomaly, PVDIS part 8](00050-second-weather-forecast-for-solar-anomaly/result.md) | Does the GEFS ensemble, alone or blended with ENS, forecast the solar anomaly better than ENS, and does that lower the demand forecast's error? | Refuted at the gate: GEFS's cross-validated squared error is 1.109 times ENS's and the blend's 1.002, against the 0.99 needed; the demand contrasts were not run | Not for production | Pending |

### 00001: ERA5-trained stage 1a against ENS-trained, for solar and wind

**Stage 1a trained on ERA5 forecasts solar and wind power better than the same columns trained on
ENS weather, up to a lead of 6 days.** The experiment applied an XGBoost model trained on ERA5 and
CAMS to each of the 51 ENS members, averaged the members, and compared the mean absolute error with
an ENS-trained model. It used 6 solar farms and 3 wind farms on one held-out year (53 weekly runs,
row set `fold_a_tuesdays_v1`).

- **Headline result:** the error fell by 1.70 points of capacity for wind (interval −2.33 to −1.08)
  and 0.57 for solar (−0.83 to −0.32). Beyond day 6 the gain is not material.
- **The gain comes from training on ERA5 weather,** not from the length of the history: a model
  trained on only the ENS months had the same error.
- **A likely reason (not tested directly):** a model trained on forecast weather learns the
  forecast's errors, which flattens the weather-to-power curve, most of all for wind, where power
  rises steeply with wind speed. A model trained on ERA5 learns the sharper curve.
- **Wind peaks gain most.** In the top decile of observed wind power the gain is 6.14 points.
- **A flaw in the first run:** the production-recipe control memorised calendar position, which
  inflated the first result by 1.13 points for solar and 3.40 for wind. Every arm now omits calendar
  columns, and every later experiment includes a climatology row.
- **Write-up and pull request:** [result](00001-era5-trained-stage1a-vs-ens-trained/result.md),
  [PR #1184 (Study: stage 1 pilot, an ERA5-trained weather-to-power model against the ENS-trained
  control)](https://github.com/openclimatefix/nged-substation-forecast/pull/1184).

![Planned contrasts by lead band](00001-era5-trained-stage1a-vs-ens-trained/figure_2.svg)

### 00002: a stage 2 calibrator against the pooled stage 1a mixture

**Pooling the ERA5-trained stage 1a quantiles over the ENS members beats an ENS-trained quantile
model, and a stage 2 calibrator with a learned blend did not improve on the pooled mixture.** The
experiment scored the continuous ranked probability score (CRPS) on the same 9 generator series.

- **Headline result:** the pooled mixture beat the ENS-trained quantile model by 0.62 points of
  capacity for wind (−0.99 to −0.26) and 0.24 for solar (−0.36 to −0.06). Stage 2 against the
  mixture was not material (solar +0.40, wind +0.26).
- **Training on the long ERA5 history** changed the CRPS by only 0.01 to 0.09 points against
  training on the ENS months.
- **A likely reason stage 2 did not help:** it had little to correct. The mixture is already close
  to calibrated for one product, and the calibrator was trained on about 15 months of ENS data for 9
  series.
- **What the test did not stress:** it used one weather product and no lagged power, so it did not
  test the two jobs stage 2 is meant to do.
- **Write-up and pull request:**
  [result](00002-stage2-calibrator-and-mixture-vs-ens-quantile/result.md),
  [PR #1189 (Stage 2 pilot: shared calibrator and learned blend against the pooled stage 1a
  mixture)](https://github.com/openclimatefix/nged-substation-forecast/pull/1189).

![CRPS by lead band](00002-stage2-calibrator-and-mixture-vs-ens-quantile/figure_1.svg)

### 00003: ERA5-trained stage 1a against ENS-trained, for demand series

**For demand series, the ERA5-trained model forecasts worse than the ENS-trained model.** The
experiment repeated 00001 on 22 series that are not wind or solar farms, on a new row set of 73 runs
(every fifth day) so that every weekday is represented in every lead band.

- **Headline result:** with weather only, the ERA5-trained error is larger by 1.37 points of
  capacity
  (+0.68 to +2.17). With calendar columns and lagged power the two are equal (+0.10, −0.26 to
  +0.50).
- **The likely cause is drift:** the error falls steadily as the training start moves later, from
  2019
  to 2023, which suggests that the customer mix and the embedded generation changed.
- **Write-up and pull request:**
  [result](00003-era5-trained-stage1a-vs-ens-trained-demand/result.md),
  [PR #1192 (Experiment 00003: ERA5-trained stage 1a against ENS-trained, for demand
  series)](https://github.com/openclimatefix/nged-substation-forecast/pull/1192).

![Contrasts by lead band](00003-era5-trained-stage1a-vs-ens-trained-demand/figure_2.svg)

### 00005: where lagged power enters a two-stage demand forecast

**For Primary substations, recent power helps more in the stage 2 calibrator than in stage 1b.** The
experiment gave both placements identical inputs and calibrated both, on 20 substation series and
the 73 runs of the validation year. Every lagged arm is a ceiling, because the recent residuals use
ERA5 weather at past times.

- **Headline result:** the primary contrast, stage 1b with a calibrator minus the calibrator with
  lags, was +0.84 points of capacity for leads up to 48 hours (99% interval +0.28 to +1.50, 9 of 12
  months favouring the calibrator). Lagged power in the calibrator scored 5.31 against 6.14 for
  stage 1b and 7.11 for no lags. Using both placements lowered the quantile score by at most 0.12
  points.
- **Finding for later experiments:** a lag-free calibrator lowered the quantile score by about 1
  point of capacity at short leads and raised the median's error. Stage 1a alone scored no better
  than day-type climatology. The grid supply points' error is dominated by drift.
- **Write-up and pull request:** [result](00005-lag-placement-stage1b-vs-stage2-demand/result.md),
  [PR #1195 (Experiment 5: where lagged power enters a two-stage demand
  forecast)](https://github.com/openclimatefix/nged-substation-forecast/pull/1195).

![Lagged power in the calibrator beats lagged power in stage 1b at every lead](00005-lag-placement-stage1b-vs-stage2-demand/figure_2.svg)

### 00006: which inputs work best for the ERA5-trained demand model (stage 1a)

**The clock is the largest single input for Primary substations, but it lowers the error by only
about 0.5 points of capacity, and the choice of training window matters as much.** The experiment
trained one XGBoost model per substation on ERA5 weather and scored held-out ERA5 months (it does
not
forecast), for 14 scored Primary substations, two bulk supply points, and two grid supply points.
It chose inputs on 2024-07 to 2025-06 and scored them once on 2025-07 to 2026-06.

- **Headline result:** hour and weekday lower the Primary error by 0.55 points (99.4% interval
  -0.71 to -0.41) on the selection year and 0.46 (-0.61 to -0.30) on the confirmation year, against
  1.5 predicted. The chosen Primary set adds a COVID indicator and the days to Christmas, which
  lowers the error by a further 0.29 on the confirmation year, supported at 95% and not at the
  planned level, and the Christmas column gains little over a year and a lot at Christmas, where it
  lowers the Primary error by 1.62 points over 15 December to 7 January. For grid supply points the
  COVID indicator beats the clock.
- **Training history and lockdown handling matter about as much as any column:** recency weights
  lower the confirmation-year error by 0.46 more. A 2023-09 start lowers it by 0.86 more, but it
  did not help on the selection year and needs a fresh year before anyone adopts it.
- **Time of year adds nothing, because the weather block already carries the season.** Windchill
  changes the error by less than 0.01 points, and trailing temperature means do not help.
- **Different substation types need different sets:** bulk supply points gain from holiday flags,
  and grid supply points gain from the COVID indicator, and have errors about four times larger.
- **Write-up and pull request:** [result](00006-stage1a-demand-input-feature-sweep/result.md),
  [PR #1202 (Experiment 00006: which input features work best for the ERA5-trained demand
  model)](https://github.com/openclimatefix/nged-substation-forecast/pull/1202).

![Error on the confirmation year by arm and substation type](00006-stage1a-demand-input-feature-sweep/figure_1.svg)

### 00012: inputs and training start for the solar and wind stage 1a model

**No input change lowered the solar or wind error by the 0.15 points the plan counted as a gain.**
The experiment tested a CAMS-trained model, a stage 0 model that calibrates ENS irradiance to CAMS,
a
clear-sky-ratio target (solar), a power-curve base margin (wind), a smaller weather block, and two
later training starts, on 6 solar farms and 3 wind farms (row set `fold_a_tuesdays_v1`).

- **Headline result:** CAMS-trained minus ERA5-trained for solar is -0.140 points of capacity (99.5%
  interval -0.277 to -0.044), largest within 24 h at -0.35. The stage 0 model made the 51-member
  average 0.160 higher (-0.017 to +0.382), although it improved the control member alone from
  8.34 to 7.82. The clear-sky-ratio target (-0.048) and the base margin (-0.030) were equivalent to
  the benchmark.
- **Finding for later experiments:** training starts in 2023-09 and 2024-04 were within 0.3 points
  of 2019-09 for solar and wind, and a gain of 0.15 points is not ruled out.
- **Write-up and pull request:**
  [result](00012-solar-wind-stage1a-inputs-cams-and-training-start/result.md),
  [PR #1217 (Plan experiment 12: inputs and training start for the solar and wind stage 1a
  model)](https://github.com/openclimatefix/nged-substation-forecast/pull/1217).

![Planned contrasts](00012-solar-wind-stage1a-inputs-cams-and-training-start/figure_2.svg)

### 00021: the error correction on BSP and GSP demand, and its variants

**The 14-day error correction did not lower the BSP and GSP quantile score, and no variant beat the
plain 14-day additive form.** The experiment post-processed N1's stored forecasts with power that is
2 days old, on 73 forecast runs with ENS weather.

- **Headline result:** the correction minus N1 on the 4 BSP and GSP series was +0.36 points of
  capacity over all leads (99.375% interval -0.19 to +0.98). On the 16 Primary substations it was
  -0.99 (-1.82 to -0.18), the same as with power 3.5 hours old.
- **Absolute scores (all leads, BSP and GSP):** N1 13.66, corrected 14.02, `manual_heuristic` 14.70,
  `climatology` 15.11.
- **Variants (Primary, minus the 14-day additive form):** 7-day window +0.19, 28-day window -0.07
  (-0.15 to +0.01), multiplicative +2.28, per-half-hour +0.02, recency-weighted +0.04.
- **Finding for later experiments:** N1's BSP series have no bias (+0.01), so a bias correction has
  nothing to remove there; the GSP series are biased low by 5.2 points, the correction removes 3.2,
  and the score still does not improve.
- **Write-up and pull request:** [result](00021-error-correction-variants-bsp-gsp-demand/result.md),
  [PR #1228 (Experiment 21: which error correction works best, and does it help BSP and GSP
  demand?)](https://github.com/openclimatefix/nged-substation-forecast/pull/1228).

![The eight planned contrasts](00021-error-correction-variants-bsp-gsp-demand/figure_1.svg)

### 00022: the weather ceiling for Primary demand

**ERA5 weather in place of ENS weather at serving time lowered the Primary demand quantile score by
only 0.10 points, so better weather products have little to offer Primary demand.** The experiment
used experiment 19's full stack on the 16 Primary substations and 73 forecast runs. ERA5 arrives
about 5 days late, so the arm is a ceiling and no serving system could use it.

- **Headline result:** the all-variable ERA5 arm minus the ENS arm was -0.10 points of capacity over
  all leads (99.38% interval -0.16 to -0.04); the ENS arm scored 5.32 and the ERA5 arm 5.22.
- **By variable:** shortwave radiation alone gave -0.08 (-0.13 to -0.04); temperature gave -0.01
  (-0.04 to +0.01).
- **By lead:** the gain was -0.08 at 3 to 6 days and -0.15 at 11 to 15 days, and ERA5 was slightly
  worse (+0.03) under 24 hours.
- **Finding for later experiments:** if a second weather product is tried for Primary demand, test
  its shortwave radiation first.
- **Write-up and pull request:** [result](00022-era5-weather-ceiling-primary-demand/result.md), [PR
  #1227 (Experiment 22: how much would Primary demand gain from ERA5 weather in place of ENS
  weather?)](https://github.com/openclimatefix/nged-substation-forecast/pull/1227).

![Each ERA5 arm minus the ENS arm](00022-era5-weather-ceiling-primary-demand/figure_1.svg)

### 00025: price-change features for stage 1 demand

**Adding the change in the half-hourly price to the price level lowered the Primary demand error by
only 0.04 to 0.05 points, and national demand and wind scored 0.22 and 0.40 points below the price
change arm.** The experiment
trained the ERA5 stage 1 model with the price at the target half-hour and its changes over 1, 2, 4,
and 12 half-hours. The price is known only after the half-hour, so no score is a forecast skill.

- **Headline result:** changes minus level only was -0.037 points of capacity (97.5% interval
  -0.056 to -0.018) in the selection year and -0.050 (-0.083 to -0.018) in the confirmation year;
  changes minus other-week changes was -0.039 and -0.047. The pre-registered threshold was 0.1.
- **Comparison arms:** national demand and wind scored 0.22 and 0.40 points below the price change
  arm, and the price level alone was 0.10 worse than no price in the confirmation year (the
  interval includes zero).
- **BSP and GSP:** the four series show no consistent sign.
- **Finding for later experiments:** try national demand and wind before any price column, and keep
  a control with the real price level and other-week changes.
- **Write-up and pull request:** [result](00025-price-change-features-stage1-demand/result.md), [PR
  #1231 (Experiment 25: does the change in the intra-day price help a stage 1 demand model explain
  power?)](https://github.com/openclimatefix/nged-substation-forecast/pull/1231).

![Primary contrasts with 97.5% intervals](00025-price-change-features-stage1-demand/figure_1.svg)

### 00029: a differentiable physics stage 1 for the solar farms

**Fitted PV physics did not beat the XGBoost stage 1 for the 6 solar farms, and added nothing when
given to it as a column.** The experiment fitted a PyTorch PV model with priors on tilt, azimuth, DC
capacity, and the AC-to-DC ratio to each farm's metered power, ran it on every ENS member, and
passed its output to a per-farm XGBoost quantile layer (row set `fold_a_tuesdays_v1`).

- **Headline result:** the PyTorch physics stage 1 minus the XGBoost stage 1 is +0.079 points of
  capacity (98.75% interval +0.029 to +0.135), and the hybrid (physics plus the nine weather
  columns) is -0.000 (-0.015 to +0.018). The XGBoost stage 1 scores 5.026.
- **Finding for later experiments:** the physics point forecast has a member-average mean absolute
  error of 7.53, against 7.64 for the XGBoost stage 1a of experiment 00012 (a descriptive
  comparison), so physics is a candidate for interpretation and checks of a farm's orientation, not
  for the quantile pipeline. The fitted
  tilts (18 to 35 degrees) and azimuths (within 1.3 degrees of south) are plausible. A synthetic
  test recovers parameters well only when the data come from the fitted chain: a different sky
  model gave a +6.2 degree tilt bias.
- **Write-up and pull request:** [result](00029-physics-stage1-solar-pytorch/result.md), [PR #1236
  (Experiment 29: does a differentiable physics stage 1 beat the XGBoost stage 1 for solar
  farms?)](https://github.com/openclimatefix/nged-substation-forecast/pull/1236).

![Contrasts](00029-physics-stage1-solar-pytorch/figure_2.svg)

### 00031: embedded solar capacity features for Primary demand

**None of the capacity features lowered the all-lead quantile score of Primary demand forecasts.**
The arms gave N1 trailing estimates of each substation's embedded solar capacity (from recent power,
irradiance, and the Embedded Capacity Register) as columns, a linear offset, or a sunshine-shaped
correction (row set `fold_a_every5d_v1`, 16 Primary substations).

- **Headline result:** quantile score, in points of capacity, against the 00019 stack without its
  half-life. Register-based PV products (A3) +0.006 (99.17% interval -0.022 to +0.033); the shaped
  correction (A5) +0.016 (-0.069 to +0.100); the capacity statistics as columns +0.167 (+0.091 to
  +0.243), a loss in all 12 months.
- **Finding for later experiments:** in 95% breakdowns the shaped correction lowers the score by
  0.37 in sunny daylight (-0.51 to -0.25) and raises it by 0.68 under heavy cloud; a possible cause,
  untested, is ENS irradiance that is biased against CAMS, so calibrate ENS to CAMS first. True
  PV_Live generation lowers the score by 0.82 at the 2 GSPs (-1.06 to -0.55).
- **Cut for time:** the no-fit physics baseline and the post hoc tweak were not run.
- **Write-up and pull request:** [result](00031-embedded-pv-features-primary-demand/result.md),
  [PR #1239 (Experiment 31: do embedded-solar capacity features lower the error of Primary demand forecasts?)](https://github.com/openclimatefix/nged-substation-forecast/pull/1239).

![Contrasts](00031-embedded-pv-features-primary-demand/figure_1.svg)

### What the experiments taught us so far

- **A 14-day error correction put N1 below `manual_heuristic` for Primary demand.** The correction
  lowered the quantile score by 0.99 points on N1, a model frozen at its training end, and the
  0.5-year
  half-life added nothing to that frozen model (00019).
- **Use the 14-day additive error correction for Primary demand, with power 2 days old, and do not
  yet use it for BSP and GSP.** It scored the same with 2-day-old power as with 3.5-hour-old power,
  a 7-day window and a ratio form were worse, and on 4 BSP and GSP series it did not lower the
  quantile score (00021).
- **A weather product as accurate as ERA5 would lower the Primary demand score by 1.9%.** ERA5
  weather
  in place of ENS weather lowered the quantile score by 0.10 points, nearly all from shortwave
  radiation and none up to 2 days ahead (00022).
- **Check for calendar memorisation.** When the same valid time repeats across many ENS runs, a
  model
  with calendar columns can memorise it. Give each experiment a climatology row, and group folds by
  valid time.
- **Choose a row set that covers every weekday.** The 53 Tuesday runs of 00001 and 00002 put every
  case of the first day on a Tuesday or Wednesday, which does not suit demand.
- **Training on ERA5 weather helps solar and wind, but a longer history does not add to it, and for
  demand a long history hurt.** The solar and wind gain appeared even when the ERA5-trained model
  saw only the ENS-era months, so the gain comes from the weather it was trained on and not from the
  years. For demand, training from 2019 was worse than training from 2023, probably because the
  customer mix and embedded generation drifted.
- **Put recent power in the stage 2 calibrator, and calibrate before judging it.** For Primary
  substations, the calibrator with recent-origin inputs beat a stage 1b correction, and a lag-free
  calibrator is worth about 1 point of capacity at short leads (00005).
- **The term and return-to-work columns showed no detectable gain on top of the Christmas column for
  Primary demand.** Over the three validation years, the return and term columns moved the Primary
  error by +0.06 points from 2 to 20 January (99.2% interval -0.17 to +0.30), +0.20 at Easter (-0.19
  to +0.59), and -0.29 over the half terms (-0.68 to +0.07), and Both was not 0.2 points better than
  a calendar shifted by 17 days (00011). The arm with the full history and the COVID indicator
  scored 0.26 points higher on the whole year than the base arm.
- **A change that helps the point forecast of solar or wind need not help the pooled mixture's
  CRPS.** CAMS-trained solar lowered the member-average error by 0.14 points (00012) and the
  mixture's CRPS by 0.009 (00020), and a stage 0 mapping widened the member spread by a quarter. A
  power-curve base margin lowered the wind CRPS by 0.055 (00020).

### 00007: does a static CAMS correction remove the stage 1a daytime residual?

**The cheap first test for P2 and P4 was inconclusive.** A correction with fixed coefficients on
ERA5 shortwave and CAMS irradiance, fitted per substation on 2024-07 to 2025-06 and scored on
2025-07 to 2026-06, removed 46.6% of the daylight squared residual at 13 Primary substations
(98.33% interval 38.4% to 53.1%), just under the planned 50%. A fixed daytime shift alone removed
39.1%, CAMS given ERA5 added a partial R squared of 0.11 (0.08 to 0.13), and the stability ratio was
0.84. One substation sets the pooled figure: without it the share is 18.2%. A CAMS-trained stage 1a
(the second stage of P2) is worth its own test, and the slope statistics of P2 wait for it. See
[`result.md`](00007-cams-static-correction-stage1a-residual/result.md) and [PR #1210 (Experiment 7:
does a static CAMS correction remove the stage 1a daytime
residual?)](https://github.com/openclimatefix/nged-substation-forecast/pull/1210).

![Share removed by substation](00007-cams-static-correction-stage1a-residual/figure_1.svg)

### 00008: summaries of recent power in the stage 2 calibrator

**Adding the minimum, mean, and maximum of the last 24 hours and of the same weekday to the
calibrator with raw lags changed the score by no resolvable amount.** The experiment reused 00005's
stage 1 statistics on the same 73 runs, and every arm with ERA5 residuals is a ceiling.

- **Headline result:** C1 (summaries added to L3) was -0.02 points of capacity for leads up to 48
  hours (98.75% interval -0.14 to +0.13). Replacing the raw power lags with summaries was worse
  (+0.29).
- **Finding for later experiments:** summaries alone, without ERA5 residuals, scored 1.18 points
  better than no recent power up to 48 hours. They are available at forecast time, but the stage 1
  statistics beneath them come from an ERA5-trained stage 1a.
- **Write-up and pull request:**
  [result](00008-lag-summaries-in-stage2-calibrator-demand/result.md),
  [PR #1209 (Experiment 00008: summaries of recent power in the stage 2
  calibrator)](https://github.com/openclimatefix/nged-substation-forecast/pull/1209).

![Summaries alone clearly beat no recent power](00008-lag-summaries-in-stage2-calibrator-demand/figure_2.svg)

### 00009: a stage 1a trained on CAMS irradiance, for demand

**Training stage 1a on CAMS irradiance in place of ERA5 shortwave lowered the Primary quantile score
by 0.035 points of capacity, a sixth of the 0.2 points the plan required.** The contrast over all
leads at the 16 Primary substations is -0.035 (97.5% interval -0.070 to +0.001), and -0.043 (-0.081
to -0.004) without the substation that set experiment 00007's pooled figure. The score falls by 0.07
to 0.08 points in the first 2 days, no gain is resolved from day 7, and the gain appears in daylight
only. At the observed weather, where the CAMS-trained arm also reads CAMS, the daylight error of the
median falls by 0.88 points, so the smaller gain at ENS is probably a mismatch at application: the
ENS members' shortwave averages 7% above ERA5's, which this experiment did not test. The paired
interval was far narrower than the plan guessed (0.035 points either side, not 0.2). See
[`result.md`](00009-cams-trained-stage1a-vs-era5-trained-demand/result.md) and [PR #1213 (Experiment
9: does a stage 1a trained on CAMS irradiance beat one trained on ERA5
shortwave?)](https://github.com/openclimatefix/nged-substation-forecast/pull/1213).

![Contrast of the CAMS-trained arm by scope](00009-cams-trained-stage1a-vs-era5-trained-demand/figure_1.svg)

### 00013: Primary substation demand against the stored baselines

**The upgraded single-stage model N1 did not beat `manual_heuristic`, and the production recipe
trails
`manual_heuristic` by 1.69 points of capacity at every lead.** The experiment scored
`manual_heuristic`,
`climatology`, the production recipe, a stage 1 trained on ERA5, and N1 (an XGBoost quantile model
trained on the ENS control member, with lagged power measured from the forecast origin) on the 16
Primary substations and the 73 runs of `fold_a_every5d_v1`.

- **Headline result:** N1 minus `manual_heuristic` was +0.22 points of capacity over all leads (99%
  interval -0.54 to +0.97) and -0.57 up to 48 hours (-1.33 to +0.27) in the mean absolute error of
  the
  median. The production recipe minus `manual_heuristic` was +1.69 (+0.98 to +2.59).
- **Absolute scores (all leads):** `manual_heuristic` 8.23, N1 8.45, production 9.92, `climatology`
  10.33 points of capacity.
- **Why the production recipe loses:** about 0.6 points is a level offset (every arm forecast the
  validation year 2 to 3 points too low, `manual_heuristic` by 1.3), and the rest matches the lack
  of
  recent power, which lagged power from the origin recovers (1.15 points, C4).
- **Finding for later experiments:** compare Primary arms against `manual_heuristic`, and test a
  drift correction in P8.
- **Write-up and pull request:** [result](00013-primary-demand-against-stored-baselines/result.md),
  [PR #1218 (Experiment 13: Primary substation demand against the stored
  baselines)](https://github.com/openclimatefix/nged-substation-forecast/pull/1218).

![The five planned contrasts](00013-primary-demand-against-stored-baselines/figure_2.svg)

### 00020: stacking the candidate changes for solar and wind

**For wind, a power-curve base margin and a spread factor together lower the pooled mixture's CRPS
by 0.071 points of capacity, and the base margin carries the gain; for solar, nothing helped.** The
experiment stacked a CAMS-trained stage 1a, a stage 0 quantile mapping, the base margin (wind only),
and the spread factor against the production recipe's pooled ERA5-trained mixture, and dropped each
component in turn, for 6 solar farms and 3 wind farms (row set `fold_a_tuesdays_v1`).

- **Headline result:** wind full stack minus recipe, -0.071 (99.58% interval -0.133 to -0.014);
  solar
  full stack minus the production recipe, +0.035 (-0.075 to +0.116).
- **Components:** the wind base margin alone gives -0.055 (-0.110 to -0.007). For solar,
  CAMS-trained
  gives -0.009, stage 0 +0.036, and the spread factor +0.006, each with an interval that includes
  zero.
- **Finding for later experiments:** the stage 0 quantile mapping widens the member spread by about
  a
  quarter and leaves monthly ratios to CAMS from 0.83 to 1.10, and it did not lower the solar CRPS
  in this experiment.
- **Write-up and pull request:**
  [result](00020-solar-wind-stack-cams-stage0-spread-margin/result.md),
  [PR #1226 (Experiment 20: do the candidate changes for solar and wind add up to a better pooled
  mixture?)](https://github.com/openclimatefix/nged-substation-forecast/pull/1226).

![Planned contrasts](00020-solar-wind-stack-cams-stage0-spread-margin/figure_1.svg)

### 00018: the staged chain against the single-stage model for Primary demand

**The staged chain, built only from weather that serving has and with NGED's real telemetry delay,
had a quantile score not distinguishable from N1's and a point estimate above
`manual_heuristic`'s.** The experiment gave the stage 2 calibrator lag anomalies from ERA5T, CAMS,
and the ENS control member's early leads, with the newest usable power 7 hours before the 09:00 UTC
origin.

- **Headline result:** the chain minus N1 was +0.08 points of capacity over all leads (99.17%
  interval -0.11 to +0.25), -0.27 up to 48 hours (-0.62 to +0.03). The chain minus
  `manual_heuristic` was +0.52 (-0.25 to +1.25).
- **Absolute scores (all leads):** `manual_heuristic` 5.89, N1 6.33, chain 6.41.
- **What the serving sources cost:** 0.13 points against final ERA5, and 3.5 more hours of telemetry
  delay cost another 0.13. Lag anomalies lowered the score by 0.04 against raw lagged power.
- **Finding for later experiments:** a spread factor of 1.15 lifts the 5% to 95% coverage from 86%
  to 90%, and every arm under-forecast the validation year by 1.3 to 3.5 points.
- **Write-up and pull request:**
  [result](00018-staged-chain-vs-n1-serving-latency-lag-anomalies-demand/result.md),
  [PR #1224 (Experiment 18: does the staged chain beat the single-stage model for Primary
  substation demand?)](https://github.com/openclimatefix/nged-substation-forecast/pull/1224).

![The six planned contrasts](00018-staged-chain-vs-n1-serving-latency-lag-anomalies-demand/figure_2.svg)

### 00014: a stage 0 for the CAMS-trained demand model

**Mapping the ENS shortwave onto CAMS lowered the Primary quantile score of the CAMS-trained stage
1a by 0.067 points of capacity, but the verdict is inconclusive.** The contrast over all leads at
the 16 Primary substations is -0.067 (97.5% interval -0.094 to -0.039), in 12 of 12 months and
entirely in daylight, so the CAMS-trained arm with stage 0 is 0.102 points better than the
ERA5-trained arm. The pre-registered rule also required the corrected shortwave to stay within 5%
of CAMS in every month, and it was up to 18.6% off in December although the year's mean matched
(ratio 1.000). The gain is smallest in the first day, whose interval reaches zero. The mapping
widened the member spread by 27%, against a predicted 10%, and a post hoc monthly mapping did no
better. See [`result.md`](00014-cams-trained-stage1-with-stage0-ens-correction-demand/result.md) and
[PR #1219 (Experiment 14: does a stage 0 that maps ENS irradiance onto CAMS help the CAMS-trained
demand
model?)](https://github.com/openclimatefix/nged-substation-forecast/pull/1219).

![Ratio of ENS daylight shortwave to CAMS](00014-cams-trained-stage1-with-stage0-ens-correction-demand/figure_1.svg)

### 00010: the lagged electricity price in the stage 1a demand model

**Adding the Elexon price from 7 days earlier to the ERA5-trained demand model did not reliably
help at the substation hosting the largest battery, and the experiment could not test a battery
response.** The
price is Elexon's Market Index Data (the `APXMIDP` half-hourly index, free, 98.5% of half-hours
covered from 2019). The comparison arms were demand and wind at the same lag, and a price taken from
an earlier week.

- **Headline result:** at B2, the only listed series with enough history in the selection year, the
  price lowered the error by 0.92 points of capacity (98.3% interval -2.10 to +0.07), and by 0.06
  (-0.38 to +0.23) in the confirmation year. The pre-registered contrast C4, the gain at the listed
  series minus the gain at the others, was -0.55 (-1.43 to +0.20) and then +0.47 (+0.06 to +0.91),
  the wrong sign.
- **Finding for later experiments:** NGED's register lists battery storage of at least 5% of
  capacity at only 2 of the 20 series (B2 and P4), and P4 has too little history for the selection
  year, so this series set cannot test a battery response. The shuffled price from an earlier week
  gave about five sixths of the selection-year gain.
- **Write-up and pull request:** [result](00010-elexon-price-lags-stage1a-demand/result.md),
  [PR #1215 (Experiment 10: lagged electricity price in the stage 1a demand
  model)](https://github.com/openclimatefix/nged-substation-forecast/pull/1215).

![Planned contrasts with 98.3% intervals](00010-elexon-price-lags-stage1a-demand/figure_2.svg)

### 00011: school-term and return-to-work columns in the stage 1a demand model

**The return-to-work and school-term columns did not lower the Primary demand error after Christmas,
at Easter, or over the half-term holidays.** All 20 series lie in a box that is essentially
Lincolnshire, so the experiment used Lincolnshire County Council's recommended term calendar for
2021-22 to 2025-26, collected by hand and validated by script. The base arm was the XGBoost arm of
experiment 00006 with the 14-day Christmas column, trained from 2021-09-01.

- **Headline result:** for Primary substations, the return and term columns together changed the
  error by +0.06 points of capacity from 2 to 20 January (99.2% interval -0.17 to +0.30), +0.20 over
  Easter (-0.19 to +0.59), and -0.29 over the half terms (-0.68 to +0.07), against 7.18 for the base
  arm over the whole year. Both was not 0.2 points better than either calendar shifted by 17 days,
  so C6 failed, and C5 failed because its interval reaches -0.11.
- **Verdict wording:** the rule as pre-registered gives "refuted". Its refutation clause was looser
  than intended. Read as intended, the verdict is inconclusive for Easter and the half terms,
  because the data do not rule out a gain of 0.15 points there.
- **Finding for later experiments:** the base arm's Primary error averages 8.4 points from 2 to 20
  January, and the new columns change it by +0.06. The arm with the full history and the COVID
  indicator scored 0.26 points higher on the whole year than the base arm (99.2% interval 0.12 to
  0.43).
- **Write-up and pull request:**
  [result](00011-school-term-and-return-to-work-columns-stage1a-demand/result.md), [PR #1214
  (Experiment 11: school-term and return-to-work columns for the stage 1a demand
  model)](https://github.com/openclimatefix/nged-substation-forecast/pull/1214).

![Change in error by window and column set](00011-school-term-and-return-to-work-columns-stage1a-demand/figure_1.svg)

### 00017: training window, recency, and retraining rule for demand

**Question.** Does training the stage 1a demand model on recent rows, retraining it monthly, or
correcting it by its last 14 days of error lower the Primary substation demand error, as drift
suggests?

**Verdict: supported.** On 16 Primary substations, with monthly rolling origins from January 2022 to
June 2026 on ERA5 weather, all-history training retrained monthly scores 7.62 points of capacity.
The 14-day correction lowers it by 1.52 points (99.44% interval 1.13 to 1.95), a half-life of 0.5
years by 0.92 (0.64 to 1.22), and a 12-month window by 0.84 (0.53 to 1.19). Retraining yearly
instead
of monthly adds 0.52 points, and a model never retrained adds 3.80.

**What changes later experiments.** The bias of the all-history model runs from forecast too high
(2022 to 2025) to too low (2026), so a window change needs several years of rolling origins to be
judged. The correction stacks with a 0.5-year half-life (5.82 points, post hoc).

![Contrasts against the all-history model](00017-training-window-and-retraining-rule-drift-demand/figure_1.svg)

Details: [result.md](00017-training-window-and-retraining-rule-drift-demand/result.md),
[PR #1223](https://github.com/openclimatefix/nged-substation-forecast/pull/1223).

### 00016: calibration of the pooled mixture for solar and wind

**A scalar spread factor brings the pooled mixture's 5% to 95% band for solar from about 95% to
about 90% coverage, with no change in the CRPS.** The experiment corrected the pooled mixture of
00002's ERA5-trained stage 1a with a spread factor, a spread factor per lead band, a
location-and-scale calibrator, and a free quantile calibrator, each fitted on earlier validation
months, for 6 solar farms and 3 wind farms (row set `fold_a_tuesdays_v1`).

- **Headline result:** uncorrected solar covers 94.3% to 95.3% of non-trivial cases in every lead
  band, and `spread` gives 89.5% to 91.8% with a CRPS change of +0.006 points of capacity (99.4%
  interval -0.004 to +0.024).
- **Wind:** the uncorrected band covers 95.3% and 94.4% in the first two bands and about 90% after
  that. No planned arm brings the first two bands within 2.5 points of 90, and the fit per lead band
  existed for only 3 months of the first band. A post hoc spread factor on two lead groups
  does, with a CRPS change of -0.021 (-0.042 to -0.002), a small gain.
- **Finding for later experiments:** a location-and-scale calibrator per lead band undershoots
  90% in two wind bands (86.7% and 83.9%) and its CRPS interval is 0.36 points wide.
- **Write-up and pull request:** [result](00016-pooled-mixture-calibration-solar-wind/result.md),
  [PR #1221 (Experiment 16: calibration of the pooled mixture for solar and
  wind)](https://github.com/openclimatefix/nged-substation-forecast/pull/1221).

![Coverage of the 5% to 95% band by lead band](00016-pooled-mixture-calibration-solar-wind/figure_1.svg)

### 00019: stacking small gains on N1 for Primary demand

**Stacking a 14-day error correction, a recency half-life, and a spread factor on N1 beat
`manual_heuristic` on the quantile score, and the correction supplied nearly all of the gain.** The
experiment used the 16 Primary substations, ENS weather, and 73 forecast runs.

- **Headline result:** the stack minus `manual_heuristic` was -0.56 points of capacity over all
  leads
  (99.44% interval -0.93 to -0.26), and the stack minus N1 was -1.00 (-1.88 to -0.14).
- **Absolute scores (all leads):** stack 5.32, `manual_heuristic` 5.89, N1 6.33.
- **Components:** dropping the correction from the stack raised the score by 0.97 (+0.15 to +1.81);
  the 0.5-year half-life moved it by -0.01 (-0.03 to +0.02); dropping the spread factor raised it by
  0.03 (+0.005 to +0.05).
- **Finding for later experiments:** N1 forecast the validation year 2.6 points too low, and the
  correction cut that bias to 0.5. The correction was tested on forecasts 5 days apart and on a
  model
  that is never retrained.
- **Write-up and pull request:** [result](00019-primary-demand-stack-vs-manual-heuristic/result.md),
  [PR #1225 (Experiment 19: does stacking small gains beat manual_heuristic for Primary
  demand?)](https://github.com/openclimatefix/nged-substation-forecast/pull/1225).

![The nine planned contrasts](00019-primary-demand-stack-vs-manual-heuristic/figure_2.svg)

### 00015: the same-time electricity price in a stage 1 demand model

**The price at the same half-hour as the target lowered the pooled demand error in the selection
year, but a price from another week did almost as well, and the gain did not repeat in the
confirmation year.** The stage 1 demand model maps the observed price and ERA5 weather at time t to
power at time t
on 17 series (18 in the confirmation year), scored at observed values.

- **Headline result:** price against no price, pooled: -0.52 points of capacity (97.5% interval
  -0.88 to -0.11) in the selection year and -0.18 (-0.50 to +0.08) in the confirmation year. Price
  against a price from another week: -0.07 (-0.17 to +0.03) and -0.06 (-0.18 to +0.07).
- **Finding for later experiments:** national demand and wind at time t did no worse than the price
  in either year (pooled intervals include zero; 0.43 points lower in the confirmation year), so the
  price may be a proxy for national conditions. The gain was largest from 12:00 to 20:00 and at the
  series with the largest errors, and the other-week price reproduced most of it there.
- **Write-up and pull request:** [result](00015-same-time-price-stage1-demand/result.md), [PR #1220
  (Experiment 15: same-time price in a stage 1 demand
  model)](https://github.com/openclimatefix/nged-substation-forecast/pull/1220).

![Pooled contrasts with 97.5% intervals](00015-same-time-price-stage1-demand/figure_1.svg)

### 00023: recent observed power for solar and wind

**Recent power did not help the stage 2 calibrator for solar or wind, and a 14-day correction built
from the mean error made the forecast worse.** The experiment gave the calibrator lagged power and
summaries of the latest day for 6 solar farms and 3 wind farms, and shifted the pooled mixture by
its own mean error over the previous 14 days (row set `fold_a_tuesdays_v1`).

- **Headline result:** lagged power added to the calibrator, solar -0.012 (99.58% interval -0.122 to
  +0.088), wind +0.012 (-0.090 to +0.146). The 14-day correction on the mixture, solar +0.437
  (+0.215 to +0.664), wind +0.413 (-0.003 to +0.924).
- **Finding for later experiments:** a calibrator trained on the ENS control member alone scored
  0.27
  (solar) and 0.31 (wind) points worse than the pooled mixture before any lag was added. Shifting by
  the median error (post hoc) instead of the mean error moved the CRPS by +0.010 (solar) and -0.064
  (wind), both with intervals that include zero.
- **Write-up and pull request:**
  [result](00023-lagged-power-and-error-correction-solar-wind/result.md),
  [PR #1229 (Experiment 23: does recent observed power help solar and wind forecasts?)](https://github.com/openclimatefix/nged-substation-forecast/pull/1229).

![Contrasts](00023-lagged-power-and-error-correction-solar-wind/figure_1.svg)

### 00026: effective-capacity features for solar and wind

**A trailing effective-capacity feature did not lower the CRPS detectably for solar or wind.** The
experiment added the 99th percentile of a farm's power at the 10% of steps with the most CAMS
irradiance (solar) or ERA5 100 m wind (wind) over the last 28 days, read with a 5-day cutoff, to the
stage 1a quantile models of 6 solar farms and 3 wind farms (row set `fold_a_tuesdays_v1`).

- **Headline result:** single feature minus no feature, solar -0.018 (98.75% interval -0.162 to
  +0.096), wind +0.013 (-0.054 to +0.101).
- **Finding for later experiments:** the single feature moved solar farm E's CRPS by -0.235
  points (its shuffled copy by -0.034) and raised it for solar farm A and wind farm W3. Adding the
  wind power-over-wind-cubed feature lowered the wind CRPS by 0.061 (-0.107 to -0.025), an
  exploratory contrast.
- **Write-up and pull request:**
  [result](00026-capacity-features-solar-wind/result.md),
  [PR #1233 (Experiment 26: does a recent effective-capacity feature
  lower the solar and wind forecast error?)](https://github.com/openclimatefix/nged-substation-forecast/pull/1233).

![Contrasts](00026-capacity-features-solar-wind/figure_1.svg)

### 00027: PV_Live at the lag times for GSP demand

**PV_Live at the lag times did not help the stage 2 calibrator for the two grid supply points
(GSPs), Walpole and Bicker Fen.** The experiment gave the calibrator PV_Live's estimate of the GSP's
solar generation at the lag times, as a level, as a share of PV_Live's capacity, and added to the
lagged delta (row set `fold_a_every5d_v1`).

- **Headline result:** the quantile score of the arm with PV_Live minus the arm without, over all
  leads: delta plus PV_Live -0.03 (98.33% interval -0.13 to +0.09), PV_Live level -0.08 (-0.26 to
  +0.08), PV_Live share -0.06 (-0.22 to +0.09). The arm without PV_Live scores 19.04.
- **Finding for later experiments:** PV_Live at the valid time lowered stage 1a's point error at
  Bicker Fen by 1.28 points (-1.90 to -0.67) and not at Walpole (-0.03, -0.27 to +0.23), so a
  PV_Live
  forecast as a stage 1 input is the open question (exploratory; the valid-time value is not known
  at the forecast origin).
- **Write-up and pull request:** [result](00027-pvlive-lags-in-stage2-gsp-demand/result.md),
  [PR #1234 (Experiment 27: does PV_Live at the lag times help the stage 2 forecast of GSP demand?)](https://github.com/openclimatefix/nged-substation-forecast/pull/1234).

![Contrasts](00027-pvlive-lags-in-stage2-gsp-demand/figure_1.svg)

### 00030: lagged power columns for Primary demand

**Adding all 13 lags of `manual_heuristic` to N1 did not lower its quantile score, because the 7
annual analogues hurt and the 14-day correction removes the gain of the 6 weekly ones.** N1 is the
single-stage model; arms added weekly analogues, annual analogues, summaries, recent values, and
same-day-type analogues (row set `fold_a_every5d_v1`, 16 Primary substations).

- **Headline result:** quantile score, in points of capacity. C3, all 13 lags minus N1, -0.01
  (99.375% interval -0.45 to +0.43, seed 0). The weekly analogues alone, -0.30 (-0.63 to -0.02, 3
  seeds); the annual analogues on top, +0.29 (+0.13 to +0.49). With the correction N1 scores 5.34,
  and a post hoc arm with the weekly mean and median scores 5.29 (-0.05 against N1, -0.08 to -0.02).
- **Finding for later experiments:** judge every lag set against N1 with the correction (5.34) and
  the 00019 stack (5.32), not N1 alone (6.33).
- **Cut for time:** 3 seeds for 4 arms only, and the residual-lag arm was not run.
- **Write-up and pull request:** [result](00030-lagged-power-feature-sets-primary-demand/result.md),
  [PR #1237 (Experiment 30: which lagged power columns lower the error of Primary substation demand forecasts)](https://github.com/openclimatefix/nged-substation-forecast/pull/1237).

![Quantile score by arm](00030-lagged-power-feature-sets-primary-demand/figure_1.svg)

### 00028: a solar forecast for the GSPs in the stage 2 calibrator

**A forecast of each grid supply point's (GSP's) PV_Live solar generation did not clearly lower the
stage 2 quantile score for Walpole and Bicker Fen.** A simple XGBoost model, trained on ERA5 weather
and applied to the 51 ENS members, gave the calibrator the member mean and spread of the solar share
at the target time (row set `fold_a_every5d_v1`).

- **Headline result:** solar forecast minus none over all leads -0.14 (95% interval -0.34 to
  +0.04), and the true PV_Live value as an oracle -0.50 (-0.91 to -0.12), mostly at leads of 7 to
  15 days.
- **Finding for later experiments:** the solar forecast beat a climatology up to day 6 (mean
  absolute error 5.0 to 7.0 points of PV_Live capacity against 8.8 to 9.5), and its member mean was
  too high by 1.8 to 3.0 points at every lead band.
- **Write-up and pull request:** [result](00028-pvlive-forecast-in-stage2-gsp-demand/result.md),
  [PR #1235 (Experiment 28: does a GSP solar forecast help the stage 2 demand forecast?)](https://github.com/openclimatefix/nged-substation-forecast/pull/1235).

![Contrasts](00028-pvlive-forecast-in-stage2-gsp-demand/figure_1.svg)

### 00024: which groupings of substations can share one XGBoost model

**Training one XGBoost model per substation type, or per size band, costs 0.15 points of capacity
against one model per substation, and one model across all substations costs 0.29.** The experiment
refitted 00019's model on 20 demand series (8 Primary substations that meter in MW, 8 in MVA, 2 BSP,
2 GSP) with every grouping, on 73 forecast runs with ENS weather.

- **Headline result:** by type with the series identifier minus per series, Primary quantile score
  over all leads: +0.15 points (99.44% interval +0.03 to +0.26); all series in one model: +0.29
  (+0.19 to +0.42). Absolute Primary scores: per series 5.32, by type 5.47, all series 5.61.
- **Finding for later experiments:** the series identifier is essential. Without it the by-type
  pool loses a further 0.82 points and the all-series pool 0.85. On the 16 Primary substations a
  random
  grouping scores the same as the by-type grouping (5.47 each), so the number of models matters
  more than which Primary substations share one.
- **Write-up and pull request:** [result](00024-pooling-groupings-demand-xgboost/result.md),
  [PR #1230 (Experiment 24: which groupings of substations can share one XGBoost model (plan))](https://github.com/openclimatefix/nged-substation-forecast/pull/1230).

![The nine planned contrasts](00024-pooling-groupings-demand-xgboost/figure_1.svg)

### 00032: fingerprints of each substation in a pooled demand model

**Adding location, training-window statistics, or recent-window statistics to the by-type pooled
model changed the Primary quantile score by at most 0.02 points, and the fingerprints made a
substation held out of the pool worse.** The experiment added three fingerprint blocks to 00024's
by-type pool with the series identifier, and refitted each pool without one of 8 Primary
substations to forecast it as a new substation. The pre-registered verdict is inconclusive, because
the lower limits of the four warm contrasts C1 to C4 are all below zero (-0.002 to -0.049).

- **Headline result:** identifier and both fingerprints minus identifier alone, Primary quantile
  score over all leads: +0.02 points (99.375% interval -0.05 to +0.10; absolute 5.49 against 5.47).
  Held-out substation with training-window statistics minus no fingerprint: +1.39 (+0.99 to +1.91).
- **Finding for later experiments:** training-window statistics replace the identifier for series
  seen in training (+0.01 points, upper limit +0.02), but a pool of 7 other series learns them as 7
  identifiers, so they do not carry a new substation. Post hoc, recent-window statistics alone
  lowered the held-out score by 0.23 points (-0.29 to -0.16, 95%, one seed).
- **Write-up and pull request:** [result](00032-substation-fingerprints-pooled-demand/result.md),
  [PR #1241 (Experiment 32: do substation fingerprints improve a pooled demand model?)](https://github.com/openclimatefix/nged-substation-forecast/pull/1241).

![The eight planned contrasts](00032-substation-fingerprints-pooled-demand/figure_1.svg)

### 00033: the best-guess architecture against production, N1, and `manual_heuristic`

**The best guess lowered the quantile score below production's by 4.89 points of capacity for
Primary demand (98.33% interval -5.66 to -4.06), 3.81 for solar (-5.07 to -2.64), and 6.27 for wind
(-7.65 to -5.28).** The experiment compared the architecture that experiments 1 to 32 point to with
the stored production forecast, N1, `manual_heuristic`, `climatology`, six notable ideas, and an
ERA5-weather ceiling, on the quantile score, MAE of the median, twCRPS and Brier score at
`historical_p99`, and the exceedance rates of the 0.90 and 0.95 quantiles. The pre-registered
verdict is supported.

- **Headline result:** Primary demand scores 5.32 against 9.52 for production, 5.89 for
  `manual_heuristic`, and 6.33 for N1. Production's 5% to 95% band covers 10% to 20% of cases, and
  the 14-day correction accounts for 1.38 of the 4.89-point Primary gap.
- **Finding for later experiments:** monthly retraining without the correction scored 5.12 on one
  seed, below the best guess's 5.32 on the same seed, and the correction did not add to it, with the
  cause untested. The twCRPS at `historical_p99` mostly does not separate the arms; solar and wind
  N1 came within 0.1 and 0.3 points of the ERA5-trained best guess. Better weather would lower the
  wind score by 5.46 and the Primary demand score by 0.11.
- **Write-up and pull request:** [result](00033-best-guess-architecture-vs-production/result.md),
  [PR #1242 (Experiment 33: best-guess architecture against production, N1, and manual_heuristic)](https://github.com/openclimatefix/nged-substation-forecast/pull/1242).

![The three decisive contrasts](00033-best-guess-architecture-vs-production/figure_2_decisive.svg)

### 00035: a lead-weighted blend of quantiles from stored models

**Averaging the quantiles of three stored Primary demand models lowered the quantile score by 0.10
points; solar and wind gains were too small to resolve.** The experiment refitted nothing. It
averaged each quantile level of two or three stored models per technology, with weights chosen
only on earlier months, and scored the result on the cases of experiment 33. The verdict is
supported, on Primary demand alone.

- **Headline result:** the lead-band blend of the monthly-retrained N1, `a7_cs`, and the stack
  minus the monthly-retrained N1 (the best single member, chosen with hindsight), Primary quantile
  score over all leads: -0.097 points (98.33%
  interval -0.187 to -0.026; absolute 5.025 against 5.122). Solar -0.039 (-0.115 to +0.031) and
  wind -0.074 (-0.194 to +0.055) are null.
- **Finding for later experiments:** one fixed weight vector scores as well as weights per lead
  band (+0.005, -0.008 to +0.023 for Primary demand), and the blend scored 0.048 points worse when
  the
  14-day correction was applied afterwards instead of to each member first (+0.008 to +0.101, 95%).
- **Write-up and pull request:**
  [result](00035-lead-weighted-quantile-blend-of-stored-models/result.md),
  [PR #1244 (Experiment 35: does a lead-weighted blend of quantiles beat the best single model?)](https://github.com/openclimatefix/nged-substation-forecast/pull/1244).

![Blends minus the best single member](00035-lead-weighted-quantile-blend-of-stored-models/figure_1.svg)

### 00036: forecasts of the series beneath a BSP or GSP

**Adding forecasts of the series beneath a bulk supply point (BSP) did not lower its quantile
score, and the ERA5-trained N1 gained 0.36 to 0.40 points of capacity at the grid supply points
(GSPs) with intervals that include zero.** The experiment gave the N1 model of Boston, Skegness,
Walpole, and Bicker Fen the summed median forecasts of the Primary substations and metered
generators beneath, a regression on those forecasts, an N1 trained on ERA5, and wind and solar
columns. The pre-registered verdict is inconclusive.

- **Headline result:** B1 minus B0 over all leads is +0.04 (99.17% interval -0.18 to +0.25) at
  Boston and +0.19 (-0.10 to +0.73) at Skegness, and B3 minus B0 is -0.36 (-1.60 to +0.43) at
  Walpole and -0.40 (-1.84 to +0.97) at Bicker Fen. An oracle that sees the observed children scores
  5.58 and 3.42 points below N1 at the two BSPs.
- **Finding for later experiments:** the oracle's gain is not headroom for a forecast of the
  children, because the forecast columns gained nothing. The ERA5-trained N1 gained 0.41 points
  over all leads (95% interval 0.15 to 0.66, exploratory) at Boston, and 0.66 to 0.88 points up to
  day 6.
- **Write-up and pull request:** [result](00036-hierarchy-aware-bsp-gsp-forecasts/result.md),
  [PR #1245 (Experiment 36: do forecasts of the series beneath a BSP or GSP lower its error?)](https://github.com/openclimatefix/nged-substation-forecast/pull/1245).

![The planned contrasts and B3 at the BSPs](00036-hierarchy-aware-bsp-gsp-forecasts/figure_1.svg)

### 00037: logged switching in the telemetry

**A detector that checks neighbouring substations recovered a third of the MW of the 18 visible
validation transfers, but the verdict is inconclusive.** The experiment turned the NGED switching
logs into 612 closures of 2 hours to 60 days, found that 37 of them show a visible step (about 6
by chance), and scored two detectors that never read the logs against them, with thresholds set on
odd months and recall measured on even months.

- **Headline result:** the neighbour-balance detector D1 (4-hour window, threshold 4) reaches a
  size-weighted recall of 0.326 (95% interval 0.093 to 0.579) at 0.38 unexplained alarms per
  series-month, against 0.036 (0.000 to 0.131) for the single-series detector D0 at 0.14. Shifted
  logs give 0.004. D1 misses the 0.35 bar and clears the 0.20 bar, so the verdict is inconclusive.
- **Finding for later experiments:** at about 0.5 alarms per series-month D0 reaches 0.224 and D1
  0.362, so the pre-registered grid overstated the gap. A 24-hour D0 reaches 0.276. The stack's
  threshold-weighted score share in confirmed half-hours is 40%, but one 370-hour closure on series
  18 holds nearly all of it.
- **Write-up and pull request:** [result](00037-switching-first-look/result.md),
  [PR #1246 (Experiment 37: do logged switching closures show up in the telemetry, and does a
  neighbour-balance detector beat a plain one?)](https://github.com/openclimatefix/nged-substation-forecast/pull/1246).

![Validation recall against unexplained alarms](00037-switching-first-look/figure_1_headline.svg)

### 00038: the value of detecting switching events

**Leaving the half-hours that the neighbour-balance detector flags out of the stack's error
correction raised the forecast's tail error, so the verdict is refuted.** The experiment masked
detected intervals from the 14-day and 7-day corrections of the proxy stack (experiment 00017's
ERA5-weather baseline) and of experiment 00019's real stack, and scored half-hours outside logged
closures against an oracle mask from the logs and a random mask of equal volume.

- **Headline result:** the primary contrast, D1 mask with a 7-day window against the plain 7-day
  correction, is +57.1% (95% interval +0.4% to +133.5%) in threshold-weighted error at each series'
  P99. The oracle mask lowers the 14-day error by 8.9% (-21.8% to +0.0%), and 95% of that gain
  sits in January 2026, the month of the 17-18 closure. The random mask changes it by +0.1%.
- **Finding for later experiments:** a post hoc revision of the interval builder gives -5.4%
  (-15.1% to +1.5%), but raises the mean absolute error by 0.25 points, and 3 of 17 long closures
  improve. The plain 7-day window beats the 14-day window by 5.2% (-10.8% to +0.0%) in the proxy
  stack but loses by 8.5% (+2.4% to +10.7%) in the real stack.
- **Write-up and pull request:** [result](00038-switching-mask-value/result.md),
  [PR #1247 (Experiment 38: is detecting a switching event worth anything to the forecast?)](https://github.com/openclimatefix/nged-substation-forecast/pull/1247).

![Per-event and aggregate change in the threshold-weighted error](00038-switching-mask-value/figure_1_headline.svg)

### 00039: separating embedded solar from Primary substation demand

**A fit of a calendar-plus-temperature demand term minus one CAMS-driven PV shape recovers injected
solar-farm output to within 25%, and the real series pass four checks at 6 of 8 Primaries, so the
pre-registered verdict is `recovers-but-confounded`.** The experiment injected the output of 6
metered solar farms, scaled to 20% of each Primary's effective capacity, into the demand of the 8
directional Primary substations (218 units), and ran four checks (placebo sky, season, cloud-demand
term, day type) on the real series.

- **Headline result:** the half-hourly fit recovers a median 0.773 of the injected capacity (95%
  interval 0.740 to 0.818). The daily fit recovers 0.854 (0.815 to 0.866), which passes the
  recovery rule. In 2025/26 Kirton fails the season and cloud-term checks, and Wrangle T1 fails
  the placebo check by a hair, so the placebo check passes at 7 of 8 Primaries and all four at 6.
- **Finding for later experiments:** injection cannot detect a demand confound, because a linear fit
  returns the same increment on any demand. The fitted capacity is 0.9 to 3.2 MW per Primary
  before dividing by the recovery ratio. The next experiment is PVDIS-2B, a richer demand model.
- **Write-up and pull request:** [result](00039-pv-disaggregation-cams-injection/result.md),
  [PR #1248 (Experiment 39: can a CAMS-driven fit separate embedded solar from Primary substation demand?)](https://github.com/openclimatefix/nged-substation-forecast/pull/1248).

![Median recovery ratio by model and irradiance point](00039-pv-disaggregation-cams-injection/figure_01.svg)

### 00040: how far the capacity fitted to embedded solar can be trusted

**A cross-arm misfit test is calibrated and detects four of five cloud-driven loads at 25% of the
true capacity, but it detects a day-long cloud load in only 0.35 of draws, so the verdict is
`underpowered`.** The test compares the capacity that six least-squares fits assign to the same
series, using known injected solar-farm output and eight known loads in the real series of the 8
directional Primary substations.

- **Headline result:** pure injected solar output gives a median capacity over truth of 0.992 (95%
  interval 0.951 to 1.032) and a false-positive rate of 0.055. Kirton is flagged in 6 of 7
  season-years and no other Primary is flagged in any. The bound on the capacity at the other seven
  is 0.18 to 0.23 at five, 0.47 at Trusthorpe, and 0.71 at Wrangle T1, where the test has almost
  no power.
- **Finding for later experiments:** the two loads that the misfit test cannot see, a day-long cloud
  load and a load that rises with sunshine, set the bound. Evening demand after dark responds to
  cloud at some Primaries and offers a measurement of the first.
- **Write-up and pull request:** [result](00040-pv-capacity-error-bars/result.md),
  [PR #1249 (Experiment 40: how far can the capacity fitted to embedded solar be trusted?)](https://github.com/openclimatefix/nged-substation-forecast/pull/1249).

![Capacity per Primary in 2025/26 with its interval and bound](00040-pv-capacity-error-bars/figure_01.svg)

### 00034: how often to retrain the Primary demand model, and which cheap proxy stands in

**Refitting the Primary demand model monthly lowered the quantile score from 5.32 to 5.12 points of
capacity, refitting weekly lowered it by a further 0.07, and the 14-day correction stopped
helping.** The experiment refitted N1 for 16 Primary substations at six cadences and scored 73
forecast runs from July 2025 to June 2026. The pre-registered verdict is inconclusive: the
99.375% interval on monthly minus the stack is -0.44 to +0.08.

- **Headline result:** monthly retraining minus the stack, quantile score over all leads: -0.20
  points (99.375% interval -0.44 to +0.08; 95% interval -0.38 to -0.01). Weekly minus monthly:
  -0.07 (-0.18 to -0.00). The correction at weekly cadence: +0.23 (+0.03 to +0.44).
- **Finding for later experiments:** of four cheap proxies for monthly retraining, only monthly
  refits on a rolling 12-month window (150 rounds, `eta` 0.15) reproduced full monthly
  retraining's contrasts, at about a third of the GPU time. A frozen reference overstates a
  feature that follows recent demand level (weekly analogues: -0.05 frozen, -0.01 retrained).
- **Write-up and pull request:** [result](00034-retraining-cadence-and-cheap-proxies-demand/result.md),
  [PR #1243 (Experiment 34: retraining cadence and cheap proxies for Primary demand (plan))](https://github.com/openclimatefix/nged-substation-forecast/pull/1243).

![Difference from the stack by retraining cadence](00034-retraining-cadence-and-cheap-proxies-demand/figure_1.svg)

### 00041: a labeller for logged switching events, gated on known answers

**A Sonnet labeller called no placebo card visible but found 0.74 of large, long injected
transfers (25 of 34; 0.57 to 0.85), under the 0.80 bar, so the verdict is partly supported.** On
300 known-answer cards it called 0 of 120 placebo cards visible (95% interval 0.00 to 0.03). Where
a metered receiver exists the recall is 0.89 (46 of 52; 0.77 to 0.95), and where only the donor
steps it is 0.60. A rules-only labeller on the same features reaches 0.65 with 3 of 120 placebo
cards visible. The receivers-named bar was waived by the maintainer after the rules-only results.

- **Finding for later experiments:** Sonnet reaches 0.80 only for transfers of 25% or more of the
  receiving load lasting over 1 day, so detector scoring treats every other band as don't-care.
- **Process finding:** the first test set's injected cards differed from placebo cards in the
  minute of the log time, the sheet's sides, and the clock time, and a reviewer caught it; the
  gate was rerun on a rebuilt set. SW2-0b needs the same check on real cards against sentinels.
- **Cost:** 14.9 million subagent tokens for 1,156 card labellings, about 12,900 per card, with no
  Opus run.
- **Agreement:** the two Sonnet labellers agree with kappa 0.79.
- **Write-up and pull request:** [result](00041-sw2-0a-labeller-gate/result.md),
  [PR #1253 (Experiment 41: SW2-0a switching-event labeller and its known-answer gate (plan))](https://github.com/openclimatefix/nged-substation-forecast/pull/1253).

![Recall of injected transfers by size and duration band](00041-sw2-0a-labeller-gate/figure_2_recall_by_band.svg)

### 00041, part 2 (SW2-0b): the labelled set of real switching jobs

**Of 537 logged switching jobs of 2 hours to 60 days, 66 (12%) show a visible transfer of load
between substations, and 9 of 148 detector alarms with no logged closure do.** Two independent
Sonnet runs labelled 1,017 summary cards (817 real, 200 hidden known-answer cards) and Opus settled
66 disagreements. No hidden placebo card (0 of 100) and no quiet alarm window (0 of 40) was called
visible. Few of the 440 not-visible jobs are proven quiet, because Sonnet misses transfers under 25%
of the receiving load and under 1 day, and 13 of the 35 conserved calls rest on one substation
with mostly missing readings.

- **Finding for later experiments:** score detectors against the 82 visible cards, and treat the
  331 not-visible operations of under 1 day as don't-care.
- **Cost:** 20.1 million subagent tokens, a third above the estimate of 15 million.
- **Write-up:** [labelled set](00041-sw2-0a-labeller-gate/labelled-set.md).

![Final labels by card kind](00041-sw2-0a-labeller-gate/figure_4_labelled_counts.svg)

### 00043: how well do the two existing switching detectors find transfers?

**On 600 injected transfers, the neighbour-balance detector D1 finds fewer large, long transfers
than the single-series detector D0 over a 24-hour window, so the hypothesis is refuted there.** D0
finds 47 of 160 (0.29) and D1 finds 34 (0.21); D1 minus D0 is -0.081 (95% interval -0.130 to
-0.037). Over a 4-hour window D1 finds 40 and D0 26 (+0.087, +0.044 to +0.136). A detector counts as
finding a transfer only when its start is within 2 hours and its end within 6 hours or 10% of the
length.

- **Finding for later experiments:** both detectors find 0.08 to 0.17 of all injected transfers, and
  D1 finds almost none of the 84 headline transfers that have no metered receiver (5 of 84 at the
  4-hour window, 2 of 84 at 24 hours).
- **Precision:** about half of the alarms match a logged closure; precision is 0.42 to 0.57 whatever
  the setting.
- **Write-up and pull request:** [result](00043-sw2-1-detector-scoreboard/result.md),
  [PR #1262 (Experiment 43: how well do the two existing switching detectors find transfers?)](https://github.com/openclimatefix/nged-substation-forecast/pull/1262).

![Share of injected transfers found, by band](00043-sw2-1-detector-scoreboard/figure_2_recall_by_band.svg)

### 00044: how much labelled switching events hurt the best guess

**A perfect switching detector would lower experiment 33's best guess's quantile score for Primary
demand by 0.42% (95% interval -0.47% to 1.96%), under the plan's 3% threshold, so the verdict is
refuted.** The experiment fitted nothing. It split the best guess's scores for the 16 Primary
substations from July 2025 to June 2026 by whether the target half-hour, or its lagged power (the
earlier power readings the forecast takes as inputs), lies inside a labelled switching event of 25%
or more of the receiving load lasting over 24 hours.

- **Headline result:** flagged half-hours are 1.35% (0.04% to 3.22%) of the half-hours scored and
  carry 3.4% (0.04% to 8.68%) of the summed quantile score, 2.5 times their share of half-hours.
  Only 6 such events fall in the year, on 3 series, so the intervals are wide and say little about
  the other 13 series.
- **Sensitivity:** with every visible event, whatever its size or length, the quantile-score bound
  is 2.35% (-0.22% to 5.73%). The flagged half-hours carry 55% (2% to 70%) of the tail score, which
  rests on a few short events.
- **Finding for later experiments:** the stored `moved_share_of_receiver` of the labelled set is
  missing (NaN) for 62 of the 81 labelled events that touch a Primary series, partly from series
  ids written as strings, so scores against it need the share recomputed.
- **Write-up and pull request:** [result](00044-sw2-2-switching-share-of-best-guess-score/result.md),
  [PR #1261](https://github.com/openclimatefix/nged-substation-forecast/pull/1261).

![Shares of half-hours and of each score](00044-sw2-2-switching-share-of-best-guess-score/figure_2_shares.svg)

### 00046: how much of the best guess's daylight error is embedded solar? (PVDIS part 3)

**We asked how much of a Primary substation's daylight forecast error comes from solar panels behind
its meter, by adding the panels' capacity times the sunshine anomaly (sunnier or cloudier than the
average for that month and half-hour) to a demand forecast without radiation inputs.** Built from the
true sunshine, which no real forecast has, the correction lowers the quantile score (the
probabilistic forecast error score) by 0.966 points (-1.431 to -0.595), about 20% of the 4.724
points the demand forecast scores. Built from the ECMWF ensemble forecast of sunshine, it lowers the
score by 0.056 points (97.5% interval -0.191 to +0.064), against a threshold of 0.35, so the verdict
is `refuted`. The ensemble anomaly tracks the true one for about 2 days and then fades, and the gain
follows it: -0.262 at 0 to 24 hours, -0.277 at 1 to 2 days, and about 0 beyond.

- **Finding for later experiments:** the sunshine correction built from the ensemble forecast scored
  +0.048 (-0.059 to +0.141) against the best-guess recipe with its radiation inputs, so it is no
  better. It is also not shown to be no worse, because the sign changes by month and by series. The
  next experiment in the series (PVDIS-5, which retests the correction with weekly capacity
  estimates) is a non-inferiority and explanation test with a margin set in advance, with a gain
  test only for 0 to 48 hours.
- **Coefficient:** a fixed coefficient of 1 was too large, because the residual's slope on the
  oracle correction is -0.771. A coefficient fitted on the training window gives -0.143 (-0.231 to
  -0.057, exploratory) against the plain correction.
- **Fragile small gain:** without one series (anonymised as D9) the correction gains -0.113 (-0.230
  to -0.000, rounded; the interval touches zero), and it is positive in June to August, so it is not
  a consistent gain. Its lower bound stays above -0.35 in every pooled scope.
- **Exploratory midday read:** at 09:00 to 15:00 UTC on day+1 the correction gains -0.434 (-0.559 to
  -0.283) in April to September and scores no worse than the best-guess recipe, and the quantiles
  move from too narrow to slightly too wide. In June to August at day+2 and day+3 it scores worse
  than the best-guess recipe. Whether that justifies porting is not decided.
- **Write-up:** [result](00046-embedded-solar-share-of-primary-forecast-error/result.md).

![The planned contrasts and the fitted-coefficient reads](00046-embedded-solar-share-of-primary-forecast-error/figure_1_contrasts.svg)

### 00050: does a second weather forecast beat ENS at the solar anomaly? (PVDIS part 8)

**We asked whether the American GEFS ensemble forecast, alone or blended with the ECMWF ensemble
(ENS), predicts the sunshine anomaly (sunnier or cloudier than usual) better than ENS, because
experiment 00046 found that the sunshine forecast limits the gain from a solar correction.** In
cross-validation on the training window, GEFS made 10.9% more squared error against the satellite-
based CAMS anomaly than ENS did, and was worse in all 9 held-out months. The fitted blend made 0.2%
more, against the 1% less the gate needed, so the verdict is `refuted` and the demand contrasts were
not run. On the validation year GEFS correlates with the CAMS anomaly 0.072 less than ENS at 0 to 24
hours (95% interval -0.100 to -0.039), and the blend 0.002 more (-0.001 to +0.006).

- **Finding for later experiments:** a second global ensemble shares ENS's errors, so a gain inside
  2 days needs a product of a different kind, such as a high-resolution regional model or a
  satellite-based nowcast. The fitted blend gives GEFS about 0.1 weight inside 2 days and none from
  2 to 10 days.
- **Unverified:** the 00 UTC GEFS run's availability by 09:00 UTC was not observed in that window,
  so the same-day run, the optimistic case for GEFS, was assumed.
- **Write-up:** [result](00050-second-weather-forecast-for-solar-anomaly/result.md).

![Correlation with the CAMS anomaly, GEFS and the blend minus ENS](00050-second-weather-forecast-for-solar-anomaly/figure_1_headline.svg)
