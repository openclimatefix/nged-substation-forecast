---
id: "00039"
slug: pv-disaggregation-cams-injection
status: planned
size: medium
hypothesis: >-
  On the 8 directional (MW) Primary substations, a calendar-plus-temperature demand term minus one
  PV shape driven by Copernicus Atmosphere Monitoring Service (CAMS) irradiance recovers the P99
  capacity of real solar-farm output injected into the real demand series to within 25%, and the
  same fits on the un-injected series pass four checks that a weather-correlated demand confound
  would fail.
prediction: >-
  Median capacity ratio R (recovered over injected P99) for M1 with CAMS at the substation is 0.77
  (95% interval 0.70 to 0.83). The daily fit M1d raises it by 0.08, CAMS at the farm raises it by
  0.08, and the richer demand term M1L lowers it by 0.04. Half-hourly nMAE is 0.12. On the real
  series K1 passes at 8 of 8 Primaries in 2025/26, K2 at 7 of 8, K3 fails at Kirton and probably
  at Wrangle T1, and K4 is unknown. The expected verdict is `separates`, narrowly. The scratch
  numbers behind this prediction are exploratory and were not blind.
decision_rule: >-
  Four planned contrasts on medians over (Primary, farm, season-year) units, with 95% two-way
  cluster-bootstrap intervals (Primaries and farms resampled independently). C1 is R of M1 with
  CAMS at the substation (the headline). C2 is R of M1d minus C1. C3 is R of M1 with CAMS at the
  farm minus C1. C4 is R of M1L minus C1. M1d replaces M1 when C2 is at least +0.05 with an
  interval above zero, because M1d costs nothing extra. On the better of M1 and M1d, recovery
  passes when the median R is in [0.75, 1.25], its 95% interval lies inside [0.65, 1.35], and the
  median nMAE is at most 0.15. The real checks pass when at least 6 of the 8 Primaries pass K1 to
  K4 in their latest season-year and K1 passes at all 8. Verdicts: `separates` (both pass);
  `recovers-but-confounded` (recovery passes, real checks fail); `biased-correctable` (median R in
  [0.5, 0.75) or (1.25, 1.5], 10th to 90th percentile spread under 0.20); `fails` (median R under
  0.5 or nMAE over 0.25 at both irradiance points). A verdict never rests on a breakdown. The
  experiment stops without a verdict if the known-answer self-tests or the timestamp scan fail.
  There is no cost-of-porting threshold, because the aim is explanation and capacity estimation, and
  not a production forecast column.
benchmark: >-
  None. The scored quantity is recovery of a known injected capacity. The ±25% band is the
  maintainer's ranking prediction HD1, fixed before any of these fits.
row_set_id: none
metric: >-
  Capacity ratio R (recovered over injected P99), half-hourly nMAE of the recovered PV series as a
  share of the injected P99, and the share of Primary-years passing each real check
verdict: pending
production_status: not-for-production
adopted_in: none
code: ml-experiments/00039-pv-disaggregation-cams-injection/
---

# Experiment 39: can a CAMS-driven fit separate embedded solar from Primary substation demand?

**Planned; no code has run.** This page is committed before any fit so the commit order shows that
the prediction and the decision rule came first.

**Question.** The aim is explanation and capacity estimation: how many megawatts of embedded
solar sit behind each Primary substation, and what they generated at each half-hour. It is not
forecast skill. Capacity throughout means the 99th percentile (P99) of the PV component's own
half-hourly output.

**Why two halves.** For a linear estimator, synthetic injection cannot detect a demand confound:
the capacity fitted to `demand + injection` minus the capacity fitted to `demand` equals the
capacity fitted to the injection alone. Real demand therefore sets only the noise of the recovery.
Injection tests the PV shape and the irradiance. Four checks on the real series (K1 to K4) test
for weather-correlated demand, the failure
[Hensley and Bull (2018)](https://doi.org/10.3847/1538-4357/aaa489) warn yields a low residual and
a biased component.

## Size

- **Stored data:** a private work folder and one committed `result.md`; nothing enters a Delta
  table. Simple on this trigger.
- **Production serving path:** not touched.
- **Degradation rule:** not touched.
- **More than one defensible design:** yes (half-hourly or daily fit, irradiance point, demand
  richness), so those are the planned arms C1 to C4.
- **Code whose callers cannot be named without searching:** no. The scripts call only
  `studies.*`.
- **Size: medium.** Sonnet implements, one Sonnet reviewer reads the code for leakage, the
  injection arithmetic and the four checks, Sonnet runs and writes up, and the manager's Opus agent
  reviews the result.

## Design

- **Series.** The 8 directional MW Primaries decide the verdict. The 8 MVA Primaries and the
  Boston and Skegness bulk supply points are descriptive only. Solar farms A to F (anonymised) are
  the injected sources, each normalised by its own P99, with dropouts (power at or below zero under
  GHI above 100 W/m²) removed.
- **Weather source at lag times.** There are no lags. The fit uses CAMS irradiance and ERA5
  temperature at the same half-hours as the power. Neither is needed at forecast time, because this
  is an estimation experiment on past data. CAMS arrives with a delay, so a live use would run on a
  trailing window.
- **Models.** M1: daylight half-hours, `power = cell(month, local half-hour, day type) + b * HDD -
  c * u`, ordinary least squares within cells, day-clustered standard errors. M1d: the same on daily
  daylight sums. M1L: M1 plus a flat cloud term `1 - clear-sky index`.
- **Injection.** 8 Primaries x 6 farms x season-years x `k` in {0.05, 0.10, 0.20, 0.40} of the
  Primary's effective capacity x 2 irradiance points x 3 models. Known-answer self-tests run first
  and stop the script on breach.
- **Real checks.** K1 placebo sky, K2 season consistency, K3 demand richness, K4 day-type
  consistency, with the thresholds in the decision rule above and the design notes.
- **Figures.** 19, raw data first; Solar farm A to F labels; no generator named.

## What the next experiment should answer

To be written from the verdict, by the design's rule: `biased-correctable` leads to removing the
attenuation, `recovers-but-confounded` to enriching the demand term until `c` stops moving, and
`separates` to a capacity that grows over 2019 to 2026. This experiment does not start it.
