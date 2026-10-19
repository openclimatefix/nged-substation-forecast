---
id: "00102"
slug: simp1-simplifying-the-bg2-best-guess-before-production
theme: production-reference-and-best-guess
also_relevant_to:
  - lagged-power-and-error-correction
  - generator-forecasting
  - solar-disaggregation
status: done
size: complex
hypothesis: >-
  A single weekly-refitted XGBoost model without the three power lags, the solar offset, the 14-day
  correction, or the blend scores within 0.15 points of capacity of BG2 at the 16 Primary
  substations, within 0.30 at the 4 bulk and grid supply points, within 0.10 at the 6 solar farms, and
  within 0.30 at the 3 wind farms. We expect the Primary claim to fail and the other three to be
  close calls (see the prediction). Separately, the solar offset helps most where it should, at midday
  in summer at the 5 Primary substations that have one.
prediction: >-
  Dropping the three power lags raises the score by 0.05 to 0.30 points at Primaries, 0.10 to 0.50 at
  supply points, 0.10 to 0.40 at solar farms, and 0.10 to 0.50 at wind farms, because earlier
  experiments found lagged power helps most in the first day. The simplest Primary candidate scores
  0.20 to 0.40 above BG2 (the weekly model alone is already 0.095 above, experiment 87), so we expect
  C1 to fail its 0.15 threshold; the ranges for supply points and wind straddle their 0.30
  thresholds, so C2 and C4 are genuinely open. The best
  2-member blend is within 0.05 of BG2. The solar offset lowers the score by 0.05 to 0.15 at the 5
  offset Primaries in midday summer rows, and by less than 0.01 at night.
decision_rule: >-
  Five verdict contrasts, each read at 99% (5% divided by five, Bonferroni), whole origin months
  resampled, 20,000 resamples. C1 to C4 are the simplest candidate of each target class minus BG2 on
  the same runs. The simplest candidate is, for Primary, the single weekly-refitted N1 without lags,
  offset, correction, or blend and with BG2's spread factor of 1.15; for supply points, the weekly N1
  without lags and with the spread factor of 1.15 (as in BG2's `b_weekly`, which is weekly); for
  solar and wind farms, BG2's ENS-trained N1 without lags, refitted monthly and with no spread
  factor, like the stored arm `g_ens_weekly`, which experiment 87 also refitted monthly. A
  simplification is safe to take to production when the interval's upper end lies below the
  threshold: +0.15 (Primary), +0.30 (supply points), +0.10 (solar farms), +0.30 (wind farms). The
  Primary and supply point thresholds are about half of BG2's own gain over
  the stack in experiment 87 (0.283 and 0.634), because a simplification that gives back half of a
  gain has not earned its place; the farm thresholds are experiment 87's non-inferiority margins. C5
  is BG2 minus BG2 without the offset (stored arms p_s6_bg2 and p_s5_blend_no_offset) on the 5 offset
  Primaries, for valid times from 10:00 to 14:00 UK time in April to September; the offset helps when
  the interval lies wholly below zero (threshold zero: the offset is already built). The overall
  verdict is `supported` when C1 to C4 are all safe and C5 helps, `partly supported` when at least
  one holds, and `refuted` when none does. Every other arm, every single-lag or single-part
  simplification, and every comparison with `manual_heuristic` is read at 95% and decides no
  verdict; each is labelled safe when its upper end lies below its class's threshold.
benchmark: >-
  BG2 (experiment 87: `p_s6_bg2`, `b_weekly`, `g_ens_weekly`), `manual_heuristic`, the stack
  (`p_s2_stack`, `b_stack`), `climatology`, and the production-style `xgboost_baseline`, all stored.
row_set_id: fold_a_daily_gap_v1
metric: >-
  Quantile score (twice the mass-weighted mean pinball loss) in points of capacity over 11 levels
  (0.01, 0.05 to 0.95, 0.99). Reported and deciding nothing: mean absolute error of the median, and
  coverage and pinball loss at 0.01, 0.05, 0.95, and 0.99.
verdict: partly supported
best_arm: >-
  The lag-free forecast at wind farms (10.973 points of capacity on the quantile score, against 11.032
  for BG2) and, at Primary substations, the BG2 blend cut to three models (weekly, analogue, frozen
  with correction), 5.258 against 5.260.
headline: >-
  Removing the three recent-power inputs cost 0.806 points at the 16 Primary substations (99%
  interval 0.367 to 1.569) and 0.278 at the 4 supply points (0.040 to 0.720), and it changed the solar
  and wind farm scores by +0.010 (-0.107 to +0.110) and -0.059 (-0.184 to +0.058). Two Primary
  substations carry most of the Primary cost. The solar offset lowered the score by 0.184 points
  (-0.297 to -0.103) at midday in summer at its 5 substations.
lessons: >-
  The recent-power inputs are the one part of BG2 that cannot be removed at Primary and supply point
  models, and two of the 16 Primary substations account for most of that. The blend, the offset, the
  14-day correction, and the spread factor each add 0.1 points or less over the whole year, and a
  blend of three models matches the five. The farm models do not need the power inputs.
production_status: candidate
adopted_in: none
arms: >-
  Quantile score over 11 levels on the 292 runs, in points of capacity, with 95% intervals. Primary:
  BG2 5.260 (4.987 to 5.534); BG2 blend of weekly, analogue, and frozen with correction 5.258;
  analogue and frozen pair 5.276; BG2 without the 14-day correction 5.320; weekly model with lags
  5.355; monthly model with lags 5.478; weekly model without lags and offset (simplest candidate)
  6.066 (5.483 to 6.811); monthly model without lags 6.381; stack 5.543; `manual_heuristic` 6.182;
  `climatology` 7.368. Supply points: BG2 12.925 (11.569 to 14.177); weekly without lags 13.203
  (11.768 to 14.584); monthly without lags 13.379; stack 13.559; `manual_heuristic` 14.506. Solar
  farms (monthly refits): BG2 5.012 (3.788 to 6.392); without lags 5.022 (3.799 to 6.392);
  `manual_heuristic` 5.718. Wind farms (monthly refits): BG2 11.032 (9.641 to 12.303); without lags
  10.973 (9.592 to 12.250); `manual_heuristic` 13.906.
advice:
  - >-
    Keep the three recent-power inputs in the Primary and supply point models, because removing them
    raised the Primary score by 0.806 points (99% interval 0.367 to 1.569) and the supply point score
    by 0.278 (0.040 to 0.720) (C1, C2).
  - >-
    Find out why Primary K and Primary M lose 3.9 to 6.5 points without lags, before designing any
    input that tracks a substation's level, because the other 14 Primaries lose 0.17 points (99%
    interval 0.08 to 0.30) (lag cost by substation).
  - >-
    Drop the power lags from the solar and wind farm models only after a check on more than 6 solar
    and 3 wind farms, because removing them changed the scores by +0.010 (-0.107 to +0.110) and -0.059
    (-0.184 to +0.058) and one solar farm decides whether the solar limit of 0.10 holds (C3, C4,
    leave-one-farm-out).
  - >-
    Serve a blend of three models (weekly, analogue, frozen with correction) in place of the blend of
    five, because the three-model blend differs from BG2 by -0.002 points (-0.006 to 0.000); the
    choice of the three was made on a disjoint set of runs (blend subsets).
  - >-
    Test the solar offset on midday summer rows at the substations that have one, because it lowers
    the score by 0.184 points (-0.297 to -0.103) there and by 0.010 over all rows (C5).
  - >-
    Keep the 14-day correction in the frozen model, because removing it raises the Primary score by
    0.060 points (0.021 to 0.094) (BG2 without the 14-day correction).
take_home_message: >-
  Removing the three recent-power inputs from the best-guess forecast makes it clearly worse at
  Primary substations (0.8 points of capacity, about 13%) and at supply points (0.3 points), and
  almost all of the Primary loss comes from 2 of the 16 substations. The farm forecasts do not need
  those inputs. The blend of 5 models can shrink to 3, and the solar offset earns its place at midday
  in summer, where it lowers the score by 0.18 points at the 5 substations that use it.
code: ml-experiments/00102-simp1-simplifying-the-bg2-best-guess-before-production/
---

# Experiment 102: SIMP1, which parts of the BG2 best guess can be removed before production

**Take-home message: the three recent-power inputs are the one part of the best guess that cannot be
removed at Primary substations and supply points, and two of the 16 Primary substations account for
most of that cost.** A Primary substation steps 33 kV or 66 kV down to 11 kV, and a supply point is a
bulk supply point (BSP) or grid supply point (GSP), where the transmission network feeds the
distribution network. BG2 is the best-guess forecast of experiment 87. The score is the quantile
score in points of capacity, where one point is 1% of a series' capacity and lower is better. The
simplest Primary forecast, a single weekly-refitted XGBoost model without the power inputs, the solar
offset, the 14-day correction, or the blend, scores 6.066 against BG2's 5.260. The loss is 0.806
points (99% interval 0.367 to 1.569), far above the 0.15 that would count as safe, and the forecast is
no better than `manual_heuristic`, NGED's rule of thumb, at 6.182. The lag result is driven by two
series: without Primary K and Primary M, the loss falls to 0.173 points (0.081 to 0.295). The verdict
is `partly supported`: 2 of the 5 pre-registered contrasts hold.

We asked which parts of BG2 earn their place in production. We removed each part in turn from the
best guess (the blend of five models, the three recent-power inputs, the solar offset, the 14-day
correction, the spread factor of 1.15), scored each result against BG2 on the same 292 daily
forecast runs, and also scored the solar offset on midday summer rows. The blend, offset,
correction, and spread factor each cost 0.1 points or less when removed, and a blend of weekly,
analogue, and frozen models matches BG2's five models. At the farms the power inputs make no
measurable difference: -0.059 points at wind farms and +0.010 at solar farms.

![Each simplification against BG2 at Primary substations](figure_1a_primary_contrasts.svg)

![Cost of removing the power inputs at each Primary substation](figure_7_lag_cost_by_series.svg)

![Primary substations at midday in summer, at night, and in other hours](figure_3_midday_summer.svg)

![Each simplification against BG2 at supply points](figure_1b_bsp_gsp_contrasts.svg)

![Each simplification against BG2 at solar farms](figure_1c_solar_contrasts.svg)

![Each simplification against BG2 at wind farms](figure_1d_wind_contrasts.svg)

![Score by forecast lead](figure_2_by_lead.svg)

![Blend weights by month](figure_6_blend_weights.svg)

**Example forecasts.** Each figure shows BG2, the simplifications, and `manual_heuristic` for one
series in the week with the highest mean power in summer, the week with the highest mean power in
winter, and the week with the largest error. The series are Primary G, Solar farm A, Wind farm A,
and one supply point.

![Primary G, summer](figure_4_primary_summer.svg)

![Primary G, winter](figure_4_primary_winter.svg)

![Primary G, large error](figure_4_primary_large_error.svg)

![Supply point, summer](figure_4_supply_summer.svg)

![Supply point, winter](figure_4_supply_winter.svg)

![Supply point, large error](figure_4_supply_large_error.svg)

![Solar farm A, summer](figure_4_solar_summer.svg)

![Solar farm A, winter](figure_4_solar_winter.svg)

![Solar farm A, large error](figure_4_solar_large_error.svg)

![Wind farm A, summer](figure_4_wind_summer.svg)

![Wind farm A, winter](figure_4_wind_winter.svg)

![Wind farm A, large error](figure_4_wind_large_error.svg)

**The five blend members.** Each figure shows the median forecast of the weekly, monthly, analogue,
pooled-by-type, and frozen-with-correction members, with the blend and the actual, at one Primary
substation. The same three weeks are drawn for Primary G and Primary K in this folder
(`figure_5_members_g_*` and `figure_5_members_k_*`).

![Blend members at Primary A, summer](figure_5_members_a_summer.svg)

![Blend members at Primary A, winter](figure_5_members_a_winter.svg)

![Blend members at Primary A, large error](figure_5_members_a_large_error.svg)

## Results

- **Architecture and arms.** Every arm is BG2 (experiment 87) with one part removed. The removal
  arms are the five blend members alone, every blend of two or three members, BG2 with the frozen
  member uncorrected, BG2 without the offset, BG2 without the spread factor (the last two stored in
  experiment 87), and the single XGBoost model without the three power inputs. The weekly model
  without lags is the Primary and supply point simplest candidate; the monthly model without lags
  is the farm candidate and the monthly contrast at Primaries. Every no-lag model is BG2's model
  with the three lag columns removed (same rows, weights, 500 rounds, seed 0).
- **Row set and schedule.** The 292 runs of `fold_a_daily_gap_v1`. Primary and supply point
  contrasts C1 and C2 use weekly refits, as BG2 does. The farm contrasts and the monthly lag contrast
  use monthly refits, as BG2's farm arms do. The lag cost has the same sign and a similar size under
  both schedules (0.903 points monthly and 0.710 weekly, at Primaries). GPU use was 13.0 hours of
  wall time.
- **Lag weather.** The lags are the observed power of the series, not weather, so no past-weather
  source enters the lag columns, and every lag is at least 3.5 hours older than the forecast
  origin.
- **The five verdict contrasts, each at 99%.** Primary C1: +0.806 (0.367 to 1.569), not safe
  against 0.15. Supply points C2: +0.278 (0.040 to 0.720), not shown safe against 0.30. Solar farms
  C3: +0.010 (-0.107 to +0.110), not shown safe against 0.10, because the upper end exceeds the limit
  by 0.010. Wind farms C4: -0.059 (-0.184 to +0.058), safe against 0.30. Solar offset C5, midday in
  summer at the 5 offset Primaries: -0.184 (-0.297 to -0.103), the offset helps.
- **The lag cost sits in two Primary substations.** Weekly refits, no-lag minus with-lag, per
  substation: Primary M +5.69 and Primary K +3.94; Primary I +0.63, Primary P +0.52, Primary E +0.51;
  the other 11 range from -0.09 to +0.22. Leaving out Primary K and Primary M, the lag cost is
  +0.136 points (99% interval 0.055 to 0.232) and C1 is +0.173 (0.081 to 0.295). With all 16, C1 is
  +0.806. Both of those two substations lose in the same direction under weekly and monthly refits
  (figure 7).
- **Where each simplification lands, Primary.** Differences from BG2 (5.260), 95% intervals:
  weekly model with lags +0.095 (0.050 to 0.140); blend of weekly, analogue, and frozen -0.002
  (-0.006 to 0.000); analogue and frozen pair +0.016 (-0.005 to +0.041); without the 14-day
  correction +0.060 (0.021 to 0.094); without the spread factor +0.017 (0.003 to 0.033); without the
  offset +0.010 (0.000 to 0.019); each member alone +0.091 (weekly), +0.154 (analogue), +0.210
  (monthly), +0.270 (frozen), +0.336 (pooled). Against the stack (5.543) the simplest candidate is
  +0.523 (0.124 to 1.071) worse; against `manual_heuristic` it is -0.115 (-0.433 to +0.430), no
  difference.
- **Supply points.** Weekly without lags 13.203, monthly without lags 13.379, against BG2 12.925. The
  simplest candidate beats `manual_heuristic` by 1.300 points (0.784 to 1.803) and the stack by 0.356
  (0.118 to 0.677).
- **Farms.** Solar farms: BG2 5.012, without lags 5.022; `manual_heuristic` 5.718. The fitted
  spread factor changes the score by +0.002 (-0.005 to +0.013). Wind farms: BG2 11.032, without lags
  10.973; the power-curve margin changes the score by -0.057 (-0.098 to -0.024); `manual_heuristic`
  13.906. The no-lag farm forecasts beat `manual_heuristic` by 0.693 (solar) and 2.930 (wind) points.
- **Midday in summer, night, and other hours (offset effect, BG2 minus BG2 without the offset).** At
  the 5 offset Primaries: midday in summer -0.184 (-0.297 to -0.103), night -0.014 (-0.044 to +0.019),
  other hours -0.017 (-0.056 to +0.021), all rows -0.031 (-0.071 to +0.011). At all 16 Primaries:
  midday in summer -0.058 (-0.093 to -0.032), all rows -0.010 (-0.022 to +0.003). The score at the 5
  offset Primaries at midday in summer is 5.319 with the offset and 5.503 without it. Midday in
  summer is the half-hours that start from 10:00 to 13:30 UK time in April to September, and night is
  the half-hours that start from 23:00 to 03:30.
- **Lead time.** The simplest Primary candidate loses at every lead band, and the loss shrinks with
  lead: +1.085 at 0 to 24 hours, +0.993 at 1 to 2 days, +0.906 at 2 to 6 days, +0.743 at 6 to 10 days,
  and +0.670 at 10 days and beyond (all 99% intervals exclude zero).
- **Blend weights.** After the first month (equal weights), the blend gives the monthly and the
  pooled-by-type members zero weight in every month. The weekly member takes 0.25 to 0.50, the
  analogue member 0.25 in every month, and the frozen member with correction 0.25 to 0.50. This
  matches the best three-model blend.

| Tail check, 292 runs | Quantile | Primary BG2 | Primary simplest | Wind BG2 | Wind simplest |
|---|---|---|---|---|---|
| Share of actuals below the quantile | 0.01 | 0.007 | 0.004 | 0.003 | 0.002 |
| | 0.05 | 0.042 | 0.035 | 0.037 | 0.029 |
| | 0.95 | 0.937 | 0.935 | 0.951 | 0.957 |
| | 0.99 | 0.986 | 0.991 | 0.987 | 0.989 |
| Pinball loss | 0.01 / 0.05 | 0.280 / 0.901 | 0.317 / 1.071 | 0.269 / 1.265 | 0.269 / 1.261 |
| | 0.95 / 0.99 | 1.208 / 0.415 | 1.419 / 0.447 | 2.708 / 0.697 | 2.692 / 0.694 |

The tails of the lag-free forecasts stay close to nominal at the farms and at Primaries. Primary
loses most at the 0.05 and 0.95 pinball loss (+0.170 and +0.211).

| Class and limit | Simplest recipe within the limit | Cost against BG2 |
|---|---|---|
| Primary substations, 0.15 | A single weekly model with lags | +0.095 (0.050 to 0.140) |
| Primary substations, 0.15 | Blend of weekly, analogue, and frozen | -0.002 (-0.006 to 0.000) |
| Supply points, 0.30 | None found; without lags | +0.278 (0.040 to 0.720) |
| Solar farms, 0.10 | Without lags (borderline) | +0.010 (-0.107 to +0.110) |
| Wind farms, 0.30 | Without lags | -0.059 (-0.184 to +0.058) |

## Checks

- **Median forecast minus mean forecast, per site, in points of capacity.** BG2 at Primaries ranges
  from -3.01 (Primary K) to +0.09, with all but Primary K and Primary I (-1.96) within -0.9 to +0.1.
  The simplest candidate ranges from -3.99 to +0.46. `manual_heuristic` has the same shape (-3.11 at
  Primary K). At farms the gap is -1.4 to -2.2 at solar and -5.4 to -7.6 at wind for both BG2 and the
  lag-free forecast, so removing the lags shifts it by less than 0.6 points. The supply points range
  from -5.9 to +3.9 for BG2 and `manual_heuristic` alike.
- **Error of the mean forecast beside each score contrast.** C1 +1.084 (0.648 to 1.717), C2 +0.241
  (0.030 to 0.494), C3 +0.041 (-0.043 to +0.121), C4 -0.055 (-0.165 to +0.050), all 95%. The error of
  the median forecast agrees in sign with the score contrast for every contrast except two spread-factor
  contrasts, whose median is unchanged by construction. For C5 at the offset Primaries at midday in
  summer, the median error changes by -0.246 (-0.362 to -0.148).
- **Triggers fired.** Trigger 2 (a result contradicts its prediction): the lag cost of 0.806 is
  above the predicted 0.05 to 0.30; the per-substation table above is the tested explanation.
  Trigger 4 (leaving a site out flips a planned contrast): leaving out solar farm E turns C3 from
  not safe into safe (-0.024 to +0.018 at 99%). Trigger 5 (a nested arm beats the arm that contains
  it): wind farms without lags score 0.059 points lower than with lags, within the interval.
  Triggers 1, 3, 6, and 7 did not fire. The audit is pending; its findings and triage will be
  recorded here.

## Caveats and what to try next

- **The two-substation result needs a reason.** Primary K and Primary M lose 3.9 to 6.5 points
  without lags. The lags may carry a level shift, such as a new connection, that the weather and
  calendar inputs cannot see. This experiment does not test that cause.
- **The best pair and best triple are exploratory.** They were chosen on the 73 runs of
  `fold_a_every5d_v1`, which interleave in time with the 292 runs scored, so their numbers decide no
  verdict.
- **Not run.** The curtailment-cost metric (it needs the private node table), a no-lag blend,
  and an offset on no-lag models. The offset contrast C5 compares blends that each choose their own
  monthly weights, so it includes a small weights effect.
- **Selection bias.** The BG2 recipe was tuned on the same validation year, so every simplification
  is judged against an optimistic reference.
- **Next.** Find out what the lags carry at Primary K and Primary M; test the three-model blend and
  the farm models without lags on a second year; give the supply points a lag-free alternative only
  if a level input can replace the lags.
