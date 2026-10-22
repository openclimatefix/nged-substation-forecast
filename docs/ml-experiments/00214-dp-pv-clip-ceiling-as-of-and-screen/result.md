---
id: '00214'
slug: dp-pv-clip-ceiling-as-of-and-screen
theme: differentiable-physics-embedded-generation
also_relevant_to:
  - production-reference-and-best-guess
status: finished
size: >-
  medium. Stored: private files under data/ml-experiments/00214-dp-pv-clip-ceiling-as-of-and-screen/
  only. Serving path: not touched. Degradation rule: not touched. Designs: more than one defensible
  design for the as-of rule, but the brief fixes the 12-month window and the 50% to 100% grid.
  Callers: this folder's scripts copy experiment 180's fit, forecast, and scoring code, and call
  `studies.dp_gam` unchanged. Asset graph: not touched.
hypothesis: >-
  The export-limit signature of experiment 180 is a stable property of D5, not a fit to the scored
  year. A ceiling chosen at each monthly refit only from the 12 months of forecasts before that
  refit keeps most of experiment 180's gain at D5, and the same signature is absent at the other
  Primaries, with D9 the most likely exception.
prediction: >-
  Written before any fit. (a) The as-of ceiling at D5 lies in 50% to 70% at 10 or more of the 12
  monthly cuts of the scored year, and the as-of arm keeps at least 0.08 MW of experiment 180's
  0.133 MW gain on D5 sunny daylight (control minus as-of arm). (b) At least one other Primary
  gains 0.03 MW or more with a clip, and the rest move by under 0.01 MW. Part (b) is the
  uncertain part: experiment 180's stage 0 found the implied solar output per unit shape falls
  towards midday only at D5 (D10, D11, D14, and D16 are flat or rising, and D9's is negative), so
  the author expects (b) to fail and the clip to be a D5 property.
decision_rule: >-
  Planned contrasts, each on sunny daylight half-hours, all leads to 168 hours, mean absolute error
  of the median forecast (q50) in MW, as-of arm minus control, month-block bootstrap on paired
  differences at the 98.33% level (three planned contrasts: D5 as-of, the best other Primary's
  as-of arm, and the rest pooled). `supported`: (a) and (b) both hold, with the D5 interval
  wholly below zero, the other Primary's interval wholly below zero, and the placebos D10 and D16
  moving by under 0.02 MW. `partly supported`: (a) holds and no other Primary gains 0.03 MW, which
  is reported as "the clip is a D5 property". `not supported`: the as-of ceiling at D5 leaves
  50% to 70% at 3 or more cuts, or the as-of arm keeps under 0.08 MW. `inconclusive`: the D5
  interval includes zero. The threshold is 0.08 MW because an as-of clip adds one per-substation
  selection step to each refit, so it needs a gain a quarter of the way to experiment 180's
  target of closing the gap to BG2.
benchmark: >-
  The control (the fixed fleet shape, experiment 180's arm c) and experiment 180's fixed-60% arm
  (arm k60), on `fold_a_daily_gap_v1`
row_set_id: fold_a_daily_gap_v1
metric: >-
  Mean absolute error of q50 in MW and pinball loss averaged over 11 quantiles in MW, on sunny
  daylight half-hours (the top third of a substation's daylight half-hours ranked by the fixed
  fleet shape, as in experiment 180), and over all half-hours, with the tail slice of the sunny
  rows (coverage and pinball at q01, q05, q95, q99)
code: ml-experiments/00214-dp-pv-clip-ceiling-as-of-and-screen/
verdict: partly supported
production_status: candidate
adopted_in: none
date: 2026-10-22
best_arm: >-
  Ceiling chosen as-of at D5 (60% at 11 of 12 refits, 50% at one): -0.129 MW on sunny daylight
  (98.33% interval -0.212 to -0.029)
headline: >-
  D5, sunny daylight, all leads, scored year, mean absolute error minus the control's: as-of
  ceiling -0.129 MW (-0.212 to -0.029), fixed 60% -0.133 MW (-0.212 to -0.035). D9 with an as-of
  ceiling -0.032 MW (-0.076 to +0.007). D10, D11, D14, and D16 at 60%: +0.004, +0.017, -0.001, and
  -0.004 MW, each interval spanning zero.
lessons:
  - >-
    D5's ceiling does not depend on hindsight: from earlier data alone the 60% ceiling is chosen at
    11 of 12 refits and 50% at the other, and the gain is 97% of the fixed-60% gain.
  - >-
    Only D5 shows the export-limit signature in all three summers (figure 3). D9 gains a point
    estimate of 0.03 MW, but its interval includes zero and its diagnostic is not interpretable.
  - >-
    A 60% clip at D11 raises the sunny error by 0.017 MW (interval -0.005 to +0.041), so the clip
    should be switched on per substation and not as a fleet default.
advice:
  - >-
    Choose the D5 clip ceiling at each refit from the previous 12 months of forecasts, with 50% to
    90% candidates and no clip as a candidate; it picks 60% almost always (figure 2).
  - >-
    Do not apply the clip to other Primaries by default; D11 gets worse by 0.017 MW and D10, D14,
    and D16 do not move (figure 1).
  - >-
    Re-test D9 once it has two or more summers of data before the scored year; the screen could
    not run on its 2022 and 2023 summers (figure 3).
take_home_message: >-
  The 60% cap on D5's solar output holds up when it is chosen only from earlier data: at each
  monthly refit the cap picked from the previous 12 months was 60% 11 times out of 12, and the
  forecast keeps 97% of the gain. No other Primary shows the same flattening of net power; D9 gains
  a little (0.03 MW, not clearly different from zero) and D11 loses a little, so the cap is a D5
  property for now.
---

# Does the 60% export clip at D5 hold when its ceiling is chosen as-of?

**The 60% ceiling at D5 survives being chosen only from earlier data, and no other Primary shows the
same signature clearly enough to say the clip generalises.** [Experiment
180](../00180-dp-sunny-daylight-fix-d5-solar-shape/result.md) found that capping the solar output a
demand forecast subtracts from a Primary substation's power at 60% of the solar capacity held for
D5 lowers the forecast's error on sunny half-hours. We asked two follow-up questions. First, if the
ceiling is picked at each monthly refit from the previous 12 months of forecasts, instead of on
the months the result was later scored on, does the gain remain? Second, do the other five
Primaries show the same flattening of net power as the sun rises? As-of selection kept 97% of the
gain (-0.129 MW against -0.133 MW). The screen found the flattening at D5 only. D9 improved by 0.032
MW with an interval that includes zero, and D11 worsened by 0.017 MW.

![Figure 1. Sunny-daylight error against the control, by Primary](figure_01_headline.svg)

![Figure 2. The clip ceiling at D5 and D9, scored year](figure_02_ceiling_sweep.svg)

![Figure 3. The screen: implied solar output per unit shape, by summer](figure_03_screen.svg)

**Verdict `partly supported`.** The pre-registered D5 part holds. The as-of ceiling lay in 50% to
70% at 12 of 12 refits (60% at 11, 50% at one), and the as-of arm keeps 0.129 MW of the 0.133 MW
gain, against a required 0.08 MW. The D5 interval is wholly below zero and both placebos move by
under 0.02 MW. The other-Primary part is mixed. D9's gain of 0.03 MW or more is met by the point
estimate but not by the interval, which the plan required to lie wholly below zero for `supported`.
The plan did not say how to label that case, so we chose `partly supported`. D11 moves by more than
the 0.01 MW the plan allowed for the rest.

## Results

- **Rows and scope:** experiment 178's rows (the scored year, 292 runs of `fold_a_daily_gap_v1`,
  outside the switching events, leads 0.5 to 168 hours). Sunny daylight is the top third of
  the substation's daylight half-hours by the fixed fleet shape. D5 has 14,045 such half-hours and
  D9 has 15,415.
- **Schedule and models:** the same monthly refits as experiment 180 (T-all, calibrated ensemble
  mean, stage-2 arm `d`). Only the ceiling differs between arms, so the contrast does not depend on
  the retraining schedule.
- **Reproduction:** the new fixed-60% arm at D5 matches experiment 180's stored forecasts exactly
  (maximum difference 0 in q50 and q99).
- **Screen (figure 3):** the mean of the top two deciles over the mean of deciles 4 to 6 is 0.77,
  0.68, and 0.73 at D5 in the summers of 2022, 2023, and 2024, against 0.98 to 3.74 at D10, D11, D14,
  and D16. D9 has 0 and 28 clear half-hours in 2022 and 2023 and a negative decile profile in 2024,
  so its ratio is undefined and it fails the screen.
- **As-of ceiling at D5:** 60% at 11 cuts and 50% at the cut of 2026-03-01 (error 0.468 MW at both
  50% and 60%). At the first cut the window errors were 0.494 MW at 60%, 0.502 at 50%, 0.519 at 70%,
  and 0.591 at no clip.
- **Absolute scores at D5, sunny daylight (MW):** control mean absolute error 0.655, as-of 0.527,
  fixed 60% 0.523. Pinball averaged over 11 quantiles 0.185, 0.155, and 0.154.
- **Gap to BG2 at D5 (arm minus BG2):** +0.073 MW (0.028 to 0.099) as-of, +0.069 MW fixed 60%,
  +0.202 MW control.
- **D9:** the as-of ceiling is 60% at 10 of 12 cuts and 50% at two. As-of minus control is -0.032 MW
  (-0.076 to +0.007), and fixed 60% is -0.037 MW (-0.080 to 0.000). Its gap to BG2 falls from
  +0.079 MW to +0.047 MW at the as-of arm (-0.008 to +0.080). Control, as-of: 0.497 and 0.465 MW.
- **The rest at 60% (arm minus control, sunny MW):** D10 +0.004 (-0.024 to +0.037), D11 +0.017
  (-0.005 to +0.041), D14 -0.001 (-0.049 to +0.049), D16 -0.004 (-0.050 to +0.048); pooled
  +0.004 (-0.029 to +0.043).
- **Tail calibration at D5, sunny rows (coverage, pinball in MW), as-of against control:** q01 0.056
  and 0.051 (0.0230 and 0.0252), q05 0.178 and 0.168 (0.0841 and 0.0972), q95 0.996 and 0.967
  (0.0765 and 0.0806), q99 0.999 and 0.993 (0.0209 and 0.0218). The upper quantiles over-cover in
  both arms, and the clip raises the over-coverage.
- **Replicate months:** the as-of window grows from 3 to 11 months and again picks 60% at every cut
  (-0.124 MW at D5, -0.031 at D9, the same as fixed 60%). The tuning months were used to pick 60% in
  experiment 180, so this is a repeat of the choice and not independent evidence.
- **Ceiling sweep (figure 2):** at D5 the gain peaks at 50% to 60% (-0.126 and -0.133 MW) and is
  -0.007 MW at 90%. At D9 the shape is the same at one third of the size.
- **Caveats:** D9 has data from 2024, so its as-of windows hold 12 months of forecasts but its
  fits have a short history. The as-of rule picks the ceiling by sunny error, the quantity scored.
- **Compute:** 11 arm-and-substation fits (about 45 s each on one core) and the scoring, at most 4
  cores, inside a 16 GB scope.

## Checks

- **Median minus mean forecast, per site:** 0.000 MW at D5 and D9 (sunny rows, as-of arm). The
  stored mean and median agree, so the spread does not shift the forecast.
- **Error of the mean forecast beside each quantile-score contrast:** the mean-forecast contrast
  equals the median-forecast contrast to the third decimal (D5 as-of -0.129 MW, D9 -0.032 MW), so
  trigger 3 cannot fire.
- **Triggers:** none fired. The reproduction gate passed, both placebos stayed inside 0.02 MW, no
  nested arm beats the arm containing it, and the D5 interval has about the same width as experiment
  180's. Dropping D11 from the pooled rest moves the pooled estimate from +0.004 MW to about 0 and
  does not change the label. D9's wide interval is reported, not treated as a flip.

## What to try next

- **Port the D5 clip as an as-of rule:** choose the ceiling at each refit from the previous 12
  months, candidates 50% to 90% and no clip. It costs one selection per refit.
- **Re-test D9 with more history:** D9's gain is within noise and its screen could not run, so a
  later release with two or more summers of D9 data can repeat the test.
- **Suggestion 3 of the brief (combination check)** builds on the D5 clip and can use this as-of
  rule in place of the fixed ceiling.
