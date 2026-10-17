---
id: "00081"
slug: r11-control-member-second-validation-year-demand
status: planned
size: complex
hypothesis: >-
  The Primary substation demand findings of experiments 48 and 49 hold on a second set of forecast
  runs that the project has not scored: the daily 00:00 UTC ECMWF ensemble (ENS) control-member runs
  of 2023-07-01 to 2024-03-31. On those runs, for the 16 Primary substations, weekly refitting of
  the XGBoost demand model N1 without error correction scores below the stack (frozen N1 with the
  14-day correction and the spread factor 1.15), weekly refitting scores below monthly refitting,
  and dropping the Christmas column or lengthening the half-life from 0.5 to 2 years changes the
  score of the monthly rolling-window refit by less than 0.03 points of capacity.
prediction: >-
  Weekly minus the stack is -0.05 to -0.25 (experiment 48 found -0.20 on 51-member forecasts, and
  one control member gives a noisier forecast that a correction can repair less, so the gap may
  shrink or grow). Weekly minus monthly is -0.03 to -0.15. Monthly minus the stack is -0.20 to
  +0.10, unresolved as in experiment 48. Dropping the Christmas column moves the score by -0.03 to
  +0.03 over the whole fold and by more inside 2023-12-18 to 2024-01-04. The 2-year half-life moves
  it by -0.03 to +0.03. The blend of four models scores 0.03 to 0.10 below its best member. In
  2023-07 to 2023-09, weekly refitting gains more over the stack than in 2023-10 to 2024-03,
  because weekly refits adopt the 9 km product (introduced 2023-06-27) sooner than the frozen model.
decision_rule: >-
  Six planned contrasts on the quantile score (twice the mass-weighted mean pinball loss over 9
  levels) in points of capacity, over the 16 Primaries and all runs of fold B, seed 0, paired, with
  the ISO weeks of forecast start resampled (about 40 blocks) and 50,000 resamples, at 99.17% (a
  Bonferroni family of six). The control-only runs span 9 months, so months cannot be the block;
  a t-interval over the 9 monthly mean differences is reported beside each contrast. C1 weekly
  (no correction) minus the stack. C2 monthly minus the stack. C3 weekly minus monthly. C4 the
  monthly rolling 12-month refit without the Christmas column minus the same refit with it. C5 the
  monthly rolling 12-month refit with a half-life of 2 years minus the same refit with 0.5 years.
  C6 the blend (one weight vector per month, chosen on earlier months, over weekly, monthly,
  rolling, and stack) minus its best single member chosen with hindsight. A contrast replicates
  when its estimate has the sign the earlier experiment found and its interval excludes zero
  (rule R-a). A contrast holds when pooled when the precision-weighted mean of the earlier
  estimate and this one, with standard errors taken from the half-widths of the two intervals
  (each divided by the normal quantile of its level), has a 95% interval that excludes zero and
  both estimates have the earlier sign (rule R-b). The earlier estimates are C1 -0.20 and C3 -0.13
  (experiment 48), C2 -0.08 (experiment 48), C4 -0.017, C5 -0.003, and C6 -0.076 (experiment 49).
  Thresholds are set by what a change costs to port. C1, C2, and C3 have threshold zero, because
  retraining replaces the correction step and adds a scheduled job. C6 has threshold -0.05,
  because the blend adds three XGBoost models. C4 and C5 follow experiment 49: a component stays
  when its interval lies wholly above zero and can go when its interval lies wholly inside
  plus or minus 0.03. Verdict, from fold B: supported when C1 and C3 replicate by R-a; partly
  supported when one replicates by R-a or both hold by R-b; refuted when the C1 interval lies
  wholly above zero; inconclusive otherwise. Fold C (below) cannot change the verdict. It says
  how to read it. When C1 on fold B excludes zero and C1 on fold C does not, the page reports the
  gain as possibly caused by adapting to the 9 km product. Every other read is exploratory and
  labelled so, at 95%: lead bands, valid months, each series, and the first quarter (2023-07 to
  2023-09) against the rest.
benchmark: >-
  The stack (frozen N1 fitted on the control member up to 2023-06-30, half-life 2 years, 500
  rounds, with the 14-day correction and the spread factor), `manual_heuristic`, and
  `climatology`, each scored on the same runs.
row_set_id: control_fold_b_daily_v1
metric: >-
  Quantile score in points of capacity (1 point is 1% of the series' effective capacity); MAE of
  the median and the exceedance rate of the 0.95 quantile are reported and decide nothing. Every
  number is a control-only number and is never compared with a 51-member score.
code: ml-experiments/00081-r11-control-member-second-validation-year-demand/
production_status: not-for-production
adopted_in: none
---

# Experiment 81: do the weekly-retraining and VP1 findings hold on a second set of forecast runs?

**Status: planned. No code has run.** This page is the plan, committed before any fit. The
sections below are the question, the design, what the fold cannot say, and the sizing.

## The question

**Two findings that decide the next model release rest on one validation year.** Experiment 48
found that refitting the Primary substation demand model every week scored 0.20 points of capacity
below the best setup then known, but the gap crosses zero with a t-interval over the 12 months and
shrinks to -0.08 without three of the 16 substations. Experiment 49 found that dropping the
Christmas column and lengthening the half-life changed the score by under 0.02 points, on one
winter. We ask whether the same contrasts have the same sign on forecast runs from a different
period, 2023-07 to 2024-03, for which the project holds only the ENS control member.

## Design

**Rows and folds.** The ENS control-member backfill (1,105 daily runs, 2021-03-21 to 2024-03-31,
no run for 2022-09-10) gives two folds. Fold B, the one the maintainer asked for, is the 275 daily
00:00 UTC runs of 2023-07-01 to 2024-03-31 (row set `control_fold_b_daily_v1`). The ECMWF ensemble
moved from 18 km to 9 km native resolution on 2023-06-27, so every run in fold B is 9 km while 2.3
years of its training history is 18 km. Fold C is the 273 runs of 2022-07-01 to 2023-03-31 (row
set `control_fold_c_daily_v1`), where every run and every training row is 18 km. Fold C has 15
months of training history before it starts and runs the same arms, so that a retraining gain
that appears only in fold B can be read as adaptation to the new product. Neither fold scores a
valid time on or after 2026-07-01.

**Training rule.** Every model trains on its own history: the ENS control member's rows from
2021-03-21 to the refit date. No ERA5-trained model is used. Frames are built as in experiment 48
(production's half-hourly upsampling, the 16 Primaries' H3 cells, power lags from the observed
power), from one control member in place of 51.

**Arms (Primary, 16 series, N1 recipe of experiment 34, spread factor 1.15).**

- **Stack:** frozen N1 (fitted to 2023-06-30 in fold B, to 2022-06-30 in fold C; half-life 2
  years; 500 rounds) with the 14-day correction built from the forecasts as issued.
- **Weekly:** full refit every week, half-life 0.5 years, no correction.
- **Monthly:** the same, every month.
- **Rolling:** experiment 49's P-roll, a refit every month on the latest 12 months, 150 rounds,
  learning rate 0.15, half-life 0.5 years.
- **Rolling without the Christmas column,** and **rolling with a 2-year half-life.**
- **Frozen without correction** (descriptive), `manual_heuristic`, and `climatology`.
- **Blend:** weekly, monthly, rolling, and stack combined by a monthly weight vector, as
  experiment 49 did in serving form. It has four members, not the five of experiment 49: the
  weekly-analogue and pooled members are left out, so C6 asks whether blending a corrected frozen
  model with retrained models helps, not whether experiment 49's five-model result repeats.

**Arms left out, with the reason.** Substation fingerprints are not run, because experiment 49
bounded their effect to under 0.05 points and the porting threshold is 0.10, so no second year can
change the decision. The solar arms are a second stage (below).

**Stage B, solar and analogues (exploratory, conditional).** Experiments 64, 78, and 74 gave the
forecast a solar term as a starting offset (`am`), compared it with a zero start (`am0`), and
compared analogues with it. Those contrasts use daylight April-to-September rows, and fold B holds
92 days of them (2023-07 to 2023-09) at 5 series, with about 13 weeks, against 29 in experiment 64.
They also need a capacity table and a calibrated sunshine anomaly that, in experiment 64, come from
the ensemble mean, which one member cannot give. Stage B starts only if a feasibility check, printed
before any fit, finds a capacity estimate for each of the 5 series that uses data before 2023-07-01
and a CAMS shape for the whole of fold B. It then reports `am` minus `d1l` and `am` minus `am0`
at 95% as exploratory reads of sign only, and runs no verdict. If the check fails, the page says
Stage B was not run.

## What the control-only fold cannot say

- **Absolute skill.** One member cannot give an ensemble mean, spread, CRPS, coverage, or a
  51-member quantile score. Every number here is labelled "control-only fold" and none is compared
  with experiment 48's or 49's absolute scores; only the sign and size of paired contrasts are.
- **Whether a gain is retraining or adaptation to the 9 km product.** In fold B the frozen stack
  was fitted on 18 km data and scores 9 km data, while a refit adopts 9 km rows as they arrive.
  Fold C removes the product change but has a shorter history (15 months) and a different year.
- **Independence of the two years.** Both years score the same 16 substations, so pooled intervals
  treat years as independent while the series are shared.
- **Anything about the live seam.** Our own ENS archive starts on 2024-04-01 with a different
  product, so these results say nothing about the product that production serves.
- **A year's worth of months.** Fold B has 9 months and 40 weekly blocks. The intervals resolve
  less than experiment 48's.
- **Generators.** Only Primaries are scored. No NGED generator series appears.

## Sizing, by the five triggers

- **Stored:** predictions under `data/ml-experiments/`, private; nothing in production.
- **Serving path:** not touched.
- **Degradation rule:** not touched.
- **More than one defensible design:** yes, the choice of folds and of replication rule, which is why
  this plan has one Opus review.
- **Callers not nameable without searching:** no; the scripts copy experiments 34, 48, and 49.

The experiment is complex because it builds new frames from a new archive and its plan could change
the verdict on weekly retraining. Review set: one Opus plan review, one Sonnet code review, one
Sonnet write-up check, one Sonnet prose review.

**Earlier advice followed.** Test weekly retraining and the Christmas column on a further year
(48, 49); read contrasts at a Bonferroni level (34); leave the spread factor at 1.15 (49).
Departures: month blocks become ISO-week blocks, because the fold spans 9 months.

**Resources.** About 3 GPU hours per fold for the weekly, monthly, rolling, stack, and variant fits
(training sets are smaller than in experiment 48, with one member and 2.3 years), run as parallel
workers inside slot E (6 cores) with one GPU fit at a time. No GPU fit starts before the ML manager
says go.
