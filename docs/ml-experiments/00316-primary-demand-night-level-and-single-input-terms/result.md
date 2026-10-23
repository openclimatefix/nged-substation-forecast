---
id: '00316'
slug: primary-demand-night-level-and-single-input-terms
title: >-
  Is the band heads' gain at one Primary substation a night-time level error, and which single-input
  term of the best-guess demand model is behind its April-to-June gap to BG2?
kind: other
theme: production-reference-and-best-guess
also_relevant_to:
  - calibration-and-uncertainty
status: done
size: >-
  medium. Stored: private files under data/ml-experiments/00316-*/ only. Serving path: not touched.
  Degradation rule: not touched. Designs: more than one is defensible for the stage 1 graft (this
  plan pre-registers a surrogate-based emulation of grafting, because arm D's term contributions are
  not stored). Callers: this folder's scripts, which read the stored forecasts of experiments 251,
  266, 305, 311, and 314 and call reviewed `studies.*` modules without changing them. CPU only, no
  downloads. A stage 2 correction head would need its own addendum before it runs.
hypothesis: >-
  (Stage 0) The gain of the band heads of experiments 311 and 314 at D10, an anonymised Primary
  substation, is mostly a night-time level error of Primary demand research default 1 and not a
  better description of the uncertainty. (Stage 1) The April-to-June gap between arm D and BG2 at
  D5 and D9, two anonymised Primaries, sits mainly in arm D's response to sunshine and not in its
  recent-power or calendar terms.
prediction: >-
  Written before any score of the new arms is read; each line can fail. Lead range 0.5 to 168 hours,
  shorter than NGED's 1 to 10 days. (S0a) At least 70% of D10's gain under the pinball head (arm P)
  lies on night half-hours (21:00 to 07:30 UTC). (S0b) An as-of night offset (median residual of the
  previous 28 nights' half-hours, added to every quantile of D10's night half-hours, no head)
  reproduces at least 70% of D10's gain under arm P; under 30% means the head does something else.
  (S1a) Grafting the solar response recovers at least 50% of the advantage of the depth-1 XGBoost
  refit over arm D on the worst quarter of April-to-June daylight half-hours at D5 and D9. (S1b)
  Grafting the lag handling and grafting the calendar profile each recover under 25%. (S1c) Grafting
  all five families together recovers at least 50% (the check that the emulation is usable).
decision_rule: >-
  Intervals resample calendar months of forecast initialisation (12 in the scored year; the 3
  months of April to June for stage 1, so stage 1 intervals are wide and the page says so). S0a and
  S0b are read on the lower bound of the 95% interval of each share. Stage 1 names a family when
  its share S is at least 0.30; the 95% interval is reported beside it. Stage 2 runs only if
  stage 1 names a family, or if no single family reaches 0.30 but the graft of solar, lags, and
  calendar together reaches 0.50 (the brief's two rules for that case disagree, and this plan
  resolves them that way). A stage 2 addendum with its arms, cut protocol, and minimum gain (0.03
  points, a new head) is committed before stage 2 runs. Nothing is ported by stages 0 and 1, so no
  porting threshold applies to them.
benchmark: >-
  Primary demand research default 1 (experiment 251's arm 1) on row set fold_a_daily_gap_v1; BG2,
  `manual_heuristic`, and `climatology` beside it.
row_set_id: fold_a_daily_gap_v1
metric: >-
  Quantile score (twice the mean pinball loss over 11 quantiles, in points of effective capacity;
  smaller is better).
code: ml-experiments/00316-primary-demand-night-level-and-single-input-terms/
verdict: inconclusive
best_arm: >-
  Not applicable: a diagnostic. At D10 the pinball band head scores 3.024 and Primary demand
  research default 1 scores 3.169; the as-of night offset scores 3.148.
headline: >-
  The band heads' gain at D10 is 0.144 points (95% interval 0.049 to 0.262), 88% of it on night
  half-hours, but an as-of night offset reproduces only 14% of it (-16% to 99%): the heads narrow
  D10's night-time bands and do not correct a level error.
lessons:
  - >-
    The heads narrow D10's night-time bands. Keeping only the head's narrower width scores 63% of the
    gain, and keeping only its upward shift of the band centre lowers the score.
  - >-
    A depth-1 surrogate of arm D fits its median with R-squared 0.82 to 0.93, too loosely to graft
    onto, so stage 1 could not name a family.
production_status: not-for-production
adopted_in: none
date: 2026-10-23
take_home_message: >-
  The band heads help at one Primary substation because Primary demand research default 1 draws its
  night-time bands about 7 points too wide there, not because it forecasts the night level too high:
  a night level correction reproduces only 14% of the heads' gain. The test of which single-input
  term of the best-guess model trails BG2 in April to June could not be run, because a simple
  stand-in for the model was too loose to swap terms into.
advice:
  - >-
    Test a band-width correction for D10's night half-hours before building another band head,
    because narrowing the bands alone scores 63% of the heads' gain and a night level offset only 14%
    (figure 1).
  - >-
    Do not add an as-of night level term to the default to chase D10's band-head gain, because the
    offset gains 0.021 points against the pinball head's 0.144 (figure 1).
  - >-
    Compute arm D's own term contributions from its fitted additive model before comparing its terms
    with a refit's, because a depth-1 surrogate of its median misses too much for a graft to be
    scored (figure 3).
  - >-
    Report each band head's gain at D10 and at the other five Primaries separately, because the
    other five together gain 0.007 points (-0.014 to 0.026) under the pinball head (Results).
---

**Take-home: the band heads help at D10, an anonymised Primary substation, because the default's
night-time bands there are too wide, not because its night level is wrong.** The default forecast
sits about 1 to 2 points above the actual demand at D10 at every hour, but correcting that level at
night recovers only 14% of the heads' gain. The heads narrow D10's 5% to 95% band by 7.5 points at
night on average, and keeping only that narrowing scores 63% of the gain. The second question, which
single-input term of the best-guess model trails BG2 in April to June, could not be answered: the
stand-in used for the model's own terms was too loose. Verdict: inconclusive; stage 2 was not run.

Two questions were asked of Primary demand research default 1 ("the default"), a stack of additive
terms with neural-network heads. Lead range is 0.5 to 168 hours, shorter than NGED's 1 to 10 days.
First, experiments 311 and 314 trained heads that reshape the default's forecast bands, and nearly
all of their gain is at D10. We split that gain by hour and by what the head changes, and scored a
night level offset with no head. Second, a depth-1 XGBoost refit (experiment 305) keeps most of
BG2's April-to-June advantage over arm D, the best-guess stack, at D5 and D9. We tried to find which
family of arm D's single-input terms accounts for that by swapping the refit's terms into arm D one
family at a time.

Quantile score is twice the mean pinball loss over 11 quantiles, in percentage points of effective
capacity; smaller is better. Stage 0 uses the 561,207 half-hours of experiment 251 (six Primaries,
2025-07-02 to 2026-06-30, rows outside hand-filtered events) with month-block intervals (12
blocks). Stage 1 uses experiment 305's April-to-June rows with ISO-week blocks (14 weeks). The plan
said month blocks for stage 1; this was changed before any stage 1 score was read, because 3 blocks
are too few. Rows are daily 09:00 UTC runs.

![D10 gain by arm](figure_1_d10_gain_by_arm.svg)

![D10 residual and gain by hour](figure_2_d10_hours.svg)

![Graft shares](figure_3_graft_shares.svg)

![Term curves](figure_4_term_curves.svg)

## Results

- **S0a held.** 88% of the pinball head's D10 gain is on night half-hours (21:00 to 07:30 UTC), with
  a 95% interval of 78% to 132%. The mixture head gives 85% (74% to 126%). Night gain is 0.278 points
  (0.130 to 0.456) and day gain 0.031 (-0.031 to 0.098).
- **S0b failed.** The as-of night offset (median residual of the previous 28 nights, mean -1.13
  points at night) scores 3.148 at D10 against the default's 3.169 and the pinball head's 3.024. It
  reproduces 14% of the head's gain (95% interval -16% to 99%). That is under the 30% line that
  means the head does something else, though the interval is wide.
- **The head narrows the bands.** The head's median equals the default's. At D10 night half-hours
  the 5% to 95% band narrows by 7.5 points on average and its centre moves up by 0.22 points; by day
  the band narrows by 3.5 points and the centre moves up by 0.55. Applying only the width change
  scores 3.077 (63% of the gain, 42% to 116%); applying only the centre shift scores 3.293, worse than
  the default (-86% of the gain, -237% to -30%).
- **By level.** At night the largest gains are at the 0.10, 0.25, and 0.40 quantiles (0.044, 0.057,
  and 0.040 points per level on the all-level mean).
- **The other five Primaries.** Together they gain 0.007 points (-0.014 to 0.026) under the pinball
  head and 0.012 (-0.005 to 0.028) under the mixture head.
- **Absolute D10 quantile scores.** Default 3.169, pinball head 3.024, mixture head 3.011, offset
  3.148. The default's six-Primary score, 2.579, matches experiment 314's gate.
- **Stage 1 gate failed.** A depth-1 XGBoost surrogate of arm D's median reproduces it with
  R-squared 0.82 to 0.93 (mean RMSE 4.6 points). Grafting all five families scores S = 0.018
  (-0.265 to 0.335) of the refit's advantage on the worst quarter at D5 and D9, against the 0.50 the
  plan required, and -8.3 on all daylight half-hours. The emulation is unusable, so no family is
  named.
- **Descriptive only.** On the worst quarter at D5 and D9 the solar graft scores S = 0.295 (0.116 to
  0.512), lags -0.250, calendar 0.008, temperature 0.151, other weather 0.142. These do not meet the
  gate. The curves in figure 4 show the refit's recent-power response is steeper than the
  surrogate's (about -13 to +12 points against -3 to +8), but arm D carries recent power through its
  own filter, which the surrogate cannot see.
- **Stage 2 not run**, by the stop rule: no single family reached S of 0.30 with an interval above
  it, and the three-family graft is under 0.50.

## Checks

- **Median minus mean forecast per site** (default, points): D5 0.01, D9 -0.75, D10 0.72, D11 -0.50,
  D14 -0.20, D16 -0.84.
- **Mean-forecast error beside the contrast.** At D10 the mean absolute error of the median is 5.362
  for both default and head (the head keeps the median). The mean absolute error of the mean of the
  11 quantiles is 5.680 for the default and 5.526 for the head, the same direction as the quantile
  score.
- **Triggers that fired:** 1 (the stage 1 emulation gate, a control, failed), 2 (S0b contradicts its
  prediction; the explanation tested is the width-only and centre-only arms above), and 7 (the
  ranked list's D10 night level term and pooled solar term depend on this result). The instruction
  for this session was Sonnet only, so no Opus audit was run, and the PR is left for the manager to
  decide on the audit.

## Caveats and next steps

- The offset's interval is wide (12 months, one substation), so "under 30%" is a point estimate.
- The offset used the shortest-lead residuals of the default, and a lead-matched or temperature-smooth
  offset was not tried.
- The graft is an emulation; the true graft needs arm D's own term contributions from its fitted
  additive model.
- Try next: a band-width correction for D10's night half-hours without a neural head; and for stage
  1, recompute arm D's term contributions, or refit the XGBoost with one family removed at a time.
