---
id: '00298'
slug: lsc-long-deep-detector-and-refit-cleaning
title: >-
  Does leaving detected load shifts out of the refit of a no-lag forecast stop it learning them as
  normal?
theme: lsc-detectors
also_relevant_to:
  - retraining-and-training-history
status: done
size: >-
  complex by the sizing triggers, run with the reviews the maintainer's brief allows (Sonnet only).
  Stored: private daily-residual panels, alarm tables and per-world bias tables under
  data/ml-experiments/00298-lsc-long-deep-detector-and-refit-cleaning/ only. Serving path: not
  touched. Degradation rule: not touched. Designs: more than one is defensible for the daily
  statistic, the depth and end rules, and the cleaning cap, so all are fixed below before any score
  is read. Callers: this folder's scripts, which call the reviewed `studies.dp_gam` fit and predict
  functions unchanged and copy experiment 274's and 290's harness patterns.
hypothesis: >-
  A trivial detector on the daily mean residual of the lagless convex additive model (GAM_ONLY)
  finds load shifts that last days to weeks and cover at least 10% of a Primary substation's load,
  which the existing hour-scale detectors miss, and excluding the days it flags from the monthly
  refit stops the forecast learning those shifts as the normal pattern.
prediction: >-
  Written before any score is read. (1) The best detector among A to C finds at least 12 of the 17
  unlabelled block days and at least 15 of D8's 29 drops (or the engine evidence of experiment 291
  says the drops are not load shifts), at no more than 2 false alarms per Primary-year in even
  months. (2) The neighbour-confirmed variants D to F cut false alarms per Primary-year by at least
  half at no more than 20% recall loss among events with metered partners, and reject at least 80%
  of injected permanent steps. (3) Cleaning with the best variant cuts the share of a 30-day, 25%
  shift learned as normal from the no-cleaning share to under 5%, at a rise in mean absolute error
  on clean rows of under 0.01 points. We expect (1) to fail on the block days, which experiment
  181 found are 12 hours or
  less and 1 to 4 noise units tall, so that a daily mean cannot see them.
decision_rule: >-
  Pre-registered before any score is read; Bonferroni level 98.75% across the four contrasts below.
  C1: the number of the 17 block days and of the 29 drops found (an alarm span covering any part
  of the event's UTC days, or within one day of it), and false alarms per Primary-year in even
  months, for the variant among A to C that has the highest recall on odd-month injected shifts at
  the threshold that gives 2 false alarms per Primary-year in odd months. C2: for D to F against
  the same variant, the ratio of false alarms per Primary-year and the difference in recall among
  events whose donor has at least one metered neighbour, and the share of injected permanent steps
  rejected. C3: the share of a 30-day, 25% shift learned as normal in the 48 hours after its end,
  oracle-cleaned and detector-cleaned against uncleaned, over 10 Primaries and 4 onset dates. C4:
  the change in mean absolute error on rows with no shift, as the difference in mean absolute
  error against the clean
  series between the cleaned and the uncleaned forecast. Intervals resample Primaries and whole
  events (percentile bootstrap, 2,000 resamples). The detector is `supported` when predictions 1
  to 3 all hold, `partly supported` when the cleaning prediction (3) holds with some detector
  variant, `not supported` otherwise. The result is a statement about whether long shifts stay a
  stated limit, so there is no porting work to weigh.
benchmark: >-
  Experiment 274's no-cleaning cell for the share learned (19% for 30 days, 25%), recomputed in
  this experiment with unmasked fits and 4 onset dates; experiment 290's range of 9% to 29%;
  RO-NC (the refit-offset neighbour-count detector), the pair alarm and the learned score of
  experiment 195 on the same real events.
row_set_id: >-
  Own rows: 16 Primary substations' daily residuals from 2022-01 to 2026-06 (real record), and 10
  Primaries at 4 onset dates (injected); not comparable with the leaderboard
metric: >-
  Detector: recall by event class, false alarms per Primary-year, onset and end error in hours.
  Cleaning: share learned as normal (forecast trained on the injected series minus the forecast
  trained on the clean series, divided by the mean injected transfer, over the 48 hours after the
  end) and mean absolute error against the clean series.
code: ml-experiments/00298-lsc-long-deep-detector-and-refit-cleaning/
verdict: not supported
best_arm: >-
  Cleaning with the pair-test detector E3 for the 30-day, 25% case: share learned as normal 10.1%
  (95% interval 6.9% to 13.2%), against 19.4% with no cleaning and 2.8% with the oracle.
headline: >-
  Detectors A to C and E3 on the daily mean residual match 10 to 12 of the 19 long-and-deep real
  load shifts, no more than LR_b20 (12). Leaving the days a detector flags out of the refit cuts
  the share of a 30-day, 25% transfer learned as normal from 19.4% to 10% to 13%, against 2.8% for
  the oracle.
lessons: >-
  Leaving flagged days out of the refit works only as far as the detector is early and complete.
  The oracle shows the ceiling (2.8% learned), and the 6 to 9 points the detectors recover are a
  third to a half of the gap. The neighbour check cuts false alarms by about 70% (68% to 73%) but
  loses 27% to 56% of the transfers that have a metered partner.
take_home_message: >-
  A simple detector on the daily mean error of the forecast that uses no recent power finds about
  as many of the long, deep real load shifts as the learned score LR_b20 (10 to 12 of 19 for four
  of the six variants). The cumulative-sum variant B also finds 27 of the 29 multi-day drops at
  D8, which are probably a gas engine and not load shifts; variant C finds 11. Leaving the days a
  detector flags out of the monthly refit cuts the share of a 30-day transfer that the forecast
  learns as normal from about 19% to about 10% to 13%. Excluding the true shift days instead cuts
  the share to 2.8%. The detectors recover less because each needs 1 to 3 days to fire and misses
  about 40% of long injected transfers and over half of the real ones. Tell NGED that transfers of
  14 days or more are partly learned as normal, and try a detector that fires on the first day
  next.
advice:
  - >-
    Expect the forecast to learn the first 1 to 3 days of every transfer as normal, because a
    daily-mean detector fires 24 to 48 hours after a transfer starts (Part 1, timing bullet).
  - >-
    When testing transfers of 90 days or longer, allow the excluded run to exceed 60 days, or keep
    excluding until the daily residual returns to its normal range; the 60-day cap alone left 27%
    learned, and no other cap was tested (Part 2, long shifts and steps).
  - >-
    Do not use the pair test E3 as a neighbour check: its recall barely changes with no metered
    partner (73%), with one (81%), and with a shifted-week placebo (74%) (Part 1, pair-test
    bullet).
  - >-
    Use variants D and F, not E, when a neighbour confirmation is wanted, and expect to lose 27%
    to 56% of the transfers that have a metered partner at the same threshold (Part 1, neighbour-
    checks bullet).
  - >-
    Judge a detector for long shifts on the 19 long-and-deep real shifts and on the D8 drops, not
    on the block days: no daily-mean variant found more than 3 of 17 block days (Part 1, real-
    record bullet).
production_status: not-for-production
adopted_in: none
---

**Six detectors that read the daily mean forecast error find about as many long, deep real load
shifts as LR_b20, a learned score from [experiment
302](../00302-lsc-learned-score-without-nged-labels/result.md), and leaving their flagged days out
of the monthly refit cuts what a 30-day transfer teaches the forecast by a third to a half, but not
to the 5% that was pre-registered.** A load shift (also called a transfer) is a spell in which an
operator moves part of one Primary substation's demand onto a neighbouring Primary. Primary
substations step 33 kV down to 11 kV. We used the generalised additive forecast with no lagged power
(GAM_ONLY), refitted every month. Detectors A to C test the daily mean of its error. Detectors D to
F add a check against the neighbouring Primaries. We scored them on the real record and on transfers
that we added to the demand series. We then refitted GAM_ONLY with the days a detector had flagged
left out of training. The "share learned as normal" is the part of an added transfer that the
refitted forecast absorbs into its idea of normal load. The "oracle" is cleaning that leaves out
exactly the true transfer days. The verdict is `not supported`, because pre-registered predictions 1
and 2 failed, and prediction 3 failed on its 5% target. The oracle shows the ceiling: leaving out
the true days cuts the share learned to 2.8%.

## Detectors on the 19 real load shifts that are long and large

The headline comparison is on the 19 real load shifts that last at least 2 days and move at least
10% of load, the group that [experiment
302](../00302-lsc-learned-score-without-nged-labels/result.md) fixed before this experiment's
results were read. The other 16 of the 35 labelled real transfers over 1 day are "other". "Matches"
counts shifts with an alarm during them or within a day. "Alarms per match" is every alarm over the
whole record divided by the matches. "MW-weighted" weights each match by the shift's recorded
megawatts. `printed_stratum.txt` holds the full table. RO-NC is the refit-offset neighbour-count
detector.

| Detector | Matches, 19 long and large | Alarms per match | MW-weighted | Matches, other 16 | MW-weighted |
|---|---|---|---|---|---|
| LR_b20 (experiment 302) | 12 | 8.8 | 0.72 | 9 | 0.69 |
| RO-NC, 2nd operating point | 12 | 10.8 | 0.71 | 10 | 0.80 |
| RO-NC, carried operating point | 16 | 18.1 | 0.86 | 13 | 0.92 |
| Learned re-ranker | 18 | 21.3 | 0.95 | 14 | 0.94 |
| A3 (mean of 3 days) | 10 | 13.3 | 0.65 | 4 | 0.43 |
| B (cumulative sum) | 10 | 15.0 | 0.66 | 3 | 0.18 |
| C (3 days against 28) | 11 | 11.4 | 0.67 | 5 | 0.38 |
| E3 (pair test) | 12 | 11.4 | 0.68 | 2 | 0.09 |
| D-A3 (neighbour check) | 8 | 16.4 | 0.54 | 4 | 0.34 |
| F-C (neighbour cover) | 9 | 13.8 | 0.58 | 4 | 0.22 |

**On the 19 long and large shifts the daily-mean detectors match 8 to 12, against 12 for LR_b20,
with more alarms per match; on the other 16 shifts they are clearly worse.** RO-NC and the learned
re-ranker match more at their carried operating points, but at 18 to 21 alarms per match.

![Real load shifts matched](real_matches_by_stratum.svg)

The same split on the injected shifts (25% of load, no metered partner, even-month onsets, 2 or more
days against 1 day): variant C finds 62% of 1,440 long shifts (3.2 alarms per match, MW-weighted
0.85) and 13% of 240 one-day shifts. E3 finds 61% and 6%. F-C finds 55% and 8%.

## Part 1: the detectors

![Injected recall by length](injected_recall_by_duration.svg)

- **Real record, tuned to 2 false alarms per Primary-year in odd months.** A "world" below is one
  injected-transfer scenario at one Primary. Variant C, which odd-month injected transfers picked as
  the best any-alarm variant, found 0 of 17 block days, 11 of 29 D8 drops, and 16 of the 35 real
  transfers, with 1.48 false alarms per Primary-year in even months. Variant B found 27 of 29 D8
  drops and 1 of 17 block days at 1.59 false alarms. The block days are 12 hours or less, so a daily
  mean cannot see them. Prediction 1 fails on the block days and on the drops for C (11 against 15
  needed), and holds on the false-alarm rate. The drops are probably a gas engine ([experiment
  291](../00291-d8-drops-engine-spells-load-shift/result.md)), so they are not counted as load
  shifts.
- **Neighbour checks at the same threshold** cut false alarms by 68% to 73% (D-A3 against A3: 76 to
  24 over the record), but lose 27% to 56% of the 22 real transfers that have two metered sides.
  Prediction 2's limit of a 20% recall loss fails.
- **The pair test E3 does not check neighbours.** On injected transfers its recall is 73% with no
  metered partner, 81% with one, and 74% with a shifted-week placebo. D and F separate better (F-C
  at the same threshold: 65% with the partner, 42% with none, 39% with the placebo).
- **Permanent steps are not rejected.** Variants D to F alarm on 50% to 99% of injected steps whose
  receiver is unmetered, against the 20% or less that prediction 2 allowed.
- **Timing.** For found injected shifts of 6 days or more, the median alarm comes 24 to 48 hours
  after the start for A2, A3, C, E3, and F-C, and 72 to 96 hours for A7 and B. The median start
  error is 0 hours at day resolution, and the median end error is 0 hours except for F-C (-48 hours)
  and D-C (-24 hours).
- **Depth.** Recall for 7 to 90 days at 10% of load is 16% to 44%, at 25% it is 66% to 83%, and at
  50% it is 66% to 92%.

## Part 2: cleaning the refit

![Share learned by length](share_learned_by_duration.svg)

| 30 days, 25% of load | Share learned, 48 h after the end | 2 to 33 days after |
|---|---|---|
| No cleaning | 19.4% (18.2 to 20.5) | 12.0% |
| Oracle | 2.8% (0.4 to 5.1) | 2.0% |
| Detector C | 13.0% (10.4 to 15.7) | 6.0% |
| Detector E3 | 10.1% (6.9 to 13.2) | 4.5% |
| Detector F-C | 13.2% (10.3 to 16.0) | 5.9% |

- **Paired difference from no cleaning:** oracle -16.6 points (-18.9 to -14.4), C -6.3 (-9.3 to
  -3.6), E3 -9.3 (-12.6 to -6.1), F-C -6.1 (-9.2 to -3.5). Prediction 3's target of under 5% fails
  for every detector.
- **Short shifts.** Cleaning removes most of the learning for 3 to 14 days (14 days: 18.1%
  uncleaned, 3.9% oracle, 6.1% to 7.4% detectors). A 7-day, 25% shift is learned at 9.3% with no
  cleaning and at 3.0% to 3.4% with a detector.
- **Long shifts and steps.** A 90-day shift is learned at 32% with no cleaning and at 22% to 28%
  with the oracle or any detector, because the run of excluded days is capped at 60. A permanent
  step of 25% is 45% learned after 12 months with no cleaning, 24% with the oracle, and 24% (E3) to
  34% (F-C) with detectors.
- **Clean rows.** On the 14 Primaries with no shift, cleaning changes the mean absolute error by
  +0.013 (-0.028 to +0.068) points for C, +0.007 (-0.030 to +0.053) for E3, and -0.008 (-0.033 to
  +0.024) for F-C. Prediction 3's limit is a rise of under 0.01 points, which E3 and F-C meet and C
  does not. D8 (+0.31 for C) and D12 (+0.16 to +0.26 for E3 and F-C) carry the positive values.
- **Self-reinforcement.** Refitting without the flagged days and re-running the detector on the new
  residuals flags 74% to 97% of the same days again, so the exclusion is stable. The recheck uses
  in-sample residuals for the days that stayed in training.
- **Wrong exclusions.** Over the whole run each detector flags 7.5 to 8.2 days per 30-day world, and
  5 to 7 days per world of 1 to 7 days, outside the true shift.
- **Near-a-shift scores** against the true normal-arrangement series (30 days, 25%): mean continuous
  ranked probability score 4.66 points with no cleaning, 4.54 for the oracle, and 4.59 to 4.61 for
  the detectors. The threshold-weighted score is 0.63 for every arm. The spread is a fixed Gaussian
  width fitted to the clean forecast's own error, so these scores rank the centres only.

## Checks

- **Median forecast against mean forecast.** GAM_ONLY forecasts the mean only, so there is no median
  to compare with it.
- **Error of the mean forecast beside the scores.** The mean absolute error against the true series
  falls in step with the continuous ranked probability score (6.26 uncleaned, 6.11 oracle, 6.17 to
  6.20 detectors).
- **Audit triggers fired.** A control failed: E3 does no neighbour check (73% with no partner, 81%
  with one, 74% with the placebo). The result contradicts the pre-registered predictions, and the
  explanation we tested, the detector's delay of 1 to 3 days, fits but was not isolated. A nested
  arm lost to the arm it contains: the oracle, which should contain every detector arm, leaves 27.4%
  learned for a 90-day, 25% shift against 22.3% for E3 (intervals 24.6 to 30.6 and 17.9 to 26.3),
  because the oracle's run of excluded days is capped at 60 while the detectors' separate runs are
  not.
- **No Opus audit was run,** because this session was limited to Sonnet models.
- **Reviews.** One Sonnet code review found 12 issues. We fixed four (a pair-noise cache shared
  between worlds, a placebo shifted by less than the transfer, hits that ignored the sign of the
  alarm, and stored-alarm counts on a different basis) and documented the rest below. One Sonnet
  review checked the numbers and one checked the prose.

## Caveats and what to try next

- Thresholds for D-A7, D-C, and F-A7 reach the bottom of the grid, so they are not at equal
  false-alarm rate with A to C. Recall on the real record is counted in all months, so it is partly
  in the tuning months. Variant B's long spans count as hits on any overlap.
- The hindsight start moves back by up to the window length minus one (6 days for A7 and E7), not
  the planned 2. The partner in a metered world is refitted but never cleaned, and the load level
  used by the depth rule is the clean level.
- Permanent steps start at the four fixed onsets and run 15 months, not 3, 6, and 12 months before
  the end of the data. The weather is ERA5 and CAMS, which arrive days late, so these detectors had
  better weather data than a live service would have, and live results will be worse.
- D15 has no panel because its history starts in 2025, so 15 of the 16 Primaries are scored.
- **Next.** The gap between the detectors (10% to 13% learned) and the oracle (2.8%) is mostly the 1
  to 3 days before a detector fires and the long transfers it misses. The result points to the next
  experiment: a detector that fires on the first day, or a refit that downweights the most recent
  days when the residual is high.
