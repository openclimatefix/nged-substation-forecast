---
id: "00042"
slug: label-exclusion-ceiling-lagless-demand
status: planned
size: medium
hypothesis: >-
  Training the lagless XGBoost demand model with the half-hours of labelled switching removed from
  its training target lowers its quantile score outside switching, by more than a random removal of
  the same number of half-hours does.
prediction: >-
  The labelled-minus-random gap is between -0.30 and 0 points of capacity on the series whose
  training target changes, for the ERA5-trained model (48 labelled visible events, 26 of them longer
  than 1 day, in its 2022-01 to 2025-06 window), and its interval includes zero when pooled over all
  20 series. The ENS-trained model has only 14 such events (9 longer than 1 day) in its 15-month
  window, so its gap is expected to be smaller and unresolved.
decision_rule: >-
  Four planned contrasts, C2 and C4 for each of two training sources (ERA5-trained, the headline,
  and ENS-trained, the secondary), on the quantile score outside switching, over all leads, on the
  series whose training target the exclusion changes (the "affected series"). C2 is the
  trusted-labelled arm minus the random arm, and C4 is the all-visible-labelled arm minus the random
  arm. Each is read at 98.75% (Bonferroni across four) with whole months and seeds resampled. The
  contrasts of each labelled arm against no exclusion (C1 trusted, C3 all visible) are reported
  without a verdict. A contrast counts as a gain when its interval lies wholly below zero. The
  verdict reads on C2 of the ERA5-trained source. Supported: C2 lies wholly below -0.10 points of
  capacity (the threshold: a detector would be a new pipeline step, so the gap must exceed what a
  step is worth). Small gain: C2 lies wholly below zero but not wholly below -0.10. Refuted: C2 lies
  wholly above zero. Inconclusive: the interval includes zero. If a source has fewer than 3 trusted
  events in its training window, that source's verdict rests on its C4 and the page says so. The
  page also gives the pooled-over-20-series equivalent of the threshold, and states before any
  verdict the minimum detectable gap implied by the seed spread of the no-exclusion arm; if that gap
  exceeds 0.10, an inconclusive verdict is the expected outcome.
benchmark: >-
  The same model with no exclusion, and the same model with a random exclusion of the same number of
  half-hours in blocks of the same lengths.
row_set_id: fold_a_every5d_v1
metric: >-
  Quantile score (twice the mass-weighted mean pinball loss over nine levels, in points of
  capacity), outside labelled switching and inside it; MAE of the median as a secondary.
pull_request: none yet
code: ml-experiments/00042-label-exclusion-ceiling-lagless-demand/
---

# Experiment 42: does removing labelled switching from the training target lower demand error?

**Plan, written before any fit.** We ask whether a demand forecast improves when the half-hours of
known switching jobs are left out of the data the forecasting model learns from. Switching jobs move
load between substations for hours or days, so the metered demand in those half-hours is not what
the substation normally draws. We train the same XGBoost (gradient-boosted trees) model three ways:
on all the data, with the labelled switching removed, and with the same number of randomly chosen
half-hours removed. If removing real switching beats removing random half-hours, the gap is the most
a detector that finds switching could gain by cleaning the training data. The same gap is also the
first evidence for a forecast of demand under the normal running arrangement (the demand with no
switching applied).

## Design

- **Model.** Experiment 00033's N1 recipe for Primary demand with the three power lag columns
  removed (the "lagless" model): XGBoost quantile model per series, 9 levels, 500 rounds, half-life
  2 years, applied to the 51 ENS members and pooled. Two training sources. The *ERA5-trained* source
  (headline) trains on ERA5 weather at valid times 2022-01-01 to 2025-06-30, as experiment 00033's
  `era5_retrained` arm does, because the labels start in 2022 and this window holds 48 labelled
  visible events. The *ENS-trained* source (secondary, experiment 00033's recipe, called N1 there)
  trains on the ENS control member with valid times 2024-04-01 to 2025-06-30. ERA5 arrives about 5
  days late, but both sources are scored on ENS members at serving time, so no past-weather lag
  enters. A lagless model keeps switching out of its inputs, so only the target is affected.
- **Rows.** The 73 runs of `fold_a_every5d_v1`, 20 demand series, valid times 2025-07-01 to
  2026-06-30, the rows of experiment 00033. Three model seeds (0 to 2).
- **Exclusion.** A training row (a run and a valid time) is dropped when its valid time lies in a
  labelled interval, `onset_utc` to `end_utc`, of an event, with the interval clipped to the
  training window (an event that ends after 2025-06-30 loses only its part before that date, which
  is allowed because this experiment is an upper bound that uses the labels' later end, and the
  report counts such events), for every series named as a side, the donor, or a receiver of the
  event. Nothing else changes. The trusted rule's `max_side_missing_share` comes from experiment
  41's labelling, which read the whole of each event, so it is oracle information too. No scoring
  row is touched. The labels are from experiment 00041 (SW2-0a and SW2-0b), real operations labelled
  `conserved` or `one_sided`.
- **Label sets.** *Trusted*: moved load of at least 25% of the receiving series' mean load over the
  365 days before onset (recomputed here, because the stored share is missing for 50 of the 66
  visible events), longer than 1 day, and `max_side_missing_share` of at most 0.5. The 13 conserved
  operations that rest on one substation with over half its readings missing never count as
  conserved; in the all-visible set they count as one-sided. *All visible*: every `conserved` and
  `one_sided` operation, any size and duration. Counted from the label file, the ERA5 window holds
  48 events (26 longer than 1 day, 147 event-days) and the ENS window 14 (9 longer than 1 day, 49
  event-days). Before scoring, the report prints the number of trusted events and the share of each
  affected series' training rows removed, because a gap that removes 0.1% of the rows cannot be
  interpreted. The recomputed share is printed beside experiment 41's `moved_share_of_donor_range`
  for the events that have both, as a consistency check.
- **Random control.** For each labelled series and label set, six random draws. Each draw keeps the
  number of excluded half-hours and the block lengths of the real events, measures the real block
  lengths after clipping to the training window, places the blocks at random starts inside the
  training window, in the calendar months of the real events' starts, that neither overlap each
  other, nor any labelled interval of any visibility, nor the window edges. Rejection sampling uses
  a generator seeded by the draw number and the series. The control therefore removes clean rows by
  design. The report gives the removed share of the recency weight per arm. Draw `i` is fitted with
  model seed `i mod 3`, so each seed carries two draws. The random arm's score per seed is the mean
  over its draws, and the report also gives C2 per draw and the spread across the six draws.
- **Affected series.** A series is affected under a label set when at least one event removes
  training rows from it. Unaffected series have identical training data in all arms, so the no
  exclusion predictions are reused for them, after a check: one unaffected series is refitted under
  a labelled arm with the production worker settings and must reproduce its no-exclusion predictions
  exactly, otherwise every series is refitted in every arm, and the pooled 20-series score is
  reported as well.
- **Scoring.** "Switching" in the scoring window is every half-hour inside a visible labelled
  interval of the series, from labels in 2025-07 to 2026-06. The headline scores the half-hours
  outside; the half-hours inside are reported separately and are few. Absolute quantile scores are
  reported for every arm beside the contrasts, with a month-by-half-hour `climatology` row and
  `manual_heuristic` for scale.
- **Compute.** The slot's 6 cores and at most 12 GB of the graphics card's memory (GPU): three
  parallel fit workers with 2 threads each. Each worker writes its arm and seed to its own file
  atomically.

- **Per-event view.** The report adds a descriptive gap for each affected series and each labelled
  event's scoring months, because events cluster in a few series and months and the month-resampled
  interval will be wide.

## Sizing

- **Stored:** predictions under `data/ml-experiments/00042-*` only, never committed.
- **Serving path:** not touched.
- **Degradation rule:** not touched.
- **More than one design:** yes (the trust rule and the random-block shape), so both are fixed above
  before the labels are joined to any score.
- **Spans unfamiliar code:** no. The fit copies experiment 00033's demand code and its scoring
  helpers.

Medium: one Sonnet plan review, a Sonnet code review for leakage and for the fit stopping at the
training end, a Sonnet check of the write-up against the printed report, and a Sonnet prose review.

## Earlier advice followed

- Experiment 00033 (advice on absolute scores and the interval rule): absolute scores are reported
  and a gain needs an interval over seeds that excludes zero.
- Experiment 00041: labels under 25% of the receiving load or under 1 day are not trusted, so the
  headline uses only larger, longer events and the rest are a sensitivity check.
- Departure: the lagless model is not experiment 00033's best guess, because the best guess uses
  lags that would carry switching into the inputs.

## Results

Not yet run.
