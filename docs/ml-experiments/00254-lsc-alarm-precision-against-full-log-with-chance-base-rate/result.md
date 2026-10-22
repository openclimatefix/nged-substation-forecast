---
id: "00254"
slug: lsc-alarm-precision-against-full-log-with-chance-base-rate
title: >-
  How many LSC alarms are real load shifts, judged against the full log and a chance base rate?
theme: lsc-detectors
also_relevant_to: []
status: planned
size: >-
  simple. Stored: private tables under
  data/ml-experiments/00254-lsc-alarm-precision-against-full-log-with-chance-base-rate/ only.
  Serving path: not touched. Degradation rule: not touched. Designs: one defensible design, because
  the operating points are frozen from experiment 227. Callers: this folder's scripts only; they
  read the alarm tables of experiments 195 and 227 and the log of experiment 37.
hypothesis: >-
  Experiment 195's loose rule (a logged closure within 24 hours "explains" an alarm) overcounts real
  alarms, because a random window is explained about 45% of the time (experiment 159), and the
  strict rule (experiment 195's step-and-end match) undercounts them, because the log's times are
  approximate. Once the share expected by chance is removed, a large part of the learned alarm
  source LR_b20's alarms that are not strict matches still coincide with a logged closure.
prediction: >-
  Written before any table exists. For LR_b20, at least 30% of the alarms that are not strict
  matches have a logged closure on the same Primary substation or a connected neighbour starting or
  ending within 24 hours of the step, at an excess of at least 15 percentage points over the mean
  of the week-shifted base rate, so its corrected false-alarm rate is under 3 per matched load
  shift (against 4.0 if every non-match were false). It fails if LR_b20's log-near share is inside
  the 2.5% to 97.5% range of the shifted shares; the page then says that the unmatched alarms are
  false at the rate the strict count implies, and that the precision of an event table is about 1
  in 5.
decision_rule: >-
  Pre-registered. Sources, frozen from experiment 227 with nothing tuned: LR_b20 (the headline),
  LR_b80, RO_b80, and PG_g70, on all 16 series and the whole real record, each learned alarm taken
  from the fold that holds out its Primary pair. The unit is a row of experiment 227's interval
  table (alarms on one series that start within 24 hours merge), stamped at its step. Classes, in
  this order: strict match (experiment 195's rule: step within 2 hours of a logged onset of the 35
  labelled load shifts over 1 day, end within the larger of 6 hours and 10% of the duration);
  log-near (a closure of the 1,040-row log of experiment 37, kept at 2 hours to 60 days, on the same
  series or a connected neighbour, starts or ends within 24 hours of the step; 6 and 48 hours are
  reported beside); balanced-opposite candidate; D8 drop edge; block day; bank holiday or clock
  change; none. Chance base rate: every alarm of a series is shifted by the same whole number of
  weeks (circular within the record, 200 shifts of 3 to 100 weeks either way, so hour of day and
  weekday are kept), and the log-near share is recomputed. Excess = observed share minus the mean of
  the shifted shares, reported with the 2.5% and 97.5% points of the shifts. Corrected precision =
  (strict matches + excess log-near alarms among the non-matches) / all alarms; its interval comes
  from resampling whole calendar months of alarms (1,000 resamples) with each alarm's own chance
  probability. False alarms per matched load shift = (all alarms minus estimated real alarms) /
  strict matches. The same table is made for each Primary substation, with D8, D9, and D10 shown
  separately. The verdict is `supported` if LR_b20 meets both bars of the prediction, `not
  supported` if its log-near share is inside the shifted range, and `partly supported` otherwise.
  No port threshold applies, because the experiment changes no code in production; the numbers feed
  the event table for NGED and the harm accounting of later experiments.
benchmark: >-
  Experiment 227: LR_b20 matches 21 of 35 logged load shifts at 5.0 alarms per match, which counts
  every alarm that is not a strict match as false.
row_set_id: >-
  Experiment 227's alarm tables on the whole real record; no forecast rows, so no leaderboard
metric: >-
  Share of alarms in each class, excess log-near share over the week-shifted base rate, corrected
  precision, and false alarms per matched load shift.
---

Planned; the write-up replaces this page when the tables are done.
