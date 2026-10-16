# Experiment 41, part 2 (SW2-0b): which logged switching jobs show up in the power readings?

We asked which of the electricity network's logged switching jobs since January 2022 really moved
load from one substation to another, so that later experiments have a list of real events to test
detectors against. We turned each job into a short summary card of power readings and let two
independent runs of the AI model Sonnet read every card, say whether load visibly moved, and say
when it started and ended. A third run of a stronger model (Opus) settled the 66 cards where the two
Sonnet runs disagreed or were unsure. We hid 200 cards with known answers among the real ones to
check the labellers kept working. Of 537 logged jobs lasting 2 hours to 60 days, 66 (12%) show a
visible transfer. The labellers cannot see small or short transfers, so most of the other 440 jobs
are not proven quiet.

![Excess power around four labelled real operations](figure_3_real_examples.svg)

## What was labelled

**The set holds 1,017 cards: 817 real and 200 hidden known-answer cards.** The real cards are 537
logged operations of 2 hours to 60 days (jobs that overlap in time and share a substation are one
operation), 42 edge cards for the first and last 72 hours of 21 closures over 60 days, 50 closures
under 2 hours, 150 alarms raised by experiment 37's detectors where no closure was logged, and 40
quiet windows with no alarm. An alarm card puts the alarm's time where the log time would be and
draws a duration from the real operations. The hidden cards are 100 placebo windows, 80 injected
transfers, and 20 injected transfers with the log time put 1 to 3 hours off, built as in
[experiment 41's gate](result.md).

![Final labels by card kind](figure_4_labelled_counts.svg)

## The hidden cards match the real ones, except in missing data

**A check on the card text found the hidden cards match the real operations on every field except
the amount of missing data.** The check compares real operation cards with hidden cards on the
minute of the log time, whether side B is unmetered, the clock hour, the duration, the number of
neighbour rows, the card length, and the unavailable values per hourly row. Each of the first six
fields passes a two-sample test at p of 0.24 or more. The hidden cards were drawn to match the
real cards' missing data, but real cards still show 1.6 unavailable values per row against 0.9 on hidden cards, because real jobs sit among
more gaps in the telemetry.

## Hidden cards: labeller 1 against the known answers

**The first Sonnet run called none of the 100 placebo windows visible and found 0.60 of the
injected transfers.** Large, long transfers (25% or more of the receiving load, 6 hours or more)
were found in 19 of 21. The recall of injected transfers by unavailable values per row, with the
number of hidden cards in each group, is in the table.

| Unavailable values per row | Injected found | Placebo called visible | Share of real operations |
|---|---|---|---|
| under 1 | 26 of 42 | 0 of 62 | 0.49 |
| 1 to 2 | 13 of 22 | 0 of 27 | 0.23 |
| 2 to 3 | 9 of 13 | 0 of 5 | 0.14 |
| 3 or more | 0 of 3 | 0 of 6 | 0.14 |

**Recall does not fall steadily with missing data, but only 3 hidden cards sit in the most
missing-data group, so weighting by the real mix is not reliable.** Weighted to the real operations'
mix, the recall is 0.53 (against 0.60 unweighted), and the placebo rate stays 0 of 100. The
weighted figure rests on 3 cards in its last group.

## Real cards

- **Operations:** 537 operations, with 35 conserved (two substations step in opposite directions),
  31 one-sided (one substation steps), 440 not visible, and 31 that cannot be assessed because power
  readings are missing. The median transfer is 1.0 MW (10th to 90th percentile 0.55 to 2.5 MW) and
  0.30 of the donor's usual range.
- **By length:** of the 66 visible operations, 16 last 2 to 6 hours, 15 last 6 to 24 hours, and 35
  last over 1 day.
- **By size:** a share of the receiving substation's load can be computed for 16 of them: 2 under
  10%, 6 from 10% to 25%, and 8 of 25% or more. The other 50 have no metered receiver named by the
  labellers.
- **Edge cards:** 5 of the 42 are visible (1 at the start, 4 at the end), and 16 cannot be assessed.
- **Short closures:** 2 of 50 closures under 2 hours are visible and 10 cannot be assessed. Those
  closures are in the don't-care bands of the gate.

## Alarms: how many unexplained detector alarms are real

**Of 150 alarms with no logged closure, 9 (6%) are judged to show a transfer, and 0 of 40 quiet
windows do.** The 9 are 4 conserved and 5 one-sided. This sets a lower bound on the detectors'
precision of 0.06, because the logs miss real switching, and the labellers cannot see every real
transfer. By detector (37 alarms each): the 8-stamp D0 gives 4 of 37, the 8-stamp D1 gives 2 of 37,
the 48-stamp D1 gives 3 of 37, and the 48-stamp D0 gives 0 of 37. By the duration drawn for the
card: 4 of 56 for 2 to 6 hours, 2 of 40 for 6 to 24 hours, and 3 of 52 for over 1 day. Size bands
are not reported, because only 3 of the 9 have a share of receiving load. The 9 visible alarms have
a median transfer of 2.5 MW.

## What the 440 not-visible operations tell us

**None of the 440 not-visible operations is a proven quiet job.** Sonnet's recall on hidden
transfers is below 0.80 in every band that lasts under 1 day, so the 331 not-visible operations of 2
to 24 hours (210 of 2 to 6 hours and 121 of 6 to 24 hours) are don't-care operations. Of the 109
that last over 1 day, only a transfer of 25% or more of the receiving load would be seen with a
recall near 0.83, so even those exclude only large transfers. The 31 operations that cannot be
assessed are also don't-care.

## Disagreements and cost

- **Agreement:** the two Sonnet runs agree with Cohen's kappa 0.78 on the four labels and 0.83 on
  visible against not visible, over 529 cards (every card the first run called visible or low
  confidence, plus a random quarter of the rest).
- **Opus:** adjudicated 66 cards (52 disagreements and 14 agreed low-confidence visible calls).
  After adjudication 15 are conserved, 17 one-sided, 25 not visible, and 9 cannot be assessed.
- **Cost:** 20.1 million subagent tokens: 12.3 million for the first run on all 1,017 cards, 6.8
  million for the second run on 529, and 1.1 million for Opus. The manager relayed the
  maintainer's go-ahead for SW2-0b with an estimate of about 15 million tokens, with a second
  labeller on half the cards; the second labeller ran on 529 cards (52%) and the total was a third
  higher than estimated, because each Sonnet run cost 12,000 to 13,000 tokens per card.
- **Total for experiment 41:** 35.0 million subagent tokens across the gate (14.9 million) and
  this set (20.1 million).

## What this does not show

- **The labels depend on two runs of one model.** Both labellers are Sonnet with the same brief, so
  a shared blind spot would not show as disagreement.
- **The known-answer cards are easier than real cards on missing data.** The hidden cards carry
  fewer gaps (0.9 against 1.6 values per row), so their recall flatters the real cards.
- **Alarm cards use a drawn duration.** An alarm is not a closure, so its onset and end labels have
  no logged times to check against.
- **The set holds 66 visible operations and 82 visible cards in all.** Interval estimates on any
  F1 by band will be wide.

The labelled set is `data/ml-experiments/00041-sw2-0a-labeller-gate/real/labelled_set.parquet`
(private, one row per card, with the labels, the moved MW with its 10th to 90th percentile, and
the two moved-load shares computed in Python). Code: `ml-experiments/00041-sw2-0a-labeller-gate/`,
with the numbers printed by `e00041_labelled_set.py`, `e00041_finalize.py`, and
`e00041_sentinel_check.py`.
