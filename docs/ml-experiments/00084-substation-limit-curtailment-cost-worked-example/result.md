---
id: "00084"
slug: substation-limit-curtailment-cost-worked-example
status: done
size: medium
hypothesis: >-
  The substation-limit curtailment metric (`studies.node_curtailment`) can run end to end on the
  pilot's two bulk supply points, Boston and Skegness, and the pilot holds enough half-hours beyond
  a scaled limit to rank forecasts. This is a worked example of new code, not a test of a
  forecasting idea.
prediction: >-
  None was registered before the run. The plan for the metric expected about 0 to 1 half-hours over
  the published thermal limit and 19, 46 and 323 over the scaled limits, so it expected no ranking
  at the thermal limit.
decision_rule: >-
  None. The experiment reports event counts first and says whether they are large enough to rank
  forecasts.
benchmark: >-
  The perfect forecast (every quantile equal to the metered value), and the NGED baseline: the
  manual heuristic's substation forecast with every generator assumed to run at full capacity. The
  baseline is the maintainer's unconfirmed understanding of NGED's practice.
row_set_id: fold_a_daily_gap_v1
metric: >-
  Avoidable cost in GBP at a limit of 0.9 times each node's earlier maximum export: the energy proxy
  at GBP 2,000 per MWh, Curtailment Limit 0, minus the perfect forecast's cost, over 2025-07-02 to
  2026-06-30.
code: ml-experiments/00084-substation-limit-curtailment-cost-worked-example/
verdict: inconclusive
verdict_note: >-
  A worked example with no registered contrast. At 0.9 times the maximum the cost rests on 19
  half-hours in one or two months, and every arm's month-resampled 95% interval reaches GBP 0.
best_arm: >-
  None at the headline limit. The best guess has the lowest avoidable cost at all three scaled
  limits (GBP 571,823 at 0.9 times the maximum), but its paired differences from climatology and the
  NGED baseline have intervals that include 0 at 0.9.
headline: >-
  Two bulk supply points have 19 half-hours beyond a limit of 0.9 times their earlier maximum in the
  scored year, so the pilot proves the arithmetic and cannot rank forecasts at the headline limit.
lessons: >-
  The metric runs and reproduces a hand-worked answer exactly. Ranking forecasts needs far more
  events than two substations give.
production_status: not-for-production
adopted_in: none
advice:
  - >-
    Do not rank forecasts by curtailment cost on the Boston and Skegness pilot: wait for the
    2,500-series delivery, or use a limit of 0.5 times the maximum for a rehearsal only (counts in
    this page's figure 1).
  - >-
    Score curtailment at a scaled limit, and show the published thermal limit as a reference only,
    because NGED curtails early to keep headroom for one more fault (this page, figure 2).
  - >-
    Label every curtailment cost "calibration to history, not a saving" (this page, figure 4).
---

# Experiment 84: the substation-limit curtailment metric on the pilot, a worked example

**We asked what it would have cost if the electricity network operator had capped nearby generators
whenever a forecast said a substation would export more than a limit, and whether two substations
hold enough such half-hours to tell forecasts apart.** We scored the stored demand and generator
forecasts of experiment 48 for Boston and Skegness over the year to 30 June 2026. NGED curtails
early, so that one more fault still leaves headroom, and never lets a substation approach its
published thermal limit. The limit we score against is therefore a stand-in for that headroom: 0.9,
0.75, or 0.5 times each substation's own maximum export before the scored year. The answer is that
two substations hold too few events to rank forecasts. At 0.9 times the maximum there are 19
half-hours beyond the limit. The next step is the January 2027 delivery of about 2,500 series, which
the same code can score unchanged.

**Verdict: inconclusive (no registered contrast).** Every cost below is a calibration to history,
not a saving. At 0.9 times the maximum, the month-resampled 95% interval of every arm reaches GBP 0.
The best guess has the lowest avoidable cost at all three limits, but at 0.9 its paired differences
from climatology and the NGED baseline have intervals that include 0. The published thermal limits
are NGED's, Supported by NGED Open Data. Questions for NGED are collected in [issue #1318 (Questions
to ask NGED about
curtailment)](https://github.com/openclimatefix/nged-substation-forecast/issues/1318).

## The event census comes first

![Half-hours beyond each limit, by node and season](figure_1_event_census.svg)

![Net export by season, with the limits marked](figure_2_export_distribution.svg)

**The counts are too small to rank forecasts at the headline limit.** Counts below are half-hours of
metered flow beyond the limit in the scored year, from cleaned unflagged power. The published
thermal limit is a ceiling NGED does not operate to, shown as a reference.

| Node | Published thermal limit (reference) | 0.9 x maximum | 0.75 x maximum | 0.5 x maximum |
|---|---|---|---|---|
| Boston | 1 (45.0 MW) | 17 | 36 | 215 |
| Skegness | 0 (29.8 MW) | 2 | 10 | 108 |
| Bicker Fen (sign unconfirmed, not scored) | 0 | 0 | 84 | 734 |

- **Scored nodes:** Boston and Skegness together have 19 half-hours at 0.9 times the maximum, 46 at
  0.75, and 323 at 0.5. Only at 0.5, half the observed peak, are there hundreds.
- **Import:** the scaled import counts are in figure 1. No published import limit is used, and
  curtailing generators cannot relieve an import limit, so import is counted and not priced.
- **Signs:** Boston and Skegness are -1 (a positive reading is import), from our own data. Bicker Fen
  is +1 and unconfirmed. Walpole's sign is unknown, so the metric excludes it.

## Does the code reproduce a hand-worked answer?

![One made-up half-hour through the whole chain](figure_3_known_answer.svg)

**Yes: all eight quantities agree exactly.** The example is made up (a 45.0 MW limit and two
generators of 10 MW and 20 MW). It includes a generator share cut to the generator's own forecast
and a residual breach, and `e00084_known_answer.py` fails if the code and the hand answer disagree.

## Cost per arm

![Avoidable cost per arm at three scaled limits](figure_4_avoidable_cost.svg)

**Absolute cost of the perfect forecast, and avoidable cost of each other arm, in GBP over the
scored year at GBP 2,000 per MWh (energy proxy, Curtailment Limit 0).** Avoidable cost is the arm's
cost minus the perfect forecast's. All figures are a calibration to history, not a saving.

| Arm | 0.9 x maximum | 0.75 x maximum | 0.5 x maximum |
|---|---|---|---|
| Perfect forecast (absolute) | 87,460 | 252,428 | 1,341,908 |
| Best guess (weekly retrained) | 571,823 | 2,449,203 | 9,385,270 |
| Best-guess stack | 1,572,512 | 4,889,923 | 13,161,772 |
| Climatology | 1,318,078 | 4,638,378 | 20,187,791 |
| NGED baseline (unconfirmed) | 1,323,425 | 4,136,392 | 18,073,166 |
| Best-guess nodes, actual generation | 779,645 | 2,718,739 | 11,574,478 |

- **0.9 times the maximum:** no interval excludes zero. The best guess minus the NGED baseline is
  -GBP 751,603 (95% interval -2,293,480 to +14,988), and minus climatology is -GBP 746,255
  (-2,288,392 to +30,815). The stack minus the best guess is +GBP 1,000,690 (0 to 2,865,606).
- **0.75 times the maximum:** the stack costs GBP 2,440,720 more than the best guess (96,371 to
  6,686,043), and the best guess is GBP 2,189,175 cheaper than climatology (-6,255,250 to 0).
- **0.5 times the maximum:** the best guess is GBP 10,802,520 cheaper than climatology (-20,880,964
  to -3,342,553), and GBP 8,687,896 cheaper than the NGED baseline (-26,251,726 to +2,339,946).
  This is a rehearsal at a limit nobody operates at.
- **Published thermal limit:** one Boston half-hour (0.038 MWh over) is scored, and every
  non-perfect arm costs GBP 1,540 against the perfect forecast's GBP 1,225.
- **DCUSA form and other prices:** the energy proxy scales with price, so other prices rank the
  arms as GBP 2,000 does. The DCUSA form charges each instructed generator its whole headroom and
  ranks the arms differently (figure 4, right).

## What to watch

- **Generators without a forecast:** a node row is scored whenever the node forecast and actual are
  finite. Of the 27,936 node rows scored, 100,632 generator-half-hours had no finite forecast and
  were not capped, so those generators gave no relief and no loss. None had a forecast with no
  metered output.
- **Forecast capped at capacity:** a generator forecast above its effective capacity is capped at
  the capacity, as `node_curtailment` requires. All arms are capped alike, and metered output above
  capacity stays in the score.
- **The NGED baseline is unconfirmed.** It is the maintainer's understanding that NGED forecasts
  substation flow only with the manual heuristic, so each generator is assumed to run at full
  capacity. NGED has not confirmed it.
- **Thermal limits:** Boston and Skegness use the smaller of the two transformers' published limits.
  Boston's all-history maximum export, 45.08 MW, sits on the 45.0 MW limit, so 45.0 MW may be an
  operating ceiling or a clipped meter.
- **Anonymisation:** no generator appears singly. Every generator figure is a sum over at least 4
  generators, in GBP or MWh, and the table joining generators to substations stays in the private
  data folder.

## What the next experiment should answer

**Whether a ranking appears once the January 2027 delivery gives about 2,500 series.** The same
function scores any number of nodes from longer tables.

Reproduce with `uv run python ml-experiments/00084-*/e00084_report.py`, then `e00084_known_answer.py`
and `e00084_charts.py`. The printed report is in
`data/ml-experiments/00084-substation-limit-curtailment-cost-worked-example/report.md`.
