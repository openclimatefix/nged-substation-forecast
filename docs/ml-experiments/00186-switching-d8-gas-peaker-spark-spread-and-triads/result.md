---
id: "00186"
slug: switching-d8-gas-peaker-spark-spread-and-triads
theme: switching-detectors
also_relevant_to:
  - differentiable-physics-embedded-generation
status: planned
size: >-
  medium. Stored: private tables and public price files (under 100 MB) under
  data/ml-experiments/00186-switching-d8-gas-peaker-spark-spread-and-triads/ only. Serving path:
  not touched. Degradation rule: not touched. Designs: more than one is defensible (how to match
  ordinary days, which efficiency, which flatness test), so this plan fixes one of each before any
  result is read. Callers: this folder's scripts, which read experiment 159's stored residual panel
  and drops and experiment 185's public price files.
hypothesis: >-
  A gas-fired generator (an embedded peaker or small combined-heat-and-power engine) below D8 runs
  when its margin is positive, and its output lowers D8's net power by a roughly constant block of
  1.2 to 5.4 MW for hours to days. D8's multi-day drops (experiment 159) therefore look like flat
  on/off blocks, start and stop with the spark spread, and may concentrate in winter evening peaks
  when the plant chases triads.
prediction: >-
  Written before any result is read. Drop days have a positive-spread share at least 20 points above
  matched ordinary days (95% resampled interval excludes zero), and at least 20 of the 29 drops are
  flat blocks (depth varying under 25% within the drop). I expect this to fail: experiment 182
  found the shortfall deeper at midday in 24 of 29 drops, which a gas plant does not predict, and
  experiment 185 found the system price explains almost none of the forecast's daily loss.
decision_rule: >-
  Both halves of the prediction must hold at the central efficiency (40%) for the hypothesis to be
  supported. If neither holds, the peaker hypothesis is not supported and the page asks NGED for the
  plant's name or the operator's contact. If exactly one holds, the verdict is `partly supported`.
  Spark spread is the system price minus gas price (National Gas daily system average price,
  pence per kWh, times 10 for pounds per MWh thermal) divided by electrical efficiency (35%, 40%,
  45%), minus a variable operating cost of 5 pounds per MWh. Matched ordinary days share D8's day of
  week and season (month window of plus or minus 30 days) and day-mean temperature within 2 degrees,
  1,000 draws of one matched day per drop day. No threshold needs porting cost, since the output is
  a diagnosis, not a model change.
benchmark: >-
  Matched ordinary days at D8, the same summaries at D3, D14, and D5, and experiment 185's system
  price alone.
row_set_id: none (describes stored residuals, no forecast is scored)
metric: >-
  Share of drop days with a positive spread minus the matched-day share (percentage points, 95%
  resampled interval); drops that are flat blocks out of 29.
code: ml-experiments/00186-switching-d8-gas-peaker-spark-spread-and-triads/
date: 2026-10-22
---

Planned. The write-up replaces this page when the scripts have run.
