---
id: "00291"
slug: d8-drops-engine-spells-load-shift
title: >-
  Are D8's multi-day drops generator-engine spells?
theme: lsc-detectors
also_relevant_to:
  - differentiable-physics-embedded-generation
status: planned
size: >-
  simple. Stored: private tables under data/ml-experiments/00291-d8-drops-engine-spells-load-shift/
  only. Serving path: not touched. Degradation rule: not touched. Designs: one (the overlap, the
  placebo and the clustering statistic are fixed below before any number is read). Callers: this
  folder's scripts only; they read the stored drops of experiment 159, the engine detectors of
  experiments 196 and 199, and the alarms of experiments 130 and 195.
hypothesis: >-
  D8's 29 multi-day drops (experiment 159) are spells in which a gas engine of up to three units
  runs, and D8's own metered power no longer includes the engine's output (experiment 194). If so,
  the engine detector of experiment 196 flags the drops' half-hours far more often than matched
  windows elsewhere on D8, the drops' depths cluster at multiples of one engine unit, and most of
  D8's load-shift alarms that fall inside drops are engine edges.
prediction: >-
  Written before any result. At least 70% of the in-scope drops (28 of 29 start on or after
  2023-08-15; the first, in 2022, precedes the engine's appearance in D8) overlap engine-running
  states, meaning at least half of the drop's half-hours have the smoothed probability of one or
  more units above 0.5 under experiment 196's detector (arm v1). The share of overlapping drops
  exceeds the share in matched placebo windows on D8 (one-sided p below 0.05). The depths cluster
  at multiples of one unit of 1.97 MW: the mean distance from a depth to the nearest multiple, in
  units, has a 95% bootstrap interval wholly below 0.25, the value for a uniform spread. Expected
  weak points: the drops and the detector read the same D8 residual, so some overlap is built in;
  the detector over-flags D8 from 2025-03 (40% of half-hours), where 17 of the 29 drops lie.
decision_rule: >-
  Fixed before any number is read. The deciding arm is v1 (experiment 196's stored fixed-noise
  smoothed posterior for D8), on all half-hours of each in-scope drop. `supported`: overlap is at
  least 70% (Wilson 95% interval reported), the overlap count exceeds the matched placebo (below),
  and the clustering interval lies wholly below 0.25. `partly supported`: exactly one of the
  overlap-above-placebo test and the clustering test holds, or both hold only in the stratum from
  2025-03 onward (where the detector over-flags). `not supported`: the overlap is not above the
  matched placebo; the page then says the drops are not explained by the engine and D8's cause
  stays unknown. `inconclusive`: the matched placebo overlap is itself at least 90%, so the
  detector cannot tell the drops from other stretches. The night-only arm N (experiment 199's
  night evidence, drop share taken over night half-hours) is reported beside v1 and does not decide;
  a disagreement in direction is stated. Matched placebo: for each in-scope drop, a window of the
  same length shifted by a whole number of weeks, drawn uniformly from D8 half-hours between
  2023-08-15 and 2026-06-30 that overlap no drop and lie on the same side of 2025-03-01 as the
  drop; 5,000 draws of one placebo window per drop give the distribution of the overlap count. The
  same windows are scored on D5, D10, and D11 (where the detector over-flags) as a control. The
  clustering test uses 5,000 bootstrap resamples of the drops; the unit of 1.97 MW is the spacing
  of the four output levels of the recorded engine (0, 1.96, 3.93, 5.93 MW) over its 44 days of
  known answer, and 1.75 MW (experiment 196's fitted value) is a sensitivity. The pre-2025-03
  stratum (11 in-scope drops, where the detector's known-answer gate applies) is reported on its
  own. No change to production follows from this experiment, so there is no porting threshold.
benchmark: >-
  Experiments 159 (the drops), 194 (the engine's history and price-following), 196 and 199 (engine
  detectors), 130 and 195 (load-shift alarms); matched placebo windows on D8 and controls D5, D10,
  and D11.
row_set_id: >-
  none (describes stored residuals, detector states and alarms; no forecast is scored)
metric: >-
  Share of each drop's half-hours flagged as engine running; share of drops with at least half
  flagged; mean distance of depths and onset steps from the nearest multiple of one unit; share of
  load-shift alarms inside engine spells.
code: ml-experiments/00291-d8-drops-engine-spells-load-shift/
---

**Planned; the write-up replaces this page when the experiment has run.** The hypothesis,
prediction and decision rule above were committed before any number was read.
