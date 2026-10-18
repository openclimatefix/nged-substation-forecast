---
id: "00092"
slug: lv0-drift-diagnostics-on-stored-predictions-demand
status: planned
size: simple
hypothesis: >-
  Most of the drift in Primary substation demand since 2022 is a per-series level or smooth trend,
  not a change of daily shape, so a slow per-series level term, not a new weather response, is
  what a long-history or pooled XGBoost model is missing.
prediction: >-
  The all-hours component of the drift (diagnostic D1) holds over half of the drift variance in at
  least 10 of the 16 Primaries, and diagnostic D6 puts over half of the weekly-minus-monthly refit
  gap in series-months flagged by drift (a step, a top-quartile slope, 5% capacity growth, or a
  switching event).
decision_rule: >-
  Supported when both parts of the prediction hold. Proceed to LV1 (level-normalised long history)
  when D1 or D2 shows drift of at least 1 point of capacity a year in at least a quarter of the
  series, or D4 or D5 links the loss to drift at r > 0.4. When none holds, the level hypothesis is
  weakened before any fit and AGE1 runs alone. Every interval is a month-block bootstrap. No
  threshold is needed for porting, because the experiment fits and ports nothing.
benchmark: none (diagnostics on stored predictions)
row_set_id: none (reads experiments 17, 24, 48, 78, 81, and 89 as stored)
metric: points of capacity (power over effective capacity, times 100)
---

**Plan.** The experiment fits nothing. It reads the stored predictions of experiments 17, 24, 48,
81, 89, and 78 and runs six diagnostics from `briefs/long_history.md`: D1 drift fingerprint by
hour band and season, D2 step census, D3 common versus own drift, D4 age effect, D5 pooling gap
versus drift, and D6 weekly-minus-monthly refit gap versus drift.

Size, answered for each of the five triggers: nothing new is stored beyond figures and one small
table; the production serving path is untouched; no degradation rule is touched; one design tests
the hypothesis; the callers of the code are `studies.*` helpers I name in the scripts.
