# Log of research default versions

**This log lists every version of each family's research default, newest first.** The [rule
page](research-default.md) defines the families, the adoption conditions, and the eight items in
each row. The pull request of the adopting experiment adds the row, and a row is never edited
afterwards except to change its status.

## Primary demand

| Version | Change | Adopting experiment | Scored-year gain | Replicate gain | Absolute headline score | Benchmarks (BG2, `manual_heuristic`, `climatology`) | Stored forecasts | Code commit | Status |
|---|---|---|---|---|---|---|---|---|---|
| Primary demand research default 2 | Default plus the two-number solar summary S extended from 0.5 to 168 hours to 0.5 to 240 hours | [319](00319-primary-demand-default-1-solar-summary-and-heuristic-blend-at-7-to-10-days/result.md) | -0.064 points (2.0%) at 168 to 240 hours, -0.019 points at 0.5 to 240 hours (99.5% interval wholly below zero) | -0.034 points at 168 to 240 hours (95% interval -0.050 to -0.017) | 2.749 points at 0.5 to 240 hours, 3.152 at 168 to 240 hours | BG2 not scored past 168 hours; manual heuristic 3.414 and climatology 3.664 at 0.5 to 240 hours | The private files `work/arms/scored_long_lead.parquet` and `replicate_long_lead.parquet` of experiment 319 | `COMMIT` | Current |
| Primary demand research default 1 | The GAM stack A+H+S | [220](00220-dp-bg-stage4-assembly-with-drop-one/result.md) and [251](00251-dp-final-best-guess-assembly-clip-and-coast-against-bg2/result.md) | First version | First version | To be filled from experiment 251 when the first adoption is logged | To be filled with the first adoption | To be named with the first adoption | To be named with the first adoption | First version, set by the rule |

## Metered solar farms

| Version | Change | Adopting experiment | Scored-year gain | Replicate gain | Absolute headline score | Benchmarks (`manual_heuristic`, `climatology`, XGBoost reference) | Stored forecasts | Code commit | Status |
|---|---|---|---|---|---|---|---|---|---|
| Metered solar farms research default 1 | The spread-binned plant P2 with an additive band, a mixture-density neural network (five seeds averaged), and XGBoost, blended per quantile with sorted quantiles | [300](00300-metered-farm-best-guess-assembly/result.md) | First version | First version | Blend 5.27 points of capacity with each farm's own history and 5.45 with none (zero history) | 8.69, 8.64, and 5.45 (own history) and 5.56 (zero history) | The predictions of experiment 300, written once under its id | `d59e93445` | First version, set by the rule |

**Single-family scores of Metered solar farms research default 1** (quantile score in points of
capacity, own history then zero history): the mixture-density neural network alone 5.34 and 5.55,
the additive model alone 5.43 and 5.59, and plant-fed XGBoost (XGBoost on the plant forecast, its
ensemble spread, the unit mean, and the availability ratio) 5.45 and 5.56. The first production
update uses one model family, so these rows show what each family scores without the blend.

**The scored lead range of this row is 12 to 36 hours, and NGED's range is 1 to 10 days.**
Experiment 310 scores the row from 1 to 10 days. The frozen XGBoost reference is the XGBoost forecast
of experiment 243 with the floors of experiment 224.

## Wind farms

No version exists. The family accepts no adoption until its first default is named.

## Bulk and grid supply points

No version exists. The family accepts no adoption until its first default is named.
