# Log of research default versions

**This log lists every version of each family's research default, newest first.** The [rule
page](research-default.md) defines the families, the adoption conditions, and the eight items in
each row. The pull request of the adopting experiment adds the row, and a row is never edited
afterwards except to change its status.

## Primary demand

| Version | Change | Adopting experiment | Scored-year gain | Replicate gain | Absolute headline score | Benchmarks (BG2, `manual_heuristic`, `climatology`) | Stored forecasts | Code commit | Status |
|---|---|---|---|---|---|---|---|---|---|
| Primary demand research default 3 | Default 2 plus the experiment 319 heuristic blend on daylight half-hours only (sun above 5 degrees) at 168 to 240 hours | [322](00322-primary-demand-default-2-daylight-only-heuristic-blend-and-sunshine-graft-replication/result.md) | -0.075 points (2.4%) at 168 to 240 hours, -0.022 points at 0.5 to 240 hours (99.5% interval -0.151 to -0.028 at 168 to 240 hours) | -0.027 points at 168 to 240 hours (99.5% interval -0.079 to +0.008) | 2.727 points at 0.5 to 240 hours, 3.077 at 168 to 240 hours | BG2 not scored past 168 hours; manual heuristic 3.414 and climatology 3.664 at 0.5 to 240 hours | The private files `work/arms/scored_long_lead.parquet` and `replicate_long_lead.parquet` of experiment 322 (columns `d1_*`) | `ed338cd81` | Current |
| Primary demand research default 2 | Default plus the two-number solar summary S extended from 0.5 to 168 hours to 0.5 to 240 hours | [319](00319-primary-demand-default-1-solar-summary-and-heuristic-blend-at-7-to-10-days/result.md) | -0.064 points (2.0%) at 168 to 240 hours, -0.019 points at 0.5 to 240 hours (99.5% interval wholly below zero) | -0.034 points at 168 to 240 hours (95% interval -0.050 to -0.017) | 2.749 points at 0.5 to 240 hours, 3.152 at 168 to 240 hours | BG2 not scored past 168 hours; manual heuristic 3.414 and climatology 3.664 at 0.5 to 240 hours | The private files `work/arms/scored_long_lead.parquet` and `replicate_long_lead.parquet` of experiment 319 | `20f0afe6f` | Superseded by Primary demand research default 3 |
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

| Version | Change | Adopting experiment | Scored-year gain | Replicate gain | Absolute headline score | Benchmarks (`climatology`; XGBoost without the offset table; `manual_heuristic` and BG2 not scored for wind farms) | Stored forecasts | Code commit | Status |
|---|---|---|---|---|---|---|---|---|---|
| Wind farms research default 2 | The frozen physics plant of each wind farm, run on each of the 51 ensemble members, followed by a mixture-density neural-network head (experiment 347's arm y2). The maintainer decided on 2026-10-26 to adopt it although it fails the upper-tail condition: 2.28% of observations exceed q99 against a nominal 1%, and the one-sided 95% upper bound of the harm to the 0.99 pinball loss is +0.084 points over every lead against a limit of +0.05 | [347](00347-wind-physics-hybrid-plant-then-mdn/result.md), adopted by the maintainer's decision of 2026-10-26, overruling the upper-tail condition | -0.271 points (2.4%) over every lead from 0.5 hours to 15 days (99.75% interval -0.472 to -0.057); -0.234 points over true days 1 to 10 (99.75% interval -0.506 to +0.052, which includes zero) | -0.375 points over every lead (95% interval -0.535 to -0.243) and -0.376 over true days 1 to 10 (95% interval -0.563 to -0.226) | 11.081 points over every lead and 10.518 over true days 1 to 10, on the 96,479 kept rows of `fold_a_tuesdays_v1` | Wind farms research default 1 11.352 and 10.752; `climatology` 13.842 and 13.860; XGBoost without the offset table 11.697 and 11.102 | The private predictions of experiment 347, written once under its id | `2edd4e6b9` | Current |
| Wind farms research default 1 | The ERA5-trained XGBoost of experiment [33](00033-best-guess-architecture-vs-production/result.md) on the 51 ensemble members with the spread factor, plus the per-quantile offset table of experiment [259](00259-wind-plant-plus-quantile-offset-head/result.md) (experiment [303](00303-wind-adapter-in-production-setting/result.md)'s P1). The wind-farm experiments that followed (336, 338 to 340, 342 to 344, and 346 to 348) scored their changes against this default, and their pages call it "Wind farms research default 1" | [303](00303-wind-adapter-in-production-setting/result.md) | First version | First version | 11.352 points on the kept rows over every lead, and 10.752 over true days 1 to 10 (from experiment 347's refit) | XGBoost without the offset table 11.697 and 11.102; `climatology` 13.842 and 13.860; `manual_heuristic` and BG2 not scored for wind farms | The private predictions of experiment 303 (its arm P1), reproduced to three decimal places (11.352) by the refits in experiments 336 and 347 | `334c127e4` | Superseded by Wind farms research default 2 |

**The adoption of Wind farms research default 2 overrules the rule.** Experiment 347 passes the gain
condition over every lead and the replicate condition, and fails the upper-tail condition and the
gain condition over true days 1 to 10. A neural-network head given only the physics plant's median
and the farm reproduces 83% of the gain over every lead. The maintainer adopted the arm anyway.

**The scored lead range is 0.5 hours to 15 days, and NGED's range is 1 to 10 days.** The gain over
true days 1 to 10 has an interval that includes zero.

**Experiment 348 tested four tail fixes inside the head and found none that calibrates the top
quantile.** Its blend of the physics plant's residual band with Wind farms research default 1's
XGBoost quantiles gains less (0.147 points over true days 1 to 10) but leaves the 0.99 pinball loss
unchanged. The blend is a candidate for a later version and does not change this adoption.

## Bulk and grid supply points

No version exists. The family accepts no adoption until its first default is named.
