# Calibration and uncertainty

**Where we tested it, the forecast's 5% to 95% band was already within a few points of its nominal
90% coverage, and simple corrections fixed solar's over-wide band but did not make any forecast
measurably more accurate.** Coverage is the share of measured values that fall inside a band, and a
calibrated 5% to 95% band covers 90%. The two gains that cleared their thresholds both came from
combining or conditioning models and not from recalibrating: pooling ERA5-trained quantiles over the
51 members of the ECMWF ensemble (ENS) beat a model trained on ENS weather by 0.62 points of
capacity for wind and 0.24 for solar, and averaging the quantiles of three stored models lowered the
Primary substation quantile score by 0.10 points (98.33% interval -0.19 to -0.03). Every result
rests on one validation year (July 2025 to June 2026) and a small set of series: 6 solar farms, 3
wind farms, and 16 Primary substations. A point of capacity is 1% of a series' capacity, and a lower
score is better.

## Does it work?

**Pooling the ERA5-trained quantiles over the ENS members beat a quantile model trained on ENS
weather, and a stage 2 calibrator on top did not help.** In [experiment
00002](../../00002-stage2-calibrator-and-mixture-vs-ens-quantile/result.md), the pooled mixture
scored 0.62 points lower for wind (95% interval -0.99 to -0.26) and 0.24 points lower for solar
(-0.36 to -0.06) on the continuous ranked probability score (CRPS). The stage 2 calibrator scored
worse than the mixture by 0.40 points for solar (+0.26 to +0.61), and by 0.26 for wind with an
interval that includes zero (-0.17 to +0.64). The experiment used no lagged power, so it cannot say
whether a calibrator that sees recent power would help.

**A single spread factor brought solar's band from over-covering to about 90%, and the CRPS did not
change detectably.** In [experiment
00016](../../00016-pooled-mixture-calibration-solar-wind/result.md), the uncorrected solar band
covered 94.3% to 95.3% of cases in every lead band. One factor (fitted 0.90 to 1.00, median 0.95)
moved coverage to 89.5% to 91.8%, with a CRPS change of +0.006 (99.4% interval -0.004 to +0.024).
Wind over-covered in the first two days only (95.3% and 94.4%), and no planned correction fixed
that. A spread factor on two lead groups, chosen after seeing the results, brought every wind band
to 90.7% to 92.5% and changed the CRPS by -0.021 (-0.042 to -0.002).

**For wind, the power-curve base margin carries the gain and the spread factor adds nothing
detectable.** In [experiment
00020](../../00020-solar-wind-stack-cams-stage0-spread-margin/result.md), the base margin lowered
the wind CRPS by 0.055 (99.58% interval -0.110 to -0.007), and the margin with the spread factor by
0.071 (-0.133 to -0.014). Dropping the spread factor from that stack changed the CRPS by +0.016
(-0.004 to +0.036). For solar, none of a CAMS-trained stage 1a model (-0.009), a stage 0 quantile
mapping (+0.036), or a spread factor (+0.006) helped detectably, and the full solar stack was 0.035
worse (-0.075 to +0.116). With 6 solar farms the solar intervals are 0.19 to 0.30 points wide, so a
gain under 0.1 points could not be resolved.

**The stored best-guess quantiles for Primary substations were already close to nominal, and no
recalibration fitted on earlier months beat them.** In [experiment
00097](../../00097-cal1-post-hoc-recalibration-of-bg2-quantiles-tails/result.md), the stored q05
covered 0.035 and the stored q95 covered 0.938 at 12 to 36 hours lead. No arm passed both
pre-registered tests, and the verdict is `refuted`. The experiment has only 9 scored months, so it
shows that the stored tails are close to nominal and cannot show that no recalibration could help.
The coverage of q95 swung from 0.88 to 0.97 between months, which a table fitted on earlier months
cannot follow, and the stored median covered only 0.45.

**Averaging the quantiles of three stored models lowered the Primary demand quantile score by 0.10
points, and solar and wind gains were unresolved.** In [experiment
00035](../../00035-lead-weighted-quantile-blend-of-stored-models/result.md), the blend of the
monthly-retrained N1 (a single-stage XGBoost quantile model trained on ENS weather), `a7_cs`, and
the stack scored 5.025 against 5.122 for the best member (-0.097, 98.33% interval -0.187 to -0.026).
Solar (-0.039, -0.115 to +0.031) and wind (-0.074, -0.194 to +0.055) had intervals that include
zero.

**Training on the CRPS of several ensemble members at once recovered only about half of the harm
that extra members do to the pinball loss.** In [experiment
00099](../../00099-crps1-crps-type-loss-over-ens-members-primary-demand-pilot/result.md), the CRPS
loss beat the pinball loss on the same 6 members per run by 0.039 points (95% interval -0.057 to
-0.018), a `small gain`. Training the pinball loss on 6 members instead of the control member made
it 0.052 points worse (+0.029 to +0.074). Against the control-member recipe the CRPS arm is +0.036
points (-0.022 to +0.111), so the pilot cannot say whether the CRPS arm is better or worse. Its 12
to 36 hour tails stayed outside 0.01 of nominal (q95 covered 0.924, q05 covered 0.061).

**Starting XGBoost from zero did not lower the quantile score of any production-style model.** In
[experiment 00080](../../00080-bm0-zero-base-margin-xgboost-quantile-models/result.md), the Primary
substation difference was -0.017 points (98.75% interval -0.072 to +0.054), and the solar farm
difference was -0.010 (-0.036 to +0.028). The verdict is `inconclusive`: no gain could be shown.
Four times as many boosting rounds scored 0.211 points worse (+0.100 to +0.352) at 8 Primary
substations.

![Coverage of the 5% to 95% band by lead band](../../00016-pooled-mixture-calibration-solar-wind/figure_1.svg)

**Figure 1: the uncorrected 5% to 95% band over-covers for solar in every lead band and for wind in
the first two days, and a spread factor corrects solar.** Source: [experiment
00016](../../00016-pooled-mixture-calibration-solar-wind/result.md).

![Wind and solar CRPS differences for each part of the stack](../../00020-solar-wind-stack-cams-stage0-spread-margin/figure_1.svg)

**Figure 2: the stack of changes helps wind and does not help solar.** Negative values are better.
Source: [experiment 00020](../../00020-solar-wind-stack-cams-stage0-spread-margin/result.md).

![Primary substation tail coverage of each recalibration](../../00097-cal1-post-hoc-recalibration-of-bg2-quantiles-tails/figure_1_primary_tail_coverage.svg)

**Figure 3: no recalibration of the stored Primary substation quantiles moved tail coverage by more
than its own interval.** Source: [experiment
00097](../../00097-cal1-post-hoc-recalibration-of-bg2-quantiles-tails/result.md).

![Blends minus the best single member](../../00035-lead-weighted-quantile-blend-of-stored-models/figure_1.svg)

**Figure 4: blending helps Primary demand, and the solar and wind gains are unresolved.** Negative
values are better. Source: [experiment
00035](../../00035-lead-weighted-quantile-blend-of-stored-models/result.md).

## Best design so far

**For Primary substations, the best design is a blend of the stored quantile models and no
recalibration.** Average the quantiles of the monthly-retrained N1, `a7_cs`, and the stack, applying
the 14-day correction to each member before blending ([experiment
00035](../../00035-lead-weighted-quantile-blend-of-stored-models/result.md)). One fixed weight
vector scored 5.020 against 5.025 for weights per lead band. Leave the stored quantiles otherwise as
they are ([experiment
00097](../../00097-cal1-post-hoc-recalibration-of-bg2-quantiles-tails/result.md)), and keep training
on the control member with the pinball loss ([experiment
00099](../../00099-crps1-crps-type-loss-over-ens-members-primary-demand-pilot/result.md)).

**For solar and wind farms, the best design is the pooled ERA5-trained mixture with small
additions.** Pool the ERA5-trained quantiles over the ENS members ([experiment
00002](../../00002-stage2-calibrator-and-mixture-vs-ens-quantile/result.md)). Add a spread factor
for solar if a 90% band is wanted ([experiment
00016](../../00016-pooled-mixture-calibration-solar-wind/result.md)), and a power-curve base margin
for wind ([experiment 00020](../../00020-solar-wind-stack-cams-stage0-spread-margin/result.md)). No
experiment has beaten the pooled mixture's CRPS for solar.

## What to implement in production

Four experiments have `production_status: candidate`. The first three change how the solar and wind
mixture is made, and the fourth adds models to the Primary demand service. No experiment has
`adopted` status, and a release plan decides whether any of them is ported.

- **[00002](../../00002-stage2-calibrator-and-mixture-vs-ens-quantile/result.md), pooled mixture:**
  pool the ERA5-trained quantiles over the ENS members for solar and wind. The stage 2 calibrator
  itself did not help.
- **[00016](../../00016-pooled-mixture-calibration-solar-wind/result.md), solar spread factor:**
  scale the solar mixture's quantiles about the median by one fitted factor. The 5% to 95% band
  covered 94.3% to 95.3% before the scaling and 89.5% to 91.8% after it, and the CRPS did not change
  detectably. The wind two-group factor was chosen after seeing the results and has not been tested
  on another year.
- **[00020](../../00020-solar-wind-stack-cams-stage0-spread-margin/result.md), wind base margin:**
  add a power-curve base margin to the wind stage 1a quantile model, which lowers the CRPS by 0.055
  points (99.58% interval -0.110 to -0.007). Nothing detectably helped solar.
- **[00035](../../00035-lead-weighted-quantile-blend-of-stored-models/result.md), Primary quantile
  blend:** serve a fixed-weight average of the quantiles of the monthly-retrained N1, `a7_cs`, and
  the stack. The blend lowers the quantile score by 0.10 points (98.33% interval -0.19 to -0.03)
  against the best member. It needs three models, one of them refitted monthly.

## Gotchas and implementation details

Each item names the experiment that taught it, what goes wrong without it, and what to check before
assuming it still applies.

**Report the pooled mixture beside any calibrator.** Without the mixture as the reference, a
calibrator that is worse than pooling can look like a gain over a single-member model ([experiment
00002](../../00002-stage2-calibrator-and-mixture-vs-ens-quantile/result.md)). Check again if the
calibrator gets new inputs, such as lagged power, which that experiment did not use.

**Fit every correction on valid times before the first run it is applied to.** A fit that sees later
outcomes fakes a calibration gain. The first month, and any lead band with fewer than 5,000 earlier
non-trivial cases, stays uncorrected: the wind 0 to 24 h band had a fit in only 3 months and the 1
to 2 day band in 8 ([experiment
00016](../../00016-pooled-mixture-calibration-solar-wind/result.md)). [experiment
00097](../../00097-cal1-post-hoc-recalibration-of-bg2-quantiles-tails/result.md) asserted before
scoring that every calibration row preceded its scored month. Check again if the series count or the
validation period grows, because the 5,000-case minimum is reached earlier then.

**Compute solar coverage on non-trivial cases only.** Night-time solar zeros sit inside every band
and inflate the share, so [experiment
00016](../../00016-pooled-mixture-calibration-solar-wind/result.md) counted only cases whose
uncorrected 95% quantile exceeded 1% of capacity. Check again for any technology other than solar,
or any forecast that is not zero at night.

**Refit each arm's own spread factor.** A factor fitted for one set of quantiles does not fit a
different set ([experiment
00020](../../00020-solar-wind-stack-cams-stage0-spread-margin/result.md)). Check again if the
quantiles come from a model that this experiment did not stack.

**Report coverage and pinball loss at q01, q05, q95, and q99 with the number of exceedances, keep
the stored quantiles at all 11 levels, and judge tails over more than 9 months.** Nine scored months
gave intervals of plus or minus 0.02 on q95 coverage, wider than the 0.01 bar, and a cell with fewer
than 30 days holding an exceedance was untestable. Recalibration can only be tested from stored
forecasts ([experiment
00097](../../00097-cal1-post-hoc-recalibration-of-bg2-quantiles-tails/result.md)). Check again once
2 years of scored runs exist.

**A power-curve base margin makes XGBoost ignore its base score.** A zero-start arm on a wind model
with the margin changes nothing, so test the starting value only on wind models without the margin
([experiment 00080](../../00080-bm0-zero-base-margin-xgboost-quantile-models/result.md)). Check
again if the XGBoost version or the objective changes.

**Measure the noise floor of GPU XGBoost with a repeat fit.** Two fits of the same 500-round model
scored 0.004 points apart at Primary substations and 0.007 at grid supply points, and a contrast
smaller than that is noise ([experiment
00080](../../00080-bm0-zero-base-margin-xgboost-quantile-models/result.md)). Check again on a
different GPU or XGBoost version.

**Hold the retraining schedule fixed within a contrast.** The CRPS-loss arms refit monthly with 150
rounds and sit 0.19 to 0.23 points above the weekly 500-round best guess, which probably reflects
the schedule, so their absolute scores cannot be set beside the best guess ([experiment
00099](../../00099-crps1-crps-type-loss-over-ens-members-primary-demand-pilot/result.md)). Check
again if the schedule of either side changes.

**Pool pinball-trained quantiles with equal weights, not the CRPS atom masses.** The CRPS masses
scored 0.025 points worse on the pinball arm ([experiment
00099](../../00099-crps1-crps-type-loss-over-ens-members-primary-demand-pilot/result.md)). Check
again if the pinball arm is replaced by a CRPS-trained one.

**Apply the 14-day correction to each member before blending, not to the blend.** Correcting after
the blend scored 0.048 points worse (95% interval +0.008 to +0.101) ([experiment
00035](../../00035-lead-weighted-quantile-blend-of-stored-models/result.md)). Check again if the
members are no longer corrected individually.

## What we have not settled

**Open questions:** whether a calibrator that sees recent power helps solar and wind; whether the
wind two-group spread factor holds on another year; whether a CRPS-type loss on all 51 members helps
Primary tails; whether the blend gains for solar and wind exist on more than 6 solar farms and 3
wind farms; and whether the month-to-month swing in Primary tail coverage can be tracked with at
least 2 years of scored runs. The [Research plan](plan.md) lists what is planned next, and the
[Experiments](experiments.md) page lists every experiment tried.
