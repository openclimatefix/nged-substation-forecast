---
id: "00014"
slug: cams-trained-stage1-with-stage0-ens-correction-demand
status: planned
size: complex
hypothesis: >-
  A stage 0 that maps the ECMWF ensemble (ENS) member's shortwave irradiance onto the irradiance of
  the Copernicus Atmosphere Monitoring Service (CAMS) lowers the Primary substation quantile score
  of the CAMS-trained stage 1a XGBoost demand model, relative to feeding that model the raw ENS
  shortwave, because the ENS shortwave is about 7% above CAMS and stage 1a never saw that offset in
  training.
prediction: >-
  On the 73 ENS runs of `fold_a_every5d_v1`, stage 0 lowers the Primary quantile score over all
  leads by 0.03 to 0.10 points of capacity (C2), mostly in daylight and in the first 6 days. After
  stage 0, the validation-year mean daylight shortwave of the ENS members is within 2% of CAMS
  (before: about 7% above), and the spread of the 51 members is within 10% of its value before.
decision_rule: >-
  C2 is the quantile score of the CAMS-trained arm fed stage-0-corrected ENS minus that of the same
  arm fed raw ENS, pooled as the mean over every scored row of the 16 Primary series and over the
  three seeds. C1 is the CAMS-trained arm fed raw ENS minus the ERA5-trained arm, reproduced from
  experiment 00009's fitted models and not refitted. Intervals resample whole validation months and
  the three seeds, at 97.5% (Bonferroni across C1 and C2), with 10,000 resamples. The hypothesis is
  supported if C2 is at most -0.03 with its interval wholly below zero and the daylight bias
  check passes. It is refuted if the interval of C2 lies wholly above -0.03 and the bias check
  passes. Otherwise, including when the bias check fails, it is inconclusive. The bias check uses
  the 16 Primary series, daylight rows, and all leads: the validation-year mean of the corrected
  ENS shortwave is within 2% of CAMS, and each validation month is within 5%. Because 00009's C1
  had a 97.5% half-width of 0.035 points, a true C2 near zero is unlikely to meet the refutation
  condition, so an inconclusive verdict is a likely outcome; the write-up reports the half-width.
benchmark: >-
  The CAMS-trained arm fed raw ENS (experiment 00009's fitted models, same rows), the ERA5-trained
  arm of 00009, and 00005's month-by-half-hour climatology `r2`.
row_set_id: fold_a_every5d_v1
metric: >-
  Quantile score: twice the mass-weighted mean pinball loss over 9 quantile levels, an estimate of
  the continuous ranked probability score, in points of capacity (power over effective capacity,
  times 100); and the mean absolute error of the median
---

# Does a stage 0 that maps ENS irradiance onto CAMS help the CAMS-trained demand model?

**This experiment is P20 of the [research plan](../research-plan.md), the follow-up to [experiment
00009](../00009-cams-trained-stage1a-vs-era5-trained-demand/result.md).** Experiment 00009 found
that a stage 1a trained on CAMS irradiance lowered the Primary quantile score by only 0.035
points when fed ENS shortwave, and measured the ENS daylight shortwave at 281 W/m² against 263 for
CAMS. This experiment tests that offset as a cause.

## Size

- **What gets stored:** prediction sets and tables under this experiment's private folder. Nothing
  enters a Delta table.
- **Production serving path:** not touched.
- **Degradation rule:** not touched.
- **Designs that would test the hypothesis defensibly:** three. Empirical quantile mapping,
  planned here, keeps the spread of the 51 members. A regression of CAMS on ENS would pull every
  member towards the conditional mean and shrink the spread. A single multiplicative factor would
  fix only the mean offset.
- **Callers that need searching for:** none. The scripts live in this experiment's folder, and they
  reuse the code and the fitted models of 00009.

The only trigger that fires is the second design: the experiment adds a method that more than one
design could implement. It therefore gets one review of the plan, one of the code, and one of the
write-up, all by Sonnet. The earlier `advice` lists were read. This plan follows "Keep the stage 1a
training weather in ERA5 until a CAMS-trained arm has been tried with the ENS shortwave
bias-corrected" and "Score any irradiance-input change by lead band as well as in total" (00009),
and sets its threshold of -0.03 points near the 97.5% half-width of 00009's C1 (0.035 points).
A gain of that size is at the edge of what the rows can resolve.

## Design

**No stage 1a model is refitted.** The ERA5-trained and CAMS-trained boosters of 00009 (20 series, three
seeds) are loaded as they are. Only the input they read at validation changes, so C2 differs from
00009's CAMS-trained arm in one input: the ENS shortwave column.

**Stage 0 is an empirical quantile mapping from ENS shortwave to CAMS irradiance.** The mapping is
fitted on the training times before 2025-06-17 (14 days before the first validation run), from
the stage 1 pilot's ENS training table (one ensemble member per run, 456 runs from 2024-04-01;
the script prints which member and reports it, since the control member is not exchangeable with
the perturbed ones), filtered to `valid_time` before 2025-06-17 (not to init time, because the table
runs to 2025-06-30), with an assertion on the maximum valid time. The table holds 22 series, and
the mapping uses only the 20 modelled ones paired with
CAMS held half-hourly at the series' own point. The pairs are pooled over the 20 series and
binned by lead band (the five bands of 00009) and by solar zenith (6 bins of equal cosine width,
daylight only, zenith below 80°). Within a bin, 99 quantiles of the ENS values are matched to the
same quantiles of the CAMS values, and the mapping is piecewise linear between them and clamped
to the end points outside the fitted range. A bin with fewer than 1,000 pairs borrows the mapping of
its neighbouring zenith bin. At validation,
each member's value is mapped on its own, so the 51 members keep their spread. Night rows pass
through unchanged. CAMS is needed only at training times, never at forecast time.

**The stage 0 output is checked before stage 1a reads it.** The mismatch table lists the mean daylight shortwave
of CAMS, raw ENS, and corrected ENS by month of the validation year, with the ratio to CAMS, plus the
mean spread of the 51 members before and after.

**Two planned contrasts, kept separate, both on Primary substations over all leads.** C1 is the
CAMS-trained arm on raw ENS minus the ERA5-trained arm, recomputed here from 00009's models on the
same rows; a value differing from 00009's -0.035 by more than 0.005 is reported and read as the
noise floor of GPU prediction. C2 is the stage 0 contrast. The threshold of -0.03 is a little under
the half-width of 00009's interval, so a gain of that size is at the edge of what the rows resolve.

**Exploratory contrasts are read at 95%.** They are: C2 for the four bulk supply point and grid
supply point series together (the CAMS values exist for all 20 series, which 00009 used); C2 by lead
band and for daylight and night separately; X1, a quantile mapping of ENS onto ERA5 shortwave at
the same cells, fed to the ERA5-trained arm, minus that arm on raw ENS (does a stage 0 help
whichever product stage 1a trained on); and C2 for each Primary series.

**Leakage and calendar checks.** Stage 0 sees no valid time from 2025-06-17 on. Stage 1a models are
00009's, trained before the same date. No calendar column is added.

**Known limits.** CAMS is held hourly while the ENS shortwave is interpolated to half-hours, which
the marginal mapping ignores. The mapping is fitted on 14 months, is unconditional on the day's
weather, and corrects the offset in distribution, not the forecast error of an individual member.
The training table holds one member per run, assumed exchangeable with the other 50.

## Results

To be written after the run.
