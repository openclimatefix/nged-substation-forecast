# Plan: do lagged power features improve a solar power forecast? (issue #1138)

## Problem

Issue #1138 theorises that an XGBoost model forecasting solar power should gain from lagged power
features. The lags could calibrate a forecast to soiling, panel degradation, a failed string, part
of a plant offline, aerosol episodes, and changes in the weather forecast product. In a model
trained across all plants, the lags could also tell the model each plant's quirks (such as its
DC:AC ratio) without a physical model of the plant. The maintainer also asked whether lags help the
forecast quantify its uncertainty, and asked for a wide sweep of ideas, wacky ones included, before
any careful comparison. Nothing in the repository tests any of this. The persistence baselines read
lagged power, but no study gives an XGBoost model lagged power next to weather.

## Planned solution

A two-phase study on the 6 NGED utility-scale solar plants, driven by the mean of ECMWF's 51-member
ensemble forecast (ENS), reading the matched-lead study's inputs that are already on disk. **Both
phases are two views of one table of out-of-fold losses**, produced by the existing `run_all` and
`out_of_fold_losses`, so nothing is fitted twice.

- **Phase 1, a broad shallow sweep:** 18 fitted arms (14 lag ideas, the baseline, two controls, and
  an interpolation bound), plus post-model corrections that need no fitting. Each arm is scored as
  its own absolute error on the first 10 calendar months, with no pairwise comparison and no
  significance claim. The page shows one ranked chart. The sweep generates hypotheses.
- **Phase 2, a rigorous comparison:** five planned paired contrasts, fixed before any fit, one of
  which names "the best sweep arm", chosen by a rule written before the sweep. Phase 2 adds the
  sensitivity hyperparameter setting, the quantile models, a positive control, a global model with a
  fingerprint mini-sweep, and longer leads.

A point forecast and a probabilistic forecast (nine quantiles) are both scored. The page is
figure-led. The whole study ships as one PR.

## Verdict, size and departures

**Verdict: worth doing, with the changes below to the issue's wording.**

**Size: complex.** The five triggers:

- **What gets stored:** a published `docs/studies/` page and files under `data/studies/`; no Patito
  model, Delta table, or asset. Fires under the `study` skill's rule that a published page counts as
  what gets stored.
- **Production serving path:** not touched. Nothing under `src/` or any package except
  `packages/studies/` changes.
- **Degradation rule:** none touched.
- **More than one defensible design:** yes. The lag definition, the two-step leak control, the
  control arms, the global model's folds, and the sweep's selection rule each admit several designs.
- **Callers not nameable without searching:** none; the plan adds one function to
  `packages/studies/` and edits no existing one.

**Reviews:** simplicity review 1, the correctness review, simplicity review 2, the ideas review, and
the whole-plan consistency review (all done, triaged below); two Opus scientific-validity reviews,
the diff review, the mutation pass (the package changes), and the prose reviews and personas the
`study` skill sets. The study skill's merge conditions apply; the diff stays inside `studies/`,
`packages/studies/`, and `docs/studies/`.

**Departures from the issue body:**

1. **"24-hour lag" becomes "the same clock hour on the latest fully observed day".** A forecast
   issued for a target hour after the run starts cannot read power 24 h before the target at every
   hour. The lag of a lead-day-`N` target is `24 (N + 1)` hours (48 h at lead-day 1, 72 h at
   lead-day 2), the lag `baselines.diurnal_persistence` reads; the correctness review confirmed the
   two agree on every row where both exist. The page defines "lag" this way once.
2. **The ENS mean, not IFS HRES, is the weather product.** The issue suggests IFS from Open-Meteo.
   The ENS mean is the live service's input and the more accurate product (8.79 against 9.76 points
   at day 1 on the matched-lead rows). IFS HRES is an exploratory replicate of two arms.
3. **The sweep covers ideas the issue does not list**, at the maintainer's request.
4. **No new PV dataset** (next section).

## Do we need the UK PV dataset?

**Recommendation: no, for this study.** The target use is utility-scale plants; the
`openclimatefix/uk_pv` data is domestic rooftop PV, whose failures and soiling average out across
many small systems. It would also need a second weather download, and whether its time coverage
overlaps the ENS archive is unchecked. The shared rows hold 35,263 rows over 21 months (December 2024 to
September 2026, with January 2026 dropped), but the power record that lags are read from ends at
2026-07-01 (`studies.power.scan_power` stops at the final-test date, which the study does not
open), so **18 scored months** (2024-12 to 2026-06, without 2026-01) remain, and 4,246 to 6,976 rows per plant. The 6
plants sit in one 25 km box, so the effective sample size is weather episodes (months), not plants.
An effect is detected only if it is large; a null is read as "an effect as large as X is not
excluded", and a positive control sized to realistic soiling losses measures the detection limit.
The six similar plants are weak evidence for the global-model "plant quirks" hypothesis, and the
page scopes that result narrowly. The Scope section says domestic PV and other regions are
untested.

## Design

### Rows and inputs (reused from the matched-lead study)

- **Weather and power inputs:** the matched-lead inputs on disk under
  `data/studies/per_study/nwp_forecast_comparison/`: `original/solar_forecast_inputs.parquet`
  (the ENS mean's `ens_mean_day<N>_{ghi,temp}` for `N` in 0 to 3, `ghi_cams`, and the persistence
  and clear-sky columns), the `leads_day10*` and `day4_shared` files (the ENS mean at the longer
  lead days the matched-lead study published; day 4 exists and is not fitted), and
  `leads_day10d/solar_extra_lead_inputs.parquet` (`ifs_single_day<N>_{ghi,temp}`, fitted at days 1
  and 2 only, for the IFS HRES replicate). They are joined on `(site, time)`. The baseline arm has
  seven columns: hour of day, day of year, an era code, the sun's elevation and azimuth, and the ENS
  mean's global horizontal irradiance and 2 m air temperature at its own lead.
- **Shared rows, folds and `era_code`:** rebuilt in `build_lag_frame.py` with `cut_eras` and the
  matched-lead study's era constants (eras start 2025-10 and 2026-02; fold offsets
  `{0: 0, 1: 0, 2: 3}`), because a study script may not import another study folder. The build
  asserts 35,263 rows (the matched-lead published count). Per-plant folds are never recomputed after
  a row drop.
- **Row set:** the 35,263 shared rows minus the rows where B0's columns or L1's strict lag are null.
  Every other arm's columns may be null, and XGBoost handles them natively: CTX7's raw lags, IM
  (null when no hour has ended by 09:00 UTC, as in winter), AN, the stage-1 lag columns, and the
  window statistics under their minimum count (5 of 7 days for the weekly ones). The IFS HRES
  replicate is scored on that row set minus its own gap days. Each drop's count is printed. A strict
  7-of-7 rule would drop about 24% of rows, so no arm uses one.
- **Origin convention:** the existing one (`baselines.issue_time`): runs are issued at 09:00 UTC on
  the run's own day for every lead-day from 1. Lead-day 1, the day-ahead lead the live service uses,
  is the headline. Lead-day 0 is run in the longer-lead sweep only, issued at the run's init time
  (00 UTC) with a 24 h lag, and every figure showing it is labelled as not usable for its early
  hours in a live service, because the 00 UTC run is published later.
- **Targets and scaling:** per-plant fits use `power_mw` as the target, exactly as the published arm
  does, so `out_of_fold_losses` and `clamp_to_cap` work unchanged. The global fit divides
  `power_mw`, `cap_mw`, and every input in megawatts (lags, window statistics, IM's energy, TF's
  ratio, CK's ceiling, AN's power) by `effective_capacity_mw`, then sets `effective_capacity_mw` to
  1.0, so the existing loss columns stay in fractions of capacity. Dimensionless inputs (RP's ratio,
  DT's shares, IM's hour count) are not rescaled. Without the rescaling, the global model would
  learn each plant's capacity from lags in megawatts.
- **Lags are read from the full cleaned hourly power table** (not from the daylight-filtered
  inputs), strictly, with no tolerance, unlike `diurnal_persistence`, which falls back up to 7 days.
  NGED timestamps are already corrected at ingest and are not shifted again. The script asserts
  `lag_time <= issue_time` on every row and that the strict lag equals
  `diurnal_persistence_day<N>` wherever both exist.
- **Cleaning removes the case the issue names.** Outage hours, commissioning-ramp hours and
  export-cap hours are dropped from targets and lags, so "part of a plant offline" is outside what
  the study measures. The page says so.
- **No padding to equal column counts.** At `colsample_bytree=1` a duplicate column never wins a
  split, so padding with duplicates changes nothing and cannot remove the small bias towards a wider
  arm. The study instead measures that bias with the null controls N2 and N2-k, and says the arms
  have different widths.

### The sweep arms

All arms are fitted per plant, at lead-day 1, at the primary hyperparameter setting, with three
seeds and all five folds, through `run_all`, point model only (the exceptions are in Phase 2).
Phase 1 reads these losses on the first 10 calendar months (2024-12 to 2025-09) and nothing else.
Every window below is anchored at the issue day and reads nothing after the issue time.

| Family | Arm (extra columns) |
|---|---|
| Reference | B0 baseline (0); no-ML persistence and diurnal persistence (existing columns, not fitted) |
| Single lag | L1 latest whole day, same clock hour (1); IM the issue morning: observed energy in the hours ending at or before 09:00 UTC on the issue day, its ratio to the ENS mean's day-0 forecast irradiance (from the issue day's 00 UTC run) over the same hours, and the hour count (3) |
| Lag with weather | L2 L1 plus the forecast irradiance and temperature at the lag hour, from the target's lead day (3); CTX7 raw context: lags and lag-hour forecast irradiance for days 1 to 7 (14) |
| Weekly statistics | W7 minimum, maximum, mean of the same clock hour over the 7 whole days before the issue day, at least 5 present (3) |
| Slow trackers | Q30 the 30-day 90th percentile and median of the same clock hour (2); CK the clipping ceiling: expanding 99.5th percentile and 60-day maximum of hourly power, and the share of clear hours near the ceiling (3) |
| Transfer function | TF the 30-day ratio of power to forecast irradiance at the same clock hour (hours above 50 W m⁻², at least 15 days present), and that ratio times the target's forecast irradiance (2); AN the analogue ensemble: the mean, forecast clear-sky index, and spread of the observed power (rescaled by clear-sky irradiance) on the 5 days in the last 30 whose forecast clear-sky index was closest to the target's (3); PC the ratio of power to satellite irradiance (CAMS) over days 1 to 7 before the issue day, and the ratio of satellite to forecast irradiance over days 1 to 30 (since February 2026 CAMS's point service has a latency of 1 day: at 09:00 UTC it serves every interval through the end of the previous day, as [the data-sources roadmap](docs/roadmap/data-sources.md) records, so the whole of day 1 is available; a gain would justify ingesting CAMS into the live system; 2) |
| Two-step | S2 stage-1 prediction at the target hour and the lag hour, the observed lag, and their difference (4); S3 mean stage-1 residual over the last 1, 7, and 30 whole days before the issue day (3) |
| Cross-plant | RP the plant's 7-day capacity-normalised energy divided by the mean of the other plants', which isolates a plant-specific fault from shared weather (2, with a 1-day version) |
| Interpolation bound | T1 days since 2024-03-01 (1): month-block folds interleave, so trees can interpolate a test month's level from months on both sides, including future months; its gain is what interpolation buys, never drift a live forecast could use |
| Post-model (no fit; computed in the report from the stage-1 columns) | R1s the baseline's prediction times the 7 whole days' observed-over-predicted energy ratio before the issue day, shrunk halfway to 1; R2 the same with a per-clock-hour 30-day ratio; R4 predictions and upper quantiles clamped at CK's ceiling; R5 the baseline's point forecast plus the quantiles of stage-1 residuals over the last 30 days within the target's forecast clear-sky tercile, scored on CRPS against the baseline's quantile model |
| Combination | KS the kitchen sink: L1, IM, W7, Q30, TF, S3, RP (about 16) |
| Controls | N1 one observed hour from days 8 to 28 before the issue day at the same clock hour (1); N2 a lag at the same clock hour from a uniformly random observed day in months outside the target row's fold, the true null (1); N2-k, fitted only if X adds more than three columns, with as many random-day lags as X adds columns |

**Two-step arms use the existing `{fold}` column pattern.** Stage 1 is the baseline XGBoost model,
fitted at the primary setting with one seed and reused for every stage-2 fit. It predicts every
daylight hour of the input file (from 2024-03), not only the shared rows, because the trailing
windows of S3, R1s, R2 and R5 read stage-1 predictions at hours before the first shared row. For
scored fold `k`, an hour in fold `j` is predicted by the model that withheld folds `k` and `j`, and
an hour in no fold by the model that withheld `k`. The columns are written as `stage1_target_fold{k}`
and `stage1_lag_fold{k}`, as `run_hybrid_experiment` does for its physics predictions, so no
stage-1 prediction comes from a model that saw its own hour. Stage 1 predicts the lag hour from the
lag hour's weather at the target's lead day, available at the issue time.

**The shortlist rule, fixed before any fit:** X is the arm with the lowest phase-1 mean absolute
error, over the 10 screening months, among IM, CTX7, W7, Q30, CK, TF, AN, PC, S3, RP, and KS. The
excluded arms are B0, L1, L2, S2, N1, and N2 (phase 2 carries those regardless), T1 (an
interpolation bound), the persistence references and the
post-model corrections R1s, R2, R4 and R5 (none is fitted, so none has a sensitivity setting or
quantile model), and the global-only arms. Phase 2 carries X.

### Phase 2: the rigorous comparison

- **Settings and seeds:** both hyperparameter settings and three seeds for B0, L1, L2, S2, X, N1,
  and N2; the sensitivity setting is added to the primary fits above.
- **Quantile models:** fitted for B0, L1, X, and N2 at the primary setting, and for B0 and L1 at the
  sensitivity setting, through `out_of_fold_losses(with_quantiles=True)`, which already returns
  `crps_capped_mw`. CRPS is divided by `effective_capacity_mw` and labelled "CRPS approximated from
  nine quantiles (0.1 to 0.9)". A B0-versus-L1 figure shows the share of rows inside the 10% to 90%
  interval against a nominal 80%, and the interval width in clear, mixed and cloudy hours (by the
  forecast clear-sky index). This wrapper is the only quantile code written for the study.
- **Global scope:** B0, L1 and X at lead-day 1, one XGBoost model across the 6 plants with no plant
  identifier, coordinates or capacity column. The plants are pooled into one site-sorted frame
  whose `fold` is one fleet-wide assignment (`cut_eras` with `site` set to one constant for the
  whole fleet, the same era months and offsets, checked with `raise_on_uncovered_months`), so every
  fold is a set of calendar months at every plant and training excludes all plants' rows in the
  scored months by construction. The fleet-wide folds differ from the per-plant folds, so the
  global-against-per-plant comparison is exploratory. The global scope skips arms built from the
  `{fold}` stage-1 columns (S2, S3, KS), because those columns are indexed by the per-plant folds; if
  the shortlist rule picks one of them as X, the global X is the best remaining eligible arm. The ENS mean's 0.25° grid cells partly
  identify the plant, which the page says. Both settings, three seeds; quantiles for B0 and L1.
- **Global fingerprint mini-sweep (purpose: a "fingerprint" of each system for a model trained on
  many systems):** G-B0 and G-L1 are the global scope's B0 and L1 fits, reused. G-ID is G-B0 plus an
  integer plant code, the in-sample upper bound for any static fingerprint. G-FP is G-B0 plus the
  fingerprint pack: TF, CK, and the diurnal centroid shift and shoulder share DT, computed over the
  30 days before the issue day. A per-plant model already knows its own plant, so a fingerprint can
  only help there where the property changes; the global scope is the only place it is tested.
  G-FP close to G-ID, which is close to per-plant B0, shows the features carry plant identity. A
  leave-one-plant-out run (exploratory; the plant's scored months withheld at all plants) of G-B0,
  G-L1, and G-FP tests whether a fingerprint helps a plant the model has never seen. Six plants
  cannot show that a fingerprint scales to hundreds of systems, because trees cannot interpolate
  between five points, and the page says so.
- **Longer leads, to day 14 (exploratory):** NGED want forecasts to at least day 10 and perhaps day
  14. B0, L1, W7, Q30, T1, and N2 are fitted at lead-days 0, 1, 2, 3, 5, 7, 10, and 14 (day 1 is
  also the full sweep), point model, primary setting, three seeds. The lag of a lead-day-`N` target
  is still `24 (N + 1)` hours, and every window or percentile is anchored at the issue day, not the
  target day, so at day 14 the one-lag arm reads power 15 days old and the weekly window covers days
  15 to 21 before the target. Each lead-day has its own row set (a row needs that lead's columns
  and B0's and L1's lags at that lead), and arms within a lead-day share rows. The figure of each
  arm's gain over B0 against lead also draws the climatology baseline (`baselines.climatology`),
  because at day 14 climatology beats the ENS mean and slow trackers are the lag ideas that could
  still help.
- **IFS HRES replicate:** B0 and L1 with the IFS HRES columns at lead-days 1 and 2, point model,
  primary setting (exploratory).
- **No purge.** A training row whose input is a test label does not put that label in the model's
  output path, and B0 already trains next to test-month boundaries. The page says so in one
  sentence.
- **Device and slot:** every fit runs on the GPU with 4 concurrent workers (see "Compute
  estimate"), one device within each planned contrast. Fits need a slot from the Study MAIN
  COORDINATOR and checkpoint after every few arm-settings, because a reboot is planned.
- **Reproduction check:** B0 with the ENS mean at lead-day 1, refitted on the GPU (the study's
  device) on the 35,263 shared rows, asserting its mean absolute error is within 0.03 points of the
  published 8.771%. The tolerance is the published solar device-to-device range
  (−0.024 to +0.014 points), so the check confirms the pipeline is wired correctly without
  requiring bit-for-bit equality, and there is no CPU leg and no checksum.

### Controls

- **Positive control:** the same pipeline on a synthetic target in which power in a seeded random
  half of the scored months (exactly 9 of the 18, with the other calendar months drawn
  independently) is multiplied by (1 − s), applied to the hourly power table before the
  lags are built, for s of 2%, 5% and 10%. Choosing months at random keeps every B0 column (the era
  code, the day of year) uninformative about which months are shifted, which a single step date
  would not. B0 and L1, per-plant, lead-day 1, primary setting, three
  seeds (`out_of_fold_losses` always fits all three). "Recovers" means L1 minus B0 at the primary setting has a 99% interval wholly below zero and
  a point estimate at least 2% of B0's mean absolute error; the smallest s that L1 recovers is the
  detection limit at the primary setting.
- **Null and level-only controls:** N2 keeps neither the plant's level nor yesterday's weather, so
  its difference from B0 is the bias of one added uninformative column plus the noise floor. N2-k
  does the same for a wide arm. N1 keeps the plant's slow level but drops yesterday's own weather,
  so L1 minus N1 measures the value of day-specific information beyond the month's level.

### Deciding contrasts (planned, written before any result)

All at lead-day 1, at both hyperparameter settings, with 99% intervals (Bonferroni over five,
through `bootstrap_difference_at_level`), judged with the existing `bracket_verdict` and
`combine_setting_verdicts`.

- **P1:** L1 minus B0, mean absolute error, per-plant, all 18 months.
- **P2:** X minus L1, mean absolute error, per-plant, on the months after 2025-09 only (8 months,
  2025-10 to 2026-06 without 2026-01, that the sweep never screened at any plant), where X is the
  shortlist rule's arm. These months span the second and third eras of the weather forecast product
  (from 2025-10 and 2026-02) and hold no summer, so a summer-only effect cannot be tested by P2.
  A split leaving 10 months to screen and 8 to test was chosen because the minimum for an interval
  is 6 months.
- **P3:** S2 minus L2, mean absolute error, per-plant, all months (both arms carry lagged-weather
  information; the two-step design is the question).
- **P4:** CRPS of L1 minus CRPS of B0, per-plant, all months.
- **P5:** L1 minus B0, mean absolute error, global, all months.

**Decision rule, fixed before any result:** a contrast "helps" if, at both hyperparameter settings,
its 99% interval lies wholly below zero and its point estimate is at least 2% of the baseline's own
value of that metric (about 0.2 points for mean absolute error, the smallest effect worth acting
on, and about five times the 0.4% bias of a wider arm); a smaller gain is reported as detectable but
not worth the complexity. Every other number is exploratory and labelled so: the sweep, S2 minus
L1, N1, N2 and N2-k comparisons, the post-model corrections R1s to R5, the global fingerprint
mini-sweep and leave-one-plant-out, the positive control, lead-days 0 and 2 to 14, global against
per-plant, coverage and sharpness (descriptive), the IFS HRES replicate, and any analysis added
after the first run (post hoc).

### Charts (the story, as figures)

1. **Figure 1, the headline:** P1 to P5 with their 99% intervals, both settings as markers, under
   the bottom line.
2. **The data:** forecast irradiance against measured power for each anonymised plant over one
   stated week per rule (clearest, most variable, dullest), showing steps, gaps and outages; the
   lag hour against the target hour as a scatter.
3. **The techniques with known answers:** how each lag column is built for one example day; the
   synthetic level shifts and whether L1 recovers them; N1 and N2 beside L1; the two-step design's
   withheld folds drawn as months.
4. **The XGBoost models work:** out-of-fold predictions against measured output for each plant over
   the same weeks; each arm's absolute error per plant; B0 beside the published figure.
5. **The sweep:** one ranked chart of every fitted arm's absolute error with the baseline as a
   line, labelled as a screen, and a separate panel for R5 (scored on CRPS).
6. **Results:** each longer-lead arm's (B0, L1, W7, Q30, T1, N2) gain over B0 against lead-day with
   climatology; monthly paired differences; per-plant against global; reliability, coverage and
   width; feature importance (descriptive) by feature group.
7. **The global fingerprint:** G-B0, G-L1, G-ID, G-FP, and their leave-one-plant-out versions.

Plants are anonymised A to F (`studies.anonymise.site_labels_for`); nothing prints an identifier,
name, or coordinate, and power is shown only as a share of capacity.

## Compute estimate

Benchmarked on the idle workstation (with a download using about 5 cores), small fits of 5,000 to
28,000 rows with 12 to 22 columns, 500 rounds at the primary setting and 1,200 at the sensitivity
setting. A single fit on the GPU took 1.0 to 1.3 times less time than on 4 CPU threads at 5,000
rows (a nine-quantile fit: 4.4 s against 5.8 s) and 2 to 2.6 times less at 28,000 rows (5.3 s
against 13.6 s). For throughput, 8 fits (a point and a quantile model each, 5,000 rows) took 34 s on
the CPU with 2 workers of 4 threads (the 8-core budget) and 18.5 s on the GPU with 4 workers; 8 GPU
workers gave no gain. GPU memory peaked at 0.6 GB with 4 workers. An earlier benchmark, run while
another job held the GPU at 99%, found the GPU slower; the idle result supersedes it.

| Stage | Fits | Single-fit time on 4 CPU threads |
|---|---|---|
| Primary sweep, point model: 18 arms x 6 plants x 5 folds x 3 seeds | 1,620 | about 30 min |
| Sensitivity setting, point model: 7 arms | 630 | about 12 min |
| Quantile models, primary: B0, L1, X, N2 | 360 | about 40 min |
| Quantile models, sensitivity: B0, L1 | 180 | about 30 min |
| Global scope: B0, L1, X, both settings; quantiles for B0 and L1 | 150 | about 25 min |
| Global fingerprint mini-sweep: G-ID, G-FP, and leave-one-plant-out of G-B0, G-L1, G-FP | about 300 | about 12 min |
| Longer leads: 6 arms x 7 extra lead-days (0, 2, 3, 5, 7, 10, 14) x 6 plants x 5 folds x 3 seeds | 3,780 | about 45 min |
| Stage-1 models, positive control, IFS HRES replicate | about 800 | about 12 min |

Feature importance (descriptive) comes from a separate point-model refit of B0, L1, L2, S2 and X at
the primary setting, seed 0, one per plant and fold (about 150 fits), because `out_of_fold_losses`
does not return the boosters.

**Total: about 3.5 h of single-fit time on 4 CPU threads. At the benchmarked GPU throughput (about
2.8 times a single CPU fit) that is about 1.3 h of wall time, so the slot request is 2 h, because the 6 per-plant jobs of an arm run in two waves on 4 workers** (GPU,
4 workers using 4 cores, 1 GB of GPU memory). On the 8-core CPU budget the same work would take
about 1.8 h.

## Follow-ups after the first science review (post hoc)

The first Opus science review of the results asked for re-runs and analyses. All are post hoc and
labelled so on the page; the published planned contrasts P1 to P5 are unchanged and not re-run.

- **Positive controls that can be passed:** an oracle arm (B0 plus the true shift factor) at 5% and
  10%, to separate a pipeline defect from a weak feature; the same month-level control on W7, Q30,
  TF, AN and PC (and R1s, which needs no fit); and a second, more physical control with
  plant-specific persistent steps (random start, 4 to 12 weeks, at 5% and 10%) for the oracle, L1,
  W7, Q30, TF and AN. "Recovers" is judged by whether the 99% interval lies wholly below zero and
  by the share of the oracle's gain recovered, because the 2% rule cannot be met by an ideal
  estimator below 10%.
- **Long leads:** a no-fit 50/50 blend of B0 and climatology; a CL arm (B0 plus the out-of-fold
  climatology as a column) and W7+CL at days 7, 10 and 14; N2-3, a width-matched null for W7; and the
  sensitivity setting for B0, N2, W7 and Q30 at days 10 and 14.
- **PC with a 2-day latency** (days 2 to 8 and 2 to 31) at lead-day 1, because 3 of the 8 later
  months fall before CAMS's February 2026 latency cut.
- **Fingerprint decomposition:** G-ID+TF, G-FP without CK, leave-one-plant-out of G-FP without CK,
  and per-plant B0 on the fleet-wide folds so global against per-plant is paired.
- **No-fit analyses:** every arm's gain over B0 on the 8 months the sweep never screened, split by
  forecast-product era; per-quantile hit rates; the count of exploratory intervals; the hindsight
  per-plant-month scaling bound.

## Walk through

**The happy path.** `build_lag_frame.py` rebuilds the shared rows and folds (`cut_eras`), joins the
ENS-mean and IFS HRES day columns, reads the full cleaned hourly power (`studies.power`), builds the
window-statistic arms with `same_clock_hour_window` and the other arms in the script, asserts the
lag arithmetic, applies the row set, and writes the frame under `data/studies/lag_features/`.
`fit_lag_arms.py` fits the stage-1 models with withheld folds (writing the `{fold}` columns), runs
every arm through `run_all`, prints the shortlist rule's X, then adds the sensitivity, quantile,
global, longer-lead and control fits, checkpointing, and saves per-row losses and out-of-fold
predictions. `report_lag_features.py` filters the losses to the screening months for phase 1 and to
the full or held-out months for phase 2, computes intervals (`bootstrap_difference_at_level`,
`bootstrap_absolute`), and prints every table the page quotes with each arm's column list.
`lag_features_charts.py` draws the figures from the saved files and the report. The page
`docs/studies/lag-features.md` quotes the report.

**Branches.** Per-plant arms go through `run_all`; the global arms pass the pooled frame to
`out_of_fold_losses` as one `site_rows`. The positive control modifies the hourly power table before
the lags are built.

**Error handling.** A study fails fast: an uncovered calendar month raises
(`raise_on_uncovered_months`); a failed lag-arithmetic assertion raises; a shared-row check that
does not hold raises. Dropped rows are counted and printed.

## What changes, file by file

- `studies/lag_features/build_lag_frame.py` — the shared rows and folds, the strict lags, every
  sweep arm's columns, N1, N2, the positive control, the row set, the assertions, and one fixed
  `{arm: column tuple}` mapping printed into the report.
- `studies/lag_features/fit_lag_arms.py` — stage 1, the arm runs, the shortlist rule, the
  sensitivity, quantile, global, longer-lead (0, 2, 3, 5, 7, 10, 14) and control runs.
- `studies/lag_features/report_lag_features.py`, `lag_features_charts.py`, `README.md`.
- `packages/studies/src/studies/baselines.py` — one added function,
  `same_clock_hour_window(keys, hourly, day, first_days_back, last_days_back, statistic,
  min_count)`, which L1, CTX7's lags, W7, Q30, N1, and every longer-lead window use (the other arms
  are built in `build_lag_frame.py`), with one parametrised test in
  `packages/studies/tests/test_baselines.py`.
- `docs/studies/lag-features.md`, an entry in `docs/studies/index.md` and `mkdocs.yml`, and
  `docs/studies/assets/lag_features_*.svg` (optimised with `svgo`).

## Tests

- **`same_clock_hour_window`:** a parametrised fixture at lead-days 0, 1 and 2 gives each day,
  including the issue day's clock hour, a distinct value, plus other clock hours and other sites;
  it asserts only the intended days of the same clock hour and site are used, the issue day is
  excluded, and a window below `min_count` returns null. Lead-day 0 is included because it moves the
  issue time to 00 UTC. Each case fails on the bug it names (including the issue day, the wrong
  clock hour, a mixed site, an off-by-one window edge).
- **Lag arithmetic, the issue-morning cut-off, the fleet-wide fold coverage, and B0's reproduction
  check:** assertions inside the scripts, since study scripts have no unit tests; each raises on
  violation and the report prints the result.

## Design-philosophy check

R&D code: fail fast, as the inherent-stability page requires of training and metrics code. Nothing
runs in production. The study does not choose the production model; a follow-up issue would carry
any result into `ml_core` feature code. The page cites no hypothesis label unless one on
persistence or lag features exists in `engineering-hypotheses.md` (checked at write-up).

## Docs to update

The new study page, the studies index and nav, the `studies/README.md` row, and the family README.
No roadmap item completes, so there is no ship-time triage.

## Verification

`uv run ruff check .`, `uv run ruff format --check .`, `uv run ty check`, `uv run pytest
--run-studies -n auto packages/studies`, `uv run pymarkdown scan -r docs README.md CLAUDE.md
packages/*/README.md`, `uv run mkdocs build --strict` and reading the rendered page, pydoclint and
the docs-link checker.

## Risks and open questions

1. **Sample:** 18 scored months at 6 plants in one 25 km box. P2 rests on 8 months in two eras of
   the weather forecast product with no summer. Intervals from a percentile bootstrap over 18 month clusters
   tend to under-cover, plants are pooled and weighted by row count, and the test does not cover
   differences between plants. The page says all of this. No new PV data.
2. **Power-data latency:** the study assumes observed power up to the forecast's issue time (up to
   09:00 UTC on the issue day for IM). The live system currently receives NGED power with a lag of
   at least 6 hours, which the maintainer has asked the study to ignore, because PV power can be
   obtained with a delay of minutes. The page states that IM and the shorter-lag arms need that
   faster feed, while lags of a day or more are unaffected.
3. **Unequal column widths:** padding is inert, so phase-2 contrasts compare arms of different
   widths. N2 measures the bias of one extra column, and N2-k that of a wide X. The maintainer has
   accepted the departure from the `study` skill's equal-count rule; the page says why.
4. **CAMS ingestion:** PC is eligible as X. A gain from PC would justify ingesting CAMS into the live
   system. The arm depends on CAMS's 1-day latency (documented in the data-sources roadmap, and
   shorter than the 2 days older third-party documentation gives); in the study's history before
   February 2026 CAMS's latency was longer, so the backtest reads CAMS slightly fresher than the
   service then offered, and the page says so.

## Triage of the reviews

**Simplicity review 1.** Accepted: no cycle-change eras (shared folds); reuse of the on-disk
inputs; the existing origin convention; a smaller phase-2 grid; the positive control sized to a
realistic shift; S2 stage 1 at one setting and seed. The suggested cut of leave-one-plant-out was
later reversed (it became exploratory, see the ideas review). Modified: N1 stays as a level-only
control, with N2 added as the true null; the lag is read strictly. The maintainer's two-phase
request reverses the suggestion to run only five arms.

**Correctness review.** Accepted: the global fit leaking through neighbouring plants (now solved by
a fleet-wide fold); `power_mw` as the per-plant target and capacity-normalised global inputs; N1
drawn from days 8 to 28 before the issue day; no purge; normalised CRPS; the data facts (18 usable months,
7 missing runs, the empty 2024-11-12 split removed); stage-1 predictions that withhold the hour's
own fold, and lagged weather at the target's lead day; the minimum-count rule and lags from the full
cleaned table; P3 as S2 minus L2; 99% intervals; a threshold for the CRPS contrast; a multiplicative
positive control applied before the lags; script assertions; the "partly identify the plant"
wording; the cleaning caveat; the interval caveats.

**Ideas review (the maintainer asked for more calibration and fingerprint ideas).** Accepted: AN,
TF, IM, RP (replacing the fleet mean), CK, PC (study-only), CTX7 (replacing L7), the post-model
corrections R1s, R2, R4 and R5, and the global fingerprint mini-sweep with G-ID as the in-sample
upper bound and a leave-one-plant-out run (exploratory, the only test of in-context learning); DS
dropped; the slope dropped from Q30; T1 relabelled as an interpolation bound. Left for a follow-up:
rain-wash features (an ENS precipitation extract; a null is likely uninformative), snow, a
temperature-coefficient fingerprint, aerosol optical depth as a feature, and sequence models.

**Simplicity review 2.** Accepted: no padding (inert); phases as two views of one loss table; the
`{fold}` pattern for two-step arms; quantile models for B0, L1, X, and N2 only, with Y dropped; P2
split by calendar month rather than fold; the global model through `out_of_fold_losses` with a
fleet-wide fold; OF, NB, L2d and CSI cut and TR merged into Q30; one generic window function;
reproduction steps 2 and 3 dropped; one PR. Lead-day 0 was dropped and later restored at the
maintainer's request, in the longer-lead sweep only.

**Whole-plan consistency review.** Accepted: the shortlist rule's exclusions (M1); the row set
(M2); rebuilding the shared rows in the build script instead of importing another study folder
(M3); `cut_eras` for the global folds (M4); the positive control's pass criterion and a known date
off the era boundaries (M5); N2 drawn outside the target row's fold, and N2-k for a wide X (S1,
S2); the reproduction check retargeted to the ENS-mean B0 (later simplified to a GPU-only check within 0.03 points of the published error)
(S3); the device text (S4); the compute rows and the double-counted global arms (S5); stage-1
predictions at every daylight hour (S6); stale text (S7); the exploratory list (S8); the window
function's scope (S9); the global scaling rule (S10); the word "fingerprint" kept for plant
properties only (S11); lead-day availability (S12); the charts (S13); the ENS departure (S14); the
one-era note on P2 (S15); the window anchors (S16). Open questions answered: PC is eligible as X (the
maintainer would consider ingesting CAMS); IM's power-latency assumption is stated as a risk; N2-k is fitted for a wide X; the
reproduction target is the ENS-mean B0; P2's single era is stated as a limitation.
