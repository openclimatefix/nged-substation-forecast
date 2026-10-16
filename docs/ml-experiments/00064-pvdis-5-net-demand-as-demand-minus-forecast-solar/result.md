---
id: "00064"
slug: pvdis-5-net-demand-as-demand-minus-forecast-solar
status: planned
size: complex
hypothesis: >-
  At the 5 Primary substations (substations that feed 11 kV circuits from the 33 kV network) with a
  clean solar-capacity estimate, a demand forecast that carries an explicit solar term (the
  substation's embedded solar capacity times a calibrated forecast of the solar output) scores no
  worse than experiment 00033's best-guess recipe `d0` by more than 0.10 points of capacity on
  daylight April-to-September rows, and the solar term carries real capacity information, because the
  same arm with capacities shuffled between forecast dates scores worse. The solar term is built in
  four ways: added to the forecast after the fit with a coefficient fitted on the training window
  (`sf`), given to the XGBoost model as an input column (`a`), given to it as a starting offset so
  that the trees fit only the remainder (`am`), or removed from the training target and added back
  after the fit (`b`, "demand minus solar").
prediction: >-
  `sf` is the likeliest arm to pass. Experiment 00046's exploratory fitted-coefficient shift scored
  0.038 points below `d0` (97.5% interval -0.126 to +0.034), so `sf` is expected at -0.04 (-0.12 to
  +0.04) against `d0` over all leads, and at -0.10 to -0.25 at leads under 48 hours, where the ENS
  anomaly tracks the satellite-based anomaly. `a` is expected at +0.00 to +0.10 against `d0`,
  because experiment 00031's nearest arm lost 0.036 points. `am` is expected to beat `a` by 0.02 to
  0.08 on the sunniest 10% of rows, where `a`'s trees cannot extrapolate past the capacities seen in
  training, and to tie it elsewhere. `b` is expected at +0.05 to +0.25 against `d0`, because the
  gross target adds the noise of the satellite-based sunshine estimate to the label, and the ENS
  forecast of the subtracted term loses its skill beyond 2 days. The fitted coefficient is expected
  between -0.4 and -0.7 (slope of the demand residual on the true-sunshine shift) for the full
  `d1l` model, as in experiment 00051, and nearer -1 for a model without lagged power and without
  cloud-proxy inputs, if the lower slope is the model already absorbing part of the anomaly. The
  full-leads upper end of the non-inferiority read is therefore expected below +0.10 for `sf` and
  not for `b`. A shuffled-capacity twin is expected to gain nothing against `d0` for `sf` and `am`
  and to gain part of what the real arm gains for `a`, as experiment 00031's shuffled control did at
  leads under 48 hours.
decision_rule: >-
  Eight decisive reads, 4 arms (`sf`, `a`, `am`, `b`) times 2 contrasts, at 99.375% (Bonferroni
  across 8 reads), 50,000 resamples of whole ISO weeks of forecast start (about 29 blocks) and of
  seeds, on the quantile score in points of capacity over the 5 verdict series' daylight
  April-to-September rows. Contrast N is the arm minus `d0`. Contrast T is the arm minus its own
  shuffled-capacity twin. The non-inferiority margin is 0.10 points, fixed here from a power
  calculation on experiment 00046's stored predictions: the paired contrast of the fitted ensemble
  shift against `d0` has a 97.5% half-width of 0.071 points with the same 29 weekly blocks, which
  exceeds 0.05, so the design cannot resolve a margin of 0.05, and 0.10 equals about the whole
  0.104-point effect of `d0`'s radiation inputs (experiment 00046). An arm is non-inferior when the
  upper end of its N interval is below +0.10. An arm carries capacity information when the upper end
  of its T interval is below zero. `supported` when at least one arm is both. `refuted` when no arm
  is non-inferior and every arm's N interval lies wholly above zero. `inconclusive` otherwise,
  including an arm that is non-inferior but does not beat its twin, which the write-up then reports
  as "matches, but the capacity adds nothing". A gain is reported separately, and counts when the
  upper end of N is below -0.10 (the same size as the margin, because a skill gain smaller than the
  margin cannot be told from the explanation-only case), over all leads or in the 0-to-48-hour read
  that is exploratory at 95%. Nothing here is gated on the 0.35-point threshold of experiment 00046,
  because PVDIS-5 is a test of explanation and non-inferiority first. All other reads are
  exploratory and labelled so, at 95%: the A-against-A-margin contrast, per-member against
  ensemble-mean shift, the temperature derating, each arm's effective weight, the sunniest decile,
  the lead-band and valid-month reads, and the slope diagnostic. The cost of porting `sf` is a
  capacity table to read and a fitted coefficient table to keep current, with no new model fit in
  production; `a`, `am`, and `b` add the same table and change the training target or the inputs.
benchmark: >-
  `d0`, experiment 00033's best-guess recipe (N1 with the 14-day correction and the 1.15 spread
  factor), as refitted in experiment 00046 with 3 seeds (stored predictions, no refit). `d1l`, the
  same recipe without `shortwave` and `longwave`, is the base of `sf`, `a`, `am`, and `b`. The
  reference rows `climatology` and `manual_heuristic` are experiment 00033's stored predictions on
  the same rows. The P-roll row 5.150 of experiment 00049 is not used, because its rows (weekly
  refits, other series) differ from this experiment's rows.
row_set_id: fold_a_every5d_v1
metric: >-
  Quantile score (twice the mass-weighted mean pinball loss over nine levels, the estimate of the
  CRPS) in points of capacity (1 point is 1% of the series' effective capacity), after the 14-day
  correction and the spread factor, on daylight April-to-September rows with a CAMS unit PV shape
  and an observed power, as in experiment 00046.
code: ml-experiments/00064-pvdis-5-net-demand-as-demand-minus-forecast-solar/
verdict: null
best_arm: null
headline: null
lessons: null
production_status: null
adopted_in: null
advice: []
---

# Experiment 64: demand minus forecast solar (PVDIS-5), the plan

This page is the pre-registered plan. It is committed before any code runs, and the sections after
"Size" are replaced by the write-up when the experiment finishes.

## The question, in plain words

We ask whether a forecast of a Primary substation's power can be split into "demand" and "solar
output" without getting worse, and whether the split ever gets better. A Primary substation's meter
reads demand minus the output of solar panels behind it, so a sunny day lowers the metered power.
[Experiment 46](../00046-embedded-solar-share-of-primary-forecast-error/result.md) subtracted a
forecast of the panels' output from a demand forecast and gained almost nothing over all lead times,
although knowing the true sunshine would have gained about 1 point. Experiment 51 and the solar
design session found three reasons the subtraction was too crude: the coefficient of 1 was too
large, the spread of the ensemble members added noise, and the capacity was a single 2023/24 figure.
This experiment repairs those three and tries four ways to attach the solar term to the forecast.

## Size

- **What gets stored:** predictions of the new arms under
  `data/ml-experiments/00064-.../predictions/`, private and never committed, and nothing else; no
  contract, Delta table, or parquet in the pipeline changes.
- **Production serving path:** not touched.
- **A degradation rule:** not touched; a missing capacity or ENS solar value gives a zero shift in
  every arm.
- **More than one defensible design:** yes (four attachments of the solar term), which is why the
  four are separate arms.
- **Code whose callers could not be named without searching:** no; the scripts copy experiment
  00046's helpers.

One trigger fires, the one about more than one defensible design. The experiment also adds a new
method (gross-demand training with lagged gross inputs) and four arms with twins, so it gets the
full process: plan, one Opus plan review, Sonnet implementation, Sonnet code review, Sonnet run
and write-up, and Sonnet write-up and prose reviews.

## Earlier advice followed and departed from

Followed: fit the solar coefficient on the training window by lead band and season instead of fixing
it at 1 (00046, 00051); use the ensemble-mean shift and keep the per-member shift only as a
comparison (00051); test non-inferiority with a margin fixed after a power calculation (00046,
00051); pre-register a gain read for 0 to 48 hours only (00046); report coverage of the 0.05 and 0.95
quantiles in sunny and cloudy rows (00046); read the sign of each contrast by valid month (00046);
use no second weather forecast, because experiments 50 and 53 found none that improves ENS (00050,
00053); read the season-year capacity constant, not a weekly path (00045). Departed: none.

## Design

**Rows.** The 5 verdict series of experiment 00046 (D9, D10, D11, D14, D16; their capacity status in
experiment 45 is `clean` or `low-power`), the 3 seeds, and the scored rows of experiment 00046:
validation runs of fold A (every fifth day, valid 2025-07-01 to 2026-06-30), daylight half-hours of
April to September with a CAMS unit shape and an observed power. The training window is the same
as experiment 00046's (ENS control member, up to 2025-06-30, half-life 2 years). No row on or after
2026-07-01 is scored.

**Capacity.** The season-year constant of experiment 45 (`capacity_weekly.parquet`, rows with
`recommended` true), one value per Primary per season-year (September to August). A training row
reads the constant of its own season-year. A validation row reads the constant of 2024/25, the
latest season-year that was observed before the fold's training end, held flat. That departs from
a strict real-time vintage in one respect: the 2024/25 constant was fitted on data up to August
2025, two months past the training end; the write-up says so. Experiment 46's 2023/24 capacity is
not used.

**Solar shape.** The CAMS unit PV shape `u` (experiment 46) for the past, and the calibrated ENS
shape for the forecast: the CAMS climatology for the local month and half-hour plus the calibrated
ENS anomaly, where the calibration slope is fitted per series and lead band on the training window
(experiment 46's `calibration.parquet`). Ensemble-mean calibrated shape for the headline arms; the
per-member calibrated shape for the comparison arms.

**Effective weight `w`.** For each series, lead band (the 5 bands of experiment 46), and season
(April to June, July to September), `w` is the slope through the origin of the out-of-fold residual
of the median on the oracle shift (minus the capacity times the CAMS anomaly), fitted on the daylight
training rows of that cell. The out-of-fold residuals come from 4 time-block folds of the training
window that group by the valid time's calendar month (month index modulo 4), seed 0 only, so each
held-out month is predicted by a model that has not seen its valid times. The same table of `w` is
used by `sf` and `am`.

**Arms** (all use experiment 00046's `d1l` inputs and recipe unless stated):

- `sf`: the stored `d1l` quantiles shifted by minus `w` times the capacity times the
  ensemble-mean calibrated ENS anomaly over the effective capacity (predict-only on the stored
  boosters, no refit). `sf_member` shifts each member by its own calibrated anomaly, then pools.
- `a`: `d1l` plus one input column, the capacity times the calibrated ENS shape of the member (the
  control member in training, each member at inference), refitted.
- `am`: as `a` without the column, with the XGBoost `base_margin` set to minus `w` times the capacity
  times the ensemble-mean calibrated shape (control member in training), so the trees fit the
  remainder. The `base_margin` for the nine quantile outputs is tested on a smoke run first.
- `b`: trained on gross demand, the metered power plus the capacity times the CAMS unit shape, with
  the `d1l` inputs. Each lagged input is the net lag plus the capacity times the shape at the lagged
  time, taken from CAMS when the lagged time is at least 24 hours before the forecast origin and
  from the same ENS run's ensemble-mean shape (valid time on or after the run's 00:00 UTC) otherwise,
  and from the climatology when neither exists; the construction is identical in training and at
  inference. At inference each member's gross quantiles are shifted down by the capacity times the
  ensemble-mean calibrated shape (coefficient 1 by construction), pooled, and given the spread
  factor and the 14-day correction on the net residuals. `b_member` shifts each member by its own
  shape.
- The shuffled-capacity twins `sf_twin`, `a_twin`, `am_twin`, and `b_twin` use the same code with the
  capacity values permuted between forecast dates within each series and calendar month (a fixed
  seed), as experiment 00031's A6 did.
- `sf_temp` (exploratory): `sf` with its shift multiplied by `1 - 0.004 * (T_cell - T_ref)`, where
  `T_cell` is the member-mean 2 m temperature plus 25 K times the shape, and `T_ref` is the
  training-window mean of `T_cell` on daylight rows. The report gives how much of the slope of the
  sunny-row quantile score on temperature the derating removes.
- Reference rows: `d0`, `d1l`, `d1l_ens_shift` (experiment 00046's fixed-1 member shift), and
  experiment 00033's `climatology` and `manual_heuristic` (stored predictions).

**Slope diagnostic.** The cause of the weak slope (-0.42 to -0.69 in experiment 51) is either
noise in the satellite-based estimate of the fleet's sunshine (regression dilution) or the model
already absorbing part of the anomaly through lagged power and cloud-proxy inputs. Four variants of
`d1l` are cross-fitted as above (seed 0): all inputs, no lagged-power columns, no cloud-proxy columns
(`dew_point_temperature_2m`, `precipitation`, `precipitation_type`, `longwave`, and the two
pressures), and neither. The write-up gives the slope of the out-of-fold residual on the oracle shift
for each. If the slope for "neither" is near -1 and the slope for "all inputs" is near -0.5,
absorption by the inputs explains it; if all four are near -0.5, dilution does.

**Arm sets, compute, and checks.** About 90 booster fits (6 refitted arms, 5 series, 3 seeds, about
100 s each) and 80 cross-fits (4 variants, 4 folds, 5 series), run 4 at a time inside the slot of
8 cores and 12 GB of GPU memory, one job per series and seed, each written atomically. Before the
fits: a smoke run on one series; a zero-shift check (every shift arm equals `d1l` when `w` and the
capacity are zero); the pooled-mixture check that `sf` with `w` = 0 reproduces the stored `d1l`
within 1e-6; and the training-window cut-off check (no input read after 2025-06-30 23:30 UTC in
training). The fit stops at the fold's training end.

**Reports.** Absolute quantile score of every arm and the reference rows, the eight decisive reads,
the lead-band reads (0-24 h, 1-2 d, 2-6 d, 6-10 d, 10+ d), the valid-month signs, coverage of the 0.05
and 0.95 quantiles in sunny and cloudy rows, the sunniest decile of rows by clear-sky index, each arm's
effective weight (the slope of its median minus `d1l`'s median on minus the capacity times the
ensemble-mean calibrated anomaly) against the slopes of experiment 51, the temperature read, and the
per-series reads. Anonymised series labels only.
