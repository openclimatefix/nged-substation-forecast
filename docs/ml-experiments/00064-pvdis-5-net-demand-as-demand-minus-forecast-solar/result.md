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
  not for `b`. A placebo twin (the forecast sunshine taken from another date) is expected to score
  worse than its real arm by 0.10 to 0.40 points for `sf`, `am`, and `b`, and by 0.02 to 0.15 for `a`, whose
  trees can learn to ignore a column that does not help. The 0.10 margin is read at 98.75%, where the
  half-width of the paired contrast of the fitted ensemble shift against `d0` is 0.079 points.
decision_rule: >-
  Two families of reads, each Bonferroni-corrected across the 4 arms (`sf`, `a`, `am`, and `b`), so
  98.75% intervals from 50,000 resamples of whole ISO weeks of forecast start (about 29 blocks) and
  of seeds, on the quantile score in points of capacity over the 5 verdict series' daylight
  April-to-September rows. The verdict family has two contrasts per arm. Contrast N is the arm minus
  `d0`. Contrast T is the arm minus its own placebo twin, which takes the forecast solar shape from
  another date (see the design). An arm passes the verdict family when both reads pass, which is an
  intersection test, so the two reads within an arm need no further correction. The non-inferiority
  margin is 0.10 points, fixed here from a power calculation on experiment 00046's stored
  predictions: at 98.75% the paired contrast of the fitted ensemble shift against `d0` has a
  half-width of 0.079 points with the same 29 weekly blocks, which exceeds 0.05, so the design
  cannot resolve a margin of 0.05, and 0.10 is about the whole 0.104-point effect of `d0`'s
  radiation inputs (experiment 00046). An arm is non-inferior when the upper end of its N interval
  is below +0.10. An arm carries solar information when the upper end of its T interval is below
  zero (the placebo keeps the capacity and the distribution of the shape and removes only the
  timing of the sunshine, so T measures what the forecast of the sunshine adds). `supported` when at
  least one arm is both. `refuted` when no arm is non-inferior and every arm's N interval lies
  wholly above zero. `inconclusive` otherwise, including an arm that is non-inferior but does not
  beat its placebo, which the write-up then reports as "matches, but the solar term adds nothing".
  The gain family has one read per arm, N over leads under 48 hours, also at 98.75%. It does not
  change the verdict. A gain is reported when the upper end is below -0.10 (the same size as the
  margin, because a skill gain smaller than the margin cannot be told from the explanation-only
  case). Nothing here is gated on the 0.35-point threshold of experiment 00046, because PVDIS-5
  tests explanation and non-inferiority first. All other reads are exploratory, labelled so, at 95%:
  the A-against-A-margin contrast, per-member against ensemble-mean shift, the temperature
  derating, each arm's effective weight, the sunniest decile, the lead-band and valid-month reads,
  and the slope diagnostic. The cost of porting `sf` is a capacity table to read and a table of
  fitted coefficients to keep current, with no new model fit in production; `a`, `am`, and `b` add
  the same tables and change the training target or the inputs.
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
00051); pre-register a gain read for 0 to 48 hours only, as its own corrected family (00046);
report coverage of the 0.05 and 0.95 quantiles in sunny and cloudy rows (00046); read the sign of each contrast by valid month (00046);
use no second weather forecast, because experiments 50 and 53 found none that improves ENS (00050,
00053); read the season-year capacity constant, not a weekly path (00045). Departed: none.

## Design

**Rows.** The 5 verdict series of experiment 00046 (D9, D10, D11, D14, D16; their capacity status in
experiment 45 is `clean` or `low-power`), the 3 seeds, and the scored rows of experiment 00046:
validation runs of fold A (every fifth day, valid 2025-07-01 to 2026-06-30), daylight half-hours of
April to September with a CAMS unit shape and an observed power. The training window is the same
as experiment 00046's (ENS control member, 2024-04-01 to 2025-06-30, half-life 2 years). No row on
or after 2026-07-01 is scored.

**Capacity.** One value per series, used for every training row and every validation row. It is the
2024/25 season-year constant of experiment 45, refitted with `fit_season_year_constant` on data
before 2025-07-01 only (the fit's `origin_week` set to the first week after the training end), so
that no scored actual enters the capacity. Experiment 45's final constant for D14 is zero in
2023/24 and 2024/25 (all weeks unobserved), so D14 reads experiment 40's 2023/24 `C`, the value
experiment 46 used; a pre-fit check stops the run if any verdict series has a capacity of zero. The
value is stale by the validation year: the 2025/26 constants of three series are 13% to 18% above
the 2024/25 ones, which biases the placebo contrast against the solar term. The write-up says so.

**Solar shape.** The CAMS unit PV shape `u` (experiment 46) for the past. For the forecast, the CAMS
climatology for the local month and half-hour plus the calibrated ENS anomaly, where the calibration
slope is fitted per series and lead band on the training window (experiment 46's
`calibration.parquet`). The headline arms use the ensemble-mean calibrated shape. The comparison
arms use each member's calibrated shape.

**Fitted weight `w`.** For each series, lead band (the 5 bands of experiment 46), and season (April
to June, July to September), `w` is the slope through the origin of the out-of-fold residual of the
median (actual minus median) on the shift that `sf` applies, minus the capacity times the
ensemble-mean calibrated ENS anomaly over the effective capacity, fitted on the daylight training
rows of that cell. This is the slope for the regressor that is applied at forecast time. The slope
on the oracle shift (the CAMS anomaly), which experiment 51 reported, is fitted too and reported
beside it. The out-of-fold residuals come from 4 time-block folds of the training window that group
by the valid time's calendar month (month index modulo 4), seed 0 only. The cells for July to
September rest on one summer (2024), and adjacent months train each other across folds; the
write-up says so. The same table of `w` is used by `sf` and `am`.

**Arms** (all use experiment 00046's `d1l` inputs and recipe unless stated):

- `sf`: the stored `d1l` quantiles shifted by `w` times the capacity times the ensemble-mean
  calibrated ENS anomaly over the effective capacity (predict-only on the stored boosters, no
  refit). `sf_member` shifts each member by its own calibrated anomaly, then pools.
- `a`: `d1l` plus one input column, the capacity times the calibrated ENS shape (the ensemble mean
  in training and in each member's row at inference), refitted.
- `am`: `d1l` with the XGBoost `base_margin` set to minus `w` times the capacity times the
  calibrated anomaly of the ensemble mean over the effective capacity, so the trees fit the
  remainder. The margin carries the anomaly only; the climatology is left to the trees. Training and
  inference both use the ensemble-mean anomaly (`u_mean` in `ens_train`). The smoke run asserts that
  the margin has shape (rows, 9) and is set on every inference matrix, that a zero-tree model
  returns exactly the margin, and that `w` multiplies the anomaly only.
- `b`: trained on gross demand, the metered power plus the capacity times the CAMS unit shape, with
  the `d1l` inputs. Training rows whose CAMS shape is missing are dropped. Each lagged input is the
  net lag plus the capacity times the shape at the lagged time. The shape comes from CAMS when the
  lagged time is more than 24 hours before the forecast origin (the delay is taken from experiment
  46's `TELEMETRY_DELAY` and the CAMS publication lag the study recorded, to be checked in the code
  review), and otherwise from the ensemble-mean shape of the latest ENS run available at the origin
  that covers the lagged time (today's 00:00 UTC run, else yesterday's, read from the NWP archive).
  The construction is identical in training and at inference, and the report gives the share of each
  source for every lag column. At inference each member's gross quantiles are shifted down by the
  capacity times the ensemble-mean calibrated shape (coefficient 1 by construction), pooled, and
  given the spread factor and the 14-day correction on the net residuals. `b_member` shifts each
  member by its own shape.
- The placebo twins `sf_twin`, `a_twin`, `am_twin`, and `b_twin` replace the forecast solar shape
  of every row by the shape of another forecast run in the same series, calendar month, half-hour,
  and lead band, drawn without replacement with a fixed seed, separately within the training rows
  and within the validation rows. The capacity and the distribution of the shape are kept, and the
  timing of the sunshine is destroyed. In `sf_twin` and `b_twin` only the inference-side shape is
  swapped; in `a_twin` and `am_twin` the column or margin is swapped in training and at inference.
  Experiment 00031's shuffled capacity could not serve here, because one capacity value per series
  leaves nothing to shuffle.
- `sf_temp` (exploratory): `sf` with its shift multiplied by `1 - 0.004 * (T_cell - T_ref)`, where
  `T_cell` is the member-mean 2 m temperature plus 25 K times the shape, and `T_ref` is the
  training-window mean of `T_cell` on daylight rows. The report gives how much of the slope of the
  sunny-row quantile score on temperature the derating removes.
- Reference rows: `d0`, `d1l`, `d1l_ens_shift` (experiment 00046's fixed-1 member shift, which
  used the 2023/24 capacity), and experiment 00033's `climatology` and `manual_heuristic` (stored
  predictions).

**Slope diagnostic.** The cause of the weak slope (-0.42 to -0.69 in experiment 51) is either
noise in the satellite-based estimate of the fleet's sunshine (regression dilution) or the model
already absorbing part of the anomaly through lagged power and cloud-proxy inputs. Four variants of
`d1l` are cross-fitted as above (seed 0): all inputs, no lagged-power columns (`lag_origin`,
`lag_same_time_latest_day`, `lag_same_time_latest_week`), no cloud-proxy columns
(`dew_point_temperature_2m`, `precipitation`, `precipitation_type`, `pressure_surface`, and
`pressure_msl`; `d1l` has no radiation columns already), and neither. The write-up gives the slope of
the out-of-fold residual on the oracle shift for each. If the slope for "neither" is near -1 and the
slope for "all inputs" is near -0.5, absorption by the inputs explains it; if all four are near
-0.5, dilution does.

**Compute and checks.** About 90 booster fits (6 refitted arms, 5 series, 3 seeds, about 100 s
each) plus the placebo refits of `a` and `am` (60), and 80 cross-fits (4 variants, 4 folds, 5
series), run 4 at a time inside the slot of 8 cores and 12 GB of GPU memory, one job per series and
seed, each written atomically. Before the fits: a smoke run on one series; a zero-shift check (every
shift arm equals `d1l` when `w` is zero); the check that `sf` with `w` = 0 reproduces the stored
`d1l` within 1e-6; the capacity check above; and the training-window check (no input read after
2025-06-30 23:30 UTC in training). The fit stops at the fold's training end.

**Reports.** Absolute quantile score of every arm and the reference rows, the verdict and gain
reads, the lead-band reads (0-24 h, 1-2 d, 2-6 d, 6-10 d, 10+ d), the valid-month signs, coverage of
the 0.05 and 0.95 quantiles in sunny and cloudy rows, the sunniest decile of rows by clear-sky index,
each arm's effective weight (the slope of its median minus `d1l`'s median on minus the capacity times
the ensemble-mean calibrated anomaly) against the slopes of experiment 51, the temperature read, and
the per-series reads. Anonymised series labels only.
