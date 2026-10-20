---
id: "00119"
slug: differentiable-physics-enc-weather-patch-encoder-on-ens
title: "Differentiable physics, ENC: a weather-patch encoder around each Primary, on ENS"
part: "ENC"
theme: differentiable-physics-forecast-engine
also_relevant_to:
  - weather-products-and-blending
  - calibration-and-uncertainty
  - differentiable-physics-demand-calendar-and-weather
status: planned
size: >-
  medium. Stored private files only; the serving path, the degradation rules, and the asset graph
  are untouched. Several designs are defensible (patch, output, placebo, replicate), and the plan
  pre-registers one of each. The code is confined to this experiment's folder.
hypothesis: >-
  A small neural network that reads the ENS grid boxes around a Primary, and not only the one box
  GAM-K reads, lowers GAM-K's quantile score a little on average and more on the cold, showery,
  and badly forecast days NGED cares about, either as a correction to GAM-K's quantiles or as a
  replacement for its single-box temperature and irradiance.
prediction: >-
  C1, ENC-c minus GAM-K, is -0.02 to -0.08 points over leads 0.5 to 168 hours and -0.03 to -0.10 at
  D+1, in both years, with 3 to 5 of 5 sites better. C3 (against the placebo from another date) is
  -0.02 to -0.07, most of C1. Hard-day contrasts are wide and carry no label.
decision_rule: >-
  The verdict is C1's label. Supported if the scored-year interval lies wholly below zero, the
  estimate is at least 0.04 points (the cost of a neural network and a patch-extraction step in
  serving), the replicate year's estimate is below zero and at least half as large, at least 4 of
  5 sites improve in each year, C3 is below zero in both years, and the known-answer gate passed.
  The plan lists small gain, partly supported, refuted, and inconclusive.
benchmark: >-
  GAM-K as stored by experiment 111 (reproduced); the encoder given only the centre box, a distant
  site's patch, and the same site's patch from another date; climatology; manual_heuristic; BG2
row_set_id: >-
  experiment 111's GAM-K predictions at D5, D10, D11, D14, D16 (D9 beside), plus GAM-K fits for
  2024-04 to 2024-06: 456 training runs and the 291 scored runs of 2025-07 to 2026-06
metric: >-
  Quantile score (twice the mean pinball loss over experiment 87's 11 quantiles), points of
  effective capacity, outside switching events, leads 0.5 to 168 hours; D+1 and hard days beside it
production_status: not-for-production
code: ml-experiments/00119-differentiable-physics-enc-weather-patch-encoder-on-ens/
---

**Planned; no result yet.** The plan below was committed before any code ran.

## What ENC asks, and the prior it starts from

**ENC asks whether the weather around a Primary carries information that GAM-K's single ENS box
does not.** GAM-K reads the ENS temperature and calibrated irradiance at one box per Primary. A
front, a sea breeze, or a convective line a box away moves demand, and the box-level ensemble
spread only partly stands in for that. The idea follows neural post-processing of ensemble
forecasts ([Rasp and Lerch, 2018](https://doi.org/10.1175/MWR-D-18-0187.1)).

**The prior is a small average gain.** ERA5 ([Hersbach et al., 2020](https://doi.org/10.1002/qj.3803))
quality weather lowered the Primary score by only about 0.10 points in the weather-products theme,
and the patch partly duplicates the spread of the 51-member ECMWF ensemble
([Molteni et al., 1996](https://doi.org/10.1002/qj.49712252905)) that GAM-K's mixture carries. The
expected average gain is 0.02 to 0.08 points, and larger on hard days.

**Builds on** experiments [111][e111] (GAM-K), [112][e112] (start dependence), and [113][e113]
(weak identification).

[e111]:
    https://openclimatefix.github.io/nged-substation-forecast/ml-experiments/00111-differentiable-physics-6-ens-forecast-engine-against-bg2/result/
[e112]:
    https://openclimatefix.github.io/nged-substation-forecast/ml-experiments/00112-differentiable-physics-vp-variable-projection-fit-of-pooled-model/result/
[e113]:
    https://openclimatefix.github.io/nged-substation-forecast/ml-experiments/00113-differentiable-physics-n5-convex-identification-at-realistic-noise/result/

**Wind and solar sites are not part of ENC.** The maintainer expects a larger gain there, but
GAM-K is a demand model and 3 wind farms cannot rank an arm. A follow-up waits for C1 and C3.

## Data on local disk, and what it leaves

**Only `data/NWP/nwp_model_id=ECMWF_ENS_0_25_degree` is read. No fetch, no AWS.** On 2026-10-20:

- **Runs:** 932 daily 00:00 UTC runs, 2024-04-01 to 2026-10-19, one per day. Extraction checks
  every run for 51 members (0 to 50) and reports any short run.
- **Leads:** 0 to 360 hours, 3-hourly to 144 and 6-hourly after: 53 steps up to 168 hours.
- **Grid:** 1,671 H3 resolution-5 cells ([H3](https://h3geo.org)) over Great Britain, about 1,500
  distinct ENS values per field, so neighbouring cells can share one 0.25° ENS box. The patch is a
  Primary's cell plus three hexagonal rings (37 cells), fixed by geography alone.
- **Fields:** the encoder reads five of 17 columns (temperature, dew point, shortwave,
  precipitation, mean-sea-level pressure) as member mean and standard deviation. Experiment 111's
  `ens_store` holds only the centre box, so the patch frames are new (at most 259 cells).

| Months | Count | Role |
|---|---|---|
| 2024-04 to 2025-06 | 15 (456 runs) | F1 training year (GAM-K fits for 2024-04 to 06 made first) |
| 2025-07 to 2026-06 | 12 (291 runs) | F1 scored year; F2 training year |
| 2024-07 to 2025-06 | 12 (365 runs) | F2 scored year (replicate) |
| 2026-07 to 2026-10-19 | 3.6 | reserved for a finalist (skill rule): never used in ENC |

**Tuning is month-block cross-validation**
([Roberts et al., 2017](https://doi.org/10.1111/ecog.02881)) with 7 days purged either side: F1 has
five blocks of 3 months (12 train, 3 tune), F2 four (9 train, 3 tune). The final fit uses the whole
training year, for the median of the blocks' early-stopped epochs, fixed before scoring.

## The two designs, and the arms

**Both designs take the stored GAM-K predictions as the base and change nothing in GAM-K.** The
fits for 2024-04 to 2024-06 use experiment 111's code and protocol (each month's fit stops before
the month) and are made before any scoring.

**Encoder input and size.** One sample is a (site, run, 3-hourly step) triple: about 145,000
samples in F1, but only about 456 independent weather days. Inputs are 5 fields × (member mean,
member standard deviation) on 37 cells, each field minus the centre cell's value, plus the centre
cell's absolute values, the lead, half-hour-of-day sine and cosine, and GAM-K's mean forecast.
**All means and standard deviations used to standardise inputs come from the direction's training
months only** (per lead band), and the scored year is transformed with those numbers. A half-hour
takes the patch of the last 3-hourly step at or before its valid time (causal, no interpolation),
for every arm and placebo. The encoder shares weights across sites (the weights see a cell's offset,
not its coordinates), has at most 3,000 parameters, and gets one site-offset scalar per site under
a heavy penalty.

| Arm | Output and input | Role |
|---|---|---|
| K0 | GAM-K as stored | benchmark, reproduced |
| ENC-c | signed shifts of GAM-K's 11 quantiles | design (a): decision arm |
| ENC-c0 | ENC-c, centre cell only | patch against network flexibility |
| ENC-cps | ENC-c trained and scored on the most distant decision site's patch, same date | placebo: regional weather |
| ENC-cpt | ENC-c trained and scored on the same site's patch from another date | placebo: timing |
| ENC-r | temperature and irradiance offsets through K0's sensitivities | design (b) |
| ENC-r0 | ENC-r, centre cell only | design (b)'s control |

- **Design (a), correction (ENC-c).** The output layer is zero-initialised, so the untrained arm is
  K0 exactly. It adds a free signed shift to each of the 11 quantiles, then sorts them, so the
  bands can narrow as well as widen. The loss is the pinball loss
  ([Koenker and Bassett, 1978](https://doi.org/10.2307/1913643);
  [Gneiting and Raftery, 2007](https://doi.org/10.1198/016214506000001437)) of the corrected
  quantiles against `actual`. A factor λ in {0, 0.5, 1} scales the shifts and is chosen
  out of fold and reported per arm; λ = 0 is labelled "no gain found", not `refuted`.
- **Placebos are trained on the foreign input, not only scored on it.** ENC-cps uses the decision
  site farthest from the Primary (no shared cells); ENC-cpt draws, for each (site, run), another
  run of the same year within ±15 days of the same day of year and the same lead (seeded, at least
  2 days away). **C3 uses ENC-cpt; ENC-cps is descriptive:** it keeps regional weather, so ENC-c
  minus ENC-cps reads as local detail beyond regional weather.
- **Design (b), replacement (ENC-r).** The encoder outputs ΔT in ±5 K and an irradiance multiplier
  in [0.7, 1.3] per (site, step). The forecast moves by K0's own sensitivities ∂F/∂T and ∂F/∂G per
  site and half-hour, from finite differences (±1 K, ±10%) of **the same per-month out-of-sample
  GAM-K fit that produced each stored prediction**, so no scored-year information enters (about 1
  CPU-hour). **First-order screening version:** the shift is applied after the filter, whose
  residual stream and spread stay K0's; the exact version is the confirmation tier. The GAM cannot
  be refitted on encoded weather, because ENS covers 27 months.
- **Seeds and training.** Seeds 11 to 15; an arm's forecast is their mean, and per-seed scores are
  the start-dependence reference. AdamW
  ([Loshchilov and Hutter, 2019](https://arxiv.org/abs/1711.05101)), learning rate 3e-3, batch
  4,096, at most 60 epochs, early stopping on out-of-fold pinball loss (patience 8), dropout 0.2 on
  whole cells ([Srivastava et al., 2014](https://jmlr.org/papers/v15/srivastava14a.html)), width
  16, weight decay c / N_days (c in {0.4, 4, 40}; N_days = training days). The best c is recorded
  in `decisions.json` before any scored run. ENC-c0 and the placebos inherit ENC-c's; ENC-r has its
  own grid, and ENC-r0 inherits it. Out-of-fold scores are optimistic, so **every reported score
  comes from the final fit on the scored year**.

## Hard cases, defined before scoring

**The three subsets are fixed before any ENC prediction is scored (`hard_days.json`, committed
with the planned result).** Days are UTC days; thresholds pool 2024-07-01 to 2026-06-30.

1. **Coldest 5%.** Daily mean temperature over the five decision sites, from the centre-cell
   control member of each day's 00:00 UTC run at leads 9 to 33 hours (experiment 111's analysis
   proxy): about 36 days shared by all sites, about 18 per year. No power data.
2. **Most convective or showery 5%.** ENS has no convective-available-potential-energy field, so
   the proxy is the spatial spread of precipitation: the standard deviation across the 37 patch
   cells of the control member's precipitation at each 3-hourly step, averaged over the day.
   Among days with at least the median wet day's patch-mean total, the top 5% by that index, per
   site (about 36 days per site). No power data.
3. **Largest GAM-K errors 5%.** The 5% of (site, day) pairs with the largest mean absolute error
   of K0's `q50` at D+1. **This subset is defined from the scored year's outcomes and flatters any
   arm that differs from K0** (regression to the mean), so only the placebo-corrected contrast is
   read, and it is not an out-of-sample finding like C1.

**Power.** About 36 independent days give a 99% half-width near 0.2 points (per-day spread near
0.5), so every subset under 60 independent days per year is **descriptive: no label**. The gate's
sparse injection is repeated on each subset's days to show how much of a true 0.05-point gain is
recovered, with its interval.

## Known-answer gate (blocking, sized as a power check, before any F1 or F2 scored fit)

**A failed blocking test stops scoring until fixed; if unfixable, the verdict is `inconclusive`.**
The gate runs on F1's training year only (five blocks), so the scored year stays unopened.

1. **Injected patch signal is recovered at realistic noise.** Add to `actual` a known zero-mean
   function of the patch, g = 0.5 × (mean temperature east of the centre minus west, standardised)
   and 0.5 × (mean shortwave north minus south, standardised), scaled by α. **Dense:** every day,
   sized from the real residual standard deviation (about 3.5 points) so that an exactly recovered
   g lowers the score by 0.10, 0.05, or 0.025 points. **Sparse:** g only on the 10% of days with
   the largest east-minus-west gradient, sized for 0.05 points over all days (real signal is
   sparse). **ENC-r's injection:** a standardised east-minus-west ΔT added through K0's
   sensitivities, sized for 0.05, dense and sparse. **Blocking pass, for ENC-c and ENC-r, dense and
   sparse, at 0.05:** at least half of the gain recovered at 4 of 5 decision sites, with the
   contrast's interval wholly below zero. A gate that cannot detect 0.05 makes a
   real C1 whose interval includes zero `inconclusive`. ENC-c0, ENC-cps, and ENC-cpt recover under
   20% of every injection (the signal is not in their input).
2. **No false gain on clean data.** With α = 0, ENC-c, ENC-r, and each placebo differ from K0 by
   at most ±0.01, and the first two intervals include zero.
3. **Reproduction:** at zero initialisation ENC-c and ENC-r equal K0's quantiles to 1e-6 (float64).
4. **Causality and leakage, also a pytest in the experiment folder:** changing any ENS run issued
   after τ, or any reading after τ minus 3.5 hours, leaves every arm's forecast at τ unchanged
   exactly; a tuning block and its purge never enter that block's fit; changing any scored-year
   value moves no standardisation statistic, epoch count, or setting.
5. **Seed stability and overfit:** the five seeds' ENC-c − K0 on the training blocks spread by at
   most 0.02 points (experiment 112: starts 1.6 points apart meant an under-regularised fit), and
   training pinball loss is within 25% of out-of-fold loss, else c moves up one step.

## Directions, sites, and rows

- **F1** trains on 2024-04 to 2025-06 and scores 2025-07-02 to 2026-06-29 (the 291 stored runs): the
  verdict. **F2** trains on 2025-07 to 2026-06 and scores 2024-07 to 2025-06 (365 runs): the replicate.
- **F2 reverses time** because local ENS holds only two scoring years. **F2's settings are chosen on
  F2's own training year.** Two caveats: K0's 2025-26 fits saw more history than its 2024-25 fits,
  and ECMWF may have changed its model cycle between the years
  ([IFS documentation](https://www.ecmwf.int/en/publications/ifs-documentation)). The cycle dates
  are checked against the run dates before F2 is read.
- **Sites.** D5, D10, D11, D14, D16 decide (4 of 5); D9 beside. D15, D18, MVA Primaries are out.
- **Rows:** present rows outside the hand-filtered events, shared across a contrast's arms, all-rows
  score beside. **Lead bands:** 0.5 to 168 hours (headline), D+1 (15.5 to 39), 0 to 24, 24 to 48,
  48 to 144, 144 to 168.
- **Robustness rows:** November to March against April to October; leaving one site out; working
  days against weekends and bank holidays. Predictions carry the keys, `q01` to `q99`, `actual`,
  `lead_hours`, `arm`, `seed`, `switching_mask`.

## Planned contrasts and margins

**Five planned contrasts, 99% paired intervals (Bonferroni over five) from `studies.bootstrap`,
which resamples whole calendar months** ([Künsch, 1989](https://doi.org/10.1214/aos/1176347265)),
3 resampling seeds, with site counts, every arm's absolute score, the MAE of the median, the
mean-forecast error beside the quantile-score contrast, and the replicate year's estimate.

- **C1, the correction (decision):** ENC-c minus K0, over leads 0.5 to 168 hours.
- **C2, patch against network flexibility:** ENC-c minus ENC-c0 (descriptive).
- **C3, date-aligned information:** ENC-c minus ENC-cpt. ENC-c minus ENC-cps is C3b, descriptive.
- **C4, the replacement:** ENC-r minus K0 (labelled; it does not change the verdict).
- **C5, hard days:** per subset, ENC-c minus K0 and ENC-c minus ENC-cpt (descriptive, see Power).

**Noise reference.** Part 4's C4 half-width was about ±0.023 at 97.5% (near ±0.027 at 99%); W1's
D+1 screen had about ±0.04 at 98.75%. **The smallest gain worth porting is 0.04 points**, above
W1's 0.03, because the encoder adds a neural network and a patch-extraction step to serving.
Hard-day subsets use a paired 7-day block bootstrap of whole days (some months hold no rows).

## Predictions

Written before any fit; each line can fail, and none overrides the decision rule.

- **C1:** ENC-c minus K0 is -0.02 to -0.08 over all leads and -0.03 to -0.10 at D+1, in both
  years, 3 to 5 of 5 sites better; q05, q95 coverage within 3 points of nominal, q01, q99 within 1.5.
- **C2:** -0.01 to -0.05. **C3:** -0.02 to -0.07, most of C1. **C3b:** about half of C3 (a distant
  patch keeps regional weather). **C4:** -0.01 to -0.05, smaller than C1 (a constrained output).
- **C5:** ENC-c minus ENC-cpt is -0.10 to -0.40 on the coldest 5%, -0.05 to -0.30 on the showery
  5%, and -0.05 to -0.25 on the largest errors; each interval is wide, and no label is given.

## Decision rule

**The verdict is C1's label.** C2 to C5 are labelled in the bullets and the take-home only, and a
second year is required, as in W1.

- **`supported`:** C1's scored-year (F1) interval lies wholly below zero, its estimate is at least
  0.04 points, F2's estimate is below zero and at least half of F1's, at least 4 of 5 sites
  improve in each year, C3 is below zero in both years, and the gate passed.
- **`small gain`:** C1's scored-year interval lies wholly below zero with an estimate under 0.04
  points, and F2's estimate is below zero.
- **`partly supported`:** C1's estimate is below zero in both years but the scored-year interval
  includes zero or fewer than 4 sites improve; or C1 is `supported` or `small gain` while C3 is
  not below zero in one year (the gain does not come from date-aligned patch information).
- **`refuted`:** C1's interval lies wholly above zero in either year.
- **`inconclusive`:** anything else, a gate test that cannot be fixed, or the gate's power
  finding with C1's interval including zero.

**Hand-ons:** the encoder is a candidate for GAM-K if C1 is `supported`, or `small gain` with C3
below zero in both years; if C4 beats C1, the replacement is. Patch extraction moves to a
reviewed `studies` module only when C1 is `supported`.

## Checks: the seven audit-before-merge triggers

Also report median minus mean forecast per site and arm, and mean-forecast error beside each contrast.

1. **A control fails:** a gate test, K0's reproduction, or a placebo recovering the injection.
2. **A result contradicts a prediction or finding:** C1 beating the 0.10 ERA5 bound.
3. **Quantile score and MAE of the median disagree in sign** on C1 to C4.
4. **Leaving one decision site out flips a sign or label** of C1 to C4.
5. **A nested arm beats its container:** ENC-c0, ENC-cps, or ENC-cpt beating ENC-c.
6. **An interval over twice or under half its sister's:** W1's KW minus K0; 111's GAM-K minus GAM-0.
7. **A later design depends on an unreplicated result:** encoder adopted while F2 disagrees.

## What the write-up must contain

The take-home says how much the patch lowers the score on average and on the hard days, whether
the gain is local detail beyond regional weather (C3b), and what it narrows the next step to.

- **Figures, raw data first:** the patch fields on three days (coldest, showery, ordinary); the
  hard-day calendars; the C1 to C5 headline chart with intervals, site counts, and both years;
  score by lead band per arm with BG2, GAM-0, and K0; gain by month and hour band; the gate
  (dense and sparse); training against out-of-fold loss; the five seeds; the tails.
- **Tails and absolute scores:** coverage and pinball at q01, q05, q95, q99 per arm and lead band
  (`studies.tail_calibration`); every arm's and reference's absolute score on shared rows.
- **Substation cards** ("Provisional: fitted to past weather, forecast on ENS, screening run";
  ENC-c's F1 fit): per decision substation, by D-label, the number of distinct ENS boxes in its
  patch; the learned weight of each distinct box (the sensitivity of the median shift to that
  box's temperature), drawn on the hexagon patch without coordinates, with the spread across the
  five seeds and five blocks as the error bar; the shift against lead; and **example forecasts**
  on four weeks fixed here, before any score is read: the Monday-to-Sunday weeks starting
  2025-07-21, 2025-10-20, 2026-01-19, and 2026-04-20 (the third Monday of the second month of each
  season; the next fully present week if a run is missing). Each panel shows the median and 80%
  and 90% bands of K0 and ENC-c against the observed series, switching events marked.
  **Line on every card:** "Provisional; patch weights are not identified
  individually (neighbouring cells share an ENS box); read the ring and east-west contrast, not
  one cell", and no physical direction is claimed from one site.
- **Caveats:** one frozen fit and no refits; F2's two caveats; the largest-error subset is selected
  on outcomes; ENC-r is first-order and applied after the filter; the encoder sees member mean and
  spread, not 51 members; the 3.5-hour delay is assumed; experiment 111's inherited leaks (ENS
  calibration fitted on the odd months, hindsight pinned capacity, ERA5 in the GAM's last training
  rows) and the hindsight masks apply to K0 and the encoder alike.

## Privacy

**D-labels only, with dates, for D5, D9, D10, D11, D14, and D16.** D15, D18, and their sum are
generator data: ENC does not use them (any number from them would be G-c, normalised, no dates,
never beside a D-label). The page may name interconnected pairs such as D5-D16 but no substation
names, no other topology, and never a generator's public name with its series.

## Run order and compute

**Order:** extract the patch frames; make the 2024-04 to 06 GAM-K fits and the ENC-r sensitivities;
write `hard_days.json`; code and gate tests 3 to 5; **time one ENC-c fit, record it in
`decisions.json`**; F1 grid, gate, fit, scoring; F2 grid, fit, scoring; ENC-r arms; report.

**Compute estimate, at the planned 1.0 minute per fit (GPU: one RTX A6000 share, up to 10 GB):**

| Item | Fits | GPU hours |
|---|---|---|
| Grids: ENC-c and ENC-r, 3 values of c × 9 blocks each | 54 | 0.90 |
| Gate on F1 blocks: ENC-c 5 injections, ENC-r 4, ENC-c0 and 2 placebos 1 each, × 5 blocks | 60 | 1.00 |
| Final fits: 6 arms × 5 seeds × 2 directions | 60 | 1.00 |
| Overhead: reruns, hard-day injections | | 0.50 |
| **Total** | **174** | **about 3.4** |

Peak GPU memory is under 3 GB; a fit under 30 seconds lets the 5 seeds be stacked. **If the timed
fit exceeds 1.1 minutes, cut in this order:** seeds from five to three, ENC-r0, the gate's 0.10
size, the sparse hard-day injections; never ENC-c0, the placebos, or the 0.05 gates. Data
extraction is CPU (about 2 CPU-hours).

## Pre-fit clarifications (2026-10-20, before any encoder is fitted)

**Winter breakdown (maintainer priority, relayed by the manager).** The best model goes to
production in about a month and NGED first sees it in winter, so the page reports a winter breakdown
beside every annual score.

- **Winter rows:** November to March, by the run date's month, for every arm and every contrast
  (C1 to C5), with paired intervals from the same month-resampled bootstrap on the rows where all
  compared arms have predictions, and the annual score beside it. Evening peaks within winter are
  the half-hours 16:00 to 20:00 UTC.
- **Cold snap:** the day belongs to the coldest 5% of days (`hard_days.json`, subset 1, 37 days
  over the two years, about 18 per year, defined before any score). The cold-snap panel of each
  substation card shows K0 and ENC-c against the observed series on that year's cold-snap days
  that fall in the scored year, with the day count stated, and says "too few days" for fewer than
  10 days with rows at the site.
- **Independent winters:** the local ENS holds two winters, 2024-25 (scored in F2) and 2025-26
  (scored in F1). Each direction's winter estimate rests on one winter, so no winter result is
  labelled by itself, and the page says so.
- **Verdict rule:** an arm that loses in winter (winter estimate above zero with an interval
  wholly above zero in either year, or above zero in both years) is held back even if the annual
  mean improves, and the verdict says so in its first sentence. This does not change the decision
  rule's labels.

**Corrections to the numbers above, found while building the data.** (1) The first 9 runs of
2024-04 have no week of analysis-proxy weather before their origin, so experiment 111 never
forecast them: F1 trains on 447 runs (2024-04-10 to 2025-06-30), not 456. (2) A run's half-hours
reach 177 hours after its 00:00 initialisation, so the encoder reads 54 ENS steps (3-hourly to
144 hours, 6-hourly to 174 hours), not 53. (3) Experiment 111 stored GAM-K for the tuning months
(odd months of 2024-07 to 2025-06), the replicate months (even months), and the scored year, but
not for 2024-04 to 2024-06; ENC re-forecast those 91 runs from experiment 111's stored monthly
fits, and the check that the same code re-forecasts 4 stored runs per site gave a largest
difference of 0 (`e00119_gamk.py`).

**Two plan flaws, fixed before any gate or scored fit (2026-10-20).** The manager accepted both
rulings. They were made after the data extraction, the K0 reproduction, and the first run of the
settings grid (the grid showed an out-of-fold gain of about 0.035 points for ENC-c on the clean
training year), and before any known-answer-gate fit and before any scored-year prediction.

- **Gate test 2 cannot be a gate on real data.** The plan requires every arm to differ from K0 by
  at most 0.01 points on the clean training year. A real gain of the predicted size (0.02 to 0.08
  points) would fail that test, so it could never tell a false gain from a true one. Test 2 is
  applied to permuted-actual data instead: each row's `actual` is replaced by the `actual` of the
  same site at the same lead from another run of the same year (2 to 15 days away), so the patch
  has no link to the target. The bounds are unchanged (at most 0.01 points, and the intervals of
  ENC-c and ENC-r include zero). The literal clean-data reading is run and printed, and the page
  reports it as descriptive.
- **No decision site is free of shared ENS cells.** The six patches overlap by 12 to 30 of their 37
  cells, so the plan's "farthest decision site, no shared cells" does not exist. ENC-cps uses the
  farthest decision site by distance, and the page reports the shared-cell count for each site.
  Shared cells mean the placebo can see some of the real patch's weather, so a gain by ENC-cps is
  read as a leak-through warning and not as noise. ENC-cps stays descriptive (C3b).
