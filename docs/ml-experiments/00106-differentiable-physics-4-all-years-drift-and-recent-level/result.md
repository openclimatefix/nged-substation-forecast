---
id: "00106"
slug: differentiable-physics-4-all-years-drift-and-recent-level
part: 4
theme: differentiable-physics-long-history-and-drift
also_relevant_to: [lagged-power-and-error-correction, retraining-and-training-history]
status: planned
size: >-
  complex for the module changes (drift and pinned capacity in the pooled model, a Kalman
  assimilation module, batched refits by training cut, all in packages/studies and reused by Part
  6); medium for the experiment scripts. See "Size triggers".
hypothesis: >-
  The DP trained on every usable year from 2019-09 with a drift term forecasts Primary demand better
  than the same model on its best window, and assimilating recent residuals through a Kalman filter
  makes it better again, by more than a simple moving-average correction does.
prediction: see "Predictions"
decision_rule: see "Decision rule"
benchmark: >-
  The same DP on its best window (12 months or 2 years, with or without drift, chosen on 2023-24);
  Part 2's convex model G-all+D1 (2.78) and G1 (2.82) as stored; lagless XGBoost with its default
  start, a zero start, and experiment 64's solar offset with each start; an exponentially weighted
  mean residual correction; climatology; manual_heuristic; lagged XGBoost as a descriptive reference
  at leads of 24.5 to 48 hours and 144.5 to 168 hours
row_set_id: >-
  dp_mw_v1 and dp_mva_dark_v1 (experiment 103); MW sites scored 2025-07-01 to 2026-06-30, replicate
  2024-07-01 to 2025-06-30
metric: >-
  Quantile score (twice the mean pinball loss over experiment 87's 11 quantiles) in points of
  effective capacity, on rows outside the hand-filtered switching events at the scored site, at the
  day-ahead lead (day D+1 from a 09:00 UTC initialisation on day D)
verdict: <supported, partly supported, refuted, or inconclusive>
production_status: not-for-production
take_home_message: <written after the results>
date: <date>
pr: <pr>
advice: <written after the results>
code: ml-experiments/00106-<slug>/
---

# Differentiable physics, part 4: all years with drift, and recent-level assimilation

**Status: planned.** This page is the plan, committed before any code runs. The write-up
replaces it when the experiment finishes.

## What Part 4 asks, and the bar it starts from

**Part 4 asks two questions.** First, the series' long-history criterion: does the best drift arm
trained from 2019-09 beat the same model on its best window? Second, the 2×2 of drift against
assimilation: does a Kalman filter on recent residuals, the DP's analogue of XGBoost's power lags,
add to drift, replace drift, or replace refitting? SIMP1 (experiment 102, provisional) found that
removing the lags makes Primary demand about 0.9 points worse, so assimilation is where the largest
gain should be.

**Part 2's corrected scored-year numbers (G-all+D1 2.78, G1 2.82, lagged XGBoost 2.49 at 24.5 to
48 hours) are unreplicated**: the second audit (`e104_audit2_result.md`) found two of three
history-window contrasts reverse sign in the tuning year. Three findings shape this plan:

- **A one-year history contrast is not a finding.** Every decision contrast is scored on a second
  frozen year, and every decision needs the replicate's sign to agree ("Decision rule").
- **Winter and summer pull opposite ways** (all years with drift −0.40 in November to March, +0.19
  in April to October; D14's gain all winter, D10's loss all summer). Every contrast is split by
  season.
- **G-all's loss to G1 is mostly D16** (without D16 +0.03). Every contrast is reported with and
  without D16.

**D1 is the default drift, not D1 with steps.** Part 2's `tuned.json` names `d1_steps` on a 0.215
gain measured on the year the step penalty was tuned on; out of sample steps lost (+0.08, +0.53 at
D10 alone, where the steps track D10's unlabelled November 2024 jump and the May to June 2025 rise).

## Which model Part 4 runs on

**Part 4 runs its grid on model M, chosen by this rule when Part 3's result is final, before any
Part 4 scored fit:**

1. **M = P (the pooled DP)** with Part 3's handed-on configuration if all four hold: Part 3's
   verdict is `supported` or `partly supported`; Part 3's repaired gate passes; the gate review's
   module split of the injected-farm midday movement (its fix 3.3) and its second-seed noise floor
   (fix 3.4) have been run and reported; and P-all minus G-all is at most +0.10 points in Part 3's
   scored year and, if Part 3 scores its tuning year as a replicate, in that year too. The rule
   reads Part 3's scored year, which the write-up says.
2. **Otherwise M = G**, Part 2's convex per-site model (`studies.dp_gam`). Part 4 then needs no
   pooled drift module.

**The convex grid runs whatever M is:** it takes minutes on the CPU, carries C5, and is the first
test of Part 6's GAM-K arm.

**Capacity is pinned in every primary arm (maintainer-delegated ruling, 2026-10-19).** A capacity
that moves with the fit's starting point adds noise to every history contrast, and Part 3's interim
result is that the pooled capacity is not identified by the data (different starts reach different
capacities at nearly the same fit; noise floor about 3 points).

- **The pinned capacity.** For a fit with cut c, each site's weekly capacity is the `frozen`
  vintage of `capacity_weekly_vintages.parquet` at the latest `origin_monday` on or before c, read
  week by week for training rows (the vintage holds weekly values back to 2019-09-16). After c the
  forecast holds the median of the vintage's last 13 weeks, not its last week, which
  `data-processing.md` warns has 3.63 times the path error. D14 is never read for 2023-09-04 to
  2025-08-25 and is held at its last readable value, as in Part 2. The write-up prints each site's
  pinned value beside G's fitted capacity.
- **Before the first vintage.** The vintages start at origin 2023-08-07, so the tuning cut
  (2023-07-01) has none. That fit, the DP and the XGBoost solar offset alike, takes the earliest
  vintage scaled back by the series plan's season-year growth prior. Neither brief gives the prior
  a number, so the module PR fixes it before any fit. The earliest vintage uses data up to 5 weeks
  after the tuning cut; this leak touches only the tuning-year choices, and the write-up discloses
  it.
- **The fitted-capacity variant (`-fc`)** inherits Part 3's capacity fix if one is settled and
  has passed Part 3's repaired gate; otherwise it starts capacity at the pinned vintage, not at
  softplus(−3). It reports how far the fit moves from the vintage, never enters a decision
  contrast, and on M = G is Part 2's behaviour.
- **Pinning does not close the network's route for solar.** The Part 3 gate review found the
  interaction network carrying solar (the shared midday output moves when a farm is injected), and a
  wrong pinned value gives the network more reason to compensate. Gate test 4 measures it.
  Whether the capacity can be identified at all is Part 5's question; Part 4 only points to it.

## Size triggers

**Stored data:** private files under `data/ml-experiments/<id>-<slug>/` and the feature cache only.
**Serving path and degradation rules:** not touched. **More than one defensible design:** yes (the
filter's states and spread, its hyperparameter fit, the final knot, the refit batching); this plan
pre-registers one of each. **Callers:** Parts 4 to 6's scripts only; torch stays in
`packages/studies`. The module changes get their own PR off `main`, reviewed and merged as in Part 3
(Opus code review, leakage first: the filter's information cut, then each refit's training cut).

## Module changes (tests under 5 minutes each)

- **Drift in the pooled model (`studies.dp_pooled`, only if M = P).** Replace the `drift`
  switch's `NotImplementedError` with D1 (a level per site and hour band on monthly knots,
  second-difference penalty scaled by training site-years, as in `dp_gam`) and `d1_steps`. The
  drift is centred per site, and it holds its last value at forecast time.
- **Pinned capacity (`dp_gam`, `dp_pooled`).** `capacity="pinned"` takes a weekly capacity per
  site in points as a fixed input and removes the capacity parameters (a unit test, not evidence).
- **COVID flag (`dp_gam`, `dp_pooled`).** An indicator per hour band over the fixed lockdown window,
  for the one arm that keeps the lockdown rows.
- **The final knot (`dp_gam`, `dp_pooled`).** `final_knot` is `free` (Part 2), `shrunk` (the held
  value is the last knot shrunk by λ towards the mean of the last 12 knots, at forecast time only),
  or `pinned` (the last knot tied to the one before it in the fit). Part 2's audit found the last
  knot follows the slope of the last two months.
- **Batched refits by training cut.** Each copy builds its own D1 knot grid counted back from its
  own cut c, so no knot lies after c; its row weights zero every row ending after c, and its centring
  means and any feature scaling come from its weighted rows only. `fit_many` fits a 4-weekly (13
  cuts) or weekly (52 cuts) schedule as copies. The module PR measures peak GPU memory for B = 6 and
  B = 12 on all years (Part 3's 10 GB estimate was for 2 years), which sets the batch size.
- **New module `studies.dp_assimilation`:**
    - **Kalman filter.** Per site and hour band, two hidden states on that band's residual stream: a
      persistent level ℓ_b (random walk, step variance q per day) and a transient anomaly z_b
      (AR(1), factor φ per half-hour, innovation variance σ_z²), observed as ℓ_b + z_b plus noise
      σ_ε². Time is wall-clock half-hours, so z decays across the band's daily gap and across the
      delay; a missing reading skips the update. The forecast at a target h half-hours after the
      band's last observed reading is the DP mean + ℓ_b + φ^h·z_b.
    - **Information cut.** At initialisation τ (09:00 UTC daily) the filter sees only residuals
      whose half-hour ends at least 3.5 hours before τ, the telemetry delay of experiments 87 and
      104 (`e00087_common.TELEMETRY_DELAY` plus the half-hour a reading covers). The 3.5 hours is an
      assumption, not a measured value, and goes on `data-processing.md` as one. The sweep is 0, 9,
      and 24 hours.
    - **No switching masks in the filter**, as the series plan rules. The masks apply only to
      fitting the DP and to scoring.
    - **Hyperparameters.** log q, logit φ, log σ_z, and log σ_ε per site, under a shared normal
      prior across sites, fitted by marginal likelihood on 2023-24. Each base arm gets its own set,
      fitted on its own residuals from its 2023-07-01 fit: A0's for A2, A1's for A3, A1s's for
      A3s, W's for WK, and X-best's for X-best+K.
    - **Across a refit,** the filter carries its state forward on each fit's out-of-sample
      residuals and switches to the new fit's residuals at c. It never re-runs on a new fit's
      in-sample residuals, which would reset ℓ towards zero.
    - **EWMA.** Per site and band, an exponentially weighted mean of the raw residuals with a 14-day
      half-life (roughly BG2's 14-day correction), held flat over every lead, under the same cut.
    - **Spread for corrected arms.** The arm's replica runs through the same filter over its
      following 12 months. The quantiles are the corrected mean plus the replica's post-filter
      residual quantiles per hour band, calendar quarter, and lead band, centred on their median.
- **Tests:** (1) the filter matches a NumPy Kalman reference to 1e-6; (2) changing any residual
  after the information cut leaves the forecast at τ unchanged exactly, for the Kalman filter, the
  EWMA, and XGBoost with the filter; (3) a batched refit copy matches a genuinely separate fit built
  through the real cut path (`build_pooled_batch` at c, or `dp_gam`'s own window), never a second
  row-weighted copy, within a tolerance set from the measured seed-to-seed floor (or in float64 on
  the CPU); (4) the pooled D1 on one synthetic site with the network off matches the convex D1
  path within 0.2 points; (5) leakage asserts as in Parts 2 and 3, including that no copy has a
  knot after its cut.

## Years, sites, and rows

**Three years, each with its own frozen cut, so that every Part 4 choice is made on a year that is
never scored:**

| Year | Cut (training ends) | Role |
|---|---|---|
| 2023-07-01 to 2024-06-30 | 2023-07-01 | tuning: drift step, step penalty, final knot, filter hyperparameters, W, XGBoost bar |
| 2024-07-01 to 2025-06-30 | 2024-07-01 | replicate: every decision contrast, out of sample for Part 4's choices |
| 2025-07-01 to 2026-06-30 | 2025-07-01 | scored: the verdict |

**What the tuning year can and cannot tune.** At the 2023-07-01 cut, all years with lockdown out is
only about 2 to 2.5 years at the long sites, almost the 2-year window, so 2023-24 tunes the knot,
the penalties, and the filter, but nothing about years. The replicate year is in-sample for the
settings inherited from Part 3 (K, the deviation penalty) and, slightly, for the filter-arm spread
(its replica runs over 2023-24, the filter's fitting year). The write-up says both and reports the
tuned settings' replicate-year scores without re-choosing.

- **Training windows.** All years: every row from each site's first usable date (D11 2019-09-17,
  D10 2019-12-01, D5 2020-02-01, D14 2020-07-01, D16 2021-02-01, D9 2023-10-01) to the cut,
  `default_rows == True`, the lockdown window 2020-03-23 to 2021-07-19 out. "From 2019-09" buys 3.5
  to 4.5 extra site-years at the long sites, not 6.
- **Pooled training set (if M = P), switching masks, and decision sites** as Part 3: decision
  sites D5, D10, D11, D14, and D16 (Phase A needs 4 of 5); D9 reported beside them (it has 9 months
  at the 2024-07 cut, so its replicate-year C1 is zero by construction); D18 descriptive.
- **Initialisations and leads.** One initialisation per day at 09:00 UTC, targets for the next 7
  days; the headline is day D+1 (leads 15.5 to 39 hours), and every assimilation contrast is also
  reported per day of lead.
- **Predictions written:** `time_series_id`, `valid_time`, `power_fcst_init_time`, `q01`, `q05`
  to `q95`, `q99` (all 11 levels), `actual`, `lead_hours`, `arm`, `seed`, and, for filter arms,
  `level` and `anomaly`, so the gain can be split between ℓ and z.
- **Robustness rows, for every contrast:** November to March against April to October; without
  D16; D10 with its two unlabelled switching windows dropped from scoring (2025-11-17 to 2025-12-14
  scored, 2024-11-08 to 2024-12-01 replicate); and the all-rows score beside the masked headline.

## Arms

**Every arm is screening tier unless it says otherwise:** one frozen fit per cut, one seed, on the
GPU for M = P and the CPU for G, about 10 minutes per arm. The filter and EWMA run on a frozen fit's
residuals in seconds, so they add arms, not fits.

| Arm | Drift | Assimilation | Notes |
|---|---|---|---|
| A0 | none | none | |
| A1 | D1 | none | the default drift |
| A1s | D1 + steps | none | paired with A1 |
| A2 | none | Kalman | |
| A3 | D1 | Kalman | the assimilation decision arm |
| A3s | D1 + steps | Kalman | paired with A3 |
| A4 | D1 | EWMA | the simple fix, on A3's base |
| A4-0 | none | EWMA | the series plan's baseline, on A2's base (exploratory) |
| A3-ℓ, A3-z | D1 | Kalman, one state | which state carries the gain (exploratory) |
| A1-fc, A3-fc | D1 | none, Kalman | capacity fitted (labelled variant) |
| W | chosen | none | M's best window: {12 months, 2 years} × {no drift, D1}, chosen on 2023-24 |
| WK | as W | Kalman | the best window with the filter (exploratory) |
| A1-cov | D1 | none | lockdown rows kept, COVID flag on (exploratory) |

**The final knot of A1 and A3** is `free`, `shrunk` (λ on {0.25, 0.5, 0.75}), or `pinned`, chosen
on 2023-24 by quantile score; a variant other than `free` is adopted only if it beats `free` by at
least 0.05 points there. **Refit cadence:** A1 and A3 frozen, 4-weekly, and weekly over both
replicate and scored years, on G always; on M, 4-weekly first, weekly only if the module PR
measures all-years B = 12 under 10 GB and the first weekly batch runs in under 12 minutes. **D10's
November 2024 jump:** A1 and A1s refitted at the 2025-07-01 cut with D10's rows 2024-11-08 to
2024-12-01 masked (one batched copy each).

**XGBoost bar (CPU, lagless, experiment 87's Primary hyperparameters without the three power lags,
11 quantiles, Part 2's ERA5 and CAMS inputs).** X-best is the default-start arm on the window (12
months, 2 years, or all years) that scores best on 2023-24, the window chosen with the plain arm
because the offset adds nothing when CAMS is an input. Beside it run X-bestz (zero start),
X-bestamq (quantile start plus experiment 64's solar offset), and X-bestam (zero plus offset), the
offset built as in `e00104_xgb.py` with each site's weight w reported. The C6 bar is the 2023-24
winner of X-best and X-bestamq. Stored sets cover part of the scored year (Part 2's `x_12*` and
`x_all*`, Part 3's `x_2y*`); the module PR lists the missing combinations (at least the
quantile-start sets on 12 months and all years) and fits them on the CPU.

- **XGBoost with the filter (X-best+K, exploratory)**, so a gain of A3 over lagless XGBoost is not
  credited to the DP when the filter alone gives it.
- **Lagged XGBoost (descriptive)** on X-best's window at leads of 24.5 to 48 and 144.5 to 168
  hours, with the same 3.5-hour telemetry delay as the filter.
- **References:** Part 2's stored G-all+D1 and G1; climatology; `manual_heuristic`.

**Tuning (on 2023-24, fitted at the 2023-07-01 cut, by quantile score):** the drift step on {300,
1,000, 3,000, 10,000, 30,000} (Part 2's choice sat at the grid edge); the step penalty on {30, 100,
300, 1,000}; the final knot and λ; W; the XGBoost window. Every candidate is scored with A1's replica
spread (one replica fit at a 2022-07-01 cut), not a replica of its own. The filters' hyperparameters
are fitted by marginal likelihood. D9 has no history at that cut and is not in the tuning score.

## Known-answer gate (blocking, after tuning, before any replicate-year or scored-year fit)

**A failed blocking test stops scoring until it is fixed.** If it cannot be fixed inside the part,
the verdict is `inconclusive`. Each test runs with the interaction network in its decision-arm
setting, because a test with the network off cannot see the solar route the Part 3 review found.

1. **Synthetic drift recovery.** Eight synthetic sites from real calendar and weather tensors over
   2019-09 to 2025-06, with known drift paths per band: a ±10-point ramp at four sites, one +8 point
   step on a known date at two sites, a 3-point rise in the last 3 months before the cut at one
   site, and no drift at one. Pass if A1's path is within 1.5 points (root mean square) of the truth
   at every site; the held value at the cut is within 1.5 points at the late-rise site under the
   chosen final knot; A1s places each true step within one knot and 25% of its size; and the
   no-drift site's steps sum to under 1 point in absolute value.
2. **The filter tracks a seasonal misfit causally and does not invent one.** (a) Synthetic,
   blocking: into white-noise synthetic residuals inject a known 3-point annual cycle (the residual a
   heating slope 30% too low leaves). Pass if ℓ follows the injected cycle with its amplitude within
   25%, lags it by at most 7 days, and never leads it (a lead would mean the filter sees the
   future); and on the same residuals without the cycle, ℓ's annual harmonic is under 0.5 points.
   (b) Real, a reported diagnostic: on 2023-24, the annual harmonic of A2's ℓ against M's own
   heating term, and ℓ's correlation with the 7-day mean temperature, per site. No retuning follows
   from (b): part of a winter ℓ is expected, since D14's winter gain is a heating misfit ℓ should
   carry.
3. **Injected transfers.** Inject transfers with `e00043_inject.py` (non-circular shapes) into
   2023-24 at D10, D11, D14, and D16, only where no real hand-filtered transfer lies within 48 hours,
   and run A3's filter. Pass if ℓ + z carries 50% of the moved MW by the end of day 2, carries 0.6 to
   1.3 times the moved MW from day 2 to the end (experiment 88's detector carried 1.5 times), and
   falls under 25% within 3 days of the end. The filter on the 66 real transfers is reported beside
   it, with transfers that have no metered receiver counted apart.
4. **The drift does not take the solar term.** Experiment 39's injected-farm harness, one farm
   switched on mid-window at D10, D11, D14, and D16 (batched), fitted with pinned A1. With capacity
   pinned the midday drift band should absorb the farm; blocking pass if the drift outside the midday
   bands moves by less than 0.5 points at every injected site. Reported, not blocking: the shared
   modules' midday movement (the network's route), and A1-fc's recovered-to-injected capacity
   ratio against [0.75, 1.25]. Only ratios appear: the farms are metered generators.
5. **Module tests 1 to 5** pass.

## Planned contrasts

Six planned contrasts, Bonferroni-corrected (99.17% intervals from `studies.bootstrap`, resampling
whole calendar months of stored predictions, paired across arms), at day D+1, each with site counts,
every arm's absolute score, the mean-forecast error, the replicate year's estimate and count, and
the robustness rows:

- **C1, long history (the series criterion):** A1 minus W. Reported beside it: A0 minus W, the
  years-without-drift contrast.
- **C2, assimilation on top of drift:** A3 minus A1.
- **C3, the Kalman structure against the simple fix:** A3 minus A4.
- **C4, is drift still needed once the filter runs:** A2 minus A3, non-inferiority, margin 0.10.
- **C5, does the filter replace refitting:** A3 frozen minus A1 weekly on G, non-inferiority,
  margin 0.10 (on M it is exploratory).
- **C6, the DP against lagless XGBoost:** A1 minus the C6 bar.

**C3, C4, and C5 cannot be decided at this noise level.** One year's screening interval is about
±0.4 points at 99.17%, against a predicted C3 of −0.02 to −0.15 and a margin of 0.10 for C4 and C5.
Each of the three is therefore reported descriptively, with no verdict word, wherever its interval
cannot exclude zero (C3) or the margin (C4, C5). The choice each hands to Part 6 follows the
pre-registered rule in "Decision rule", labelled a provisional choice, not a finding.

**Exploratory at 95%,** grouped:

- **Steps and knots:** A1s minus A1 and A3s minus A3, both also with D10's jump masked; each
  final-knot variant minus `free`, by month since the cut.
- **Filter:** A2 minus A0, A4-0 minus A0, A2 minus A4-0; A3-ℓ and A3-z minus A3; WK minus A3; the
  refit cadence curves; C5 on M; the delay sweep; every assimilation contrast per day of lead.
- **XGBoost:** A3 minus X-best+K; A3 minus lagged XGBoost at 24.5 to 48 hours; X-bestz minus
  X-best and X-bestam minus X-bestz; A1 minus Part 2's G-all+D1 and G1 (scored year only).
- **Descriptive rows (no interval where scarce):** A1-cov; the `-fc` variants; D9; D18; the
  coastal sites D5 and D9; bank holidays, the Christmas run, and days below 0 °C.

## Predictions

Written before any fit; each line can fail, and none of them overrides the decision rule.

- **C1:** A1 minus W is −0.05 to −0.30 in the scored year, with the replicate the same sign; the
  November-to-March part is more negative than the April-to-October part at 4 of 5 sites; dropping
  D16 moves C1 by under 0.10.
- **C2:** A3 minus A1 is −0.3 to −0.8 at day D+1, interval wholly below zero, 5 of 5 sites better
  in both years, and under a third of that gain is left at day D+7.
- **C3:** A3 minus A4 is −0.02 to −0.15. **C4:** A2 minus A3 lies within ±0.10. **C5:** A3 frozen
  minus A1 weekly is below zero.
- **C6:** A1 minus the C6 bar is −0.1 to −0.5 (Part 2's G-all+D1 minus X-all was −0.37).
- **States:** ℓ carries over 70% of C2's day-D+1 gain, and z's share falls under 10% by day D+3.
- **Steps out of sample:** A1s minus A1 lies within ±0.10, and masking D10's November 2024 jump
  halves D10's part of any steps loss.
- **Final knot:** the variant chosen on 2023-24 beats `free` in months 0 to 4 after the cut.
- **Refit cadence:** A1 frozen minus A1 weekly is +0.1 to +0.4; A3 frozen minus A3 weekly is under
  half of that.
- **Tails:** A3's q95 coverage is within 3 points of nominal and nearer nominal than A1's in
  November to February; q01 and q99 within 1.5 points.
- **Against lagged XGBoost:** A3 is within 0.2 points at leads of 24.5 to 48 hours.
- **Run time:** every screening arm under 10 minutes; screening under 2 hours of wall clock; peak GPU
  memory under 10 GB.

## Decision rule

**The verdict is C1's label under the series' Phase A rule, with a replicate check.** C2 to C6 are
labelled in the bullets and reported in the take-home, not in the `verdict` field.

- **`supported`:** C1's estimate is below zero, at least 4 of the 5 decision sites improve, the
  replicate year's estimate is also below zero, and the gate passed.
- **`partly supported`:** C1's estimate is below zero but fewer than 4 sites improve, or the
  replicate year reverses the sign.
- **`refuted`:** C1's interval lies wholly above zero.
- **`inconclusive`:** anything else, or a gate test that cannot be fixed inside the part.

**Every choice handed to Part 6 needs the same sign in both years:**

- **The filter goes to Part 6** if C2 is below zero in both years at 4 or more of 5 sites in both;
  the joint training of filter and DP (the series plan's stage two) is then the confirmation-tier
  step.
- **Kalman or EWMA:** the Kalman filter is kept only if C3 is below zero in both years at 4 or more
  of 5 sites in both; otherwise the EWMA, the simpler correction, goes to Part 6.
- **C4 and C5** pass only with a point estimate of at most +0.10 in both years, and a pass is
  labelled "not shown worse", never "drift not needed" or "refits not needed", until the
  confirmation tier. A pass of C4 lets Part 6 run the filter without drift; a pass of C5 is reported
  as an option for a slower refit schedule, not a commitment.
- **The criterion's scope.** A `supported` C1 on M = P meets the series' long-history criterion;
  on M = G it meets it only for the convex model, and the write-up says so.

## Run order, compute, and time

**Order:** (1) the module PR (pooled drift if M = P, pinned capacity with the growth-prior number
fixed, the COVID flag, final-knot options, per-cut batched refits, `studies.dp_assimilation`, tests
1 to 5), about 6 hours of agent time, whose convex and filter parts start while Part 3 runs; (2)
the model rule, with M recorded in `decisions.json`; (3) tuning at the 2023-07-01 cut, about 30
minutes; (4) the gate, about 20 minutes; (5) the grid at the 2024-07-01 and 2025-07-01 cuts; (6)
the XGBoost fits and references on the CPU, alongside; (7) refit cadence, G first, then M 4-weekly,
then M weekly if the measurements allow; (8) the delay sweep and the exploratory filter arms.

**Fits are reused as replicas.** The replica of each 2025-07-01 fit is the matching 2024-07-01 fit,
and the replica of each 2024-07-01 fit is the matching 2023-07-01 tuning fit, which halves the
grid's fits. Per cut that leaves A0, A1, A1s, A1-fc, W, the `pinned`-knot A1, and at the 2025-07-01
cut the 2 D10-masked copies: 7 or 8 fits, about 1 hour of GPU at B = 12, or minutes on the CPU if M
= G. The refit cadence adds about 1 hour (approved). Report, figures, cards, and write-up take about
4 hours plus the numbers review, any audit, and the prose sweep: about a day and a half in all from
approval to a merge-ready PR.

**The GPU share is up to 10 GB. If the 2-hour budget is at risk, cut in this order:** the `-fc`
variants, the `pinned`-knot fits, A1-cov (recorded as deferred to the confirmation tier), then M's
weekly refits. The filter-only arms (A2 to A4-0, A3-ℓ, A3-z, WK, X-best+K, the delay sweep) take
seconds and are never cut.

## Data paths

As Part 3's plan, plus Part 2's `work/tuned.json` and centred stored predictions, Part 3's
handed-on configuration and `x_2y*` predictions, and `e00043_inject.py` for the transfers.

## What the write-up must contain (beyond the series plan's standing rules)

Title "Differentiable physics, part 4: all years with drift, and recent-level assimilation"; front
matter as above (theme `differentiable-physics-long-history-and-drift`, `also_relevant_to`, a
two-or-three-sentence `take_home_message`, `part: 4`). The take-home says in plain words whether
more years help once the model has a drift term, in both years and both seasons; how much feeding
the model the latest observed demand gains, and for how many days ahead; and whether that makes
drift or refitting unnecessary. The intro says which model (P or G) the grid ran on and why.

- **Figures:** a winter and a summer week per decision site, actual against A1, A3, and W; the
  headline chart of C1 to C6 with intervals, site counts, and the replicate year beside each; every
  arm's absolute score per site; A3's gain over A1 by day of lead, split into ℓ and z; the drift
  paths with the held value and each final-knot variant; the refit cadence curves; the gate; the
  tails (coverage and pinball at q01, q05, q95, and q99, every arm); the score by month since the cut.
- **Substation cards** (D-labels only, labelled provisional, from A3's fit at the 2025-07-01 cut):
  the series plan's panels, with capacity per season-year shown as three values (the convex
  model's fit, the pooled fit A3-fc, and the pinned vintage) under the caveat that the pooled
  capacity is not identified by the data, plus the filter's ℓ and z over the scored year. One
  cross-substation figure compares capacity, heating slope, drift range, and the filter's level
  step.
- **Checks:** median minus mean forecast per site and arm; the mean-forecast error beside each
  quantile-score contrast; leaving one site out for C1 to C6; which of the `ml-experiment` skill's
  seven audit triggers fired (trigger 7 by construction, since Part 6 depends on this result).
- **Caveats, Part 4's own:** the filter's residuals use ERA5 temperature and CAMS irradiance up to
  the information cut, which a live service would not have, so A3 and A4 are a ceiling, flattered
  more than A1; on ENS in Part 6 the transient state also absorbs weather forecast error; the 3.5-hour
  delay is assumed; the 5-week vintage leak at the tuning cut; the replicate year's in-sample
  settings; M's choice reads Part 3's scored year.

## Decisions recorded, and departures from the series plan

**Decisions (manager, 2026-10-19):** the telemetry delay is 3.5 hours; Part 4 tunes on 2023-24;
the fitted-capacity variant inherits Part 3's fix if settled and passed, else starts at the pinned
vintage; the extra GPU hour for refit cadence is approved, with C5 always on the convex model and
pooled refits 4-weekly first, weekly only if measured memory and time allow.

**Departures from the series plan, each deliberate:**

- **The brief's A4 sits on no drift;** here A4 sits on D1, so C3 compares the two corrections on
  one base, and the brief's version is A4-0.
- **"The same model on its best window"** is W, chosen with or without D1, so C1 isolates years
  whenever W keeps D1.
- **The brief widens the quantiles with the filter's own variance;** here the spread comes from the
  replica's post-filter residuals, following the series rule that a spread is fitted on cross-fitted
  residuals.
- **D2 (a shared slow factor) is dropped:** new module code for an exploratory arm.

**Open question left:** none; the growth prior's number is fixed in the module PR before any fit.
