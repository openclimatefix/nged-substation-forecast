--- id: "00108" slug: differentiable-physics-shared-anomaly-factor-across-substations title:
"Differentiable physics: a shared anomaly factor across substations" theme:
lagged-power-and-error-correction also_relevant_to: [differentiable-physics-long-history-and-drift]
status: planned size: >- medium: a small module PR (a spread over a correction summed from several
traces, with its leakage test) plus experiment scripts hypothesis: >- The residuals of the frozen
convex demand model carry a component shared across Primaries, and a shared state estimated from the
other substations' residuals, available at the forecast origin, lowers the day-ahead quantile score
by more than the per-site Kalman filter of Part 4 already does. prediction: see "Predictions"
decision_rule: see "Decision rule" benchmark: >- Part 4's A2 (convex model on all years, no drift,
pinned capacity, per-site Kalman filter), rebuilt from its stored configuration; Part 4's A0 (the
same model uncorrected); a placebo factor from the contributor panel shifted by whole weeks,
refitted per draw; climatology and manual_heuristic as stored by Part 4 row_set_id: dp_mw_v1
(experiment 103); scored 2025-07-01 to 2026-06-30, replicate 2024-07-01 to 2025-06-30, tuning
2023-07-01 to 2024-06-30 metric: >- Quantile score (twice the mean pinball loss over experiment 87's
11 quantiles) in points of effective capacity, on rows outside the hand-filtered switching events at
the scored site, at day D+1 from a 09:00 UTC initialisation on day D (leads of 15.5 to 39 hours)
verdict: <supported, partly supported, refuted, or inconclusive> production_status:
not-for-production take_home_message: <written after the results> date: <date> pr: <pr> advice:
<written after the results> code:
ml-experiments/00108-differentiable-physics-shared-anomaly-factor-across-substations/ ---

# Differentiable physics: a shared anomaly factor across substations

## What the experiment asks

**The deciding question is whether a shared state adds anything beyond each site's own Kalman
level.** A shared anomaly lasting a day is already in each site's ℓ, so a shared state can add only
a less noisy estimate of the shared part and a different persistence for it. A 09:00 UTC forecast
sees residuals to 05:30 UTC, so a multi-day cold spell is visible from its second day and a match
evening never is. The sites lie within about 37 km in Lincolnshire, so the factor will mostly carry
a shared weather-response misfit; and CAMS irradiance uses CAMS aerosols, so dust is partly in the
inputs already, unlike on ENS in Part 6.

**Size triggers.** Private stored files only; serving path and degradation rules untouched; more
than one defensible design (one pre-registered); the module change touches
`studies.dp_assimilation` (callers: Part 4's and Part 6's scripts) and merges, after an Opus code
review (leakage first), before any real score.

## Base model, sites, and residual streams

- **Base:** Part 4's A0 (`studies.dp_gam`, all years, lockdown out, no drift, pinned capacity). The
  bar is A2, A0 plus the per-site Kalman filter with Part 4's `a0` hyperparameters. No arm carries
  drift: the Part 4 audit ruled it adds nothing once the filter runs. - **Scored sites:** D5, D10,
  D11, D14, and D16 decide; D9 is reported beside them. - **Contributors (decision arm):** the six
  MW sites, so the composition is fixed across bands and seasons. A site contributes once it has a
  fit with at least 6 months of training rows: each leave-one-out factor has 4 contributors in the
  tuning year (D9 has no fit at 2023-07-01) and 5 later, and is missing where fewer than 3 have a
  reading. - **Contributors (exploratory KF-mva):** the MW sites plus the MVA Primaries D2, D3, D4,
  D8, D12, D19, D21, and D22 on dark rows, fitted with a zero-point `PinnedCapacity` built in the
  script. Its factor is a median of up to 13 at night and 4 or 5 at midday, with a seasonal mix at
  dusk. - **Fits:** every contributor at each 1 July from 2021 to 2025, CPU, float64. The 2024 and
  2025 MW fits must reproduce Part 4's stored A0 and A2 to 1e-4 points, and `a0` refitted from this
  experiment's 2023 residuals must match `tuned.json` to 1e-3, or the experiment stops. -
  **Streams:** r_s(t) = observed − A0 mean, stitched with `stitch_out_of_sample_residuals`, so none
  is in-sample. No switching mask enters a stream, σ, or β; masks apply to fitting and scoring.

## The factor and the decision arm

**The factor is a leave-one-out median of scaled residuals, filtered per hour band by Part 4's
two-state Kalman filter.** Every step uses data from before the cut or the information cut.

1. **Scale:** u_s(t) = r_s(t) / σ_{s,b}, with σ_{s,b} = 1.4826 × the median absolute deviation of
   site s's in-sample residuals in band b over the 12 months before the cut, every row.
2. **Factor:** F_{−s}(t) = the median of u_j(t) over contributors j ≠ s with a reading at t.
3. **Loading:** β_s is the Huber-loss slope of u_s on F_{−s} with a fitted intercept, on every row
   of the tuning year, shrunk halfway to the mean over sites; the intercept stays in e_s.
4. **Split:** e_s(t) = r_s(t) − β_s σ_{s,b} F_{−s}(t).
5. **Two filters:** `run_kalman` on e_s (per-site set) and on F_{−s} (one set for all sites), both
   fitted by marginal likelihood on the tuning year. The local set is reported beside `a0`, since C1
   credits both the factor and that refit; KF-frz keeps the local set at `a0`.
6. **Forecast:** KF = A0 mean + local correction + β_s σ_{s,b} × shared correction, via
   `correction_at` at the 3.5-hour cutoff, for the scored site and every contributor alike.
7. **Spread:** the replica's post-filter residual quantiles per band, quarter, and lead band, as
   Part 4 (replicas: the year before each scored year; 2022-23 for the tuning year).

**With β_s = 0 and the `a0` set, KF is exactly A2** (asserted to 1e-6), so A2 is nested in KF. The
two-step design approximates a joint all-site filter.

**The spread needs a module change.** `post_filter_lead_spread` takes one stream and one trace; KF
sums two traces built from the site's and the contributors' replica residuals. The module PR
generalises it to a correction summed over several traces, with a test that perturbing any
residual, at the site or a contributor, after each case's cutoff leaves forecast and spread
unchanged exactly. Part 4's code review A found a leak-shaped mutant the tests missed in this path.

## Arms

Every arm corrects the same frozen A0 fits and runs in seconds.

| Arm | Correction | Role |
|---|---|---|
| A0 | none | base, rebuilt; must match Part 4 |
| A2 | per-site Kalman, `a0` set | the bar, rebuilt; must match Part 4 |
| F | shared filter only, on A0 | is there a usable shared component |
| KF | local + shared filters | the decision arm |
| KF-plc-k | KF on the shifted panel, refitted, k = 46 to 58 | placebo control |
| KF-frz | KF with the local set frozen at `a0` | separates the refit from the factor |
| KF-mva | KF with MVA dark-row contributors | exploratory |
| KF-9h | KF with contributors 9 hours stale | unequal telemetry delay (exploratory) |
| KF-hind | A0 + local correction + β_s σ_{s,b} × F_{−s} at the target time | hindsight reference, never a forecast |

Exploratory **F-last** (the batch brief's literal method) adds to A0 β_s σ_{s,b} × the last usable
hour's mean F_{−s}, decayed as exp(−lead / T), T from {6, 12, 24, 48, 96} hours on the tuning year.
Every arm writes `days_ahead` 1 to 7; A0, A2, F, and KF add a same-day row (targets 09:30 to 24:00
UTC on day D); filter arms add `local_level`, `local_anomaly`, `shared_level`, and `shared_anomaly`.

**The placebo shifts the whole contributor panel back by one common k whole weeks, k = 46 to 58 (13
draws),** which keeps season, weekday, and the factor's size, persistence, and cross-site
correlation, and breaks only its alignment with the target, the mechanism C1 claims. **Each draw
refits β and the shared hyperparameters on the tuning year** (local set kept), so a placebo can win
through tuning freedom alone: the honest null for C1.

## Known-answer gate (blocking, after tuning, before any replicate-year or scored-year score)

**Known anomalies go into the tuning year's streams, tuning frozen,** at least 48 hours from any
hand-filtered transfer and 10 days apart, with site loadings λ_s drawn from 0.5 to 1.5 (seeded). A
failed test stops scoring; if it cannot be fixed, the verdict is `inconclusive`.

| Episode | Count | Shape | Analogue |
|---|---|---|---|
| Common, multi-day | 8 | +a·λ_s points in every band for 2 or 3 days, a in {2, 4} | cold spell |
| Common, midday | 4 | +3·λ_s points in the midday band for 2 days | dust episode (less PV) |
| Common, one evening | 4 | +3·λ_s points over 16:00 to 23:59 local on one day | big match |
| Local | 4 | +4 points at one site only for 3 days | a local transfer or load |

1. **Recovery:** the injected change in the all-site median regresses on the median of
   a·λ_s/σ_{s,b} with slope 0.8 to 1.2 (scaled units).
2. **Shared carry:** from an episode's second day, the shared component alone (β_s σ_{s,b} × the
   shared correction's injected change) carries 40% to 120% of the injected points at D+1.
3. **Shared release:** within 3 days of an episode's end it is under 25% of its peak.
4. **The factor adds value:** over days 2 and 3 of the common multi-day and midday episodes, KF's
   mean absolute correction error (injected points minus the correction's injected change) is
   below A2's. This fails if the per-site levels already carry everything.
5. **No foresight:** for the one-evening episodes, every forecast issued before the first injected
   half-hour passes the cut is bit-identical with and without the injection.
6. **Locality:** for a local episode, the injected site's shared correction does not move, and the
   other sites' corrections move by under 0.3 points on average.
7. **Nesting, reproduction, and the module's leakage test** pass.

The factor and A2 on the 66 real hand-filtered transfers are reported, not blocking.

## Power check (before any replicate-year or scored-year score)

**The 0.03-point margin may sit inside the noise.** In Part 4's stored predictions, A3 − A4 (two
corrections on one base) is −0.20 [−0.42, −0.06] scored and −0.20 [−0.35, −0.09] replicate at
98.33%, a half-width of 0.13 to 0.18; KF − A2 should be narrower by an unknown amount. Before
scoring, the script measures on the tuning year the half-width of C1's 98.33% interval (in-sample
for this experiment's tuning, so it measures noise only) and 2.4 × the standard deviation of the 13
placebo draws' KF-plc − A2. **The minimum detectable effect (MDE) is the larger.** If it is at most
0.08 points the scored-year C1 decides; otherwise the pooled 24-month C1 (replicate and scored
years, months resampled across both) decides. **A pooled C1 has no independent replicate, so its
verdict is capped at `inconclusive` unless its interval lies wholly below zero.** The MDE and the
choice go in `decisions.json`, and to the manager, before the first replicate-year score.

## Planned contrasts

Three contrasts at day D+1 on the decision sites, with 98.33% Bonferroni intervals from
`studies.bootstrap` (whole months, paired), site counts, absolute scores, the error of the mean
forecast, the replicate year's estimate, and the pooled 24-month estimate:

- **C1, does a shared factor add beyond the per-site level (the decision):** KF minus A2.
- **C2, is there a usable shared component at all:** F minus A0.
- **C3, does the alignment across sites matter:** KF minus the mean of the 13 placebo arms' scores.

**Robustness rows:** November to March against April to October; without D16; D10's unlabelled
windows dropped (2025-11-17 to 12-14, 2024-11-08 to 12-01); all rows; each decision site left out
of score and contributors. **Exploratory at 95%:** C1 per lead day, same-day, and band; KF-hind,
KF-frz, KF-mva, and KF-9h each minus KF; F-last minus F; KF minus A2 on the 20 largest days.

**Comparison with Part 4's per-site level, descriptive:** the pairwise correlation of A2's stored
ℓ across sites per band; the share of each site's ℓ variance the other sites' median ℓ explains;
and KF's correction split into shared and local parts by season.

## Predictions

Written before any fit; each line can fail, and none overrides the decision rule.

- **C2:** −0.15 to −0.50, interval below zero, 4 or 5 of 5 sites better.
- **C1:** −0.08 to +0.02, because A2's levels already carry most of the persistent shared part; any
  gain sits in days 1 and 2 and November to March; on the same-day row −0.05 to −0.20.
- **KF-hind minus KF:** −0.3 to −0.8 (most shared variance is invisible at the origin).
- **A2's levels:** median pairwise correlation of day-ahead ℓ across sites 0.3 to 0.6.
- **Placebo:** KF beats at least 11 of the 13 draws. **Power:** MDE 0.05 to 0.12 points.
- **KF-frz** within ±0.02 of KF; **KF-mva** within ±0.03; **KF-9h** loses at most half of C1.
- **Tails:** KF's q95 and q99 coverage within 1 point of A2's; on the 20 largest days KF's q99
  pinball below A2's.
- **The 20 largest days:** at least 10 are bank holidays, Christmas-run days, or cold-spell days
  (mean ERA5 temperature 3 °C or more below its 30-day mean); at most 3 match a football match or a
  reported dust episode.

## Decision rule

**Gains are negative. The deciding C1 is the scored-year C1, or the pooled C1 if the power check
says so; "the replicate" is the replicate-year C1, unused when the pooled C1 decides.** The placebo
control passes when KF − A2 is below every refitted draw's KF-plc − A2 and C3's interval lies wholly
below zero. The margin, 0.03 points, is about 1.3% of a day-ahead score of about 2.29 (Part 4's A3,
the arm with drift, unreviewed); below it a gain does not justify a cross-site dependency in
production.

- **`supported`:** the estimate is at or below −0.03, the interval lies wholly below zero, at least
  4 of the 5 decision sites improve, the replicate (if used) has the same sign, and both the gate
  and the placebo control passed. - **`partly supported`:** not `supported`, the gate and the
  placebo control passed, and either the interval lies wholly below zero, or the estimate is at or
  below −0.03 with the replicate (or, when the pooled C1 decides, each year) the same sign. -
  **`refuted`:** the interval lies wholly above zero. - **`inconclusive`:** everything else,
  including any estimate between −0.03 and zero with an interval spanning zero, a failed placebo
  control, or a gate test that cannot be fixed. When several conditions fail, the lowest label any
  of them allows applies.

**C2 and C3 are labelled in the bullets and take-home.** If C2 holds and C1 does not, the take-home
says a shared component exists and the per-site filter already carries it. A `supported` C1 goes to
a later joint all-site filter experiment and Part 6's GAM-K arm as a candidate for a
confirmation-tier replication.

## The 20 largest shared-anomaly days

**The list is computed from the data alone, before any matching, and no event enters any fit.**
Rank the three years' days by the largest absolute band mean of the hindsight all-site median of
scaled residuals; show date, weekday, band, value, contributor count, and the temperature anomaly.
Afterwards, match the list to bank holidays, the Christmas run, cold spells, England's Euro 2024
matches (16, 20, 25, and 30 June and 6, 10, and 14 July 2024), the Women's Euro 2025 final (27
July 2025), the 2026 World Cup (from 11 June 2026), and Saharan dust episodes in Met Office or
national news reports, each with a source URL.

## Run order, compute, and time

**Order:** module PR; scripts; fits and checks; tuning; gate; power check; arms; contrasts;
write-up. **Compute, CPU only, at most 6 cores, no GPU, about 50 minutes of wall clock plus 20 to 40
for the placebo refits:** 70 fits (14 sites × 5 cuts) about 5 minutes on 6 workers; tuning 10; the
13 refitted placebo draws about 20 to 40 as parallel workers; gate 5; arms 10; power check,
bootstrap, and report 15. Agent time: about 2 hours for the module PR, 3 for the scripts, and 2 for
the write-up, plus reviews.

## What the write-up must contain (beyond the series brief's standing rules)

The take-home says plainly whether the substations share an anomaly, how much of it the origin can
see, and whether it adds anything to Part 4's per-site filter, for how many days ahead.

- **Figures:** C1 to C3 with intervals (first); the placebo draws with KF marked; C1 by day of lead
  with KF-hind; the hindsight factor per band with the 20 days marked; the gate; A2's ℓ across
  sites against the factor; the tails (q01, q05, q95, q99, every arm); every arm's score per site.
  Tables: the 20 days; each site's β, σ, and local hyperparameters beside `a0`. Substation cards
  (D-labels, provisional): one added panel per scored site, local against shared level.
- **Checks:** median minus mean forecast per site and arm; the error of the mean beside each
  contrast; leave-one-site-out for C1 to C3; the triggers: (1) a gate test or the placebo control
  fails; (2) an unexplained failed prediction, or a rebuilt A2 that disagrees with Part 4; (3) a
  contrast changes direction between quantile score and error of the mean; (4) leaving one site out
  flips C1's sign or verdict; (5) A2 beats KF by more than C1's half-width (A2 is nested in KF), or
  a refitted placebo beats A2 by more than its own half-width; (6) not applicable, no sister
  contrast on these rows (C1's width is compared descriptively with the placebo spread); (7) fires
  if C1 is `supported`, since a joint all-site filter and Part 6 would build on it.
- **Caveats:** ERA5 and CAMS up to the cut make F and KF a ceiling; on ENS the shared state also
  absorbs regional forecast error; the 3.5-hour delay is assumed and equal; the replicate year's
  replica is the tuning year; the hindsight switching masks; one small area; and two inherited
  capacity-vintage leaks: the 2021 and 2022 fits (warm-up, the tuning replica, placebo panels) read
  the 2023-08-07 vintage 13 to 25 months late, and D9's 2024-07-01 fit reads one 9.5 months late,
  which reaches every decision site's replicate-year factor.
- **Data lessons** go on `docs/ml-experiments/data-processing.md` in the same PR.

**Privacy.** Substations appear by D-label only. D15, D18, and the two "battery or biofuel" series
are never used or shown. The 20-day table and the factor figures are fleet-level.
