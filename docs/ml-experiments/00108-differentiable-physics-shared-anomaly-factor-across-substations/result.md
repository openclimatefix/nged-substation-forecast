---
id: "00108"
slug: differentiable-physics-shared-anomaly-factor-across-substations
theme: lagged-power-and-error-correction
also_relevant_to: [differentiable-physics-long-history-and-drift]
status: done
size: >-
  medium: a small module PR (a spread over a correction summed from several traces, with its leakage
  test) plus experiment scripts
hypothesis: >-
  The residuals of the frozen convex demand model carry a component shared across Primaries, and a
  shared state estimated from the other substations' residuals, available at the forecast origin,
  lowers the day-ahead quantile score by more than the per-site Kalman filter of Part 4 already does.
prediction: see "Predictions"
decision_rule: see "Decision rule"
benchmark: >-
  Part 4's A2 (convex model on all years, no drift, pinned capacity, per-site Kalman filter), rebuilt
  from its stored configuration; Part 4's A0 (the same model uncorrected); a placebo factor from the
  contributor panel shifted by whole weeks, refitted per draw; climatology and manual_heuristic as
  stored by Part 4
row_set_id: >-
  dp_mw_v1 (experiment 103); scored 2025-07-01 to 2026-06-30, replicate 2024-07-01 to 2025-06-30,
  tuning 2023-07-01 to 2024-06-30
metric: >-
  Quantile score (twice the mean pinball loss over experiment 87's 11 quantiles) in points of
  effective capacity, on rows outside the hand-filtered switching events at the scored site, at day
  D+1 from a 09:00 UTC initialisation on day D (leads of 15.5 to 39 hours)
verdict: inconclusive
production_status: not-for-production
adopted_in: none
take_home_message: >-
  The experiment stopped at its known-answer test, so it cannot say whether a shared factor
  improves the forecast. Over the injected episodes the shared filter, tuned on 2023-24, carried
  under 5% of the injected points (the plan asked for 40% or more), still held 35% of its level
  after three full days without the anomaly, and corrected no better than the per-site filter
  (2.402 against 2.397 points of mean absolute error over days 2 and 3 of the episodes). Part of
  the test was mis-specified, so a corrected test is the next step before any real score.
date: 2026-10-19
pr: 1384
advice:
  - >-
    Judge a shared filter's carry against what a day-ahead forecast can see, not against the full
    injected points: the loading ceiling is 0.458, just above the 40% limit, and this run carried
    10% of a perfect tracker even on episode days (gate test 2).
  - >-
    Inject episodes of three or more days, or score only the third day onward, because a 09:00 UTC
    origin sees the first day's night band and, in summer time, one morning half-hour (gate test 2).
  - >-
    Expect a shared state tuned by marginal likelihood on 2023-24 to decay by about 30% per day
    after a step, so a limit of 25% after three days cannot pass; set the limit from the idealised
    response (figure 2) before running the gate (gate test 3).
  - >-
    Do not re-run a failed gate on the same injections after changing the test; write the revised
    test down first and use new injections (gate rule).
code: ml-experiments/00108-differentiable-physics-shared-anomaly-factor-across-substations/
---

# Differentiable physics: a shared anomaly factor across substations

**The experiment stopped at its pre-registered known-answer gate, so it did not test whether a
shared anomaly factor improves the day-ahead forecast.** The gate injected 20 anomalies of known
size into the tuning year (2023-07 to 2024-06). On the 12 common episodes the shared filter, tuned
on that year, carried 4.7% of the injected points (the plan asked for 40 to 120%). The shared part
peaks two days after an episode ends because of the day-ahead lag, and 3 days after the end it was
still at 74 to 75% of that peak (the plan asked for under 25%). After three full days without the
anomaly, the tuned filter still holds 35% of its level. The shared filter also corrected injected
anomalies no better than the per-site filter of experiment 106: 2.402 against 2.397 points of mean
absolute error, measured over days 2 and 3 of the episodes. No arm was scored on the replicate or
scored years, and no contrast was computed. The verdict is `inconclusive`.

**Part of the gate was mis-specified, and part of the failure is real.** A 09:00 UTC day-ahead
forecast sees almost none of a two-day episode, so the carry test could hardly pass: a contemporaneous
perfect tracker would carry only 46%, because the loadings cap it near 0.45. The release failure is
real: the tuned shared filter decays by about 30% per day. The corrected criteria below are
recommendations from this result, not results, and this run fails the corrected carry criterion
too (10% of a perfect tracker).

## Figures

![Share of the shared part's peak still present 3 days after each of the 12 common episodes ended.
Every episode is far above the 25% limit; the peak itself falls two days after an episode ends.](figure_01_gate_release.svg)

![The tuned shared filter's state after a noise-free three-day step of the factor. The step is on
days 0 to 2. The
state reaches 1 at the first reading and falls to under 25% on the 7th day after the step ends. The
figure uses synthetic 6-hour bands, not the real 6, 5, 5, and 8-hour bands.](figure_02_shared_response.svg)

![The 20 largest shared-anomaly days of the tuning year, by band mean of the all-site median of
scaled residuals.](figure_03_largest_days.svg)

## Gate results

Tuning was frozen before the gate. Every tuning input is restricted to the tuning year and the
filters only look backwards. The 2024 and 2025 fits and the reproduction step read later years, but
no new arm was scored on them. All numbers are in [report.txt](report.txt), printed by
`e00108_report.py`.

| Test | Result | Limit | Verdict |
|---|---|---|---|
| 1. Recovery slope of the injected change in the all-site median | 1.10 | 0.8 to 1.2 | pass |
| 2. Shared part's carry of the injected points on episode days from day 2 | 4.7% | 40 to 120% | fail |
| 3. Shared part 3 days after an episode ends, share of its peak (peak is 2 days after the end) | 74 to 75% (worst 0.754) | under 25% | fail |
| 4. Mean absolute correction error, KF against A2, days 2 and 3 of the episodes | 2.402 against 2.397 | KF below A2 | fail |
| 5. No foresight: forecasts issued before an evening episode | bit-identical, 218,160 cases | identical | pass |
| 6. Locality: a local episode's effect on other sites | own site 0.0, others 0.043 points | 0 and under 0.3 | pass |
| 7a. Nesting: KF with beta 0 and the `a0` filters against A2 | gap 0.0 | 1e-6 | pass |

- **Reproduction passed.** A0 and A2 reproduce experiment 106's stored sets (`replicate` and
  `scored`, days 1 to 7) with a maximum difference of 0.0 points, and the refitted `a0` filter
  hyperparameters equal `tuned.json` exactly.
- **Episodes:** 8 common multi-day, 4 common midday, 4 common one-evening, and 4 local, placed with
  seed 108 at least 48 hours from every hand-filtered switching row and at least 10 days apart.
- **A window bug was fixed after the first print.** Test 2 and test 4 first counted the day after
  each episode as an episode day, which printed 18% for test 2 (and 39% of a perfect tracker) and
  1.999 against 1.994 for test 4. The thresholds, the tuning, and the injections did not change.
  Tests 1, 3, 5, 6, and 7a use their own windows and are unchanged.
- **Test 2 by day:** the shared part carried 3% of the injected points on an episode's second day
  and 16% on its third. Against a perfect tracker of the factor (loading times scale times the
  injected scaled change), it carried 10%.
- **Test 3 by filter:** the tuned shared filter has `q` 0.0207 per day, `phi` 0.818 per half-hour,
  `sigma_z` 0.401, and `sigma_eps` 0.134, in units of each site's residual scale. Its noise-free
  response to a three-day unit step (days 0 to 2) is 1.00 on days 2 to 4 and then 0.68, 0.48, 0.35,
  0.26, 0.19. It falls under 25% on the 7th day after the step ends, and holds 35% after three full
  days without the step. The 74 to 75% in the gate is mostly lag: the shared part peaks two days
  after an episode ends, so 3 days after the end is one day past the peak.
- **Frozen loadings** (Huber slope, shrunk halfway to the mean): D5 0.50, D10 0.40, D11 0.34, D14
  0.44, D16 0.58; D9 has no tuning-year fit and takes the mean, 0.45.

## Design points

- **The origin sees little of a short episode.** A 09:00 UTC initialisation reads residuals to
  05:30 UTC, so the forecast for an episode's second day sees the first day's night band and, in
  summer time, one morning half-hour, and nothing in the other bands. Seven of the 12 common
  episodes fall in summer time. Test 2 counted all four bands from the second day.
- **The carry limit ignored what a day-ahead forecast can see.** Six of the eight multi-day
  episodes last 2 days, so the forecasts for them see almost none of the episode. The loadings also
  cap a perfect tracker at 0.458, just above the 40% limit.
- **Recommended corrections, not run:** judge carry against the loading-scaled injected change (40 to
  120% of a perfect tracker), use episodes of three or more days or score from the third day, and set
  the release limit from the idealised response before running. This run fails the loading-scaled
  carry criterion too (10% of a perfect tracker). A re-run needs new injections.

## The 20 largest shared-anomaly days (descriptive, no verdict)

The list is ranked from the tuning year's residual streams alone (the largest absolute band mean
of the all-site median of scaled residuals, five sites on every day), and was fixed before any event
was matched. The temperature anomaly is the day's mean ERA5 temperature at D5 minus its trailing
30-day mean. The table is in [report.txt](report.txt).

- **Cold spell:** 9 of the 20 days fall in January 2024, and 4 of them (8, 13, 14, and 15 January) are
  3 °C or more below the 30-day mean, with 15 January at −7.5 °C. Two more days (22 September 2023
  and 21 October 2023) also meet the 3 °C criterion, so 6 of 20 do. A news report of a Met Office
  reading puts the lowest January temperature since 2019 on 16 January 2024, at Dalwhinnie in
  Scotland, not in the trial area
  ([LBC](https://www.lbc.co.uk/news/exact-location-met-office-weather-alert-snow)); I did not find
  a source for 8 to 15 January itself.
- **Bank holidays and the Christmas run:** none of the 20 days.
- **Football:** 2 of the 20. England's home matches against Malta on 17 November 2023 and against
  Iceland on 7 June 2024, both kicking off at 19:45 at Wembley, fall on listed Friday evenings
  (−2.14 and −1.93, evening band)
  ([England Football](https://www.englandfootball.com/england/mens-senior-team/fixtures-results/2023-24/england-mens-malta-uefa-euro-qualifier-wembley-friday-17-november-2023),
  [FotMob](https://www.fotmob.com/en-GB/matches/england-vs-iceland/2ebcim)). England's four Euro
  2024 matches inside the tuning year (16, 20, 25, and 30 June) do not fall on listed days
  ([Sky Sports](https://skysports.com/football/news/33161/13020967/englands-euro-2024-fixtures-and-route)).
- **Saharan dust:** none on the listed dates. Reports place dust episodes around 6 September 2023
  ([The Star](https://www.thestar.co.uk/read-this/september-heatwave-windows-and-cars-left-dusty-as-sahara-sand-blows-across-uk-amid-high-temperatures-4284844)),
  8 to 11 October 2023 (a Shropshire Star report whose link no longer resolves), and 26 to 28
  January 2024
  ([NASA Earth Observatory](https://science.nasa.gov/earth/earth-observatory/saharan-dust-blows-toward-europe-152384)).
  The nearest listed days are 25 and 29 January, 1 and 22 September, and 21 October. None coincides.
- **Sign:** 17 of the 20 band means are negative, which means demand below the frozen model's mean.

## Checks

The audit triggers are checked against what ran. No forecast beyond the gate and the reproduction
was scored, so there is no median-minus-mean forecast, no error of the mean forecast beside a
contrast, and no leave-one-site-out contrast to report.

- **Trigger 1 fired:** three gate tests failed (2, 3, and 4). An Opus numbers review and audit
  (private brief `data/ml-experiments/briefs/dp_pr1384_numbers_audit_result.md`) reproduced every
  printed number, found the window bug above, and found wording errors; all were accepted and
  fixed on this page, with none rejected.
- **Trigger 2:** the rebuilt A2 agrees with experiment 106 to 0.0 points. The failures in tests 2 to
  4 were not predicted in the plan, which expected the gate to pass. Tests 2 and 3 are explained
  above. Test 4's failure has a visibility explanation: over days 2 and 3 of the episodes, the
  day-ahead forecasts have seen at most the first day of an episode and the shared part carries
  under 5% of the injected points, so the test says little about whether a shared state adds value.
- **Triggers 3 to 7:** not applicable, because no contrast was computed.

## What was not run

- **Not run, because a failed gate test stops scoring:** the power check, the 13 refitted placebo
  draws, the arms F, KF, KF-frz, KF-mva, KF-9h, KF-hind, the contrasts C1 to C3, and the replicate-year and scored-year scores. The code for the
  arms and the gate is in `ml-experiments/00108-*/`.
- **Known limit of the stale-contributor arm:** in KF-9h, a local stream built as the residual
  minus the loading times the scale times the factor reads factor values that are not yet usable
  at the local 3.5-hour cutoff, which changed 14 of 100 day-1 corrections in a module check. The
  module cannot detect this, and the arm's script would have to delay the factor inside the local
  stream too.
- **Caveats:** ERA5 and CAMS inputs up to the cut would make any result a ceiling; the 3.5-hour delay
  is assumed; the gate injected into the five MW sites in one small area; and the 2021 and 2022
  fits read the 2023-08-07 capacity vintage 13 to 25 months late.

## What to try next

- Re-register the gate with the corrected criteria above, on new injections, and run it before any
  score. If it passes with a shared filter tuned for persistence, run the plan's contrasts.
- Otherwise, record that this shared-filter design could not be shown to work, without concluding
  that a shared state adds nothing.
