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
  The experiment stopped at its known-answer gate, where three of seven tests failed, so it cannot
  say whether a shared factor improves the forecast. On injected anomalies the shared filter, tuned
  on 2023-24, carried under 5% of the injected points (the plan asked for 40% or more), and it
  corrected no better than the per-site filter (2.402 against 2.397 points of mean absolute error
  over days 2 and 3 of the episodes). Part of the test was mis-specified, so a corrected test would
  be needed before any real score.
date: 2026-10-19
pr: 1384
advice:
  - >-
    Judge a shared filter's carry against what a day-ahead forecast can see, not against the full
    injected points: the loading ceiling is 0.458, just above the 40% limit, and this run carried
    10% of a perfect tracker even on episode days (gate test 2).
  - >-
    Inject episodes of 3 or more days, or score only the third day onward, because a 09:00 UTC
    origin sees the first day's night band and, in summer time, one morning half-hour (gate test 2).
  - >-
    The shared filter tuned by marginal likelihood on 2023-24 decays by about 30% per day after a
    step, so a limit of 25% 3 days after an episode cannot pass for that filter; set the limit from
    the idealised response (figure 2) before running the gate (gate test 3).
  - >-
    Do not re-run a failed gate on the same injections after changing the test; write the revised
    test down first and use new injections (gate rule).
code: ml-experiments/00108-differentiable-physics-shared-anomaly-factor-across-substations/
---

# Differentiable physics: a shared anomaly factor across substations

**The experiment stopped at its pre-registered known-answer gate, so it did not test whether a
shared anomaly factor improves the day-ahead forecast.** A known-answer gate is a set of tests,
written before the run, that inject anomalies of known size and check that the filters recover them.
A failed test stops the experiment before any real score. The gate injected 20 anomalies into the
tuning year (2023-07 to 2024-06). On the 12 multi-day and midday episodes the shared Kalman filter,
tuned on that year, had a carry of 4.7%: carry is the share of the injected anomaly that the shared
part reproduces, and the plan asked for 40 to 120%. Release is how fast the shared part falls back
after an anomaly ends. The plan asked for under 25% of the peak 3 days after the end, and the shared
part held 74 to 75%. The shared filter also corrected injected anomalies no better than the per-site
filter of experiment 106: 2.402 against 2.397 points of mean absolute error, measured over days 2
and 3 of the episodes. No arm was scored on the replicate or scored years, and no contrast was
computed. The verdict is `inconclusive`.

**The first print of the gate contained a code bug, which is fixed.** The first print of test 2
(carry 18%) and of test 4 (1.999 against 1.994) counted the day after each episode as an episode
day. The bug was fixed without changing any threshold, tuning value, or injection, and all numbers
on this page are the corrected ones.

**Part of the gate was mis-specified.** A 09:00 UTC day-ahead forecast sees almost none of a 2-day
episode, so a filter could hardly pass the carry test. A perfect tracker, a filter that knew the
injected change exactly and applied each site's loading on the factor (about 0.45), would carry 46%,
only just above the 40% limit. The release limit was also mis-specified: the shared part peaks 2
days after an episode ends because of the day-ahead lag, so 3 days after the end is one day past the
peak, and most of the 74 to 75% is that lag. The slow decay is still real in the noise-free test
below: the tuned shared filter decays by about 30% per day.

**The corrected criteria are recommendations, not results.** This run fails the corrected carry
criterion too, with 10% of a perfect tracker.

## Figures

Day 1 is the first day of an episode, and the step response in figure 2 is numbered the same way.

![Share of the shared part's peak still present 3 days after each of the 12 multi-day and midday
episodes ended. Every episode is far above the 25% limit; the peak itself falls 2 days after an
episode ends.](figure_01_gate_release.svg)

![The tuned shared filter's state after a noise-free 3-day step of the factor on days 1 to 3. The
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
| 4. Mean absolute correction error, shared Kalman filter plus per-site filter (KF) against the per-site filter alone (A2), days 2 and 3 of the episodes | 2.402 against 2.397 | KF below A2 | fail |
| 5. No foresight: forecasts issued before an evening episode | bit-identical, 218,160 cases | identical | pass |
| 6. Locality: a local episode's effect on other sites | own site 0.0, others 0.043 points | 0 and under 0.3 | pass |
| 7a. Nesting: KF with beta 0 and the `a0` filters against A2 | gap 0.0 | 1e-6 | pass |

- **Reproduction passed.** A0 (the frozen demand model with no correction) and A2 reproduce
  experiment 106's stored sets (`replicate` and `scored`, days 1 to 7) with a maximum difference of
  0.0 points, and the refitted `a0` filter hyperparameters equal `tuned.json` exactly.
- **Episodes:** 8 common multi-day, 4 common midday, 4 common one-evening, and 4 local; the carry
  and release tests use the 12 multi-day and midday episodes. They were placed with seed 108, at
  least 48 hours from every hand-filtered switching row and at least 10 days apart.
- **Window bug:** the first print counted the day after each episode as an episode day in tests 2
  and 4, as the second paragraph says. Tests 1, 3, 5, 6, and 7a use their own windows and are
  unchanged.
- **Test 2 by day:** the shared part carried 3% of the injected points on an episode's second day
  and 16% on its third. Against a perfect tracker, it carried 10%.
- **Test 3 by filter:** the tuned shared filter has `q` 0.0207 per day, `phi` 0.818 per half-hour,
  `sigma_z` 0.401, and `sigma_eps` 0.134, in units of each site's residual scale. Its response to a
  noise-free 3-day unit step on days 1 to 3 is 1.00 on days 3 to 5 and then 0.68, 0.48, 0.35, 0.26,
  0.19. That is synthetic, not real data: the filter falls under 25% on the 7th day after the step
  ends, and still holds 35% on the 4th day after it. In the gate, the shared part peaks 2 days after
  an episode ends, so the 74 to 75% at 3 days is mostly that lag.
- **Frozen loadings** (Huber slope of each site's scaled residual on the factor, shrunk halfway to
  the mean): D5 0.50, D10 0.40, D11 0.34, D14 0.44, D16 0.58; D9 has no tuning-year fit and takes
  the mean, 0.45.

## Design points

- **The origin sees little of a short episode.** A 09:00 UTC initialisation reads residuals to
  05:30 UTC, so the forecast for an episode's second day sees the first day's night band and, in
  summer time, one morning half-hour, and nothing in the other bands. Seven of the 12 common
  episodes fall in summer time. Test 2 counted all four bands from the second day.
- **The carry limit ignored what a day-ahead forecast can see.** Six of the eight multi-day
  episodes last 2 days, so the forecasts for them see almost none of the episode. The loadings also
  cap a perfect tracker at 0.458, just above the 40% limit.
- **Recommended corrections, not run:** judge carry against the loading-scaled injected change (40
  to 120% of a perfect tracker), use episodes of 3 or more days or score from the third day, and set
  the release limit from the idealised response before running. This run fails the loading-scaled
  carry criterion too (10% of a perfect tracker). A re-run needs new injections.

## The 20 largest shared-anomaly days (descriptive, no verdict)

The list is ranked from the tuning year's residual streams alone (the largest absolute band mean of
the all-site median of scaled residuals, the five Primary substations D5, D10, D11, D14, and D16 on
every day), and was fixed before any event was matched. The temperature anomaly is the day's mean
ERA5 temperature at D5 minus its trailing 30-day mean. The table is in [report.txt](report.txt).

- **Cold spell:** 9 of the 20 days fall in January 2024, and 4 of those 9 days (8, 13, 14, and 15
  January) are 3 °C or more below the 30-day mean, with 15 January at −7.5 °C. Two more days (22
  September 2023 and 21 October 2023) also meet the 3 °C criterion, so 6 of 20 do. A news report
  of a Met Office reading puts the lowest January temperature since 2019 on 16 January 2024, at
  Dalwhinnie in Scotland, not in the trial area
  ([LBC](https://www.lbc.co.uk/news/exact-location-met-office-weather-alert-snow)).
  No source was found for 8 to 15 January itself.
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
- **Base rates:** the same measure over all 367 days of the tuning year gives a negative sign on 83%
  of days, a temperature 3 °C or more below the 30-day mean on 17%, and a day in January on 8%. So
  17 negative days is what chance gives, while 6 cold days (3.4 expected) and 9 January days (1.7
  expected) are more than chance. England has about 8 bank holidays a year (0.4 expected among 20
  days), and England's men play a few home matches a year (well under 1 expected). No test of these
  matches was made.

## Checks

The audit triggers are checked against what ran. No forecast beyond the gate and the reproduction
was scored, so there is no median-minus-mean forecast, no error of the mean forecast beside a
contrast, and no leave-one-site-out contrast to report.

- **Trigger 1 fired:** three gate tests failed (2, 3, and 4). The `ml-experiment` skill lists seven
  audit triggers, which are conditions that require an audit or an explanation. An Opus numbers
  review and audit, kept outside the repository, reproduced every printed number, found the window
  bug above, and found wording errors; all were accepted and fixed on this page, with none rejected.
- **Trigger 2:** the rebuilt A2 agrees with experiment 106 to 0.0 points. The failures in tests 2 to
  4 were not predicted in the plan, which expected the gate to pass. Tests 2 and 3 are explained
  above. Test 4's failure has a visibility explanation: over days 2 and 3 of the episodes, the
  day-ahead forecasts have seen at most the first day of an episode and the shared part carries
  under 5% of the injected points, so the test says little about whether a shared state adds value.
- **Triggers 3 to 7:** not applicable, because no contrast was computed.

## What was not run

- **Not run, because a failed gate test stops scoring:** the power check, the 13 refitted placebo
  draws (the same analysis with the factor shifted back by whole weeks, to show how much a
  meaningless factor would help), the arms with the shared filter alone, with the local filter
  frozen, with contributors 9 hours late, and with a hindsight factor, the contrasts C1 to C3 of the
  plan, and the replicate-year and scored-year scores. The code for the arms and the gate is in
  `ml-experiments/00108-*/`.
- **Leakage limit of the unrun 9-hour arm:** if the local filter's stream were built from the
  residual minus the loading times the factor, it would read factor values that are not yet usable
  at the local 3.5-hour cutoff. This changed 14 of 100 day-1 corrections in a module check, and the
  module cannot detect it.
- **Caveats:** ERA5 and CAMS inputs up to the cut would make any result a ceiling; the 3.5-hour
  delay is assumed; the gate injected into the five Primary substations D5, D10, D11, D14, and D16
  in one small area; and the 2021 and 2022 fits read the 2023-08-07 capacity vintage 13 to 25 months
  late.

## What to try next

- Re-register the gate with the corrected criteria above, on new injections, and run it before any
  score. If it passes with a shared filter tuned for persistence, run the plan's contrasts.
- Otherwise, record that this shared-filter design could not be shown to work, without concluding
  that a shared state adds nothing.
