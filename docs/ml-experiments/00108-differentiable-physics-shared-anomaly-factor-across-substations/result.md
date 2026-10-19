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
  improves the forecast. The shared filter, tuned on 2023-24, was slow to release a multi-day
  anomaly (74 to 75% of its peak was still present 3 days after it ended) and no better than the
  per-site filter at correcting injected anomalies (1.999 against 1.994 points of mean absolute
  error). Part of the test was mis-specified, so a corrected test is the next step before any
  real score.
date: 2026-10-19
pr: 1383
advice:
  - >-
    Judge a shared filter's carry against the beta-scaled injected change, not against the full
    injected points, because a shrunk loading near 0.45 caps the carry below the 40% limit (gate
    test 2).
  - >-
    Inject episodes of three or more days, or score only the third day onward, because a 09:00 UTC
    origin sees 5.5 hours of an episode's first day, which is the night band only (gate test 2).
  - >-
    Expect a shared state tuned by marginal likelihood on 2023-24 to decay by about 30% per day
    after a step, so a release limit of 25% at 3 days cannot pass; set the limit from the
    idealised response (figure 2) before running the gate (gate test 3).
  - >-
    Do not re-run a failed gate on the same injections after changing the test; write the revised
    test down first and use new injections (gate rule).
code: ml-experiments/00108-differentiable-physics-shared-anomaly-factor-across-substations/
---

# Differentiable physics: a shared anomaly factor across substations

**The experiment stopped at its pre-registered known-answer gate, so it did not test whether a
shared anomaly factor improves the day-ahead forecast.** The gate injected 20 anomalies of known
size into the tuning year (2023-07 to 2024-06). The shared filter, tuned on that year, carried 18% of
the injected points (the plan asked for 40 to 120%) and kept 74 to 75% of its peak 3 days after an
anomaly ended (the plan asked for under 25%). It also corrected injected anomalies no better than the
per-site filter of experiment 106: 1.999 against 1.994 points of mean absolute error over days 2
and 3.
No replicate-year or scored-year row was read, and no contrast was computed. The verdict is
`inconclusive`.

**Part of the gate was mis-specified, and part of the failure is real.** The carry limit of 40%
sat just under the ceiling that the shrunk loadings allow (mean 0.45), and a 09:00 UTC origin sees
only the night band of an episode's first day, so the carry test could hardly pass for any filter.
The release failure is real: the tuned shared filter decays by about 30% per day. The corrected
criteria below are recommendations from this result, not results.

## Figures

![Share of the shared part's peak still present 3 days after each of the 12 common episodes ended.
Every episode is far above the 25% limit.](figure_01_gate_release.svg)

![The tuned shared filter's state after a noise-free three-day step of the factor. The state
reaches 1 within a day and falls to under 25% only on day 9.](figure_02_shared_response.svg)

![The 20 largest shared-anomaly days of the tuning year, by band mean of the all-site median of
scaled residuals.](figure_03_largest_days.svg)

## Gate results

Tuning was frozen before the gate. All numbers are in [report.txt](report.txt), printed by
`e00108_report.py`.

| Test | Result | Limit | Verdict |
|---|---|---|---|
| 1. Recovery slope of the injected change in the all-site median | 1.10 | 0.8 to 1.2 | pass |
| 2. Shared part's carry of the injected points from day 2 | 18% | 40 to 120% | fail |
| 3. Shared part 3 days after an episode ends, share of its peak | 74 to 75% (worst 0.754) | under 25% | fail |
| 4. Mean absolute correction error, KF against A2, days 2 and 3 | 1.999 against 1.994 | KF below A2 | fail |
| 5. No foresight: forecasts issued before an evening episode | bit-identical, 218,160 cases | identical | pass |
| 6. Locality: a local episode's effect on other sites | own site 0.0, others 0.043 points | 0 and under 0.3 | pass |
| 7a. Nesting: KF with beta 0 and the `a0` filters against A2 | gap 0.0 | 1e-6 | pass |

- **Reproduction passed.** A0 and A2 reproduce experiment 106's stored sets (`replicate` and
  `scored`, days 1 to 7) with a maximum difference of 0.0 points, and the refitted `a0` filter
  hyperparameters equal `tuned.json` exactly.
- **Episodes:** 8 common multi-day, 4 common midday, 4 common one-evening, and 4 local, placed with
  seed 108 at least 48 hours from every hand-filtered switching row and at least 10 days apart.
- **Test 2 by day:** the shared part carried 3% of the injected points on an episode's second day
  and 16% on its third. Against a perfect tracker of the factor (loading times scale times the
  injected scaled change), it carried 39%.
- **Test 3 by filter:** the tuned shared filter has `q` 0.0207 per day, `phi` 0.818 per half-hour,
  `sigma_z` 0.401, and `sigma_eps` 0.134, in units of each site's residual scale. Its noise-free
  response to a three-day unit step is 1.00 on day 2 and 0.68, 0.48, 0.35, 0.26, 0.19 on the
  following days, so it needs about six days after the step to fall under 25%.
- **Frozen loadings** (Huber slope, shrunk halfway to the mean): D5 0.50, D10 0.40, D11 0.34, D14
  0.44, D16 0.58; D9 has no tuning-year fit and takes the mean, 0.45.

## Design points

- **The origin sees little of a short episode.** A 09:00 UTC initialisation reads residuals to
  05:30 UTC, so the forecast for an episode's second day sees the night band of its first day and
  nothing in the other three bands. Test 2 counted all four bands from the second day.
- **The carry limit was set without the loading's ceiling.** The shared part is the loading times the
  scale times the filter state, so even a perfect tracker carries about the loading, mean 0.45.
- **Recommended corrections, not run:** judge carry against the loading-scaled injected change (40 to
  120% of a perfect tracker), use episodes of three or more days or score from the third day, and set
  the release limit from the idealised response before running. A re-run needs new injections.

## The 20 largest shared-anomaly days (descriptive, no verdict)

The list is ranked from the tuning year's residual streams alone (the largest absolute band mean
of the all-site median of scaled residuals, five sites on every day), and was fixed before any event
was matched. The temperature anomaly is the day's mean ERA5 temperature at D5 minus its trailing
30-day mean. The table is in [report.txt](report.txt).

- **Cold spell:** 9 of the 20 days fall in January 2024, and 4 of them (8, 13, 14, and 15 January) are
  3 °C or more below the 30-day mean, with 15 January at −7.5 °C. Two more days (22 September 2023
  and 21 October 2023) also meet the 3 °C criterion, so 6 of 20 do. A Met Office-sourced report puts
  the lowest January temperature since 2019 on 16 January 2024
  ([LBC](https://www.lbc.co.uk/news/exact-location-met-office-weather-alert-snow)); I did not find
  a source for 8 to 15 January itself.
- **Bank holidays and the Christmas run:** none of the 20 days.
- **Football:** none. England's four Euro 2024 matches inside the tuning year were on 16, 20, 25,
  and 30 June 2024
  ([Sky Sports](https://skysports.com/football/news/33161/13020967/englands-euro-2024-fixtures-and-route),
  [England Football](https://www.englandfootball.com/england/mens-senior-team/fixtures-results)); the
  nearest listed days are 7 and 28 June.
- **Saharan dust:** none on the listed dates. Reports place dust episodes around 6 September 2023
  ([The Star](https://www.thestar.co.uk/read-this/september-heatwave-windows-and-cars-left-dusty-as-sahara-sand-blows-across-uk-amid-high-temperatures-4284844)),
  8 to 11 October 2023
  ([Shropshire Star](https://cdn-www.shropshirestar.mna.arcpublishing.com/news/environment/2023/10/09/weather-experts-spot-huge-saharan-dust-clouds-heading-to-the-uk)),
  and 26 to 28 January 2024 (Met Office post and
  [NASA Earth Observatory](https://www.earthobservatory.nasa.gov/images/152384/saharan-dust-blows-toward-europe)).
  The nearest listed days are 22 and 29 January and 1 and 22 September. None coincides.
- **Sign:** 17 of the 20 band means are negative, which means demand below the frozen model's mean.

## Checks

The audit triggers are checked against what ran. No forecast beyond the gate and the reproduction
was scored, so there is no median-minus-mean forecast, no error of the mean forecast beside a
contrast, and no leave-one-site-out contrast to report.

- **Trigger 1 fired:** three gate tests failed (2, 3, and 4).
- **Trigger 2:** the rebuilt A2 agrees with experiment 106 to 0.0 points. The failures in tests 2 to
  4 were not predicted in the plan, which expected the gate to pass, and are explained above.
- **Triggers 3 to 7:** not applicable, because no contrast was computed.

## What was not run

- **Not run:** the power check, the 13 refitted placebo draws, the arms F, KF, KF-frz, KF-mva, KF-9h,
  KF-hind, the contrasts C1 to C3, and the replicate-year and scored-year scores. The code for the
  arms and the gate is in `ml-experiments/00108-*/`.
- **Caveats:** ERA5 and CAMS inputs up to the cut would make any result a ceiling; the 3.5-hour delay
  is assumed; the gate injected into the five MW sites in one small area; and the 2021 and 2022
  fits read the 2023-08-07 capacity vintage 13 to 25 months late.

## What to try next

- Re-register the gate with the corrected criteria above, on new injections, and run it before any
  score. If it passes with a shared filter tuned for persistence, run the plan's contrasts.
- Otherwise the per-site filter of experiment 106 already carries what a shared state could add at
  day-ahead leads in this area.
