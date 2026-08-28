# Status of the Flexpectation live service

**Flexpectation's forecasting service has run live on Amazon Web Services (AWS) since 15 July 2026,
about 6 weeks.** The service produces a probabilistic power forecast for each time series in the
trial area, half-hourly out to 14 days ahead, refreshed every 6 hours, and writes every forecast to
the `power_forecasts` table that NGED read. The AWS running cost is estimated at £25 to £35 a
month.

## What the service has done so far

**Version 0.1 served every forecast slot it was scheduled for.** Between the 18:00 UTC slot on
15 July 2026 and the 18:00 UTC slot on 13 August 2026 the schedule called for 117 consecutive
6-hourly slots, and all 117 produced a forecast for all 28 time series then in scope. The count was
verified by reading the forecast table back off storage rather than from the scheduler's own record:
all 117 scheduled forecast times are present, every consecutive pair is exactly 6 hours apart, and
all 28 time series appear in every one of the 117.

**The one input failure in that window degraded the forecast instead of stopping it, which is the
behaviour the design exists to produce.** Dynamical.org first published the 2026-08-09 00Z run of
the European Centre for Medium-Range Weather Forecasts (ECMWF) ensemble with one weather variable
wholly missing, and repaired the run 3 hours 25 minutes later. Version 0.1 treated a wholly-missing
variable as fatal, so no weather data was stored for 9 August, and four forecast slots ran on the
previous day's weather run instead, at 36 to 54 hours old against the 12 to 30 hours a healthy slot
sees. The forecast never stopped, the degradation was bounded, and the degradation was
reconstructable afterwards because every forecast row records which weather run produced it.
Restoring the missing run by hand is the only human intervention the service has needed, and it cost
about a minute.

**The upgrade from version 0.1 to version 0.2 on 13 August 2026 went in without incident, and
version 0.2 has served every slot since.** The version 0.1 stack was retired at about 19:00 UTC that
evening, and version 0.2's first forecast slot ran at 00:00 UTC the following morning, covering 31
time series — three more than version 0.1, under a newly promoted model. From then to the 06:00 UTC
slot on 28 August 2026 the schedule called for 58 consecutive slots, and all 58 produced a forecast
for all 31 time series. No weather run was missed in that window: every slot forecast from weather
data between 12 and 30 hours old, which is the healthy band for a once-daily run. No human
intervention has been needed since the upgrade.

| Period | Version | Time series | Forecast slots served | Interventions |
|---|---|---|---|---|
| 15 July to 13 August 2026 | 0.1 | 28 | 117 of 117 | 1 |
| 14 to 28 August 2026 | 0.2 | 31 | 58 of 58 | 0 |

**Version 0.2 adds the telemetry version 0.1 did not have.** An automated check now reads each
slot's forecast rows back and reports how many weather runs were missed, so a slot forecasting from
stale inputs is recorded as degraded rather than passing unremarked. A wholly-missing weather
variable is retried for 4 hours rather than treated as fatal, which is what would have made the
9 August intervention unnecessary. And the deployment reports errors to Sentry, which the version
0.1 deployment never did. One gap is open and tracked: a slot that is merely degraded, with nothing
having actually failed, still reports itself healthy to the alerting, so the degradation reaches the
checks page rather than an operator.

## The forecast itself is still a placeholder

**The accuracy of the forecast the service produces today should not be read as a statement of what
the project can achieve.** The deployed model is the deliberately naive baseline: one XGBoost model
per time series, trained on lightly cleaned data, with no hyperparameter tuning, no comparison
against a baseline forecaster, no detection of switching events, and no estimate of how much of each
generator's capacity is available on the day. Forecast quality was explicitly out of scope for
versions 0.1 and 0.2.

**What versions 0.1 and 0.2 bought instead is the machinery for running machine-learning experiments
quickly and putting a model into production safely.** Every experiment is scored on fixed
cross-validation folds and written to an MLflow leaderboard, and every run is stamped with the git
commit and with the version of each input table it read, so any result can be reproduced exactly.
Promotion is a single named asset, so the live service always serves an identified model rather than
whatever was most recently trained, and a change of champion model is one deliberate act rather than
a redeployment. Continuous integration runs the linters, the type checker, and the test suite on
every change. The intervention log records every occasion a human had to touch the running service,
because that measurement cannot be reconstructed after the fact.

## What comes next

**Three milestones follow, ordered so that measurement precedes improvement.** Version 0.3 builds
the performance analysis: the full metric set, including two cost-savings figures in pounds — one
for flexibility procurement and one for curtailment — persistence and climatology baselines that
make a score interpretable, time-slice breakdowns from nowcasting to 14 days, and a versioned suite
of failure scenarios that score each candidate model on degraded inputs as well as healthy ones.
Version 0.4 improves the automatic cleaning of NGED's power data, and surfaces what the cleaning
finds to NGED as warnings alongside the forecast. Version 0.5 is the first milestone whose object is
forecast skill: a backlog of XGBoost experiments, run against the leaderboard version 0.3 provides.

**Running the experiments before the measurement exists would mean choosing a champion model
blind.** A leaderboard that cannot say whether a candidate model beats persistence, or how it
behaves when the weather feed is stale, cannot settle which of two models to put in front of NGED.
Building the measurement first is what makes the version 0.5 experiments decidable, and it is also
what lets NGED see the value of a forecast improvement in pounds rather than in error percentages.

## See also

- [Intervention log](https://openclimatefix.github.io/nged-substation-forecast/live_service/intervention-log/)
  — the append-only record of every human intervention, and the verified slot counts quoted above.
- [Roadmap](https://openclimatefix.github.io/nged-substation-forecast/roadmap/) — the full contents
  of versions 0.3, 0.4, and 0.5, and the milestones beyond.
- [AWS running costs](https://openclimatefix.github.io/nged-substation-forecast/architecture/aws-costs/)
  — where the £25 to £35 a month comes from.
