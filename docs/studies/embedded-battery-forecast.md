# How well can an embedded battery's output be forecast, and how many of NGED's batteries have a published plan?

**An embedded battery's half-hourly output cannot be forecast day ahead better than a 56-day
trailing average of its own recent output, from any public input we tested.** The day-ahead price
and the plans that other batteries file lower the error by 0.3% to 1.3%, but the XGBoost forecast
built on them stays level with the average (provisional). Three things would change this answer:
a longer history than 11 scored months, a forecast of the price better than the one tested, and
the battery's own planned output at the time of the forecast, which only about 5% of NGED's
connected batteries publish. The study did not test batteries below 1 MW (most of NGED's),
domestic batteries, or any input that NGED holds and the public does not.

## Summary

**All numbers on this page are provisional until the second science review and the prose review
finish.** The questions below are the ones the study asks. A forecast here is a set of 13
quantiles for each half-hour. The score is the CRPS, the continuous ranked probability score,
which rewards a forecast that is both sharp and honest about its spread and is lower for a better
forecast. A "point of p99" is 1% of a battery's own 99th-percentile absolute output, so scores
compare batteries of different sizes. The **climatology** is the benchmark: the spread of the
battery's own output at the same time of day over the 56 days before the forecast, on days of the
same type (working day, or weekend and bank holiday).

- **Can the output of a battery with a published plan be forecast day ahead? Not better than the
  climatology.**
    - Tested: an XGBoost quantile model given the actual day-ahead price at 18:00 the day before,
      for 35 battery BMUs. A BMU, or Balancing Mechanism Unit, is a battery that the system
      operator schedules individually.
    - Its CRPS was 12.05 points of p99 against 11.99 for the climatology, a difference of +0.066
      points [-0.075, +0.216] that is not statistically significant at the 5% level (provisional).
      An improvement larger than 0.075 points, 0.6% of the climatology's CRPS, is excluded
      (provisional).
    - The price does carry information: the same XGBoost model given the real price beat itself
      given another day's price by 0.139 points [-0.173, -0.105], 1.1% (provisional).
    - A price forecast available at 06:00 the day before did slightly better than another day's
      price (-0.035 points [-0.062, -0.011]), but that XGBoost model was still worse than the
      climatology by +0.195 points [+0.080, +0.331] (provisional).
    - Half the batteries beat the climatology slightly (18 of 35), and two batteries that were
      switched off until spring 2026 do much worse (provisional).
- **Can a battery without a published plan be forecast? We could test only one battery, and the
  answer there is also no better than the climatology.**
    - Tested: the same XGBoost model for one NGED battery, called battery A, at 18:00 the day
      before. Its CRPS was 10.06 points of p99 against 9.98 for the climatology, +0.079 points
      [-0.068, +0.228], not statistically significant at the 5% level (provisional).
    - An improvement larger than 0.068 points, 0.7% of the climatology's CRPS, is excluded
      (provisional).
    - One battery cannot show how the answer varies between batteries, and its meter may include
      site load that nothing here tests.
- **Does a neighbour's published plan help? A little, and only if the plan is public at the time
  of the forecast.**
    - A battery's Final Physical Notification (FPN) is the planned output that a BMU files with the
      system operator. At gate closure, one hour before the half-hour, the notification is final.
    - Tested: for the 35 BMUs one hour ahead, an XGBoost model given the other lead parties'
      notifications beat the same model given another day's notifications by 0.159 points
      [-0.203, -0.113], 1.3% of the climatology's CRPS (provisional).
    - Those notifications recover 5.2% [4.1%, 6.2%] of the gain that the battery's own
      notification gives, which is 3.2 points (provisional). That share is below the tenth the
      plan named as the threshold for "says little".
    - The model given others' notifications was not significantly better than the climatology
      (-0.120 points [-0.283, +0.073]), and the plans that make D4 work are assumed to be public at
      gate closure, which nothing here verified.
    - Resampling the 12 lead parties as well as the months, D4's interval is still below zero
      [-0.236, -0.064] (provisional).
- **How many of NGED's batteries have a published plan? At most 4.5% of the connected storage
  rows, and the registers match none of them by name and capacity.**
    - The Embedded Capacity Register (ECR) lists 176 connected storage rows of 50 kW or more in
      NGED's four licence areas, and 141 of them are under 1 MW.
    - The BMU register holds 8 embedded storage BMUs in NGED's four grid supply point groups, 4.5%
      of the rows, and 47% of the 699 MW (provisional).
    - Comparing names and capacities matched none of the 8 BMUs to an ECR row, so the count is a
      bound from two registers and not a count of matched batteries.
- **What the methods can and cannot support.** The study measures a month-resampled difference in
  CRPS for 35 BMUs of about 50 MW over October 2025 to August 2026. It supports claims about
  those batteries in that year. It does not support a claim about batteries below 1 MW, about
  other years, or about a battery whose output includes site load.

![Figure 1: planned contrasts](assets/embedded_battery_forecast_headline.svg)

**Take-home bullets, one per use of the data:**

- **For a forecaster of a battery with a published plan, use the 56-day climatology as the
  day-ahead baseline.** No XGBoost forecast tested beat it.
- **For a forecaster at gate closure, the battery's own notification is the input that matters.**
  Other batteries' notifications add about 5% of its value, if they are public.
- **For planning NGED's counts, treat at most 5% of the connected batteries as having a published
  plan.**

> **How this page was made.** The research question came from a human. Everything else — the code
> behind every result, the analysis, the figures, and the text — was written by Claude, Anthropic's
> AI model (for this page, Claude Sonnet 5.5). Several independent Claude reviewers have checked
> the method, the evidence, and the prose adversarially.

## Key findings

- **The day-ahead price helps an XGBoost forecast only slightly, and the effect survives
  resampling lead parties.** D2 is [-0.209, -0.069] when lead parties are resampled; see
  [the price tells the XGBoost model something](#the-price-tells-the-xgboost-model-something-but-not-enough).
- **A price forecast at 06:00 recovers about a quarter of the gain of the actual price, and its
  significance does not survive resampling lead parties.** See
  [the forecast price](#a-price-forecast-at-0600-recovers-about-a-quarter-of-the-actual-prices-gain).
- **Two batteries that were idle until spring 2026 do not explain the null result.** See
  [the null result is not an artefact of two idle batteries](#the-null-result-is-not-an-artefact-of-two-idle-batteries).
- **The XGBoost quantile forecast is close to calibrated, and the climatology's lower tail is too
  narrow.** See [the forecasts work](#the-forecasts-work).

## Introduction

**NGED needs to know what the batteries on its distribution network will do tomorrow, and almost
none of them publish a plan.** A BMU files a Final Physical Notification, and most embedded
batteries are not BMUs, so a forecaster of those sees only prices, the calendar, the weather, and
the battery's own recent output. The [battery and solar separation
study](battery-pv-separation.md) used after-the-event inputs throughout, so that study bounds a
forecast without measuring one. This study measures forecasts that use only what a forecaster
knows at each issue time. A census counts how many connected batteries are BMUs. Part A forecasts
35 battery BMUs, which have public output, prices, and notifications. Part B forecasts NGED
battery A, which has no notification, and says what Part A can and cannot transfer.

| Issue time | When | Price input | Own output known up to | Notification |
|---|---|---|---|---|
| 06:00 the day before ("DA-early") | The production forecast's slot before the auction result | A forecast of the day-ahead price | 06:00 the day before | None exists |
| 18:00 the day before ("DA-late") | After the N2EX auction result | The actual N2EX price | 18:00 the day before | Exists, but a public copy at 18:00 is not available |
| One hour before the half-hour ("ID-1h", gate closure) | Gate closure | The actual N2EX price | The half-hour that ended at issue | Own and others' final notifications, for BMUs |

The **lead** of a forecast is the time from its issue to the half-hour it describes.

## Data and methods

**The forecast is 13 quantiles for each half-hour, never a Gaussian.** A battery's output is
bounded and has three modes: charging hard, idle at exactly zero, and exporting hard. The 13
levels are the production service's delivery quantiles (p1, p2, p5, p10, p20, p35, p50, p65, p80,
p90, p95, p98, p99). Each forecast is sorted to repair quantile crossing and clipped to the
battery's 0.1st and 99.9th percentile of output on the training months. The score is CRPS
approximated from the 13 quantiles (the gap-weighted sum of pinball losses), as a percentage of the
battery's own p99. It does not cover the tails beyond p1 and p99.

![Figure 2: one week of a public battery's output and the day-ahead price](assets/embedded_battery_forecast_example_week.svg)

**Four kinds of forecast are compared.** The climatology is described above. The *persistence plus
past errors* forecast adds the training months' error quantiles to the latest observed output at
the same time of day. The *rank rule* places the battery's charging and export at the cheapest and
dearest hours of the day, then adds error quantiles by schedule state. The XGBoost quantile model
(`reg:quantileerror` at the 13 levels) is given 13 columns: half-hour of day, day of week, the
price, its rank within the day, the day's mean price and range, the rank rule's value, the
persistence value, the climatology's median, and four slots for notifications. A slot with no
source holds its own value from seven days earlier. Every arm of the XGBoost model has 13 columns,
with `colsample_bytree` set to 1. Here an "arm" is one choice of inputs, such as "given another
day's price".

**The price inputs bracket what a price forecast could deliver.** *Actual* is the N2EX price. *A
week earlier* is the price at the same hour seven days before. *Forecast* is an XGBoost model of
the price, given the NESO wind forecast as published before the issue time, the calendar, and the
earlier prices, fitted out of fold. *Another day's* is the price of a different day of the same
calendar month and day type, which is the negative control: it keeps the price's typical daily
shape and removes the target day's decisions.

**Only prices that are public at the issue time enter the forecasts.** The N2EX auction's delivery
day is the UK local day, so in summer the last UTC hour of a day (23:00 to 24:00) is priced by the
next day's auction, whose results come out by 10:00 that day. At 18:00 the day before, and at
gate closure before 10:00, that hour's price and the day's rank, mean, range, and rank-rule value
use the price from a week earlier instead. The perfect-price arm at 06:00 keeps every hour,
because it is an upper bound.

**Rows, folds, and intervals.** Scoring starts on 1 October 2025, so that the trailing
climatology has at least 28 days of history, and ends on 31 August 2026: 16,080 half-hours per
battery, the same rows for every arm. The folds are four contiguous blocks of three whole months.
An interval resamples whole calendar months and one of three fitting seeds, 2,000 times, with the
same months for both arms of a contrast. This test covers the month-to-month weather and the seed,
not differences between batteries. A second XGBoost hyperparameter setting is run for every
planned contrast, and a planned contrast is "met" only if the difference is negative and
statistically significant at the 5% level under both settings.

**Planned and exploratory.** A *planned* contrast was written into the study plan before any
result existed. Every other number is exploratory, and anything added after the first run is post
hoc. The planned contrasts are: **D1**, XGBoost given the actual price minus the climatology at
18:00 the day before; **D2**, XGBoost given the actual price minus XGBoost given another day's
price, at 18:00; **D3**, XGBoost given the forecast price minus XGBoost given another day's
price, at 06:00; **D4**, XGBoost given other lead parties' notifications minus XGBoost given
another day's notifications, one hour ahead; and **D5**, D1 for NGED battery A. No correction is
made for multiple comparisons across the five. An exploratory row with no real effect has a
nominal 5% chance of reaching statistical significance at the 5% level, the number of such
spurious rows is unknown because the number of rows with no real effect is unknown, and the rows'
shared months make spurious results cluster.

**The testbed is every embedded battery BMU with enough public data.** A BMU is in the testbed if
its type is embedded and at least 95% of the year's half-hours are present in both Elexon's
settled output and the half-hourly Physical Notification file: 35 BMUs, with a median generation
capacity of 49 MW, run by 12 lead parties. One lead party runs 14 of the 35, and 3 of the 35 sit
in NGED's grid supply point groups. Another lead party's notifications are the others'
notifications in D4, so a battery's own optimiser does not see its own plan. NGED battery A is
shown only as a fraction of its own p99, on days numbered 1 to 7.

**Controls.** A known-answer check on 40 synthetic batteries fed the pipeline the real prices.
For 20 batteries that follow the optimal schedule for the real price, an XGBoost model given the
actual price beat one given another day's price by 5.006 points [-5.303, -4.652] (all 20 batteries
significant); for 20 price-blind batteries the difference was 0.000 points [-0.016, +0.013] (1 of
20 significant, with at most 2 allowed). The positive control is about 35 times the real effect
(D2), so it shows that the pipeline works, and not that it would detect an effect of D2's size
against a real battery's noise.

**Amendments after the first science review are post hoc.** The plan's analysis scores every
half-hour. After the review, three readings were added beside it: the fits with each battery's idle
lead-in dropped, the contrasts with the two batteries that have one removed, and an interval that
resamples the lead parties as well as the months. A calendar month is idle when at least 99% of
its present half-hours are exactly zero, and the idle lead-in is the run of idle months at the
start of a battery's series. The published-price rule above was added by the same review.

![Figure 3: connected storage rows by size class](assets/embedded_battery_forecast_census.svg)

## Results

### The forecasts work

![Figure 4: the two forecasts as fans over one week](assets/embedded_battery_forecast_fans.svg)

**The XGBoost forecast is close to calibrated, and its p10 to p90 band covers 76.5% of outcomes
against the nominal 80%.** The climatology's band covers 77.6% and is 15% wider (48.5 points of
p99 against 42.3). The climatology is too narrow in its lower tail: 5.9% of outcomes fall below its
p1, and 2.6% above its p99.

![Figure 5: reliability at 18:00 the day before](assets/embedded_battery_forecast_reliability.svg)

### Every forecast is level with the climatology or worse

![Figure 6: CRPS by forecast and issue time](assets/embedded_battery_forecast_leaderboards.svg)

**At 18:00 the day before, XGBoost with the actual price scores 12.05 points of p99 against 11.99
for the climatology.** The rank rule scores 12.74 and persistence 17.45. At 06:00 the XGBoost
model given the forecast price scores 12.18, which is 0.195 points [+0.080, +0.331] worse than the
climatology (exploratory). One hour ahead, with no notification, the XGBoost model scores 12.00
against 11.95 for the climatology (+0.048 [-0.097, +0.211], exploratory). Across leads the XGBoost
models' skill against the climatology stays within 0.03 of zero, and in the 12-to-18-hour lead bin
of the 18:00 forecast it is +0.013 [+0.001, +0.024], one of 12 exploratory bins.

![Figure 7: CRPS skill against the climatology by lead](assets/embedded_battery_forecast_skill_by_lead.svg)

![Figure 8: each battery's skill at 18:00 the day before](assets/embedded_battery_forecast_per_battery.svg)

### The price tells the XGBoost model something, but not enough

**The real price beats another day's price by 0.139 points (D2), and the second setting agrees
(-0.136 [-0.170, -0.104]).** That is 1.1% of the climatology's CRPS. The price from a week earlier
recovers almost none of it (-0.012 [-0.029, +0.004], exploratory), so the gain is the target day's
price and not the month's typical shape. D1, the same model against the climatology, is +0.066
points [-0.075, +0.216] at the primary setting and -0.002 [-0.138, +0.153] at the second. The
climatology adapts every day, while the XGBoost model is fitted across month blocks, some later than
the month it is tested on, so D1 tests this XGBoost model against an adaptive baseline and not what
public inputs can carry in principle.

### A price forecast at 06:00 recovers about a quarter of the actual price's gain

**The price forecast's MAE is 14.92 GBP per MWh against 22.29 for the price a week earlier (ratio
0.669), so the XGBoost forecast of the price is not weak by the plan's rule.** It is worse than the
week-earlier price in January and February 2026. D3, the XGBoost model given the forecast price
minus the same model given another day's price, is -0.035 points [-0.062, -0.011] (second setting
-0.044), 0.3% of the climatology's CRPS. That model is still 0.195 points worse than the
climatology. The perfect price at 06:00 gives -0.144 [-0.181, -0.108] against another day's price
(exploratory).

### Other batteries' plans help a little

**Others' notifications improve the forecast one hour ahead by 0.159 points (D4), and the battery's
own notification improves it by 3.205 points (exploratory).** The share recovered is 5.2% [4.1%,
6.2%]; leaving out the five batteries whose notification is exactly zero in at least 80% of
half-hours changes it to 5.1%. The false-alarm rule holds: the shuffled-notification arm gains
0.009 points, far below D4's 0.159. Leaving out the largest lead party (14 batteries) from the
neighbour set gives -0.157 [-0.202, -0.111] (exploratory). Same-lead-party notifications, for the
29 batteries that have another unit of their own party, give -0.607 [-0.715, -0.490] against no
notification (exploratory). A battery's own notification at gate closure gives a CRPS skill of
+0.264 against the climatology (32 of 35 batteries positive, exploratory); a forecaster has it only
if Elexon makes the notification public at gate closure, which the study did not verify.

### The null result is not an artefact of two idle batteries

**Two batteries of one lead party publish exactly zero until March or April 2026.** They alone
made the first review suspect D1. Dropping their idle lead-in from training and scoring leaves D1 at
+0.068 points [-0.060, +0.202], and removing both batteries leaves it at +0.007 [-0.094, +0.104]
(post hoc). D2, D3, and D4 move by at most 0.010 points under either reading.

![Figure 9: D1 to D4 under four ways of choosing the rows or the interval](assets/embedded_battery_forecast_sensitivity.svg)

**Resampling the 12 lead parties as well as the months widens every interval.** D2 is [-0.209,
-0.069] and D4 is [-0.236, -0.064], so both stay below zero. D1 is [-0.098, +0.360] and D3 is
[-0.069, +0.009], so D3 crosses zero (post hoc). The per-party means of D3 have both signs, so the
D3 result is specific to these 35 batteries in this year.

### NGED battery A is also level with the climatology

![Figure 10: NGED battery A over seven numbered days](assets/embedded_battery_forecast_nged_week.svg)

**D5 is +0.079 points [-0.068, +0.228], and the second setting gives +0.017 [-0.104, +0.140].**
An improvement larger than 0.068 points is excluded at the primary setting and 0.104 at the
second. With the fleet's notifications as inputs one hour ahead, the XGBoost model's CRPS is 9.92
against 9.95 for the climatology and gains 0.123 points [-0.160, -0.087] over another day's
notifications (exploratory). The census finds no reviewed storage BMU that matches NGED battery A,
so by those keys the battery is not a BMU.

## Discussion: what to use

- **For a day-ahead forecast of any embedded battery, use the 56-day climatology until a
  forecast beats it.** The XGBoost forecasts are level with it and wider in calibration terms only
  at the lower tail.
- **For NGED's planning, a forecast that needs the battery's own notification applies to at most
  4.5% of the connected storage rows.** The notification is the single input that moved the error
  by more than 1.3%.
- **If a better answer is needed, test the forecast of the price first.** The actual price recovers
  1.1% of the CRPS, and a forecast price recovers about a quarter of that. A price forecast of
  that quality would still leave the forecast level with the climatology.
- **Archiving Final Physical Notifications as they are filed would allow a day-ahead notification
  forecast.** The public API keeps only the final version, so this study has no day-ahead
  notification arm.

## Limitations

- **The testbed is large batteries.** The 35 BMUs have a median capacity of 49 MW, while the
  median connected storage row in the ECR is 0.19 MW and 141 of 176 rows are below 1 MW. A result
  for the testbed does not transfer to small batteries.
- **BMUs take balancing actions.** A BMU's output includes acceptances in the Balancing Mechanism,
  which a non-BMU battery lacks, so the part of Part A that depends on being a BMU does not
  transfer to Part B.
- **Eleven scored months are the sample.** Each fold holds two or three months, and every
  interval rests on 11 months. One lead party runs 14 of 35 batteries.
- **NGED battery A is one battery,** and its meter may include site consumption.
- **The notification is assumed public at gate closure.** The API keeps only the final version, and
  the minute at which Elexon publishes it is unverified.
- **The simple forecasts are crude by construction.** The persistence forecast adds a residual
  spread to a weakly correlated centre, and the rank rule pools residuals across the day. Neither
  result shows that recent output or the price rank carries no information. The one-hour-ahead
  XGBoost model with no notification sees one lag of the battery's output, so it says nothing
  about real-time telemetry.
- **The score ignores the tails beyond p1 and p99,** where the climatology is miscalibrated.
- **The census is a bound.** The registers match none of the 8 BMUs by name and capacity; the
  ECR copy is the August 2026 workbook and was not compared with NGED's portal version.
- **The XGBoost fits ran on the CPU** with two hyperparameter settings and three seeds, and the
  deterministic forecasts hold no random draw, so the seed term of their intervals is zero.
- **The shuffled arms draw from the whole calendar month,** future days included. They therefore
  know the month's price level in advance, which suits a negative control.

## Scope

The study says nothing about batteries below 1 MW, domestic batteries, the frequency-response
contracts that batteries hold, state of charge, any input that only NGED holds (such as metered
flows at the primary substation), forecasts beyond one day ahead, or any year before September
2025.

## Data and code availability

**Inputs.** The N2EX day-ahead prices are NESO open data. Elexon supplies the settled output
(B1610), the Physical Notifications, the BMU register, and the NESO wind forecast. The Embedded
Capacity Register is NGED's workbook; this page carries only counts and bands from it. NGED battery
A's series is private, and the page shows it only as a fraction of its own p99. The code is under
`studies/embedded_battery_forecast/` and `packages/studies/`, at the commit of the pull request
that added this page. XGBoost runs the `reg:quantileerror` objective, and the hyperparameters are
`PRIMARY_HYPER_PARAMETERS` and `SENSITIVITY_HYPER_PARAMETERS` in `studies.cross_validation`.

## Reproducing the figures

```bash
export OMP_NUM_THREADS=2
uv run python studies/embedded_battery_forecast/census_report.py
uv run python studies/embedded_battery_forecast/forecast_inputs.py
uv run python studies/embedded_battery_forecast/forecast_synthetic.py primary
uv run python studies/embedded_battery_forecast/forecast_price_model.py
uv run python studies/embedded_battery_forecast/forecast_bmus.py primary
uv run python studies/embedded_battery_forecast/forecast_bmus.py sensitivity
uv run python studies/embedded_battery_forecast/forecast_nged_battery_a.py primary
uv run python studies/embedded_battery_forecast/forecast_nged_battery_a.py sensitivity
FIT_VARIANT=idle_dropped uv run python studies/embedded_battery_forecast/forecast_bmus.py primary
FIT_VARIANT=idle_dropped uv run python studies/embedded_battery_forecast/forecast_bmus.py sensitivity
FIT_VARIANT=pre_review uv run python studies/embedded_battery_forecast/forecast_report.py
FIT_VARIANT=idle_dropped uv run python studies/embedded_battery_forecast/forecast_report.py
uv run python studies/embedded_battery_forecast/forecast_report.py
uv run python studies/embedded_battery_forecast/forecast_charts.py
```
