# How well can an embedded battery's output be forecast, and how many of NGED's batteries have a published plan?

**For 35 large embedded batteries from October 2025 to August 2026, no forecast we built from the
day-ahead price, the calendar, and the battery's own metered output beat a 56-day trailing
climatology at 18:00 the day before (provisional).** The climatology is a forecast in its own
right: the spread of the battery's own output at the same time of day over the 56 days before the
forecast, on days of the same type. It is also a loose one. Its middle forecast is wrong by
15.5% of the battery's largest typical output on an average half-hour, and its p10 to p90 band
(the range that should hold 80% of outcomes) is about half that output wide. A better XGBoost
forecast than the one we built, one that adapts as fast as the climatology does, was not tested.
The study did not test batteries below 1 MW, which are most of NGED's connected batteries,
domestic batteries, or any input that only NGED holds. Whether a battery publishes a plan of its
own output, a Final Physical Notification (FPN) filed with the system operator, matters far more
than any price input one hour ahead, but the registers suggest that few of NGED's batteries do.

## Summary

**All numbers on this page are provisional until the prose review finishes.** This page asks
four questions. Every forecast here is 13 quantiles for each half-hour. A quantile is a value
that the outcome falls below with a stated probability, so the 10th quantile (p10) is exceeded
nine times in ten. The score is the CRPS, the continuous ranked probability score: the average,
over the 13 quantiles, of a penalty that grows with the distance between the outcome and the
quantile, weighted so that a forecast scores well only if it is both narrow (sharp) and honest
about its spread. The CRPS is in the units of the outcome, and smaller is better. A "point of
p99" is 1% of a battery's own 99th-percentile absolute output, so the CRPS is comparable between
batteries of different sizes, and a negative difference between two CRPS values favours the first
forecast. Square brackets after a number give its 95% interval. A planned contrast is "excluded"
when its interval does not contain it.

The forecasts compared are the **climatology** described above, and XGBoost quantile models. XGBoost
is a gradient-boosted decision-tree library, fitted here to predict the 13 quantiles from the
price, the calendar, and the battery's recent output. Five contrasts were written into the study
plan before any result existed. **D1**: XGBoost given the actual price minus the climatology, at
18:00 the day before. **D2**: XGBoost given the actual price minus XGBoost given the price of
another day, at 18:00. **D3**: XGBoost given a forecast price minus XGBoost given another day's
price, at 06:00 the day before. **D4**: XGBoost given other lead parties' FPNs minus XGBoost given
another day's FPNs, one hour ahead. **D5**: D1 for NGED battery A. The "another day" forecast is the
negative control: it keeps the price's typical daily shape and removes the target day's decisions,
so any gain over it comes from the target day.

An FPN is the output that a Balancing Mechanism Unit (BMU), the unit in which a battery's output
is notified, metered, and settled with the system operator, files with the operator for each
half-hour. It is final at gate closure, one hour before the half-hour. A lead party is the company
that trades for the BMU, which is not necessarily the company that decides when the battery
charges.

- **Can a BMU's output be forecast day ahead? Not better than the climatology.**
    - Tested: XGBoost given the actual price at 18:00 the day before, for 35 BMUs, against the
      climatology (D1).
    - Its CRPS was 12.05 points of p99 against 11.99 for the climatology, a difference of +0.066
      points [-0.075, +0.216], which is not statistically significant at the 5% level
      (provisional).
    - An improvement larger than 0.14 points, 1.1% of the climatology's CRPS, is excluded at both
      XGBoost settings (0.075 at the first setting alone) (provisional).
    - The price does carry information: XGBoost given the real price beat XGBoost given another
      day's price by 0.139 points [-0.173, -0.105], 1.1% of the climatology's CRPS (D2,
      provisional).
    - A price forecast available at 06:00 beat another day's price by 0.035 points [-0.062,
      -0.011] (D3), but XGBoost given it was worse than the climatology by 0.195 points [+0.080,
      +0.331] (provisional).
    - Across the 35 batteries, 18 have a positive point estimate of their CRPS improvement over the
      climatology, with a median of +0.1% of the climatology's CRPS (provisional). Two batteries with
      exactly zero output until spring 2026 do much worse.
- **Can a battery without a published plan be forecast? We could test one battery, with the same
  answer.**
    - Tested: XGBoost given the actual price, for one NGED battery, called battery A, against the
      climatology at 18:00 the day before (D5).
    - Its CRPS was 10.06 points of p99 against 9.98, +0.079 points [-0.068, +0.228], not
      statistically significant at the 5% level (provisional). An improvement larger than 0.10
      points, 1.0% of the climatology's CRPS, is excluded at both XGBoost settings (provisional).
    - One battery cannot show how the answer varies, and its meter may include site load.
- **Does a neighbour's FPN help? A little, and only if the FPN is public at gate closure.**
    - Tested: for the 35 BMUs one hour ahead, XGBoost given other lead parties' FPNs against
      XGBoost given another day's FPNs (D4).
    - The gain was 0.159 points [-0.203, -0.113], 1.3% of the CRPS of XGBoost given another
      day's FPNs (provisional).
    - Other lead parties' FPNs recover 5.2% of the gain that the battery's own FPN gives, 3.2
      points (95% interval 2.0% to 7.5% when lead parties are resampled as well as months;
      provisional). The study plan named a tenth as the threshold below which the fleet's FPNs
      "say little".
    - XGBoost given others' FPNs was not significantly better than the climatology (-0.120
      points [-0.283, +0.073]; exploratory).
- **How many of NGED's batteries have an FPN? The registers suggest few, and cannot say how
  few.**
    - NGED's Embedded Capacity Register (ECR) lists 176 connected storage rows of 50 kW or more in
      NGED's four licence areas, and 141 rows are under 1 MW. A licence area is a region in which
      NGED operates.
    - The BMU register lists 8 embedded storage BMUs in NGED's four grid supply point groups, the
      groups of substations where the transmission network meets NGED's: 8 BMUs to 176 rows, or
      4.5% (2.8% counting the 5 BMUs with a full year of settled output; provisional).
    - Comparing names and capacities matched none of the 8 BMUs to an ECR row, and one BMU may
      stand for several rows or for none, so the true share of rows with an FPN is not known.
- **What the methods can and cannot support.** The study measures month-resampled differences in
  CRPS for 35 BMUs of about 50 MW over one year. It supports claims about those batteries in that
  year. It does not support claims about batteries below 1 MW, about other years, or about a
  battery whose meter includes site load. The forecasts also assume that the battery's own output
  is known at the issue time (see "Data and methods").

![Figure 1: planned contrasts](assets/embedded_battery_forecast_headline.svg)

**Take-home bullets, one per use of the data:**

- **For a day-ahead forecast of a large embedded battery, use the 56-day climatology as the
  baseline.** No XGBoost forecast tested beat it.
- **For a forecast one hour ahead, the battery's own FPN is the input that matters,** if it is
  public at gate closure. Other lead parties' FPNs add about 5% of its value.
- **For NGED's planning, expect only a small share of connected batteries to have an FPN.** The
  registers give 8 BMUs against 176 storage rows.

> **How this page was made.** The research question came from a human. Everything else — the code
> behind every result, the analysis, the figures, and the text — was written by Claude, Anthropic's
> AI model (for this page, Claude Sonnet 5.5, Claude Sonnet 5, and Claude Opus 5.5). Several
> independent Claude reviewers have checked the method, the evidence, and the prose
> adversarially.

## Key findings

- **The day-ahead price lowers the XGBoost error by 1.1% (actual price) and 0.3% (forecast
  price), measured against the climatology's CRPS, and the first effect survives resampling lead
  parties.** D2 is [-0.209, -0.069] when lead parties are resampled; see
  [the price tells XGBoost something](#the-price-tells-xgboost-something-but-not-enough).
- **A price forecast at 06:00 recovers about a quarter of the actual price's gain, and its
  significance does not survive resampling lead parties.** See
  [the forecast price](#a-price-forecast-at-0600-recovers-about-a-quarter-of-the-actual-prices-gain).
- **Dropping the idle months of two batteries does not rescue them, and removing them leaves D1
  null.** See [the idle batteries](#two-idle-batteries-do-not-explain-the-null-result).
- **The XGBoost forecast is slightly overconfident (its p10 to p90 band covers 76.5% of
  outcomes), and the climatology's lower-tail miss is mostly ties at exactly zero.** See
  [the forecasts](#the-forecasts-and-their-calibration).

## Introduction

**A forecast of what embedded batteries will do tomorrow, half-hour by half-hour, is an input to
planning a distribution network, and almost no battery publishes its own plan.** A BMU files an
FPN, and most embedded batteries are not BMUs, so a forecaster of those sees only prices, the
calendar, the weather, and the battery's own recent output. The [battery and solar separation
study](battery-pv-separation.md) used after-the-event inputs throughout, so that study bounds a
forecast without measuring one. This study measures forecasts built at three issue times, with
one assumption stated under "Data and methods". A census counts how many connected batteries are
BMUs. Part A forecasts 35 battery BMUs, which have public output, prices, and FPNs. Part B
forecasts NGED battery A, which has no FPN, and says what Part A can and cannot transfer.

| Issue time | When | Price input | FPN |
|---|---|---|---|
| 06:00 the day before ("DA-early") | 06:00 UTC, before the day-ahead auction result is out | A forecast of the day-ahead price | None exists yet |
| 18:00 the day before ("DA-late") | 18:00 UTC, after the result | The actual price | Exists, but a public copy at 18:00 is not available |
| Gate closure ("ID-1h") | One hour before the half-hour | The actual price | Own and others' final FPNs, for BMUs |

The **lead** of a forecast is the time from its issue to the half-hour it describes. The
day-ahead auction is the N2EX auction run for NESO (the National Energy System Operator), which
sets the hourly price for the next day, with results public by 10:00 UK time on the day before.

## Data and methods

**The forecast is 13 quantiles for each half-hour, never a Gaussian.** A battery's output is
bounded and has three modes: charging hard, idle at exactly zero, and exporting hard, so a
Gaussian would put its probability where the battery rarely is. The 13 levels are p1, p2, p5, p10,
p20, p35, p50, p65, p80, p90, p95, p98, and p99. Each forecast is sorted to repair quantile
crossing and clipped to the 0.1st and 99.9th percentile of the battery's output on the training
months. The CRPS is approximated from the 13 quantiles as the weighted sum of pinball losses (a
pinball loss penalises an outcome above a quantile in proportion to the quantile's level, and an
outcome below it in proportion to one minus the level), in points of p99. It does not cover the
tails beyond p1 and p99. The **CRPS skill** used in some figures is one minus the ratio of a
forecast's CRPS to the climatology's: it is positive when the forecast is better, and 0.01 means
a CRPS 1% lower.

![Figure 2: one week of a public battery's output and the day-ahead price](assets/embedded_battery_forecast_example_week.svg)

**Four kinds of forecast are compared.** The climatology is described at the top of the page. The
*persistence plus past errors* forecast adds the training months' error quantiles to the latest
observed output at the same time of day. The *rank rule* places charging and export at the
cheapest and dearest hours of the day, then adds error quantiles by schedule state. The XGBoost
quantile model (`reg:quantileerror` at the 13 levels) is given 13 columns: half-hour of day, day
of week, the price, its rank within the day, the day's mean price and range, the rank rule's
value, the persistence value, the climatology's median, and four slots for FPNs. A slot with no
source holds its own value from seven days earlier (for NGED battery A, the climatology's median
from seven days earlier). Every XGBoost forecast has 13 columns, with `colsample_bytree` set
to 1. Each choice of inputs, such as "given another day's price", is called an arm in the report.

**The price inputs bracket what a price forecast could deliver.** *Actual* is the N2EX price. *A
week earlier* is the price at the same hour seven days before. *Forecast* is an XGBoost model of
the price, given the wind forecast that NESO publishes through Elexon's data portal as published
before the issue time, the calendar, and the earlier prices, fitted out of fold. *Another day's*
is the price of a different day of the same calendar month and day type, the negative control
described above.

**Only prices that are public at the issue time enter the forecasts.** The auction prices each
hour of the UK day, midnight to midnight UK time. In summer UK clocks are one hour ahead of UTC,
so the hour 23:00 to 24:00 UTC already belongs to the next UK day, and its price is not public
until 10:00 UTC that day. A forecast issued before then replaces that hour's price, and the day's
rank, mean, range, and rank-rule value, with the price at the same hour a week earlier. The
perfect-price forecast at 06:00 keeps every hour, because it is an upper bound.

**The battery's own output is assumed known in real time.** Elexon publishes settled output
about seven days after the half-hour, so a forecaster using public data alone would have neither
the persistence value nor the latest week of the climatology. The study assumes the network
operator reads each battery's output from its own telemetry, as NGED does for NGED battery A.
That assumption favours XGBoost and persistence over the climatology, so the D1 result is
conservative.

**Rows, folds, and intervals.** The data run from September 2025 to August 2026. Scoring starts
on 1 October 2025, so the climatology has at least 28 days of history (56 days from late
October), and ends on 31 August 2026: 11 scored months and 16,080 half-hours per battery, the
same rows for every forecast. The folds are four contiguous blocks of three whole months, so
the first holds two scored months. An interval resamples whole calendar months, and one of three
fitting seeds, 2,000 times, with the same months for both forecasts of a contrast. Resampling
means drawing months at random with replacement and recomputing the difference. This test covers
the month-to-month weather and the seed, not differences between batteries. A second XGBoost
hyperparameter setting is run for every planned contrast, and a planned contrast is "met" only
if the difference is negative and statistically significant at the 5% level under both settings.

**Planned and exploratory.** A *planned* contrast, D1 to D5 above, was written into the study
plan before any result existed. Every other number is exploratory, and anything added after the
first run is post hoc. No correction is made for multiple comparisons across the five. An
exploratory row with no real effect has a nominal 5% chance of reaching statistical significance
at the 5% level, the number of such spurious rows is unknown because the number of rows with no
real effect is unknown, and the rows' shared months make spurious results cluster.

**The testbed is every embedded battery BMU with enough public data.** A BMU is in the testbed if
its type is embedded and at least 95% of the year's half-hours are present in both Elexon's
settled output and the half-hourly FPN file: 35 BMUs, with a median generation capacity of 49 MW,
run by 12 lead parties. One lead party runs 14 of the 35, and 3 of the 35 sit in NGED's grid
supply point groups. In D4 the neighbours are batteries of other lead parties. A lead party is
the trading counterparty, not necessarily the party that schedules the battery, so some
neighbours may share the target's scheduler. NGED battery A is shown only as a fraction of its
own p99, on days numbered 1 to 7.

**Controls.** A known-answer check on 40 synthetic batteries fed the pipeline the real prices.
For 20 batteries that follow the optimal schedule for the real price, XGBoost given the actual
price beat XGBoost given another day's price by 5.006 points [-5.303, -4.652] (all 20 batteries
significant); for 20 price-blind batteries the difference was 0.000 points [-0.016, +0.013] (1 of
20 significant, with at most 2 allowed). The positive control is about 35 times the real effect
(D2), so it shows that the pipeline works, and not that it would detect an effect of D2's size
against a real battery's noise.

**Amendments after the first science review are post hoc.** The plan's analysis scores every
half-hour. After the review, three readings were added beside it: the fits with each battery's
idle lead-in dropped, the contrasts with the two batteries that have an idle lead-in removed, and
an interval that resamples the lead parties as well as the months. A calendar month is idle when
at least 99% of its present half-hours are exactly zero, and the idle lead-in is the run of idle
months at the start of a battery's series. The published-price rule above was added by the same
review.

![Figure 3: connected storage rows by size class](assets/embedded_battery_forecast_census.svg)

## Results

### The forecasts and their calibration

![Figure 4: the two forecasts as fans over one week](assets/embedded_battery_forecast_fans.svg)

**The XGBoost forecast is slightly overconfident: its p10 to p90 band covers 76.5% of outcomes
against the nominal 80%, and its p1 to p99 band covers 96.3% against 98%.** The climatology's
p10 to p90 band covers 77.6% and is 6.2 points of p99 wider (48.5 against 42.3). At the lower
end, 2.9% of outcomes fall strictly below the climatology's p1, and another 2.95% equal it, almost
all of them exactly zero outputs of the two batteries that were idle until spring 2026. Figure 5
counts a tie as half.

![Figure 5: reliability at 18:00 the day before](assets/embedded_battery_forecast_reliability.svg)

### Every XGBoost forecast is level with the climatology or worse

![Figure 6: CRPS by forecast and issue time](assets/embedded_battery_forecast_leaderboards.svg)

**At 18:00 the day before, XGBoost given the actual price scores 12.05 points of p99 against
11.99 for the climatology.** The rank rule scores 12.74 and persistence 17.45. At 06:00, XGBoost
given the forecast price scores 12.18, which is 0.195 points [+0.080, +0.331] worse than the
climatology (exploratory). One hour ahead, with no FPN, XGBoost scores 12.00 against 11.95 for
the climatology (+0.048 [-0.097, +0.211], exploratory). Across leads the XGBoost forecasts' skill
stays near zero, and is significantly positive in one 6-hour bin only (Figure 7; exploratory).

![Figure 7: CRPS skill against the climatology by lead](assets/embedded_battery_forecast_skill_by_lead.svg)

**For individual batteries, 18 of 35 have a positive point estimate of skill at 18:00 the day
before, with a median of +0.001, and two batteries are far below zero.** No per-battery test was
run, so these are point estimates (Figure 8).

![Figure 8: each battery's skill at 18:00 the day before](assets/embedded_battery_forecast_per_battery.svg)

### The price tells XGBoost something, but not enough

**The real price beats another day's price by 0.139 points (D2), and the second setting agrees
(-0.136 [-0.170, -0.104]).** That is 1.1% of the climatology's CRPS. The price from a week
earlier recovers almost none of it (-0.012 [-0.029, +0.004], exploratory), so the gain is the
target day's price and not the month's typical shape. D1 is +0.066 points [-0.075, +0.216] at
the first setting and -0.002 [-0.138, +0.153] at the second. The climatology adapts every day,
while XGBoost is fitted across month blocks, some later than the month it is tested on, so D1
tests this XGBoost model against an adaptive baseline, and not what public inputs can carry in
principle.

### A price forecast at 06:00 recovers about a quarter of the actual price's gain

**The price forecast's mean absolute error is 14.92 GBP per MWh against 22.29 for the price a
week earlier (ratio 0.669).** The ratio is below the 0.8 that the study plan set as the mark of a
weak price forecast. The forecast is worse than the week-earlier price in January 2026 (22.89
against 17.43) and February 2026 (12.63 against 11.25). D3 is -0.035 points [-0.062, -0.011]
(second setting -0.044), 0.3% of the climatology's CRPS, and XGBoost given the forecast price is
still 0.195 points worse than the climatology. The perfect price at 06:00 gives -0.144 [-0.181,
-0.108] against another day's price (exploratory).

### Other lead parties' FPNs help a little

**Others' FPNs improve the forecast one hour ahead by 0.159 points (D4), and the battery's own FPN
improves it by 3.205 points (exploratory).** The share recovered is 5.2% [4.1%, 6.2%] with months
resampled, and 2.0% to 7.5% with lead parties resampled too; leaving out the five batteries whose
FPN is exactly zero in at least 80% of half-hours changes it to 5.1%. The negative control passes:
XGBoost given another day's others' FPNs gains 0.009 points over no FPN, far below D4's 0.159.
Leaving out the largest lead party (14 batteries) from the neighbour set gives -0.157 [-0.202,
-0.111] (exploratory). Same-lead-party FPNs, for the 29 batteries that have another unit of their
own party, give -0.607 points [-0.715, -0.490] (4.8%) against no FPN (exploratory). The battery's
own FPN gives a CRPS skill of +0.264 against the climatology (32 of 35 batteries positive,
exploratory). A forecaster has it only if Elexon makes the FPN public at gate closure, which the
study did not verify.

### Two idle batteries do not explain the null result

**Two batteries of one lead party have exactly zero metered output until March or April 2026.**
The data cannot tell switched off from not yet commissioned. Dropping their idle lead-in from
training and scoring leaves D1 at +0.068 points [-0.060, +0.202]. Removing both batteries leaves
it at +0.007 [-0.094, +0.104] at the first setting and -0.066 [-0.162, +0.030] at the second,
still not statistically significant at the 5% level (post hoc). D2, D3, and D4 move by at most
0.010 points under either reading. Dropping the idle months does not rescue the two batteries:
their skill against the climatology stays near -0.2. With four three-month folds, a battery that
starts in spring has two or three working months to train on, while the climatology needs eight
weeks.

![Figure 9: D1 to D4 under four ways of choosing the rows or the interval](assets/embedded_battery_forecast_sensitivity.svg)

**Resampling the 12 lead parties as well as the months widens every interval.** D2 is [-0.209,
-0.069] and D4 is [-0.236, -0.064], so both stay below zero. D1 is [-0.098, +0.360] and D3 is
[-0.069, +0.009], so D3 crosses zero (post hoc; the intervals rest on 12 lead parties, one of
which runs 14 batteries). The per-party means of D3 have both signs, so the D3 result is specific
to these 35 batteries in this year.

### NGED battery A is also level with the climatology

![Figure 10: NGED battery A over seven numbered days](assets/embedded_battery_forecast_nged_week.svg)

**D5 is +0.079 points [-0.068, +0.228], and the second setting gives +0.017 [-0.104, +0.140].**
An improvement larger than 0.068 points is excluded at the first setting and 0.104 at the second.
With the fleet's FPNs as inputs one hour ahead, XGBoost's CRPS is 9.92 against 9.95 for the
climatology, and gains 0.123 points [-0.160, -0.087] over another day's FPNs (exploratory). The
census finds no reviewed storage BMU that matches NGED battery A, so by those keys the battery is
not a BMU.

## Discussion: what to use

- **For a day-ahead forecast of the large batteries tested, and of NGED battery A, use the 56-day
  climatology until a forecast beats it.** The XGBoost forecasts are level with it in CRPS,
  slightly sharper, and better calibrated in the lower tail. The study did not test batteries
  below 1 MW, so the climatology is a starting point there, and not a tested result.
- **For NGED's planning, expect a forecast that needs the battery's own FPN to apply to a small
  share of connected batteries.** The registers give 8 BMUs against 176 storage rows. The own FPN
  moved the error by 3.2 points one hour ahead (27%), same-lead-party FPNs by 0.6 points (5%),
  and others' FPNs by 0.16 points (1.3%).
- **What would change these answers:** an XGBoost model that adapts as fast as the climatology
  (see Limitations), more than 11 scored months, and an FPN archive.
- **Archiving FPNs as they are filed would allow a day-ahead FPN forecast.** The public API keeps
  only the final version, so this study has no day-ahead FPN forecast.

## Limitations

- **The testbed is large batteries.** The 35 BMUs have a median capacity of 49 MW, while the
  median connected storage row in the ECR is 0.19 MW and 141 of 176 rows are below 1 MW. A result
  for the testbed need not hold for small batteries.
- **Part A may not transfer to Part B.** A BMU is dispatched through the Balancing Mechanism,
  which a non-BMU battery is not, and NGED battery A is one battery whose meter may include site
  load.
- **Eleven scored months are the sample.** Every interval rests on 11 months, and one lead party
  runs 14 of 35 batteries.
- **The FPN is assumed public at gate closure.** The API keeps only the final version, and the
  minute at which Elexon publishes it is unverified.
- **XGBoost was not given the climatology's adaptivity.** The XGBoost model sees the climatology
  only through its median and is retrained once per three-month fold, so a battery that changes
  behaviour mid-year handicaps it. An XGBoost model that adapts as fast as the climatology was
  not tested, and dropping the idle lead-in leaves the two idle batteries' skill near -0.2, so
  the idle-month explanation of D1 is not supported.
- **The simple forecasts are crude.** The persistence forecast adds a residual spread to a weakly
  correlated centre, and the rank rule pools residuals across the day. Neither result shows that
  recent output or the price rank carries no information. The one-hour-ahead XGBoost model with
  no FPN sees one lag of the battery's output, so it says nothing about real-time telemetry.
- **The score ignores the tails beyond p1 and p99.**
- **The census is a ratio of two registers.** The registers match none of the 8 BMUs to an ECR row
  by name and capacity; the 699 MW is the total export capacity of the 176 rows, and the ECR copy
  is the August 2026 workbook, not compared with NGED's portal version.
- **Fits ran on the CPU** with two hyperparameter settings and three seeds. The deterministic
  forecasts hold no random draw, so the seed term of their intervals is zero.
- **The another-day arms draw from the whole calendar month,** future days included, so they know
  the month's price level in advance, which suits a negative control.

## Scope

The study says nothing about batteries below 1 MW, domestic batteries, the frequency-response
contracts that batteries hold, state of charge, any input that only NGED holds (such as metered
flows at the primary substation), forecasts beyond one day ahead, or any period other than
September 2025 to August 2026.

## Data and code availability

**Inputs.** The N2EX day-ahead prices are NESO open data. Elexon supplies the settled output
(B1610), the FPNs, the BMU register, and the wind forecast that NESO publishes through it. The
Embedded Capacity Register is NGED's workbook, marked shared and not open, so only a holder of the
workbook can rerun the census; this page carries only counts and bands from it. NGED battery A's
series is private, and the page shows it only as a fraction of its own p99. The code is under
`studies/embedded_battery_forecast/` and `packages/studies/`, at the commit of the pull request
that added this page. XGBoost runs the `reg:quantileerror` objective, and the hyperparameters are
`PRIMARY_HYPER_PARAMETERS` and `SENSITIVITY_HYPER_PARAMETERS` in `studies.cross_validation`.

## Reproducing the figures

The `pre_review` report reads fits archived from the first run, which a clean checkout does not
hold, so a clean checkout cannot rebuild the table that compares the earlier whole-UTC-day
prices. Every other step below rebuilds from the inputs.

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
FIT_VARIANT=idle_dropped uv run python studies/embedded_battery_forecast/forecast_report.py
uv run python studies/embedded_battery_forecast/forecast_report.py
uv run python studies/embedded_battery_forecast/forecast_charts.py
```
