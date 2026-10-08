# Plan: how well can an aggregate's half-hourly flow reveal the MW and MWh of an unmetered battery?

**Question.** The maintainer asked: "How well can we detect the capacity (MW and MWh) of an
unmetered battery (or fleet of unmetered batteries) from an aggregate signal, such as from the half
hourly data of a primary substation?" The study answers three measurable parts for every battery
size from a domestic fleet up to a merchant battery. The first part is how often a detector flags a
battery that is not there (the false-alarm rate). The second is how small a battery the detector
still finds. The third is the posterior distribution of the battery's power (MW) and energy
capacity (MWh), and whether its credible intervals hold the truth as often as they claim. Every
part is measured first on built test sums with a known battery, then on NGED flows.

**The prior study shows why the question is hard.** In
[battery-pv-separation](https://openclimatefix.github.io/nged-substation-forecast/studies/battery-pv-separation/),
the residual of the joint solar-and-battery linear programme fell as the assumed battery power grew,
by the same amount for real aggregate output and for calendar replicas with no battery in them
(rung 7: 0.29 and 0.28 of the no-battery residual at 50 MW). A battery schedule that is free to take
any shape absorbs demand noise as readily as a real battery's swings. This study therefore gives
the battery no free shape: each battery's schedule is simulated from a public signal that a real
battery follows, such as a tariff window or a price, and only a few physical parameters are
inferred.

## Size, by the five triggers

- Changes what gets stored: yes. A published page, a results folder, and new code in
  `packages/studies/`.
- Touches the production serving path: no.
- Touches a degradation rule: no.
- Admits more than one defensible design: yes. The signal set, the priors, and the inference method
  each have defensible alternatives.
- Spans code whose callers cannot be named without searching: no. The functions moved into
  `packages/studies/` are called only by `studies/battery_pv_separation/`, named below.

So the study is complex. It gets both plan reviews, both diff reviews (the mutation pass included,
because `packages/studies/` changes), two Opus scientific-validity reviews, and an Opus prose review
with persona and evidence reviews.

## What the prior study settled, and this study does not repeat

- How four public batteries respond to the day-ahead price, and that the price rank adds nothing to
  the time of day for NGED battery A (rungs 1 and 2).
- That a state of charge integrated over a whole year is inflated by drift: 5.0 to 9.0 hours at p99
  power for a year fit, against 2.7 to 5.1 hours for 3-month blocks (rung 3).
- That free price regressors recover about a fifth of a battery's size and gain much of their fit
  from generic daily structure (rungs 4 and 5).
- How well the joint linear programme recovers solar, and how its costs move the fitted solar
  (rungs 6, 7, 7b, and 7c). The rung 7c known-answer test scores the fitted solar against an assumed
  battery; this study reuses its construction of sums with a demand-like part, not its question.

## The signals: which ones the estimator uses, and which are only controls

**The estimator uses only signals that are public, free, known before delivery, and shared by many
batteries, because a shared signal is what makes a battery or a fleet visible in an aggregate.** The
signal list comes from the research note `prose_out/small_battery_signals.md`. Its judgements of how
much flow each signal explains are reasoned, not measured, and the claims marked unverified below
stay unverified until the study checks them.

| Signal | Role | Batteries it schedules | What must be obtained |
|---|---|---|---|
| Fixed tariff window edges: Intelligent Octopus Go cheap 23:30 to 05:30, Octopus Go 00:30 to 05:30, Octopus Flux cheap 02:00 to 05:00 with high export 16:00 to 19:00 | Estimator template | Domestic fleets, which step together at the window minute | Nothing; the times are constants (window times as published on the tariff pages) |
| Octopus Agile half-hourly import price, East Midlands region | Estimator template | Domestic batteries on Agile, moving together within a region | A download from Octopus's REST API (below) |
| N2EX day-ahead price | Estimator template | Merchant batteries connected to the distribution network | Already on disk for September 2025 to August 2026 |
| Distribution-charge red band, weekdays 16:00 to 19:00 | Estimator template | Commercial and industrial batteries behind the meter, which avoid importing in the red band | Nothing; a constant. The band for NGED's East Midlands licence area is unverified, and the study checks it in NGED's published charging statement before the first run |

**Grid frequency and Demand Flexibility Service events are left out of the study altogether.** In
the Elexon download for September 2025 to September 2026, the half-hourly minimum frequency fell
below the 49.7 Hz static-response trigger in 6 half-hours on 4 days, after faulty readings are
dropped, and Demand Flexibility Service events fall on tens of days a year. Neither signal acts
often enough to size a battery, so neither would change a conclusion, and leaving both out removes
a download, a simulated fleet, and a figure.

**Electric-vehicle chargers on the same tariffs step at the same window edges.** A charge edge alone
therefore cannot separate a battery fleet from an electric-vehicle fleet. Every battery template is
energy-neutral: a charge block and a discharge block whose energies differ by the round-trip
efficiency. The estimator also carries a charge-only nuisance template at each window edge, so an
electric-vehicle step is not credited to a battery. A battery is credited only where the matching
discharge is found.

**Downloads.** One small download is needed: Octopus Agile import rates for the East Midlands region
(region letter B in Octopus's tariff codes; all 8 primaries metered in MW are in NGED's East
Midlands licence area), for every Agile product version covering September 2025 to August 2026,
through the public `standard-unit-rates` endpoint: about 17,500 values in a few dozen requests. The
endpoint needs no account, key, or payment. The fetch script records the terms page at fetch time
and stops for the maintainer if the terms require accepting a licence. If the download stops, the
Agile template is dropped and the domestic fleet uses the three fixed-window tariffs only.

**The study year is September 2025 to August 2026 only.** The N2EX prices on disk, the B1610
battery outputs, and `window_half_hours` all cover that year. An earlier era was considered for the
primary screen and cut: only 4 of the 8 primaries metered in MW have data from 2019, and the era
needs a second N2EX download and a second time grid.

## The estimator: a template posterior

**One estimator gives the posterior of MW and MWh; a model-free step statistic is its comparator.**

**The template posterior.** Within a 3-month block, the aggregate is modelled as

```text
y(t) = calendar baseline + fleet-curve solar - sum over classes c of a_c * template_c(t; d_c, eta) + noise
```

where `a_c >= 0` is the power of battery class `c` in MW, and `template_c` is a one-megawatt
schedule, positive for export, simulated from that class's signal for a duration `d_c` in hours
and a round-trip efficiency `eta` (passed to `lp_schedule` as `eta_one_way = sqrt(eta)`).
The three battery classes are:

- **Merchant**: `studies.battery_dispatch.lp_schedule` on the N2EX day-ahead price.
- **Domestic fleet**: one template per tariff (Intelligent Octopus Go, Octopus Go, Flux, and
  Agile), each with its own non-negative power, so the tariff membership shares are absorbed into
  the linear powers and need no prior of their own. All four share one duration. Each template
  charges from its window start at full power until full and discharges across the evening (16:00
  to 23:30) until empty, except Flux, which discharges from 16:00 to 19:00; the Agile template is a
  price-taker on each 23:00-to-23:00 Agile delivery day, whose prices are published the afternoon
  before. Each template is the mean over a fixed log-normal spread of durations (standard deviation
  0.25 in natural log), which softens the end of the charge.
- **Commercial and industrial**: discharge across the weekday red band, recharge overnight.

The baseline (`studies.pv_separation.baseline_design`, `flexibility="daily"` within a block) and
the four fleet curves (`studies.pv_separation.solar_basis`) enter linearly, as do the class powers
and the charge-only nuisance templates, so for a fixed set of durations the model is
linear-Gaussian.

**Power, duration, and round-trip efficiency each get an explicit prior and a posterior; the
state-of-charge limits and the cycle cap are fixed, and a sensitivity arm moves them.** The priors
come from the prior study and the research note; the domestic durations are unverified.

| Parameter | Prior | Why this prior |
|---|---|---|
| Power of each class or tariff (MW) | Half-normal, scale 20% of the aggregate's p99 | Non-negative, and puts prior mass on every share from 0.5% to 40% without favouring any |
| Duration (MWh per MW), merchant | Log-normal, median 2 hours, 95% between 1 and 4 hours | Distribution-connected merchant batteries are mostly 1 to 2 hours, with newer ones longer |
| Duration, domestic fleet | Log-normal, median 2 hours, 95% between 1.2 and 3.5 hours | Home products of 5 to 13.5 kWh at 3 to 5 kW (unverified) |
| Duration, commercial and industrial | Log-normal, median 1.5 hours, 95% between 1 and 3 hours | A red band of 3 hours needs at most 3 hours, and most sites cover part of it |
| Round-trip efficiency, shared by all classes | Beta, mean 0.87, 95% between 0.80 and 0.92 | The prior study fitted one-way 0.92 to 0.94 (round trip 0.85 to 0.88) on public batteries; home systems with an inverter each way are likely lower |

**Two physical parameters stay fixed, because each only rescales what the grid already infers.**
The state-of-charge limits (5% and 95%) set the usable share of the energy capacity, which trades
one for one against the duration, so the duration posterior is read as usable hours at full power,
and the page says so. The cycle cap of 1 a day (`lp_schedule`'s `cycles_per_day_cap`, which stands
in for a throughput cost; `lp_schedule` has no cost argument) only bites on days with two price
peaks. Neither would change a verdict unless the posterior for MWh moved by more than its own
interval width, and the second setting tests that: it re-runs every planned contrast with the
limits at 0% and 100% and a cap of 2 cycles a day.

**Inference: an exact grid over duration and efficiency, exact integration over power.** Each
class's duration takes 6 log-spaced grid values from 0.5 to 6 hours (0.5, 0.82, 1.35, 2.2, 3.6, and
6.0), and the shared round-trip efficiency takes 4 values (0.78, 0.83, 0.88, and 0.93), each grid
point weighted by its prior. The posterior is therefore a sum over 6 x 6 x 6 x 4 = 864 combinations
and needs no sampling over duration or efficiency. The powers enter the model linearly, so their
integral is exact rather than gridded, which is the limit of an infinitely fine power grid. The
templates do not depend on the aggregate, so each block's 144 template columns (36 columns for each
efficiency: 6 durations for the merchant class, 6 for commercial and industrial, and 6 times four
tariff columns for the domestic class) and their projections off the baseline and solar columns are
computed once and shared by every sum in that block. For each sum and combination, the baseline,
solar, and nuisance coefficients are integrated out analytically under a wide Gaussian prior. The
class powers have a half-normal prior, so the combination's evidence is the unconstrained Gaussian
evidence times the posterior probability of the positive orthant over the prior's
(`scipy.stats.multivariate_normal.cdf`, at most 6 dimensions). The class powers are then sampled
from their truncated Gaussian conditional (Gibbs sampling with `scipy.stats.truncnorm`) for the
combinations holding 99% of the posterior mass. The noise is serially correlated, so the residual is
prewhitened with a first-order autoregression fitted to the scored aggregate's own residual from the
highest-evidence combination, iterated twice; the noise parameters carry no truth, so estimating
them from the scored sum is not leakage. The report gives, for every sum, the marginal posterior of
each class's MW, of its MWh (`a_c * duration_c`), and of the round-trip efficiency, with 50% and 90%
credible intervals. All of this is numpy and scipy, and no posterior can collapse onto a few
importance weights.

**The detection statistic** is the log Bayes factor of the model with batteries against the model
with none. The threshold is the 95th percentile of the statistic over the blocks of the other demand
series with no added battery, so its nominal false-alarm rate is 5%.

**No real primary is known to be battery-free, so the threshold detects an added battery, not a
battery.** Each primary may already hold domestic and commercial batteries, which is the question
itself. In rungs 1 to 3 the threshold therefore answers "is there a battery beyond what the series
already holds", which is what the known-answer sums test. The rung 5 screen cannot use that
threshold to claim a primary holds a battery, and uses a within-series placebo instead (rung 5).

**The comparator: the step tail.** Fit the same baseline and solar by least squares, take the
residual's half-hour changes `dr`, and compute `P_step = max(0, (q99.5(|dr|) - 2.81 * sigma) / 2)`,
with `sigma = 1.4826 * MAD(dr)`. A battery switching from full charge to full discharge adds a step
of up to twice its power, and 2.81 is the 99.5th percentile of the absolute value of a standard
normal variable. The step tail uses no signal and no prior, and gives no MWh.

**What the posterior can and cannot mean.** The credible intervals are conditional on the template
family. A battery driven by a signal outside the table, such as a private aggregator's dispatch, is
not described, and the calibration check below measures how much that matters for the known-answer
batteries only.

## Definitions

- **Aggregate**: `y(t) = demand(t) + solar_flow(t) - battery(t)`, positive for import, battery
  positive for export, on the half-hour grid of `battery_synthetic.window_half_hours` (September
  2025 to August 2026, UTC). That grid is redefined in `capacity_inputs.py`, because a study script
  may not import another study's folder.
- **Battery share**: the battery's or fleet's power as a percentage of the demand series' p99
  absolute flow. Shares run 0, 0.5, 1, 2, 5, 10, 20, and 40%, so a fleet of about
  ten 5 kW home batteries on a 10 MW primary sits at the 0.5% end.
- **Noise unit**: `sigma_step`, the robust standard deviation of the half-hour changes of the
  demand series' residual with no added battery. Sizes are reported as `P / sigma_step` as well as
  shares, so NGED can apply the result to any primary from that primary's own noise.
- **Power error**: `|P_hat - P| / P`, with `P_hat` the posterior median, for scoring the contrasts.
  The step tail returns exactly 0 for a small battery, so a log error would be minus infinity there.
  Charts show `log2(P_hat / P)` with `P_hat` floored at 1% of `P`, and say so. **Energy error** is
  the same with MWh.
- **Blocks**: every fit is per 3-month block (September to November, December to February, March to
  May, June to August).

## Planned contrasts (fixed before any result)

- **C1, false alarms.** On the 44 blocks (11 demand series, 4 blocks each) with no added battery,
  each scored against the threshold built from the other 10 series, the log Bayes factor flags a
  share of blocks not significantly above 5%: a one-sided exact binomial test of the flag count
  against 5% does not reject at the 5% level. The count and its Clopper-Pearson interval are
  reported.
- **C2, calibration.** On the simulated known-answer sums (rungs 1 and 2), the 90% credible interval
  for power holds the truth in at least 80% of sums, and the 50% interval in 40% to 60%, judged on
  the point coverage. The same coverages are reported for energy.
- **C3, posterior against step tail.** At shares of 5% and above in rung 1, the mean of the paired
  difference in power error (posterior median minus step tail) is below zero, with its 95%
  cluster-bootstrap interval excluding zero.
- **C4, energy separately from power.** For rung 1 batteries of 1 and 4 hours at shares of 10% and
  above, the mean of the paired difference in energy error (posterior median minus the rule
  `2 hours * P_hat`) is below zero, with its 95% interval excluding zero. The ratio of the posterior
  width of the duration to its prior width is reported beside C4, because a ratio near 1 means the
  aggregate taught the estimator nothing about duration.
- **C5, fleets of real batteries.** On rung 3's fleets of public batteries, the mean of the paired
  difference in `|log2(P_hat / peak)| - |log2(P_hat / registered sum)|` is below zero, with its 95%
  interval excluding zero, where the peak is the fleet's coincident peak (the p99 of the summed
  output) and the registered sum is the sum of the registered generation capacities.

Every other number is exploratory, and anything added after the first results is post hoc.

**Folds.** The detection threshold for a scored block comes only from the other demand series, so
no series' own noise sets its own threshold. The noise parameters of a sum are estimated from that
sum, because they carry no truth. **Intervals** on errors, coverages, and rates resample whole
demand series and whole batteries (a two-level cluster bootstrap, 2,000 resamples), as the prior
study's rungs 4 to 6 resampled whole aggregates. A per-block posterior is not a per-month score, so
the month-resampled default does not apply, and the page says why. Rates also carry Clopper-Pearson
intervals with the count of blocks beside them. **Second setting**: every planned contrast is
re-run with the duration priors' log-standard-deviations doubled, the efficiency prior's 95% range
widened to 0.70 to 0.95, the state-of-charge limits at 0% and 100%, and a cap of 2 cycles a day, so
a verdict that rests on those choices shows.

**Controls.** The positive control is a simulated 2-hour merchant battery at a 40% share on the
calendar replica of a demand half; the 90% interval must hold `P` and `E`, and the posterior median
must lie within 10% of each. If the positive control fails, nothing else is scored. The negative
controls are the blocks with no added battery; tariff templates shifted one hour early, which a
real window-edge signal should beat; and Agile prices from 7 days earlier.

**The simulated truth is never drawn from the estimator's own template family.** Rung 1's merchant
batteries are dispatched by `lp_schedule` with the one-way efficiency drawn uniformly from 0.88 to
0.95 (round trip 0.77 to 0.90, mostly between grid values), the state-of-charge limits from 0% to
10% and 90% to 100%, a cap of 1 or 2 cycles a day, and durations of 1, 2, and 4 hours, which lie
between the estimator's grid values, while the estimator keeps its fixed limits and cap. A
calibration that holds only when the truth is a template would otherwise pass C2 for free.

## The ladder (five rungs)

**Rung 1: a simulated battery with an exact answer.** Merchant schedules from `lp_schedule` on the
N2EX price, with durations of 1, 2, and 4 hours, subtracted from each demand half at every share.
Demand halves: the 8 NGED primaries metered in MW and the 3 public demand-like BMUs of rung 7c
(`DEMAND_BMUS` in `battery_rung7c_known_answer.py`), 11 series in all. Calendar replicas are used
for the positive control only: a replica is a monthly mean with almost no noise, so scoring it
beside real demand would flatter every detection rate. No solar arm is added: the primaries already
carry their own embedded solar, and the model carries the four solar columns. This rung holds the
positive control and answers C1, C3, and C4, and C2 for merchant batteries. Rung 1 is
11 series x 4 blocks x 3 durations x 7 shares = 924 sums, plus the 44 blocks with no added
battery.

**Rung 2: a fleet of small batteries that move as one (simulated).** No ground truth exists for any
real domestic fleet, so this rung is simulated, and the page says so in its first sentence. Each
fleet is built home by home: a number of homes, a 3 to 5 kW battery per home with a duration drawn
from the domestic prior, and a tariff for each home drawn from known membership shares. Fleet sizes
run over the same shares as rung 1, on the same 11 series and 4 blocks, with one fleet draw per
share. The simulation's truth is exact; its realism is not checked against any real fleet, so rung 2
shows what the estimator can recover if domestic batteries behave as the templates assume. The
simulator draws its parameters from distributions centred away from the priors' medians, so a fleet
the priors describe badly is part of the test. This rung answers C2 and C4 for fleets and gives the
detection curve for domestic sizes.

**Rung 3: real public batteries, alone and in fleets.** The four batteries of the prior study, four
of the smallest `battery_hint` BMUs in the census list with at least 95% of half-hours present, and
fleets of 2, 4, and 8 of the 101 `battery_hint` BMUs in `bmu_list.csv` (all 101 have B1610 output on
disk; fixed seed, 5 draws per size), added to the demand halves at shares of 5%, 10%, 20%, and 40%.
That is 23 batteries or fleets x 11 series x 4 blocks x 4 shares = 4,048 sums. Power truth is the
registered generation capacity, with the metered p99 beside it. No public energy capacity exists, so
the energy reference is the prior study's 3-month-block fit (`smallest_capacity`, moved from
`battery_rung3.py`) applied to the battery's own metered output. This rung answers C5, and tests
whether real dispatch leaves the calibration of C2 intact.

**Rung 4: NGED battery A inside a BSP flow (a real known answer).** A pre-plan check found that NGED
battery A's half-hour changes correlate with one bulk supply point's (BSP's) raw flow (correlation
-0.28 over September 2025 to August 2026, negative as expected when export reduces import), and at
0.07 or below with every primary and both grid supply points. NGED battery A's p99 output is a few
percent of that BSP flow's p99. The BSP flow starts after NGED battery A's first sizeable reading,
so no before-and-after test exists. The rung scores the BSP flow as it is, and with NGED battery A's
metered output added back (a matched null for the same BSP), and with 1, 3, and 10 further copies of
NGED battery A subtracted to find the multiple at which the detector flags it. The correlation is
evidence of connection, not a network-topology record, and the page says so. Exploratory, one site.

**Rung 5: the screen of the 8 NGED primaries metered in MW, reported per primary.** Each primary,
labelled S1 to S8, gets the posterior of each battery class's MW and MWh for the study year. The
threshold of rungs 1 to 3 is built from the other primaries, which may hold batteries too, so the
screen's evidence is a within-series placebo instead: each primary's log Bayes factor for the real
templates is ranked against its log Bayes factors for 12 placebo template sets on the same primary
(tariff windows shifted by -3, -2, -1, +1, +2, and +3 hours, and N2EX and Agile prices taken from 6
other weeks). A real battery class should beat every placebo; a primary whose real templates rank
first of 13 is reported as showing evidence, and the page states that 1 in 13 would rank first by
chance. NGED's Embedded Capacity Register (August 2026,
on disk) gives a partial answer key for connected storage of 50 kW and above: 2 of the 8 primaries
have a connected storage entry. The register lists no storage capacity in MWh and no duration for
any of its storage rows, so it can check presence and MW size class, never MWh. Below 50 kW the
register is silent, so a domestic fleet is never in it. The page reports, per primary, whether the
register lists connected storage, and never prints a generator's name or registered capacity, nor
pairs a primary's posterior with a register entry's capacity.

**NGED's primary series are labelled "Disaggregated Demand", read by the plan as demand with
metered generation subtracted.** The inputs step checks NGED's definition of that label before the
first run, because a series with estimated embedded generation added back would change what the
solar columns and the battery templates mean. A battery that NGED meters
may therefore be absent from a primary's flow already, and what remains is the unmetered battery
the question asks about. Primaries metered in MVA are out of scope, because MVA has no sign.

**Each rung is reported only on the evidence below it.** If the positive control fails, nothing is
scored. If rung 1 finds nothing at a 40% share on real demand, rungs 2 to 5 are reported as a
demonstration that the method fails.

**Compute budget.** Each block's template columns and projections are computed once, so a sum costs
144 inner products and 864 small evidence evaluations, each with an orthant probability of at most
6 dimensions (`multivariate_normal.cdf` with `maxpts=2000`, since scipy's default is a million
points per dimension). A timing on this workstation at `OMP_NUM_THREADS=2` put one 6-dimensional
orthant probability at 1.8 ms with `maxpts=2000`, so a sum takes about 1.6 seconds. Rungs 1 to 3
hold about 5,300 sums (924, 308, and 4,048), each run at both settings, which is about 4.7 hours of
one core, or about 1.2 hours on 4 workers. The report records the first full sum's timing and the
grid size; if rungs 1 to 3 would exceed 2 hours, the cut order in the work breakdown applies.

## What counts as "not identifiable", and what it would mean for NGED

**Power is not identifiable at a size** where the detector finds fewer than half of the batteries at
the 5% false-alarm threshold, or the 90% credible interval for power spans more than a factor of 4.
**Energy is not identified separately from power** where the duration's posterior width is more than
0.7 of its prior width and C4 fails. **A fleet's power is identified only as its coincident peak**
if C5 holds; the coincident peak is what matters at the network's peak, so that outcome is not a
shortfall. **Every credible interval is untrustworthy** wherever C2 fails, and the page then reports
coverage instead of intervals.

**If small batteries are not identifiable, a primary's half-hourly flow cannot inventory domestic
and commercial batteries.** NGED would need connection records, which the register's 50 kW floor and
missing MWh limit, or metering, and a forecast would treat unregistered batteries as uncertainty
near the evening peak and at the tariff window edges. **If they are identifiable above some size**,
the page gives that size in units of `sigma_step`, so NGED can compute the smallest detectable
battery at each of its primaries.

## The page

**The page opens with the conclusions and one headline figure; the rest is figures, each with a
few sentences.**

- **Conclusions** at the top: at most five bolded sentences, one each for false alarms, the
  smallest detectable merchant battery, the smallest detectable domestic fleet, whether MWh is
  identified separately from MW, and what the primary screen found.
- **One sentence on frequency** beside the signal list: grid frequency was reviewed, and its
  half-hourly mean barely moves, so the estimator does not use it.
- **Headline figure**: detection probability against battery size in `P / sigma_step` (log axis,
  0.5% to 40% shares marked), one line each for a merchant battery, a domestic fleet, and a real
  public battery, with the 5% false-alarm level; beneath it, on the same axis, the median width of
  the 90% credible interval for power.

**The figures run in three groups: what the data looks like, then known-answer cases with the true
value drawn on every panel, then the harder cases.** A case where the technique fails gets a figure
as plain as a case where it works: the same axes, the truth marked, and a title that says it failed.
Charts holding an NGED primary show it by label (S1 to S8); charts holding NGED battery A show it
normalised by its p99 with days numbered 1 to 7 and no dates.

## Figure list

| Figure | What it shows | Rung | Message |
|---|---|---|---|
| 1 (headline) | Detection probability and median 90% interval width for power against `P / sigma_step`, for a merchant battery, a domestic fleet, and a real public battery, with the 5% false-alarm level | 1 to 3 | A battery is found reliably above a stated size, and below that size the estimator says it cannot tell. |
| 2 | Two weeks of each of the 8 primaries metered in MW, in MW, one winter and one summer | Data | Primary flows differ in level, noise, and daily shape, and the noise sets what can be detected. |
| 3 | Mean daily profile of each primary, weekdays against weekends, with 16:00 to 19:00 and the tariff window starts marked | Data | Any battery signal has to show against a daily shape far larger than the battery. |
| 4 | Each primary's mean half-hour change by time of day, with the window edges (23:30, 00:30, 02:00, 05:30, 16:00, 19:00) marked | Data | Window edges appear, or fail to appear, as steps in the primaries' own data. |
| 5 | One week of the N2EX day-ahead price and the East Midlands Agile price | Data | The two prices share a shape, and Agile's cheap slots move day to day. |
| 6 | One week of a public demand-like BMU with and without a public battery subtracted, in MW, with the battery beneath | Data | A known battery is visible by eye when large and invisible when small. |
| 7 | NGED battery A's output beneath its BSP's flow, both normalised by p99, with the flow after NGED battery A is added back | Data, rung 4 | A real battery of a few percent of a BSP flow is hard to see by eye. |
| 8 | A simulated domestic fleet's output for three nights, home by home and summed, beside the templates | Data, rung 2 | A fleet on one tariff steps at the window minute, and a spread of durations blurs the end of the charge. |
| 9 | The positive control: posterior of MW and of MWh, with the truth marked | 1 | The estimator recovers an easy battery, so later failures belong to the data, not to the code. |
| 10 | Posterior of MW against MWh for three rung 1 sums (40%, 10%, and 2% shares), truth marked | 1 | Large batteries give tight posteriors around the truth, and small ones give posteriors no narrower than the prior. |
| 11 | Posterior median against truth for power and energy, log axes, coloured by share, with 90% intervals | 1, 2 | Estimates track the truth above a size and scatter below it. |
| 12 | Coverage of the 50% and 90% intervals for power and energy, by rung and share, against the nominal lines (C2) | 1, 2, 3 | The intervals mean what they say, or the figure shows where they are too narrow. |
| 13 | The distribution of the log Bayes factor over the blocks with no added battery, with the threshold, and the false-alarm rate per series (C1) | 1 | The detector flags about 5% of blocks with no added battery, as designed. |
| 14 | Paired contrasts C3 and C4 with intervals, and the duration's posterior width over its prior width | 1, 2 | The posterior beats the step statistic, and MWh is or is not learned beyond the prior. |
| 15 | Detection probability against fleet size for the simulated domestic fleet, with the 1-hour-shifted templates as a negative control | 2 | A fleet is found from its window edge, and the shifted window finds nothing. |
| 16 | Real public batteries: posterior against registered power and against the energy reference, beside the simulated batteries at the same shares | 3 | Real dispatch is harder to size than a simulated schedule, by the amount shown. |
| 17 | Fleets of real batteries: posterior median against coincident peak and against the sum of registered powers (C5) | 3 | A fleet is seen as its coincident peak. |
| 18 | NGED battery A in its BSP: the log Bayes factor and the MW posterior against the multiple of NGED battery A (0 to 11), with the threshold | 4 | A real battery of this share is or is not detected, and the multiple at which detection starts. |
| 19 | The primary screen: each primary's log Bayes factor for the real templates beside its 12 placebo template sets, and its MW posterior per battery class, with the register's storage presence marked | 5 | Which primaries show evidence of a battery beyond the placebos, and how that agrees with the register. |

## Work breakdown for a Sonnet agent

Run with `OMP_NUM_THREADS=2` and at most 4 workers. Every script gets a fresh code review before its
first run. Results go to `data/studies/per_study/unmetered_battery_capacity/`.

1. **Branch**: work on `unmetered-battery-capacity`, which already holds the merged
   battery-pv-separation code (pull request 1100).
2. **Download** `studies/market_downloads/fetch_agile.py`, following the `data-download` skill. Run
   the `data-validation` checklist on it. Check the red band in NGED's East Midlands charging
   statement, and record it in the report.
3. **Move and add code in `packages/studies/`, with tests.** New module
   `studies/battery_templates.py`: the tariff, Agile, red-band, and charge-only templates, and the
   home-by-home fleet simulator. New module `studies/template_posterior.py`: prewhitening, the
   linear-Gaussian marginal likelihood with the orthant correction, the evidence over the duration
   grid, the truncated Gibbs step, and the log Bayes factor. Move `calendar_replica` (from
   `battery_rung7.py`, taking the half-hour grid as an argument, because `window_half_hours` lives
   in the study folder) and `smallest_capacity` with `cell_energy_path` (from `battery_rung3.py`)
   into `studies/battery_capacity.py`. Repoint every caller: `battery_rung3.py`, `battery_rung7.py`,
   `nged_battery_a_rungs.py`, and `nged_battery_a_charts.py`. Re-run `battery_rung3.py` and
   `nged_battery_a_rungs.py` to show their outputs unchanged bit for bit. The duplicate
   `calendar_replica` in `studies/solar_disaggregation/stage3b_real_controls.py` is out of scope and
   stays.
4. **Inputs** `studies/unmetered_battery_capacity/capacity_inputs.py`: the 8 primaries metered in
   MW, the BSP flow (found by the correlation check, recomputed in code), and NGED battery A, read
   by metadata filter; the register's per-primary storage presence, read from `literature/NGED/NGED
   ECR AUG 2026.xlsx` in the main checkout (neither `fastexcel` nor `openpyxl` is installed, so add
   one to the `studies` package's dependencies, or convert the sheet to CSV once and record how);
   NGED's definition of "Disaggregated Demand"; data-validation checks into `report_inputs.md`.
   Nothing that identifies a generator is written.
5. **Templates** `capacity_templates.py`: every block's 144 template columns, for both settings,
   saved. Time one sum's posterior and record it with the grid size; if rungs 1 to 3 would take more
   than 2 hours, cut in this order and record the cut: rung 3's fleet draws from 5 to 3; rung 3's 5%
   share; the efficiency grid from 4 to 3 values.
6. **Positive control**, then **nulls and thresholds** (`capacity_nulls.py`).
7. **Rungs 1 to 5**, one script each, saving every posterior's grid weights and power draws so a
   chart needs no refit.
8. **Report** `capacity_report.py`: every table the page quotes, C1 to C5 at both settings,
   coverage, and the controls.
9. **First Opus science review**, triage, re-run what it asks. **Charts and page**
   (`docs/studies/unmetered-battery-capacity.md`, beside the battery page in `mkdocs.yml`), then the
   **second Opus science review**, the diff reviews, the mutation pass, and the prose, persona (an
   NGED planning engineer, a home-battery optimiser developer, a sceptical Bayesian statistician),
   and evidence reviews.

## Tests (each would fail on `main` today)

- Tariff templates: charge starts exactly at each window's first half-hour, and each template's
  discharged energy equals its charged energy times the round-trip efficiency.
- Charge-only nuisance template: has no discharge half-hour.
- Fleet simulator: the fleet sum equals the sum of the homes, and a fleet with no duration spread
  reproduces a single template times the home count.
- Marginal likelihood: matches a brute-force numerical integral on a 3-column problem.
- Truncated Gibbs step: never returns a negative power, and recovers a known posterior mean on a
  conjugate case.
- Orthant correction: with one power column, the corrected evidence matches a brute-force integral
  over the half-normal prior.
- Duration grid: on a noise-free sum built from a grid template, the posterior puts most of its
  mass on that template's duration.
- Moved functions: pinned by a fixture before the move, with unequal values across half-hours.

## Design-philosophy check

Study code is R&D: every script fails fast on a missing input, nothing runs in production, and no
asset check changes. The study informs the disaggregation roadmap's battery component, which the
roadmap ranks "very poor" for physics-based disaggregation, and commits the project to nothing.

## Verification commands

```bash
uv run ruff check studies/unmetered_battery_capacity studies/market_downloads packages/studies --fix
uv run ruff format studies/unmetered_battery_capacity studies/market_downloads packages/studies
uv run ty check
uv run pytest --run-studies -n auto packages/studies
uv run pymarkdown scan -r docs/studies/unmetered-battery-capacity.md
uv run mkdocs build --strict
```

Plus `pydoclint` and the docs-link checker, as CI runs them.

## What the plan review changed

One reviewer ran both lenses, simplicity then correctness, and edited the plan in place.

- **Cut**: the grid-frequency signal, the fleet answering the 49.7 Hz trigger, and its figure; the
  Demand Flexibility Service download and control; the September 2019 to August 2020 era, with its
  N2EX download (only 4 of the 8 primaries have data that early); rung 1's solar arm and its
  calendar-replica halves; and rung 3's 2% share and half its fleet draws.
- **Replaced**: importance sampling over 4,000 prior draws of about a dozen parameters with an exact
  864-point grid over the three durations and a shared round-trip efficiency, with power integrated
  exactly. With some 4,400 half-hours per block the likelihood is sharp enough that a few importance
  weights would carry the whole posterior, and the planned fallback still left about eight
  dimensions. Tariff membership shares became per-tariff linear powers. The state-of-charge limits
  and the cycle cap are fixed, with a sensitivity arm in the second setting.
- **Corrected**: the battery sign in the model; the count of `battery_hint` BMUs (101, not 149); the
  throughput-cost parameter, which `lp_schedule` does not have; the noise parameters, which now
  come from the scored sum rather than from other series; a power error that was minus infinity
  whenever the step tail returns 0; pass rules for C1, C3, C4, and C5, which had none; simulated
  truths drawn from the estimator's own template family, which would pass C2 for free; the rung 5
  screen, whose threshold assumed the other primaries hold no battery, now a within-series placebo;
  the callers of the moved functions; and the import of `window_half_hours`, which crosses a study
  boundary.

**Left risky.** The first-order autoregression may not whiten half-hourly demand residuals, in
which case C2 fails and the page reports coverage instead of intervals. Whether NGED's
"Disaggregated Demand" has embedded generation added back is unchecked. The register needs an Excel
reader that is not installed. The compute budget rests on an orthant probability costing a few
milliseconds, which the first timed sum must confirm.

## Open questions for the maintainer

None. The maintainer has settled the priors, the per-primary publication, the size range, the MVA
scope, and the register's use. The download step stops for the maintainer only if a provider's terms
turn out to require accepting a licence.
