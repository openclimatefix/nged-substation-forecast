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
| Fixed tariff window edges: Intelligent Octopus Go cheap 23:30 to 05:30, Octopus Go 00:30 to 05:30, Octopus Flux cheap 02:00 to 05:00 with high export 16:00 to 19:00 | Estimator template | Domestic fleets, which step together at the window minute | Nothing; the times are constants (window times as published on the tariff pages, unverified for 2019 to 2020) |
| Octopus Agile half-hourly import price, East Midlands region | Estimator template | Domestic batteries on Agile, moving together within a region | A download from Octopus's REST API (below) |
| N2EX day-ahead price | Estimator template | Merchant batteries connected to the distribution network | Already on disk for September 2025 onwards; 2019 to 2020 needs a download |
| Distribution-charge red band, weekdays 16:00 to 19:00 | Estimator template | Commercial and industrial batteries behind the meter, which avoid importing in the red band | Nothing; a constant. The band for NGED's East Midlands licence area is unverified, and the study checks it in NGED's published charging statement before the first run |
| Demand Flexibility Service event half-hours | Control and explanation only | Enrolled homes and sites, on tens of event days a year | NESO's utilisation report, by grid supply point (a download) |
| Grid frequency, including the 49.7 Hz static-response trigger | Explanation only | Batteries holding frequency response | Already on disk (Elexon, 15-second values) |

**Frequency stays out of the estimator because the half-hourly mean barely moves.** In the Elexon
download for September 2025 to September 2026, the half-hourly mean frequency has a standard
deviation of 0.058 Hz. After 188 half-hours with samples at or below 45 Hz are dropped as faulty
readings, the minimum fell below 49.7 Hz in 6 half-hours on 4 days. A fleet answering the static
trigger therefore acts a handful of times a year, which no estimator can size; the study reports
the aggregate's change in those half-hours as an explanation, and does not claim it as evidence.
Whether static response is still procured from domestic batteries in 2026 is unverified.

**Demand Flexibility Service events stay out of the estimator because they are rare and the
response is not a battery's alone.** The study checks, as an exploratory control, whether the
fitted battery's output steps in the event half-hours.

**Electric-vehicle chargers on the same tariffs step at the same window edges.** A charge edge alone
therefore cannot separate a battery fleet from an electric-vehicle fleet. Every battery template is
energy-neutral: a charge block and a discharge block whose energies differ by the round-trip
efficiency. The estimator also carries a charge-only nuisance template at each window edge, so an
electric-vehicle step is not credited to a battery. A battery is credited only where the matching
discharge is found.

**Downloads.** Three small downloads are needed, each free, with no account, no key, no cost, and
no licence acceptance, according to the research note. The fetch script records each provider's
terms page at fetch time, and stops for the maintainer if the terms turn out to require acceptance:

- Octopus Agile import rates for the East Midlands region (region letter B in Octopus's tariff
  codes, unverified), for every Agile product version covering September 2019 to August 2020 and
  September 2025 to August 2026, through the `standard-unit-rates` endpoint: about 35,000 values
  in a few dozen requests.
- N2EX day-ahead prices for September 2019 to August 2020, from the same NESO dataset the prior
  study used.
- NESO's Demand Flexibility Service utilisation report, for the study year.

## The estimator: a template posterior

**One estimator gives the posterior of MW and MWh; a model-free step statistic is its comparator.**

**The template posterior.** Within a 3-month block, the aggregate is modelled as

    y(t) = calendar baseline + fleet-curve solar + sum over battery classes of a_c * template_c(t; theta) + noise

where `a_c >= 0` is the power of battery class `c` in MW, and `template_c` is a one-megawatt
schedule simulated from that class's signal. The three battery classes are:

- **Merchant**: `studies.battery_dispatch.lp_schedule` on the N2EX day-ahead price.
- **Domestic fleet**: a mixture of the four tariff templates (Intelligent Octopus Go, Octopus Go,
  Flux, and Agile), weighted by membership shares. Each template charges from its window start
  at full power until full and discharges across the evening (16:00 to 23:30) until empty, except
  Flux, which discharges from 16:00 to 19:00; the Agile template is a daily price-taker on the Agile
  import price. The fleet's spread of durations softens the edges.
- **Commercial and industrial**: discharge across the weekday red band, recharge overnight.

The baseline (`studies.pv_separation.baseline_design`, daily flexibility within a block) and the
four fleet curves (`studies.pv_separation.solar_basis`) enter linearly, as do the class powers
`a_c`, so for fixed physical parameters `theta` the model is linear-Gaussian. The charge-only
nuisance templates enter the same way.

**Priors on the physical parameters `theta`** (the values come from the prior study and the
research note, and the domestic durations are unverified):

| Parameter | Prior |
|---|---|
| One-way efficiency | Beta, centred on 0.93 (round trip about 0.87), 95% between 0.89 and 0.96; the prior study fitted 0.92 to 0.94 |
| Duration (MWh per MW), merchant | Log-normal, median 2 hours, 95% between 1 and 4 hours |
| Duration, domestic fleet | Log-normal, median 2 hours, 95% between 1.2 and 3.5 hours, for products of 5 to 13.5 kWh at 3 to 5 kW |
| Duration, commercial and industrial | Log-normal, median 1.5 hours, 95% between 1 and 3 hours |
| State-of-charge limits | Lower uniform on 0% to 10%, upper uniform on 90% to 100% |
| Throughput or degradation cost, merchant and Agile | Uniform on £0 to £20 per MWh |
| Spread of durations within a domestic fleet | Log-normal spread with standard deviation uniform on 0 to 0.4 (natural log) |
| Tariff membership shares | Dirichlet(1, 1, 1, 1) over the four tariffs |
| Class power `a_c` | Half-normal with scale 20% of the aggregate's p99 |

**Inference: importance sampling over `theta`, exact integration over the linear terms.** The
templates do not depend on the aggregate, so a library of 4,000 prior draws of `theta`, with each
draw's templates, is simulated once and shared by every sum. For each sum and draw, the baseline,
solar, and nuisance coefficients are integrated out analytically under a wide Gaussian prior, and
the class powers are sampled from their truncated Gaussian conditional (Gibbs sampling with
`scipy.stats.truncnorm`). The draw's weight is its marginal likelihood. The noise is serially
correlated, so the residual is prewhitened with a fitted first-order autoregression before the
likelihood is evaluated. MWh is the posterior of `a_c * duration_c`. All of this is numpy and
scipy; a draw costs a few small matrix updates, so 4,000 draws per sum run overnight on 4 workers.
The run reports the effective sample size of every posterior; where it falls below 200, the
fallback fixes the state-of-charge limits and the throughput cost at their prior medians and
re-runs, and the report records the fallback.

**The detection statistic** is the log Bayes factor of the model with batteries against the model
with none. The threshold is the 95th percentile of the statistic over battery-free training blocks,
so its nominal false-alarm rate is 5%.

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
  positive for export, on the half-hour grid of `battery_synthetic.window_half_hours`, UTC.
- **Battery share**: the battery's or fleet's power as a percentage of the battery-free
  aggregate's p99 absolute flow. Shares run 0, 0.5, 1, 2, 5, 10, 20, and 40%, so a fleet of about
  ten 5 kW home batteries on a 10 MW primary sits at the 0.5% end.
- **Noise unit**: `sigma_step`, the robust standard deviation of the half-hour changes of the
  battery-free aggregate's residual. Sizes are reported as `P / sigma_step` as well as shares, so
  NGED can apply the result to any primary from that primary's own noise.
- **Power error**: `log2(P_hat / P)` with `P_hat` the posterior median.
- **Blocks**: every fit is per 3-month block (September to November, December to February, March to
  May, June to August).

## Planned contrasts (fixed before any result)

- **C1, false alarms.** On held-out battery-free blocks of the study year, the log Bayes factor at its
  5% threshold flags at most 5% of blocks.
- **C2, calibration.** On the simulated known-answer sums (rungs 1 and 2), the 90% credible interval
  for power holds the truth in at least 80% of sums, and the 50% interval in 40% to 60%. The same
  coverages are reported for energy.
- **C3, posterior against step tail.** At shares of 5% and above, the posterior median's absolute
  power error is smaller than the step tail's (paired difference).
- **C4, energy separately from power.** For simulated batteries of 1 and 4 hours at shares of 10%
  and above, the posterior median energy is closer to the truth than the rule `2 hours * P_hat`.
  The ratio of the posterior width of the duration to its prior width is reported beside C4, because
  a ratio near 1 means the aggregate taught the estimator nothing about duration.
- **C5, fleets of real batteries.** On sums of several public batteries, the posterior median power
  is closer to the fleet's coincident peak (the p99 of the summed output) than to the sum of the
  registered powers.

Every other number is exploratory, and anything added after the first results is post hoc.

**Folds.** The null threshold, the autoregression, and any calibration for a scored block come only
from other demand series in other 3-month blocks, so no block's weather leaks into its own
calibration. **Intervals** on errors, coverages, and rates resample whole demand series and whole
batteries (a two-level cluster bootstrap, 2,000 resamples), as the prior study's rungs 4 to 6
resampled whole aggregates. A per-block posterior is not a per-month score, so the month-resampled
default does not apply, and the page says why. Rates also carry Clopper-Pearson intervals with the
count of blocks beside them. **Second setting**: every planned contrast is re-run with priors twice
as wide on every duration and on efficiency, so a verdict that rests on the priors shows.

**Controls.** The positive control is a simulated 2-hour merchant battery at a 40% share on the
calendar replica of a demand half; the 90% interval must hold `P` and `E`, and the posterior median
must lie within 10% of each. If the positive control fails, nothing else is scored. The negative
controls are the battery-free blocks; tariff templates shifted one hour early, which a real
window-edge signal should beat; and Agile prices from 7 days earlier.

## The ladder (five rungs)

**Rung 1: a simulated battery with an exact answer.** Merchant schedules from `lp_schedule` on the
N2EX price, with durations of 1, 2, and 4 hours, subtracted from each demand half at every share.
Demand halves: the 8 NGED primaries metered in MW, the 3 public demand-like BMUs of rung 7c
(`DEMAND_BMUS`), and the calendar replica of each. A solar arm adds a 25% share of rung 4's solar
sets at the 5% and 20% battery shares. This rung holds the positive control and answers C1 to C4.

**Rung 2: a fleet of small batteries that move as one (simulated).** No ground truth exists for any
real domestic fleet, so this rung is simulated, and the page says so in its first sentence. Each
fleet is built home by home: a number of homes, a 3 to 5 kW battery per home with a duration drawn
from the domestic prior, and a tariff for each home drawn from known membership shares. A second
fleet answers the 49.7 Hz trigger at the 6 real trigger half-hours as well as following its tariff.
The simulation's truth is exact; its realism is not checked against any real fleet, so rung 2 shows
what the estimator can recover if domestic batteries behave as the templates assume. The
simulator draws its parameters from distributions centred away from the priors' medians, so a fleet
the priors describe badly is part of the test. This rung answers C2 and C4 for fleets and gives the
detection curve for domestic sizes.

**Rung 3: real public batteries, alone and in fleets.** The four batteries of the prior study, four
of the smallest `battery_hint` BMUs in the census list with at least 95% of half-hours present, and
fleets of 2, 4, and 8 of the 149 `battery_hint` BMUs (fixed seed, 10 draws per size), added to the
demand halves at shares of 2% to 40%. Power truth is the registered generation capacity, with the
metered p99 beside it. No public energy capacity exists, so the energy reference is the prior
study's 3-month-block fit (`battery_rung3.smallest_capacity`) applied to the battery's own metered
output. This rung answers C5, and tests whether real dispatch leaves the calibration of C2 intact.

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
labelled S1 to S8, gets its detection statistic against the held-out threshold and the posterior of
each battery class's MW and MWh, for the study year and for September 2019 to August 2020, when
fewer distribution-connected batteries existed. NGED's Embedded Capacity Register (August 2026,
on disk) gives a partial answer key for connected storage of 50 kW and above: 2 of the 8 primaries
have a connected storage entry. The register lists no storage capacity in MWh and no duration for
any of its storage rows, so it can check presence and MW size class, never MWh. Below 50 kW the
register is silent, so a domestic fleet is never in it. The page reports, per primary, whether the
register lists connected storage, and never prints a generator's name or registered capacity, nor
pairs a primary's posterior with a register entry's capacity.

**NGED's primary series already have metered generation subtracted.** A battery that NGED meters
may therefore be absent from a primary's flow already, and what remains is the unmetered battery
the question asks about. Primaries metered in MVA are out of scope, because MVA has no sign.

**Each rung is reported only on the evidence below it.** If the positive control fails, nothing is
scored. If rung 1 finds nothing at a 40% share on real demand, rungs 2 to 5 are reported as a
demonstration that the method fails.

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
| 9 | The half-hours with frequency below 49.7 Hz, with the aggregate's change in each | Data | The static trigger fires a handful of times a year, too seldom to size a fleet. |
| 10 | The positive control: posterior of MW and of MWh, with the truth marked | 1 | The estimator recovers an easy battery, so later failures belong to the data, not to the code. |
| 11 | Posterior of MW against MWh for three rung 1 sums (40%, 10%, and 2% shares), truth marked | 1 | Large batteries give tight posteriors around the truth, and small ones give posteriors no narrower than the prior. |
| 12 | Posterior median against truth for power and energy, log axes, coloured by share, with 90% intervals | 1, 2 | Estimates track the truth above a size and scatter below it. |
| 13 | Coverage of the 50% and 90% intervals for power and energy, by rung and share, against the nominal lines (C2) | 1, 2, 3 | The intervals mean what they say, or the figure shows where they are too narrow. |
| 14 | The battery-free null distribution of the log Bayes factor with the threshold, and the false-alarm rate per null source (C1) | 1 | The detector flags about 5% of battery-free blocks, as designed. |
| 15 | Paired contrasts C3 and C4 with intervals, and the duration's posterior width over its prior width | 1, 2 | The posterior beats the step statistic, and MWh is or is not learned beyond the prior. |
| 16 | Detection probability against fleet size for the simulated domestic fleet, with the 1-hour-shifted templates as a negative control | 2 | A fleet is found from its window edge, and the shifted window finds nothing. |
| 17 | Real public batteries: posterior against registered power and against the energy reference, beside the simulated batteries at the same shares | 3 | Real dispatch is harder to size than a simulated schedule, by the amount shown. |
| 18 | Fleets of real batteries: posterior median against coincident peak and against the sum of registered powers (C5) | 3 | A fleet is seen as its coincident peak. |
| 19 | NGED battery A in its BSP: the log Bayes factor and the MW posterior against the multiple of NGED battery A (0 to 11), with the threshold | 4 | A real battery of this share is or is not detected, and the multiple at which detection starts. |
| 20 | The primary screen: each primary's log Bayes factor against the threshold and its MW posterior per battery class, for both eras, with the register's storage presence marked | 5 | Which primaries show evidence of a battery, and how that agrees with the register. |

## Work breakdown for a Sonnet agent

Run with `OMP_NUM_THREADS=2` and at most 4 workers. Every script gets a fresh code review before its
first run. Results go to `data/studies/per_study/unmetered_battery_capacity/`.

1. **Branch** from `main` after the battery-pv-separation pull request merges, or from that branch.
2. **Downloads** `studies/market_downloads/fetch_agile_and_dfs.py`, following the `data-download`
   skill: Agile, the earlier N2EX year, and the Demand Flexibility Service report. Run the
   `data-validation` checklist on each. Check the red band and the Agile region letter, and record
   both in the report.
3. **Move and add code in `packages/studies/`, with tests.** New module `studies/battery_templates.py`:
   the tariff, Agile, red-band, and charge-only templates, and the home-by-home fleet simulator. New
   module `studies/template_posterior.py`: prewhitening, the linear-Gaussian marginal likelihood, the
   truncated Gibbs step, importance weights, effective sample size, and the log Bayes factor. Move
   `calendar_replica` (from `battery_rung7.py`) and `smallest_capacity` with `cell_energy_path`
   (from `battery_rung3.py`) into `studies/battery_capacity.py`, repoint the prior study's imports,
   and re-run `battery_rung3.py` to show its outputs unchanged bit for bit.
4. **Inputs** `studies/unmetered_battery_capacity/capacity_inputs.py`: the 8 primaries metered in MW,
   the BSP flow (found by the correlation check, recomputed in code), and NGED battery A, read by
   metadata filter; the register's per-primary storage presence; data-validation checks into
   `report_inputs.md`. Nothing that identifies a generator is written.
5. **Template library** `capacity_templates.py`: 4,000 prior draws and their templates per block,
   saved. Time one sum's posterior; if rungs 1 to 3 would take more than 8 hours, cut in this order
   and record the cut: drop the solar arm to the 20% share; cut fleet draws from 10 to 5; cut the
   library to 2,000 draws.
6. **Positive control**, then **nulls and thresholds** (`capacity_nulls.py`).
7. **Rungs 1 to 5**, one script each, saving every posterior's draws and weights so a chart needs no
   refit.
8. **Report** `capacity_report.py`: every table the page quotes, C1 to C5 at both prior settings,
   coverage, effective sample sizes, and the controls.
9. **First Opus science review**, triage, re-run what it asks. **Charts and page**
   (`docs/studies/unmetered-battery-capacity.md`, beside the battery page in `mkdocs.yml`), then the
   **second Opus science review**, the diff reviews, the mutation pass, and the prose, persona
   (an NGED planning engineer, a home-battery optimiser developer, a sceptical Bayesian
   statistician), and evidence reviews.

## Tests (each would fail on `main` today)

- Tariff templates: charge starts exactly at each window's first half-hour, and each template's
  discharged energy equals its charged energy times the round-trip efficiency.
- Charge-only nuisance template: has no discharge half-hour.
- Fleet simulator: the fleet sum equals the sum of the homes, and a fleet with no duration spread
  reproduces a single template times the home count.
- Marginal likelihood: matches a brute-force numerical integral on a 3-column problem.
- Truncated Gibbs step: never returns a negative power, and recovers a known posterior mean on a
  conjugate case.
- Importance sampling: on a simulated sum whose `theta` is one of the library's draws, the posterior
  mass concentrates on that draw's neighbourhood; effective sample size equals the draw count when
  every weight is equal.
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

## Open questions for the maintainer

None. The maintainer has settled the priors, the per-primary publication, the size range, the MVA
scope, and the register's use. The download step stops for the maintainer only if a provider's terms
turn out to require accepting a licence.
