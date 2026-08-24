# The current state of the art in energy forecasting: a summary

This is a short version of a literature review Open Climate Fix carried out for National Grid
Electricity Distribution, as part of the Flexpectation project. The summary is meant to be read on
its own. The full review, which cites eighty sources and gives the evidence
behind every claim here, is published at
<https://openclimatefix.github.io/nged-substation-forecast/background/energy-forecasting-review/>,
and is referred to below as "the full review".

Before we discuss the literature, there is a very important caveat to admit up-front: In 2026, no
honest narrative review of the energy forecasting literature can claim to reveal the canonical
"state of the art"! That is because (almost) all energy forecasting papers measure performance in
different ways, against different datasets. It's like an international football tournament where
every team plays by different rules, with different size goals.

Energy forecasting researchers have done great work over the years. But, unfortunately, the
literature does not tell us how those approaches compare against each other, especially in messy
"real world" energy forecasting scenarios. This isn't anyone's fault; it's a systemic failure. And
the industry is already aware of this problem, and people are trying to fix it. But, at the time of
writing, the literature cannot yet tell us the current state of the art solutions for the problems
that NGED cares about.

Having said that, there have been some valiant attempts to compare multiple forecasting approaches
against the same dataset. But none of these attempts directly address the main challenges relevant
to Flexpectation. Before we discuss those attempts, we must emphasise two reasons for optimism:

Firstly, whilst we might not know exactly which algorithms provide the best forecasting performance,
we do know how to go about finding out. There's no magic. Machine learning is an empirical science,
and progress in it comes largely from testing many ideas under identical conditions and measuring
carefully — because most ideas fail. John Jumper, who shared the 2024 Nobel Prize in Chemistry for
his work on AlphaFold, puts the share of research ideas that fail at around 90%, and treats that
rate as an ordinary and necessary feature of doing research rather than as evidence of doing it
badly ([Nobel Week interview](https://www.nobelprize.org/prizes/chemistry/2024/jumper/interview/), 6
December 2024, from 14:12). If roughly one idea in ten survives contact with the data, ten attempts
is simply the price of one result. So our task is to run hundreds of ML experiments, and then
measure performance against the same dataset, using the same performance metrics.

Secondly - and perhaps most importantly - the fact that the industry doesn't yet know the state of
the art is a huge opportunity for the Flexpectation project: We are in a very privileged position
where we can try hundreds of ideas, and test the best ideas in the real world. We have a fantastic
opportunity to make a significant contribution to the energy forecasting industry by publishing our
"leaderboards of ML experiments", and hence help the industry as a whole to better understand how
multiple approaches perform.

## What we read, and what Flexpectation is

Flexpectation forecasts net demand — demand minus whatever generation sits behind the substation —
at NGED's grid supply points, bulk supply points and primary substations. The forecast is
half-hourly, runs 14 days ahead, is reissued every six hours, and states a range of possible loads
with a probability attached to each rather than a single number. Version one covers a 32-series
trial area; the scale-up from 2027 covers the network.

We read most of the papers an argument rests on in full; the other papers were available to us only
as an abstract, a preprint or part of a paper, and wherever a claim rests on a partial read we say
so at the point the claim is made. We also read the published deliverables of twelve
network-innovation projects in GB. Papers may be missing for no better reason than that we did not
find them, and the section "What this review excluded, and why" lists what we knowingly left out.
Every statement below that we found no published work on something is a statement about our search
rather than about the field: if you know of work that fills one of these gaps, we would rather cite
it than repeat it, and we will correct this review.

One concurrent project is cited more than any paper: Northern Powergrid's Artificial Forecasting, an
Ofgem Strategic Innovation Fund programme whose Alpha and Beta deliverables are both public, and
which has its own section below. [Haben et al. (2021)](https://arxiv.org/abs/2106.00006) reviewed a
final list of 221 low-voltage forecasting papers published to 2020, noting that the number they
actually read is slightly smaller. [Shukla and Hong (2024)](https://doi.org/10.1049/stg2.12162)
reports the BigDEAL Challenge 2022, a competition themed on forecasting the *timing* of peak demand
rather than its size, which drew 78 teams from 27 countries and published its data alongside the
paper.

Almost every number in this review depends on where in the network it was measured, so here is the
voltage ladder of a distribution network, from the top down:

- **Grid supply point** — where the distribution network meets the transmission system, 400 kV or
  275 kV down to 132 kV. Hundreds of thousands of customers sit below one.
- **Bulk supply point** — 132 kV down to 33 kV or 66 kV. Tens of thousands of customers.
- **Primary substation** — 33 kV or 66 kV down to 11 kV. A few thousand customers.
- **Secondary substation** — 11 kV down to 400 V. Tens to a few hundred customers.
- **Feeder and individual customer** — the bottom of the ladder, at 400 V.

NGED owns 52 grid supply points, 271 bulk supply points and 1,161 primary substations. The 32 series
of the trial area are 16 of those primary substations, two grid supply points, two bulk supply
points and the 12 metered generators described under problem 2 below. **Flexpectation forecasts no
secondary substations**, neither in the trial area nor in the network-wide scale-up, though several
of the studies below do. GB is separately divided into 14 *grid supply point groups*, each a whole
distribution region containing many grid supply points, and several studies below forecast those
regions, which are far larger than any single substation.

## How to read the numbers in this review

**Two kinds of published number transfer to a different network, and other kinds do not.** A ratio
against a baseline transfers, but only if the paper says what the baseline was and which substations
it was averaged over. Those baselines differ far more than the prose in most papers suggests —
yesterday's value at the same time, the average of the last four weeks, a day-type persistence rule
and the long-run seasonal average all appear among the studies reviewed here, and a percentage gain
against one baseline is not a percentage gain against another. A skill score — how much less error a
forecast has than a stated benchmark, as a percentage — needs its benchmark named for the same
reason.

**Errors normalised by something physical also transfer.** An error expressed as a fraction of a
substation's firm capacity or transformer rating comes far closer to meaning the same thing at every
substation than an error expressed as a fraction of the load that happened to occur. Normalising by
a rating is not exact, because a rating is itself a convention standing in for a limit that moves
with air temperature, wind and the duration of the overload, so a paper that normalises this way
should say which rating it used. An absolute error in kilowatts or megawatts tells NGED nothing on
its own, because it depends entirely on how big the substation was, and none of the absolute figures
below should be read as a target for this project.

**Whether a study used the weather forecast a real forecaster would have had changes what its
numbers mean.** In the table under problem 1 below, "real forecasts" means the weather forecast that
was genuinely available when the power forecast was made; "actual weather, after the fact" means
observations, or a weather model re-run after the event, that no forecaster would have had. Two of
the studies in the table below use actual weather after the fact, which makes their figures upper
bounds rather than achievable performance, because it removes the error that dominates beyond a day
or two — precisely the range NGED acts on.

**Terms used below.** **Pinball loss** scores a forecast that states a range rather than a single
number, penalising it more heavily for missing on the side it claimed was unlikely. A **day-type
persistence** benchmark predicts today's load from the most recent day of the same type — last
Tuesday for a Tuesday. A **foundation model** is trained on large numbers of unrelated time series
and then applied to a new one it has never seen. **Deterministic** means a single predicted number
with no uncertainty attached.

## The eight problems Flexpectation has to solve, and what the literature says about each

Flexpectation's specification breaks into eight problems. This section takes each in turn, says what
the problem is, reports the most relevant published results found in our literature search, and says
where those results stop short. The coverage is uneven. The first problem has enough published
results to tabulate, and the second problem is the most mature field on the list. For most of the
remaining six we found no published result that could be compared against anything, so those are
described in prose: the absence is itself the finding.

### 1. Probabilistic forecasts of net demand at substations

**The problem.** Forecast net demand — demand minus whatever generation sits behind the substation —
at every grid supply point, bulk supply point and primary substation, half-hourly, 14 days ahead,
updated every six hours, as a range of possible loads with a probability attached to each rather
than as a single number. NGED acts on the forecast one to ten days ahead, and the question NGED asks
of the forecast is "how likely is load to cross this substation's firm capacity — the load the
substation can carry safely with its largest transformer out of service?" rather than "what is the
most likely load?". Forecasting net demand is the highest priority of the eight problems, and the
other seven exist mainly to make that net-demand forecast better.

**In summary.** A large literature forecasts substation load, but very little of what we read can be
compared with the rest of that literature, and we found none of it driving a probabilistic
substation forecast from a weather ensemble across a 14-day horizon.

**What the literature reports.**

| Source | What they forecast | Level and scale | Horizon | Result, and what it was compared against | Weather |
|---|---|---|---|---|---|
| [Kaas et al. (2026)](https://arxiv.org/abs/2607.01966) | Net load, Germany | Low-voltage feeder: 200 | 4 days | A general-purpose foundation model that had never seen the data beat every purpose-trained model on average error, 3.8 kW against 4.2 kW | Actual weather, after the fact |
| [Hertel et al. (2026)](https://arxiv.org/abs/2607.15705) | Load, Germany and Portugal | Transmission, plus 200 low-voltage feeders and 287 individual customers | 4 days | Best model beat a day-type persistence forecast by 59.6% at transmission level, 42.3% at low-voltage feeders, 23.3% at individual customers | 1–3 h forecasts at the feeders, reanalysis elsewhere |
| [Browell and Fasiolo (2021)](https://arxiv.org/abs/2103.10335) | Regional net load, GB | Regional: 14 grid supply point groups | Day-ahead | Held the same risk with **up to 24.6% less upward reserve** than a fixed-tail alternative (note 1) | Real forecasts |
| [Pinheiro et al. (2023)](https://doi.org/10.1016/j.apenergy.2022.120493) | Load, Portugal | Secondary substation: 96,989 | Day-ahead | 42–47% better than the reference benchmark at system level. **At substation level, beat a naive forecast on 83–87% of network-owned and 66–70% of customer-owned sites** (note 4) | Real forecasts, 7–8 h old |
| [Gilbert et al. (2023)](https://arxiv.org/abs/2206.11745) | Load, GB | Four levels: primary substation down to household | Day-ahead | Combining forecasts gained **0.0–0.4% averaged over all periods**, but **5.7–9.0% when restricted to peaks** | None at all |
| [SSEN TRANSITION 2021](https://ssen-innovation.co.uk/transition/) | Net load, Oxfordshire | Primary substation: 13, plus their bulk supply points and 11 kV feeders | 30 min to 10 days | **11 of 13 primary substation models below 10%** mean absolute percentage error when fitted (note 2) | 40-member ICON-EU ensemble to 4 days, then one deterministic forecast to 10 days |
| [Artificial Forecasting (Northern Powergrid)](https://smarter.energynetworks.org/projects/npg_sif_006-1/) | Demand and export at primary substations; active power at secondary | Primary substation: 551 with export data, 171 modelled; secondary: 729 | Day-ahead to 11 days at primary; week- to month-ahead at secondary | **About 8% lower mean absolute error** of utilisation rate than the network's existing method (note 3) | Real forecasts at primary; none in the published secondary results |
| [Ruhhütl et al. (2023)](https://doi.org/10.1049/icp.2023.0476) | Load and generation, Austria | Substation | Day-ahead | **3 to 8% mean absolute percentage error**, varying with how industrial and how large the supplied area was; linear and Gaussian regression preferred over the alternatives tested (abstract only, note 5) | Not stated in the abstract |

*Notes.* **1.** The 24.6% saving is at the most extreme tail level [Browell and Fasiolo
(2021)](https://arxiv.org/abs/2103.10335) tested, and falls to 3.2% at the least extreme. **2.** The
two SSEN TRANSITION models that missed 10% reached 13.4% and 19.7%, and 94% of the 11 kV feeders it
built models for came in below 20%. **3.** Artificial Forecasting also captured 83% of the top 10%
of demand values inside its 5th-to-95th-percentile band, and beat its comparison benchmarks at all
eight of the near-capacity substations it was evaluated on. **4.** The beat-a-naive-forecast figures
are given as ranges because [Pinheiro et al. (2023)](https://doi.org/10.1016/j.apenergy.2022.120493)
reports two different pairs of numbers for the same statistic: 82.8% and 66.0% in the body text,
86.5% and 70.0% in the caption of the figure on the page after. We could not tell which is intended,
so the table spans both. **5.** By this review's own test, the Austrian figure does not transfer: a
mean absolute percentage error is normalised by the load that happened to occur rather than by
anything physical, and the abstract names no baseline to measure it against. The Austrian row is in
the table because it is the only substation-level study we found from a comparable European network,
and it is the one row whose number should not be read as a target.

**Even within this one table, the studies cannot be compared with each other.** [Kaas et al.
(2026)](https://arxiv.org/abs/2607.01966) and [Hertel et al.
(2026)](https://arxiv.org/abs/2607.15705) name different models as best. Neither disagreement is a
mistake: the two papers test different sets of models at different time resolutions, and the two
metrics answer different questions.

**Where the gaps are: no study we found drives substation uncertainty from a weather ensemble across
a full 14-day horizon.** Both [Haben et al. (2021)](https://arxiv.org/abs/2106.00006) and [Ludwig et
al. (2023)](https://doi.org/10.1080/01605682.2022.2115411) ask in print for the substation-level
part of it, though neither names a resolution or a horizon.

**Almost every study here optimises average accuracy, but NGED's question is about the top of the
distribution.** [Browell and Fasiolo (2021)](https://arxiv.org/abs/2103.10335) is the one study here
that models the upper tail explicitly, and what they found is a warning rather than a reassurance:
they find that "below 1% and above 99% the forecasts based on quantile regression only are not
calibrated at any GSP Group. Therefore, these quantiles are not suitable for use in
decision-making", even with five years of half-hourly data across regions far larger than a
substation. Above the 1st and 99th percentiles, Browell and Fasiolo switch to a fitted parametric
tail.

### 2. Forecasting metered generators

**The problem.** Twelve of the 32 series in the trial area are individually metered generators — six
solar farms, three wind farms, a biofuel plant, a battery and a gas generator — and each needs the
same probabilistic, half-hourly, 14-day forecast as a substation. Solar and wind are driven by
weather the ensemble supplies directly. The battery, the gas generator and the biofuel plant are
dispatched on market prices and operator decisions, and no weather forecast contains either.

**In summary.** Forecasting wind and solar from a weather forecast is the mature case, and one paper
matches Flexpectation's problem closely; nothing we found forecasts a distribution-connected
battery, gas generator or biofuel plant inside a net-demand forecast.

**At the scale of an individual generator, the closest work is on wind.** [Dantas and Browell
(2026)](https://doi.org/10.1002/we.70079) forecast 73 wind farms in GB — 34 onshore, 39 offshore —
from the ECMWF ensemble, seamlessly from 6 to 162 hours ahead. [Dantas and Browell
(2026)](https://doi.org/10.1002/we.70079) conclude that whether weather-forecast error or
weather-to-power conversion error dominates flips with lead time, and that the lead time at which it
flips varies a lot between sites. A second conclusion is more uncomfortable for a project built on
an ensemble: a deterministic forecast at higher resolution beat the ensemble at short lead times.

**Where the gap is: nothing we found forecasts a distribution-connected battery, gas generator or
biofuel plant inside a net-demand forecast.** For the battery there is at least a method to borrow.
[Bian et al. (2024)](https://doi.org/10.1109/TSG.2023.3303469) recover a price-taking storage
operator's own optimisation parameters from historical prices and observed dispatch. We found no
method worth borrowing for the gas generator or the biofuel plant; what little exists forecasts a
gas or biofuel plant's own output directly rather than as a component of a substation's net demand.

### 3. Estimating the effective capacity of metered generators

**The problem.** We call the amount of generation actually available at a metered site its
*effective capacity*: the output it could produce right now if the weather allowed, as opposed to
its nameplate rating. Turbines go out for repair, inverters degrade, and sites are curtailed — told
by the network operator to generate less than they could. A 20 MW wind farm that has been limited to
14 MW for a month is, for forecasting purposes, a different wind farm, and a model trained on its
nameplate rating cannot see the difference.

**In summary.** A method exists for each generation technology separately, but nobody has run them
across a mixed fleet at a distribution network, or tested whether estimating capacity improves the
forecast.

**For wind, one paper hits our problem exactly, and publishes its method.** [Dantas and Browell
(2026)](https://doi.org/10.1002/we.70079) needed available capacity for the same reason we do, so
rather than use a nameplate rating they estimate a time series of available capacity for each farm
and normalise that farm's power by it before modelling. Their method needs no capacity register and
no outage messages. [Viotti et al. (2026)](https://doi.org/10.1002/we.70136) point out that taking
the running maximum of production "requires monotonically increasing capacity and relies on frequent
high wind events" — and NGED's effective capacity goes *down* when a turbine is out for repair.
[Viotti et al. (2026)](https://doi.org/10.1002/we.70136) fit the most likely capacity time series
instead, and report **27.2% lower normalised mean absolute error** than the running maximum at
quantifying capacity after a new wind farm connects.

**Where the gap is: no published work we found estimates effective capacity across a mixed fleet at
a distribution network, or tests whether estimating it improves the forecast NGED buys flexibility
against.** The per-technology methods exist, and most of them work from a revenue meter alone.

### 4. Detecting switching events

**The problem.** When a cable fault or planned maintenance moves part of a network from one
substation to another, the load a substation meters steps up and its neighbour's steps down, with no
change in the underlying demand. NGED's substations spend roughly a tenth of their operating time in
an abnormal running arrangement. Switching labels exist for the 32-series trial area but not for the
wider network, so a method that is to scale to the wider network has to work from power measurements
alone.

**In summary.** One paper detects switching at a real network operator, using a bottom-up reference
series NGED does not have; the GB precedent drew the same distinction in 2016 but never measured how
often its rule was right.

**One paper detects these events at a real network operator, in order to strip them out before
estimating how much load a substation carries.** [Bouman et al.
(2024)](https://arxiv.org/abs/2405.16164), working with the Dutch network operator Alliander, study
180 primary substations at 15-minute resolution over roughly a year, detecting events that run from
a few minutes to several months. [Bouman et al. (2024)](https://arxiv.org/abs/2405.16164) is the
most directly useful paper in this review, and it leaves the forecasting half untouched, which is
what Flexpectation would add: keeping a forecast running through the events rather than deleting
them.

**Where the gaps are: the published method detects on a residual we cannot build the same way, and
the events NGED cares about are harder than the events detected.** A switch at NGED usually fans out
to two or three neighbouring substations rather than one, and the common case is a *partial*
transfer rather than a whole subgrid.

### 5. Forecasting a substation as if it were always in its normal running arrangement

**The problem.** NGED plan the network against what each substation would carry under its normal
running arrangement, so that is what the forecast has to predict — including for a substation that
has been sitting in an abnormal arrangement for weeks. That makes the target a quantity that was
never metered, and it makes the training history contaminated: past readings taken while the network
was abnormally configured describe a different substation from the one being forecast.

**In summary.** Researchers either leave the level shifts in and pay for them, rewrite the history,
or adapt to the new level; we found nobody who feeds the contamination to a model deliberately, as
information.

**Researchers respond in one of three ways, and two of the three alter the series the model is
trained to predict.** One strand leaves the level shifts in and reports the damage; a second
rewrites the history to the level it would have had if the switch had never happened, [Paredes and
Vargas (2017)](https://doi.org/10.1049/iet-gtd.2017.0129) do that across 169 real feeders and report
better medium-term forecasts for it, and Artificial Forecasting does the same in its
data-preparation pipeline. **Adaptive models are the live alternative: they track a new level once
it arrives, including one that arrives abruptly, but they never record that a switch happened.** A
model that simply adapts to a new load level cannot report what the substation would have carried
under its normal arrangement, which is the quantity NGED needs.

**Where the gap is: we found nobody who feeds a model switching-contaminated history *deliberately*,
as information rather than as damage.** Instead of correcting the series, a model could be fed the
difference between what a substation actually metered and what a model that ignores network topology
expected it to meter.

### 6. Detecting faulty metering

**The problem.** NGED's telemetry carries stuck values that repeat unchanged for hours or days,
zeros that mean "no reading" rather than "no load", physically impossible values, and gaps running
from a single half-hour to several months. Ten of the 32 series in the trial area are metered in
apparent power only, so they report magnitude without direction and reverse flow appears as a rise:
at one primary substation the meter bounces off zero on sunny days, when a solar farm behind it
exports. A model trained on uncleaned data learns the fault, and a forecast that fails silently
because its recent history was stuck is worse than one that says it is degraded.

**In summary.** Faulty metering is usually a data-cleaning step mentioned in passing rather than a
problem in its own right, the only public labelled dataset we found is Dutch, and recovering the
direction of flow from a magnitude-only meter was attempted by this network's predecessor, whose
automatic version is still open.

**The most useful published method treats faulty metering and switching as one problem.** [Bouman et
al. (2024)](https://arxiv.org/abs/2405.16164) treat measurement errors and switch events as the two
things that must be filtered out before substation measurements can be used, detect both on the same
residual. Their sign-recovery technique addresses exactly the non-directional metering defect
described above. Three network-innovation projects in GB tackled faulty metering substantively, one
of them as its whole subject — Electricity North West's ATLAS, UK Power Networks' Distribution
Network Visibility, and this network's own Time Series Data Quality.

**Where the gaps are: the fault taxonomy, a measured GB detector, and a reference series to detect
against.** None of the three GB projects above reports how often its checks are right, and none
published its labels, so there is no GB number to compare a new detector against.

### 7. Disaggregating unmetered solar and wind from a substation's net flow

**The problem.** Rooftop panels and small turbines appear only as a dent in a substation's net flow.
Recovering both the half-hourly output of that unmetered generation and its installed capacity, from
the net flow alone, is what we call *disaggregation*. Disaggregation is a different task from
estimating how much of a *metered* generator's capacity is available today, which is problem 3.
Disaggregation is a stretch goal for the trial area and a requirement for the network-wide scale-up.

**In summary.** Splitting generation out of a substation's net flow has been done where the
generation is metered or its capacity is read from a register, and uncertainty and a multi-day
horizon each appear in this literature, but never together.

**The direct predecessor of this work is running now in GB.** [UK Power Networks'
NIA_UKPN0104](https://smarter.energynetworks.org/projects/nia_ukpn0104/) (2024–2026, £389,444),
which Open Climate Fix worked on, infers the capacity of unmetered solar sitting behind each primary
substation from half-hourly substation load and weather, then forecasts that generation. Open
Climate Fix is a partner in both NIA_UKPN0104 and Flexpectation, so Flexpectation starts from
NIA_UKPN0104's method rather than from scratch.

**A Dutch network operator has published a method that splits unmetered wind and solar out of
substation measurements, by transferring from substations that do meter them.** [Teng et al.
(2023)](https://doi.org/10.1016/j.rser.2023.113662) train on ten Dutch substations that carry
complete renewable metering, then predict solar and wind power separately at substations with none,
from the substation's measured total load, weather, geospatial position and each site's known
renewable capacity, at 15-minute resolution — a root-mean-square error of 0.07 against 0.70 for a
default transfer-learning model, on a min-max-scaled target. The paper reads 0.07 as 7%, but does
not say what the scaling divides by, so the figure does not transfer to another dataset.

**GB already has an operational forecast of unmetered generation, at national scale.** NESO
publishes [embedded wind and solar
forecasts](https://www.neso.energy/data-portal/embedded-wind-and-solar-forecasts) half-hourly to 14
days ahead, the same resolution and horizon Flexpectation delivers. The forecast is a single number
per half-hour, with no uncertainty attached, and it covers GB as one region rather than substation
by substation.

**Where the gaps are: doing it without a metered training set, inferring the capacity rather than
being told it, and putting uncertainty and a multi-day horizon in the same forecast at substation
level.** [Teng et al. (2023)](https://doi.org/10.1016/j.rser.2023.113662) need a population of
fully-metered substations to transfer from, and are given the existence and capacity of each
renewable facility rather than inferring it — whereas inferring that capacity is half of what NGED
needs.

### 8. Disaggregating other distributed energy resources: heat pumps, electric-vehicle chargers and batteries

**The problem.** Heat pumps, electric-vehicle chargers and price-sensitive domestic batteries change
the shape of a substation's load in ways a model trained on history cannot anticipate, because the
number of them behind any given substation is growing quickly. The stretch goal is to disaggregate
and forecast them separately rather than letting them sit inside net demand.

**In summary.** Heat pumps, chargers and batteries are the largest gap in the review and the largest
deliberate omission from our search: in the one study we found that measures charger forecast skill
against aggregation, only the site with more than a hundred charge points was significantly better
than a naive benchmark, though some models at one much smaller site also beat it, heat-pump
diversity is untested in the cold weather that matters, and no diversity factor helps for domestic
batteries at all.

**Detecting heat pumps, chargers and batteries and forecasting them are separately hard, and not in
the order we expected.** Northern Powergrid's [smart-meter detection
trial](https://smarter.energynetworks.org/projects/npg_nia_-49/), on 1,500 monitored premises, found
that "EV identification at premises level was found to be relatively straightforward" and that
"aggregation does mask some signals, although EV usage is still clearly identifiable at feeder and
substation level", while "the detection of ASHP [air-source heat pumps] is frustrated by the low
levels of adoption". So the spiky, synchronised charging that makes electric vehicles hard to
*forecast* is what makes them easy to *detect* in aggregate; heat pumps are the reverse.

**Where the gaps are: forecast skill at substation aggregation, and the tariff-driven peak.**
Nothing we found forecasts an aggregate of heat pumps, chargers and batteries behind a GB primary
substation, states its own uncertainty, and is scored against the evening peak that the network
actually cares about. Reading the electrification literature properly is the first deliverable on
this strand, before any model.

## How we will know whether each of these worked

The eight problems above need three different kinds of evaluation, and this literature is far
stronger on the first than on the other two. Forecasting has settled practice we can adopt.
Estimating something nobody measures — an effective capacity, an unmetered solar output — has six
possible substitutes for ground truth, of which this literature uses four. Detecting rare events has
good academic practice and, in GB, no precedent that measured anything at all.

**Every forecasting paper we read that describes its split keeps most training data out of the
future of its test data, with one exception, and the training window usually grows rather than
slides.** **One length rule is worth adopting outright.** [Pinheiro et al.
(2023)](https://doi.org/10.1016/j.apenergy.2022.120493) held out the whole of 2019 and note that
"one year is the minimum acceptable to test a forecasting model whose target value shows annual
seasonality". Substation load shows exactly that seasonality, so any fold shorter than a year cannot
tell us whether a model handles winter, and winter is when NGED buys flexibility.

**Not one of the papers we read addresses the leakage a frequently reissued forecast creates, and
Flexpectation is the most exposed design of the lot.** When a forecast covering 14 days is reissued
every six hours, every target half-hour is covered by 56 separate forecasts. Count them as
independent and a significance test will report a confidence the data does not support; let a target
half-hour fall on both sides of a train-test boundary and the test set is contaminated outright. We
will report what we did about it rather than leave it implicit, and we treat it as an open
methodological question rather than a solved one.

**There is no ground truth for an effective capacity or an unmetered solar output, and the papers
that estimate them say so.** This literature uses four substitutes for truth, each of which fails
differently, and leaves two more on the table. The four in use are to hold out sites that are
metered and pretend they are not; to inject a change into real data and see whether the method
recovers it; to compare against an independent tool rather than against truth; and to measure
whether the estimate improves the forecast it was built to improve. The two left on the table are to
check an estimate against physics rather than against an answer, and to meter one substation
completely for a period and use it only as validation.

**Flexpectation will run all six substitutes and treat agreement between the six as the signal,
because no one substitute is trustworthy alone.** They are not six attempts at the same measurement,
and each fails differently. Every number we publish will say which of the six substitutes produced
it.

**Detection needs different metrics, and the best-worked example in this review chose them
deliberately.** [Bouman et al. (2024)](https://arxiv.org/abs/2405.16164) score precision, recall and
an F-score with β set to 1.5 rather than 1, "to give a higher importance to the recall term, as the
potential impact of a false negative is higher than that of a false positive in power grid expansion
planning". That asymmetry holds for Flexpectation too: a missed switching event silently corrupts
the history a model trains on, whereas a false alarm costs an engineer a look.

**The honest headline from the one paper that measured properly is that detecting switching and
metering faults is hard.** [Bouman et al. (2024)](https://arxiv.org/abs/2405.16164) report F-scores
near 0.2 on the shortest events and around 0.5 on the longest, and conclude that performance "is
relatively low across the board, even on the train data. This indicates that the problem is hard to
learn, though it generalizes fairly well". Any target we set for problems 4 and 6 should start from
those F-scores rather than from an intuition about how obvious a switching event looks on a chart.

**None of the three GB projects we checked offers a number to compare against, which we checked
rather than assumed.** Publishing precision and recall against a stated label set, with the labels
released, would therefore be the first time we know of that a GB network has done so, and it is the
cheapest of this review's commitments to keep.

## What published leaderboards did, and what a single team can borrow from them

Building leaderboards is one of Flexpectation's deliverables, so the design of a leaderboard is
itself a question the literature can be asked about. **What Flexpectation is building is a
leaderboard, not a competition, and the distinction changes which published lessons apply.** Our
leaderboards carry our own experiments — one per class of time series, so solar farms, wind farms,
batteries and the demand at primary substations each get their own, with grid and bulk supply points
sharing a board because their measurements are the same kind of thing — They will be public to view
and reproducible, but we are not inviting other teams to submit entries. Anyone who wants to
benchmark against us can rerun the setup for themselves. That means the published lessons about
attracting entrants, prize pots and qualifying rounds do not apply to us, while the lessons about
protocol — what makes a comparison trustworthy — apply with more force, because a competition gets
some of its integrity free from having rivals who would like to catch each other out, and we will
not have any.

**Energy forecasting has run competitions on common data for over a decade, and of the competitions
whose target we could establish, only one got near the level NGED acts on.** The Global Energy
Forecasting Competitions of 2012, 2014 and 2017 covered hierarchical load, price, wind and solar,
published their data, and drew hundreds of contestants from more than 60 countries ([Hong et al.
(2020)](https://doi.org/10.1109/OAJPE.2020.3029979)). The closest of those competitions to a
distribution network is the second track of GEFCom2017, which asked for probabilistic forecasts of
183 delivery-point meters of a US utility and drew 177 entrants in total across both its tracks
([Hyndman (2020)](https://doi.org/10.1016/j.ijforecast.2019.03.015)). What we found no example of is
a *standing* leaderboard at distribution-substation level — one that keeps accepting new entries
after its competition closes. That is the gap Flexpectation's leaderboards fall into, though the
search behind that statement is ours and we would be glad to be pointed at a counter-example.

**The mechanism that makes a leaderboard trustworthy is time, not policing.** TS-Arena's central
idea is that a forecast is submitted before the outturn it will be scored against physically exists,
which "makes test-set contamination impossible by design". HEFTCom made the same argument from
experience: because the competition ran on the real, unknown future, "data leakage, accidental or
deliberate, was impossible". A half-hourly forecasting service is unusually well placed here: every
day supplies 48 fresh evaluation points that can never be reused, and the condition that the answer
did not exist when the model was frozen holds automatically.

**The specific way a single team fools itself is not fabrication but running the baseline badly.**
[Kleinebrahm et al. (2026)](https://arxiv.org/abs/2604.24705) put it as a general problem with
published comparisons: competing methods "are not always implemented or optimized with equal care",
so reported differences "may reflect differences in implementation quality rather than inherent
methodological advantages". [Hong et al. (2020)](https://doi.org/10.1109/OAJPE.2020.3029979) put it
more bluntly, that "sometimes the parameters are manipulated, so that the competing models are being
dominated by the proposed ones". So we run every entry through the same evaluation interface, and
run each baseline from its authors' own code at its authors' recommended defaults — the rule [Meyer
et al. (2026)](https://arxiv.org/abs/2512.20761)'s TS-Arena imposes on itself. **Run two baselines
that bracket the answer, not one.** [Doubleday et al.
(2020)](https://doi.org/10.1016/j.solener.2020.05.051) distinguish the two jobs a benchmark does: a
yardstick, which need not be a good forecast, and a point on the yardstick, which "should be close
to the state of the art". They recommend carrying both, so that a new method can be positioned
between them rather than merely declared better than something. That is the shape our leaderboards
take: persistence and climatology as the naive yardstick, and NGED's incumbent method as the point
on the yardstick a new model has to reach.

**Our own leaderboard has this problem today, and we would rather say so than discover it later.**
The fold that Flexpectation currently reports serves as both the model-selection set and the
reported result, so every hyperparameter choice and feature ablation is adjudicated on the same
twelve months the leaderboard publishes. With hundreds of experiments planned, the winner's reported
skill will be optimistically biased. The structural fix is a final-test window that no model
selection is allowed to touch, and it is scheduled. Until it lands, three things hold: leaderboard
numbers are selection metrics rather than estimates of future skill, differences smaller than
fold-level noise should not drive decisions, and the number of experiments run against a fold is
itself a statistic worth publishing beside the fold's results.

**Rankings travel better than absolute numbers do.** Where a benchmark has enough data behind it,
the ordering of models survives a change of test set even when the accuracy level does not, and that
decides what a leaderboard should report as its headline. [Fildes
(2020)](https://doi.org/10.1016/j.ijforecast.2019.04.012), reviewing the M4 competition, compared
its daily micro series against a real retail forecasting problem and found the same method scoring
1.665% on one and 11.1% on the other. His conclusion is a direct endorsement of what Flexpectation
is doing: "each organization needs to organize its own forecasting competition for its own
forecasting problems, and should not rely on even large benchmark data sets", with the published
competition useful for narrowing "the pool of methods to be considered" rather than for predicting
your own error. So a leaderboard should lead with ranks and with margins over a stated baseline, and
treat an absolute skill number as valid only on the distribution it was measured on.

**A finite evaluation window can rank the wrong model first, and several months is not obviously
enough.** [Messner et al. (2020)](https://doi.org/10.1002/we.2497)'s conclusion is the sharpest
warning we found about reading a leaderboard: "evaluation results based on a finite data set are
always subject to some degree of uncertainty and the best ranked forecast does not necessarily have
to be the truly best one. Depending on the actual setup, e.g., in a benchmarking exercise to hire a
forecaster, it should be remembered that even periods of several months may still yield uncertainty
in terms of who the best forecaster truly is." HEFTCom's own competition period was three months.

**What a leaderboard without entrants cannot do, we should not claim it does.** Three of the
strongest results in the benchmarks above are unavailable to us. CASP's finding that its field
plateaued for fourteen years is a statement about protein structure prediction only because dozens
of groups were trying independently; a plateau on our leaderboard would be ambiguous between a hard
problem and a team that did not think of the right idea. What our leaderboard can do is narrower and
still worth having: show which approaches beat a stated baseline on NGED's own data, under one
protocol, with the forecasts, the metric definitions and the code published so that anyone can check
the arithmetic or rerun the comparison themselves.

## Six findings that recur across the studies we read

The six findings below describe this literature rather than laws of nature: each is what several
teams measured on their own networks, and a network that differs from theirs may well behave
differently.

### 1. In the load-forecasting studies we read, each further step up in model sophistication bought a much smaller margin than the effort put into it would suggest

[Pinheiro et al. (2023)](https://doi.org/10.1016/j.apenergy.2022.120493), running a live system
drawing on 96,989 Portuguese secondary substations and fitting per-asset models at 84,663 of them,
tuned a system-level gradient-boosted tree. At system level, the gradient-boosted tree scored 199 MW
root-mean-square error and the generalised additive model scored 191 MW, so the gradient-boosted
tree was 4% worse than the simpler model. [Pinheiro et al.
(2023)](https://doi.org/10.1016/j.apenergy.2022.120493) concluded there was no accuracy gain to be
had and rejected the gradient-boosted tree on the cost of tuning it and on the loss of
interpretability, keeping the generalised additive model. Artificial Forecasting also found that
gradient-boosted trees did not beat a simpler model, when forecasting customer export at primary
substations. What did help them was refitting the model every month rather than redesigning it.

### 2. In every study that forecast more than one voltage level, accuracy got worse further down the network

[Hertel et al. (2026)](https://arxiv.org/abs/2607.15705) ran the same models against a day-type
persistence baseline on three datasets — a German transmission control area, 200 German low-voltage
feeders and 287 individual Portuguese clients — and the margin over that baseline shrank from 59.6%
to 42.3% to 23.3% as aggregation fell. What shrank is the headroom above a naive rule rather than
the accuracy itself, which is the more useful reading: their own gloss is that it is easier to beat
a simple approach on highly aggregated data than on volatile feeder- and client-level data. The
model did not get worse; the problem got harder.

**Rising error does not mean falling usefulness.** A forecast at a primary substation may carry a
larger percentage error than a forecast at a grid supply point and still support flexibility
procurement just as well, because what NGED needs from the forecast is a reliable answer to "will
this substation exceed its firm capacity?", and that question can be answered well even when the
load itself is hard to predict precisely. Whether decision-usefulness really is flat across voltage
levels is something this project can measure, and we intend to.

### 3. In the one study we found reporting results substation by substation at scale, the trained model did not beat a naive "same time yesterday" rule at a substantial minority of substations — and we know that only because the authors reported it

[Pinheiro et al. (2023)](https://doi.org/10.1016/j.apenergy.2022.120493) found that their model beat
a "same time yesterday" forecast at 83–87% of network-owned secondary substations but at only 66–70%
of customer-owned ones. We do not know that NGED's primary substations will behave the same way, and
they may not, because a primary substation aggregates far more customers than a Portuguese secondary
substation does.

### 4. In the studies we read, standard accuracy measures rewarded flat forecasts that would be of little use for flexibility procurement

A forecast that predicts the right peak at the wrong time is penalised twice by mean absolute error
— once for the peak it predicted that did not happen, and once for the peak that did happen and the
forecast missed. A flat, featureless forecast avoids both penalties. Meteorologists named that
effect the double penalty decades ago, and their conclusion transfers: a score that forgives a peak
predicted an hour late is generally no longer a **proper scoring rule** — a score a forecaster
cannot improve by publishing anything other than what they genuinely believe. A peak-aware score
therefore belongs alongside a proper score, not instead of one. Two teams independently concluded
that mean absolute error was the wrong measure for peaks. Artificial Forecasting built a metric over
the top 10% of demand values and made it the primary measure for comparing their models, reporting
it both against actual demand and normalised to transformer rating.

### 5. In the study we read most closely, a forecast stated its own uncertainty badly and a single accuracy score did not reveal it

[Kaas et al. (2026)](https://arxiv.org/abs/2607.01966) scored models on 200 German low-voltage
feeders with an overload-decision metric evaluated at each model's 95th percentile for consumer
peaks and its 5th for producer peaks. The two models that came first and second on consumer peaks in
the quantile variant of that metric — Chronos-Bolt, a time-series foundation model, and a
weekly-naive baseline — turned out to have 90% ranges containing the true value only 62% and 58% of
the time across the series as a whole, and 43% and 49% of the time at the consumer peaks themselves.
In the results of [Kaas et al. (2026)](https://arxiv.org/abs/2607.01966), a model that understates
its uncertainty raises fewer false alarms, so it scores well on a threshold-crossing test while
being exactly the model an operator should not trust near a capacity limit.

### 6. In the low-voltage papers reviewed up to 2020, weather forecasts were barely used and weather ensembles almost never

Of the 221 low-voltage forecasting papers [Haben et al. (2021)](https://arxiv.org/abs/2106.00006)
reviewed up to 2020, three used a weather *forecast* and none used an *ensemble* of weather
forecasts. [Pinheiro et al. (2023)](https://doi.org/10.1016/j.apenergy.2022.120493), published after
that review closed, is a fourth paper using a real weather forecast — but its inputs are single
point forecasts rather than an ensemble. [Pinheiro et al.
(2023)](https://doi.org/10.1016/j.apenergy.2022.120493) therefore overturns the first half of that
finding but not the second: even the largest deployment in this review used no weather ensemble.

## Three findings that cut against this project's plan

Three results in this literature point against Flexpectation's plan, and we intend to test all three
rather than route around them.

### Finer-grained weather data has not always paid

[Browell and Fasiolo (2021)](https://arxiv.org/abs/2103.10335) added spatial statistics derived from
gridded numerical weather prediction to their model of 14 grid supply point groups in GB. Those
spatial statistics helped significantly in two of the 14 regions, hurt significantly in three, and
made no measurable difference in the remaining nine. Weather itself was worth a great deal to them —
adding wind and irradiance cut their pinball loss by 40% overall, and by 60% in North Scotland
against 10% in London — so the question is not whether weather matters but whether *finer* weather
does. Artificial Forecasting obtained postcode-level weather forecasts for two wind-connected
primary substations after their wind-connected models had performed poorly, and reported that the
postcode-level forecasts "did not notably improve model performance", naming better weather data as
a next step.

### Weather has bought less than expected at low voltage in the past

[Haben et al. (2019)](https://doi.org/10.1016/j.ijforecast.2018.10.007) tested 100 real low-voltage
feeders with both forecast and observed temperature, and found that temperature had no effect on
forecast accuracy, or a negative one. [Haben et al.
(2019)](https://doi.org/10.1016/j.ijforecast.2018.10.007) used data collected in 2014 and 2015, and
we expect how much weather matters at a substation to be changing quickly, because the thing that
makes a substation weather-dependent is embedded solar generation and heat pumps, and there are far
more of both on the network now than there were then. That is a prediction, though, not a
measurement — and the Scottish primary-substation sensitivities of [Fox et al.
(2018)](https://doi.org/10.34890/134), measured on ten years of data ending in the mid-2010s and
described under "What GB networks have already built" below, say weather was already moving primary
substation demand well before the mid-2010s.

### A model trained on none of NGED's data may match a model trained on all of it

[Kaas et al. (2026)](https://arxiv.org/abs/2607.01966) tested Chronos-2, a general-purpose
time-series model that had never seen their data, against models trained on the first 160 of those
feeders and scored, like Chronos-2, on all 200. Chronos-2 beat every purpose-trained competitor on
mean absolute error, 3.8 kW against 4.2 kW. If heavily engineered models do not clearly beat an
off-the-shelf model given none of the target network's data, that is important information about the
value of any programme of heavy engineering.

## What GB networks have already built

**Scottish and Southern Electricity Networks' TRANSITION** (Network Innovation Competition,
Oxfordshire; its load-forecasting deliverable reported 2021) is the closest precedent for
Flexpectation's method. It forecast net load at 13 primary substations, their bulk supply points and
their 33 kV and 11 kV feeders, from 30 minutes to 10 days ahead. TRANSITION split each substation's
net load — demand minus whatever generation behind that substation happened to produce — into demand
and generation, forecast the two separately, then recombined them. Two things TRANSITION did not set
out to do are what Flexpectation adds: its ensemble covered only the first four days, so from day
four to day ten a single deterministic forecast was all it had, whereas NGED acts out to fourteen;
and it was a 13-substation trial rather than a network-wide deployment. Everything else in its
design is the shape Flexpectation is building.

**[NGED's own Electricity Flexibility and Forecasting System,
EFFS](https://smarter.energynetworks.org/projects/wpden03/)** (Network Innovation Competition,
2018–2021, budgeted at £3,338,798 and spending £2,948,281) forecast grid supply points, bulk supply
points, primary substation transformers and generation sites from an hour to six months ahead,
feeding automated constraint identification. Its evaluation independently selected XGBoost as the
best balance of accuracy against effort — the same starting point Flexpectation uses. EFFS's
forecasts were deterministic, with no uncertainty attached, which is the step this project adds.

**[SSEN FastTrack](https://smarter.energynetworks.org/projects/10166254/)** (Strategic Innovation
Fund, Alpha 2025–2026) models how the distribution connections queue — around 180 GW and growing —
will load the network, which is a planning question rather than the operational one Flexpectation
asks. **[SP Energy Networks'
Predict4Resilience](https://smarter.energynetworks.org/projects/10061710/)** drives a probability
distribution of network faults per district from an ensemble weather forecast, up to seven days
ahead, in a tool built with control-room engineers, which its Beta phase is taking into live trials
— the GB precedent for putting ensemble-derived distributions in front of network operators.

**The Dutch operator Alliander runs [OpenSTEF](https://lfenergy.org/projects/openstef/)**, an
open-source forecasting stack under the Linux Foundation's LF Energy umbrella, in live operation
across thousands of grid connection points to 48 hours ahead. It is the only operational network
forecasting system in this review whose code can be read rather than inferred from a deliverable,
**The second is far larger than any project here.** Enedis, the French distribution network
operator, has forecast consumption and generation at all 2,300 of its high-voltage-to-medium-voltage
substations since 2015, and is now extending that to a finer geographic grid ([Cordier et al.
(2024)](https://doi.org/10.1049/icp.2024.2058), whose abstract we read rather than the full paper).
Forecasting operationally at the scale Flexpectation reaches in 2027 is therefore a decade old
somewhere else, which is reassuring about the engineering and says nothing about the forecast
quality, because the abstract we read reports no accuracy figures.

### Northern Powergrid's Artificial Forecasting is further ahead, and sets the bar

**One concurrent project matters more than any paper here.** Artificial Forecasting is an Ofgem
Strategic Innovation Fund programme, with about £3.9m of grant across its three phases, run by
Northern Powergrid with Faculty, EV.energy and Oaktree Power, the final Beta phase running to
February 2027. Artificial Forecasting does much of what Flexpectation does at primary substations,
it also covers secondary substations, which Flexpectation does not, and at the time of writing it is
further ahead than Flexpectation.

**Artificial Forecasting has run operationally through a full winter flexibility procurement
cycle.** A forecasting service for primary substations is deployed and has passed the network's
architecture review board, data governance and information security checks for its current
deployment. It was used operationally by Northern Powergrid's System Forecasting team through a full
winter flexibility procurement cycle to support week-ahead dispatch decisions. It produces
half-hourly probabilistic forecasts with 5th-to-95th-percentile bands, flags forecast exceedances of
firm capacity, and is benchmarked against the network's existing growth-based and persistence
methods and a rolling four-week baseline. The deliverable states, without publishing the figures
behind it, that performance did not materially degrade on average across the 11-day horizon.
Artificial Forecasting's value case puts whole-life net present value at around £60m for one
network, or £250m if three further networks adopt it, driven by a 3% reduction in spending on
reinforcement — building bigger transformers and cables — in the current price-control period rising
to 6% in the next, and a 25% improvement in the cost-effectiveness of contracted flexibility. The
project pairs those figures with the appropriate qualification: it reports early Beta evidence, from
one winter procurement cycle, supporting the performance assumptions behind the value case, which
"remains appropriate, subject to further validation".

**Artificial Forecasting is independent evidence that short-term substation forecasting is
operationally useful**, that networks will change their procurement process around it, and that a
benefits case has been made and accepted. Because it is public, operational and benchmarked against
a real incumbent method, it also sets the clearest available bar for what "working" looks like.
Artificial Forecasting's core intellectual property is to be made available royalty-free to other GB
networks, and we would rather build on it than rebuild it — a shared evaluation protocol between two
GB networks would be worth more to both than two separate ones.

## Set against this literature, what we plan is ambitious, and here is why we think it can be done

**Measured against the studies we found, the plan sits outside the published literature in five ways
at once — which is a statement about where the gaps in our search are, rather than about the quality
of the work that fills the rest of the field.** No study in this review drives a substation forecast
from a weather ensemble across a 14-day horizon. None models the upper tail explicitly at substation
level; the one study that models it explicitly at all works on regions far larger than a substation.
None puts unmetered generation inside a probabilistic forecast at substation level over a multi-day
horizon, though unmetered generation, probabilistic forecasting at substation level and a multi-day
horizon each exist on their own. None tracks the available capacity of a mixed fleet of solar, wind
and dispatchable generators at one distribution network, or measures whether doing so improves the
forecast. None turns switching-contaminated history into a useful input rather than deleting it,
rewriting it, or absorbing the cost of leaving it in. Flexpectation attempts all eight problems
above, across four families of model:

- a heavily-tuned version of the gradient-boosting approach that wins most tabular forecasting
  competitions, and which NGED's own EFFS project independently selected;
- weather and time encoders pre-trained on large datasets, so that a model for one substation can
  start from what has been learned across all of them;
- models that use the connectivity map explicitly;
- differentiable physics — building known physical behaviour directly into the model, so that it has
  to learn only what the physics cannot supply: the response of a solar panel and of a wind turbine
  on the generation side, and the thermal response of buildings on the demand side.

**Only the first of those four strands — the heavily-tuned gradient-boosting model — is in scope for
version one.** The other three strands belong to the network-wide scale-up from 2027, as does the
disaggregation of unmetered generation.

**The main reason for attempting all eight at once is that they may be one problem rather than
eight.** A switching event, a turbine out for repair and a stuck meter all surface in the same
place: as a discrepancy between what a substation metered and what the weather and the calendar say
it should have metered. Every study reviewed above that touches more than one of the eight solves
them as a pipeline. In every case one stage's output is frozen before the next stage sees it, so an
error made early cannot be corrected later and the forecast error never gets to tell the capacity
estimator it was wrong.

**So the question we want to answer is whether one model that estimates capacity, switching state
and demand together beats that pipeline.** NGED's specification leaves room for it, asking that
these phenomena be handled rather than that they be handled explicitly.

**The first reason for confidence is that experiments are nearly free.** The core forecast already
exists and runs today, on an experiment framework that makes one more experiment cost compute time
rather than staff time. That is what makes it realistic to run on the order of hundreds of
machine-learning experiments a month, and it is the same argument the introduction to this review
makes: if roughly nine ideas in ten fail, the only affordable way to find the one that works is to
make each attempt cheap.

**Several of the four model families will not work, and that is what makes them research directions
rather than engineering tasks.** The honest expectation is that some deliver clearly, some produce a
negative result worth publishing, and some are abandoned. Both NGED and this project count a
negative result as an outcome: evidence that switching cannot be recovered from power data alone,
for instance, would be worth having, because it would justify extracting switching labels from
operational systems instead of continuing to look.

## What this review excluded, and why

Six bodies of work were deliberately left out, and the full review says why for each:
behind-the-meter solar disaggregation below NGED's level of aggregation, general concept-drift
detection, network topology detection from synchrophasor measurements, the bulk of the low-voltage
forecasting literature (covered second-hand through the [Haben et al.
(2021)](https://arxiv.org/abs/2106.00006) review of 221 papers), differentiable physics applied to
substation demand, and — the largest omission — the electrification literature on heat pumps,
chargers and domestic batteries, which is large enough to need a review of its own.

**CIRED is the venue this audience is most likely to read — it is where European distribution
network operators publish their own operational work, so CIRED is where a claim of ours is most
likely to be contradicted.** We therefore searched it in full: the titles and abstracts of every
paper in the CIRED main conferences and workshops of 2017 and 2020 to 2025, about 3,600 of them; the
2018 and 2019 proceedings, which are not indexed, by keyword against their open full-text archive;
and the 305 papers accepted for the Brussels workshop of June 2026 by title, those proceedings not
yet being published. Nothing there contradicts what this review reports missing, and the absences
are worth stating, because CIRED is where the counter-example would have been. Those absences are as
good as the search behind them: a method a paper uses without naming it in its title or abstract
would not have surfaced. No CIRED paper drives a load or generation forecast from a weather
ensemble. Fourteen forecast probabilistically at all, of which one is at substation scale —
[Mesarcik et al. (2025)](https://doi.org/10.1049/icp.2025.1968), day-ahead, on ten years of
measurements from 312 Dutch substations. Nothing scores the upper tail, nothing keeps
switching-contaminated history usable, and nothing estimates how much of a generator's capacity is
available.

## Publishing results that others can compare against

**We will publish the telemetry, the evaluation protocol, the metric definitions and the code that
computes them, so that someone outside the project can check the results.** **Energy forecasting's
own senior figures say that published results in the field cannot be compared with each other.**
[Hong et al. (2020)](https://doi.org/10.1109/OAJPE.2020.3029979), a review written by six of the
field's most senior figures, concludes that "most papers can never be replicated, because the data
have never been published". **Incomparable results are what this review ran into at every one of the
eight problems.** Even the eight studies in the one table above, all forecasting electricity demand
somewhere on a network, differ in target, level, horizon and weather assumption in nearly every row,
so almost none of them can be compared directly with any other.

This review makes nine commitments to publish or report. Collected in one place, they are:

- **Every ratio comes with its reference forecast, the population it was scored on, and the number
  of ensemble members that produced it.** [Weigel et al. (2007)](https://doi.org/10.1175/MWR3280.1)
  show that a ranked probability skill score is biased downwards by an amount that depends on
  ensemble size, so a score from our 51 members is not comparable with one from a study using ten
  until their correction is applied. We apply it.
- **Accuracy is reported separately for each class of asset** — grid supply points, bulk supply
  points, primary substations and metered generators — each against its own stated naive baseline,
  because a single project-wide accuracy target would mean different things at different levels.
- **The fraction of series that beat their naive baseline is published alongside the average
  error**, never the average alone.
- **The battery, the gas generator and the biofuel plant are reported separately** from the wind and
  solar sites, because the battery, the gas generator and the biofuel plant are dispatched on market
  signals no weather forecast contains.
- **A peak-aware score is reported alongside a proper scoring rule**, never instead of one.
- **The tail is scored with a threshold-weighted continuous ranked probability score**, weighted
  above a fixed per-series threshold set at the 99th percentile of that series' own measured
  history, rather than by selecting the periods in which an exceedance happened. The obvious alternative — keep
  only the periods in which load crossed the limit, and score those — is not merely noisy but
  biased: [Lerch et al. (2017)](https://doi.org/10.1214/16-STS588) show that choosing which periods
  to score on the basis of what happened rewards a forecaster who over-predicts extremes, and can
  rank a deliberately biased forecast above an honest one. [Gneiting and Ranjan
  (2011)](https://doi.org/10.1198/jbes.2010.08110)'s threshold-weighted score puts the emphasis
  inside the score instead, and stays a proper scoring rule while doing it.
- **Coverage — how often reality fell inside the range the forecast claimed — is broken down by
  season, by forecast lead time and by how heavily loaded the substation was.** A coverage figure
  averaged over a year can read as a healthy 90% while being 99% in the quiet months and 70% at the
  winter peaks, and the winter peaks are the only periods NGED buys flexibility for. Breaking it
  down is the point, and conformal prediction does not remove the need to: [Foygel Barber et al.
  (2020)](https://doi.org/10.1093/imaiai/iaaa017) prove that a distribution-free guarantee holds
  only on average across all conditions, never separately for the conditions that matter, so a
  conformal forecast can promise 90% coverage overall while failing at the peaks.
- **Each metered generator's series is normalised by its estimated effective capacity** before
  training — unless the comparison described under problem 3 shows the normalisation is not needed —
  and that estimate is tracked as it changes.
- **Negative results are published too**, including whether an off-the-shelf model given none of our
  data matches our own, and whether sustained experimentation stops yielding improvements.

## References

Every source cited above, in alphabetical order by first author. The full review cites 40 further sources that this summary does not.

- Bian, Y., Zheng, N., Zheng, Y., Xu, B. and Shi, Y. (2024). [Predicting Strategic Energy Storage
  Behaviors](https://doi.org/10.1109/TSG.2023.3303469). *IEEE Transactions on Smart Grid*.
- Bouman, R., Schmeitz, L., Buise, L., Heres, J., Shapovalova, Y. and Heskes, T. (2024). [Acquiring
  Better Load Estimates by Combining Anomaly and Change Point Detection in Power Grid Time-series
  Measurements](https://arxiv.org/abs/2405.16164).
- Browell, J. and Fasiolo, M. (2021). [Probabilistic Forecasting of Regional Net-load with
  Conditional Extremes and Gridded NWP](https://arxiv.org/abs/2103.10335).
- Cordier, G. et al. (2024). [Methods and techniques used to produce electricity forecasts on
  Enedis’ distribution network at a finer grid than the HV/MV
  substation](https://doi.org/10.1049/icp.2024.2058). *IET Conference Proceedings*.
- Dantas, G. and Browell, J. (2026). [Seamless Short‐ to Mid‐Term Probabilistic Wind Power
  Forecasting](https://doi.org/10.1002/we.70079). *Wind Energy*.
- Doubleday, K., Van Scyoc Hernandez, V. and Hodge, B. M. (2020). [Benchmark probabilistic solar
  forecasts: Characteristics and recommendations](https://doi.org/10.1016/j.solener.2020.05.051).
  *Solar Energy*.
- Fildes, R. (2020). [Learning from forecasting
  competitions](https://doi.org/10.1016/j.ijforecast.2019.04.012). *International Journal of
  Forecasting*.
- Fox, J., Plecas, M., Neilson, D., Cannon, D. and Parr, J. (2018). [Analysis of local demand trends
  and forecasting through weather correction and benefit to DSO transistion and microgrids Analysis
  of local demand trends and forecasting through weather correction and benefit to DSO transition
  and microgrids](https://doi.org/10.34890/134). *AIM*.
- Foygel Barber, R., Candès, E. J., Ramdas, A. and Tibshirani, R. J. (2020). [The limits of
  distribution-free conditional predictive inference](https://doi.org/10.1093/imaiai/iaaa017).
  *Information and Inference: A Journal of the IMA*.
- Gilbert, C., Browell, J. and Stephen, B. (2023). [Probabilistic load forecasting for the low
  voltage network: forecast fusion and daily peaks](https://arxiv.org/abs/2206.11745).
- Gneiting, T. and Ranjan, R. (2011). [Comparing Density Forecasts Using Threshold- and
  Quantile-Weighted Scoring Rules](https://doi.org/10.1198/jbes.2010.08110). *Journal of Business &
  Economic Statistics*.
- Haben, S., Giasemidis, G., Ziel, F. and Arora, S. (2019). [Short term load forecasting and the
  effect of temperature at the low voltage level](https://doi.org/10.1016/j.ijforecast.2018.10.007).
  *International Journal of Forecasting*.
- Haben, S., Arora, S., Giasemidis, G., Voss, M. and Greetham, D. V. (2021). [Review of Low Voltage
  Load Forecasting: Methods, Applications, and Recommendations](https://arxiv.org/abs/2106.00006).
- Hertel, M., Pütz, S., Kolar, J., Schäfer, B., Mikut, R. and Hagenmeyer, V. (2026). [A Benchmark
  for Electrical Load Forecasting Across Grid Levels: Time-Series Transformers Outperform
  Established Methods](https://arxiv.org/abs/2607.15705).
- Hong, T., Pinson, P., Wang, Y., Weron, R., Yang, D. and Zareipour, H. (2020). [Energy Forecasting:
  A Review and Outlook](https://doi.org/10.1109/OAJPE.2020.3029979). *IEEE Open Access Journal of
  Power and Energy*.
- Hyndman, R. J. (2020). [A brief history of forecasting
  competitions](https://doi.org/10.1016/j.ijforecast.2019.03.015). *International Journal of
  Forecasting*.
- Jumper, J. (2024). [Nobel Week
  interview](https://www.nobelprize.org/prizes/chemistry/2024/jumper/interview/).
- Kaas, B., Treutlein, M., Gerber, H. B., Neumann, O., Phatthanakhuha, C., Resch, O., Mikut, R. and
  Hagenmeyer, V. (2026). [Probabilistic Low-Voltage Peak Load Forecasting with Time Series
  Foundation Models Evaluated on Application-Oriented Metrics](https://arxiv.org/abs/2607.01966).
- Kleinebrahm, M. et al. (2026). [Energy-Arena: A Dynamic Benchmark for Operational Energy
  Forecasting](https://arxiv.org/abs/2604.24705).
- Lerch, S., Thorarinsdottir, T. L., Ravazzolo, F. and Gneiting, T. (2017). [Forecaster’s Dilemma:
  Extreme Events and Forecast Evaluation](https://doi.org/10.1214/16-STS588). *Statistical Science*.
- LF Energy. [OpenSTEF](https://lfenergy.org/projects/openstef/).
- Ludwig, N., Arora, S. and Taylor, J. W. (2023). [Probabilistic load forecasting using
  post-processed weather ensemble predictions](https://doi.org/10.1080/01605682.2022.2115411).
  *Journal of the Operational Research Society*.

- Mesarcik, M., Loke, J., Wildeboer, J. and Lucassen, B. (2025). [Probabilistic day-ahead power
  forecasting in the medium-voltage grid using state space
  models](https://doi.org/10.1049/icp.2025.1968). *IET Conference Proceedings*.
- Messner, J. W., Pinson, P., Browell, J., Bjerregård, M. B. and Schicker, I. (2020). [Evaluation of
  wind power forecasts — An up-to-date view](https://doi.org/10.1002/we.2497). *Wind Energy*.
- Meyer, M., Kaltenpoth, S., Albers, H., Zalipski, K. and Müller, O. (2026). [TS-Arena: A Live
  Forecast Pre-Registration Platform](https://arxiv.org/abs/2512.20761).
- National Energy System Operator. [Embedded wind and solar
  forecasts](https://www.neso.energy/data-portal/embedded-wind-and-solar-forecasts).
- Northern Powergrid (2024). [Artificial Forecasting, Alpha
  phase](https://smarter.energynetworks.org/projects/npg_sif_006-1/).
- Northern Powergrid (2024). [Detecting LCTs from Smart Meter Consumption
  Data](https://smarter.energynetworks.org/projects/npg_nia_-49/).
- Paredes, G. and Vargas, L. (2017). [Adjustment of discrete load changes in feeder databases for
  improving medium‐term demand forecasting](https://doi.org/10.1049/iet-gtd.2017.0129). *IET
  Generation, Transmission & Distribution*.
- Pinheiro, M. G., Madeira, S. C. and Francisco, A. P. (2023). [Short-term electricity load
  forecasting—A systematic approach from system level to secondary
  substations](https://doi.org/10.1016/j.apenergy.2022.120493). *Applied Energy*.
- Ruhhütl, M., Schmaranz, R. and Dietrichsteiner, T. (2023). [Load and generation forecast on
  substation level](https://doi.org/10.1049/icp.2023.0476). *IET Conference Proceedings*.
- Scottish and Southern Electricity Networks (2021).
  [TRANSITION](https://ssen-innovation.co.uk/transition/).
- Scottish and Southern Electricity Networks (2025). [FastTrack, Alpha Round
  4](https://smarter.energynetworks.org/projects/10166254/).
- Shukla, S. and Hong, T. (2024). [BigDEAL Challenge 2022: Forecasting peak timing of electricity
  demand](https://doi.org/10.1049/stg2.12162). *IET Smart Grid*.
- SP Energy Networks (2023).
  [Predict4Resilience](https://smarter.energynetworks.org/projects/10061710/).
- Teng, S., Cambier van Nooten, C., van Doorn, J., Ottenbros, A., Huijbregts, M. and Jansen, J.
  (2023). [Near real-time predictions of renewable electricity production at substation level via
  domain adaptation zero-shot learning in sequence](https://doi.org/10.1016/j.rser.2023.113662).
  *Renewable and Sustainable Energy Reviews*.
- UK Power Networks. [NIA_UKPN0104](https://smarter.energynetworks.org/projects/nia_ukpn0104/).
- Viotti, O., Arnqvist, J. and Olauson, J. (2026). [Estimating Wind‐Power Capacity Time Series From
  Production Data Using a Power Curve Model and Quadratic
  Optimization](https://doi.org/10.1002/we.70136). *Wind Energy*.
- Weigel, A. P., Liniger, M. A. and Appenzeller, C. (2007). [The Discrete Brier and Ranked
  Probability Skill Scores](https://doi.org/10.1175/MWR3280.1). *Monthly Weather Review*.
- Western Power Distribution (2021). [Electricity Flexibility and Forecasting System
  (EFFS)](https://smarter.energynetworks.org/projects/wpden03/).
