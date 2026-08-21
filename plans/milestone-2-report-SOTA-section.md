# The current state of the art in energy forecasting

Before we discuss the literature, there is a very important caveat to admit up-front: In 2026, no honest review of the energy forecasting literature can claim to reveal the canonical "state of the art"! That is because (almost) all energy forecasting papers measure performance in different ways, against different datasets. It's like an international football tournament where every team plays by different rules, with different size goals.

Energy forecasting researchers have done great work over the years. But, unfortunately, the literature does not tell us how those approaches compare against each other, especially in messy "real world" energy forecasting scenarios. This isn't anyone's fault; it's a systemic failure. And the industry is already aware of this problem, and people are trying to fix it. But, at the time of writing, the literature cannot yet tell us the current state of the art solutions for the problems that NGED cares about.

Having said that, there have been some valiant attempts to compare multiple forecasting approaches against the same dataset. But none of these attempts directly address the main challenges relevant to Flexpectation. Before we discuss those attempts, we must emphasise two reasons for optimism:

Firstly, whilst we might not know exactly which algorithms provide the best forecasting performance, we do know how to go about finding out. There's no magic. Machine learning is an empirical science, and progress in it comes largely from testing many ideas under identical conditions and measuring carefully — because most ideas fail. John Jumper, who shared the 2024 Nobel Prize in Chemistry for his work on AlphaFold, puts the share of research ideas that fail at around 90%, and treats that rate as an ordinary and necessary feature of doing research rather than as evidence of doing it badly ([Nobel Week interview](https://www.nobelprize.org/prizes/chemistry/2024/jumper/interview/), 6 December 2024, from 14:12). If roughly one idea in ten survives contact with the data, ten attempts is simply the price of one result. So our task is to run hundreds of ML experiments, and then measure performance against the same dataset, using the same performance metrics. This, in turn, requires us to design and build a framework that makes it easy to run hundreds of ML experiments per month. At the time of writing, we have implemented the first version of this framework, and we will continue to evolve the framework over the course of the project.

Secondly - and perhaps most importantly - the fact that the industry doesn't yet know the state of the art is a huge opportunity for the Flexpectation project: We are in a very privileged position where we can try hundreds of ideas, and test the best ideas in the real world. We have a fantastic opportunity to make a significant contribution to the energy forecasting industry by publishing our "leaderboards of ML experiments", and hence help the industry as a whole to better understand how multiple approaches perform.

## What we read

We read eleven papers in full and drew on four more that were only partly available to us, each
flagged where it appears. We also read the published deliverables of five concurrent GB network
projects. The selection was deliberate rather than systematic: a paper earned its place by bearing
on a decision Flexpectation actually faces and by changing something we believed. Papers may be
missing for no better reason than that we did not find them, and the last section says what we
knowingly left out. A further group of papers is cited once each, for one specific result, rather
than reviewed.

Flexpectation is a Network Innovation Allowance project of £841,733, running from January 2026 to
March 2028. It forecasts 32 time series in NGED's trial area — 16 primary substations, two grid
supply points, two bulk supply points and 12 metered generators — at half-hourly resolution, 14 days
ahead, updated every six hours, with each forecast expressed as a set of quantiles rather than a
single number. From 2027 it scales towards roughly 2,500 series across NGED's network.

Almost every number in this review depends on where in the network it was measured, so it is worth
fixing the ladder first:

- **Grid supply point** — where the distribution network meets the transmission system, carrying
  hundreds of thousands of customers. NGED has 52.
- **Bulk supply point** — the level below, typically 132 kV to 33 kV. NGED has 271.
- **Primary substation** — 33 kV to 11 kV, typically a few thousand customers. NGED has 1,161, of
  which 16 are in the trial area.
- **Secondary substation** — 11 kV to 400 V, from tens to a few hundred customers. **Flexpectation
  forecasts none of these**, at either stage, though several of the studies below are at this level.
- **Feeder and individual customer** — the bottom of the ladder, and the level at which most of the
  low-voltage forecasting literature works.

## The best published results, and why they cannot be compared

Every entry below is best-in-class for the problem its authors set themselves, and almost none can
be compared directly with any other, because the target, the level, the horizon and the weather
assumption differ in nearly every row.

| Source | What they forecast | Level and scale | Horizon | Best result, and what it beat | Weather |
|---|---|---|---|---|---|
| [Browell et al. 2025 (HEFTCom)](https://doi.org/10.1016/j.ijforecast.2025.10.005) | Combined wind and solar output, GB | National, 2 wind and 1 solar fleet | Day-ahead | Winning team scored 22.18 MWh mean pinball loss against the organisers' quick-start benchmark of 53.58; their more competitive reference entry scored 25.38, and the next teams 23.18 and 24.64. Revenue of £88.9m against a £105.2m perfect-decision ceiling | Real forecasts, live |
| [Kaas et al. 2026](https://arxiv.org/abs/2607.01966) | Net load, Germany | 200 low-voltage feeders | 4 days | A general-purpose "foundation" model that had never seen the data beat every purpose-trained model on average error, 3.8 kW against 4.2 kW | Actual weather, after the fact |
| [Hertel et al. 2026](https://arxiv.org/abs/2607.15705) | Load, Germany and Portugal | Transmission, 200 feeders, 287 customers | 4 days | Best model beat a day-type persistence forecast by 59.6% at transmission level, 42.3% at low-voltage feeders, 23.3% at individual customers | Reanalysis and 1–3 h forecasts |
| [Browell & Fasiolo 2021](https://arxiv.org/abs/2103.10335) | Regional net load, GB | 14 grid supply point groups | Day-ahead | Their conditional tail model held the same risk with up to 24.6% less upward reserve than a fixed-tail alternative (3.2% at the least extreme level tested). Adding wind and irradiance cut pinball loss 40% overall — 10% in London, 60% in North Scotland | Real forecasts |
| [Pinheiro et al. 2023](https://doi.org/10.1016/j.apenergy.2022.120493) | Load, Portugal | 96,989 secondary substations | Day-ahead | 42–47% better than the reference benchmark at system level. **At substation level, beat a naive forecast on 83–87% of network-owned and 66–70% of customer-owned sites** | Real forecasts, 7–8 h old |
| [Gilbert et al. 2023](https://arxiv.org/abs/2206.11745) | Load, GB | 4 levels, primary substation to household | Day-ahead | Combining forecasts gained **0.0–0.4% averaged over all periods**, but **5.7–9.0% when restricted to peaks** | None at all |
| [Bouman et al. 2024](https://arxiv.org/abs/2405.16164) | Switch-event and anomaly detection, Netherlands | 180 primary substations | Not a forecast | Annual maximum and minimum load estimates within a 10% margin in 88% and 91% of cases | None |
| [SSEN TRANSITION 2021](https://ssen-innovation.co.uk/transition/) | Net load, Oxfordshire | 13 primary substations, plus bulk supply points and 11 kV feeders | 30 min to 10 days | All primary substation models below 10% mean absolute percentage error in calibration except two (13.4% and 19.7%); 94% of 11 kV feeders below 20% | **40-member ICON-EU ensemble to 4 days**, then one deterministic forecast to 10 days |
| **[Artificial Forecasting (Northern Powergrid)](https://smarter.energynetworks.org/projects/npg_sif_006-1/)** | **Demand and export at primary substations; net demand at secondary** | **551 primary, 729 secondary substations** | **Day- to week-ahead at primary; week- to month-ahead at secondary** | **About 8% lower mean absolute error of utilisation rate; 83% of the top 10% of demand values captured inside its 5th–95th percentile band; better than a rolling four-week baseline at 8 of 8 near-capacity substations** | **Real forecasts at primary; none at secondary** |

*Weather column:* "real forecasts" means the forecast that was genuinely available when the forecast
was made; "actual weather, after the fact" means observations or reanalysis that no forecaster would
have had. The difference is the subject of the second point below.

Four further sources carry findings rather than comparable scores, and we use them for those
findings alone. [Haben et al. 2021](https://arxiv.org/abs/2106.00006) reviewed 221 low-voltage
forecasting papers published to 2020: **three** used a weather *forecast* and **none** used a
weather ensemble. [Shukla & Hong 2024](https://doi.org/10.1049/stg2.12162), reporting the BigDEAL
competition across three neighbouring US distribution companies, found that team rankings on peak
*size* were the least correlated of the three tracks, while rankings on peak *timing* and peak
*shape* went together; only four of thirteen finalist teams beat the organisers' benchmark. The
[GEFCom2017 competition](https://doi.org/10.1016/j.ijforecast.2019.02.006) on hierarchical New
England load at 2–9 weeks is paywalled and everything we know of it is second-hand.
[Energy-Arena](https://arxiv.org/abs/2604.24705) is a live public leaderboard rather than a
competition — we could not extract the full paper and worked from its abstract and the running
platform, which today carries 24 deterministic challenges across prices, load, wind and solar.

The sharpest illustration comes from two papers published a fortnight apart, by overlapping groups
at the Karlsruhe Institute of Technology, on the same 200 German low-voltage feeders. [Kaas et al.
(2026)](https://arxiv.org/abs/2607.01966) and [Hertel et al.
(2026)](https://arxiv.org/abs/2607.15705) name different models as best. Inside Kaas et al. (2026),
mean absolute error and an overload-decision metric name different winners again. Neither
disagreement is a mistake: the two papers test different sets of models at different time
resolutions, and the two metrics answer different questions. Between them, Kaas et al. (2026) and
Hertel et al. (2026) are the clearest demonstration available that the choice of metric, dataset and
horizon decides which model wins.

### Three things decide what a headline number means

**The level of aggregation.** [Hertel et al.](https://arxiv.org/abs/2607.15705) ran the same models
against the same benchmark at three levels of the grid and beat it by 59.6% at transmission and
23.3% at individual customers. The model did not get worse; the problem got harder. A headline
percentage therefore says more about where it was measured than about the method, and this is the
single most important thing to take from this review, because it sets what to expect at NGED's
primary substations.

**Weather known after the event.** Two of the studies above use the weather as it was known
immediately afterwards — short-range forecasts issued one to three hours ahead, or reanalysis —
rather than the weather that was forecast days out. They do this deliberately, so that differences
between models are not swamped by weather-forecast error. That is the right choice for their
question and the wrong one for ours, because it removes the error that dominates beyond a day or
two, which is precisely the range NGED acts on. Their figures are upper bounds, not achievable
performance.

**Which periods are averaged.** [Gilbert et al.](https://arxiv.org/abs/2206.11745)'s forecast
combination looks worthless averaged across every half-hour of their test period and clearly worth
having at the daily peak — the same comparison, two answers. An average over every period is
dominated by the quiet ones, and the quiet ones are not why a network buys flexibility.

### Which published numbers do transfer

Only two kinds. **Ratios against a stated baseline on a stated population** transfer, provided the
baseline is stated. Those baselines differ far more than the prose in most papers suggests —
yesterday's value at the same time, the average of the last four weeks, a day-type persistence rule
and the long-run seasonal average all appear in the table above, and a percentage gain against one
baseline is not a percentage gain against another. **Errors normalised by something physical** also
transfer: an error expressed as a fraction of a substation's firm capacity or transformer rating
means the same thing at every substation, whereas an error expressed as a fraction of the load that
happened to occur does not. Absolute errors in kW or MW transfer to nothing, and none of the
absolute figures above should be read as a target for this project.

## What the literature does agree on

Six findings recur across the independent studies we read. They are findings about this literature,
not laws of nature: each is what several teams have measured on their own networks, and a network
that differs from theirs may well behave differently.

**1. In every study we read, sophisticated models beat simple models by much less than their authors
expected.** [Pinheiro et al. (2023)](https://doi.org/10.1016/j.apenergy.2022.120493), running a live
system across 96,989 Portuguese secondary substations, tuned a gradient-boosted tree by exhaustive
grid search. On the national series, the gradient-boosted tree scored 199 MW root-mean-square error
and the generalised additive model scored 191 MW, so the gradient-boosted tree was 4% worse than the
simpler model. Pinheiro et al. (2023) rejected the gradient-boosted tree on both accuracy and
interpretability, and kept the generalised additive model. Artificial Forecasting reached the same
conclusion about gradient-boosted trees when forecasting customer export at primary substations:
compared against the Bayesian ridge regression they went on to adopt, boosted trees "helped some
substations but harmed others", so they kept the Bayesian ridge regression. Their deliverable
reports the direction of that effect but not its size. Artificial Forecasting also tested a temporal
convolutional network at 729 secondary substations against a linear regression on the last four
weeks' periodic average: the neural network won by about one percentage point on the peak-focused
metric its authors treat as primary, lost on the daily-maximum and weekly-maximum metrics, and lost
on five of six metrics at the 24 substations with the worst data quality. In the BigDEAL
competition, only four of thirteen finalist teams beat the organisers' benchmark at all. None of
this says sophisticated models cannot win. It says that in these studies the margin was small, that
interpretable models stayed competitive, and that data quality and choice of metric mattered at
least as much as model complexity.

**2. In every study that measured it, forecast accuracy gets worse as you move down the voltage
levels.** [Hertel et al. (2026)](https://arxiv.org/abs/2607.15705) ran the same models against the
same benchmark at three levels and beat that benchmark by 59.6% at transmission level, 42.3% at
low-voltage feeders and 23.3% at individual customers. This is probably not a fact about forecasting
so much as a fact about averaging: a grid supply point aggregates hundreds of thousands of
customers, whose individual quirks cancel out, while a single feeder aggregates a few dozen, whose
quirks do not. Predicting the temperature of a kilogram of air is easier than predicting the motion
of each molecule in it, and for the same reason.

Two consequences follow. We will report accuracy separately for each class of asset — grid supply
points, bulk supply points, primary substations and metered generators — against a stated naive
baseline for each, because a single project-wide accuracy target would mean different things at
different levels. And we will not assume that rising error means falling usefulness. A forecast at a
primary substation may carry a larger percentage error than a forecast at a grid supply point and
still support flexibility procurement just as well, because what NGED needs from the forecast is a
reliable answer to "will this substation exceed its firm capacity?", and that question can be
answered well even when the load itself is hard to predict precisely. Whether decision-usefulness
really is flat across voltage levels is something this project can measure, and we intend to.

**3. In the one study that measured it at scale, a substantial minority of substations were not
forecast better by a trained model than by a naive rule.** Pinheiro et al. (2023) found that their
model beat a "same time yesterday" forecast at 83–87% of network-owned secondary substations but at
only 66–70% of customer-owned ones. Those customer-owned sites serve a single customer — one large
building or one industrial process — where load follows decisions no weather model can see. We do
not know that NGED's primary substations will behave the same way, and they may not, because a
primary substation aggregates far more customers than a Portuguese secondary substation does. But
the risk is real enough that we will report the fraction of series that beat a naive baseline
alongside average error, rather than reporting the average alone.

**4. Standard accuracy measures quietly reward forecasts that are useless for flexibility
procurement.** A forecast that predicts the right peak at the wrong time is penalised twice by mean
absolute error — once for the peak it predicted that did not happen, and once for the peak that
happened and it missed. A flat, featureless forecast avoids both penalties. Mean absolute error and
root-mean-square error therefore systematically favour smooth forecasts over peaky ones, which is
backwards for a network buying flexibility to keep load below a capacity limit. Two teams reached
this conclusion independently and acted on it. Pinheiro et al. (2023) scored their substation models
with Haben's adjusted error, a peak-aware measure, for exactly this reason. Artificial Forecasting
built a metric over the top 10% of demand values, normalised to transformer rating, and made it the
primary measure for comparing their models.

**5. A forecast can state its own uncertainty badly, and one scoring choice will not reveal it.**
[Kaas et al. (2026)](https://arxiv.org/abs/2607.01966) scored models on 200 German low-voltage
feeders with an overload-decision metric evaluated at each model's 95th percentile. The two models
that topped that metric on the consumer side turned out to have 90% ranges containing the true value
only 62% and 58% of the time across the series as a whole, and under half the time at the peaks
themselves. A model that understates its uncertainty raises fewer false alarms, so it scores well on
a threshold-crossing test while being exactly the model an operator should not trust near a capacity
limit. The same paper supplies its own counter-example: ranked on average error rather than on the
overload metric, the winning model was also the best calibrated, at 89.75% coverage against a
nominal 90%. Whenever we publish a probabilistic forecast, we will publish how often reality fell
inside its stated range.

**6. Weather forecasts are barely used at low voltage, and weather ensembles almost never.** Of the
221 low-voltage forecasting papers [Haben et al. (2021)](https://arxiv.org/abs/2106.00006) reviewed
up to 2020, three used a weather *forecast* and none used an *ensemble* of weather forecasts.
Pinheiro et al. (2023), published after that review closed, is a fourth paper using a real weather
forecast — but its inputs are deterministic point forecasts, so the ensemble half of the finding
survives the largest deployment since. Artificial Forecasting's published secondary-substation
results use no weather at all, because the weather archive available to them reached only 16 days
ahead while their forecasts were month-ahead.

### Three findings that cut against this project's plan

A review that found only support for its own programme would not be worth reading. Three results in
this literature point the other way. We state them here, and we intend to test all three rather than
route around them.

**Finer-grained weather data has not always paid.** [Browell and Fasiolo
(2021)](https://arxiv.org/abs/2103.10335) added spatial statistics derived from gridded numerical
weather prediction to their model of 14 GB grid supply point groups. Those spatial statistics helped
significantly in two of the 14 regions, hurt significantly in three, and made no measurable
difference in the remaining nine. Artificial Forecasting bought postcode-level weather forecasts for
two wind-connected primary substations where their models performed worst, and reported that the
postcode-level forecasts "did not notably improve model performance", naming better weather data as
a next step. Neither result says finer weather data never helps — Browell and Fasiolo (2021) found
that it did help, significantly, at two of their 14 regions. What both results say is that finer
weather data does not help everywhere, so the interesting question is *where* it helps. That
question is answerable, and answering it is part of this project: we expect finer weather data to
matter most where a substation's load is dominated by weather-driven generation or heating, which is
where NGED most needs the forecast to be right.

**Weather has bought less than expected at low voltage in the past.** [Haben et al.
(2019)](https://doi.org/10.1016/j.ijforecast.2018.10.007) tested 100 real low-voltage feeders with
both forecast and observed temperature, and found that temperature had no effect on forecast
accuracy, or a negative one. Haben et al. (2019) used data collected in the early 2010s, and we
expect the answer to be changing quickly, because the thing that makes a substation
weather-dependent is embedded solar generation and heat pumps, and there are far more of both on the
network now than there were then. A primary substation that was almost weather-independent ten years
ago may be strongly weather-dependent today. That is a prediction, though, not a measurement, and
measuring how much weather now explains at NGED's primary substations is one of the more useful
things this project can report.

**A model trained on none of NGED's data may match a model trained on all of it.** Kaas et al.
(2026) tested Chronos-2, a general-purpose time-series model that had never seen their data, against
models trained specifically on those 200 feeders. Chronos-2 beat every purpose-trained competitor on
mean absolute error, 3.8 kW against 4.2 kW. If our heavily engineered models do not clearly beat an
off-the-shelf model given none of our training data, that is important information about the value
of the whole experimental programme, and we will report it.

### An open question this review cannot settle

Finding 1 above said that sophisticated models beat simple models by much less than their authors
expected. There are two quite different explanations for that, and nothing we read separates them.

**Explanation one: substation demand has a hard limit on how well it can be predicted, and today's
models are already close to it.** Half-hourly load at a single substation is the sum of decisions
made by a few hundred customers — when they cook, when they charge a car, when a factory starts a
shift — and much of that is genuinely unpredictable from weather and calendar data, because nothing
in the weather or the calendar determines it. If this explanation is right, a simple model already
captures nearly all of the predictable part, a sophisticated model has almost nothing left to find,
and the small gains reported across this literature are the correct answer to the question.

**Explanation two: nobody has yet pushed hard enough to find out where that limit is.** The
sophisticated models in this literature are generalised additive models, gradient-boosted trees and
similar established methods, usually applied to a standard set of calendar and weather features.
Those are sensible, well-chosen tools. They are also not what a sustained modern machine-learning
effort looks like. AlphaFold, the protein-structure model that won a share of the 2024 Nobel Prize
in Chemistry, reached its result through several years of a large team running a great many
experiments against one fixed, public benchmark. That route is open to energy forecasting in
principle, but it is rare in practice, for structural reasons rather than any failing of the
researchers: a forecasting paper is typically written by a small team over months rather than years,
tests a handful of model configurations, and reports results on a dataset that no other paper uses.
Energy forecasting has therefore never accumulated the thousands of directly comparable attempts
that protein-structure prediction had accumulated before AlphaFold.

We hold explanation two loosely, and explanation one may well be the right one. Flexpectation is not
resourced like a large industrial research laboratory. What it is resourced to do is run many
experiments cheaply against one fixed benchmark, which is the part that matters for telling the two
explanations apart. If explanation one is right, sustained experimentation will converge quickly on
a small improvement over a naive forecast and then stop improving, however many further experiments
we run — and we will report that plainly. If explanation two is right, improvements should keep
arriving well past the point at which a smaller effort would have concluded there were none left to
find. Either answer is worth publishing, and the second would be worth more to the industry than to
this project alone.

## What GB networks have already built

Five concurrent or recent GB network-innovation projects bear on this work, and between them they
have built more of it than the academic literature has.

**[SSEN TRANSITION](https://ssen-innovation.co.uk/transition/)** (Network Innovation Competition,
Oxfordshire, reported 2021) is the closest precedent for Flexpectation's method. It forecast net
load at 13 primary substations, their bulk supply points and their 11 kV feeders, from 30 minutes to
10 days ahead. Uncertainty came from the 40 members of the German weather service's ICON-EU
ensemble, disaggregation split net load into demand and generation before recomposing it, and the
network connectivity map was used throughout — the project ranks "historical network connectivity
data availability" as being just as important as the demand measurements themselves. All but two of
its primary substation models calibrated below 10% mean absolute percentage error. Two limits define
what is left to do: the ensemble runs only to four days, with a single deterministic forecast
covering days four to ten, and the trial covered 13 substations rather than a network.

**[NGED's own EFFS](https://smarter.energynetworks.org/projects/wpden03/)** (Network Innovation
Competition, 2018–2021, £3,338,896) forecast grid supply points, bulk supply points, primary
substation transformers and generation sites from an hour to six months ahead, feeding automated
constraint identification. Its evaluation independently selected XGBoost as the best balance of
accuracy against effort — the same starting point Flexpectation uses. Its forecasts were
deterministic, with no uncertainty attached, which is the step this project adds.

**[UK Power Networks' NIA_UKPN0104](https://smarter.energynetworks.org/projects/nia_ukpn0104/)**
(2024–2026, £389,444, with Open Climate Fix and Sheffield Solar) infers the capacity of unmetered
solar sitting behind each primary substation from half-hourly substation load and weather, then
forecasts that generation at primary substations. It is the direct predecessor of the disaggregation
work described below, and Open Climate Fix is a partner in both.

**[SSEN FastTrack](https://smarter.energynetworks.org/projects/10166254/)** (Strategic Innovation
Fund, Alpha 2025–2026) is building a probabilistic load forecast substation by substation, rolled up
to a grid supply point view. **[SP Energy Networks'
Predict4Resilience](https://smarter.energynetworks.org/projects/10061710/)** drives a probability
distribution of network faults per district from an ensemble weather forecast, up to seven days
ahead, in an operational control room — not load forecasting, but the GB precedent for putting
ensemble-derived distributions in front of network operators. NGED also contributes to the Energy
Systems Catapult's DNO Forecasting Forum, whose published outputs include good-practice principles
for distribution-level short-term forecasting and a forecasting taxonomy written by NGED.

### Northern Powergrid's Artificial Forecasting

One concurrent project matters more than any paper here. Artificial Forecasting is an Ofgem
Strategic Innovation Fund programme, with about £3.9m of grant across its three phases, run by
Northern Powergrid with Faculty, EV.energy and Oaktree Power, the final Beta phase running to
February 2027. Its deliverables are published openly on the Energy Networks Association's Smarter
Networks Portal, though the Beta deliverables sit under a separate registration from the Alpha ones
linked above. It is doing much of what Flexpectation does at primary substations, it also covers
secondary substations, which Flexpectation does not, and at the time of writing it is further ahead.

**What Artificial Forecasting has achieved.** A forecasting service for primary substations is
deployed and has passed the network's architecture review board, data governance and information
security checks for its current deployment. It was used operationally by Northern Powergrid's System
Forecasting team through a full winter flexibility procurement cycle to support week-ahead dispatch
decisions. It produces half-hourly probabilistic forecasts with 5th-to-95th-percentile bands, flags
forecast exceedances of firm capacity, and is benchmarked against the network's existing
growth-based and persistence methods. Performance did not get much worse across an 11-day horizon on
average. Their value case puts whole-life net present value at around £60m for one network, or £250m
if three further networks adopt it, driven mainly by a 3% reduction in reinforcement spend in the
current price-control period rising to 6% in the next, and a 25% improvement in the
cost-effectiveness of contracted flexibility. Those are the figures from the Beta application; the
project reports that a year of live operation supports them, with measured savings still to
accumulate.

Artificial Forecasting is independent evidence that short-term substation forecasting is
operationally useful, that networks will change their procurement process around it, and that a
benefits case has been made and accepted. Because it is public, operational and benchmarked against
a real incumbent method, it also sets the clearest available bar for what "working" looks like.
Flexpectation is not repeating that work: the questions it takes on are the ones a deployment
programme has not needed to answer. Artificial Forecasting's core intellectual property is to be
made available royalty-free to other GB networks, and we would rather build on it than rebuild it.

## Three studies worth a closer look

### [Bouman et al. 2024](https://arxiv.org/abs/2405.16164) — switch-event detection at a Dutch network operator

This is the most directly useful paper in the review, because it takes on half of a problem that is
explicitly in this project's scope, and the part it leaves untouched is the part Flexpectation would
contribute. Working with Alliander on 180 primary substations at 15-minute resolution over roughly a
year, the authors detect the step changes caused when a cable fault or planned maintenance reroutes
part of a subgrid to a different substation — a step up at one, a step down at the other. Events run
from a few minutes to several months.

Four things transfer:

- **They detect on a residual, not on the load itself.** Alliander maintains an independent
  bottom-up estimate of each substation's load, reconstructed from customer telemetry and modelled
  profiles. They fit and rescale that estimate to the measured series, then hunt for step changes in
  the *difference* between the two. Normal daily and seasonal variation largely cancels, leaving a
  much cleaner signal. We have no bottom-up estimate, but we have our own forecast, which can play
  the same role.

- **They recover a missing sign, which solves a known problem in NGED's trial area.** Some Alliander
  substations measure only absolute current, so reverse flow appears as a rise rather than a sign
  change — the identical defect at ten of NGED's metered sites. Because their reference estimate is
  built from signed measurements, they take the sign from it. Any independently-signed reference
  would do.

- **They stratify evaluation by event length**, into four buckets from "15 minutes to 6 hours" up to
  "42 days or longer", because short anomalies are frequent while long switch events are rare but
  cover most of the affected data. Pooling the two would let the long events dominate any metric.

- **When their bottom-up estimate fails, the cause is usually wrong topology data**, not a bad
  algorithm — a warning about the network records that any disaggregation work depends on.

Their annual maximum and minimum load estimates land within a 10% margin in 88% and 91% of cases,
using deliberately interpretable methods because reinforcement decisions rest on the output. Their
purpose is capacity planning rather than forecasting, and Bouman et al. (2024) do not consider
feeding the switch events they detect into a demand forecast.

### [Gilbert, Browell & Stephen 2023](https://arxiv.org/abs/2206.11745) — why an annual average hides what happens at peak

Gilbert et al. (2023) forecast load at four levels of a hypothetical GB distribution hierarchy, from
a primary substation down to individual households, and combine a conventional half-hourly forecast
with a bespoke daily-peak forecast.

Averaged over every period, that combination gains 0.0–0.4% over the conventional forecast alone —
indistinguishable from nothing, and a result that would ordinarily end the investigation. Restricted
to the periods containing the daily peak, the same comparison gives 5.7% at the primary substation,
9.0% at secondary, 8.2% at feeder level and 6.0% at household level. The technique was always worth
having, and we know that only because the authors reported both numbers.

A second finding bears directly on the choice of metric. At household level during peak periods,
both of their conventional forecasts are worse than a trivial benchmark based only on the time of
day; only their fused forecast beats it. And the ability to predict peak *timing* falls away as you
move down the levels: better than 20% above the long-run seasonal average at the primary substation,
essentially zero at four of the feeders. Together, the peak-versus-average gap and the collapse in
peak timing are the strongest measured argument in this review for the tail and exceedance metrics
Flexpectation is building.

### [Pinheiro, Madeira & Francisco 2023](https://doi.org/10.1016/j.apenergy.2022.120493) — the closest analogue in a live setting

A production forecasting system at a Portuguese distribution network operator covers 96,989
secondary substations day-ahead, using real weather forecasts with a realistic 7–8 hour delay. It is
the only study in the review running in live production at national scale, and findings 1, 3 and 4
above all rest on it. Two of its lessons shape how we will report: the fraction of substations
beating a naive forecast belongs alongside any pooled average, and expectations for single-customer
sites should be set low from the outset.

One further result from it is worth taking: an ensemble over eight calendar regimes — the same model
and features, combined online — cut system-level root-mean-square error from 203 MW to 154 MW. It is
the cheapest positive result in the review.

## Gaps we did not find addressed, and where Flexpectation fits

Seven things we did not find addressed in the work reviewed above, academic or operational. All
seven bear on what this project is trying to do, whether as a requirement for the trial area or as
research for the network-wide scale-up. Most are questions a research paper has no reason to ask and
a deployed forecasting service has not yet needed to answer, and in several cases the authors and
engineers concerned name the gap themselves.

1. **Weather ensembles as the source of uncertainty, across the full horizon.** Finding 6 gives the
   literature count. GB practice is further ahead than the literature: SSEN's TRANSITION drove
   distribution-level uncertainty from a 40-member ensemble in 2021, though only to four days, with
   a single deterministic forecast covering days four to ten. [Taylor and
   Buizza](https://doi.org/10.1109/TPWRS.2002.800906) pushed all 51 ECMWF members through a load
   model for England and Wales daily demand at one to ten days ahead in 2002, and [Ludwig, Arora and
   Taylor](https://doi.org/10.1080/01605682.2022.2115411) revised that approach in 2023, adding a
   step we will need: raw ensembles are biased and under-dispersed, so they must be statistically
   calibrated before the load model sees them, or the resulting bands are wrong. What we did not
   find is ensemble-driven uncertainty at half-hourly resolution, per substation, across a full
   14-day horizon — and both Haben et al. (2021) and Ludwig et al. (2023) ask for exactly that in
   print. Haben et al. (2021) put it as a request "to use post-processed weather ensemble
   predictions to generate multi-step probabilistic forecasts of load at different levels of the
   low-voltage hierarchy".

2. **The upper tail, not the middle.** NGED's question is "how likely is load to cross this limit?",
   not "what is the most likely load?". Almost everything in this review optimises average accuracy,
   and HEFTCom, the largest competition here, scores only the 10th to 90th percentiles. The
   exception is instructive rather than reassuring: Browell and Fasiolo (2021) model the tail
   explicitly and set reserve at the 99.95th percentile — but they also find ordinary quantile
   regression stops being calibrated somewhere around the 1st and 99th percentiles, even with five
   years of half-hourly data across regions far larger than a substation. Our series are smaller and
   noisier, so our reliable range will be narrower, and a parametric tail is likely to be necessary
   rather than optional. They add a warning we have taken on: per-quantile pinball loss in the tail
   is too noisy to rank forecasting systems by, so publishing a 99th-percentile column would not by
   itself constitute a tail metric.

3. **A decision metric that holds risk constant, priced in pounds, at distribution level.** Most of
   this exists already in pieces. Browell and Fasiolo (2021) fix a risk appetite, compute the
   reserve volume each forecast would need to hold it, and compare — the harder half of the job,
   done at transmission level. Artificial Forecasting's Alpha work calculates the extra flexibility
   volume that forecast error would make a network procure: 20,536 kWh implied by a risk-aware
   forecast against 5,495 kWh actually needed, over two eight-day windows at one near-capacity
   substation. Its Beta phase goes further, flagging exceedances of firm capacity from the
   95th-percentile bound and scoring true- and false-positive rates against that threshold. What is
   still missing is the price: every published version of this is denominated in energy volumes.
   Faculty's appendix prices a safety margin against under-predicting periods already flagged, and
   names the exceedance a forecast misses entirely as an open item itself.

4. **Keeping switching-contaminated history usable.** Detection has been demonstrated at a real
   network operator, by Bouman et al. (2024), described above. The field then responds in one of two
   ways, and both change the target. Most delete the affected data: the main open low-voltage
   dataset ([FeederBW](https://arxiv.org/abs/2602.03521)) filters out feeders with topology changes
   by hand, warning that undetected ones remain, and [Huyghues-Beaufond et
   al.](https://doi.org/10.1016/j.apenergy.2019.114405) detect and remove structural breaks across
   342 UK medium-voltage feeders. A smaller strand rewrites it instead: [Paredes and
   Vargas](https://doi.org/10.1049/iet-gtd.2017.0129) correct six years of hourly data across 169
   real feeders to an "as if never switched" level and report better medium-term forecasts for it,
   and Artificial Forecasting does the same operationally, rescaling each step-change block onto the
   level of the most recent one so the history is kept rather than dropped — on the stated grounds
   that steps of that size "cannot be directly handled even by powerful nonlinear models". Gilbert
   et al. name adaptive handling of structural breaks as future work. All of this is defensible
   engineering. What we did not find is the alternative to correcting the target at all: feeding
   recent observations to the model as residuals against a switching-independent baseline — the
   difference between what was measured and what a model that ignores topology expected — so that a
   reading taken while the network was abnormally configured still carries information without a
   level correction having to be estimated first. Later, at the scale of NGED's full network, the
   aim is to reconstruct the demand each substation would have metered under its normal running
   arrangement. A negative result here would still be valuable: evidence that switching cannot be
   recovered from power data alone would strengthen the case for taking switching labels from
   operational systems instead — a route Artificial Forecasting has already identified, with work
   under way on how outage-planning records could feed the model.

5. **Separating out generation that nobody meters.** Where demand and generation are separated at
   all, the generation is usually metered: Artificial Forecasting models gross demand and customer
   export independently at primary substations, which is more than any paper here does. The
   unmetered solar and wind — the rooftop panels and small turbines that appear only as a dent in a
   substation's net flow — have to be estimated from that net flow. This is being worked on now: UK
   Power Networks' NIA_UKPN0104, with Open Climate Fix and Sheffield Solar, infers unmetered solar
   capacity behind each primary substation and forecasts that generation. In the peer-reviewed
   literature the nearest work stops one step short — [Kara et
   al.](https://doi.org/10.1016/j.segan.2017.11.001) and [Li et
   al.](https://doi.org/10.1109/TPWRS.2020.3035639) recover the solar signal from feeder-head and
   substation measurements without forecasting it, and the one benchmark we found on estimating
   installed capacity is at secondary substations, which is a level below ours; we read only its
   abstract. What remains open at primary substation level is the wind half, and doing all of it
   inside a 14-day probabilistic forecast.

6. **Forecasting the network as a network.** Topology enters this literature in essentially one
   form: as the summation constraint in hierarchical forecast reconciliation. [Nespoli et
   al.](https://arxiv.org/abs/1910.03976) apply it to real secondary substations and cabinets in a
   Swiss distribution grid and gain up to 10% in root-mean-square error at the top level, and [Ben
   Taieb et al.](https://doi.org/10.1080/01621459.2020.1736081) give the probabilistic version. That
   constraint carries no information about which substation neighbours which, and an abnormal
   running arrangement invalidates it by construction, which is why it is not sufficient here.
   Otherwise, information is shared across substations statistically rather than topologically —
   Artificial Forecasting pools model parameters across six load-profile clusters — and Gilbert et
   al. forecast four levels of a hierarchy separately before naming exploitation of that hierarchy
   as future work. SSEN's TRANSITION is the exception that shows the value: it used the connectivity
   map throughout and ranked that data as being as important as the demand measurements. NGED hold a
   map of which substations and metered generators connect to which. That makes it possible to
   forecast a bulk supply point both directly and by summing everything beneath it, and to treat the
   disagreement between the two answers as a check on both.

7. **Tracking how much generation is actually available.** Turbines go out for repair, inverters
   degrade and sites are curtailed. A substation whose 20 MW wind farm has been limited to 14 MW for
   a month is, for forecasting purposes, a different substation, and a model trained on nameplate
   ratings cannot see the difference. The technique is standard one level down, but always from the
   generator's own instrumentation — [available-power estimation for curtailed
   turbines](https://doi.org/10.5194/wes-6-111-2021), degradation and availability accounting for
   solar plant. Artificial Forecasting gets closest at substation level: its Alpha work calibrates
   each substation's forecast installed capacity down to the fraction actually generated over two
   years, and separately found that NESO's national generator-availability signal "almost
   universally substantially improved results" at wind-connected primary substations, while a
   feature tracking connected generation capacity over time did not help. What we found nowhere is a
   per-substation effective capacity estimated continuously from that substation's own net flow and
   tracked as it changes. NGED's specification asks us to track it over time and, optionally, to
   combine it with the forecast into a "prevailing conditions" view; we intend to use it to
   normalise each metered generator's series before training. The clearest published demonstration
   of why it matters is incidental: when Hornsea 1's export cable faulted partway through the
   HEFTCom competition, teams that forecast wind and solar separately adapted to the step change in
   available capacity, while those forecasting the combined total struggled and the organisers'
   benchmark, which ignored it, collapsed.

**How ambitious Flexpectation's research plan is, and the risk that ambition carries.** The seven
items above are not a shortlist to choose from. The plan is to attempt all of them alongside the
core forecast, across several families of model:

- a heavily-tuned version of the gradient-boosting approach that wins most tabular forecasting
  competitions, and which NGED's own EFFS project independently selected;
- weather and time encoders pre-trained on large datasets, so that a model for one substation can
  start from what has been learned across all of them;
- models that use the connectivity map explicitly;
- differentiable physics — building the known behaviour of a solar panel or a wind turbine directly
  into the model, so that it has to learn only what the physics cannot supply.

Only the first is v1 work; the rest belong to the scale-up, as do gaps 5 and 6. Physics-informed
models for solar generation exist, and one 2026 paper applies a differentiable temperature-demand
relationship at system level, but we found none applied to demand forecasting at a substation.

Attempting all seven means running on the order of hundreds of machine-learning experiments a month,
and that is possible only because of engineering already done. Most of the effort to date has gone
into a machine-learning operations framework built to current industry best practice, whose purpose
is to make one more experiment nearly free. Every experiment is fully specified by a config file. It
runs through the same pipeline that serves production, rather than a separate research copy of it.
It is tracked automatically from raw data through to result, and it lands in one comparable metrics
store. That machinery exists and works today; the leaderboard view over it is still being built. The
plan is affordable because the marginal experiment is cheap, not because the team is large.

Flexpectation's plan is riskier than a narrower one would be, and that is worth saying plainly.
Artificial Forecasting chose a focused agenda and delivered it into live operational use, which is
the right way to get a service running and is why its results are the firmest evidence in this
review. Flexpectation is running a live service for a 32-series trial area while attempting a wider
set of open research questions on a smaller budget. Several of the directions listed above will not
work — that is what makes them research directions rather than engineering tasks — and the honest
expectation is that some deliver clearly, some produce a negative result worth publishing, and some
are abandoned. Three things make that acceptable. The core forecast does not depend on any of the
seven; it exists and runs today. Each item is independently useful, so one failing does not strand
the others: switching detection, capacity estimation and disaggregation each improve the core
forecast on their own terms. And a failed experiment costs compute time rather than staff months,
which is why running a hundred of them is affordable and why most of them are allowed to fail.

There is one further contribution, about method rather than results. The central problem identified
at the start of this review is one this project is well placed to help with, and others have started
already: HEFTCom and Energy-Arena both compare methods on common data with a common metric, and
Energy-Arena keeps a live public leaderboard. Neither covers distribution-substation load, which is
the level NGED acts at, so we intend to follow their protocols where they apply rather than invent
our own. Some substation data is already public — NGED's [Connected Data
Portal](https://connecteddata.nationalgrid.co.uk/), and Northern Powergrid's release of five years
of demand at 486 primary substations — and publishing the telemetry behind our own experiments would
make the results reproducible by anyone, which is still rare in the substation literature, where
only 52 of the 221 low-voltage papers reviewed used any open dataset at all. Alongside it we will
publish the evaluation protocol, the metric definitions and the code that computes them, and a
leaderboard carrying every experiment we run. Artificial Forecasting is moving the same way, with
substation-level historical forecasts and model-performance metrics designed into its Open Data
Portal release, and a shared evaluation protocol between two GB networks would be worth more than
either alone.

## What this review excluded, and why

**Behind-the-meter solar disaggregation** is a large and active field, mostly working on US
smart-meter data at individual customer level. We excluded it as a body because most of it operates
below our level of aggregation, and kept anchor citations for when the disaggregation work begins.
The exclusion covers our reading list rather than the whole field: work at feeder aggregation and
above is real, and gap 5 above names it.

**Network topology detection** from high-resolution synchrophasor measurements is well developed,
but the measurements it needs are not available to this project. That exclusion does not cover gap
6, which is about using a connectivity map we already hold rather than inferring one.

**General concept-drift detection** was excluded because most of it addresses gradual drift and
model adaptation, whereas our problem is a discrete step change with a known physical cause. The
abrupt-drift and change-point strand of that literature is closer to our problem, and we intend to
read it properly before the switching work begins. The adaptive-model family is the live alternative
to detecting switching at all, and we will treat it as the arm to beat.

**Differentiable physics applied to substation demand forecasting** produced no strong result. There
is substantial work on physics-informed neural networks for power systems, including models that map
weather to solar output, and one 2026 paper applying a differentiable temperature-demand
relationship at system level. We found none applied to forecasting demand at a substation, where the
physics would be a panel and a turbine. Either our search terms were wrong or the intersection is
genuinely thin, and we would welcome a second opinion.

**The bulk of the low-voltage forecasting literature** is covered through the Haben et al. (2021)
review of 221 papers rather than read individually, which is the appropriate level of detail for
work that closes in 2020. The same lead author published an open-access book-length treatment in
2023, which is the better entry point for anyone following this up. We have not systematically
covered low-voltage work published since; where a specific question arises we go back to individual
papers.

**CIRED**, the distribution-network conference, is not represented here at all, and it is the venue
this audience is most likely to read. That is a gap in our search rather than in the field.

Finally, the sources we could not read in full. **GEFCom2017** remains paywalled and everything we
know of it is second-hand; nothing in this section rests on it. **Energy-Arena** we know from its
abstract and its running platform, not its full paper. **Taylor and Buizza (2002)** we read in part.
And a **2026 benchmark on estimating installed solar capacity at low-voltage substations** was
available only as an abstract, and should be read in full before the capacity-estimation work
begins.
