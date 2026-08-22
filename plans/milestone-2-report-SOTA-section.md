# The current state of the art in energy forecasting

Before we discuss the literature, there is a very important caveat to admit up-front: In 2026, no honest narrative review of the energy forecasting literature can claim to reveal the canonical "state of the art"! That is because (almost) all energy forecasting papers measure performance in different ways, against different datasets. It's like an international football tournament where every team plays by different rules, with different size goals.

Energy forecasting researchers have done great work over the years. But, unfortunately, the literature does not tell us how those approaches compare against each other, especially in messy "real world" energy forecasting scenarios. This isn't anyone's fault; it's a systemic failure. And the industry is already aware of this problem, and people are trying to fix it. But, at the time of writing, the literature cannot yet tell us the current state of the art solutions for the problems that NGED cares about.

Having said that, there have been some valiant attempts to compare multiple forecasting approaches against the same dataset. But none of these attempts directly address the main challenges relevant to Flexpectation. Before we discuss those attempts, we must emphasise two reasons for optimism:

Firstly, whilst we might not know exactly which algorithms provide the best forecasting performance, we do know how to go about finding out. There's no magic. Machine learning is an empirical science, and progress in it comes largely from testing many ideas under identical conditions and measuring carefully — because most ideas fail. John Jumper, who shared the 2024 Nobel Prize in Chemistry for his work on AlphaFold, puts the share of research ideas that fail at around 90%, and treats that rate as an ordinary and necessary feature of doing research rather than as evidence of doing it badly ([Nobel Week interview](https://www.nobelprize.org/prizes/chemistry/2024/jumper/interview/), 6 December 2024, from 14:12). If roughly one idea in ten survives contact with the data, ten attempts is simply the price of one result. So our task is to run hundreds of ML experiments, and then measure performance against the same dataset, using the same performance metrics.

Secondly - and perhaps most importantly - the fact that the industry doesn't yet know the state of the art is a huge opportunity for the Flexpectation project: We are in a very privileged position where we can try hundreds of ideas, and test the best ideas in the real world. We have a fantastic opportunity to make a significant contribution to the energy forecasting industry by publishing our "leaderboards of ML experiments", and hence help the industry as a whole to better understand how multiple approaches perform.

## What we read

This review was written for National Grid Electricity Distribution (NGED). We read fifteen papers in
full and drew on ten more that were only partly available to us — an abstract, or part of a paper.
Where one of those ten appears below, we say so. We also read the published deliverables of six
concurrent GB network projects. The selection was deliberate rather than systematic: a paper earned
its place by bearing on a decision Flexpectation actually faces and by changing something we
believed. Papers may be missing for no better reason than that we did not find them, and the section
"What this review excluded, and why" lists what we knowingly left out. A further group of papers is
cited once each, for one specific result, rather than reviewed.

One concurrent project is cited more than any paper: Northern Powergrid's Artificial Forecasting, an
Ofgem Strategic Innovation Fund programme whose Alpha and Beta deliverables are both public, and
which has its own section below. Three further sources carry findings rather than comparable scores,
and are drawn on throughout. [Haben et al. 2021](https://arxiv.org/abs/2106.00006) reviewed 221
low-voltage forecasting papers published to 2020. [Shukla and Hong
2024](https://doi.org/10.1049/stg2.12162) reports the BigDEAL competition across three neighbouring
US distribution companies. [Energy-Arena](https://arxiv.org/abs/2604.24705) is a live public
leaderboard rather than a competition — we could not extract the full paper and worked from its
abstract and the running platform, which today carries 24 deterministic challenges across prices,
load, wind and solar.

Almost every number in this review depends on where in the network it was measured, so here is the
voltage ladder of a distribution network, from the top down:

- **Grid supply point** — where the distribution network meets the transmission system, 400 kV or
  275 kV down to 132 kV. Hundreds of thousands of customers sit below one.
- **Bulk supply point** — 132 kV down to 33 kV or 66 kV. Tens of thousands of customers.
- **Primary substation** — 33 kV or 66 kV down to 11 kV. A few thousand customers.
- **Secondary substation** — 11 kV down to 400 V. Tens to a few hundred customers.
- **Feeder and individual customer** — the bottom of the ladder, at 400 V.

NGED owns 52 grid supply points, 271 bulk supply points and 1,161 primary substations, of which 16
are in the 32-series trial area. **Flexpectation forecasts no secondary substations**, neither in
the trial area nor in the network-wide scale-up, though several of the studies below do. GB is
separately divided into 14 *grid supply point groups*, each a whole distribution region containing
many grid supply points, and several studies below forecast those regions, which are far larger than
any single substation.

## How to read the numbers in this review

**Two kinds of published number transfer to a different network, and the rest do not.** A ratio
against a baseline transfers, but only if the paper says what the baseline was and which substations
it was averaged over. Those baselines differ far more than the prose in most papers suggests —
yesterday's value at the same time, the average of the last four weeks, a day-type persistence rule
and the long-run seasonal average all appear among the studies reviewed here, and a percentage gain
against one baseline is not a percentage gain against another. A skill score — how much less error a
forecast has than a stated benchmark, as a percentage — needs its benchmark named for the same
reason. Where the score is a probabilistic one computed from an ensemble, it depends on how many
members produced it: the ranked probability skill score is biased downwards for small ensembles, and
almost no paper states its ensemble size. **Errors normalised by something physical** also transfer:
an error expressed as a fraction of a substation's firm capacity or transformer rating means the
same thing at every substation, whereas an error expressed as a fraction of the load that happened
to occur does not. An absolute error in kilowatts or megawatts tells NGED nothing on its own,
because it depends entirely on how big the substation was, and none of the absolute figures below
should be read as a target for this project.

**Whether a study used the weather forecast a real forecaster would have had changes what its
numbers mean.** In the table under problem 1 below, "real forecasts" means the weather forecast that
was genuinely available when the power forecast was made; "actual weather, after the fact" means
observations, or a weather model re-run after the event, that no forecaster would have had. Two of
the studies below, [Kaas et al. (2026)](https://arxiv.org/abs/2607.01966) and [Hertel et al.
(2026)](https://arxiv.org/abs/2607.15705), both on the same 200 German low-voltage feeders, use the
second kind — short-range forecasts issued one to three hours ahead, or reanalysis. Hertel et al.
(2026) do so deliberately, because their "primary goal is to compare models under fair conditions,
which we achieve by using the same data for all"; Kaas et al. (2026) use the weather their dataset
carries, and note that moving to real four-day forecasts "will likely also introduce significantly
higher error". Either way the figures are upper bounds rather than achievable performance, because
they remove the error that dominates beyond a day or two — precisely the range NGED acts on.

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
results to tabulate, and the second is the most mature field on the list. For most of the remaining
six we found no published result that could be compared against anything, so those are described in
prose: the absence is itself the finding.

Everything below is what our search surfaced as most relevant to NGED, not a ranking of the field:
every study answers the problem its own authors set, and they set different problems. The eight are
not a shortlist to choose from — the plan is to attempt all of them, for a reason the last part of
this review sets out: they may turn out to be one problem rather than eight.

### 1. Probabilistic forecasts of net demand at substations

**The problem.** Forecast net demand — demand minus whatever generation sits behind the substation —
at every grid supply point, bulk supply point and primary substation, half-hourly, 14 days ahead,
updated every six hours, as a range of possible loads with a probability attached to each rather
than as a single number. NGED acts on the forecast one to ten days ahead, and the question it asks
of the forecast is "how likely is load to cross this substation's firm capacity — the load it can
carry safely with its largest transformer out of service?" rather than "what is the most likely
load?". This is the highest priority of the eight problems, and the other seven exist mainly to make
it better.

**The 14-day horizon sits at the edge of what a weather ensemble can supply.** [Buizza and
Leutbecher (2015)](https://doi.org/10.1002/qj.2619) put the lead time at which a weather ensemble —
the 51 slightly different forecasts the European Centre for Medium-Range Weather Forecasts (ECMWF)
runs from 51 slightly different starting conditions, whose spread shows how confident the forecast
is — stops beating a climatological distribution — the spread of weather actually observed on that
day of the year over many years — at 16 to 23 days. They measured that on upper-air variables rather
than on the near-surface temperature and irradiance that drive substation load, for which we would
expect a shorter horizon.

**What the literature reports.**

| Source | What they forecast | Level and scale | Horizon | Result, and what it was compared against | Weather |
|---|---|---|---|---|---|
| [Kaas et al. 2026](https://arxiv.org/abs/2607.01966) | Net load, Germany | Low-voltage feeder: 200 | 4 days | A general-purpose foundation model that had never seen the data beat every purpose-trained model on average error, 3.8 kW against 4.2 kW | Actual weather, after the fact |
| [Hertel et al. 2026](https://arxiv.org/abs/2607.15705) | Load, Germany and Portugal | Transmission, plus 200 low-voltage feeders and 287 individual customers | 4 days | Best model beat a day-type persistence forecast by 59.6% at transmission level, 42.3% at low-voltage feeders, 23.3% at individual customers | Reanalysis and 1–3 h forecasts |
| [Browell and Fasiolo 2021](https://arxiv.org/abs/2103.10335) | Regional net load, GB | Regional: 14 grid supply point groups | Day-ahead | Held the same risk with **up to 24.6% less upward reserve** than a fixed-tail alternative (note 1) | Real forecasts |
| [Pinheiro et al. 2023](https://doi.org/10.1016/j.apenergy.2022.120493) | Load, Portugal | Secondary substation: 96,989 | Day-ahead | 42–47% better than the reference benchmark at system level. **At substation level, beat a naive forecast on 83–87% of network-owned and 66–70% of customer-owned sites** | Real forecasts, 7–8 h old |
| [Gilbert et al. 2023](https://arxiv.org/abs/2206.11745) | Load, GB | Four levels: primary substation down to household | Day-ahead | Combining forecasts gained **0.0–0.4% averaged over all periods**, but **5.7–9.0% when restricted to peaks** | None at all |
| [SSEN TRANSITION 2021](https://ssen-innovation.co.uk/transition/) | Net load, Oxfordshire | Primary substation: 13, plus their bulk supply points and 11 kV feeders | 30 min to 10 days | **11 of 13 primary substation models below 10%** mean absolute percentage error when fitted (note 2) | 40-member ICON-EU ensemble to 4 days, then one deterministic forecast to 10 days |
| [Artificial Forecasting (Northern Powergrid)](https://smarter.energynetworks.org/projects/npg_sif_006-1/) | Demand and export at primary substations; active power at secondary | Primary substation: 551 with export data, 171 modelled; secondary: 729 | Day-ahead to 11 days at primary; week- to month-ahead at secondary | **About 8% lower mean absolute error** of utilisation rate than the network's existing method (note 3) | Real forecasts at primary; none in the published secondary results |
| [Ruhhütl et al. 2023](https://doi.org/10.1049/icp.2023.0476) | Load and generation, Austria | Substation | Day-ahead | **3 to 8% mean absolute percentage error**, varying with how industrial and how large the supplied area was; linear and Gaussian regression preferred over the alternatives tested (abstract only) | Not stated in the abstract |

*Notes.* **1.** The 24.6% saving is at the most extreme tail level Browell and Fasiolo tested, and
falls to 3.2% at the least extreme. **2.** The two SSEN TRANSITION models that missed 10% reached
13.4% and 19.7%, and 94% of the 11 kV feeders it built models for came in below 20%. **3.**
Artificial Forecasting also captured 83% of the top 10% of demand values inside its
5th-to-95th-percentile band, and beat its comparison benchmarks at all eight of the near-capacity
substations it was evaluated on.

**Even within this one table, the studies cannot be compared with each other.** The sharpest
illustration comes from two papers published a fortnight apart, by overlapping groups at the
Karlsruhe Institute of Technology, on the same 200 German low-voltage feeders. Kaas et al. (2026)
and Hertel et al. (2026) name different models as best. Inside Kaas et al. (2026), mean absolute
error and an overload-decision metric name different winners again. Neither disagreement is a
mistake: the two papers test different sets of models at different time resolutions, and the two
metrics answer different questions. "Publishing results that others can compare against", the last
section of this review, returns to what follows from that.

**One study in the table shows how much an annual average hides.** Gilbert et al. (2023) forecast
load at four levels of a hypothetical GB distribution hierarchy, from a primary substation down to
individual households — built by aggregating 742 smart meters, so their top level is, as they say
themselves, smaller than a real primary substation — and combined a conventional half-hourly
forecast with a bespoke daily-peak forecast. Averaged over every period, that combination gained
0.0–0.4% over the conventional forecast alone, indistinguishable from nothing, and a result that
would ordinarily end the investigation. Restricted to the periods containing the daily peak, the
same comparison gave 5.7% at the primary substation, 9.0% at secondary, 8.2% at feeder level and
6.0% at household level. Combining the two forecasts was always worth having, and we know that only
because Gilbert et al. (2023) reported both numbers.

**The same paper found that the ability to predict *when* the peak will happen falls away further
down the network.** At the primary substation, peak timing was predicted more than 20% more
accurately than a long-run seasonal average would have managed; at four of the feeders, no better
than that seasonal average at all. And at household level during peak periods, both of their
conventional forecasts were worse than a trivial benchmark based only on the time of day; only their
fused forecast beat it. Together, the peak-versus-average gap and the collapse in peak timing are
the strongest measured argument in this review for the tail and exceedance metrics Flexpectation is
building.

**The closest analogue to Flexpectation in a live setting is Portuguese.** Pinheiro et al. (2023)
run a production forecasting system covering 96,989 secondary substations day-ahead, using real
weather forecasts with a realistic 7–8 hour delay. It is the only study in this review running in
live production at national scale. Two of its lessons shape how we will report: the fraction of
substations beating a naive forecast belongs alongside any pooled average, and expectations for
single-customer sites should be set low from the outset.

**The cheapest positive result in this review also comes from that system.** Combining eight copies
of the same model, one fitted per calendar regime — weekday, weekend, public holiday and so on —
with the weights updated as new data arrived, cut system-level root-mean-square error by 24%.

**Where the gaps are: no study we found drives substation uncertainty from a weather ensemble across
a full 14-day horizon.** GB practice is further ahead than the academic literature here, but stops
short. [Taylor and Buizza (2002)](https://doi.org/10.1109/TPWRS.2002.800906), which we read in part,
pushed all 51 ECMWF members through a load model for midday demand in England and Wales at one to
ten days ahead in 2002, and [Ludwig, Arora and
Taylor](https://doi.org/10.1080/01605682.2022.2115411) revised that approach in 2023, adding a step
we will need: raw ensembles are biased and their spread is too narrow, so they look more certain
than they really are, and they must be bias-corrected before the load model sees them or the
resulting uncertainty bands are wrong. What we did not find is ensemble-driven uncertainty at
half-hourly resolution, per substation, across a full 14-day horizon — and both Haben et al. (2021)
and Ludwig et al. (2023) ask for exactly that in print. Haben et al. (2021) put it as a request "to
use post-processed weather ensemble predictions to generate multi-step probabilistic forecasts of
load at different levels of the LV [low-voltage] hierarchy".

**The ensemble itself is being replaced, which turns this gap into a question we can answer
directly.** ECMWF's own machine-learned ensemble,
[AIFS-ENS](https://doi.org/10.1038/s44387-026-00073-7), has been operational since 1 July 2025 with
51 members, 6-hourly to 15 days, and beats the physics ensemble on the majority of variables and
lead times; [GenCast](https://doi.org/10.1038/s41586-024-08252-9) beats it too. Flexpectation runs
on the physics ensemble today, and whether a machine-learned ensemble forecasts substation load
better is something we can measure.

**Almost every study here optimises average accuracy, but NGED's question is about the top of the
distribution.** The largest competition in this review, HEFTCom — described under problem 2 below —
scores only the 10th to 90th percentiles. Browell and Fasiolo (2021) is the one study here that
models the upper tail explicitly, and what they found is a warning rather than a reassurance: they
set reserve at a risk level of one part in two thousand — enough to cover all but about four hours a
year — but they also find that "below 1% and above 99% the forecasts based on quantile regression
only are not calibrated at any GSP Group. Therefore, these quantiles are not suitable for use in
decision-making", even with five years of half-hourly data across regions far larger than a
substation. Above those limits they switch to a fitted parametric tail. **How far into the tail a
forecast of a single substation stays trustworthy is an open question, and one this project can
answer.** Our series are smaller and noisier than the regions Browell and Fasiolo worked on, so we
expect a narrower reliable range, and a parametric tail is likely to be necessary rather than
optional. We will measure where ours stops and publish the answer, because a network buying
flexibility needs to know which percentile it can act on.

**A decision metric that holds risk constant and prices it in money has been published at
distribution level once, on a synthetic network.** [Bernecker et al.
(2025)](https://doi.org/10.1016/j.ijepes.2025.110713) fix the confidence level at which a network
operator acts, at 95%, and compare what two forecasts cost that operator in congestion management:
**3,102 euros a year using standard load profiles against 86 euros using a smart-meter-informed
forecast**, a 97% reduction, with a matching fall in voltage violations. They also give the exchange
rate NGED would want — a 1% cut in the standard deviation of forecast error is worth about 1.4% of
congestion-management cost. Two things keep the gap open: the network is a modified IEEE 33-node
test system rather than a real one, and what they compare is two *information levels*, not two
forecasting models, so the metric has never been used to rank one forecast against another at a real
substation. The rest of the metric exists in pieces. Browell and Fasiolo (2021) fix a risk appetite,
compute the reserve volume each forecast would need to hold it, and compare — the harder half of the
job, done across whole grid supply point groups. [Angus et al.
(2027)](https://doi.org/10.1016/j.epsr.2026.113545) bring that idea down to individual assets,
forecasting day-ahead how hard each of 644 low-voltage transformers in GB can safely be pushed, and
winning 10 to 12% more capacity than a fixed setting while the risk of overheating came out at
whatever percentile they asked for; we read their preprint rather than the published paper.
Artificial Forecasting's Alpha work calculates the extra flexibility volume that forecast error
would make a network procure: 20,536 kWh implied by a risk-aware forecast against 5,495 kWh actually
needed, over two eight-day windows at one near-capacity substation. Its Beta phase goes further,
making exceedance true- and false-positive rates key metrics; its Alpha phase already scored
precision and recall for the half-hours at or above 90% of a substation's firm capacity.

**What is still missing is the price on a real network.** Meteorology has priced forecast decisions
this way for decades: [Richardson (2000)](https://doi.org/10.1002/qj.49712656313) computed the
relative economic value of the ECMWF ensemble across the whole range of ratios between the cost of
acting on a forecast and the loss avoided by acting. Richardson's relative-economic-value curve is
the right shape for NGED's problem, because each substation has its own firm capacity and its own
cost of being wrong, so a single assumed cost ratio is the thing to avoid. Every published version
of it on a real distribution network, though, is denominated in energy volumes or in spare capacity
rather than in money. Artificial Forecasting does put a price on its service, but that is a business
case for a programme rather than a score that holds risk constant and can rank one forecast against
another at one substation.

**Topology enters this literature almost entirely as one thing: the summation constraint in
hierarchical forecast reconciliation.** [Nespoli et al.](https://arxiv.org/abs/1910.03976) apply it
to real secondary substations and cabinets in a Swiss distribution grid and gain up to 10% in
root-mean-square error at the upper levels of the hierarchy, and under 1% at the bottom. A summation
constraint says only that the substations beneath a bulk supply point must add up to it. It carries
no information about which substation neighbours which, and it stops holding the moment the network
is switched into an abnormal running arrangement (problem 4 below). That is why a summation
constraint is not enough for Flexpectation. The nearest thing to an exception we found is [Jung et
al. (2024)](https://doi.org/10.1049/icp.2024.1900), who feed which busbar connects to which into a
graph neural network — but they forecast voltage rather than load, and test it only in simulation;
we read their abstract rather than the full paper. Otherwise, information is shared across
substations statistically rather than topologically — one of the four models Artificial Forecasting
tested at secondary substations, a hierarchical Bayesian linear regression, trains its upper layer
across a cluster of similar substations, though the model they recommended is trained per substation
— and Gilbert et al. forecast four levels of a hierarchy separately before naming exploitation of
that hierarchy as future work. SSEN TRANSITION is the exception that shows the value: it used the
connectivity map throughout.

**The nearest answer to this question was measured on NGED's own published data, and it points away
from geography.** [Campagne et al. (2025)](https://arxiv.org/abs/2507.03690) compare four graph
neural network architectures against feed-forward and foundation-model baselines on French regional
load and on the GB distribution networks' open smart-meter feed — around two million meters and
50,000 substations across NGED's and SSEN's areas. Graph-aware models beat the baselines on both.
But which graph wins changes with granularity: spatially informed graphs worked on the coarse French
regions, whereas "for the UK data, data-driven graphs proved more suitable since that dataset
exhibits finer spatial granularity and noisier correlations". They are explicit that reproducing the
network is not the goal — "the objective in forecasting is not to reproduce the transmission network
itself, but rather to construct a representation that best reflects the correlations driving demand
patterns". Their graphs are built from geographic distance or from correlation between series, never
from electrical connectivity, so the specific question stays open.

NGED holds a map of which substations and metered generators connect to which, which no study we
found has used as a forecast input. **Does knowing the shape of the network make the forecast
better, or only more consistent?** The map makes it possible to forecast a bulk supply point both
directly and by summing everything beneath it, and to treat the disagreement between the two answers
as a check on both. We will report whether it improves accuracy as well.

### 2. Forecasting metered generators

**The problem.** Twelve of the 32 series in the trial area are individually metered generators — six
solar farms, three wind farms, a biofuel plant, a battery and a gas generator — and each needs the
same probabilistic, half-hourly, 14-day forecast as a substation. Solar and wind are driven by
weather the ensemble supplies directly. The battery, the gas generator and the biofuel plant are
dispatched on market prices and operator decisions, and no weather forecast contains either.

**Forecasting wind and solar output from a weather forecast is the most mature problem on this list,
and the one where the literature can compare itself.** [Browell et al.
2025](https://doi.org/10.1016/j.ijforecast.2025.10.005) report the Hybrid Energy Forecasting and
Trading Competition (HEFTCom), in which every team forecast the combined day-ahead output of one GB
portfolio — the 1.2 GW Hornsea 1 offshore wind farm plus the aggregate solar capacity of East
England, about 3.6 GW together — from real weather forecasts as they arrived. The winning team
scored a mean pinball loss of 22.18 MWh against the organisers' quick-start benchmark of 53.58; the
organisers also entered a more competitive reference, which scored 25.38, and the next two teams
scored 23.18 and 24.64. HEFTCom is the one place in this review where many teams forecast the same
data with the same metric, which is exactly what the rest of this literature cannot do. Its wind
half is a single offshore farm far larger than any generator NGED meters, and its solar half is a
regional aggregate rather than a plant.

**At the scale of an individual generator, the closest work is on wind.** [Dantas and Browell
(2026)](https://doi.org/10.1002/we.70079) forecast 73 wind farms in GB — 34 onshore, 39 offshore —
from the ECMWF ensemble, seamlessly from 6 to 162 hours ahead. That is the same driver, the same
horizon band and the same probabilistic form Flexpectation needs for its three wind sites, and it is
also where the effective-capacity method described under problem 3 comes from.

**Where the gap is: nothing we found forecasts a distribution-connected battery, gas generator or
biofuel plant inside a net-demand forecast.** For the battery there is at least a method to borrow.
[Bian et al. (2024)](https://doi.org/10.1109/TSG.2023.3303469) recover a price-taking storage
operator's own optimisation parameters by gradient descent on historical prices and observed
dispatch, and prove the recovered parameters converge to the true ones for a class of storage models
— their motivation, that "future power system operators must understand and predict strategic
storage arbitrage behaviors", is NGED's. We found nothing comparable for a gas generator or a
biofuel plant. Otherwise the closest the literature comes is a warning rather than a method:
Pinheiro et al. (2023) found that sites serving a single customer, whose load follows decisions no
weather model can see, were forecast markedly worse than the rest (finding 3 below). We expect the
battery, the gas generator and the biofuel plant to be the hardest series in the trial area for the
same reason, and we will report them separately rather than pooled with the wind and solar sites.

### 3. Estimating the effective capacity of metered generators

**The problem.** We call the amount of generation actually available at a metered site its
*effective capacity*: the output it could produce right now if the weather allowed, as opposed to
its nameplate rating. Turbines go out for repair, inverters degrade, and sites are curtailed — told
by the network operator to generate less than they could. A 20 MW wind farm that has been limited to
14 MW for a month is, for forecasting purposes, a different wind farm, and a model trained on its
nameplate rating cannot see the difference. The same goes for a primary substation with a large
metered generator connected behind it. This problem concerns the 12 metered generators in the trial
area, each of which has a half-hourly meter of its own; the unmetered rooftop solar and small wind
of problem 7 are a separate task.

**For wind, one paper hits our problem exactly, and publishes its method.** Dantas and Browell
(2026) needed available capacity for the same reason we do: the metered-output database they use
"does not include information related to the farms' available capacity over time", so rather than
use a nameplate rating they estimate a time series of available capacity for each farm and normalise
that farm's power by it before modelling. Their method needs no capacity register and no outage
messages. A two-hour stretch of near-constant output while the wind is above the speed at which a
turbine reaches full power marks the farm as running at everything it has, and capacity is then held
at that level until the meter exceeds it. Because their database names no turbine model, they infer
that wind speed from the site's own distribution of wind speeds, and they take the wind speed itself
from reanalysis rather than from any instrument at the farm. They did use one data source
Flexpectation will not have, excluding curtailed half-hours with published bid-acceptance volumes,
which exist for transmission-connected wind farms and not for NGED's embedded generators.

**For solar, each published route needs an input NGED does not have.** The open-source
[RdTools](https://doi.org/10.5281/zenodo.1210316) estimates degradation and soiling from a plant's
alternating-current output rather than from its internals — but its own documentation says site
irradiance is still needed to pick out the clear-sky periods it analyses, and warns that
satellite-based analysis gives less stable results than a sensor at the plant. [Mendonça Severiano
et al. (2026)](https://doi.org/10.1016/j.solener.2026.114382) need no irradiance, but work from
inverter data a network operator does not receive: they classify underperformance across 1,089
systems from that data alone — though they catch clipping, when the panels produce more than the
inverter can pass through, only about half the time, which is a warning for the six
half-hourly-metered solar farms in the trial area.

**At substation rather than generator level, Artificial Forecasting gets closest.** Its Alpha work
builds the baseline it forecasts against by scaling Northern Powergrid's own installed-capacity
projection down by the fraction of that capacity actually generated in 2021–22. Separately, it found
that the National Energy System Operator's national generator-availability signal "almost
universally substantially improved results" at wind-connected primary substations — the nearest
thing in this review to reading effective capacity off an external feed.

**Where the gap is: the research question is whether what Dantas and Browell do for wind can be done
for solar, and whether either can be done with no register of outages or curtailment.** Those are
the two inputs NGED lacks for an embedded generator, and both are load-bearing in the published
method. Where the data does exist, the state of the art simply reads effective capacity off a
register: the team that won HEFTCom clipped its forecast quantiles to the maximum capacity implied
by published outage notices, and NGED's embedded generators publish no such notices. The wind
literature otherwise assumes turbine telemetry, because its authors are the owners who have it;
Dantas and Browell are the exception, which is why theirs is the method to start from. Part of the
problem is a data question rather than a modelling one, because much distribution-connected
curtailment in GB is instructed by the network operator under active network management, so for
those sites the curtailment component of effective capacity is already known inside NGED.

**We plan to attempt this two ways.** The first is to apply Dantas and Browell's approach to solar,
substituting satellite or modelled irradiance for the wind speed their test relies on. The second is
our own: a differentiable-physics model of each generator, in which the physical parameters —
including the plant's direct-current and alternating-current capacity — are fitted as probability
distributions rather than as single numbers, so that capacity is recovered with its own uncertainty
attached and the forecast inherits that uncertainty instead of treating capacity as known. Public
data exists for testing either before it meets NGED's network: Cubico has released the
[Kelmarsh](https://doi.org/10.5281/zenodo.8252025) and
[Penmanshiel](https://doi.org/10.5281/zenodo.5946808) wind farm datasets, which carry turbine
telemetry with alarm and status events and, where available for the same period, the site's own grid
meter — so an estimator built from the meter alone can be scored against the turbine records.

**NGED's specification asks us to track effective capacity over time and, optionally, to combine it
with the forecast into a "prevailing conditions" view. We intend to go further and use it to
normalise each metered generator's series before training — but whether that normalisation earns its
place is itself testable, and one published result suggests it may not.** [de Vilmarest et al.
(2024)](https://doi.org/10.1109/TPWRS.2023.3310280) removed the embedded wind and solar capacities
from their model of GB regional net load, and a Kalman filter tracking the coefficients absorbed the
loss completely — error rose by more than 10% for the same model fitted offline, and fell by 0.4%
for the adaptive one. We will run that comparison rather than assume the normalisation is needed.

**The clearest published demonstration of why effective capacity matters is incidental.** Hornsea
1's export cable faulted on 19 January 2024, a month before HEFTCom's competition period was due to
begin; the organisers delayed the start and restarted the competition on 20 February to give teams
time to adapt. Many still struggled in the early weeks. Teams forecasting wind and solar separately
could post-process their wind forecast for the new export limit, while those forecasting the
combined total "found it harder to adapt", and the organisers' benchmark, which took no account of
the fault, "performed extremely poorly as a result".

### 4. Detecting switching events

**The problem.** When a cable fault or planned maintenance moves part of a network from one
substation to another, the load a substation meters steps up and its neighbour's steps down, with no
change in the underlying demand. NGED's substations spend roughly a tenth of their operating time in
an abnormal running arrangement. Switching labels exist for the 32-series trial area but not for the
wider network, so a method that is to scale has to work from power measurements alone.

**One paper detects these events at a real network operator in order to use them rather than delete
them.** [Bouman et al. (2024)](https://arxiv.org/abs/2405.16164), working with the Dutch network
operator Alliander, study 180 primary substations at 15-minute resolution over roughly a year,
detecting the step changes caused when a cable fault or planned maintenance reroutes part of a
subgrid to a different substation. Events run from a few minutes to several months. They estimate
annual maximum and minimum load within a 10% margin in 88% and 91% of cases. It is the most directly
useful paper in this review, and it leaves the forecasting half untouched — which is what
Flexpectation would add.

**Their central trick is to detect on a residual rather than on the load itself.** Alliander
maintains an independent bottom-up estimate of each substation's load, reconstructed from customer
telemetry and modelled profiles. They fit and rescale that estimate to the measured series, then
hunt for step changes in the *difference* between the two. Normal daily and seasonal variation
largely cancels, leaving a much cleaner signal. NGED has no bottom-up estimate of substation load,
and building one is not in Flexpectation's scope, because the project uses no telemetry from below
primary substation level. GB network operators do publish aggregated domestic smart meter data, but
attributing it to a particular substation is hard, and gets harder as the network is reconfigured
over time; for commercial and industrial metering there is no public equivalent at all.
Flexpectation does, though, produce its own forecast, which can serve as the reference series in the
same way.

**A second technique from the same paper solves a known defect in NGED's trial area.** Some
Alliander substations measure only absolute current, so reverse flow appears as a rise rather than a
sign change — the identical defect at ten of NGED's 12 metered generators. Their bottom-up estimate
is built from measurements that record the direction of flow, so Bouman et al. (2024) take the
direction from the estimate rather than from the meter. Any reference series that records direction
independently would work the same way. They also report that when their bottom-up estimate fails,
the cause is usually wrong topology data rather than a bad algorithm — a warning about the network
records that any such estimate depends on.

**Where the gaps are: the published method detects on a residual we cannot build the same way, and
the events NGED cares about are harder than the ones detected.** A switch at NGED usually fans out
to two or three neighbouring substations rather than one, and the common case is a *partial*
transfer — a continuous fraction of the load moving, with no minimum size — rather than a whole
subgrid. There is no voltage measurement at primary substation level to fall back on, and
tap-changing transformers plus half-hourly averaging would blur it if there were. So the detector
has to work unsupervised, on power alone, against events that are partial, multi-recipient and
unlabelled.

### 5. Forecasting a substation as if it were always in its normal running arrangement

**The problem.** NGED plan the network against what each substation would carry under its normal
running arrangement, so that is what the forecast has to predict — including for a substation that
has been sitting in an abnormal arrangement for weeks. That makes the target a quantity that was
never metered, and it makes the training history contaminated: past readings taken while the network
was abnormally configured describe a different substation from the one being forecast.

**Researchers respond in one of two ways, and both alter the series the model is trained to
predict.** Most delete the affected data: [Huyghues-Beaufond et
al.](https://doi.org/10.1016/j.apenergy.2019.114405) detect and remove structural breaks across 342
UK medium-voltage feeders. A smaller strand rewrites it instead, to an "as if never switched" level:
[Paredes and Vargas](https://doi.org/10.1049/iet-gtd.2017.0129) do it across 169 real feeders and
report better medium-term forecasts for it, and Artificial Forecasting does the same in its
data-preparation pipeline, rescaling a step-change block onto the level of the most recent one when
that block's median falls outside the most recent block's 10th-to-90th-percentile range, so the
history is kept rather than dropped. Artificial Forecasting argues for going further on the grounds
that demand changes of an order of magnitude, mostly caused by network reconfigurations, "cannot be
directly handled even by powerful nonlinear models like neural networks" — though they add that
changes that large are rare at their secondary substations. Gilbert et al. name adaptive handling of
structural breaks as future work.

**Adaptive models are the live alternative, and they handle gradual change rather than steps.** de
Vilmarest et al. (2024) let a Kalman filter track the drift on Browell and Fasiolo's own 14-region
GB dataset instead of correcting the history, cutting error by about 4% in 2019, 7% in 2020 and 8%
in 2021 against the same model refitted every day. But a switching event is a step, not a drift, and
a model that simply adapts to a new load level never learns that switching happened — so it cannot
report what the substation would have carried under its normal arrangement, which is the quantity
NGED needs.

**Where the gap is: we found nobody who feeds switching-contaminated history to a model as it
stands.** The question we want to settle is whether that can be done. Instead of correcting the
series, a model could be fed the difference between what a substation actually metered and what a
model that ignores network topology expected it to meter. That difference is the same quantity
Bouman et al. (2024) use to detect switch events, but used as a forecast input rather than as a
detector, so a reading taken while the network was abnormally configured would still carry
information without anyone having to estimate a level correction first. A negative result here would
still be valuable: evidence that switching cannot be recovered from power data alone would
strengthen the case for taking switching labels from operational systems instead — a route
Artificial Forecasting has already identified, naming the incorporation of planned-outage records in
its post-Beta roadmap.

### 6. Detecting faulty metering

**The problem.** NGED's telemetry carries stuck values that repeat unchanged for a day or more,
zeros that mean "no reading" rather than "no load", physically impossible values, and gaps running
from a single half-hour to several months. Ten of the 12 metered generators in the trial area report
magnitude without direction, so export appears as a rise. A model trained on uncleaned data learns
the fault, and a forecast that fails silently because its recent history was stuck is worse than one
that says it is degraded.

**Our search surfaced no paper whose subject was automatic detection of faulty metering at
distribution substations.** It appears in this literature as a data-cleaning step described in
passing rather than as a problem in its own right. The nearest things are adjacent. Bouman et al.
(2024) detect anomalies alongside switch events, on the same residual, and their sign-recovery
technique addresses exactly the non-directional metering defect described above. Mendonça Severiano
et al. (2026) classify solar underperformance, but from inverter data a distribution network
operator does not receive. And Artificial Forecasting supplies the strongest indirect evidence that
this matters: across their 729 secondary substations, data quality mattered at least as much as the
choice of model (finding 1 below).

**Where the gap is: nobody publishes how well a detector works, because nobody publishes labels.**
Detecting a stuck meter is not technically hard; knowing how often a detector is right requires a
set of half-hours labelled as faulty or not, and no such set is public for substation telemetry.
That is a gap this project can close cheaply, because the trial area is small enough to label by
hand.

### 7. Disaggregating unmetered solar and wind from a substation's net flow

**The problem.** Rooftop panels and small turbines appear only as a dent in a substation's net flow.
Recovering both the half-hourly output of that unmetered generation and its installed capacity, from
the net flow alone, is what we call *disaggregation*. It is a different task from estimating how
much of a *metered* generator's capacity is available today, which is problem 3. It is a stretch
goal for the trial area and a requirement for the network-wide scale-up.

**Where demand and generation are separated at all in this literature, the generation is metered.**
Artificial Forecasting models gross demand and customer export independently at primary substations,
which is more than any paper here does, but customer export is metered. SSEN TRANSITION split net
load into demand and generation, forecast the two separately and recombined them, again with metered
generation.

**The nearest peer-reviewed work stops one step short of a forecast.** [Kara et
al.](https://doi.org/10.1016/j.segan.2017.11.001) and [Li et
al.](https://doi.org/10.1109/TPWRS.2020.3035639) recover the solar signal from feeder-head and
substation measurements without forecasting it. The one benchmark we found on estimating installed
capacity is at secondary substations, which is a level below ours; we read only its abstract. Most
of the rest of the behind-the-meter disaggregation literature works on individual smart meters, a
level or two below a primary substation, and is excluded for that reason.

**The direct predecessor of this work is running now in GB.** [UK Power Networks'
NIA_UKPN0104](https://smarter.energynetworks.org/projects/nia_ukpn0104/) (2024–2026, £389,444), with
Open Climate Fix and Sheffield Solar, infers the capacity of unmetered solar sitting behind each
primary substation from half-hourly substation load and weather, then forecasts that generation.
Open Climate Fix is a partner in both projects, so Flexpectation starts from its method rather than
from scratch.

**One production system already splits unmetered wind and solar out of substation measurements, by
transferring from substations that do meter them.** [Teng et al.
(2023)](https://doi.org/10.1016/j.rser.2023.113662) train on ten Dutch substations that carry
complete renewable metering, then predict solar and wind power separately at substations with none,
from weather, geospatial position and each site's known renewable capacity, at 15-minute resolution
— a root-mean-square error of 0.07 against 0.70 for a conventional transfer-learning model. It ships
as the `split_energy` component of [OpenSTEF](https://lfenergy.org/projects/openstef/), Alliander's
open-source forecasting stack, which is in live operation.

**Where the gaps are: doing it without a metered training set, estimating the capacity rather than
being told it, and stating the uncertainty.** Teng et al. (2023) need a population of fully-metered
substations to transfer from, and they are given the existence and capacity of each renewable
facility rather than inferring it — whereas estimating that capacity is half of what NGED needs.
Their output is a near-real-time estimate rather than a forecast, so nothing here puts unmetered
solar or wind inside a forecast that states its own uncertainty over a horizon like ours.

### 8. Disaggregating other distributed energy resources: heat pumps, electric-vehicle chargers and batteries

**The problem.** Heat pumps, electric-vehicle chargers and price-sensitive domestic batteries change
the shape of a substation's load in ways a model trained on history cannot anticipate, because the
number of them behind any given substation is growing quickly. The stretch goal is to disaggregate
and forecast them separately rather than letting them sit inside net demand.

**The three are not equally tractable: heat pumps have a weather driver, chargers have none, and
batteries follow a control decision.** A heat pump is driven by outdoor temperature, which the
weather ensemble supplies directly, so it should behave much like any other weather-driven load. An
electric-vehicle charger has no exogenous driver at all: whether someone plugs in tonight depends on
where they drove today, and because charging behaviour is synchronised across many households,
errors add up rather than cancelling — and synchronised evening peaks are precisely what a network
constraint is about. Price-sensitive batteries are harder still, because the driver is a control
decision: two identical batteries behind the same substation can dispatch in opposite directions on
the same day depending on the tariffs their owners are on.

**This is the largest deliberate omission in the present review.** Our search covered substation and
generation forecasting, and did not cover the electrification literature, which is large and active
in its own right. The volume of work is easy to demonstrate: of the 265 papers accepted for CIRED's
Brussels workshop of June 2026, 17 concern electrification — almost as many as the 19 that name
forecasting or prediction at all. We will read that literature properly before this strand begins,
and until then this review has nothing to report about it beyond its existence.

**Where the gap is: we cannot yet say what a gap would look like.** Naming it honestly needs the
reading we have not done, so the first deliverable on this strand is a review of the electrification
literature rather than a model.

## What recurs across the studies we read

Six findings recur across the studies reviewed above. These are findings about this literature, not
laws of nature: each is what several teams measured on their own networks, and a network that
differs from theirs may well behave differently.

**1. In every load-forecasting study we read, sophisticated models beat simple models by a much
smaller margin than the effort put into them would suggest.** Pinheiro et al. (2023), running a live
system across 96,989 Portuguese secondary substations, tuned a gradient-boosted tree by exhaustive
grid search. At system level, the gradient-boosted tree scored 199 MW root-mean-square error and the
generalised additive model scored 191 MW, so the gradient-boosted tree was 4% worse than the simpler
model. Pinheiro et al. (2023) rejected the gradient-boosted tree on both accuracy and
interpretability, and kept the generalised additive model. Artificial Forecasting also found that
gradient-boosted trees did not beat a simpler model, when forecasting customer export at primary
substations. Compared against the Bayesian ridge regression they went on to adopt, boosted trees
"helped some substations but harmed others", so they kept the Bayesian ridge regression as their
default. Northern Powergrid's deliverable says which way boosted trees moved the error, but not by
how much.

In the BigDEAL load forecasting competition, only four of 13 finalist teams beat the organisers'
benchmark at all. And when Artificial Forecasting tested a neural network against a
four-week-average baseline at 729 secondary substations, the neural network lost on five of six
metrics at the 24 substations with the worst data quality. The margin was small, and data quality
and the choice of metric mattered at least as much as model complexity.

**2. In every study that forecast more than one voltage level, accuracy got worse further down the
network.** Hertel et al. (2026) ran the same models against the same benchmark at three levels and
beat that benchmark by 59.6% at transmission level, 42.3% at low-voltage feeders and 23.3% at
individual customers. [Pfeifer et al. (2021)](https://doi.org/10.1049/icp.2021.2177) measured the
same thing separately for wind power, solar power and load across a German medium-voltage region,
and report that forecasts get worse both at lower levels of aggregation and at longer horizons; we
read their abstract rather than the full paper. The model did not get worse; the problem got harder.
That pattern is probably not a fact about forecasting so much as a fact about averaging: a grid
supply point aggregates hundreds of thousands of customers, whose individual quirks cancel out,
while a single feeder aggregates a few dozen, whose quirks do not. Predicting the temperature of a
kilogram of air is easier than predicting the motion of each molecule in it, and for the same
reason.

**Rising error does not mean falling usefulness.** A forecast at a primary substation may carry a
larger percentage error than a forecast at a grid supply point and still support flexibility
procurement just as well, because what NGED needs from the forecast is a reliable answer to "will
this substation exceed its firm capacity?", and that question can be answered well even when the
load itself is hard to predict precisely. Whether decision-usefulness really is flat across voltage
levels is something this project can measure, and we intend to.

**3. In the one study that reported results substation by substation at scale, a substantial
minority of substations were not forecast better by a trained model than by a naive "same time
yesterday" rule.** Pinheiro et al. (2023) found that their model beat a "same time yesterday"
forecast at 83–87% of network-owned secondary substations but at only 66–70% of customer-owned ones.
Those customer-owned sites serve a single customer — one large building or one industrial process —
where load follows decisions no weather model can see. We do not know that NGED's primary
substations will behave the same way, and they may not, because a primary substation aggregates far
more customers than a Portuguese secondary substation does.

**4. In the studies we read, standard accuracy measures rewarded flat forecasts that would be of
little use for flexibility procurement.** A forecast that predicts the right peak at the wrong time
is penalised twice by mean absolute error — once for the peak it predicted that did not happen, and
once for the peak that happened and it missed. A flat, featureless forecast avoids both penalties.
Meteorologists named that effect the double penalty decades ago, and their conclusion transfers: a
score that forgives a peak predicted an hour late is generally no longer a **proper scoring rule** —
a score a forecaster cannot improve by publishing anything other than what they genuinely believe. A
peak-aware score therefore belongs alongside a proper score, not instead of one. Mean absolute error
and root-mean-square error therefore tend to favour smooth forecasts over peaky ones, which is
backwards for a network buying flexibility to keep load below a capacity limit. Two teams
independently concluded that mean absolute error was the wrong measure for peaks, and built their
own measure instead. Pinheiro et al. (2023) scored their substation models with Haben's adjusted
error, a peak-aware measure, for exactly this reason. Artificial Forecasting built a metric over the
top 10% of demand values and made it the primary measure for comparing their models, reporting it
both against actual demand and normalised to transformer rating.

**5. A forecast can state its own uncertainty badly, and a single accuracy score will not reveal
that the uncertainty is wrong.** Kaas et al. (2026) scored models on 200 German low-voltage feeders
with an overload-decision metric evaluated at each model's 95th percentile. The two models that
topped that metric on the consumer side turned out to have 90% ranges containing the true value only
62% and 58% of the time across the series as a whole, and under half the time at the peaks
themselves. In Kaas et al.'s results, a model that understates its uncertainty raises fewer false
alarms, so it scores well on a threshold-crossing test while being exactly the model an operator
should not trust near a capacity limit. Kaas et al. (2026) supply their own counter-example: ranked
on average error rather than on the overload metric, the winning model was also the most honest
about its own uncertainty, with reality falling inside its stated 90% range 89.75% of the time.

**6. Weather forecasts are barely used at low voltage, and weather ensembles almost never.** Of the
221 low-voltage forecasting papers Haben et al. (2021) reviewed up to 2020, three used a weather
*forecast* and none used an *ensemble* of weather forecasts. Pinheiro et al. (2023), published after
that review closed, is a fourth paper using a real weather forecast — but its inputs are single
point forecasts rather than an ensemble. Pinheiro et al. (2023) therefore overturns the first half
of Haben's finding but not the second: even the largest deployment in this review used no weather
ensemble. Artificial Forecasting's published secondary-substation results use no weather at all,
because the weather archive available to them reached only 16 days ahead while their forecasts were
month-ahead, and a substitute built from the previous year's observations at the same time of year
made every model slightly worse.

### Three findings that cut against this project's plan

Three results in this literature point against Flexpectation's plan, and we intend to test all three
rather than route around them.

**Finer-grained weather data has not always paid.** Browell and Fasiolo (2021) added spatial
statistics derived from gridded numerical weather prediction to their model of 14 grid supply point
groups in GB. Those spatial statistics helped significantly in two of the 14 regions, hurt
significantly in three, and made no measurable difference in the remaining nine. They put that down
to their own model rather than to the data, writing that another method might yet extract value from
it by building different features. Weather itself was worth a great deal to them — adding wind and
irradiance cut their pinball loss by 40% overall, and by 60% in North Scotland against 10% in London
— so the question is not whether weather matters but whether *finer* weather does. Artificial
Forecasting obtained postcode-level weather forecasts for two wind-connected primary substations
after their wind-connected models had performed poorly, and reported that the postcode-level
forecasts "did not notably improve model performance", naming better weather data as a next step.
What both results say is that finer weather data does not help everywhere, so the interesting
question is *where* it helps. That question is answerable, and answering it is part of this project:
we expect finer weather data to matter most where a substation's load is dominated by weather-driven
generation or heating, which is where NGED most needs the forecast to be right.

**Weather has bought less than expected at low voltage in the past.** [Haben et al.
(2019)](https://doi.org/10.1016/j.ijforecast.2018.10.007) tested 100 real low-voltage feeders with
both forecast and observed temperature, and found that temperature had no effect on forecast
accuracy, or a negative one. Haben et al. (2019) used data collected in the early 2010s, and we
expect how much weather matters at a substation to be changing quickly, because the thing that makes
a substation weather-dependent is embedded solar generation and heat pumps, and there are far more
of both on the network now than there were then. A primary substation that was almost
weather-independent ten years ago may be strongly weather-dependent today. That is a prediction,
though, not a measurement — and the Scottish primary-substation sensitivities of [Fox et al.
(2018)](https://doi.org/10.34890/134), measured on data ending in 2016/17 and described under "What
GB networks have already built" below, say weather was already moving primary substation demand well
before then. Measuring how much weather now explains at NGED's primary substations is one of the
more useful things this project can report.

**A model trained on none of NGED's data may match a model trained on all of it.** Kaas et al.
(2026) tested Chronos-2, a general-purpose time-series model that had never seen their data, against
models trained specifically on those 200 feeders. Chronos-2 beat every purpose-trained competitor on
mean absolute error, 3.8 kW against 4.2 kW. If heavily engineered models do not clearly beat an
off-the-shelf model given none of the target network's data, that is important information about the
value of any such experimental programme.

### An open question this review cannot settle

There are two quite different explanations for the small margin between sophisticated and simple
models reported across this literature, and nothing we read separates them.

**Explanation one: substation demand has a hard limit on how well it can be predicted, and today's
models are already close to it.** Half-hourly load at a single primary substation is the sum of
decisions made by a few thousand customers — when they cook, when they charge a car, when a factory
starts a shift — and much of that is genuinely unpredictable from weather and calendar data, because
nothing in the weather or the calendar determines it. If this explanation is right, a simple model
already captures nearly all of the predictable part, a sophisticated model has almost nothing left
to find, and the small gains reported across this literature are the correct answer to the question.

**Explanation two: nobody has yet pushed hard enough to find out how well substation demand can be
predicted.** The sophisticated models in this literature are generalised additive models,
gradient-boosted trees and similar established methods, usually applied to a standard set of
calendar and weather features. Those are sensible, well-chosen tools. They are also not what a
sustained modern machine-learning effort looks like. AlphaFold reached its result through several
years of a large team running a great many experiments against one fixed, public benchmark. That
route is open to energy forecasting in principle, but it is rare in practice, for structural reasons
rather than any failing of the researchers: a forecasting paper is typically written by a small team
over months rather than years, tests a handful of model configurations, and reports results on a
dataset that no other paper uses. Energy forecasting has therefore never accumulated the thousands
of directly comparable attempts that protein-structure prediction had accumulated before AlphaFold.

We hold explanation two loosely, and explanation one may well be the right one. Flexpectation is not
resourced like a large industrial research laboratory. What it is resourced to do is run many
experiments cheaply against one fixed benchmark, which is the part that matters for telling the two
explanations apart. If explanation one is right, sustained experimentation will converge quickly on
a small improvement over a naive forecast and then stop improving, however many further experiments
we run — and we will report that plainly. If explanation two is right, improvements should keep
arriving well past the point at which a smaller effort would have concluded there were none left to
find. Either answer is worth publishing, and the second would be worth more to the industry than to
this project alone. That is the shape of what Flexpectation offers beyond its own forecasts: not a
claim to have found the state of the art, but a run of comparable experiments on one fixed
benchmark, published as they go, so that the next team does not have to start where we started.

## What GB networks have already built

Six concurrent or recent GB network-innovation projects bear on this work, and between them they
have built more of what Flexpectation needs than the academic literature has. Five are summarised
here; the sixth, Northern Powergrid's Artificial Forecasting, gets its own section below because
this review leans on it more heavily than on any other.

**Scottish and Southern Electricity Networks' TRANSITION** (Network Innovation Competition,
Oxfordshire, reported 2021) is the closest precedent for Flexpectation's method. It forecast net
load at 13 primary substations, their bulk supply points and their 11 kV feeders, from 30 minutes to
10 days ahead. SSEN TRANSITION drew its uncertainty from the 40 members of the German weather
service's ICON-EU ensemble. It split each substation's net load — demand minus whatever generation
behind that substation happened to produce — into demand and generation, forecast the two
separately, then recombined them. And it used the network connectivity map, the record of which
substation feeds which, throughout: the project ranks "historical network connectivity data
availability" as being just as important as the demand measurements themselves. Two limits of SSEN
TRANSITION are what Flexpectation sets out to go beyond: the ensemble runs only to four days, with a
single deterministic forecast covering days four to ten, and the trial covered 13 substations rather
than a network.

**[NGED's own Electricity Flexibility and Forecasting System,
EFFS](https://smarter.energynetworks.org/projects/wpden03/)** (Network Innovation Competition,
2018–2021, £3,338,896) forecast grid supply points, bulk supply points, primary substation
transformers and generation sites from an hour to six months ahead, feeding automated constraint
identification. Its evaluation independently selected XGBoost as the best balance of accuracy
against effort — the same starting point Flexpectation uses. Its forecasts were deterministic, with
no uncertainty attached, which is the step this project adds.

**UK Power Networks' NIA_UKPN0104** is described under problem 7 above, as the direct predecessor of
Flexpectation's unmetered-solar work.

**[SSEN FastTrack](https://smarter.energynetworks.org/projects/10166254/)** (Strategic Innovation
Fund, Alpha 2025–2026) is building a probabilistic load forecast substation by substation, rolled up
to a grid supply point view. **[SP Energy Networks'
Predict4Resilience](https://smarter.energynetworks.org/projects/10061710/)** drives a probability
distribution of network faults per district from an ensemble weather forecast, up to seven days
ahead, in an operational control room — the GB precedent for putting ensemble-derived distributions
in front of network operators. SP Energy Networks has also published at Flexpectation's own voltage
level: [Fox et al. (2018)](https://doi.org/10.34890/134) ran a numerical weather prediction model
over Scotland at 1 km resolution for ten years, mapped it onto each primary substation weighted by
customer density, and used it to separate the effect of weather on peak demand from the effect of
everything else — 13 substations in the proof of concept, almost 400 in production. Demand rose by
between 1.4% and 4.8% per degree Celsius, differing substation by substation with the mix of
customers behind it. It corrects history for planning rather than forecasting forward, but it is the
GB precedent for putting gridded weather onto individual primary substations.

Two deployments outside GB belong alongside these. **The Dutch operator Alliander runs
[OpenSTEF](https://lfenergy.org/projects/openstef/)**, an open-source forecasting stack under the
Linux Foundation's LF Energy umbrella, in live operation across thousands of grid connection points
to 48 hours ahead. It is the only production system we found that separates unmetered wind and solar
out of a substation's measurements, by the method of Teng et al. (2023) described under problem 7,
and being open source it is the one whose method can be read rather than inferred from a
deliverable.

The second is far larger than any project here. Enedis, the French distribution network operator,
has forecast consumption and generation at all 2,300 of its high-voltage-to-medium-voltage
substations since 2015, and is now extending that to a finer geographic grid ([Cordier et al.
2024](https://doi.org/10.1049/icp.2024.2058), whose abstract we read rather than the full paper). A
high-voltage-to-medium-voltage substation in France is broadly the level of a GB primary substation.
Forecasting operationally at the scale Flexpectation reaches in 2027 is therefore a decade old
somewhere else, which is reassuring about the engineering and says nothing about the forecast
quality, because the paper reports none.

### Northern Powergrid's Artificial Forecasting is further ahead, and sets the bar

One concurrent project matters more than any paper here. Artificial Forecasting is an Ofgem
Strategic Innovation Fund programme, with about £3.9m of grant across its three phases, run by
Northern Powergrid with Faculty, EV.energy and Oaktree Power, the final Beta phase running to
February 2027. Its deliverables are publicly available on the Energy Networks Association's Smarter
Networks Portal, though the Beta deliverables sit under a separate project registration from the
Alpha ones linked above. Its argument is also in the peer-reviewed literature: [Wade et al.
(2024)](https://doi.org/10.1049/icp.2024.2102), by authors at Northern Powergrid and Faculty, put it
to CIRED that annual, assumptions-driven models of load at primary and secondary substations will
not support flexibility procurement, and that monthly, weekly and daily operational forecasts are
needed instead; we read its abstract rather than the full paper. Artificial Forecasting does much of
what Flexpectation does at primary substations. It also covers secondary substations, which
Flexpectation does not. And at the time of writing it is further ahead than Flexpectation.

**Artificial Forecasting has run operationally through a full winter flexibility procurement
cycle.** A forecasting service for primary substations is deployed and has passed the network's
architecture review board, data governance and information security checks for its current
deployment. It was used operationally by Northern Powergrid's System Forecasting team through a full
winter flexibility procurement cycle to support week-ahead dispatch decisions. It produces
half-hourly probabilistic forecasts with 5th-to-95th-percentile bands, flags forecast exceedances of
firm capacity, and is benchmarked against the network's existing growth-based and persistence
methods. Their published results show forecast error rising only slightly out to 11 days; the
deliverable does not give the figures. Their value case puts whole-life net present value at around
£60m for one network, or £250m if three further networks adopt it, driven mainly by a 3% reduction
in spending on reinforcement — building bigger transformers and cables — in the current
price-control period rising to 6% in the next, and a 25% improvement in the cost-effectiveness of
contracted flexibility. Those are the figures from the Beta application; the project reports that a
year of live operation supports them, with measured savings still to accumulate.

Artificial Forecasting is independent evidence that short-term substation forecasting is
operationally useful, that networks will change their procurement process around it, and that a
benefits case has been made and accepted. Because it is public, operational and benchmarked against
a real incumbent method, it also sets the clearest available bar for what "working" looks like.
Northern Powergrid's programme rightly prioritised getting a service into live operational use;
Flexpectation takes on research questions that priority left for later. Artificial Forecasting's
core intellectual property is to be made available royalty-free to other GB networks, and we would
rather build on it than rebuild it.

## Set against this literature, what we plan is ambitious, and here is why we think it can be done

**Measured against the studies above, the plan goes beyond this literature in four directions at
once.** No study in this review drives a substation forecast from a weather ensemble across a 14-day
horizon. None models the upper tail explicitly at substation level; the one study that models it
explicitly at all works on regions far larger than a substation. None estimates how much of a
*solar* generator's capacity is available without the plant's own instrumentation, and none
estimates it for any generator whose outages and curtailment go unpublished. None keeps
switching-contaminated history usable rather than deleting or rewriting it. Flexpectation attempts
all eight problems above, across four families of model:

- a heavily-tuned version of the gradient-boosting approach that wins most tabular forecasting
  competitions, and which NGED's own EFFS project independently selected;
- weather and time encoders pre-trained on large datasets, so that a model for one substation can
  start from what has been learned across all of them;
- models that use the connectivity map explicitly;
- differentiable physics — building known physical behaviour directly into the model, so that it has
  to learn only what the physics cannot supply: the response of a solar panel and of a wind turbine
  on the generation side, and the thermal response of buildings on the demand side.

Only the heavily-tuned gradient-boosting model is in scope for the first version of the service. The
pre-trained encoders, the connectivity-map models and the differentiable physics all belong to the
network-wide scale-up from 2027, as does the disaggregation of unmetered generation and forecasting
the network as a network. "What this review excluded, and why" explains why the
differentiable-physics strand is the least well supported of the four.

**The main reason for attempting all eight at once is that they may be one problem rather than
eight.** A switching event, a turbine out for repair and a stuck meter all surface in the same
place: as a discrepancy between what a substation metered and what the weather and the calendar say
it should have metered. Every study reviewed above that touches more than one of the eight solves
them as a pipeline. Dantas and Browell estimate available capacity, then normalise by it, then
forecast. Artificial Forecasting rescales step-change blocks in data preparation, then forecasts.
SSEN TRANSITION splits net load into demand and generation, forecasts each, then recombines.
Huyghues-Beaufond et al. detect structural breaks and delete them before training begins. In every
case one stage's output is frozen before the next stage sees it, so an error made early cannot be
corrected later and the forecast error never gets to tell the capacity estimator it was wrong.

**So the question we want to answer is whether one model that estimates capacity, switching state
and demand together beats that pipeline.** NGED's specification leaves room for it, asking that
these phenomena be handled rather than that they be handled explicitly. The one published result
that bears on the question points the joint way: de Vilmarest et al. (2024), described under problem
3, removed the embedded wind and solar capacities from their model of GB regional net load, and the
adaptive version got 0.4% *better*, absorbing into its own coefficients what the explicit capacity
figure had been supplying, while the static version got more than 10% worse. That is one result, on
regions far larger than a substation, for one phenomenon out of several — and there are good reasons
to doubt it generalises. A gradient-boosted tree is structurally poor at the subtraction a two-stage
residual hands it precomputed, and each of our series carries only tens of thousands of training
rows, which is not the regime in which a model reliably discovers an implicit baseline for itself.
We expect the answer to differ by model family, which is part of why the differentiable-physics
strand matters: it is the one family in which capacity, weather response and demand are estimated
jointly by construction.

**The first reason for confidence is that experiments are nearly free.** The core forecast already
exists and runs today, on an experiment framework that makes one more experiment cost compute time
rather than staff time. That is what makes it realistic to run on the order of hundreds of
machine-learning experiments a month, and it is the same argument the introduction to this review
makes: if roughly nine ideas in ten fail, the only affordable way to find the one that works is to
make each attempt cheap.

**The second is that none of the eight problems starts from nothing.** Detection of switching events
has been demonstrated at a real network operator by Bouman et al. (2024). Estimating a wind farm's
available capacity from its meter has been published, with code, by Dantas and Browell (2026).
Inferring unmetered solar behind a primary substation is being built now by UK Power Networks with
Open Climate Fix, who are a partner in both projects. For most of the eight, the work is to extend a
published method to NGED's data rather than to invent one, and Artificial Forecasting's core
intellectual property is to be made available royalty-free to other GB networks.

**The third is that the strands are independent, so one failing does not strand the others.** The
live service for the 32-series trial area is problem 1, and it does not depend on any of problems 2
to 8 landing; each of the eight is separately useful to NGED whether or not its neighbours are
solved.

**Several of these directions will not work, and that is what makes them research directions rather
than engineering tasks.** The honest expectation is that some deliver clearly, some produce a
negative result worth publishing, and some are abandoned. Both NGED and this project count a
negative result as an outcome: evidence that switching cannot be recovered from power data alone,
for instance, would be worth having, because it would justify extracting switching labels from
operational systems instead of continuing to look.

## What this review excluded, and why

**Behind-the-meter solar disaggregation was excluded because most of it works below NGED's level of
aggregation.** Separating a substation's metered flow into demand and the solar generation hidden
inside it is a large and active field, mostly working on United States smart-meter data at
individual customer level. We excluded it as a body, and kept a handful of citations to return to
when the disaggregation work begins. The exclusion covers our reading list rather than the whole
field: work at feeder aggregation and above is real, and problem 7 above names it.

**General concept-drift detection was excluded because it addresses gradual drift, and NGED's
problem is a sudden step change.** Most of that literature is about a model's accuracy decaying
slowly as the world changes under it, and about adapting the model in response, whereas our problem
is a discrete step change with a known physical cause. The abrupt-drift and change-point strand of
that literature is closer to our problem, and we intend to read it properly before the switching
work begins. A model that simply adapts to a new load level, without ever detecting that switching
happened, is the live alternative to detecting switching at all. We will treat it as the approach
our switching work has to beat.

**Differentiable physics applied to substation demand forecasting** produced no strong result,
though the ingredients exist separately. There is substantial work on physics-informed neural
networks for power systems, including models that map weather to solar output, and one 2026 paper
applying a differentiable temperature-demand relationship at system level. On the demand side the
physics that matters is the thermal response of a few thousand buildings rather than of a panel or a
turbine, and models that build that response in are a field of their own: [Di Natale et al.
(2022)](https://doi.org/10.1016/j.apenergy.2022.119806) constrain a neural network so that heat
always flows the physically correct way, and [Jiang et al.
(2025)](https://doi.org/10.1016/j.adapen.2025.100223) review that field, which they describe as
nascent. What we did not find is anyone aggregating building thermal physics up to a substation and
putting it inside a probabilistic forecast, which is the version this project would need.

**Heat pumps, electric-vehicle chargers and domestic batteries were not searched at all**, which
problem 8 above names as this review's largest deliberate omission. Our searches were framed around
substation and generation forecasting, and the electrification literature is large enough to need a
review of its own.

**Network topology detection was excluded because it needs measurements NGED does not have.**
Inferring the network's wiring from high-resolution synchrophasor measurements is well developed,
but those measurements are not available to this project. That exclusion covers neither problem 4,
which detects switching from half-hourly power alone, nor the topology question under problem 1,
which is about using a connectivity map NGED already holds rather than inferring one.

**The bulk of the low-voltage forecasting literature** is covered through the Haben et al. (2021)
review of 221 papers rather than read individually, and we have not systematically covered
low-voltage work published since it closed in 2020. The same lead author's open-access book-length
treatment of 2023 is the better entry point for anyone following this up.

**CIRED**, the International Conference on Electricity Distribution, is the venue this audience is
most likely to read — it is where European distribution network operators publish their own
operational work, so it is where a claim of ours is most likely to be contradicted. We therefore
searched it in full: the titles and abstracts of every paper in the CIRED main conferences and
workshops from 2017 to 2025, about 3,600 of them, plus the openly archived 2018 and 2019 proceedings
by keyword, and the 265 papers accepted for the Brussels workshop of June 2026 by title, those
proceedings not yet being published. Nothing there contradicts what this review reports missing, and
the absences are worth stating, because CIRED is where the counter-example would have been. No CIRED
paper drives a load or generation forecast from a weather ensemble. None forecasts load or
generation beyond 48 hours: the only 14-day forecast in the proceedings predicts feeder faults
rather than load. Fourteen forecast probabilistically at all, of which one is at substation scale —
[Mesarcik et al. (2025)](https://doi.org/10.1049/icp.2025.1968), day-ahead, on ten years of
measurements from 312 Dutch substations. Nothing scores the upper tail, nothing keeps
switching-contaminated history usable, and nothing estimates how much of a generator's capacity is
available. The closest paper to our own problem, Ruhhütl et al. (2023), appears in the table under
problem 1 above; its result is a further instance of findings 1 and 2. We read only the abstracts of
it and of Mesarcik et al. (2025), because both full texts are paywalled. The Brussels titles of June
2026 change none of this: 19 of the 265 name forecasting or prediction, none names an ensemble, and
the only horizon named is day-ahead. Two of the 19 apply time-series foundation models, so the
possibility that a model given none of a network's own data can compete is being tested in this
venue too.

## Publishing results that others can compare against

**Energy forecasting's own senior figures say that published results in the field cannot be compared
with each other.** [Hong et al. (2020)](https://doi.org/10.1109/OAJPE.2020.3029979), a review
written by six of the field's most senior figures, concludes that "most papers can never be
replicated, because the data have never been published". Hong et al. (2020) add that authors
sometimes pick the error measure that favours their own method, that significance tests are seldom
run when the differences between models are small, and that many papers compare a new model only
against models "within the immediate family". [Tawn and Browell
(2022)](https://doi.org/10.1016/j.rser.2021.111758) found eleven wind and solar papers that compared
a new model only against other models of the same type.

**Incomparable results are what this review ran into at every one of the eight problems.** Even the
eight studies in the one table above, all forecasting electricity demand somewhere on a network,
differ in target, level, horizon and weather assumption in nearly every row, so almost none of them
can be compared directly with any other. Two papers a fortnight apart on the same 200 feeders name
different winners. The other seven problems get no table at all, because there was too little to put
in one.

**Hong et al. (2020) name two remedies: publishing the underlying data, and running competitions in
which every team forecasts the same dataset.** This project is well placed to help with both, and
others have started already: HEFTCom and Energy-Arena both compare methods on common data with a
common metric, and Energy-Arena keeps a live public leaderboard. A third approach — recovering a
ranking from the published literature after the fact — shows what the alternative costs: [Nguyen and
Müsgens (2026)](https://doi.org/10.1063/5.0300682) did recover a defensible ranking of solar
forecasting methods from the published literature, by screening 1,447 studies and hand-extracting
4,687 skill scores from those that reported one, then statistically removing the effect of ten other
factors. Their finding is that ensemble-hybrid models improve on time-series models by 7 to 27
percentage points of skill score, while many advanced machine-learning methods gave inconsistent
gains. A comparison can therefore be dug out of this literature, but only at that price, and nobody
does it routinely. Publishing comparable results in the first place is much cheaper.

**Some of the data needed to do that is already public.** Neither HEFTCom nor Energy-Arena covers
distribution-substation load, which is the level NGED acts at, so we intend to follow their
protocols where they apply rather than invent our own. Some substation data is already public —
NGED's [Connected Data Portal](https://connecteddata.nationalgrid.co.uk/), and Northern Powergrid's
[open data
portal](https://northernpowergrid.opendatasoft.com/explore/dataset/primary-operational-metering/),
which publishes half-hourly metering from its primary substations. The Dutch operator Liander goes
furthest, publishing [LianderPower](https://www.liander.nl/over-ons/open-data), twelve years of
five-minute measurements from its distribution network with matched historical weather, under a
Creative Commons licence. Publishing the telemetry behind our own experiments would make the results
reproducible by anyone, which is still rare in the substation literature, where only 52 of the 221
low-voltage papers reviewed used any open dataset at all. Alongside it we will publish the
evaluation protocol, the metric definitions and the code that computes them. Artificial Forecasting
is moving the same way, with substation-level historical forecasts and model-performance metrics
designed into its Open Data Portal release, and a shared evaluation protocol between two GB networks
would be worth more than either alone.

This review makes nine commitments to publish or report. Collected in one place, and in the order a
reader would want them, they are:

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
  solar sites, because they are dispatched on market signals no weather forecast contains.
- **A peak-aware score is reported alongside a proper scoring rule**, never instead of one.
- **The tail is scored with a threshold-weighted continuous ranked probability score**, weighted
  above each substation's firm capacity, rather than by selecting the periods in which an exceedance
  happened. The obvious alternative — keep only the periods in which load crossed the limit, and
  score those — is not merely noisy but biased: [Lerch et al.
  (2017)](https://doi.org/10.1214/16-STS588) show that choosing which periods to score on the basis
  of what happened rewards a forecaster who over-predicts extremes, and can rank a deliberately
  biased forecast above an honest one. [Gneiting and Ranjan
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
