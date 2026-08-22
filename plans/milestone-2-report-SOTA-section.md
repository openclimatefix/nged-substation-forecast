# The current state of the art in energy forecasting

Before we discuss the literature, there is a very important caveat to admit up-front: In 2026, no honest narrative review of the energy forecasting literature can claim to reveal the canonical "state of the art"! That is because (almost) all energy forecasting papers measure performance in different ways, against different datasets. It's like an international football tournament where every team plays by different rules, with different size goals.

Energy forecasting researchers have done great work over the years. But, unfortunately, the literature does not tell us how those approaches compare against each other, especially in messy "real world" energy forecasting scenarios. This isn't anyone's fault; it's a systemic failure. And the industry is already aware of this problem, and people are trying to fix it. But, at the time of writing, the literature cannot yet tell us the current state of the art solutions for the problems that NGED cares about.

Having said that, there have been some valiant attempts to compare multiple forecasting approaches against the same dataset. But none of these attempts directly address the main challenges relevant to Flexpectation. Before we discuss those attempts, we must emphasise two reasons for optimism:

Firstly, whilst we might not know exactly which algorithms provide the best forecasting performance, we do know how to go about finding out. There's no magic. Machine learning is an empirical science, and progress in it comes largely from testing many ideas under identical conditions and measuring carefully — because most ideas fail. John Jumper, who shared the 2024 Nobel Prize in Chemistry for his work on AlphaFold, puts the share of research ideas that fail at around 90%, and treats that rate as an ordinary and necessary feature of doing research rather than as evidence of doing it badly ([Nobel Week interview](https://www.nobelprize.org/prizes/chemistry/2024/jumper/interview/), 6 December 2024, from 14:12). If roughly one idea in ten survives contact with the data, ten attempts is simply the price of one result. So our task is to run hundreds of ML experiments, and then measure performance against the same dataset, using the same performance metrics.

Secondly - and perhaps most importantly - the fact that the industry doesn't yet know the state of the art is a huge opportunity for the Flexpectation project: We are in a very privileged position where we can try hundreds of ideas, and test the best ideas in the real world. We have a fantastic opportunity to make a significant contribution to the energy forecasting industry by publishing our "leaderboards of ML experiments", and hence help the industry as a whole to better understand how multiple approaches perform.

## What we read

This review cites around fifty-five published sources. We read most of the ones an argument rests on
in full; the rest were available to us only as an abstract, a preprint or part of a paper, and
wherever a claim rests on a partial read we say so at the point the claim is made. We also read the
published deliverables of nine GB network projects. The selection was deliberate rather than
systematic: a paper earned its place by bearing on a decision Flexpectation actually faces and by
changing something we believed. Papers may be missing for no better reason than that we did not find
them, and the section "What this review excluded, and why" lists what we knowingly left out. A
further group of papers is cited once each, for one specific result, rather than reviewed.

One concurrent project is cited more than any paper: Northern Powergrid's Artificial Forecasting, an
Ofgem Strategic Innovation Fund programme whose Alpha and Beta deliverables are both public, and
which has its own section below. Three further sources carry findings rather than comparable scores,
and are drawn on throughout. [Haben et al. (2021)](https://arxiv.org/abs/2106.00006) reviewed 221
low-voltage forecasting papers published to 2020. [Shukla and Hong
(2024)](https://doi.org/10.1049/stg2.12162) reports the BigDEAL Challenge 2022, a competition on
forecasting the *timing* of peak demand rather than its size, which drew 78 teams from 27 countries
and published its data alongside the paper. [Energy-Arena](https://arxiv.org/abs/2604.24705) is a
live public leaderboard rather than a competition — we could not extract the full paper and worked
from its abstract and the running platform, which today carries 24 challenges across prices, load,
wind and solar — eight scored as point forecasts, eight as quantiles and eight as ensembles.

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
almost no paper states its ensemble size.

**Errors normalised by something physical also transfer.** An error expressed as a fraction of a
substation's firm capacity or transformer rating means the same thing at every substation, whereas
an error expressed as a fraction of the load that happened to occur does not. An absolute error in
kilowatts or megawatts tells NGED nothing on its own, because it depends entirely on how big the
substation was, and none of the absolute figures below should be read as a target for this project.

**Whether a study used the weather forecast a real forecaster would have had changes what its
numbers mean.** In the table under problem 1 below, "real forecasts" means the weather forecast that
was genuinely available when the power forecast was made; "actual weather, after the fact" means
observations, or a weather model re-run after the event, that no forecaster would have had. Two of
the studies below, [Kaas et al. (2026)](https://arxiv.org/abs/2607.01966) and [Hertel et al.
(2026)](https://arxiv.org/abs/2607.15705), both on the same 200 German low-voltage feeders, use
actual weather after the fact — either short-range forecasts issued one to three hours ahead, or
reanalysis. [Hertel et al. (2026)](https://arxiv.org/abs/2607.15705) do so deliberately, because
their "primary goal is to compare models under fair conditions, which we achieve by using the same
data for all"; [Kaas et al. (2026)](https://arxiv.org/abs/2607.01966) use the weather their dataset
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

**In summary.** A large literature forecasts substation load, but almost none of it can be compared
with the rest of it, and none of it drives a probabilistic substation forecast from a weather
ensemble across a 14-day horizon.

**The problem.** Forecast net demand — demand minus whatever generation sits behind the substation —
at every grid supply point, bulk supply point and primary substation, half-hourly, 14 days ahead,
updated every six hours, as a range of possible loads with a probability attached to each rather
than as a single number. NGED acts on the forecast one to ten days ahead, and the question NGED asks
of the forecast is "how likely is load to cross this substation's firm capacity — the load the
substation can carry safely with its largest transformer out of service?" rather than "what is the
most likely load?". This is the highest priority of the eight problems, and the other seven exist
mainly to make that net-demand forecast better.

**The 14-day horizon sits at the edge of what a weather ensemble can supply.** [Buizza and
Leutbecher (2015)](https://doi.org/10.1002/qj.2619) put at 16 to 23 days the lead time beyond which
a weather ensemble stops beating a climatological distribution. The ensemble is the 51 slightly
different forecasts the European Centre for Medium-Range Weather Forecasts (ECMWF) runs from 51
slightly different starting conditions, whose spread shows how confident the forecast is; the
climatological distribution is the spread of weather actually observed on that day of the year over
many years. They measured that on upper-air variables rather than on the near-surface temperature
and irradiance that drive substation load, for which we would expect a shorter horizon. That
measurement is also now eleven years old, and the ensemble has improved since: ECMWF's headline
skill scores have advanced by roughly a day per decade, so today's horizon is probably a little
longer than the figure quoted. We found no more recent study measuring the same quantity, so we
quote the 2015 figure and treat it as a lower bound rather than a current reading.

**What the literature reports.**

| Source | What they forecast | Level and scale | Horizon | Result, and what it was compared against | Weather |
|---|---|---|---|---|---|
| [Kaas et al. (2026)](https://arxiv.org/abs/2607.01966) | Net load, Germany | Low-voltage feeder: 200 | 4 days | A general-purpose foundation model that had never seen the data beat every purpose-trained model on average error, 3.8 kW against 4.2 kW | Actual weather, after the fact |
| [Hertel et al. (2026)](https://arxiv.org/abs/2607.15705) | Load, Germany and Portugal | Transmission, plus 200 low-voltage feeders and 287 individual customers | 4 days | Best model beat a day-type persistence forecast by 59.6% at transmission level, 42.3% at low-voltage feeders, 23.3% at individual customers | Reanalysis and 1–3 h forecasts |
| [Browell and Fasiolo (2021)](https://arxiv.org/abs/2103.10335) | Regional net load, GB | Regional: 14 grid supply point groups | Day-ahead | Held the same risk with **up to 24.6% less upward reserve** than a fixed-tail alternative (note 1) | Real forecasts |
| [Pinheiro et al. (2023)](https://doi.org/10.1016/j.apenergy.2022.120493) | Load, Portugal | Secondary substation: 96,989 | Day-ahead | 42–47% better than the reference benchmark at system level. **At substation level, beat a naive forecast on 83–87% of network-owned and 66–70% of customer-owned sites** | Real forecasts, 7–8 h old |
| [Gilbert et al. (2023)](https://arxiv.org/abs/2206.11745) | Load, GB | Four levels: primary substation down to household | Day-ahead | Combining forecasts gained **0.0–0.4% averaged over all periods**, but **5.7–9.0% when restricted to peaks** | None at all |
| [SSEN TRANSITION 2021](https://ssen-innovation.co.uk/transition/) | Net load, Oxfordshire | Primary substation: 13, plus their bulk supply points and 11 kV feeders | 30 min to 10 days | **11 of 13 primary substation models below 10%** mean absolute percentage error when fitted (note 2) | 40-member ICON-EU ensemble to 4 days, then one deterministic forecast to 10 days |
| [Artificial Forecasting (Northern Powergrid)](https://smarter.energynetworks.org/projects/npg_sif_006-1/) | Demand and export at primary substations; active power at secondary | Primary substation: 551 with export data, 171 modelled; secondary: 729 | Day-ahead to 11 days at primary; week- to month-ahead at secondary | **About 8% lower mean absolute error** of utilisation rate than the network's existing method (note 3) | Real forecasts at primary; none in the published secondary results |
| [Ruhhütl et al. (2023)](https://doi.org/10.1049/icp.2023.0476) | Load and generation, Austria | Substation | Day-ahead | **3 to 8% mean absolute percentage error**, varying with how industrial and how large the supplied area was; linear and Gaussian regression preferred over the alternatives tested (abstract only) | Not stated in the abstract |

*Notes.* **1.** The 24.6% saving is at the most extreme tail level [Browell and Fasiolo
(2021)](https://arxiv.org/abs/2103.10335) tested, and falls to 3.2% at the least extreme. **2.** The
two SSEN TRANSITION models that missed 10% reached 13.4% and 19.7%, and 94% of the 11 kV feeders it
built models for came in below 20%. **3.** Artificial Forecasting also captured 83% of the top 10%
of demand values inside its 5th-to-95th-percentile band, and beat its comparison benchmarks at all
eight of the near-capacity substations it was evaluated on. **4.** The beat-a-naive-forecast figures
are given as ranges because [Pinheiro et al. (2023)](https://doi.org/10.1016/j.apenergy.2022.120493)
reports two different pairs of numbers for the same statistic: 82.8% and 66.0% in the body text,
86.5% and 70.0% in the caption of the figure on the same page. We could not tell which is intended,
so the table spans both.

**Even within this one table, the studies cannot be compared with each other.** The sharpest
illustration comes from two papers published a fortnight apart, by overlapping groups at the
Karlsruhe Institute of Technology, on the same 200 German low-voltage feeders. [Kaas et al.
(2026)](https://arxiv.org/abs/2607.01966) and [Hertel et al.
(2026)](https://arxiv.org/abs/2607.15705) name different models as best. Inside [Kaas et al.
(2026)](https://arxiv.org/abs/2607.01966), mean absolute error and an overload-decision metric name
different winners again. Neither disagreement is a mistake: the two papers test different sets of
models at different time resolutions, and the two metrics answer different questions. "Publishing
results that others can compare against", the last section of this review, returns to what follows
from those results being incomparable.

**One study in the table shows how much an annual average hides.** [Gilbert et al.
(2023)](https://arxiv.org/abs/2206.11745) forecast load at four levels of a hypothetical GB
distribution hierarchy, from a primary substation down to individual households — built by
aggregating 742 smart meters, so their top level is, as they say themselves, smaller than a real
primary substation — and combined a conventional half-hourly forecast with a bespoke daily-peak
forecast. Averaged over every period, that combination gained 0.0–0.4% over the conventional
forecast alone, indistinguishable from nothing, and a result that would ordinarily end the
investigation. Restricted to the periods containing the daily peak, the same comparison gave 5.7% at
the primary substation, 9.0% at secondary, 8.2% at feeder level and 6.0% at household level.
Combining the two forecasts was always worth having, and we know that only because [Gilbert et al.
(2023)](https://arxiv.org/abs/2206.11745) reported both numbers.

**The same paper found that the ability to predict *when* the peak will happen falls away further
down the network.** At the primary substation, peak timing was predicted more than 20% more
accurately than a long-run seasonal average would have managed; at four of the feeders, no better
than that seasonal average at all. And at household level during peak periods, both of their
individual forecasts were worse than a trivial benchmark based only on the time of day; only the
combination of the two beat it. Together, the peak-versus-average gap and the collapse in peak
timing are the strongest measured argument in this review for the tail and exceedance metrics
Flexpectation is building.

**The closest analogue to Flexpectation in a live setting is a Portuguese production system.**
[Pinheiro et al. (2023)](https://doi.org/10.1016/j.apenergy.2022.120493) run a production
forecasting system covering 96,989 secondary substations day-ahead, using real weather forecasts
with a realistic 7–8 hour delay. It is the only study in this review running in live production at
national scale. Two of its lessons shape how we will report: the fraction of substations beating a
naive forecast belongs alongside any pooled average, and expectations for single-customer sites
should be set low from the outset.

**The cheapest positive result in this review also comes from that system.** Combining eight copies
of the same model, one fitted per calendar regime — weekday, weekend, public holiday and so on —
with the weights updated as new data arrived, cut system-level root-mean-square error by 24%.

**Where the gaps are: no study we found drives substation uncertainty from a weather ensemble across
a full 14-day horizon.** GB practice is further ahead than the academic literature here, but stops
short. [Taylor and Buizza (2002)](https://doi.org/10.1109/TPWRS.2002.800906), which we read in part,
pushed all 51 ECMWF members through a load model for midday demand in England and Wales at one to
ten days ahead in 2002, and [Ludwig et al. (2023)](https://doi.org/10.1080/01605682.2022.2115411)
revised that approach, adding a step we will need: raw ensembles are biased and their spread is too
narrow, so they look more certain than they really are, and they must be bias-corrected before the
load model sees them or the resulting uncertainty bands are wrong. What we did not find is
ensemble-driven uncertainty at half-hourly resolution, per substation, across a full 14-day horizon
— and both [Haben et al. (2021)](https://arxiv.org/abs/2106.00006) and [Ludwig et al.
(2023)](https://doi.org/10.1080/01605682.2022.2115411) ask for exactly that in print. [Haben et al.
(2021)](https://arxiv.org/abs/2106.00006) put it as a request "to use post-processed weather
ensemble predictions to generate multi-step probabilistic forecasts of load at different levels of
the LV [low-voltage] hierarchy".

**The ensemble itself is being replaced, which turns this gap into a question we can answer
directly.** ECMWF's own machine-learned ensemble,
[AIFS-ENS](https://doi.org/10.1038/s44387-026-00073-7), has been operational since 1 July 2025 with
51 members, 6-hourly to 15 days, and beats the physics ensemble on the majority of variables and
lead times; [GenCast](https://doi.org/10.1038/s41586-024-08252-9) beats it too. Flexpectation runs
on the physics ensemble today, and whether a machine-learned ensemble forecasts substation load
better is something we can measure.

**Almost every study here optimises average accuracy, but NGED's question is about the top of the
distribution.** The largest competition in this review, HEFTCom — described under problem 2 below —
scores only the 10th to 90th percentiles. [Browell and Fasiolo
(2021)](https://arxiv.org/abs/2103.10335) is the one study here that models the upper tail
explicitly, and what they found is a warning rather than a reassurance: they set reserve at a risk
level of one part in two thousand — enough to cover all but about four hours a year — but they also
find that "below 1% and above 99% the forecasts based on quantile regression only are not calibrated
at any GSP Group. Therefore, these quantiles are not suitable for use in decision-making", even with
five years of half-hourly data across regions far larger than a substation. Above those limits they
switch to a fitted parametric tail.

**How far into the tail a forecast of a single substation stays trustworthy is an open question, and
one this project can answer.** Our series are smaller and noisier than the regions [Browell and
Fasiolo (2021)](https://arxiv.org/abs/2103.10335) worked on, so we expect a narrower reliable range,
and a parametric tail is likely to be necessary rather than optional. We will measure where ours
stops and publish the answer, because a network buying flexibility needs to know which percentile it
can act on.

**A decision metric that holds risk constant and prices it in money has been published at
distribution level once, on a synthetic network.** [Bernecker et al.
(2025)](https://doi.org/10.1016/j.ijepes.2025.110713) fix the confidence level at which a network
operator acts, at 95%, and compare what two forecasts cost that operator in congestion management:
**3,102 euros a year using standard load profiles against 86 euros using a smart-meter-informed
forecast**, a 97% reduction, alongside a 90% fall in the number of voltage violations. They also
give the exchange rate NGED would want — a 1% cut in the standard deviation of forecast error is
worth about 1.4% of congestion-management cost. We read the sections of that paper bearing on the
cost calculation rather than the whole of it. Two things keep the gap open: the network is a
modified IEEE 33-node test system rather than a real one, and what they compare is two *information
levels*, not two forecasting models, so the metric has never been used to rank one forecast against
another at a real substation.

**The rest of the decision metric exists in pieces.** [Browell and Fasiolo
(2021)](https://arxiv.org/abs/2103.10335) fix a risk appetite, compute the reserve volume each
forecast would need to hold it, and compare — the harder half of the job, done across whole grid
supply point groups. [Angus et al. (2027)](https://doi.org/10.1016/j.epsr.2026.113545) bring that
idea down to individual assets, forecasting day-ahead how hard each of 644 low-voltage transformers
in GB can safely be pushed, and winning 10 to 12% more capacity than a fixed setting while the risk
of overheating came out at whatever percentile they asked for; we read their preprint rather than
the published paper. Artificial Forecasting's Alpha work calculates the extra flexibility volume
that forecast error would make a network procure: 20,536 kWh implied by a risk-aware forecast
against 5,495 kWh actually needed, over two eight-day windows at one near-capacity substation. Its
Beta phase goes further, making exceedance true- and false-positive rates key metrics; its Alpha
phase already scored precision and recall for the half-hours at or above 90% of a substation's firm
capacity.

**What is still missing is the price on a real network.** Meteorology has priced forecast decisions
this way for decades: [Richardson (2000)](https://doi.org/10.1002/qj.49712656313) computed the
relative economic value of the ECMWF ensemble across the whole range of ratios between the cost of
acting on a forecast and the loss avoided by acting. The relative-economic-value curve of Richardson
(2000) is the right shape for NGED's problem, because each substation has its own firm capacity and
its own cost of being wrong, so a single assumed cost ratio is the thing to avoid. Every published
version of it on a real distribution network, though, is denominated in energy volumes or in spare
capacity rather than in money. Artificial Forecasting does put a price on its service, but that is a
business case for a programme rather than a score that holds risk constant and can rank one forecast
against another at one substation.

**Topology enters this literature almost entirely as one thing: the summation constraint in
hierarchical forecast reconciliation.** [Nespoli et al. (2019)](https://arxiv.org/abs/1910.03976)
apply it to real secondary substations and cabinets in a Swiss distribution grid and gain up to 10%
in root-mean-square error at the upper levels of the hierarchy, and under 1% at the bottom. A
summation constraint says only that the substations beneath a bulk supply point must add up to it.
It carries no information about which substation neighbours which, and it stops holding the moment
the network is switched into an abnormal running arrangement (problem 4 below). That is why a
summation constraint is not enough for Flexpectation. The nearest thing to an exception we found is
[Jung et al. (2024)](https://doi.org/10.1049/icp.2024.1900), who feed which busbar connects to which
into a graph neural network — but they forecast voltage rather than load, and test their model only
in simulation; we read their abstract rather than the full paper. Otherwise, information is shared
across substations statistically rather than topologically — one of the four models Artificial
Forecasting tested at secondary substations, a hierarchical Bayesian linear regression, trains its
upper layer across a cluster of similar substations, though the model they recommended is trained
per substation — and [Gilbert et al. (2023)](https://arxiv.org/abs/2206.11745) forecast four levels
of a hierarchy separately before naming exploitation of that hierarchy as future work. SSEN
TRANSITION is the exception that shows the value: it used the connectivity map throughout.

**The nearest answer to whether the shape of the network improves the forecast was measured on
NGED's own published data, and it points away from geography.** [Campagne et al.
(2025)](https://arxiv.org/abs/2507.03690) compare seven graph neural network architectures against
feed-forward and foundation-model baselines on French regional load and on the GB distribution
networks' open smart-meter feed — around two million meters and 50,000 substations across NGED's and
SSEN's areas. Graph-aware models beat the baselines on both. But which graph wins changes with
granularity: spatially informed graphs worked on the coarse French regions, whereas "for the UK
data, data-driven graphs proved more suitable since that dataset exhibits finer spatial granularity
and noisier correlations". They are explicit that reproducing the network is not the goal — "the
objective in forecasting is not to reproduce the transmission network itself, but rather to
construct a representation that best reflects the correlations driving demand patterns". Their
graphs are built from geographic distance or from correlation between series, never from electrical
connectivity, so the specific question stays open.

**Does knowing the shape of the network make the forecast better, or only more consistent?** NGED
holds a map of which substations and metered generators connect to each other, and no study we found
has used that map as a forecast input. The map makes it possible to forecast a bulk supply point
both directly and by summing everything beneath it, and to treat the disagreement between the two
answers as a check on both. We will report whether it improves accuracy as well.

### 2. Forecasting metered generators

**In summary.** Forecasting wind and solar from a weather forecast is the mature case, and one paper
matches Flexpectation's problem closely; nothing we found forecasts a distribution-connected
battery, gas generator or biofuel plant inside a net-demand forecast.

**The problem.** Twelve of the 32 series in the trial area are individually metered generators — six
solar farms, three wind farms, a biofuel plant, a battery and a gas generator — and each needs the
same probabilistic, half-hourly, 14-day forecast as a substation. Solar and wind are driven by
weather the ensemble supplies directly. The battery, the gas generator and the biofuel plant are
dispatched on market prices and operator decisions, and no weather forecast contains either.

**Forecasting wind and solar output from a weather forecast is the most mature problem on this list,
and the one problem where different studies' results can be compared directly.** [Browell et al.
(2025)](https://doi.org/10.1016/j.ijforecast.2025.10.005) report the Hybrid Energy Forecasting and
Trading Competition (HEFTCom), in which every team forecast the combined day-ahead output of one GB
portfolio — the 1.2 GW Hornsea 1 offshore wind farm plus the aggregate solar capacity of East
England, about 3.6 GW together — from real weather forecasts as they arrived. The winning team
scored a mean pinball loss of 22.18 MWh against the organisers' quick-start benchmark of 53.58, with
the next two teams on 23.18 and 24.64. The organisers also entered a more competitive reference,
unranked, which scored 25.38. HEFTCom is the largest competition in this review and the only fully
probabilistic one, so it is the clearest case of many teams forecasting the same data with the same
metric — which is exactly what the rest of this literature cannot do. Its wind half is a single
offshore farm far larger than any generator NGED meters, and its solar half is a regional aggregate
rather than a plant.

**At the scale of an individual generator, the closest work is on wind.** [Dantas and Browell
(2026)](https://doi.org/10.1002/we.70079) forecast 73 wind farms in GB — 34 onshore, 39 offshore —
from the ECMWF ensemble, seamlessly from 6 to 162 hours ahead. That is the same driver, the same
horizon band and the same probabilistic form Flexpectation needs for its three wind sites, and it is
also where the effective-capacity method described under problem 3 comes from. Because it is the
closest published work to what Flexpectation has to build, the rest of this subsection sets out what
their method does and what we should take from it.

**Their method separates the two things that can go wrong, and that separation is the paper's main
result.** A wind power forecast can be wrong because the weather forecast was wrong, or because the
conversion from weather to power was wrong. Dantas and Browell quantify both. They fit a
gradient-boosted quantile regression tree that maps weather to power, trained on ECMWF's operational
analysis rather than on forecasts so that the errors of the weather forecast do not contaminate the
conversion model, and they then push each of the ensemble's 50 members through that tree separately.
They treat the members as exchangeable, which is why one conversion model serves all of them; note
that they use the 50 perturbed members, whereas Flexpectation reads all 51 fields ECMWF publishes,
the 50 perturbed members plus the control. Each member emerges as a normal distribution whose width
is set by that member's own predicted interquartile range, so a member the conversion model is
unsure about is given a wider kernel than one it is confident about. The 50 kernels are merged with
the beta-transformed linear opinion pool of [Gneiting and Ranjan
(2013)](https://doi.org/10.1214/13-EJS823), whose parameters are fitted separately at each lead time
by minimising CRPS. That fitting step corrects the miscalibration which arises because the members
are not independent of one another. The output is a full predictive distribution at each lead time.

**Their headline conclusion is that which of those two uncertainties dominates flips with lead time,
and that where it flips varies a lot between sites.** In their words, "weather-to-power uncertainty
dominates short-term forecast performance, while weather forecast uncertainty dominates mid-term.
Typically, the transition from one situation to the other is 2 to 3 days ahead but can vary
dramatically between wind farms. The transition typically occurs at shorter lead times for offshore
wind farms compared with onshore." Handling both is what lets one model cover 6 to 162 hours,
whereas the field had previously used a short-term model and a separate mid-term one. Flexpectation
faces the same seam over its 14-day horizon, and this paper is evidence that the seam can be removed
rather than managed.

**A second conclusion is more uncomfortable for a project built on an ensemble: a deterministic
forecast at higher resolution beat the ensemble at short lead times.** Their short-term reference
method uses ECMWF's deterministic HRES at 0.1° and hourly steps, while their own method uses the
ensemble at 0.5° and 6-hourly steps, because the archive they drew on carries no 100 m wind and no
finer ensemble. On those unequal terms "the short-term method is better than the proposed method for
horizons up to the day ahead", although it "cannot outperform the proposed method for horizons
beyond 1 day ahead". Give both methods the same resolution and the same variables and their own
method wins at every horizon. The lesson for Flexpectation is that a comparison of ensemble against
deterministic measures the resolution difference unless the resolution is equalised first.

**A third conclusion is about what an average score hides.** Averaged over five years, their method
showed no gain over the state of the art at day 0 and day 1. Restricted to the periods when the
ensemble members disagreed most — frontal passages and the like — it showed a real gain even at
those short lead times, "which was not evident in the long-run average CRPS", because a
deterministic method "is not able to discriminate between high/low weather uncertainty". They argue
the field should score those periods separately: "assessing forecast performance under
high-uncertainty events should be considered in recommended practices". Flexpectation should follow
that, because the hours when the ensemble spread is wide are the hours when a network operator has a
decision to make.

**Two things their method does not do are things Flexpectation will need.** They fit a separate
model per wind farm rather than one model across all 73, and they state that their forecasts carry
no coherence across sites or across lead times, listing "member-by-member correction to retain
spatio-temporal structure" as future work. A net-demand forecast that adds several generators and a
substation together needs precisely that coherence, which cannot be taken from this paper. Against
that, their design is cheap: it fits three trees per wind farm however many quantiles and horizons
are wanted, whereas the deterministic short-term reference needs one tree per quantile per horizon,
513 of them for the 19 quantiles and 27 horizons they report.

**Where the gap is: nothing we found forecasts a distribution-connected battery, gas generator or
biofuel plant inside a net-demand forecast.** For the battery there is at least a method to borrow.
[Bian et al. (2024)](https://doi.org/10.1109/TSG.2023.3303469) recover a price-taking storage
operator's own optimisation parameters by gradient descent on historical prices and observed
dispatch, and prove the recovered parameters converge to the true ones for a class of storage models
— we read their abstract rather than the full paper. Their motivation, that "future power system
operators must understand and predict strategic storage arbitrage behaviors", is NGED's. We found no
method worth borrowing for the gas generator or the biofuel plant; what little exists forecasts such
a plant's own output directly rather than as a component of a substation's net demand. Otherwise the
closest the literature comes is a warning rather than a method: [Pinheiro et al.
(2023)](https://doi.org/10.1016/j.apenergy.2022.120493) found that sites serving a single customer,
whose load follows decisions no weather model can see, were forecast markedly worse than the rest
(finding 3 below). We expect the battery, the gas generator and the biofuel plant to be the hardest
series in the trial area for the same reason, and we will report them separately rather than pooled
with the wind and solar sites.

### 3. Estimating the effective capacity of metered generators

**In summary.** A method exists for each generation technology separately, but nobody has run them
across a mixed fleet at a distribution network, or tested whether estimating capacity improves the
forecast.

**The problem.** We call the amount of generation actually available at a metered site its
*effective capacity*: the output it could produce right now if the weather allowed, as opposed to
its nameplate rating. Turbines go out for repair, inverters degrade, and sites are curtailed — told
by the network operator to generate less than they could. A 20 MW wind farm that has been limited to
14 MW for a month is, for forecasting purposes, a different wind farm, and a model trained on its
nameplate rating cannot see the difference. The same goes for a primary substation with a large
metered generator connected behind it. This problem concerns the 12 metered generators in the trial
area, each of which has a half-hourly meter of its own; the unmetered rooftop solar and small wind
of problem 7 are a separate task.

**For wind, one paper hits our problem exactly, and publishes its method.** [Dantas and Browell
(2026)](https://doi.org/10.1002/we.70079) needed available capacity for the same reason we do: the
metered-output database they use "does not include information related to the farms' available
capacity over time", so rather than use a nameplate rating they estimate a time series of available
capacity for each farm and normalise that farm's power by it before modelling. Their method needs no
capacity register and no outage messages. A two-hour stretch of near-constant output while the wind
is above the speed at which a turbine reaches full power marks the farm as running at everything it
has, and capacity is then held at that level until the meter exceeds it. Because their database
names no turbine model, they infer that wind speed from the site's own distribution of wind speeds,
and they take the wind speed itself from reanalysis rather than from any instrument at the farm.
They did use one data source Flexpectation will not have in the same form: they excluded curtailed
half-hours using published bid-acceptance volumes, which exist for transmission-connected wind farms
and not for NGED's embedded generators. Flexpectation has something adjacent, in that the active
network management system records curtailment for each of NGED's generator customers, but that
record is ambiguous enough that it cannot simply be dropped in where Dantas and Browell use a
bid-acceptance volume.

**That hold-until-exceeded rule has since been criticised in print, on the grounds that matter most
to NGED.** [Viotti et al. (2026)](https://doi.org/10.1002/we.70136) point out that taking the
running maximum of production "requires monotonically increasing capacity and relies on frequent
high wind events" — and NGED's effective capacity goes *down* when a turbine is out for repair,
which is the case the review exists to handle. They fit the most likely capacity time series
instead, by quadratic optimisation against a capacity factor simulated from reanalysis and a power
curve, and report **27.2% lower normalised mean absolute error** than the running maximum at
quantifying capacity after a step change. They also measure what the choice is worth downstream: a
forecasting model trained on production normalised their way scored **2.0% lower mean absolute error
and 2.3% lower root-mean-square error** day-ahead than the same model normalised by the running
maximum. Their target is a Swedish bidding zone rather than an individual farm, but they test the
de-rating case explicitly by suppressing production after a step.

**For solar, the equivalent can be done from the power signal and nothing else.** The best-known
tool, the open-source [RdTools](https://doi.org/10.5281/zenodo.1210316), does need site irradiance
to pick out the clear-sky periods it analyses, and its own documentation warns that a satellite
substitute gives less stable results. But [Meyers et al.
(2020)](https://doi.org/10.1109/JPHOTOV.2019.2957646) removed that requirement: their unsupervised
signal-processing approach "only requires a measured power signal as an input — no irradiance data,
temperature data, or system configuration information are required", and they validate it against
RdTools on the same dataset, reporting greater robustness to data anomalies; we read their abstract
and the package documentation rather than the full paper. Their approach is now the open-source
Solar Data Tools, whose pipeline detects capacity changes and clipping and estimates degradation,
with a Monte Carlo step that returns a distribution rather than a point estimate. Independent work
reaches the same place from other directions: [Cronin et al.
(2014)](https://doi.org/10.1002/pip.2310) recover relative degradation rates by comparing daily
yields across a group of systems, which maps onto the six solar farms in the trial area, and
[Peratikou and Charalambides (2022)](https://doi.org/10.1016/j.seja.2022.100015) compute clear-sky
output from photovoltaic data alone; we read the abstracts of both rather than the full papers.

**Detectors aimed at a specific fault mode work well or badly depending on which mode.** [Mendonça
Severiano et al. (2026)](https://doi.org/10.1016/j.solener.2026.114382) classify underperformance
across 1,089 systems from inverter data a network operator does not receive, and catch clipping —
when the panels produce more than the inverter can pass through — only about half the time. But
[Perry et al. (2021)](https://doi.org/10.1109/PVSC43889.2021.9518733) score clipping detectors
against expert labels on 36 systems from alternating-current power alone, and a logic-based detector
reaches an F-score of 85.0 against 56.4 for the RdTools method, with the choice of detector shifting
the estimated degradation rate by up to 0.6% a year; we read their abstract rather than the full
paper.

**At substation rather than generator level, Artificial Forecasting gets closest.** Its Alpha work
builds the baseline it forecasts against by scaling Northern Powergrid's own installed-capacity
projection down by the fraction of that capacity actually generated in 2021–22. Separately, it found
that the National Energy System Operator's national generator-availability signal "almost
universally substantially improved results" at wind-connected primary substations — the nearest
thing in this review to reading effective capacity off an external feed.

**Estimating capacity as a distribution, jointly with the forecast, has also been published.**
[Pierrot and Pinson (2024)](https://doi.org/10.1080/00401706.2024.2350421) treat a wind farm's
available capacity as the unknown, time-varying upper bound of a generalized logit-normal
distribution and track it online by normalised gradient descent, fitting the bound and the forecast
together rather than in two stages. On 14 months of ten-minute data from the Anholt offshore wind
farm, that improved the continuous ranked probability score by **34.2% over probabilistic
persistence and 17.9% over the same model with the bound held fixed**. Their motivation is NGED's,
in their own words: the bound "may change over time, while being unknown, for example in case of
curtailment actions for which information is not available or not reliable".

**Where the gap is: none of this has been done across a mixed fleet at a distribution network, or
tested for whether it improves the forecast NGED buys flexibility against.** The pieces exist, and
most of them work from a revenue meter alone. What nobody has published is the combination: solar,
wind and dispatchable sites at one distribution network operator, each with its capacity tracked
from its own meter, feeding a 14-day probabilistic forecast, with the improvement measured rather
than assumed. Where richer data does exist, the state of the art still simply reads capacity off a
register — the team that won HEFTCom clipped its forecast quantiles to the maximum capacity implied
by published outage notices, and NGED's embedded generators publish no such notices. Part of the
problem is a data question rather than a modelling one, because much distribution-connected
curtailment in GB is instructed by the network operator under active network management, so for
those sites the curtailment component is already known inside NGED.

**We plan to attempt this two ways, neither of which starts from scratch.** The first is the
two-stage route: estimate a capacity time series from the meter, then normalise by it before
training — running the method of [Dantas and Browell (2026)](https://doi.org/10.1002/we.70079), the
quadratic-optimisation alternative of [Viotti et al. (2026)](https://doi.org/10.1002/we.70136) and
the Solar Data Tools pipeline against each other on our own sites. The second is joint estimation,
of which [Pierrot and Pinson (2024)](https://doi.org/10.1080/00401706.2024.2350421) are the
published precedent: a differentiable-physics model of each generator in which the physical
parameters — including the plant's direct-current and alternating-current capacity — are fitted as
probability distributions rather than as single numbers, so that capacity is recovered with its own
uncertainty attached and the forecast inherits that uncertainty instead of treating capacity as
known. What our second approach adds beyond [Pierrot and Pinson
(2024)](https://doi.org/10.1080/00401706.2024.2350421) is fitting capacity alongside the rest of the
plant's physics rather than as a single scalar bound, and doing it for solar as well as wind. Public
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
1's export cable faulted on 19 January 2024, about two weeks before HEFTCom's competition period was
due to start. The competition began on schedule on 1 February, and it was only afterwards that the
organisers realised they had not accounted for the fault — they call it "an oversight" — so they
restarted the competition on 20 February, a month after the fault. Many teams still struggled in the
weeks that followed. Teams forecasting wind and solar separately could post-process their wind
forecast for the new export limit, while those forecasting the combined total "found it harder to
adapt", and the organisers' benchmark, which took no account of the fault, "performed extremely
poorly as a result".

### 4. Detecting switching events

**In summary.** One paper detects switching at a real network operator, using a bottom-up reference
series NGED does not have; the GB precedent drew the same distinction in 2018 but never measured how
often it was right.

**The problem.** When a cable fault or planned maintenance moves part of a network from one
substation to another, the load a substation meters steps up and its neighbour's steps down, with no
change in the underlying demand. NGED's substations spend roughly a tenth of their operating time in
an abnormal running arrangement. Switching labels exist for the 32-series trial area but not for the
wider network, so a method that is to scale has to work from power measurements alone.

**One paper detects these events at a real network operator, in order to strip them out before
estimating how much load a substation carries.** [Bouman et al.
(2024)](https://arxiv.org/abs/2405.16164), working with the Dutch network operator Alliander, study
180 primary substations at 15-minute resolution over roughly a year, detecting the step changes
caused when a cable fault or planned maintenance reroutes part of a subgrid to a different
substation. Events run from a few minutes to several months. They estimate annual maximum and
minimum load within a 10% margin in 88% and 91% of cases. Their purpose is a clean capacity-planning
figure rather than a forecast: with the contaminated periods removed, the annual minimum and maximum
load estimates stop being inflated — by up to 300% at worst when no filtering is applied. It is the
most directly useful paper in this review, and it leaves the forecasting half untouched, which is
what Flexpectation would add: keeping a forecast running through the events rather than deleting
them.

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
sign change — the identical defect at ten of the 32 series in NGED's trial area, primary substations
among them. Their bottom-up estimate is built from measurements that record the direction of flow,
so [Bouman et al. (2024)](https://arxiv.org/abs/2405.16164) take the direction from the estimate
rather than from the meter. Any reference series that records direction independently would work the
same way. They also report that when their bottom-up estimate fails, the cause is usually wrong
topology data rather than a bad algorithm — a warning about the network records that any such
estimate depends on.

**A GB network operator separated switching from bad data in 2018, with cruder tools and no
published accuracy.** Electricity North West's
[ATLAS](https://smarter.energynetworks.org/projects/nia_enwl008/) project processed five years of
half-hourly demand for "over 70 BSPs and 380 primary substations" — a larger GB fleet than
Flexpectation's trial area by two orders of magnitude. It works in stages. The first flags any
abrupt change, using a threshold of 80% of the standard deviation of the demand series. The second
then decides what kind of change it was, and this is the part that matters here: one rule handles
blocks of "unreasonably zero or negative demand", and a separate rule handles "switching operations
and network reconfigurations", each firing when the block's mean sits more than a set percentage
away from the mean of the whole series — 50% for the zero-or-negative-demand rule, 30% for the
switching rule. So the distinction between a broken meter and a reconfigured network was drawn on GB
primary substations, on power alone, without a bottom-up reference series. What ATLAS never reports
is how often either rule is right: there are no precision or recall figures anywhere in its
documents, and the project falls back on "the importance of visual sense checks of the obtained
processed demand data". That is the shape of the GB precedent — the problem was recognised and a
rule was written, but nobody measured the rule.

**Where the gaps are: the published method detects on a residual we cannot build the same way, and
the events NGED cares about are harder than the ones detected.** A switch at NGED usually fans out
to two or three neighbouring substations rather than one, and the common case is a *partial*
transfer — a continuous fraction of the load moving, with no minimum size — rather than a whole
subgrid. There is no voltage measurement at primary substation level to fall back on, and
tap-changing transformers plus half-hourly averaging would blur it if there were. So the detector
has to work unsupervised, on power alone, against events that are partial, multi-recipient and
unlabelled.

### 5. Forecasting a substation as if it were always in its normal running arrangement

**In summary.** Researchers either leave the level shifts in and pay for them, rewrite the history,
or adapt to the new level; we found nobody who feeds the contamination to a model deliberately, as
information.

**The problem.** NGED plan the network against what each substation would carry under its normal
running arrangement, so that is what the forecast has to predict — including for a substation that
has been sitting in an abnormal arrangement for weeks. That makes the target a quantity that was
never metered, and it makes the training history contaminated: past readings taken while the network
was abnormally configured describe a different substation from the one being forecast.

**Researchers respond in one of three ways, and two of the three alter the series the model is
trained to predict.** One strand leaves the level shifts in and reports the damage:
[Huyghues-Beaufond et al. (2020)](https://doi.org/10.1016/j.apenergy.2019.114405) run change-point
detection across 342 UK medium-voltage feeders, but use the change-points only to bound the segments
within which they remove *outliers*. The shifts themselves stay in training and test data, and the
paper reports that they bias the fitted parameters and hurt the forecast — while also concluding
that the forecasters "handle level-shifts well by adapting quickly to changes". A second strand
rewrites the history to an "as if never switched" level: [Paredes and Vargas
(2017)](https://doi.org/10.1049/iet-gtd.2017.0129) do it across 169 real feeders and report better
medium-term forecasts for it, and Artificial Forecasting does the same in its data-preparation
pipeline, rescaling each step-change block onto the level of the block before it whenever the newer
block's median falls outside the earlier block's 10th-to-90th-percentile range, so the history is
kept rather than dropped. Artificial Forecasting argues for going further on the grounds that demand
changes of an order of magnitude, mostly caused by network reconfigurations, "cannot be directly
handled even by powerful nonlinear models like neural networks" — though they add that changes that
large are rare at their secondary substations. [Gilbert et al.
(2023)](https://arxiv.org/abs/2206.11745) name adaptive handling of structural breaks as future
work. The third strand, models that adapt to the new level instead of touching the training series,
is the subject of the next paragraph.

**Adaptive models are the live alternative, and they handle gradual change rather than steps.** [de
Vilmarest et al. (2024)](https://doi.org/10.1109/TPWRS.2023.3310280) let a Kalman filter track the
drift on the 14-region GB dataset of [Browell and Fasiolo (2021)](https://arxiv.org/abs/2103.10335)
instead of correcting the history, cutting error by about 4% in 2019, 7% in 2020 and 8% in 2021
against the same model refitted every day. But a switching event is a step, not a drift, and a model
that simply adapts to a new load level never learns that switching happened — so it cannot report
what the substation would have carried under its normal arrangement, which is the quantity NGED
needs.

**Where the gap is: we found nobody who feeds a model switching-contaminated history *deliberately*,
as information rather than as damage.** [Huyghues-Beaufond et al.
(2020)](https://doi.org/10.1016/j.apenergy.2019.114405) leave the shifts in, but as a side-effect of
cleaning for outliers, and they report the result as a cost. The question we want to settle is
whether the contamination can be made to earn its place. Instead of correcting the series, a model
could be fed the difference between what a substation actually metered and what a model that ignores
network topology expected it to meter. That plays the same role as the residual [Bouman et al.
(2024)](https://arxiv.org/abs/2405.16164) detect on, though it is built differently: theirs is
metered load minus a topology-informed reconstruction, which goes stale the moment the network is
switched, whereas ours would be metered load minus a model that never sees topology at all. Used as
a forecast input rather than as a detector, that difference would mean a reading taken while the
network was abnormally configured would still carry information without anyone having to estimate a
level correction first. A negative result here would still be valuable: evidence that switching
cannot be recovered from power data alone would strengthen the case for taking switching labels from
operational systems instead — a route Artificial Forecasting has already identified, naming the
incorporation of planned-outage records in its post-Beta roadmap.

### 6. Detecting faulty metering

**In summary.** Faulty metering is usually a data-cleaning step mentioned in passing rather than a
problem in its own right, the only labelled dataset is Dutch, and recovering the direction of flow
from a magnitude-only meter was attempted by this network's predecessor and left unfinished.

**The problem.** NGED's telemetry carries stuck values that repeat unchanged for hours or days,
zeros that mean "no reading" rather than "no load", physically impossible values, and gaps running
from a single half-hour to several months. Ten of the 32 series in the trial area are metered in
apparent power only, so they report magnitude without direction and reverse flow appears as a rise:
at one primary substation the meter bounces off zero on sunny days, when a solar farm behind it
exports. A model trained on uncleaned data learns the fault, and a forecast that fails silently
because its recent history was stuck is worse than one that says it is degraded.

**The most useful published method treats faulty metering and switching as one problem.** [Bouman et
al. (2024)](https://arxiv.org/abs/2405.16164) treat measurement errors and switch events as the two
things that must be filtered out before substation measurements can be used, detect both on the same
residual, and report detector performance stratified by how long the event lasted. Their
sign-recovery technique addresses exactly the non-directional metering defect described above.
Earlier variations of their methodology have run at Alliander since 2021, "fully replacing the
manual, time-consuming, process", and the authors report that the ensemble method presented in this
paper has since been adopted too; its open-data portal names the associated dataset "STORM
onderstation".

**One other group has made it their subject, one voltage level down.** [Moriano et al.
(2016)](https://doi.org/10.3390/s16010085) and [Martín et al.
(2018)](https://doi.org/10.3390/s18113947) detect systematic errors in secondary-substation
monitoring equipment by comparing each measurement against a short-term load forecast, and report a
98% hit rate. Two things limit how far that carries: the errors are *injected* rather than found in
the wild, and the fault taxonomy is calibration gain and offset drift plus outliers, not the stuck
values, false zeros and multi-month gaps that dominate NGED's telemetry.

**Three GB network innovation projects made faulty metering their subject — Electricity North West's
ATLAS, described under problem 4 above, UK Power Networks' Distribution Network Visibility, and this
network's own Time Series Data Quality.** UK Power Networks' [Distribution Network
Visibility](https://www.ofgem.gov.uk/sites/default/files/docs/2014/03/dnv_cdr_version_3.0_270214.pdf)
project (Low Carbon Networks Fund, reported 2014) checked its remote terminal units against physics
rather than against a forecast: apparent power must equal the root of real power squared plus
reactive power squared, and where it does not, something is wrong with the installation — "an error
with the physical connection of the RTU CTs or voltage connections such as direction, wiring
connection, placement, ratio or dual tail issues". They ran it over 377 units and found that "95%
were found to obey the expected logic within 15 kVA, with 5% identified as probably having
installation problems", then put the check into a daily health report that ranks units for
maintenance. Alongside it they defined six named anomaly patterns — dropouts to a fixed value,
spikes, flat-lining above zero, gaps, flat-lining at zero, and out-of-range values — which is close
to the taxonomy NGED's telemetry needs. Their caution is worth repeating: "Different users have a
different view on what is a data quality point... For Control engineers operating in real time, the
data may represent a system event. However, for Planning engineers, the data may corrupt any
statistical analysis they wish to undertake." A run of implausible values is a fault to a forecaster
and a real event to a control engineer, and only the purpose settles which.

**This network's own predecessor ran the same investigation on the same telemetry, and its findings
should temper what we expect to find.** Western Power Distribution's [Time Series Data
Quality](https://smarter.energynetworks.org/projects/nia_wpd_011/) project (Network Innovation
Allowance, reported March 2017) checked SCADA analogues for zeros, for "non-varying non-zero values,
perhaps indicating a 'stuck' or incorrectly configured sensor", and for gaps, across all four
licence areas. It found that "13.8% of all analogues in the WPD South-West licence area are only
recording 0 values. (20.7% companywide)", that the share of PowerOn data points unavailable to
planners ran from 1% in the South West to 36% in the Midlands, and — the finding most relevant to
problems 3 and 7 — that "63% of all new solar sites across the company have not had their analogues
commissioned correctly". Flexpectation should expect metering defects at that prevalence rather than
as an exception.

**Everywhere else, faulty metering appears as a data-cleaning step described in passing rather than
as a problem in its own right.** [Mendonça Severiano et al.
(2026)](https://doi.org/10.1016/j.solener.2026.114382) classify solar underperformance, but from
inverter data a distribution network operator does not receive, and Artificial Forecasting supplies
only indirect evidence that it matters — across their 729 secondary substations, data quality
mattered at least as much as the choice of model (finding 1 below).

**A public labelled dataset exists, and it is Dutch.** Knowing how often a detector is right
requires measurements labelled as faulty or not, and [Bouman et al.
(2024)](https://arxiv.org/abs/2405.16164) had 180 primary substations labelled at 15-minute
resolution — since released on Alliander's [open data
portal](https://www.liander.nl/over-ons/open-data) as "STORM onderstation", explicitly so that
others can train and validate algorithms against it. That is the one place in this review where the
evaluation data for a problem is already public.

**Where the gaps are: the fault taxonomy, a measured GB detector, and a reference series to detect
against.** The Dutch labels collapse switching events and measurement errors into a single class, so
they cannot separate a stuck meter from a network reconfiguration — which is exactly the distinction
problems 4 and 6 have to make between them — and four per cent of their timestamps are labelled as
the labeller being unsure. They describe a Dutch network, and [Bouman et al.
(2024)](https://arxiv.org/abs/2405.16164) detect on a residual against a bottom-up load estimate
NGED does not have and cannot build. None of the three GB projects above reports how often its
checks are right, and none published its labels, so there is no GB number to compare a new detector
against.

**Recovering the direction of flow from a magnitude-only meter has been attempted in GB, by this
network's predecessor, and left unfinished.** [Time Series Data
Quality](https://smarter.energynetworks.org/projects/nia_wpd_011/) set out to "first detect then
assign directions to power flows where absent". What it reports achieving is more modest than that
objective — plotting every analogue made it "clear where (for example in cases of generation) the
directional sense of analogues was incorrectly set", and correction was explored "by (for example)
flipping the direction/sense of a suitable candidate feeder" where summed currents at a transformer
and along its feeders failed to reconcile by more than a threshold. That is a manual,
engineer-triggered process feeding a rectification list, not an automatic detector, and no accuracy
is reported for it. The objective is nine years old and still open. A GB labelled set, with a
taxonomy that separates metering faults from switching, is a gap this project can close cheaply,
because the trial area is small enough to label by hand.

### 7. Disaggregating unmetered solar and wind from a substation's net flow

**In summary.** Splitting generation out of a substation's net flow has been done where the
generation is metered or its capacity is read from a register, and uncertainty and a multi-day
horizon each appear in this literature, but never together.

**The problem.** Rooftop panels and small turbines appear only as a dent in a substation's net flow.
Recovering both the half-hourly output of that unmetered generation and its installed capacity, from
the net flow alone, is what we call *disaggregation*. It is a different task from estimating how
much of a *metered* generator's capacity is available today, which is problem 3. It is a stretch
goal for the trial area and a requirement for the network-wide scale-up.

**Where demand and generation are separated at all in this literature, the generation is either
metered directly or its capacity is read from a register, never inferred from measurements.**
Artificial Forecasting models gross demand and customer export independently at primary substations,
which is more than any paper here does, but customer export is metered. SSEN TRANSITION split net
load into demand and generation, forecast the two separately and recombined them. Its rooftop solar
is not metered — but neither is its capacity inferred. SSEN gathered a list of Feed-In Tariff
installations, aggregated 5.3 MW of them to one notional generator per 11 kV feeder, and drove that
with a generic solar model, noting that "there is obviously no generation output data easily
available for those generators". Looking a capacity up in a subsidy register is the step
Flexpectation cannot take, because the register stopped being complete when the Feed-In Tariff
closed.

**The nearest peer-reviewed work stops one step short of a forecast.** [Kara et al.
(2018)](https://doi.org/10.1016/j.segan.2017.11.001) and [Li et al.
(2021)](https://doi.org/10.1109/TPWRS.2020.3035639) recover the solar signal from feeder-head and
substation measurements without forecasting it. The one benchmark we found on estimating installed
capacity, [Gouveia et al. (2026)](https://doi.org/10.1016/j.ijepes.2026.111848), compares
data-driven against model-based methods at low-voltage substations, which is a level below ours; we
read only its abstract. Most of the rest of the behind-the-meter disaggregation literature works on
individual smart meters, a level or two below a primary substation, and is excluded for that reason.

**The direct predecessor of this work is running now in GB.** [UK Power Networks'
NIA_UKPN0104](https://smarter.energynetworks.org/projects/nia_ukpn0104/) (2024–2026, £389,444),
which Open Climate Fix worked on, infers the capacity of unmetered solar sitting behind each primary
substation from half-hourly substation load and weather, then forecasts that generation. Open
Climate Fix is a partner in both projects, so Flexpectation starts from its method rather than from
scratch.

**One production system already splits unmetered wind and solar out of substation measurements, by
transferring from substations that do meter them.** [Teng et al.
(2023)](https://doi.org/10.1016/j.rser.2023.113662) train on ten Dutch substations that carry
complete renewable metering, then predict solar and wind power separately at substations with none,
from weather, geospatial position and each site's known renewable capacity, at 15-minute resolution
— a root-mean-square error of 0.07 against 0.70 for a conventional transfer-learning model, both
normalised to each facility's installed capacity, so 0.07 reads as 7%; we read their abstract rather
than the full paper. The technique ships as the energy-splitting component of
[OpenSTEF](https://lfenergy.org/projects/openstef/), Alliander's open-source forecasting stack,
which is in live operation at hundreds of grid locations; OpenSTEF's own description of that
component names the technique this paper introduces.

**GB already has an operational forecast of unmetered generation, at national scale.** NESO
publishes [embedded wind and solar
forecasts](https://www.neso.energy/data-portal/embedded-wind-and-solar-forecasts) half-hourly, from
within-day to 14 days ahead, updated hourly — the same resolution and horizon Flexpectation
delivers. "Embedded" means precisely the generation this problem is about: wind and solar sitting on
the distribution network with no transmission metering, which NESO's own field definition calls
"invisible" to the transmission system. NESO also publishes its best view of installed embedded
capacity, compiled from public sources rather than inferred from measurements, and warns that it "is
not the definitive view". The forecast is a single number per half-hour, with no uncertainty
attached, and it covers GB as one region rather than substation by substation.

**Estimating unmetered *wind* capacity from measurements has been done at feeder level.** [Nikzad
and Venkatesh (2024)](https://doi.org/10.1109/OAJPE.2024.3413606) estimate the aggregate capacity of
connected distributed generation — wind and solar — from a North American utility's feeder
measurements, reporting 97.53% accuracy on capacity and 93.70% on feeder flow, and [extend the
method in 2026](https://doi.org/10.1109/TPWRD.2025.3631805) with separate wind and solar models. We
read the abstracts rather than the full papers, and could not confirm from them whether wind and
solar capacity are separated or estimated as one aggregate — which is the part that matters most to
us.

**Uncertainty and the horizon both exist in this literature, but never together.** [Wang et al.
(2018)](https://doi.org/10.1109/TPWRS.2017.2762599) run the whole pipeline this problem describes —
estimate behind-the-meter photovoltaic capacity, decompose net load into generation, demand and a
residual, forecast each, then recombine them with a copula into a probabilistic net-load forecast —
but at ISO New England scale and day-ahead. [Zhang et al.
(2022)](https://doi.org/10.1016/j.engappai.2022.104707) do probabilistic disaggregation at grid
supply point and feeder level with a multi-quantile recurrent network, scored on reliability and
sharpness. NESO covers the 14-day horizon but deterministically. [Erdener et al.
(2022)](https://doi.org/10.1016/j.rser.2022.112224) survey the field. We read the abstracts of all
four.

**Where the gaps are: doing it without a metered training set, inferring the capacity rather than
being told it, and putting uncertainty and a multi-day horizon in the same forecast at substation
level.** [Teng et al. (2023)](https://doi.org/10.1016/j.rser.2023.113662) need a population of
fully-metered substations to transfer from, and are given the existence and capacity of each
renewable facility rather than inferring it — whereas inferring that capacity is half of what NGED
needs. Their output is a near-real-time estimate rather than a forecast. SSEN TRANSITION does
combine a weather ensemble, a horizon beyond day-ahead and the level of an individual primary
substation — but it is told the unmetered capacity rather than inferring it, which is the half of
the problem NGED needs solved. Nothing we found infers that capacity from measurements and then
carries it into a probabilistic forecast at substation level.

### 8. Disaggregating other distributed energy resources: heat pumps, electric-vehicle chargers and batteries

**In summary.** This is the largest gap in the review and the largest deliberate omission from our
search: charger forecasts only beat a naive benchmark above about a hundred charge points, heat-pump
diversity is untested in the cold weather that matters, and no diversity factor helps for domestic
batteries at all.

**The problem.** Heat pumps, electric-vehicle chargers and price-sensitive domestic batteries change
the shape of a substation's load in ways a model trained on history cannot anticipate, because the
number of them behind any given substation is growing quickly. The stretch goal is to disaggregate
and forecast them separately rather than letting them sit inside net demand.

**Detecting these loads and forecasting them are separately hard, and not in the order we
expected.** Northern Powergrid's [smart-meter detection
trial](https://smarter.energynetworks.org/projects/npg_nia_-49/), on 1,500 monitored premises, found
that "EV identification at premises level was found to be relatively straightforward" and that
although "aggregation does mask some signals, EV usage is still clearly identifiable at feeder and
substation level", while "the detection of ASHP [air-source heat pumps] is frustrated by the low
levels of adoption". So the spiky, synchronised charging that makes electric vehicles hard to
*forecast* is what makes them easy to *detect* in aggregate; heat pumps are the reverse.

**Errors across many chargers cancel rather than compound, and the measurement is NGED's own.** The
[Electric Nation
trial](https://eatechnology.com/media/girhcnsc/electric-nation-customer-trial-report.pdf) — run by
this network under its former name, with 673 participants and around 137,000 charging events — fits
the demand of a group of chargers as `Group Demand = N·P + Q√N`, where P is the mean demand per
charger and Q the deviation. The mean scales with the number of chargers and the deviation only with
its square root, so relative uncertainty falls as more chargers are added, exactly as it does for
any other diversified domestic load. [Bollerslev et al.
(2022)](https://doi.org/10.1109/TTE.2021.3088275) fit the same decay to Danish driving and plug-in
behaviour and measure the exponent at 0.43 to 0.46 across battery sizes and charger ratings, against
the 0.5 that complete independence would give, which puts a number on how much synchronisation there
actually is.

**What makes electric-vehicle charging the harder network problem is when it lands, and that an
automated tariff can re-synchronise a population that had diversified.** In Electric Nation's third
trial, with a time-of-use tariff, the share of charging events starting in the 22:00 hour rose from
5.8% without the tariff to 24.7% with it — and to 37.6% among participants using the smart-charging
app, against 5.5% for those on the same tariff who did not use the app. The synchronisation came
from the scheduler, not from the tariff, and the resulting peak was higher than any previously
observed in the trial. Heat pumps, by contrast, peak in the morning rather than at the network's
evening peak.

**Heat pumps diversify in an average winter, but whether that diversity survives the cold weather
that actually matters is untested.** [Love et al.
(2017)](https://doi.org/10.1016/j.apenergy.2017.07.026) measured around 700 GB domestic heat pumps
and found demand per heat pump falling from 4.0 kW for a single unit to 1.7 kW once 275 are
aggregated, with the spread between samples falling from 1.5 kW to 0.1 kW. But a heat pump sized
small relative to a house's heat demand runs flat out for hours in cold weather, and GB design
guidance concedes that whether diversity survives colder-than-average winters needs further research
— which is precisely the condition under which a substation approaches its limit.

**No diversity factor helps for domestic batteries, and the industry agrees.** Northern Powergrid's
[code of practice for the economic development of the low-voltage
system](https://www.northernpowergrid.com/sites/default/files/assets/IMP001911_0.pdf) fits diversity
curves to measured trial data for general domestic load, heat pumps and chargers alike, and then
states that diversity "should not be applied when considering a BESS device" — a battery energy
storage system — a diversity factor of exactly one. What academic work exists separates a battery
from a household's net flow only by assuming a known control model, which is the one thing that
cannot be assumed when the point is that two identical batteries dispatch in opposite directions.

**This remains the largest deliberate omission in the review.** Our search covered substation and
generation forecasting, not electrification, and the paragraphs above are what a targeted follow-up
search surfaced rather than a proper review. The volume of work is easy to demonstrate: of the 305
papers accepted for CIRED's Brussels workshop of June 2026, 29 have a title naming electric
vehicles, chargers, heat pumps or batteries — more than the 23 whose titles name forecasting or
prediction at all.

**Where the gaps are: forecast skill at substation aggregation, and the tariff-driven peak.** The
one direct measurement of charging forecast skill against aggregation that we found is [Ostermann
and Haug (2024)](https://doi.org/10.1186/s42162-024-00319-1), who forecast "over 350,000 charging
processes at more than 500 locations across Germany" a day ahead at 15-minute resolution, scoring
the distribution with the pinball and interval scores against a naive benchmark. Aggregation is what
decides whether the forecast is worth having: at the level of a single site and of a postcode
"almost all models have values above 1 for the MASE and nRMSE, which means that the benchmark model
is better in some cases", and of their five example sites only the one with over 100 charging points
was "significantly better than those of the naive model". Pooled across the whole portfolio of more
than 500 sites, their two best models — random forest and Ada boosting — reach a normalised
root-mean-square error of 0.41 and 0.42. The lesson for NGED is that charger forecasting starts to
beat the naive benchmark somewhere above a hundred charge points, which is a larger aggregation than
most single sites and a smaller one than a primary substation. Nothing we found forecasts an
aggregate of heat pumps, chargers and batteries behind a GB primary substation, states its own
uncertainty, and is scored against the evening peak that the network actually cares about. Nothing
we found tests whether the re-synchronised peak an automated tariff creates, described above,
survives at the aggregation a primary substation carries. Reading the electrification literature
properly is the first deliverable on this strand, before any model.

## Six findings that recur across the studies we read

Six findings recur across the studies reviewed above. These are findings about this literature, not
laws of nature: each is what several teams measured on their own networks, and a network that
differs from theirs may well behave differently.

### 1. In every load-forecasting study we read, sophisticated models beat simple models by a much smaller margin than the effort put into them would suggest

[Pinheiro et al. (2023)](https://doi.org/10.1016/j.apenergy.2022.120493), running a live system
across 96,989 Portuguese secondary substations, tuned a gradient-boosted tree by exhaustive grid
search. At system level, the gradient-boosted tree scored 199 MW root-mean-square error and the
generalised additive model scored 191 MW, so the gradient-boosted tree was 4% worse than the simpler
model. [Pinheiro et al. (2023)](https://doi.org/10.1016/j.apenergy.2022.120493) concluded there was
no accuracy gain to be had and rejected the gradient-boosted tree on the cost of tuning it and on
the loss of interpretability, keeping the generalised additive model. Artificial Forecasting also
found that gradient-boosted trees did not beat a simpler model, when forecasting customer export at
primary substations. Compared against the Bayesian ridge regression they went on to adopt, boosted
trees "helped some substations but harmed others", so they kept the Bayesian ridge regression as
their default. Northern Powergrid's deliverable gives no magnitudes and no significance test in
either direction.

When Artificial Forecasting tested a neural network against a four-week-average baseline at 729
secondary substations, the neural network lost on five of six metrics at the 24 substations with the
worst data quality. The margin was small, and data quality and the choice of metric mattered at
least as much as model complexity.

### 2. In every study that forecast more than one voltage level, accuracy got worse further down the network

[Hertel et al. (2026)](https://arxiv.org/abs/2607.15705) ran the same models against a day-type
persistence baseline on three datasets — a German transmission control area, 200 German low-voltage
feeders and 287 individual Portuguese clients — and the margin over that baseline shrank from 59.6%
to 42.3% to 23.3% as aggregation fell. What shrank is the headroom above a naive rule rather than
the accuracy itself, which is the more useful reading: their own gloss is that it is easier to beat
a simple approach on highly aggregated data than on volatile feeder- and client-level data. [Pfeifer
et al. (2021)](https://doi.org/10.1049/icp.2021.2177) measured the same thing separately for wind
power, solar power and load across a medium-voltage grid region, and report that forecasts get worse
both at lower levels of aggregation and at longer horizons; we read their abstract rather than the
full paper. The model did not get worse; the problem got harder. That pattern is probably not a fact
about forecasting so much as a fact about averaging: a grid supply point aggregates hundreds of
thousands of customers, whose individual quirks cancel out, while a single feeder aggregates a few
dozen, whose quirks do not. Predicting the temperature of a kilogram of air is easier than
predicting the motion of each molecule in it, and for the same reason.

**Rising error does not mean falling usefulness.** A forecast at a primary substation may carry a
larger percentage error than a forecast at a grid supply point and still support flexibility
procurement just as well, because what NGED needs from the forecast is a reliable answer to "will
this substation exceed its firm capacity?", and that question can be answered well even when the
load itself is hard to predict precisely. Whether decision-usefulness really is flat across voltage
levels is something this project can measure, and we intend to.

### 3. In the one study reporting results substation by substation at scale, a trained model failed to beat a naive "same time yesterday" rule at a substantial minority of substations

[Pinheiro et al. (2023)](https://doi.org/10.1016/j.apenergy.2022.120493) found that their model beat
a "same time yesterday" forecast at 83–87% of network-owned secondary substations but at only 66–70%
of customer-owned ones. Those customer-owned sites serve a single customer — one large building or
one industrial process — where load follows decisions no weather model can see. We do not know that
NGED's primary substations will behave the same way, and they may not, because a primary substation
aggregates far more customers than a Portuguese secondary substation does.

### 4. In the studies we read, standard accuracy measures rewarded flat forecasts that would be of little use for flexibility procurement

A forecast that predicts the right peak at the wrong time is penalised twice by mean absolute error
— once for the peak it predicted that did not happen, and once for the peak that did happen and the
forecast missed. A flat, featureless forecast avoids both penalties. Meteorologists named that
effect the double penalty decades ago, and their conclusion transfers: a score that forgives a peak
predicted an hour late is generally no longer a **proper scoring rule** — a score a forecaster
cannot improve by publishing anything other than what they genuinely believe. A peak-aware score
therefore belongs alongside a proper score, not instead of one. Two teams independently concluded
that mean absolute error was the wrong measure for peaks. [Pinheiro et al.
(2023)](https://doi.org/10.1016/j.apenergy.2022.120493) adopted the adjusted error of [Haben et al.
(2014)](https://doi.org/10.1016/j.ijforecast.2013.08.002), a peak-aware measure, for exactly this
reason. Artificial Forecasting built a metric over the top 10% of demand values and made it the
primary measure for comparing their models, reporting it both against actual demand and normalised
to transformer rating.

### 5. A forecast can state its own uncertainty badly, and a single accuracy score will not reveal that the uncertainty is wrong

[Kaas et al. (2026)](https://arxiv.org/abs/2607.01966) scored models on 200 German low-voltage
feeders with an overload-decision metric evaluated at each model's 95th percentile. The two models
that topped that metric on the consumer side turned out to have 90% ranges containing the true value
only 62% and 58% of the time across the series as a whole, and under half the time at the peaks
themselves. In the results of [Kaas et al. (2026)](https://arxiv.org/abs/2607.01966), a model that
understates its uncertainty raises fewer false alarms, so it scores well on a threshold-crossing
test while being exactly the model an operator should not trust near a capacity limit. [Kaas et al.
(2026)](https://arxiv.org/abs/2607.01966) supply their own counter-example: ranked on average error
rather than on the overload metric, the winning model was also the most honest about its own
uncertainty, with reality falling inside its stated 90% range 89.75% of the time.

### 6. Weather forecasts are barely used at low voltage, and weather ensembles almost never

Of the 221 low-voltage forecasting papers [Haben et al. (2021)](https://arxiv.org/abs/2106.00006)
reviewed up to 2020, three used a weather *forecast* and none used an *ensemble* of weather
forecasts. [Pinheiro et al. (2023)](https://doi.org/10.1016/j.apenergy.2022.120493), published after
that review closed, is a fourth paper using a real weather forecast — but its inputs are single
point forecasts rather than an ensemble. [Pinheiro et al.
(2023)](https://doi.org/10.1016/j.apenergy.2022.120493) therefore overturns the first half of that
finding but not the second: even the largest deployment in this review used no weather ensemble.
Artificial Forecasting's published secondary-substation results use no weather at all, because the
weather archive available to them reached only 16 days ahead while their forecasts were month-ahead,
and a substitute built from the previous year's observations at the same time of year had, in their
words, "trivial or net negative" effects on every type of model they tried.

### Three findings that cut against this project's plan

Three results in this literature point against Flexpectation's plan, and we intend to test all three
rather than route around them.

#### Finer-grained weather data has not always paid

[Browell and Fasiolo (2021)](https://arxiv.org/abs/2103.10335) added spatial statistics derived from
gridded numerical weather prediction to their model of 14 grid supply point groups in GB. Those
spatial statistics helped significantly in two of the 14 regions, hurt significantly in three, and
made no measurable difference in the remaining nine. They put that down to their own model rather
than to the data, writing that another method might yet extract value from it by building different
features. Weather itself was worth a great deal to them — adding wind and irradiance cut their
pinball loss by 40% overall, and by 60% in North Scotland against 10% in London — so the question is
not whether weather matters but whether *finer* weather does. Artificial Forecasting obtained
postcode-level weather forecasts for two wind-connected primary substations after their
wind-connected models had performed poorly, and reported that the postcode-level forecasts "did not
notably improve model performance", naming better weather data as a next step. What both results say
is that finer weather data does not help everywhere, so the interesting question is *where* it
helps. That question is answerable, and answering it is part of this project: we expect finer
weather data to matter most where a substation's load is dominated by weather-driven generation or
heating, which is where NGED most needs the forecast to be right.

#### Weather has bought less than expected at low voltage in the past

[Haben et al. (2019)](https://doi.org/10.1016/j.ijforecast.2018.10.007) tested 100 real low-voltage
feeders with both forecast and observed temperature, and found that temperature had no effect on
forecast accuracy, or a negative one. [Haben et al.
(2019)](https://doi.org/10.1016/j.ijforecast.2018.10.007) used data collected in the early 2010s,
and we expect how much weather matters at a substation to be changing quickly, because the thing
that makes a substation weather-dependent is embedded solar generation and heat pumps, and there are
far more of both on the network now than there were then. A primary substation that was almost
weather-independent ten years ago may be strongly weather-dependent today. That is a prediction,
though, not a measurement — and the Scottish primary-substation sensitivities of [Fox et al.
(2018)](https://doi.org/10.34890/134), measured on ten years of data ending in the mid-2010s and
described under "What GB networks have already built" below, say weather was already moving primary
substation demand well before the mid-2010s. Measuring how much weather now explains at NGED's
primary substations is one of the more useful things this project can report.

#### A model trained on none of NGED's data may match a model trained on all of it

[Kaas et al. (2026)](https://arxiv.org/abs/2607.01966) tested Chronos-2, a general-purpose
time-series model that had never seen their data, against models trained specifically on those 200
feeders. Chronos-2 beat every purpose-trained competitor on mean absolute error, 3.8 kW against 4.2
kW. If heavily engineered models do not clearly beat an off-the-shelf model given none of the target
network's data, that is important information about the value of any such experimental programme.

### Two explanations for why sophisticated models beat simple ones by so little, and how Flexpectation can tell them apart

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

**We hold explanation two loosely, and explanation one may well be the right one.** Flexpectation is
not resourced like a large industrial research laboratory. What it is resourced to do is run many
experiments cheaply against one fixed benchmark, which is the part that matters for telling the two
explanations apart. If explanation one is right, sustained experimentation will converge quickly on
a small improvement over a naive forecast and then stop improving, however many further experiments
we run — and we will report that plainly. If explanation two is right, improvements should keep
arriving well past the point at which a smaller effort would have concluded there were none left to
find.

**Either answer is worth publishing, and explanation two would be worth more to the industry than to
this project alone.** That is the shape of what Flexpectation offers beyond its own forecasts: not a
claim to have found the state of the art, but a run of comparable experiments on one fixed
benchmark, published as they go, so that the next team does not have to start where we started.

## What GB networks have already built

Nine concurrent or recent GB network-innovation projects bear on this work, and between them they
have built more of what Flexpectation needs than the academic literature has. Five are summarised
here. The sixth, Northern Powergrid's Artificial Forecasting, gets its own section below because
this review leans on it more heavily than on any other. The remaining three — Electricity North
West's ATLAS, UK Power Networks' Distribution Network Visibility and this network's own Time Series
Data Quality — are described under problems 4 and 6 instead, because what they contribute is about
finding bad and switched measurements rather than about forecasting.

**Scottish and Southern Electricity Networks' TRANSITION** (Network Innovation Competition,
Oxfordshire, reported 2021) is the closest precedent for Flexpectation's method. It forecast net
load at 13 primary substations, their bulk supply points and their 33 kV and 11 kV feeders, from 30
minutes to 10 days ahead. SSEN TRANSITION drew its uncertainty from the 40 members of the German
weather service's ICON-EU ensemble. It split each substation's net load — demand minus whatever
generation behind that substation happened to produce — into demand and generation, forecast the two
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
Fund, Alpha 2025–2026) combines probabilistic forecasting with simulation to model how the
distribution connections queue — around 180 GW and growing — will load the network, from primary
substations up to the grid supply point. FastTrack puts a probability on how much of that queue
turns into real load and how it behaves, which is a planning question rather than the operational
one Flexpectation asks. **[SP Energy Networks'
Predict4Resilience](https://smarter.energynetworks.org/projects/10061710/)** drives a probability
distribution of network faults per district from an ensemble weather forecast, up to seven days
ahead, in a tool built with control-room engineers and now being trialled live — the GB precedent
for putting ensemble-derived distributions in front of network operators. SP Energy Networks has
also published at Flexpectation's own voltage level: [Fox et al.
(2018)](https://doi.org/10.34890/134) ran a numerical weather prediction model over Scotland at 1 km
resolution for ten years, mapped it onto each primary substation weighted by customer density, and
used it to separate the effect of weather on peak demand from the effect of everything else — 13
substations in the proof of concept, almost 400 in production. Demand fell by between 1.4% and 4.8%
for each degree Celsius of effective temperature, differing substation by substation with the mix of
customers behind it — GB primary substation demand being heating-dominated. Fox et al.'s method
corrects history for planning rather than forecasting forward, but it is the GB precedent for
putting gridded weather onto individual primary substations.

Two deployments outside GB belong alongside these.

**The Dutch operator Alliander runs [OpenSTEF](https://lfenergy.org/projects/openstef/)**, an
open-source forecasting stack under the Linux Foundation's LF Energy umbrella, in live operation
across thousands of grid connection points to 48 hours ahead. It is the only production system we
found that separates unmetered wind and solar out of a substation's measurements, by the method of
[Teng et al. (2023)](https://doi.org/10.1016/j.rser.2023.113662) described under problem 7, and
being open source it is the one whose method can be read rather than inferred from a deliverable.

**The second is far larger than any project here.** Enedis, the French distribution network
operator, has forecast consumption and generation at all 2,300 of its high-voltage-to-medium-voltage
substations since 2015, and is now extending that to a finer geographic grid ([Cordier et al.
(2024)](https://doi.org/10.1049/icp.2024.2058), whose abstract we read rather than the full paper).
A high-voltage-to-medium-voltage substation in France is broadly the level of a GB primary
substation. Forecasting operationally at the scale Flexpectation reaches in 2027 is therefore a
decade old somewhere else, which is reassuring about the engineering and says nothing about the
forecast quality, because the paper reports none.

### Northern Powergrid's Artificial Forecasting is further ahead, and sets the bar

**One concurrent project matters more than any paper here.** Artificial Forecasting is an Ofgem
Strategic Innovation Fund programme, with about £3.9m of grant across its three phases, run by
Northern Powergrid with Faculty, EV.energy and Oaktree Power, the final Beta phase running to
February 2027. Its deliverables are publicly available on the Energy Networks Association's Smarter
Networks Portal, though the [Beta
deliverables](https://smarter.energynetworks.org/projects/10145998/) sit under a separate project
registration from the Alpha ones linked above. It does much of what Flexpectation does at primary
substations, it also covers secondary substations, which Flexpectation does not, and at the time of
writing it is further ahead than Flexpectation.

**Its argument is in the peer-reviewed literature as well as in its deliverables.** [Wade et al.
(2024)](https://doi.org/10.1049/icp.2024.2102), by authors at Northern Powergrid and Faculty, put it
to CIRED that annual, assumptions-driven models of load at primary and secondary substations will
not support flexibility procurement, and that monthly, weekly and daily operational forecasts are
needed instead; we read its abstract rather than the full paper.

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
contracted flexibility. Those are the figures from the Beta application. The project's own position
is more guarded than the numbers suggest: it reports early Beta evidence, from one winter
procurement cycle, supporting the performance assumptions behind the value case, which "remains
appropriate, subject to further validation".

**Artificial Forecasting is independent evidence that short-term substation forecasting is
operationally useful**, that networks will change their procurement process around it, and that a
benefits case has been made and accepted. Because it is public, operational and benchmarked against
a real incumbent method, it also sets the clearest available bar for what "working" looks like.
Northern Powergrid's programme rightly prioritised getting a service into live operational use;
Flexpectation takes on research questions that priority left for later. Artificial Forecasting's
core intellectual property is to be made available royalty-free to other GB networks, and we would
rather build on it than rebuild it.

## Set against this literature, what we plan is ambitious, and here is why we think it can be done

**Measured against the studies above, the plan goes beyond this literature in five directions at
once.** No study in this review drives a substation forecast from a weather ensemble across a 14-day
horizon. None models the upper tail explicitly at substation level; the one study that models it
explicitly at all works on regions far larger than a substation. None puts unmetered generation
inside a probabilistic forecast at substation level over a multi-day horizon, though unmetered
generation, probabilistic forecasting at substation level and a multi-day horizon each exist on
their own. None tracks the available capacity of a mixed fleet of solar, wind and dispatchable
generators at one distribution network, or measures whether doing so improves the forecast. None
turns switching-contaminated history into a useful input rather than deleting it, rewriting it, or
absorbing the cost of leaving it in. Flexpectation attempts all eight problems above, across four
families of model:

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
them as a pipeline. [Dantas and Browell (2026)](https://doi.org/10.1002/we.70079) estimate available
capacity, then normalise by it, then forecast. Artificial Forecasting rescales step-change blocks in
data preparation, then forecasts. SSEN TRANSITION splits net load into demand and generation,
forecasts each, then recombines. [Huyghues-Beaufond et al.
(2020)](https://doi.org/10.1016/j.apenergy.2019.114405) clean outliers segment by segment before
training begins. In every case one stage's output is frozen before the next stage sees it, so an
error made early cannot be corrected later and the forecast error never gets to tell the capacity
estimator it was wrong.

**So the question we want to answer is whether one model that estimates capacity, switching state
and demand together beats that pipeline.** NGED's specification leaves room for it, asking that
these phenomena be handled rather than that they be handled explicitly. The one published result
that bears on the question points the joint way: [de Vilmarest et al.
(2024)](https://doi.org/10.1109/TPWRS.2023.3310280), described under problem 3, removed the embedded
wind and solar capacities from their model of GB regional net load, and the adaptive version got
0.4% *better*, absorbing into its own coefficients what the explicit capacity figure had been
supplying, while the static version got more than 10% worse. That is one result, on regions far
larger than a substation, for one phenomenon out of several — and there are good reasons to doubt it
generalises. A gradient-boosted tree is structurally poor at doing the subtraction that a two-stage
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
has been demonstrated at a real network operator by [Bouman et al.
(2024)](https://arxiv.org/abs/2405.16164). Estimating a wind farm's available capacity from its
meter plus reanalysis wind speed has been published, with code, by [Dantas and Browell
(2026)](https://doi.org/10.1002/we.70079). Inferring unmetered solar behind a primary substation is
being built now by UK Power Networks with Open Climate Fix, who are also a partner in Flexpectation.
For most of the eight, the work is to extend a published method to NGED's data rather than to invent
one, and Artificial Forecasting's core intellectual property is to be made available royalty-free to
other GB networks.

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

**Differentiable physics applied to substation demand forecasting produced no strong result** in our
search, though the ingredients exist separately. There is substantial work on physics-informed
neural networks for power systems, including models that map weather to solar output. On the demand
side the physics that matters is the thermal response of a few thousand buildings rather than of a
panel or a turbine, and models that build that response in are a field of their own: [Di Natale et
al. (2022)](https://doi.org/10.1016/j.apenergy.2022.119806) constrain a neural network so that heat
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

**The bulk of the low-voltage forecasting literature is covered second-hand**, through the [Haben et
al. (2021)](https://arxiv.org/abs/2106.00006) review of 221 papers rather than read individually,
and we have not systematically covered low-voltage work published since it closed in 2020. The same
lead author's open-access book-length treatment of 2023 is the better entry point for anyone
following this up.

**CIRED, the International Conference on Electricity Distribution, is the venue this audience is
most likely to read — it is where European distribution network operators publish their own
operational work, so CIRED is where a claim of ours is most likely to be contradicted.** We
therefore searched it in full: the titles and abstracts of every paper in the CIRED main conferences
and workshops of 2017 and 2020 to 2025, about 3,600 of them; the 2018 and 2019 proceedings, which
are not indexed, by keyword against their open full-text archive; and the 305 papers accepted for
the Brussels workshop of June 2026 by title, those proceedings not yet being published. Nothing
there contradicts what this review reports missing, and the absences are worth stating, because
CIRED is where the counter-example would have been. No CIRED paper drives a load or generation
forecast from a weather ensemble. None produces an operational load or generation forecast in the
days-to-fortnight band that NGED acts on: the long-horizon load forecasts in the proceedings are
annual planning forecasts, and the only 14-day forecast predicts feeder faults rather than load.
Fourteen forecast probabilistically at all, of which one is at substation scale — [Mesarcik et al.
(2025)](https://doi.org/10.1049/icp.2025.1968), day-ahead, on ten years of measurements from 312
Dutch substations. Nothing scores the upper tail, nothing keeps switching-contaminated history
usable, and nothing estimates how much of a generator's capacity is available. The closest paper to
our own problem, [Ruhhütl et al. (2023)](https://doi.org/10.1049/icp.2023.0476), appears in the
table under problem 1 above; its result is a further instance of findings 1 and 2. We read only the
abstracts of it and of [Mesarcik et al. (2025)](https://doi.org/10.1049/icp.2025.1968), because both
full texts are paywalled. The Brussels titles of June 2026 change none of this: 23 of the 305 name
forecasting or prediction, none names an ensemble, and the only short-horizon forecast named is
day-ahead; the three others that name a horizon at all name long-term planning. Two of the 23 apply
time-series foundation models, so the possibility that a model given none of a network's own data
can compete is being tested in this venue too.

## Publishing results that others can compare against

**Energy forecasting's own senior figures say that published results in the field cannot be compared
with each other.** [Hong et al. (2020)](https://doi.org/10.1109/OAJPE.2020.3029979), a review
written by six of the field's most senior figures, concludes that "most papers can never be
replicated, because the data have never been published". [Hong et al.
(2020)](https://doi.org/10.1109/OAJPE.2020.3029979) add that authors sometimes pick the error
measure that favours their own method, that significance tests are seldom run when the differences
between models are small, and that many papers compare a new model only against models "within the
immediate family". [Tawn and Browell (2022)](https://doi.org/10.1016/j.rser.2021.111758) found
eleven wind and solar papers that compared a new model only against other models of the same type.

**Incomparable results are what this review ran into at every one of the eight problems.** Even the
eight studies in the one table above, all forecasting electricity demand somewhere on a network,
differ in target, level, horizon and weather assumption in nearly every row, so almost none of them
can be compared directly with any other. Two papers a fortnight apart on the same 200 feeders name
different winners. The other seven problems get no table at all, because there was too little to put
in one.

**[Hong et al. (2020)](https://doi.org/10.1109/OAJPE.2020.3029979) name two remedies: publishing the
underlying data, and running competitions in which every team forecasts the same dataset.** This
project is well placed to help with both, and others have started already: HEFTCom and Energy-Arena
both compare methods on common data with a common metric, and Energy-Arena keeps a live public
leaderboard. A third approach — recovering a ranking from the published literature after the fact —
shows what the alternative costs: [Nguyen and Müsgens (2026)](https://doi.org/10.1063/5.0300682) did
recover a defensible ranking of solar forecasting methods from the published literature, by
screening 1,447 studies and hand-extracting 4,687 skill scores from those that reported one, then
statistically removing the effect of ten other factors. Their finding is that ensemble-hybrid models
improve on time-series models by 7 to 27 percentage points of skill score, while many advanced
machine-learning methods gave inconsistent gains. A comparison can therefore be dug out of this
literature, but only at that price, and nobody does it routinely. Publishing comparable results in
the first place is much cheaper.

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
low-voltage papers reviewed used any open dataset at all. Alongside that telemetry we will publish
the evaluation protocol, the metric definitions and the code that computes them. Artificial
Forecasting is moving the same way, with substation-level historical forecasts and model-performance
metrics designed into its Open Data Portal release, and a shared evaluation protocol between two GB
networks would be worth more than either alone.

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
  solar sites, because the battery, the gas generator and the biofuel plant are dispatched on market
  signals no weather forecast contains.
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
