# The current state of the art in energy forecasting

Before we discuss the literature, there is a very important caveat to admit up-front: In 2026, no honest narrative review of the energy forecasting literature can claim to reveal the canonical "state of the art"! That is because (almost) all energy forecasting papers measure performance in different ways, against different datasets. It's like an international football tournament where every team plays by different rules, with different size goals.

Energy forecasting researchers have done great work over the years. But, unfortunately, the literature does not tell us how those approaches compare against each other, especially in messy "real world" energy forecasting scenarios. This isn't anyone's fault; it's a systemic failure. And the industry is already aware of this problem, and people are trying to fix it. But, at the time of writing, the literature cannot yet tell us the current state of the art solutions for the challenges that NGED cares about.

Having said that, there have been some valiant attempts to compare multiple forecasting approaches against the same dataset. But none of these attempts directly address the main challenges relevant to Flexpectation. Before we discuss those attempts, we must emphasise two reasons for optimism:

Firstly, whilst we might not know exactly which algorithms provide the best forecasting performance,
we do know how to go about finding out. There's no magic. Machine learning is an empirical science,
and progress in it comes largely from testing many ideas under identical conditions and measuring
carefully — because most ideas fail. John Jumper, who shared the 2024 Nobel Prize in Chemistry for
his work on AlphaFold, puts the share of research ideas that fail at around 90%, and treats that
rate as an ordinary and necessary feature of doing research rather than as evidence of doing it
badly ([Nobel Week interview](https://youtu.be/nNM1QdmFwIs?t=852), 6 December 2024, from 14:12). If
roughly one idea in ten survives contact with the data, ten attempts is simply the price of one
result. So our task is to run hundreds of ML experiments, and then measure performance against the
same dataset, using the same performance metrics.

Secondly - and perhaps most importantly - the fact that the industry doesn't yet know the state of the art is a huge opportunity for the Flexpectation project: We are in a very privileged position where we can try hundreds of ideas, and test the best ideas in the real world. We have a fantastic opportunity to make a significant contribution to the energy forecasting industry by publishing our "leaderboards of ML experiments", and hence help the industry as a whole to better understand how multiple approaches perform.

## What we read

This review cites 98 sources. We read most of the ones an argument rests on in full; the rest were
available to us only as an abstract, a preprint, or part of a paper, and wherever a claim rests on a
partial read we say so at the point the claim is made. We also read the published deliverables of
twelve network-innovation projects in GB. The selection was deliberate rather than systematic: a
paper earned its place by bearing on a decision Flexpectation actually faces and by changing
something we believed. Papers may be missing for no better reason than that we did not find them,
and the section "What this review excluded, and why" lists what we knowingly left out. Every
statement below that we found no published work on something is a statement about our search rather
than about the field: if you know of work that fills one of these gaps, we would rather cite it than
repeat it, and we will correct this review. A further group of papers is cited once each, for one
specific result, rather than reviewed.

One concurrent project is cited more than any paper: Northern Powergrid's Artificial Forecasting, an
Ofgem Strategic Innovation Fund programme whose Alpha and Beta deliverables are both public, and
which has its own section below. Three further sources carry findings rather than comparable scores,
and are drawn on throughout. [Haben et al. (2021)](https://arxiv.org/abs/2106.00006) reviewed a
final list of 221 low-voltage forecasting papers published to 2020, noting that the number they
actually read is slightly smaller. [Shukla and Hong (2024)](https://doi.org/10.1049/stg2.12162)
reports the BigDEAL Challenge 2022, a competition themed on forecasting the *timing* of peak demand
rather than its size, which drew 78 teams from 27 countries and published its data alongside the
paper. [Energy-Arena](https://arxiv.org/abs/2604.24705) keeps a permanent leaderboard rather than
closing after a fixed period — the paper describes a platform that at the time of writing targeted
deterministic day-ahead tasks, and the running platform today carries 24 challenges across prices,
load, wind, and solar — eight scored as point forecasts, eight as quantiles, and eight as ensembles.

Almost every number in this review depends on where in the network it was measured, so here is the
voltage ladder of a distribution network, from the top down:

- **Grid supply point** — where the distribution network meets the transmission system, 400 kV or
  275 kV down to 132 kV. Hundreds of thousands of customers sit below one.
- **Bulk supply point** — 132 kV down to 33 kV or 66 kV. Tens of thousands of customers.
- **Primary substation** — 33 kV or 66 kV down to 11 kV. A few thousand customers.
- **Secondary substation** — 11 kV down to 400 V. Tens to a few hundred customers.
- **Feeder and individual customer** — the bottom of the ladder, at 400 V.

NGED owns 52 grid supply points, 271 bulk supply points, and 1,161 primary substations. The 32
series of the trial area are 16 of those primary substations, two grid supply points, two bulk
supply points, and the 12 metered generators described under challenge 2 below. **Flexpectation
forecasts no secondary substations**, neither in the trial area nor in the network-wide scale-up,
though several of the studies below do. GB is separately divided into 14 *grid supply point groups*,
each a whole distribution region containing many grid supply points, and several studies below
forecast those regions, which are far larger than any single substation.

## How to read the numbers in this review

**Two kinds of published number transfer to a different network, and the rest do not.** A ratio
against a baseline transfers, but only if the paper says what the baseline was and which substations
it was averaged over. Those baselines differ far more than the prose in most papers suggests —
yesterday's value at the same time, the average of the last four weeks, a day-type persistence rule,
and the long-run seasonal average all appear among the studies reviewed here, and a percentage gain
against one baseline is not a percentage gain against another. A skill score — how much less error a
forecast has than a stated benchmark, as a percentage — needs its benchmark named for the same
reason. Where the score is a probabilistic one computed from an ensemble, it depends on how many
members produced it: the ranked probability skill score is biased downwards for small ensembles, and
almost none of the papers we read state their ensemble size.

**Errors normalised by something physical also transfer.** An error expressed as a fraction of a
substation's firm capacity or transformer rating comes far closer to meaning the same thing at every
substation than an error expressed as a fraction of the load that happened to occur. It is not
exact, because a rating is itself a convention standing in for a limit that moves with air
temperature, wind, and the duration of the overload, so a paper that normalises this way should say
which rating it used. An absolute error in kilowatts or megawatts tells NGED nothing on its own,
because it depends entirely on how big the substation was, and none of the absolute figures below
should be read as a target for this project.

**Whether a study used the weather forecast a real forecaster would have had changes what its
numbers mean.** In the table under challenge 1 below, "real forecasts" means the weather forecast
that was genuinely available when the power forecast was made; "actual weather, after the fact"
means observations, or a weather model re-run after the event, that no forecaster would have had.
Two of the studies below use actual weather after the fact. [Kaas et al.
(2026)](https://arxiv.org/abs/2607.01966) and [Hertel et al.
(2026)](https://arxiv.org/abs/2607.15705) both forecast the same public dataset of 200 German
low-voltage feeders, which carries short-range weather forecasts issued one to three hours ahead
rather than the multi-day forecast a real forecaster would have had. [Hertel et al.
(2026)](https://arxiv.org/abs/2607.15705) also forecast a transmission control area and 287
individual customers, and for those two datasets they use ERA5 reanalysis, which is the weather as
it turned out. [Hertel et al. (2026)](https://arxiv.org/abs/2607.15705) do so deliberately, because
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

## The eight challenges Flexpectation has to solve, and what the literature says about each

Flexpectation's specification breaks into eight challenges. This section takes each in turn, says
what the challenge is, reports the most relevant published results found in our literature search,
and says where those results stop short. The coverage is uneven. The first challenge has enough
published results to tabulate, and the second is the most mature field on the list. For most of the
remaining six we found no published result that could be compared against anything, so those are
described in prose: the absence is itself the finding.

Everything below is what our search surfaced as most relevant to NGED, not a ranking of the field:
every study answers the problem its own authors set, and they set different problems. The eight are
not a shortlist to choose from — the plan is to attempt all of them, for a reason set out under "Set
against this literature, what we plan is ambitious" below: they may turn out to be one challenge
rather than eight.

### 1. Probabilistic forecasts of net demand at substations

**In summary.** A large literature forecasts substation load, but very little of what we read can be
compared with the rest of it, and we found none of it driving a probabilistic substation forecast
from a weather ensemble across a 14-day horizon.

**The challenge.** Forecast net demand — demand minus whatever generation sits behind the substation
— at every grid supply point, bulk supply point, and primary substation, half-hourly, 14 days ahead,
updated every six hours, as a range of possible loads with a probability attached to each rather
than as a single number. NGED acts on the forecast one to ten days ahead, and the question NGED asks
of the forecast is "how likely is load to cross this substation's firm capacity — the load the
substation can carry safely with its largest transformer out of service?" rather than "what is the
most likely load?". That limit is not one number. A transformer's safe rating rises as the air gets
colder and as wind carries heat away from it, so the same plant carries more on a windy January
night than on a still August afternoon; and because the plant has thermal mass, it can take a large
overload for a short period without damage, so how *long* an exceedance lasts matters as much as how
far above the rating it goes. A single firm capacity is a planning convention laid over a limit that
moves. This is the highest priority of the eight challenges, and the other seven exist mainly to
make that net-demand forecast better.

**The 14-day horizon sits at the edge of what a weather ensemble can supply.** [Buizza and
Leutbecher (2015)](https://doi.org/10.1002/qj.2619) put at 16 to 23 days the lead time beyond which
a weather ensemble stops beating a climatological distribution. Buizza and Leutbecher measured
the forecast skill horizon on upper-air variables rather than on the near-surface temperature
and irradiance that drive substation load.

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

**Even within this one table, the studies cannot be compared with each other.** The sharpest
illustration comes from two papers published a fortnight apart, by overlapping groups at the
Karlsruhe Institute of Technology, on the same 200 German low-voltage feeders. [Kaas et al.
(2026)](https://arxiv.org/abs/2607.01966) and [Hertel et al.
(2026)](https://arxiv.org/abs/2607.15705) name different models as best. Inside [Kaas et al.
(2026)](https://arxiv.org/abs/2607.01966), mean absolute error and an overload-decision metric name
different winners again. Neither disagreement is a mistake: the two papers test different sets of
models at different time resolutions, and the two metrics answer different questions.

**The closest analogue to Flexpectation in a live setting is a Portuguese production system.**
[Pinheiro et al. (2023)](https://doi.org/10.1016/j.apenergy.2022.120493) run a production
forecasting system covering 96,989 secondary substations day-ahead, using real weather forecasts
with a realistic 7–8 hour delay. It is the only study in this review running in live production at
national scale.

**The cheapest positive result in this review also comes from [Pinheiro et al.
(2023)](https://doi.org/10.1016/j.apenergy.2022.120493)'s Portuguese system.** Adding
period-specific copies of the same model — for weekends, August, public holidays, Easter, Christmas,
and so on — to a general-purpose one, with the weights updated online as new data arrived, cut
system-level root-mean-square error by 24% against that general-purpose model alone, from 203 MW to
154 MW.

**A non-weather input helped more than a better weather forecast at wind-connected substations.**
Artificial Forecasting's Alpha work added the National Energy System Operator's national demand and
operational-margin data to its substation models, and reports that the operational-margin feature
"generator availability" was "almost universally heavily used as a feature in the model and almost
universally substantially improved results" at primary substations connected to wind generation. The
project calls that result surprising, because the models it improved on already carried a forecast
wind-speed feature. No figure isolating that one feature from the rest of the inputs added in the
same round is published, so the finding says which input is worth trying rather than what it is
worth.

**Where the gaps are: no study we found drives substation uncertainty from a weather ensemble across
a full 14-day horizon.** What we did not find is ensemble-driven uncertainty at half-hourly
resolution, per substation, across a full 14-day horizon — and both [Haben et al.
(2021)](https://arxiv.org/abs/2106.00006) and [Ludwig et al.
(2023)](https://doi.org/10.1080/01605682.2022.2115411) — whose forecast is driven by the same
51-member ECMWF ensemble Flexpectation uses — ask in print for the substation-level part of it,
though neither names a resolution or a horizon.

**Almost every study here optimises average accuracy, but NGED's question is about the top of the
distribution.**  [Browell and Fasiolo
(2021)](https://arxiv.org/abs/2103.10335) is the one study here that models the upper tail
explicitly, and what they found is a warning rather than a reassurance: they work across risk levels
from 0.01% to 0.25%, one of which — 0.05%, or one part in two thousand — corresponds to reserve
being sufficient in all but about four hours a year — but they also find that "below 1% and above
99% the forecasts based on quantile regression only are not calibrated at any GSP Group. Therefore,
these quantiles are not suitable for use in decision-making", even with five years of half-hourly
data across regions far larger than a substation. Above the 1st and 99th percentiles, Browell and
Fasiolo switch to a fitted parametric tail.

**A decision metric that holds risk constant and prices it in money has been published at
distribution level once, on a synthetic network.** [Bernecker et al.
(2025)](https://doi.org/10.1016/j.ijepes.2025.110713) fix the confidence level at which a network
operator acts, at 95%, and compare what two forecasts cost that operator in congestion management:
**3,102 euros a year using standard load profiles against 86 euros using a smart-meter-informed
forecast**, a 97% reduction, alongside a 90% fall in the number of voltage violations. Bernecker et
al. also give the exchange rate NGED would want — a 1% cut in the standard deviation of forecast
error is worth about 1.4% of congestion-management cost on average across rollout levels, though the
saving varies between levels and is negative at some of them. We read the sections of that paper
bearing on the cost calculation rather than the whole of it. Two things keep the gap open: the
network is a modified IEEE 33-node test system rather than a real one, and what Bernecker et al.
compare is two *information levels*, not two forecasting models, so we found no case of the metric
being used to rank one forecast against another at a real substation.

**The rest of the decision metric exists in pieces.** [Browell and Fasiolo
(2021)](https://arxiv.org/abs/2103.10335) fix a risk appetite, compute the reserve volume each
forecast would need to hold it, and compare — the harder half of the job, done across whole grid
supply point groups. [Angus et al. (2027)](https://doi.org/10.1016/j.epsr.2026.113545) bring that
idea down to individual assets, forecasting day-ahead how hard each of 644 low-voltage transformers
in the UK can safely be pushed, and winning 10 to 12% more capacity than a fixed setting while the
risk of overheating came out at whatever percentile they asked for; we read their preprint rather
than the published paper.

**What is still missing is the price on a real network.** Meteorology has priced forecast decisions
this way for decades: [Richardson (2000)](https://doi.org/10.1002/qj.49712656313) computed the
relative economic value of the ECMWF ensemble across the whole range of ratios between the cost of
acting on a forecast and the loss avoided by acting.  Every published version of that curve on a real
distribution network, though, is denominated in energy volumes or in spare capacity rather than in
money.

**Topology enters this literature almost entirely as one thing: the summation constraint in
hierarchical forecast reconciliation.** [Nespoli et al. (2020)](https://arxiv.org/abs/1910.03976)
apply it to real secondary substations and cabinets in a Swiss distribution grid and gain up to 10%
in root-mean-square error at the upper levels of the hierarchy, and under 1% at the bottom. A
summation constraint carries no information about which substation neighbours which, and it stops
holding the moment the network is switched into an abnormal running arrangement (challenge 4 below).
That is why a summation constraint is not enough for Flexpectation. The nearest thing to an
exception we found is [Jung et al. (2024)](https://doi.org/10.1049/icp.2024.1900), who feed which
busbar connects to which into a graph neural network — but they forecast voltage rather than load,
and test their model only in simulation; we read their abstract rather than the full paper.

**The nearest answer to whether the shape of the network improves the forecast was measured on
NGED's own published data, and it points away from geography.** [Campagne et al.
(2025)](https://arxiv.org/abs/2507.03690) compare eight graph neural network architectures against
feed-forward, persistence, and foundation-model baselines on French regional load and on the GB
distribution networks' open smart-meter feed — around two million meters and 50,000 substations
across NGED's and SSEN's areas. Graph-aware models beat the baselines on both. But which graph wins
changes with granularity: spatially informed graphs worked on the coarse French regions, whereas
"for the UK data, data-driven graphs proved more suitable since that dataset exhibits finer spatial
granularity and noisier correlations". Their graphs are built from geographic distance or from
correlation between series, never from electrical connectivity, so the specific question stays open.

**Does knowing the shape of the network make the forecast better, or only more consistent?** NGED
holds a map of which substations and metered generators connect to each other, and no study we found
has used that map as a forecast input.

### 2. Forecasting metered generators

**In summary.** Forecasting wind and solar from a weather forecast is the mature case, and one paper
matches Flexpectation's challenge closely; nothing we found forecasts a distribution-connected
battery, gas generator, or biofuel plant inside a net-demand forecast.

**The challenge.** Twelve of the 32 series in the trial area are individually metered generators —
six solar farms, three wind farms, a biofuel plant, a battery, and a gas generator — and each needs
the same probabilistic, half-hourly, 14-day forecast as a substation. Solar and wind are driven by
weather the ensemble supplies directly. The battery, the gas generator, and the biofuel plant are
dispatched on market prices and operator decisions, and no weather forecast contains either.

**Forecasting wind and solar output from a weather forecast is the most mature challenge on this
list, and the one challenge where different studies' results can be compared directly.** [Browell et
al. (2025)](https://doi.org/10.1016/j.ijforecast.2025.10.005) report the Hybrid Energy Forecasting
and Trading Competition (HEFTCom), in which every team forecast the combined day-ahead output of one
GB portfolio — the 1.2 GW Hornsea 1 offshore wind farm plus the aggregate solar capacity of East
England, about 3.6 GW together — from real weather forecasts as they arrived. The winning team
scored a mean pinball loss of 22.18 MWh against the organisers' starter benchmark of 53.58, with the
next two teams on 23.18 and 24.64. The organisers also entered a more competitive reference,
unranked, which scored 25.38. HEFTCom is the competition in this review whose data is closest to
NGED's challenge, so it is the clearest case here of many teams forecasting the same data with the
same metric — which is exactly what the rest of this literature cannot do. Its wind half is a single
offshore farm far larger than any generator NGED meters, and its solar half is a regional aggregate
rather than a plant.

**At the scale of an individual generator, the closest work is on wind.** [Dantas and Browell
(2026)](https://doi.org/10.1002/we.70079) forecast 73 wind farms in GB — 34 onshore, 39 offshore —
from the ECMWF ensemble, seamlessly from 6 to 162 hours ahead.

**Their method separates the two things that can go wrong, and that separation is the paper's main
result.** A wind power forecast can be wrong because the weather forecast was wrong, or because the
conversion from weather to power was wrong. [Dantas and Browell
(2026)](https://doi.org/10.1002/we.70079) quantify both.

**[Dantas and Browell (2026)](https://doi.org/10.1002/we.70079) conclude that whether
weather-forecast error or weather-to-power conversion error dominates flips with lead time, and that
the lead time at which it flips varies a lot between sites.** Weather-to-power uncertainty dominates
the short term and weather-forecast uncertainty the mid-term, with the transition typically 2 to 3
days ahead, arriving earlier for offshore farms than onshore ones, and varying dramatically between
farms. Handling both is what lets one model cover 6 to 162 hours, whereas the field had previously
used a short-term model and a separate mid-term one. Flexpectation faces the same seam over its
14-day horizon, and this paper is evidence that the seam can be removed rather than managed.

**A second conclusion is more uncomfortable for a project built on an ensemble: a deterministic
forecast at higher resolution beat the ensemble at short lead times.** Their short-term reference
method uses ECMWF's deterministic HRES at 0.1° and hourly steps, while their own method uses the
ensemble at 0.5° and 6-hourly steps, because the archive they drew on carries no 100 m wind and no
finer ensemble. On those unequal terms "the short-term method is better than the proposed method for
horizons up to the day ahead", although it "cannot outperform the proposed method for horizons
beyond 1 day ahead". Match the two on time step and on variables and their own method matches the
short-term one on the first day and beats it from two days ahead — though even then the
deterministic reference keeps its finer 0.1° grid, which the paper flags in the caption of the
figure the comparison rests on. The lesson for Flexpectation is that a comparison of ensemble
against deterministic measures the resolution difference unless the resolution is equalised first,
and that equalising it fully is harder than it sounds.

**A third conclusion is about what an average score hides.** Averaged over a full test year, their
method showed no gain over the state of the art at day 0 and day 1. Restricted to the periods when
the ensemble members disagreed most — frontal passages and the like — it showed a real gain even at
those short lead times, "which was not evident in the long-run average CRPS", because a
deterministic method "is not able to discriminate between high/low weather uncertainty".

**Two things [Dantas and Browell (2026)](https://doi.org/10.1002/we.70079)'s method does not do are
things Flexpectation will need.** They fit a separate model per wind farm rather than one model
across all 73, and they list as future work a "member-by-member correction to retain spatio-temporal
structure in ensemble members", which "would allow for spatio-temporal coherence between forecasts
from different wind farms" — a plain signal that the forecasts as published carry no such coherence.
A net-demand forecast that adds several generators and a substation together needs precisely that
coherence, which cannot be taken from this paper.

**Gradient-boosted trees, fitted separately for each kind of generator, is what this literature does
and what won when teams were scored against each other on the same data.** [Dantas and Browell
(2026)](https://doi.org/10.1002/we.70079) model the weather-to-power relationship with quantile
regression on gradient-boosted trees, fitting a separate model for each quantile. In HEFTCom the
winning team fitted gradient-boosted trees separately for wind and for solar and separately for each
weather source, 9 of the top 10 teams forecast wind and solar separately before combining them, and
[Browell et al. (2025)](https://doi.org/10.1016/j.ijforecast.2025.10.005) conclude that
gradient-boosted trees remain competitive for day-ahead wind and solar forecasting, with performance
depending heavily on implementation. NGED's own forecasting system, EFFS, independently selected
XGBoost when it evaluated model families. Two results cut the other way, though neither is an
argument against trees: team Rnt finished third in HEFTCom's forecasting track using no tree-based
model at all, feeding embeddings from machine-learned weather-forecasting models they built in-house
— a published deep-learning weather model extended to add solar irradiance and day-ahead lead times,
driven by station observations, radar, satellite imagery, and numerical-weather-prediction analysis
— into downstream neural networks that predicted wind and solar generation. Rnt's route therefore
rests on building and running a machine-learned weather model, which is a far larger undertaking
than swapping one downstream model family for another. And [Nguyen and Müsgens
(2026)](https://doi.org/10.1063/5.0300682) meta-analyse 4,687 skill scores extracted from 188 solar
forecasting papers. Their baseline class is classic statistical time-series models — the
autoregressive integrated moving average (ARIMA) family, exponential smoothing (ETS, for error,
trend, and seasonality), and multivariate relatives such as autoregressive models with exogenous
inputs (ARX). Beyond 6 hours ahead, two classes beat that baseline. Ensemble-hybrid models gain 7.0
percentage points of skill score. An ensemble-hybrid chains one model's output into the next as an
input, and also averages several models together. Pure ensemble models gain 8.3 points. A pure
ensemble runs several models and aggregates their outputs, without the chaining. Individual
machine-learning models (including gradient-boosted trees) and regressions show no significant
advantage at all over classic statistical time-series models beyond 6 hours ahead, and taking the
weather model's own output as the forecast scores significantly worse, 14.3 percentage points of
skill score below that baseline. That class is the numerical weather prediction irradiance field
itself — usually global horizontal irradiance, at most post-processed or averaged across several
weather models — used as the forecast rather than fed as an input to a fitted model. Most of the
papers in their sample forecast irradiance rather than plant output, so for those papers no power
curve is involved at all. Their model classes follow each paper's own nomenclature, so the boundary
between an ensemble and a hybrid is fuzzy by their own account. Their own advice is to exhaust the
simple models first, because classical statistical time-series methods "still have very good
performance compared to more complex methods such as individual ML models".

**Most of NGED's metered generators are solar, and the largest meta-analysis of solar forecasting
puts the weight on exactly the input Flexpectation is built around.** [Nguyen and Müsgens
(2026)](https://doi.org/10.1063/5.0300682) fit a separate regression for each horizon band, and
beyond 6 hours ahead numerical weather prediction as an input is worth 11.6 percentage points of
skill score — the largest input effect they measure — with locally measured meteorological data
worth a further 5.1. The inputs that pay at short range carry the opposite sign out there: lagged
power costs 6.4 percentage points beyond 6 hours where it gains 8.2 within the day, and data from
neighbouring sites costs 5.5 where it gains 3.9. Two scope limits travel with those numbers. Their
sample is deterministic forecasting of solar irradiance or plant output rather than probabilistic
substation net demand, and their beyond-6-hours band lumps the whole of NGED's 1-to-14-day window
into a single category, so the figures say which inputs earn their keep at long range, not how much
they earn at day 10.

**A caution on carrying any of these numbers across to GB.** Skill score is meant to normalise away
location, but [Nguyen and Müsgens (2026)](https://doi.org/10.1063/5.0300682) find it does not: their
regressions use the warm-temperate Köppen-Geiger zone C, which is GB's, as the baseline, and the
equatorial, arid, and snow zones score 1.6 to 6.0 percentage points higher. Nguyen and Müsgens read
that as the reference model doing relatively worse where forecasting is harder, which inflates the
skill score rather than reflecting a better forecast, and conclude that transferring findings
between climate zones has to be done carefully. A GB project should therefore expect the skill
scores it can reach to sit below the ones a typical paper reports.

**For generators, the measured prize from better weather-to-power physics is largest at short lead
times, which is not where NGED acts.** Differentiable physics attacks the weather-to-power half of
the error, so on [Dantas and Browell (2026)](https://doi.org/10.1002/we.70079)'s measurement it has
most to offer inside the first 2 to 3 days of the 1-to-10-day window NGED acts on, and less beyond
it, where the weather forecast itself is the binding constraint. Adding a learned residual to a
physical generator model is established practice: [Gijón et al.
(2025)](https://arxiv.org/abs/2502.07344) fit a physics-inspired power model to a wind farm of four
turbines and train a second model on the residual, cutting the physics model's mean absolute
percentage error by 37% and its mean absolute error by 28%, with conformalised quantile regression
supplying the uncertainty. The hybrid gains that margin over the physics model alone; against a
purely data-driven model given the same eight inputs it "essentially matches" rather than beats, so
what the physics buys here is interpretability at no accuracy cost. But they predict power from
measured wind rather than forecasting it days ahead, and we found nobody putting a differentiable
generator model inside a network's probabilistic net-demand forecast. On lead time alone, then, the
larger differentiable-physics prize for Flexpectation would be on the demand side rather than the
generation side.

**The second reason to try differentiable physics on generators is the metadata NGED does not
have.** The generation forecasts in this literature are handed the numbers we lack: [Teng et al.
(2023)](https://doi.org/10.1016/j.rser.2023.113662) are given each site's capacity, and HEFTCom's
portfolio was one named 1.2 GW offshore wind farm plus the solar capacity of a region. When an
export-cable fault cut that wind farm's available capacity mid-competition, the winning team clipped
its quantiles to the capacity implied by the outage notices the farm is obliged to publish, while
the organisers' benchmark ignored the fault and, in [Browell et al.
(2025)](https://doi.org/10.1016/j.ijforecast.2025.10.005)'s words, "performed extremely poorly as a
result". NGED's embedded generators publish no such notices. NGED's Embedded Capacity Register does
give a registered capacity for generation of 50 kW and above: in the August 2026 edition, 5,598
connected generators totalling 11,456 MW, of which 4,202 sites and 5,958 MW are solar, and 4,996 of
the 5,598 name the primary substation they connect at. But a registered capacity is contractual
rather than operational — the export limit is the one "permitted as per the connection agreement" —
and the register carries no panel tilt, panel azimuth, or ratio of direct-current to
alternating-current rating for any of them. The one field that would bear on availability, the flag
for a flexible connection under Active Network Management, was unpopulated for every connected
generator in the August 2026 edition. A differentiable plant model would therefore start from the
registered capacity and fit the rest, including the day-to-day availability that a register cannot
express. Each half of that fitting has been made to work on its own: [Pierrot and Pinson
(2024)](https://doi.org/10.1080/00401706.2024.2350421) treat a wind farm's capacity as a
time-varying bound fitted jointly with the forecast, and beat probabilistic persistence by 34.2% on
continuous ranked probability score over a 5-month test period at the Anholt offshore wind farm,
drawn from 14 months of data, though their one clean test of tracking the bound on its own gained
2.43%, and [Meng et al. (2020)](https://doi.org/10.1016/j.solener.2020.09.077) infer the tilt and
azimuth of 13 roof photovoltaic systems in the Netherlands to mean absolute errors of 4.3° and 4.5°,
matching the shape of each system's hourly output against plane-of-array irradiance computed for
every candidate orientation, from a station up to 195 km away. Two details of that paper matter to
Flexpectation. Because both curves are normalised before matching, the method needs no nameplate
rating, which is the one piece of metadata it might otherwise have demanded. And it reports its
accuracy only in degrees: Meng et al. never convert an orientation error into a power error. Their
own stated limitation is that all 13 systems sit on the same standardised 42° roof, so the tilt
figure is tested against a single true tilt; the accompanying simulation study, which does span
orientations, scores 4.8° on tilt and 3.1° on azimuth across 21 notional panels, and is weakest on
the south-facing ones at 7.9° on tilt against 3.0° to 5.1° for north, east, and west. Neither method
sits inside a substation's net-demand forecast, which is where Flexpectation would have to put it.
How much the physics is worth getting right has been measured for solar: [Mayer and Gróf
(2021)](https://doi.org/10.1016/j.apenergy.2020.116239) score all 32,400 combinations of nine
irradiance-separation, ten transposition, three reflection-loss, five cell-temperature, four module,
two shading, and three inverter models against a year of 15-minute metered output from 16
ground-mounted plants at 14 sites in Hungary, and find that the best model chain has 13% lower mean
absolute error than the worst, naming irradiance separation and transposition — the step that
projects horizontal irradiance onto the plane of the array, and so depends on the array's tilt and
azimuth — as the two whose model choice matters most. Two limits on that 13%: it is the gap between
the extremes of 32,400 chains rather than a typical penalty, and every plant's tilt and azimuth came
from its design documentation and stayed fixed, so Mayer and Gróf bound the cost of choosing the
wrong physical model, not the cost of not knowing a system's orientation.

**What better orientation metadata is worth to a forecast is a number we have not found in the
literature, and the four papers closest to the question each stop short of it.** [Meng et al.
(2020)](https://doi.org/10.1016/j.solener.2020.09.077) and [Saint-Drenan et al.
(2015)](https://doi.org/10.1016/j.solener.2015.07.024) both recover a system's tilt and azimuth from
its metered alternating-current power output paired with an irradiance series measured somewhere
else, never from the power series alone. Meng et al. match normalised hourly power against
normalised plane-of-array irradiance on the clearest day of each month, transposing hourly readings
from a weather station up to 195 km away, and land within 4.3° of tilt and 4.5° of azimuth on
rooftop systems across the Netherlands. Saint-Drenan et al. fit tilt, azimuth, and an angular-loss
coefficient to a year of 15-minute power with irradiance from the HelioClim-3 satellite database and
air temperature from the COSMO-DE weather analysis, landing within 1.5° of tilt and 5° of azimuth on
the worse of two individually-metered Swiss plants. Both report their accuracy in degrees alone.
[Mayer and Gróf (2021)](https://doi.org/10.1016/j.apenergy.2020.116239) price the choice of physical
model with every plant's geometry known and held fixed. And [Amaro e Silva and Brito
(2019)](https://doi.org/10.1016/j.apenergy.2019.113807) price the mismatch between
differently-tilted surfaces in a forecast made 10 seconds ahead by watching cloud shadows cross a 1
km² grid of pyranometers in Hawaii, on synthetic photovoltaic output derived from those sensors
rather than from metered plants — neither our horizon nor our question, and the orientations there
are known throughout rather than estimated.

**Two findings from that search shape what Flexpectation should build, and they separate the two
cases the project faces.** [Saint-Drenan et al.
(2015)](https://doi.org/10.1016/j.solener.2015.07.024) report that an azimuth fitted 5° from the
true one gave better simulations than the true value, because the fit balances the systematic error
of the physical model, concluding that the output "should be seen as a set of parameters that lead
to the best simulation and not necessarily as the actual characteristics of the PV plant". That
makes accuracy in degrees the wrong target for Flexpectation: what a differentiable plant model
needs is an effective tilt and azimuth that make the forecast right, which is measurable against the
forecast we already score. For a single metered site, fitting tilt, azimuth, and the effective
direct- and alternating-current capacities is the plan — by gradient descent inside the forecast
rather than by the grid search Saint-Drenan et al. use, so that the fit stays joint and
probabilistic.

**Their second finding rules out doing the same thing to a substation, and points at what to do
instead.** Saint-Drenan et al. name Flexpectation's unmetered case as their method's failure mode:
where a power series is "the aggregated production of modules with different orientations", the
algorithm "performs poorly", because it assumes one orientation per plant. Flexpectation therefore
estimates no single orientation per substation. The fleet model represents the aggregate as a
learned mixture of east-, south-, and west-facing basis shapes — which span the fixed-tilt
orientations, and which a single tilt and azimuth cannot, because a mixed fleet produces a broad
mound of power where one south-facing array produces a sharp hill — with a tracking shape added
where ground-mounted trackers sit behind the substation, and a soft clip in place of a hard one
because many differently-sized inverters saturate at different irradiances rather than all at once.
That soft clip is also how the ratio of direct-current to alternating-current rating enters the
model: as a learned aggregate limit and a learned curvature, rather than as a register value NGED
does not hold.

**Where the gap is: nothing we found forecasts a distribution-connected battery, gas generator, or
biofuel plant inside a net-demand forecast.** For the battery there is at least a method to borrow.
[Bian et al. (2024)](https://doi.org/10.1109/TSG.2023.3303469) recover a price-taking storage
operator's own optimisation parameters by gradient descent on historical prices and observed
dispatch, and prove the recovered parameters converge to the true ones for a class of storage
models. We found no method worth borrowing for the gas generator or the biofuel plant; what little
exists forecasts a gas or biofuel plant's own output directly rather than as a component of a
substation's net demand. Otherwise the closest the literature comes is a warning rather than a
method: [Pinheiro et al. (2023)](https://doi.org/10.1016/j.apenergy.2022.120493) found that sites
serving a single customer were forecast markedly worse than the rest (finding 3 below). We read that
as the signature of load following decisions no weather model can see; Pinheiro et al. attribute it
to their model structure being tuned for the distributor's own substations.

### 3. Estimating the effective capacity of metered generators

**In summary.** A method exists for each generation technology separately, but we found none run
across a mixed fleet of individually metered generators at a distribution network, and the two
studies that measure what estimating capacity is worth downstream measure it for wind alone, at
national or single-farm scale.

**The challenge.** We call the amount of generation actually available at a metered site its
*effective capacity*: the output it could produce right now if the weather allowed, as opposed to
its nameplate rating. Turbines go out for repair, inverters degrade, and sites are curtailed — told
by the network operator to generate less than they could. A 20 MW wind farm that has been limited to
14 MW for a month is, for forecasting purposes, a different wind farm, and a model trained on its
nameplate rating cannot see the difference. The same goes for a primary substation with a large
metered generator connected behind it. This challenge concerns the 12 metered generators in the
trial area, each of which has a half-hourly meter of its own; the unmetered rooftop solar and small
wind of challenge 7 are a separate task.

**For wind, one paper hits our challenge exactly, and publishes its method.** [Dantas and Browell
(2026)](https://doi.org/10.1002/we.70079) needed available capacity for the same reason we do: the
metered-output database they use "does not include information related to the farms' available
capacity over time", so rather than use a nameplate rating they estimate a time series of available
capacity for each farm and normalise that farm's power by it before modelling. Their method needs no
capacity register and no outage messages.
Dantas and Browell did use one data source Flexpectation will not have in the same form: they
excluded curtailed half-hours using published bid-acceptance volumes, which exist for
transmission-connected wind farms and not for NGED's embedded generators. Flexpectation has
something adjacent, in that the active network management system records curtailment for each of
NGED's generator customers, but that record is ambiguous enough that it cannot simply be dropped in
where [Dantas and Browell (2026)](https://doi.org/10.1002/we.70079) use a bid-acceptance volume.

**The general shape of that rule — take the running maximum of production — has since been
criticised in print, on the grounds that matter most to NGED. The criticism is of the method rather
than of that paper: Viotti et al. use the running maximum as their own reference method and do not
cite Dantas and Browell.** [Viotti et al. (2026)](https://doi.org/10.1002/we.70136) point out that
taking the running maximum of production "requires monotonically increasing capacity and relies on
frequent high wind events" — and NGED's effective capacity goes *down* when a turbine is out for
repair, which is the case this project's effective-capacity work exists to handle. They fit the most
likely capacity time series instead, by quadratic optimisation against a capacity factor simulated
from reanalysis and a power curve, and report **27.2% lower normalised mean absolute error** than
the running maximum at quantifying capacity after a new wind farm connects. They also measure what
the choice is worth downstream: a forecasting model trained on production normalised their way
scored **2.0% lower mean absolute error and 2.3% lower root-mean-square error** day-ahead than the
same model normalised by the running maximum, across Sweden as a whole; across individual bidding
zones and parameterisations the range runs from 8.0% better to 0.6% worse. Which variant produced
which figure matters to NGED, and the two point opposite ways. The 27.2% capacity figure is scored
by the monotonic variant, the one that still assumes capacity only rises, and their non-monotonic
variant — which can track capacity downwards, as NGED needs — comes out 31% worse on that capacity
test. The 2.0% forecast figure belongs to the simplified non-monotonic parameterisation instead, and
the authors conclude that "normalization using the non-monotonic parameterization yields the best
forecasts, possibly because it captures real changes in available capacity". Their target throughout
is a Swedish bidding zone or the whole Swedish market area rather than an individual farm, and they
report that at 5-minute per-farm resolution the running maximum is already "a robust estimate of the
installed capacity per farm", so the fitting earns its advantage on hourly, region-aggregated data.
They do test the de-rating case, by suppressing production in the 30 days after a step, and there
both their method and the running maximum get worse, with no comparable improvement figure to
report. The case NGED cares about most is the one their paper answers least well.

**For solar, the equivalent can be done from the power signal and nothing else.** The best-known
tool, the open-source [RdTools](https://doi.org/10.5281/zenodo.1210316), does need site irradiance
to pick out the clear-sky periods it analyses, and its own documentation warns that a satellite
substitute gives less stable results. But [Meyers et al.
(2020)](https://doi.org/10.1109/JPHOTOV.2019.2957646) removed that requirement: their unsupervised
signal-processing approach "only requires a measured power signal as an input — no irradiance data,
temperature data, or system configuration information", and they validate that approach against
RdTools on the same dataset, reporting greater robustness to data anomalies. Their approach is now
the open-source Solar Data Tools, whose pipeline detects capacity changes and clipping and estimates
degradation, with a Monte Carlo step that returns a distribution rather than a point estimate.

**Estimating capacity jointly with the forecast, rather than in two stages, has also been
published.** [Pierrot and Pinson (2024)](https://doi.org/10.1080/00401706.2024.2350421) treat a wind
farm's available capacity as the unknown, time-varying upper bound of a generalised logit-normal
distribution and track it online by normalised gradient descent, fitting the bound and the forecast
together rather than in two stages. On 14 months of ten-minute data from the Anholt offshore wind
farm, split into 9 months of training and a 5-month test period, that improved the ten-minute-ahead
continuous ranked probability score on the test period by **34.2% over probabilistic persistence and
17.9% over a benchmark that holds the bound fixed at one**. Read that second figure carefully. The
fixed-bound benchmark is their earlier rolling maximum-likelihood method rather than the same
gradient-descent model with its bound frozen, so the 17.9% mixes the gain from tracking the bound
with the gain from changing the fitting method. Their one clean test of tracking on its own pairs
the rolling maximum-likelihood method with a varying bound against the identical method with a fixed
one, and that gained 2.43%, which they report as no "significant improvement when compared to its
equivalent with a fixed bound". Tracking a varying bound is worth having, then, but this paper does
not show it is worth 17.9% by itself.

**Where the gap is: none of this has been done across a mixed fleet of individually metered
generators at a distribution network, or tested for whether it improves the forecast NGED buys
flexibility against.** The pieces exist, and most of them work from a revenue meter alone.

**We plan to attempt that combination two ways, neither of which starts from scratch.** The first is
the two-stage route: estimate a capacity time series from the meter, then normalise by it before
training — running the quadratic-optimisation method of [Viotti et al.
(2026)](https://doi.org/10.1002/we.70136) and the Solar Data Tools pipeline against each other on
our own sites, with the running maximum of [Dantas and Browell
(2026)](https://doi.org/10.1002/we.70079) as the reference the published numbers are quoted against
rather than as a candidate, because a ratchet cannot follow a de-rating downwards. The second is
joint estimation, of which [Pierrot and Pinson
(2024)](https://doi.org/10.1080/00401706.2024.2350421) are the published precedent: a
differentiable-physics model of each generator in which the physical parameters — including the
plant's direct-current and alternating-current capacity — are fitted as probability distributions
rather than as single numbers, so that capacity is recovered with its own uncertainty attached and
the forecast inherits that uncertainty instead of treating capacity as known.

**NGED's specification asks us to track effective capacity over time and, optionally, to combine it
with the forecast into a "prevailing conditions" view. We intend to go further and use it to
normalise each metered generator's series before training — but whether that normalisation earns its
place is itself testable, and one published result suggests it may not.** [de Vilmarest et al.
(2024)](https://doi.org/10.1109/TPWRS.2023.3310280) removed the embedded wind and solar capacities
from their model of GB regional net load, and a Kalman filter tracking the coefficients absorbed the
loss completely — error rose by more than 10% for the same model fitted offline, and fell by 0.4%
for the adaptive one. The capacities they removed are the aggregate installed capacity of a whole
region's embedded generation rather than one metered generator's effective capacity, so the result
is a caution about normalisation rather than a like-for-like test of ours. We will run that
comparison rather than assume the normalisation is needed.

**The clearest published demonstration of why effective capacity matters is incidental.** Hornsea
1's export cable faulted on 19 January 2024, about two weeks before HEFTCom's competition period was
due to start. The competition began on schedule on 1 February; the organisers describe not having
accounted for the fault as "an oversight", so they restarted the competition on 20 February, a month
after the fault. Many teams still struggled in the weeks that followed. Teams forecasting wind and
solar separately could post-process their wind forecast for the new export limit, while those
forecasting the combined total "found it harder to adapt", and the organisers' benchmark, which took
no account of the fault, "performed extremely poorly as a result".

### 4. Detecting switching events

**In summary.** One paper detects switching at a real network operator, but detects it in the gap
between the substation's own meter and a second estimate of the same load, built from smart-meter
and bulk-customer readings taken below the substation — a second estimate NGED does not have.
Electricity North West's ATLAS project sorted step changes into faulty metering and network
reconfigurations on GB substations in 2016, from power measurements alone, and published no
precision or recall for either rule.

**The challenge.** When a cable fault or planned maintenance moves part of a network from one
substation to another, the load the first substation meters steps down and the load of each
substation picking up that work steps up, with no change in the underlying demand. The pick-up is
usually shared across two or three neighbouring substations rather than landing on one, and usually
only part of a substation's load moves — a continuous fraction, with no minimum size — rather than a
whole subgrid. NGED's substations spend roughly a tenth of their operating time in an abnormal
running arrangement. Switching labels exist for the 32-series trial area but not for the wider
network, so a method that is to scale to the wider network has to work from power measurements
alone.

**One paper detects these events at a real network operator, and stops at the detection.** [Bouman
et al. (2024)](https://arxiv.org/abs/2405.16164), working with the Dutch network operator Alliander,
study 180 primary substations at 15-minute resolution over roughly a year, detecting the step
changes caused when a cable fault or planned maintenance reroutes part of a subgrid to a different
substation. Events run from a few minutes to several months. What Alliander wants from the detection
is a clean maximum and minimum load for each substation, because those two extremes decide whether
the substation needs a bigger transformer or can take on more customers, and a switch pushes both of
them to the wrong value. The detected periods are therefore cut out of the history before the
extremes are read off. Forecasting is the half Flexpectation would add: a forecast that keeps
running through a switching event, rather than a history with the switched periods removed.

**[Bouman et al. (2024)](https://arxiv.org/abs/2405.16164)'s central trick is to detect on a
residual rather than on the load itself.** Alliander maintains an independent bottom-up estimate of
each substation's load, reconstructed from customer telemetry and modelled profiles. Bouman et al.
fit and rescale that bottom-up estimate to the measured series, then hunt for step changes in the
*difference* between the estimate and the measurement. Normal daily and seasonal variation largely
cancels, leaving a much cleaner signal. NGED has no bottom-up estimate of substation load, and
building one is not in Flexpectation's scope, because the project uses no telemetry from below
primary substation level.

**Flexpectation will model its own reference series rather than measure one.** What Alliander's
bottom-up estimate gives [Bouman et al. (2024)](https://arxiv.org/abs/2405.16164) — a second opinion
on what each substation's power should have been — Flexpectation plans to produce from the
substation's own meter plus weather and the calendar. The first attempt is classical: a multiple
seasonal-trend decomposition of each series into a trend and daily, weekly, and annual cycles,
leaving a remainder in which a switch shows up as a sustained level shift. The second uses the
project's existing XGBoost machinery, trained with no power-lag features, so that an earlier
switching event cannot contaminate the expected-power estimate the way a lagged reference would.
Neither route needs metering from below the substation. Whether a modelled reference detects
switching as well as a measured one is what the project has to find out.

**Flexpectation also plans to use a signal that a one-substation-at-a-time method cannot see: the
power has to go somewhere.** [Bouman et al. (2024)](https://arxiv.org/abs/2405.16164) score each
substation against its own history — "the current analysis considers one year of measurements for
one station at a time" — so nothing in the method asks whether the power that left one substation
turned up at another. Flexpectation intends to look for both sides of the transfer: when one
substation's metered power drops, the substations that picked the load up should rise at the same
moment, and their rises should sum to the drop. A step that fails to balance that way is more likely
a meter fault or a one-off than a switch, which is where a per-substation detector spends its false
positives. The catch is that an NGED transfer usually fans out across two or three neighbours, so
the search runs over subsets of neighbours rather than over pairs, and the balance holds only
approximately. We looked for a method that checks both sides — 40 title-and-abstract queries and 10
full-text queries on OpenAlex, the Semantic Scholar, Crossref, and arXiv search interfaces, the
works citing [Bouman et al. (2024)](https://arxiv.org/abs/2405.16164) under both its journal and its
arXiv identifier (4 citing works, all read at abstract level), and the titles of all 3,160 projects
on the Energy Networks Association's Smarter Networks Portal, which publishes no abstracts to search
— and found none. The closest is [Willis et al. (1984)](https://doi.org/10.1109/TPAS.1984.318713), a
regression that corrects annual peak-load curve fits for long-range planning rather than detecting
an event at a point in time, and whose abstract says the method needs neither the size nor the
direction of a transfer as an input. The title names a "load transfer coupling" regression, which
suggests the fit couples the substations that exchange load — the feature that would make it the
closest precedent — but we could not obtain the full text to check, and the abstract does not say.

**The measured accuracy is modest, and worst on the short events.** [Bouman et al.
(2024)](https://arxiv.org/abs/2405.16164) score every detector with the F1.5 score, which blends
precision — the share of flagged points that really were switching — with recall — the share of
switched points the detector flagged — weighting recall 1.5 times as heavily as precision. An F1.5
score of 1 is a perfect detector and 0 is a useless one, so higher is better. They report the score
separately for four event lengths: 15 minutes to 6 hours, 6 hours to 3 days, 3 to 42 days, and 42
days or longer. On the two shortest bands the best detectors, statistical process control and an
isolation forest, reach about 0.2, and binary segmentation scores near what random guessing would
give. On the longest band binary segmentation reaches nearly 0.5. Combining the detectors, by
flagging a point if any of them fired, raised recall but added enough false positives that the
combination did not inherit statistical process control's strength on short events: on the two
shortest bands the combined detectors scored only marginally better than binary segmentation on its
own.

**Detecting a load transfer from a substation's own metered load has been published several times,
and every method we could read works one series at a time.** [Kim et al.
(2020)](https://doi.org/10.3390/en13174358) train a long short-term memory network on a Korean
distribution line's own past load, treat its prediction as the normal state, and flag a transfer
where the measurement departs from that prediction: "the predicted load is set as the reference
value, which is considered as normal state. Finally, the actual measured data is compared with the
predicted data, and detect it as a load transfer if the difference between them exceeds the
threshold." A later paper in the same line, [Kim
(2025)](https://doi.org/10.5370/KIEE.2025.74.11.1757), drops the trained model for a pipeline close
to the one this project plans — robust seasonal-trend decomposition, then Pruned Exact Linear Time
changepoint detection, then an isolation forest over features of each candidate changepoint —
detecting transfers "using only load time series data". [Kim et al.
(2022)](https://doi.org/10.3390/en15041441), also open access, detect the same events from
polynomial and standard-pattern preprocessing rather than from a trained model, and report 7 of 9
logged transfers found on one Korean distribution line and 7 of 7 on another, which their abstract
averages to 88.89%. Read that figure for what it is — the share of logged events found, over 16
events on two lines in one year, with no false-alarm rate given. [Kim et al.
(2020)](https://doi.org/10.3390/en13174358) report the same kind of figure on the same first line,
finding 7 of 9 planned transfers, and likewise give no false-alarm rate, so recall is the only thing
either paper measures. Both Energies papers are open access under a Creative Commons licence and we
read them in full; [Kim (2025)](https://doi.org/10.5370/KIEE.2025.74.11.1757) sits behind a
subscription, so we read only its abstract.

**A GB network operator separated switching from bad data in 2016, with cruder tools and no
published accuracy.** Electricity North West's
[ATLAS](https://smarter.energynetworks.org/projects/nia_enwl008/) project processed five years of
half-hourly demand for "over 70 BSPs and 380 primary substations" — a fleet in GB more than ten
times the size of Flexpectation's trial area. It works in stages. The first flags any abrupt change,
firing where the half-hourly change in demand exceeds ±80% of the standard deviation of the demand
series. The second then decides what kind of change it was, and the second stage is the part that
matters here: one rule handles blocks of "unreasonably zero or negative demand", and a separate rule
handles "switching operations and network reconfigurations". So the distinction between a broken
meter and a reconfigured network was drawn on GB primary substations, on power alone, without a
bottom-up reference series. ATLAS was a data-preparation project rather than a detector-benchmarking
one, so it reports no precision or recall figures for either rule, and it pairs them with "the
importance of visual sense checks of the obtained processed demand data".

### 5. Forecasting a substation as if it were always in its normal running arrangement

**In summary.** Researchers respond in one of three ways: leaving the level shifts in and paying for
them, as [Huyghues-Beaufond et al. (2020)](https://doi.org/10.1016/j.apenergy.2019.114405) do;
rewriting the history, as [Paredes and Vargas (2017)](https://doi.org/10.1049/iet-gtd.2017.0129) do;
or adapting to the new level, as [de Vilmarest et al.
(2024)](https://doi.org/10.1109/TPWRS.2023.3310280) do. We found one substation study that
conditions its forecast on an operating-state label, for a switch of a different kind, and none that
both hands a model the record of when the network was abnormal and refuses to let the model predict
those periods.

**The challenge.** NGED plan the network against what each substation would carry under its normal
running arrangement, so that is what the forecast has to predict — including for a substation that
has been sitting in an abnormal arrangement for weeks. That makes the target a quantity that was
never metered, and it makes the training history contaminated: past readings taken while the network
was abnormally configured describe a different substation from the one being forecast.

**Researchers respond in one of three ways, and two of the three alter the series the model is
trained to predict.** One strand leaves the level shifts in and reports the damage:
[Huyghues-Beaufond et al. (2020)](https://doi.org/10.1016/j.apenergy.2019.114405) run change-point
detection across 342 medium-voltage feeders in the UK and use the change-points to bound the
segments within which Huyghues-Beaufond et al. remove *outliers*. A second strand rewrites the
history to the level it would have had if the switch had never happened: [Paredes and Vargas
(2017)](https://doi.org/10.1049/iet-gtd.2017.0129) do it across 169 real feeders and report better
medium-term forecasts for it, and Artificial Forecasting does the same in its data-preparation
pipeline. The fix is a level shift applied to the *older* half of each series: Paredes and Vargas
measure how far average demand moved across the step and add that difference to every reading before
it, and the variant they recommend uses a separate difference for each hour of the day and each day
of the week rather than one number for the whole series. They take the event times from expert
identification rather than from a detector, since detection was not their subject.

**Adaptive models are the live alternative: they track a new level once it arrives, including one
that arrives abruptly, but they never record that a switch happened.** [de Vilmarest et al.
(2024)](https://doi.org/10.1109/TPWRS.2023.3310280) let a Kalman filter track the drift on the
14-region GB dataset of [Browell and Fasiolo (2021)](https://arxiv.org/abs/2103.10335) instead of
correcting the history, cutting error by about 4% in 2019, 7% in 2020, and 8% in 2021 against the
same model refitted every day. But a switching event is a step, not a drift, and a model that simply
adapts to a new load level never learns that switching happened — so it cannot report what the
substation would have carried under its normal arrangement, which is the quantity NGED needs.

**Where the gap is: we found nobody who feeds a model switching-contaminated history *deliberately*,
as information rather than as damage; the nearest is a substation study that conditions on an
operating-state label for a switch of a different kind.** The question we want to settle is whether
the contamination can be made to earn its place. Instead of correcting the series, a model could be
fed the difference between what a substation actually metered and what a model that ignores network
topology expected it to meter. That plays the same role as the residual [Bouman et al.
(2024)](https://arxiv.org/abs/2405.16164) detect on, though it is built differently: theirs is
metered load minus a topology-informed reconstruction, which goes stale the moment the network is
switched, whereas ours would be metered load minus a model that never sees topology at all.

**Flexpectation v1 will try both halves of that idea at once: the abnormal periods become an input,
and they stop being a target.** The plan is to label each substation's abnormal running arrangements
explicitly, hand those labels to the model as features so it can read its own lagged power inputs
correctly when a lag falls inside an abnormal period, and drop the abnormal half-hours from the
training target, so the model is never asked to predict an abnormal arrangement and learns the
normal-arrangement quantity NGED plan against. The nearest published precedent for the first half is
[Liu et al. (2019)](https://doi.org/10.1109/ACCESS.2019.2951422), who forecast the load of each
parallel transformer inside a substation by fitting a separate regression per substation operating
condition, because "the irregular load data are too scarce to establish an ANN-based model under
various irregular conditions" — but their switching moves load between transformers inside one
substation, so the substation total stays metered throughout and the never-metered-target problem
does not arise for them. The second half has a canonical statement outside energy: [Salinas et al.
(2020)](https://doi.org/10.1016/j.ijforecast.2019.07.001) handle unobserved values in a
probabilistic forecaster by "replacing each unobserved value ... by a sample ... from the
conditional predictive distribution ... and excluding the likelihood term corresponding to the
missing observation", motivated by retail stock-outs, and they say they omitted the experiments for
it from the paper. Searching OpenAlex, Crossref, and arXiv for sample masking, zero sample weights,
gappy targets, and exclusion of anomalous periods, we found no load-forecasting study reporting what
dropping contaminated periods from the training target is worth, so Flexpectation will have to
measure that itself.

**Later research will go further and treat the normal-arrangement demand as a latent variable to be
inferred, rather than a series to be repaired first.** The route is a differentiable-physics model
of each substation, with separate photovoltaic, wind, and demand components, in which the
normal-arrangement demand is fitted as a latent quantity carrying its own uncertainty. Recovering a
demand the meter never saw is mature in fields where demand is censored: airline revenue management
calls it unconstraining, and retail and electric-vehicle-charging work calls it censored-demand
recovery, as in [Hüttel et al. (2023)](https://arxiv.org/abs/2301.06418), who model charging demand
where "the true demand is latent (unobserved), and the observations are censored". Estimating what a
curtailed wind farm would have produced is the closest analogue inside the energy sector. The
transfer is imperfect in a way worth stating plainly: censoring is one-sided, so the observed value
bounds the latent one from below, whereas an abnormal running arrangement substitutes a different
set of customers and can read either side of the normal-arrangement demand. Searching the same three
indexes for latent-demand, censored-demand, counterfactual, synthetic-control,
differentiable-physics, and physics-informed formulations applied to substation demand, we found no
published model that recovers a latent normal-running-arrangement demand for a distribution
substation.

### 6. Detecting faulty metering

**In summary.** Faulty metering is usually a data-cleaning step mentioned in passing rather than a
problem in its own right, the only public labelled dataset we found is Dutch, and recovering the
direction of flow from a magnitude-only meter was attempted by this network's predecessor, whose
automatic version is still open.

**The challenge.** NGED's telemetry carries stuck values that repeat unchanged for hours or days,
zeros that mean "no reading" rather than "no load", physically impossible values, and gaps running
from a single half-hour to several months. Ten of the 32 series in the trial area are metered in
apparent power only, so they report magnitude without direction and reverse flow appears as a rise:
at one primary substation the meter bounces off zero on sunny days, when a solar farm behind it
exports. A model trained on uncleaned data learns the fault, and a forecast that fails silently
because its recent history was stuck is worse than one that says it is degraded.

**The most useful published method treats faulty metering and switching as one challenge.** [Bouman
et al. (2024)](https://arxiv.org/abs/2405.16164) treat measurement errors and switch events as the
two things that must be filtered out before substation measurements can be used, detect both on the
same residual.. Their sign-recovery technique addresses exactly the non-directional metering defect
described above.

**One other group has made faulty metering its subject, one voltage level down.** [Moriano et al.
(2016)](https://doi.org/10.3390/s16010085) and [Martín et al.
(2018)](https://doi.org/10.3390/s18113947) detect systematic errors in secondary-substation
monitoring equipment by comparing each measurement against a short-term load forecast. Two things
limit how far the Moriano and Martín results carry: the errors are *injected* rather than found in
the wild, and the fault taxonomy is calibration gain and offset drift plus outliers, not the stuck
values, false zeros, and multi-month gaps that dominate NGED's telemetry.

**Three network-innovation projects in GB tackled faulty metering substantively, one of them as its
whole subject — Electricity North West's ATLAS, described under challenge 4 above, UK Power
Networks' Distribution Network Visibility, and this network's own Time Series Data Quality.** UK
Power Networks' [Distribution Network
Visibility](https://www.ofgem.gov.uk/sites/default/files/docs/2014/03/dnv_cdr_version_3.0_270214.pdf)
project (Low Carbon Networks Fund, reported December 2013) checked its remote terminal units against
physics rather than against a forecast. They ran it over 377 units and found that "95% were found to
obey the expected logic within 15 kVA, with 5% identified as probably having installation problems",
then put the check into a daily health report that ranks units for maintenance. A run of implausible
values is a fault to a forecaster and a real event to a control engineer, and only the purpose
settles which.

**This network's own predecessor ran the same investigation on the same telemetry, and its findings
should temper what we expect to find.** Western Power Distribution's [Time Series Data
Quality](https://smarter.energynetworks.org/projects/nia_wpd_011/) project (Network Innovation
Allowance, reported March 2017) checked SCADA analogues for zeros, for "non-varying non-zero values,
perhaps indicating a 'stuck' or incorrectly configured sensor", and for gaps, across all four
licence areas. It found that "13.8% of all analogues in the WPD South-West licence area are only
recording 0 values. (20.7% companywide)" — adding that "many of these may be valid open circuit
values, however some will reflect incorrect values" — that the share of PowerOn data points
unavailable to planners ran from 1% in the South West to 36% in the Midlands, and — the finding most
relevant to challenges 3 and 7 — that "63% of all new solar sites across the company have not had
their analogues commissioned correctly". Flexpectation should expect metering defects to be common
rather than exceptional, though none of these figures is a defect rate on its own.

**A public labelled dataset exists, and it is Dutch.** Knowing how often a detector is right
requires measurements labelled as faulty or not, and [Bouman et al.
(2024)](https://arxiv.org/abs/2405.16164) had 180 primary substations labelled at 15-minute
resolution — since released as "STORM onderstation" on the [open data
portal](https://www.liander.nl/over-ons/open-data) of Liander, Alliander's network operator,
explicitly so that others can train and validate algorithms against it. That is the one place in
this review where the evaluation data for a challenge is already public.

**Where the gaps are: the fault taxonomy, a measured GB detector, and a reference series to detect
against.** The Dutch labels collapse switching events and measurement errors into a single class, so
they cannot separate a stuck meter from a network reconfiguration — which is exactly the distinction
challenges 4 and 6 have to make between them — and nearly 4% of their timestamps are labelled as the
labeller being unsure — a figure we counted from the released dataset, because the paper does not
report it.  None of the three GB projects above reports how often its
checks are right, and none published its labels, so there is no GB number to compare a new detector
against.

**Recovering the direction of flow from a magnitude-only meter has been attempted in GB, by this
network's predecessor, and left unfinished.** [Time Series Data
Quality](https://smarter.energynetworks.org/projects/nia_wpd_011/) set out to "first detect then
assign directions to power flows where absent". What it reports achieving is more modest than that
objective — plotting every analogue made it "clear where (for example in cases of generation) the
directional sense of analogues was incorrectly set", and correction was explored "by (for example)
flipping the direction/sense of a suitable candidate feeder" where summed currents at a transformer
and along its feeders failed to reconcile by more than a threshold. The plotting and the flagging
were automated, but a candidate directional error was confirmed and corrected by an engineer,
feeding a rectification list rather than an end-to-end detector, and no accuracy is reported for it.
The automatable version of that objective is nine years old and still available. A GB labelled set,
with a taxonomy that separates metering faults from switching, is a gap this project can close
cheaply, because the trial area is small enough to label by hand.

### 7. Disaggregating unmetered solar and wind from a substation's net flow

**In summary.** Splitting generation out of a substation's net flow has been done where the
generation is metered or its capacity is read from a register. Inferring the capacity from the net
flow instead has also been done, but at low-voltage substations serving tens of customers rather
than at a primary. Uncertainty and a multi-day horizon each appear in this literature, but never
together.

**The challenge.** Rooftop panels and small turbines appear only as a dent in a substation's net
flow. Recovering both the half-hourly output of that unmetered generation and its installed
capacity, from the net flow alone, is what we call *disaggregation*. Disaggregation is a different
task from estimating how much of a *metered* generator's capacity is available today, which is
challenge 3. It is a stretch goal for the trial area and a requirement for the network-wide
scale-up.

**In the network-innovation projects that separate demand from generation, the generation is either
metered directly or its capacity is read from a register. Capacity is inferred from measurements in
the academic work below and in NIA_UKPN0104, but never then carried into a probabilistic multi-day
forecast.** Artificial Forecasting models gross demand and customer export independently at primary
substations, which is more than any paper here does, but customer export is metered, and the
generation baseline it forecasts against comes from Northern Powergrid's own installed-capacity
projection per substation, scaled down by the fraction of that capacity actually generated in 2021
and 2022. SSEN TRANSITION split net load into demand and generation, forecast the two separately,
and recombined them. Its rooftop solar is not metered — but neither is its capacity inferred. SSEN
gathered a list of Feed-In Tariff installations. Looking up a capacity in a subsidy register is the
step Flexpectation cannot take, because the register stopped being complete when the Feed-In Tariff
closed.

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
from the substation's measured total load, weather, geospatial position, and each site's known
renewable capacity, at 15-minute resolution — a root-mean-square error of 0.07 against 0.70 for a
default transfer-learning model, on a min-max-scaled target. The paper reads 0.07 as 7%, but does
not say what the scaling divides by, so the figure does not transfer to another dataset. They call
the method domain adaptation for zero-shot learning in sequence, or DAZLS.

**The two methods are relatives rather than rivals, which is the more useful thing to know.** [Teng
et al. (2023)](https://doi.org/10.1016/j.rser.2023.113662) benchmark DAZLS against "the energy
splitting model in the OpenSTEF software package" on the same data and report that DAZLS
"significantly outperforms" it — a comparison worth noting because two of the authors work at
Alliander, the operator that builds OpenSTEF, so it is a like-for-like statement from the team that
maintains both.

**Inferring installed photovoltaic capacity from a substation's own measurements has been
benchmarked properly, one voltage level below NGED's.** [Gouveia et al.
(2026)](https://doi.org/10.1016/j.ijepes.2026.111848) compare data-driven estimators against
model-based ones for the photovoltaic capacity aggregated behind a low-voltage substation, using
only the net load and irradiance series a network already holds, and no register at all. Their
substations serve 10 to 100 consumers, against the thousands behind a GB primary, and their two new
model-based methods exploit the linear relationship between net load and irradiance. Three results
carry across. The data-driven estimators were comparable to the model-based ones even when net load
and irradiance were both accurate, and "substantially outperform" them under noisy data, which is
the condition NGED's telemetry is in. Models trained on a Belgian dataset and applied unseen to
Pecan Street in the United States and AusGrid in Australia, with only approximate irradiance, stayed
under 5% mean absolute percentage error once the linear models were regularised, which is the
closest thing in this literature to evidence that a capacity estimator transfers between networks.
And they report sensitivity separately to photovoltaic penetration, number of consumers, measurement
noise, sampling rate, seasonality, and irradiance error — the sensitivity axes Flexpectation would
have to test on its own data. What they produce is a capacity figure rather than a forecast. **The
same group argues separately that the estimate is worth maintaining, not merely computing.**
[Gouveia et al. (2026b)](https://arxiv.org/abs/2604.13926) set out why aggregated capacity estimated
at low-voltage aggregation points is the practical route to knowing what is connected, given that
"limited observability, incomplete topology information, and restricted access to customer-level
data make it difficult to maintain accurate DER registries", and link that estimate to forecasting,
congestion management, flexibility quantification, and hosting-capacity assessment. That is a
preprint rather than a peer-reviewed paper, and it argues a case rather than measuring one. **GB
already has an operational forecast of unmetered generation, at national scale.** NESO publishes
[embedded wind and solar
forecasts](https://www.neso.energy/data-portal/embedded-wind-and-solar-forecasts) half-hourly, from
within-day to 14 days ahead, updated hourly — the same resolution and horizon Flexpectation
delivers. "Embedded" means precisely the generation this challenge is about: wind and solar sitting
on the distribution network with no transmission metering, which NESO's own field definition
describes as "invisible to the National Energy System Operator (NESO)". The forecast is a single
number per half-hour, with no uncertainty attached, and it covers GB as one region rather than
substation by substation.

**The model behind the first six hours of NESO's operational solar forecast was built by Open
Climate Fix, which is also Flexpectation's delivery partner and employs this review's authors, so we
have an interest to declare here.** NESO's [Solar
NowCasting](https://www.neso.energy/news/solar-nowcasting-innovation-project-improves-solar-forecasting)
project, run with Open Climate Fix under the Network Innovation Allowance, set out to forecast
exactly the generation challenge 7 is about. NESO reports that the forecast the project produced
"was 2.8 times better than our previous Photo Voltaic (PV) forecast (for forecasts up to two hours
ahead)", and that the first fully operational service reached its control room in December 2022. The
project's [own record on the Smarter Networks
Portal](https://smarter.energynetworks.org/projects/nia2_ngeso002/) reports "Accuracy improvement
over the previous model by approximately 30% for the GSP and National forecasts (4-8 hours)" and
lists "Probabilistic forecasts for all horizons" among its outcomes. And the resilience that took
the service to over 99.5% availability was implemented by NESO rather than by Open Climate Fix,
"with all the infrastructure constructed in code to allow replicability". So the combination this
review says is missing — unmetered generation, forecast probabilistically, at a spatial level below
the country — has been built once in GB, at grid supply point level rather than at primary
substations, and for solar rather than for net demand.

**The model itself is published and open source, which is unusual in this literature.**
[PVNet](https://github.com/openclimatefix/PVNet) is released under the MIT licence and described by
its authors as "a multi-modal late-fusion model for predicting renewable energy generation from
weather data", combining numerical weather prediction with satellite imagery, recent generation, and
the sun's position. The accompanying paper, [Fulton et al.
(2024)](https://www.climatechange.ai/papers/iclr2024/46), makes "0-8 hour lead time forecasts for
grid regions across Great Britain" and limits the model's inputs "to be reflective of those
available in a live production system" — a workshop paper rather than a peer-reviewed one, which we
flag because the rest of this review holds its sources to that standard, and one of its authors is
an author of this review.

**What Flexpectation takes from PVNet is a working precedent and a warning about PVNet's limits.**
 The limits
are that PVNet forecasts generation at grid supply points where the generation is the whole signal,
whereas Flexpectation must separate unmetered generation from demand inside a single net-flow
measurement at a primary substation; that its horizon is hours rather than the 14 days NGED needs;
and that the grid supply point regions it forecasts are far larger than a primary substation, so its
accuracy figures say nothing about how the same approach would perform at Flexpectation's scale.

**Uncertainty and the horizon both exist in this literature, but never together.** [Wang et al.
(2018)](https://doi.org/10.1109/TPWRS.2017.2762599) run the whole pipeline this challenge describes
— estimate behind-the-meter photovoltaic capacity, then decompose net load into generation, demand,
and a residual, forecast each, and recombine them with a copula that models how the three forecast
errors depend on each other rather than assuming they are independent — but at ISO New England
scale, and the abstract does not give the forecast horizon. [Zhang et al.
(2022)](https://doi.org/10.1016/j.engappai.2022.104707) do probabilistic disaggregation at grid
supply point and feeder level with a multi-quantile recurrent network, scored on reliability and
sharpness. NESO covers the 14-day horizon but deterministically. [Faustine et al.
(2025)](https://doi.org/10.1109/TPWRS.2024.3400123) forecast net load probabilistically at
low-voltage substations with solar behind them, by quantile regression on a feed-forward network, ;
the abstract does not state the horizon. [Erdener et al.
(2022)](https://doi.org/10.1016/j.rser.2022.112224) survey the field. We read the full paper for
[Zhang et al. (2022)](https://doi.org/10.1016/j.engappai.2022.104707) and the abstracts for the
other three.

**Where the gaps are: doing it without a metered training set, inferring the capacity rather than
being told it, and putting uncertainty and a multi-day horizon in the same forecast at substation
level.** [Teng et al. (2023)](https://doi.org/10.1016/j.rser.2023.113662) need a population of
fully-metered substations to transfer from, and are given the existence and capacity of each
renewable facility rather than inferring it — whereas inferring that capacity is half of what NGED
needs. Their output is a near-real-time estimate rather than a forecast.

### 8. Disaggregating other distributed energy resources: heat pumps, electric-vehicle chargers, and batteries

**In summary.** This is the largest gap in the review and the largest deliberate omission from our
search: in the one study we found that measures charger forecast skill against aggregation, only the
site with more than a hundred charge points was significantly better than a naive benchmark, though
some models at one much smaller site also beat it, heat-pump diversity is untested in the cold
weather that matters, and no diversity factor helps for domestic batteries at all.

**The challenge.** Heat pumps, electric-vehicle chargers, and price-sensitive domestic batteries
change the shape of a substation's load in ways a model trained on history cannot anticipate,
because the number of them behind any given substation is growing quickly. The stretch goal is to
disaggregate and forecast them separately rather than letting them sit inside net demand.

**Detecting heat pumps, chargers, and batteries and forecasting them are separately hard, and not in
the order we expected.** Northern Powergrid's [smart-meter detection
trial](https://smarter.energynetworks.org/projects/npg_nia_-49/), on 1,500 monitored premises, found
— using hand-built heuristics rather than trained models, and with no register of installations to
validate against — that "EV identification at premises level was found to be relatively
straightforward" and that "aggregation does mask some signals, although EV usage is still clearly
identifiable at feeder and substation level", while "the detection of ASHP [air-source heat pumps]
is frustrated by the low levels of adoption". So the spiky, synchronised charging that makes
electric vehicles hard to *forecast* is what makes them easy to *detect* in aggregate; heat pumps
are the reverse.

**Errors across many chargers cancel rather than compound, and the measurement is NGED's own.** The
[Electric Nation
trial](https://eatechnology.com/media/girhcnsc/electric-nation-customer-trial-report.pdf) — run by
this network under its former name, with 673 participants and over 130,000 charging events — fits
the demand of a group of chargers as `Group Demand = N·P + Q√N`, where P is the mean demand per
charger and Q the deviation. The mean scales with the number of chargers and the deviation only with
its square root, so relative uncertainty falls as more chargers are added. [Bollerslev et al.
(2022)](https://doi.org/10.1109/TTE.2021.3088275) simulate Danish driving and plug-in behaviour on
synthetic feeders and fit the exponent at between 0.42 and 0.51 across battery sizes and plug-in
behaviours at an 11 kW charger, against the 0.5 that complete independence would give.

**What makes electric-vehicle charging the harder network problem is when it lands, and that an
automated tariff can re-synchronise a population that had diversified.** In Electric Nation's third
trial, with a time-of-use tariff, the share of charging events starting in the 22:00 hour rose from
5.8% without the tariff to 24.7% with it — and to 37.6% among participants using the smart-charging
app, against 5.5% for those on the same tariff who did not use the app.

**Heat pumps diversify in an average winter, but whether that diversity survives the cold weather
that actually matters is untested.** [Love et al.
(2017)](https://doi.org/10.1016/j.apenergy.2017.07.026) measured around 700 domestic heat pumps in
GB and found demand per heat pump falling from 4.0 kW for a single unit to 1.7 kW once 275 are
aggregated, with the spread between samples falling from 1.5 kW to 0.1 kW. But a heat pump sized
small relative to a house's heat demand runs flat out for hours in cold weather, and Northern
Powergrid's [code of practice for the economic development of the low-voltage
system](https://www.northernpowergrid.com/sites/default/files/assets/IMP001911_0.pdf) notes in a
footnote that "further research is required to examine whether the increase in duty cycle (and hence
average demand) with lower than average winter ambient temperatures is material when designing a LV
system". That is precisely the condition under which a substation approaches its limit.

**No diversity factor helps for domestic batteries, and the industry agrees.** That same [code of
practice](https://www.northernpowergrid.com/sites/default/files/assets/IMP001911_0.pdf) fits
diversity curves to measured trial data for general domestic load, heat pumps, and chargers alike,
and then states that diversity "should not be applied when considering a BESS device" — a battery
energy storage system — a diversity factor of exactly one.

**The electrification literature remains the largest deliberate omission in this review.** Our
search covered substation and generation forecasting, not electrification, and the paragraphs above
are what a targeted follow-up search surfaced rather than a proper review. The volume of work is
easy to demonstrate: of the 305 papers accepted for the Brussels workshop of June 2026 held by
CIRED, the International Conference on Electricity Distribution, 28 have a title naming electric
vehicles, chargers, heat pumps, or batteries — more than the 23 whose titles name forecasting or
prediction at all.

**Where the gaps are: forecast skill at substation aggregation, and the tariff-driven peak.** The
one direct measurement of charging forecast skill against aggregation that we found is [Ostermann
and Haug (2024)](https://doi.org/10.1186/s42162-024-00319-1), who forecast aggregated charging
demand a day ahead at 15-minute resolution from "a large and novel dataset of over 350,000 charging
processes at more than 500 locations across Germany". The aggregation threshold below is a
day-ahead result, and forecast uncertainty grows with lead time, so at the 14 days NGED needs the
number of charge points required before a forecast beats a naive one should be expected to be higher
than the figure quoted here rather than the same. Aggregation is what decides whether the forecast
is worth having: "almost all models have values above 1 for the MASE and nRMSE for both the
individual sites and the zip codes, which means that the benchmark model is better in some cases",
and of their five example sites only the one with over 100 charging points was "significantly better
than those of the naive model". Nothing we found forecasts an aggregate of heat pumps, chargers, and
batteries behind a GB primary substation, states its own uncertainty, and is scored against the
evening peak that the network actually cares about. Nothing we found tests whether the
re-synchronised peak an automated tariff creates, described above, survives at the aggregation a
primary substation carries. Reading the electrification literature properly is the first deliverable
on this strand, before any model.

## How we will know whether each of these worked

The eight challenges above need three different kinds of evaluation, and this literature is far
stronger on the first than on the other two. Forecasting has settled practice we can adopt.
Estimating something nobody measures — an effective capacity, an unmetered solar output — has six
possible substitutes for ground truth, of which this literature uses four. Detecting rare events has
good academic practice and, in GB, no precedent that measured anything at all. This section says
what we will do about each, and it is placed before the recurring findings because the answer
changed how we intend to run the project rather than only how we intend to report it.

### Forecasting: challenges 1, 2, and 5

**Every forecasting paper we read that describes its split keeps most training data out of the
future of its test data, with one exception, and the training window usually grows rather than
slides.**

**One length rule is worth adopting outright.** [Pinheiro et al.
(2023)](https://doi.org/10.1016/j.apenergy.2022.120493) held out the whole of 2019 and note that
"one year is the minimum acceptable to test a forecasting model whose target value shows annual
seasonality". Substation load shows exactly that seasonality, so any fold shorter than a year cannot
tell us whether a model handles winter, and winter is when NGED buys flexibility.

**Not one of the papers we read addresses the leakage a frequently reissued forecast creates, and
Flexpectation is the most exposed design of the lot.** When a forecast covering 14 days is reissued
every six hours, every target half-hour is covered by 56 separate forecasts. Those 56 are not 56
independent observations of the model's skill: they share the weather, the recent load, and most of
the model state. Count them as independent and a significance test will report a confidence the data
does not support; let a target half-hour fall on both sides of a train-test boundary and the test
set is contaminated outright. We searched every forecasting paper in this review that reissues a
forecast more often than its horizon is long, for any treatment of the overlap that reissuing
creates — a gap or buffer between training and test, a block bootstrap, a correction to the number
of independent observations — and found one partial treatment. [Hertel et al.
(2026)](https://arxiv.org/abs/2607.15705) compare models with Diebold-Mariano tests implemented
after the R `forecast` package, whose variance estimator corrects for serial correlation in the loss
differential when it is told the forecast horizon; the paper does not say which horizon it passes,
so we cannot tell whether the correction was applied. No paper we read treats the train-test
contamination that reissuing creates. [Kaas et al. (2026)](https://arxiv.org/abs/2607.01966) are the
one paper the problem cannot reach, because their stride equals their horizon, four days each, so no
two of their forecasts share a target; that is our inference from their design rather than a claim
they make, and they give a different reason for wanting a shorter stride, that it would "provide
more insights", while describing exactly what it would create — "each data point in the dataset
covered by multiple forecasts, as opposed to a single forecast per data point in the used
configuration".

**Flexpectation's own protocol matches this literature where the literature has settled, and
inherits the open question where it has not.** We use an expanding training window with the
validation window lying strictly after it, which is what most of the papers above do — Pinheiro et
al. slide a fixed three-year window rather than expanding it, and Gilbert et al. interleave their
test blocks with training data from later in the same year. Our validation window is a complete
year, which meets [Pinheiro et al. (2023)](https://doi.org/10.1016/j.apenergy.2022.120493)'s minimum
quoted above. Power lag features shorter than the lead time are nullified, so a forecast can never
see the load it is predicting. What none of that settles is the overlap question: our forecasts are
reissued every six hours over a 14-day horizon, so the same target half-hour is scored 56 times, and
we do not yet know how much that inflates the apparent precision of a comparison between two models.
We will report what we did about it rather than leave it implicit, and we treat it as an open
methodological question rather than a solved one.

### Estimating something nobody measures: challenges 3, 7, and 8

**There is no ground truth for an effective capacity or an unmetered solar output, and the papers
that estimate them say so.**  This literature uses four substitutes for truth, each of which fails
differently, and leaves two more on the table.

**The first substitute is to hold out sites that are metered and pretend they are not.** This is the
most direct substitute available, because the answer really is known, and Flexpectation can run that
substitute: the trial area has 12 metered generators whose output can be hidden from a model that
then has to recover it.

**The second substitute is to inject a change into real data and see whether the method recovers
that change.**

**The third substitute is to compare against an independent tool rather than against truth.**

**The fourth substitute is indirect: measure whether the capacity estimate improves the forecast it
was built to improve.**

**Two further substitutes barely appear in this literature at all, and both are worth having.** The
first is to check an estimate against physics rather than against an answer. Disaggregated
components must sum to the measured net flow; disaggregated solar must be zero at night and must sit
under the clear-sky envelope; disaggregated wind must track wind speed rather than irradiance; an
inferred rooftop-solar capacity must be plausible for the area a substation serves. None of those
checks needs a label, and a violation is a detectable error whatever the truth turns out to be.
Using physical consistency to *score* an estimate, rather than to *shape* it, is close to absent
here, and it is the cheapest evaluation on this list. The second is a substation where every feeder
and every embedded generator is metered for a period, used only as validation. One fully-metered
substation would anchor everything else, and none of the papers above had one.

**Flexpectation will run all six substitutes and treat agreement between the six as the signal,
because no one substitute is trustworthy alone.** They are not six attempts at the same measurement:
the hold-out is biased towards the sites that happen to be metered; synthetic aggregation
systematically flatters, because a clean sum of metered sources has no switching events, no false
zeros, and no unmetered load, so it should always be reported as performance under idealised
aggregation rather than as real-world skill; the independent-tool comparison says only whether we
agree with an existing method; the physics checks find wrongness but never confirm rightness; and
the downstream test measures whether the estimate is *useful*, which is not the same as whether it
is *right* — an estimate that is wrong in a way the forecast does not care about will score well.
For the metered generators of challenge 3 the same logic produces a head-to-head contest between
candidate estimators in which downstream forecast skill decides, with synthetic fault injection,
precision and recall of the fitted change dates against NGED's maintenance records, robustness to
unlogged curtailment, and the calibration of each estimator's stated uncertainty as the supporting
evidence. Every number we publish will say which of the six substitutes produced it.

### Detecting rare events: challenges 4 and 6

**Detection needs different metrics, and the best-worked example in this review chose them
deliberately.** [Bouman et al. (2024)](https://arxiv.org/abs/2405.16164) score precision, recall,
and an F-score with β set to 1.5 rather than 1, "to give a higher importance to the recall term, as
the potential impact of a false negative is higher than that of a false positive in power grid
expansion planning". That asymmetry holds for Flexpectation too: a missed switching event silently
corrupts the history a model trains on, whereas a false alarm costs an engineer a look.

**Class imbalance is the design problem, not a detail.** Score by timestamp and the long events
decide the number; score by event and the short ones do. [Bouman et al.
(2024)](https://arxiv.org/abs/2405.16164)'s answer is to split events into four duration bands,
score each band separately and average, and to exclude the timestamps a labeller marked uncertain
from the scoring entirely. They set the threshold by maximising that averaged score rather than by
the conventional two- or three-standard-deviation control limit, and they resample the test stations
10,000 times to put an uncertainty on the result.

**Two other choices in this literature are worth copying.** [Perry and Muller
(2022)](https://doi.org/10.1109/PVSC48317.2022.9938675), detecting step changes across 101 manually
labelled photovoltaic power and irradiance streams, score a detection as correct if it lands "within
30 days of their labelled occurrence", which is the right shape for a problem where the exact
timestamp of a gradual shift is not knowable; the tolerance has to be stated, because the score
means nothing without it. [Martín et al. (2018)](https://doi.org/10.3390/s18113947) set their
detection threshold from instrument physics rather than from the data: transformers contribute up to
±1% error and the measurement equipment ±0.5% to ±1%, so ±2% is the inherent floor, and they set the
threshold at ±4% "to avoid detection of false gain and offset errors". A threshold derived that way
can be defended to an engineer who asks why their substation was flagged.

**The honest headline from the one paper that measured properly is that detecting switching and
metering faults is hard.** [Bouman et al. (2024)](https://arxiv.org/abs/2405.16164) report F-scores
near 0.2 on the shortest events and around 0.5 on the longest, and conclude that performance "is
relatively low across the board, even on the train data. This indicates that the problem is hard to
learn, though it generalizes fairly well". Any target we set for challenges 4 and 6 should start
from those F-scores rather than from an intuition about how obvious a switching event looks on a
chart.

**None of the three GB projects we checked offers a number to compare against, which we checked
rather than assumed.** Across Electricity North West's ATLAS — both its 2016 methodology and its
2018 closedown report — UK Power Networks' Distribution Network Visibility and this network's own
Time Series Data Quality, the words precision, recall, F-score, true positive, and false positive do
not appear at all.  Publishing precision and recall against a
stated label set, with the labels released, would therefore be the first time we know of that a GB
network has done so, and it is the cheapest of this review's commitments to keep.

## What published leaderboards did, and what a single team can borrow from them

Building leaderboards is one of Flexpectation's deliverables, so the design of a leaderboard is
itself a question the literature can be asked about. "How we will know whether each of these
worked", above, settles how a single forecast is scored; this section is about the other half — how
results from many experiments are put side by side so that the comparison means something.

**What Flexpectation is building is a leaderboard, not a competition, and the distinction changes
which published lessons apply.** Our leaderboards carry our own experiments — one per class of time
series, so solar farms, wind farms, batteries, and the demand at primary substations each get their
own, with grid and bulk supply points sharing a board because their measurements are the same kind
of thing — with every row a model, a feature set, and a processing choice scored on the same test
data with the same metrics. They will be public to view and reproducible, but we are not inviting
other teams to submit entries. Anyone who wants to benchmark against us can rerun the setup for
themselves. That means the published lessons about attracting entrants, prize pots, and qualifying
rounds do not apply to us, while the lessons about protocol — what makes a comparison trustworthy —
apply with more force, because a competition gets some of its integrity free from having rivals who
would like to catch each other out, and we will not have any.

**Energy forecasting has run competitions on common data for over a decade, and only one of them
forecast at anything like the level NGED acts on.** The Global Energy Forecasting Competitions of
2012, 2014, and 2017 covered hierarchical load, price, wind, and solar, published their data as
supplementary files to the papers introducing each competition, and drew hundreds of contestants
from more than 60 countries ([Hong et al. (2020)](https://doi.org/10.1109/OAJPE.2020.3029979)).
[Shukla and Hong (2024)](https://doi.org/10.1049/stg2.12162)'s BigDEAL Challenge 2022 was themed on
peak timing, and its final match asked for the magnitude, timing, and shape of daily peak load at
three neighbouring local distribution companies. [Browell et al.
(2025)](https://doi.org/10.1016/j.ijforecast.2025.10.005)'s HEFTCom forecast one 3.6 GW hybrid
portfolio day-ahead. [Kleinebrahm et al. (2026)](https://arxiv.org/abs/2604.24705)'s Energy-Arena
and [Meyer et al. (2026)](https://arxiv.org/abs/2512.20761)'s TS-Arena keep permanent leaderboards
open to new entries, and run continuously rather than closing. The closest of these to a
distribution network is the second track of GEFCom2017, which asked for probabilistic forecasts of
183 delivery-point meters of a US utility and drew 177 entrants in total across both its tracks
([Hyndman (2020)](https://doi.org/10.1016/j.ijforecast.2019.03.015)). BigDEAL's three local
distribution companies are whole utilities, an aggregation well above a single primary substation.
Competitions have been run on distribution-network problems, and NGED funded three of them.
[McSweeney et al. (2023)](https://doi.org/10.1109/ISGTEUROPE56780.2023.10407541) report a series
NGED ran with Energy Systems Catapult on its own network data, hosted on CodaLab, drawing "37 teams
from both academia and industry" and "over 2500 submissions". None of the three was a load forecast.
The first asked for the highest and lowest one-minute values inside each half-hour given only
half-hourly averages at a substation; the second for the daily maximum demand that a hidden
population of electric-vehicle chargers added to three primary-substation feeders; the third for
missing values across a hierarchy of primary substations, their bulk supply points, and a grid
supply point. Two of the three therefore sat at exactly the levels NGED forecasts, which is why the
absence claim below is scoped to forecasting rather than to the voltage level. The paper draws the
same conclusion this review does, that "many solutions are only tested on private data using a
single method only compared (if at all) to simple, non-competitive benchmarks", which "limits the
reproducibility and usefulness of the outputs", and pairs its own results with the caveat that they
came "despite the necessary reduction in realism" of a curated competition dataset. All three
competitions closed between December 2021 and April 2022, though their pages and data are still
readable; the paper never uses the word leaderboard, and what it recommends keeping open is the
unranked practice phase, "as it allows teams to continue experimenting within the platform". What we
found no example of is a *standing* leaderboard at distribution-substation level — one that keeps
accepting new entries after its competition closes. That is the gap Flexpectation's leaderboards
fall into, though the search behind that statement is ours and we would be glad to be pointed at a
counter-example.

**The closest published precedent is a platform whose leaderboard, at the last snapshot its authors
published, was populated entirely by models its own operators run.** That is not quite our position,
because TS-Arena does invite outside entries, but it is near enough that its self-imposed rules
transfer. It evaluates thirteen reference foundation models and three statistical baselines across
186 live energy series, all of them run by the platform team. What keeps it honest is a set of rules
the operators impose on themselves. Their reference models "act as neutral participants,
autonomously requesting context from the API Portal and submitting forecasts to it", so that they
"operate under the exact same constraints (e.g., submission windows, data access) as other
(external) participants". They run each foundation model from its authors' own repository at its
authors' recommended defaults, with no domain-specific tuning. All three of those are available to a
single team, and we intend to adopt them: our own models go through the same evaluation interface as
any baseline, and a baseline is run as its authors published it.

**The mechanism that makes a leaderboard trustworthy is time, not policing.** TS-Arena's central
idea is that a forecast is submitted before the outturn it will be scored against physically exists,
which "makes test-set contamination impossible by design". HEFTCom made the same argument from
experience: because the competition ran on the real, unknown future, "data leakage, accidental or
deliberate, was impossible". The corollary is uncomfortable for anyone relying on a fixed hold-out
set, and TS-Arena states it plainly: "leveraging any fixed dataset that is not evolving over time
and directed into the future — regardless of how carefully curated — can eventually lead to
information leakage". [Hong et al. (2020)](https://doi.org/10.1109/OAJPE.2020.3029979) name the same
failure from the other end, that "some datasets have been studied so well that the researchers may
use some of the future information to give unfair advantage of their proposed methods". A
half-hourly forecasting service is unusually well placed here: every day supplies 48 fresh
evaluation points that can never be reused, and the condition that the answer did not exist when the
model was frozen holds automatically.

**The specific way a single team fools itself is not fabrication but running the baseline badly.**
[Kleinebrahm et al. (2026)](https://arxiv.org/abs/2604.24705) put it as a general problem with
published comparisons: competing methods "are not always implemented or optimized with equal care",
so reported differences "may reflect differences in implementation quality rather than inherent
methodological advantages". [Hong et al. (2020)](https://doi.org/10.1109/OAJPE.2020.3029979) put it
more bluntly, that "sometimes the parameters are manipulated, so that the competing models are being
dominated by the proposed ones", alongside two related habits — picking the error measure that
favours the proposed method, and skipping comparison with naive models altogether. A team that runs
every entry on its own leaderboard is exposed to all three by construction, which is why the
author's-code-and-author's-defaults rule matters more for us than it does for a competition.

**Run two baselines that bracket the answer, not one.** [Doubleday et al.
(2020)](https://doi.org/10.1016/j.solener.2020.05.051) distinguish the two jobs a benchmark does: a
yardstick, which "should be consistent, accessible, and easily reproducible, though it does not
necessarily need be considered a 'good' forecast", and a point on the yardstick, which "should be
close to the state of the art". They recommend carrying both, so that a new method can be positioned
between them rather than merely declared better than something. That is the shape our leaderboards
take: persistence and climatology as the naive yardstick, and NGED's incumbent method as the point
on the yardstick a new model has to reach.

**The submission deadline, not a rule about features, is what defines a fair information set.**
[Kleinebrahm et al. (2026)](https://arxiv.org/abs/2604.24705) give a worked example of the trap:
several published papers use the day-ahead wind and solar forecasts that the European Network of
Transmission System Operators for Electricity publishes as inputs to day-ahead price models, but
those forecasts are "released only after 18:00 on the day before delivery, whereas the day-ahead
market already closes at 12:00 on that day". The feature did not exist when the forecast had to be
made. Their fix is structural rather than procedural — each competition "implicitly defines an
operational information set through the submission deadline". Flexpectation has the same hazard in
the delay between an ECMWF run and its arrival, and the same fix is available: score against the
data that had actually landed at the forecast's issue time.

**A leaderboard wears out through repeated use, and the published remedies are all forms of
rationing.** [Hyndman (2020)](https://doi.org/10.1016/j.ijforecast.2019.03.015), who co-organised a
forecasting competition himself, expects it: "over-study of a single benchmark data set means that
methods will eventually over-fit the published test data. I suspect this has happened with the M3
data over the past 20 years, and it is likely to happen with the M4 data, despite its much larger
size. Therefore, a wider range of benchmarks is desirable, and these need to be updated regularly.
Consequently, there can never be a 'final forecasting competition'." The remedies in use are crude
and effective, and all of them ration how often a score can be seen.

**The empirical evidence on how often leaderboards actually get overfitted is more reassuring than
the theory, and it does not cover our case.** What reassurance there is comes from competitions with
many independent entrants and a private split held back until the end, and the exceptions in it are
benchmarks with effectively small test sets. Our fold is small in effective sample size rather than
in row count, because consecutive half-hours are strongly correlated, which shrinks the evidence the
same way.

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
decides what a leaderboard should report as its headline. [Recht et al.
(2019)](https://arxiv.org/abs/1902.10811) found the ordering of models preserved on a freshly
collected test set while the accuracy level moved by "approximately five years of progress in a
highly active period of machine learning research". [Fildes
(2020)](https://doi.org/10.1016/j.ijforecast.2019.04.012), reviewing the M4 competition, compared
its daily micro series against a real retail forecasting problem and found the same method scoring
1.665% on one and 11.1% on the other. His conclusion is a direct endorsement of what Flexpectation
is doing: "each organization needs to organize its own forecasting competition for its own
forecasting problems, and should not rely on even large benchmark data sets", with the published
competition useful for narrowing "the pool of methods to be considered" rather than for predicting
your own error. So a leaderboard should lead with ranks and with margins over a stated baseline, and
treat an absolute skill number as valid only on the distribution it was measured on.

**A finite evaluation window can rank the wrong model first, and several months is not obviously
enough.** [Messner et al. (2020)](https://doi.org/10.1002/we.2497) demonstrate this rather than
asserting it. They fit three forecasting models with three different loss functions, so that each is
optimal for one metric by construction, and then score them on the first 200 time steps: the model
built for the quadratic loss now wins all three metrics, and the two built to win on mean absolute
error and on the quantile score both lose on their own. Their conclusion is the sharpest warning we
found about reading a leaderboard: "evaluation results based on a finite data set are always subject
to some degree of uncertainty and the best ranked forecast does not necessarily have to be the truly
best one. Depending on the actual setup, e.g., in a benchmarking exercise to hire a forecaster, it
should be remembered that even periods of several months may still yield uncertainty in terms of who
the best forecaster truly is." HEFTCom's own competition period was three months. The practical
response, which TS-Arena adopts, is to publish an interval on the ranking rather than the ranking
alone, so that a new entry near the top is visibly provisional. Their interval comes from replaying
the round order in random permutations, so it widens for models with few rounds rather than
measuring sampling error over a finite window; they warn against "treating short-term success as
proven superiority", and note that the confidence intervals of their own top models overlap.

**The AlphaFold comparison is worth stating precisely, because the precise version supports this
project better than the loose one.** CASP, the assessment that AlphaFold won, is a recurring
competition rather than a standing benchmark: every two years its organisers gather proteins "for
which the experimental structure is about to be solved or is solved but still not public" and give
the sequences to entrants ([Kryshtafovych et al. (2021)](https://doi.org/10.1002/prot.26237)). Each
target is single-use, because once the structure is published nobody can be blind to it again. The
standing benchmark in that field is a different thing, CAMEO, which takes the weekly pre-release of
forthcoming structures as its targets. AlphaFold2 was developed against neither, but against a
temporal hold-out of its own — trained to a fixed cut-off and scored on structures released after
it, which is what a live forecasting service gets for free. The blind competition was the audit; the
temporal hold-out was a check the team could run for itself, on data no rival had to supply. That is
the same division of labour Flexpectation is proposing, and it is the reason a leaderboard without
entrants is a coherent thing to build.

**A single-team leaderboard cannot buy credibility from rivals, so it has to buy it by declaring its
own gaps.** Flexpectation's is known in advance: the leaderboard runs on the 32 trial-area series
and the service is meant to reach the whole of NGED's network, so we should expect our published
numbers to flatter what happens at scale, and should say so each time we publish them.

**Two things follow from how long those benchmarks took to produce a step change.** A leaderboard's
first product is usually a credible measured plateau rather than a breakthrough, and in CASP's case
a fourteen-year plateau is what made the later jump believable ([Kryshtafovych et al.
(2021)](https://doi.org/10.1002/prot.26237)). And a benchmark of 32 series is small enough that the
constraint on what can be learned from it is likely to be its size, which is an argument for
extending it to the wider network as soon as the data allows rather than for running more
experiments against the trial area.

**What a leaderboard without entrants cannot do, we should not claim it does.** Three of the
strongest results in the benchmarks above are unavailable to us. CASP's finding that its field
plateaued for fourteen years is a statement about protein structure prediction only because dozens
of groups were trying independently; a plateau on our leaderboard would be ambiguous between a hard
problem and a team that did not think of the right idea. The M competitions' conclusions about whole
classes of method — that complex methods do not typically beat simpler ones, that combining methods
beats the methods combined ([Hyndman (2020)](https://doi.org/10.1016/j.ijforecast.2019.03.015)) —
describe what many independent people chose to try, and no single team's leaderboard can support
that kind of claim. And the reassurance about adaptive overfitting comes from competitions with at
least a thousand submissions each, entered independently against a private split held back until the
end — neither of which our leaderboard has. What our leaderboard can do is narrower and still worth
having: show which approaches beat a stated baseline on NGED's own data, under one protocol, with
the forecasts, the metric definitions, and the code published so that anyone can check the
arithmetic or rerun the comparison themselves.

## Six findings that recur across the studies we read

Six findings recur across the energy-forecasting studies reviewed under the eight challenges above.
These are findings about this literature, not laws of nature: each is what several teams measured on
their own networks, and a network that differs from theirs may well behave differently.

### 1. In the load-forecasting studies we read, each further step up in model sophistication bought a much smaller margin than the effort put into it would suggest

[Pinheiro et al. (2023)](https://doi.org/10.1016/j.apenergy.2022.120493), running a live system
drawing on 96,989 Portuguese secondary substations and fitting per-asset models at 84,663 of them,
tuned a system-level gradient-boosted tree by exhaustive grid search. At system level, the
gradient-boosted tree scored 199 MW root-mean-square error and the generalised additive model scored
191 MW, so the gradient-boosted tree was 4% worse than the simpler model. [Pinheiro et al.
(2023)](https://doi.org/10.1016/j.apenergy.2022.120493) concluded there was no accuracy gain to be
had and rejected the gradient-boosted tree on the cost of tuning it and on the loss of
interpretability, keeping the generalised additive model. Artificial Forecasting also found that
gradient-boosted trees did not clearly beat a simpler model, when forecasting customer export at
primary substations. Compared against the Bayesian ridge regression they went on to adopt, boosted
trees "helped some substations but harmed others", so they kept the Bayesian ridge regression as
their default. Northern Powergrid's deliverable gives no magnitudes and no significance test in
either direction.

When Artificial Forecasting tested a neural network against a four-week-average baseline at 729
secondary substations, the neural network lost on five of six metrics at the 24 substations with the
worst data quality. The margin was small, and data quality and the choice of metric mattered at
least as much as model complexity. [Hertel et al. (2026)](https://arxiv.org/abs/2607.15705) make the
same point from the other end of the sophistication scale: their purpose-built Transformer variant
lost to a standard encoder-decoder Transformer on all three of their datasets, and they conclude
that architectural modification is "not needed and can even lead to worse accuracy" because it
enlarges the space of hyperparameters that then has to be searched. What did help them was refitting
the model every month rather than redesigning it.

**"XGBoost" in these papers is a lighter model than the one Flexpectation plans, so read a loss by a
boosted tree with that in mind.** [Kaas et al. (2026)](https://arxiv.org/abs/2607.01966) give theirs
lagged power, weather, time, and metadata covariates and nothing beyond that: no clear-sky index, no
photovoltaic power proxy, no wind power curve, no monotone constraints, and holidays only as a
binary flag — on feeders whose target is net load with heavy solar feed-in. Their headline is better
read as a foundation model beating a lightly-featurised gradient booster than as a verdict on
gradient boosting. [Pinheiro et al. (2023)](https://doi.org/10.1016/j.apenergy.2022.120493) are the
exception, and the more uncomfortable result for us: their booster and their generalised additive
model were fitted on identical features, and the simpler model still won. Even there the shared
feature set was short — a linear trend, load lagged 24 hours and one week, time of day, nine day
types, the named public holidays, day of year, and temperature interacted with time of day and with
day of year, with temperature unfolded to its first three powers for the booster — and it carried no
irradiance and no wind, though the six weather fields they downloaded held both.

### 2. In every study that forecast more than one voltage level, accuracy got worse further down the network

[Hertel et al. (2026)](https://arxiv.org/abs/2607.15705) ran the same models against a day-type
persistence baseline on three datasets — a German transmission control area, 200 German low-voltage
feeders, and 287 individual Portuguese clients — and the margin over that baseline shrank from 59.6%
to 42.3% to 23.3% as aggregation fell. What shrank is the headroom above a naive rule rather than
the accuracy itself, which is the more useful reading: their own gloss is that it is easier to beat
a simple approach on highly aggregated data than on volatile feeder- and client-level data. [Gilbert
et al. (2023)](https://arxiv.org/abs/2206.11745) are a partial exception: they agree the forecast
gets less certain further down, but on their held-out test data the skill scores came out much
closer between levels than in cross-validation, and household peak-intensity forecasts were as
skilful against a benchmark as the aggregate ones. [Pfeifer et al.
(2021)](https://doi.org/10.1049/icp.2021.2177) measured the same thing separately for wind power,
solar power, and load across a medium-voltage grid region, and report that forecasts get worse both
at lower levels of aggregation and at longer horizons; we read their abstract rather than the full
paper. The model did not get worse; the problem got harder.

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
that mean absolute error was the wrong measure for peaks. [Pinheiro et al.
(2023)](https://doi.org/10.1016/j.apenergy.2022.120493) adopted the adjusted error of [Haben et al.
(2014)](https://doi.org/10.1016/j.ijforecast.2013.08.002), a peak-aware measure, for exactly this
reason. Artificial Forecasting built a metric over the top 10% of demand values and made it the
primary measure for comparing their models, reporting it both against actual demand and normalised
to transformer rating.

### 5. In the study we read most closely, a forecast stated its own uncertainty badly and a single accuracy score did not reveal it

[Kaas et al. (2026)](https://arxiv.org/abs/2607.01966) scored models on 200 German low-voltage
feeders with an overload-decision metric evaluated at each model's 95th percentile for consumer
peaks and its 5th for producer peaks. The two models that came first and second on consumer peaks in
the quantile variant of that metric — Chronos-Bolt, a time-series foundation model, and a
weekly-naive baseline — turned out to have 90% ranges containing the true value only 62% and 58% of
the time across the series as a whole, and 43% and 49% of the time at the consumer peaks themselves.
In the results of [Kaas et al. (2026)](https://arxiv.org/abs/2607.01966), a model that understates
its uncertainty raises fewer false alarms, so it scores well on a threshold-crossing test while
being exactly the model an operator should not trust near a capacity limit. [Kaas et al.
(2026)](https://arxiv.org/abs/2607.01966) supply their own counter-example: ranked on average error
rather than on the overload metric, the winning model was also the most honest about its own
uncertainty, with reality falling inside its stated 90% range 89.75% of the time.

### 6. In the low-voltage papers reviewed up to 2020, weather forecasts were barely used and weather ensembles almost never

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

## Three findings that cut against this project's plan

Three results in this literature point against Flexpectation's plan, and we intend to test all three
rather than route around them.

### Finer-grained weather data has not always paid

[Browell and Fasiolo (2021)](https://arxiv.org/abs/2103.10335) added spatial statistics derived from
gridded numerical weather prediction to their model of 14 grid supply point groups in GB. Those
spatial statistics helped significantly in two of the 14 regions, hurt significantly in three, and
made no measurable difference in the remaining nine. They put that down to their own model rather
than to the data, writing that another method might yet extract value from the gridded data by
building different features. Weather itself was worth a great deal to them — adding wind and
irradiance cut their pinball loss by 40% overall, and by 60% in North Scotland against 10% in London
— so the question is not whether weather matters but whether *finer* weather does. Artificial
Forecasting obtained postcode-level weather forecasts for two wind-connected primary substations
after their wind-connected models had performed poorly, and reported that the postcode-level
forecasts "did not notably improve model performance", naming better weather data as a next step.
What both results say is that finer weather data does not help everywhere, so the interesting
question is *where* it helps. That question is answerable, and answering it is part of this project:
we expect finer weather data to matter most where a substation's load is dominated by weather-driven
generation or heating, which is where NGED most needs the forecast to be right.

### Weather has bought less than expected at low voltage in the past

[Haben et al. (2019)](https://doi.org/10.1016/j.ijforecast.2018.10.007) tested 100 real low-voltage
feeders with both forecast and observed temperature, and found that temperature had no effect on
forecast accuracy, or a negative one. [Haben et al.
(2019)](https://doi.org/10.1016/j.ijforecast.2018.10.007) used data collected in 2014 and 2015, and
we expect how much weather matters at a substation to be changing quickly, because the thing that
makes a substation weather-dependent is embedded solar generation and heat pumps, and there are far
more of both on the network now than there were then. A primary substation that was almost
weather-independent ten years ago may be strongly weather-dependent today. That is a prediction,
though, not a measurement — and the Scottish primary-substation sensitivities of [Fox et al.
(2018)](https://doi.org/10.34890/134), measured on ten years of data ending in the mid-2010s and
described under "What GB networks have already built" below, say weather was already moving primary
substation demand well before the mid-2010s. Measuring how much weather now explains at NGED's
primary substations is one of the more useful things this project can report.

### A model trained on none of NGED's data may match a model trained on all of it

[Kaas et al. (2026)](https://arxiv.org/abs/2607.01966) tested Chronos-2, a general-purpose
time-series model that had never seen their data, against models trained on the first 160 of those
feeders and scored, like Chronos-2, on all 200. Chronos-2 beat every purpose-trained competitor on
mean absolute error, 3.8 kW against 4.2 kW. Their purpose-trained models were not heavily engineered
— see finding 1 above — but a model given all of a network's history and beaten by one given none of
it is still important information about the value of any programme of heavy engineering.

## What GB networks have already built

**Scottish and Southern Electricity Networks' TRANSITION** (Network Innovation Competition,
Oxfordshire; its load-forecasting deliverable reported 2021) is the closest precedent for
Flexpectation's method. It forecast net load at 13 primary substations, their bulk supply points,
and their 33 kV and 11 kV feeders, from 30 minutes to 10 days ahead.  It split each
substation's net load — demand minus whatever generation behind that substation happened to produce
— into demand and generation, forecast the two separately, then recombined them. And it used the
network connectivity map, the record of which substation feeds which, throughout: the project ranks
"historical network connectivity data availability" as "just as important as historical net demand
and generation measurements". Two things TRANSITION did not set out to do are what Flexpectation
adds: its ensemble covered only the first four days, so from day four to day ten a single
deterministic forecast was all it had, whereas NGED acts out to fourteen; and it was a 13-substation
trial rather than a network-wide deployment. Everything else in its design is the shape
Flexpectation is building.

**[NGED's own Electricity Flexibility and Forecasting System,
EFFS](https://smarter.energynetworks.org/projects/wpden03/)** (Network Innovation Competition,
2018–2021, £3,338,896 of expenditure) forecast grid supply points, bulk supply points, primary
substation transformers, and generation sites from an hour to six months ahead, feeding automated
constraint identification. Its evaluation independently selected XGBoost as the best balance of
accuracy against effort — the same starting point Flexpectation uses. Its forecasts were
deterministic, with no uncertainty attached, which is the step this project adds.

**UK Power Networks' NIA_UKPN0104** is described under challenge 7 above, as the direct predecessor
of Flexpectation's unmetered-solar work.

**[SSEN FastTrack](https://smarter.energynetworks.org/projects/10166254/)** (Strategic Innovation
Fund, Alpha 2025–2026) combines probabilistic forecasting with simulation to model how the
distribution connections queue — around 180 GW and growing — will load the network, from primary
substations up to the grid supply point. FastTrack puts a probability on how much of that queue
turns into real load and how it behaves, which is a planning question rather than the operational
one Flexpectation asks. **[SP Energy Networks'
Predict4Resilience](https://smarter.energynetworks.org/projects/10061710/)** drives a probability
distribution of network faults per district from an ensemble weather forecast, up to seven days
ahead, in a tool built with control-room engineers, which its Beta phase is taking into live trials
— the GB precedent for putting ensemble-derived distributions in front of network operators. SP
Energy Networks has also published at Flexpectation's own voltage level: [Fox et al.
(2018)](https://doi.org/10.34890/134) ran a numerical weather prediction model over Scotland at 1 km
resolution for ten years, mapped it onto each primary substation weighted by customer density, and
used it to separate the effect of weather on peak demand from the effect of everything else — 13
substations in the proof of concept, almost 400 in production. Demand fell by between 1.4% and 4.8%
for each degree Celsius of effective temperature, differing substation by substation with the mix of
customers behind it — every one of the thirteen sensitivities being negative, so demand there fell
as temperature rose. [Fox et al. (2018)](https://doi.org/10.34890/134)'s method corrects history for
planning rather than forecasting forward, but it is the GB precedent for putting gridded weather
onto individual primary substations.

Two deployments outside GB belong alongside these.

**The Dutch operator Alliander runs [OpenSTEF](https://lfenergy.org/projects/openstef/)**, an
open-source forecasting stack under the Linux Foundation's LF Energy umbrella, in live operation
across thousands of grid connection points to 48 hours ahead. It is the only operational network
forecasting system in this review whose code can be read rather than inferred from a deliverable,
and it ships a component splitter that breaks a net-load forecast into solar, wind, and residual
parts — the operational relative of challenge 7, though a far simpler one than [Teng et al.
(2023)](https://doi.org/10.1016/j.rser.2023.113662) describe.

**The second is far larger than any project here.** Enedis, the French distribution network
operator, has forecast consumption and generation at all 2,300 of its high-voltage-to-medium-voltage
substations since 2015, and is now extending that to a finer geographic grid ([Cordier et al.
(2024)](https://doi.org/10.1049/icp.2024.2058), whose abstract we read rather than the full paper).
A high-voltage-to-medium-voltage substation in France is broadly the level of a GB primary
substation. Forecasting operationally at the scale Flexpectation reaches in 2027 is therefore a
decade old somewhere else, which is reassuring about the engineering and says nothing about the
forecast quality, because the abstract we read reports no accuracy figures.

### Northern Powergrid's Artificial Forecasting is further ahead, and sets the bar

**One concurrent project matters more than any paper here.** Artificial Forecasting is an Ofgem
Strategic Innovation Fund programme, with about £3.9m of grant across its three phases, run by
Northern Powergrid with Faculty, EV.energy, and Oaktree Power, the final Beta phase running to
February 2027. Its deliverables are publicly available on the Energy Networks Association's Smarter
Networks Portal, though the [Beta
deliverables](https://smarter.energynetworks.org/projects/10145998/) sit under a separate project
registration from the Alpha ones linked above. It does much of what Flexpectation does at primary
substations, it also covers secondary substations, which Flexpectation does not, and at the time of
writing it is further ahead than Flexpectation.

**Artificial Forecasting has run operationally through a full winter flexibility procurement
cycle.** A forecasting service for primary substations is deployed and has passed the network's
architecture review board, data governance, and information security checks for its current
deployment. It was used operationally by Northern Powergrid's System Forecasting team through a full
winter flexibility procurement cycle to support week-ahead dispatch decisions. It produces
half-hourly probabilistic forecasts with 5th-to-95th-percentile bands, flags forecast exceedances of
firm capacity, and is benchmarked against the network's existing growth-based and persistence
methods and a rolling four-week baseline. The deliverable states, without publishing the figures
behind it, that performance did not materially degrade on average across the 11-day horizon. Their
value case puts whole-life net present value at around £60m for one network, or £250m if three
further networks adopt it, driven by a 3% reduction in spending on reinforcement — building bigger
transformers and cables — in the current price-control period rising to 6% in the next, and a 25%
improvement in the cost-effectiveness of contracted flexibility. Those are the figures from the Beta
application. The project pairs those figures with the appropriate qualification: it reports early
Beta evidence, from one winter procurement cycle, supporting the performance assumptions behind the
value case, which "remains appropriate, subject to further validation".

**Artificial Forecasting is independent evidence that short-term substation forecasting is
operationally useful**, that networks will change their procurement process around it, and that a
benefits case has been made and accepted. Because it is public, operational, and benchmarked against
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
horizon, though unmetered generation, probabilistic forecasting at substation level, and a multi-day
horizon each exist on their own. None tracks the available capacity of a mixed fleet of solar, wind,
and dispatchable generators at one distribution network, or measures whether doing so improves the
forecast. None turns switching-contaminated history into a useful input rather than deleting it,
rewriting it, or absorbing the cost of leaving it in. Flexpectation attempts all eight challenges
above, across four families of model:

- a heavily-tuned version of the gradient-boosting approach that wins most tabular forecasting
  competitions, and which NGED's own EFFS project independently selected;
- weather and time encoders pre-trained on large datasets, so that a model for one substation can
  start from what has been learned across all of them;
- models that use the connectivity map explicitly;
- differentiable physics — building known physical behaviour directly into the model, so that it has
  to learn only what the physics cannot supply: the response of a solar panel and of a wind turbine
  on the generation side, and the thermal response of buildings on the demand side.

**By the standard of scope in this literature, each of the four strands is a separate piece of
work.** Almost every study reviewed above takes on one of the eight challenges, at one voltage
level, with one family of model; the few that touch two solve them as a pipeline rather than
together. Pre-training weather and time encoders and then reading a substation's probabilistic
forecast off them would be a full study by that standard, and so would each of the other three
strands. That sizes the work rather than promising an output — how many of the strands survive
contact with the data is exactly what the project has to find out.

**Only the first of those four strands — the heavily-tuned gradient-boosting model — is in scope for
version one.** The pre-trained encoders, the connectivity-map models, and the differentiable physics
all belong to the network-wide scale-up from 2027, as does the disaggregation of unmetered
generation and forecasting the network as a network.

**The encoders Flexpectation plans to pre-train cover weather and time, and possibly a third for
place, and the machinery for the weather one has been built separately from any energy forecast.**
The plan is a network that turns the raw ECMWF ensemble into a calibrated probabilistic weather
forecast in physical units, which a substation model then reads, alongside a time encoder that
learns how people use the calendar — that Christmas is not an ordinary Thursday — and possibly a
space encoder holding the standing geographic context of each substation. Both halves of the weather
encoder have been built. [Rasp and Lerch (2018)](https://arxiv.org/abs/1805.09091) built the first:
a neural network that post-processes a 50-member ECMWF ensemble into calibrated probabilistic
2-metre temperature at 537 German stations 48 hours ahead, cutting mean continuous ranked
probability score from 1.16 for the raw ensemble to 0.78, with a learned per-station embedding one
of the two components the authors credit for the gain. [Mitra and Ramavajjala
(2023)](https://arxiv.org/abs/2312.00290) built the second: they freeze a weather autoencoder and
train small models on the frozen representation alone, at accuracy comparable to purpose-built
models, though the targets they predict are further weather variables rather than anything on a
network. A network operator has already fine-tuned a pre-trained weather model on its own sensors:
[Bodnar et al. (2025)](https://arxiv.org/abs/2509.25268) post-train Silurian AI's
1.5-billion-parameter Generative Forecasting Transformer on Hydro-Québec's transmission-line weather
stations, wind-farm met masts, and icing sensors, cutting mean absolute error against numerical
weather prediction benchmarks by 15% for temperature, 35% for total precipitation, and 15% for
hub-height wind speed at 6 to 72 hours ahead — but the forecasts are of weather at the assets rather
than of power, and only the icing detector is probabilistic. The nearest anyone came to joining the
two is one entrant in HEFTCom, a competition to forecast a GB wind-and-solar portfolio day-ahead:
[Browell et al. (2025)](https://doi.org/10.1016/j.ijforecast.2025.10.005) report that team Rnt fed
embeddings from their own AI weather models into downstream neural networks and finished third of
the ranked entrants. What we found nobody doing is pre-training a weather encoder against
observations and then reading a substation's probabilistic load forecast off it, or using a
differentiable model of a solar or wind farm to strip out the variance the engineering explains so
that the weather encoder trains on a clean weather signal.

**The main reason for attempting all eight at once is that they may be one challenge rather than
eight.** A switching event, a turbine out for repair, and a stuck meter all surface in the same
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

**So the question we want to answer is whether one model that estimates capacity, switching state,
and demand together beats that pipeline.** NGED's specification leaves room for it, asking that
these phenomena be handled rather than that they be handled explicitly. The one published result
that bears on the question points the joint way: [de Vilmarest et al.
(2024)](https://doi.org/10.1109/TPWRS.2023.3310280), described under challenge 3, removed the
embedded wind and solar capacities from their model of GB regional net load, and the adaptive
version got 0.4% *better*, absorbing into its own coefficients what the explicit capacity figure had
been supplying, while the offline, non-adaptive version got more than 10% worse. That is one result,
on regions far larger than a substation, for one phenomenon out of several — and there are good
reasons to doubt it generalises. We expect a gradient-boosted tree to do badly at the subtraction
that a two-stage residual hands it precomputed, and tens of thousands of training rows per series is
a small sample in which to hope a model discovers an implicit baseline for itself. Neither
expectation is measured here. We expect the answer to differ by model family, which is part of why
the differentiable-physics strand matters: it is the one family in which capacity, weather response,
and demand are estimated jointly by construction.

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

**Behind-the-meter solar disaggregation was excluded because most of it works below NGED's level of
aggregation.**  The exclusion covers our
reading list rather than the whole field: work at feeder aggregation and above is real, and
challenge 7 above names it.

**General concept-drift detection was excluded because it addresses gradual drift, and NGED's
problem is a sudden step change.** A model that simply adapts to a new load level, without ever
detecting that switching happened, is the live alternative to detecting switching at all. We will
treat it as the approach our switching work has to beat.

**Differentiable physics applied to substation demand forecasting produced no strong result** in our
search, though the ingredients exist separately. What we did not find is anyone aggregating building
thermal physics up to a substation and putting it inside a probabilistic forecast, which is the
version this project would need.

**Heat pumps, electric-vehicle chargers, and domestic batteries were not searched at all**, which
challenge 8 above names as this review's largest deliberate omission. Our searches were framed
around substation and generation forecasting, and the electrification literature is large enough to
need a review of its own.

**Network topology detection was excluded because it needs measurements NGED does not have.**
Inferring the network's wiring from high-resolution synchrophasor measurements is well developed,
but those measurements are not available to this project. That exclusion covers neither challenge 4,
which detects switching from half-hourly power alone, nor the topology question under challenge 1,
which is about using a connectivity map NGED already holds rather than inferring one.

**The bulk of the low-voltage forecasting literature is covered second-hand**, through the [Haben et
al. (2021)](https://arxiv.org/abs/2106.00006) review of 221 papers rather than read individually,
and we have not systematically covered low-voltage work published since it closed in 2020. [Haben et
al. (2023)](https://doi.org/10.1007/978-3-031-27852-5)'s open-access book-length treatment is the
better entry point for anyone following this up.

**CIRED is the venue this audience is most likely to read — it is where European distribution
network operators publish their own operational work, so CIRED is where a claim of ours is most
likely to be contradicted.** We therefore searched it in full: the titles and abstracts of every
paper in the CIRED main conferences and workshops of 2017 and 2020 to 2025, about 3,600 of them; the
2018 and 2019 proceedings, which are not indexed, by keyword against their open full-text archive;
and the 305 papers accepted for the Brussels workshop of June 2026 by title, those proceedings not
yet being published. Nothing there contradicts what this review reports missing, and the absences
are worth stating, because CIRED is where the counter-example would have been. They are as good as
the search behind them: a method a paper uses without naming it in its title or abstract would not
have surfaced. No CIRED paper drives a load or generation forecast from a weather ensemble. None
produces an operational load or generation forecast in the days-to-fortnight band that NGED acts on:
the long-horizon load forecasts in the proceedings are annual planning forecasts, and the only
14-day forecast predicts feeder faults rather than load. Fourteen forecast probabilistically at all,
of which one is at substation scale — [Mesarcik et al.
(2025)](https://doi.org/10.1049/icp.2025.1968), day-ahead, on ten years of measurements from 312
Dutch substations. Nothing scores the upper tail, nothing keeps switching-contaminated history
usable, and nothing estimates how much of a generator's capacity is available. The closest paper to
our own challenge *among the ones this exclusion covers*, [Ruhhütl et al.
(2023)](https://doi.org/10.1049/icp.2023.0476), appears in the table under challenge 1 above; its
result is a further instance of finding 1, and of the aggregation effect that finding 2 explains. We
read only the abstracts of it and of [Mesarcik et al.
(2025)](https://doi.org/10.1049/icp.2025.1968), because both full texts are paywalled. The Brussels
titles of June 2026 change none of those absences: 23 of the 305 name forecasting or prediction,
none names an ensemble, and the only short-horizon forecast named is day-ahead; the three others
that name a horizon at all name long-term planning. Two of the 23 apply time-series foundation
models, so the possibility that a model given none of a network's own data can compete is being
tested in this venue too.

## Publishing results that others can compare against

**We will publish the telemetry, the evaluation protocol, the metric definitions, and the code that
computes them, so that someone outside the project can check the results.**

**Energy forecasting's own senior figures say that published results in the field cannot be compared
with each other.** [Hong et al. (2020)](https://doi.org/10.1109/OAJPE.2020.3029979), a review
written by six of the field's most senior figures, concludes that "most papers can never be
replicated, because the data have never been published".

**Incomparable results are what this review ran into at every one of the eight challenges.** Even
the eight studies in the one table above, all forecasting electricity demand somewhere on a network,
differ in target, level, horizon, and weather assumption in nearly every row, so almost none of them
can be compared directly with any other.

This review makes nine commitments to publish or report. Collected in one place, they are:

- **Every ratio comes with its reference forecast, the population it was scored on, and the number
  of ensemble members that produced it.** [Weigel et al. (2007)](https://doi.org/10.1175/MWR3280.1)
  show that a ranked probability skill score is biased downwards by an amount that depends on
  ensemble size, so a score from our 51 members is not comparable with one from a study using ten
  until their correction is applied. We apply it.
- **Accuracy is reported separately for each class of asset** — grid supply points, bulk supply
  points, primary substations, and metered generators — each against its own stated naive baseline,
  because a single project-wide accuracy target would mean different things at different levels.
- **The fraction of series that beat their naive baseline is published alongside the average
  error**, never the average alone.
- **The battery, the gas generator, and the biofuel plant are reported separately** from the wind
  and solar sites, because the battery, the gas generator, and the biofuel plant are dispatched on
  market signals no weather forecast contains.
- **A peak-aware score is reported alongside a proper scoring rule**, never instead of one.
- **The tail is scored with a threshold-weighted continuous ranked probability score**, weighted
  above a fixed per-series threshold set at the 99th percentile of that series' own measured
  history, rather than by selecting the periods in which an exceedance happened.  The obvious alternative — keep
  only the periods in which load crossed the limit, and score those — is not merely noisy but
  biased: [Lerch et al. (2017)](https://doi.org/10.1214/16-STS588) show that choosing which periods
  to score on the basis of what happened rewards a forecaster who over-predicts extremes, and can
  rank a deliberately biased forecast above an honest one. [Gneiting and Ranjan
  (2011)](https://doi.org/10.1198/jbes.2010.08110)'s threshold-weighted score puts the emphasis
  inside the score instead, and stays a proper scoring rule while doing it.
- **Coverage — how often reality fell inside the range the forecast claimed — is broken down by
  season, by forecast lead time, and by how heavily loaded the substation was.** A coverage figure
  averaged over a year can read as a healthy 90% while being 99% in the quiet months and 70% at the
  winter peaks, and the winter peaks are the only periods NGED buys flexibility for. Breaking it
  down is the point, and conformal prediction does not remove the need to: [Foygel Barber et al.
  (2020)](https://doi.org/10.1093/imaiai/iaaa017) prove that a distribution-free guarantee holds
  only on average across all conditions, never separately for the conditions that matter, so a
  conformal forecast can promise 90% coverage overall while failing at the peaks.
- **Each metered generator's series is normalised by its estimated effective capacity** before
  training — unless the comparison described under challenge 3 shows the normalisation is not needed —
  and that estimate is tracked as it changes.
- **Negative results are published too**, including whether an off-the-shelf model given none of our
  data matches our own, and whether sustained experimentation stops yielding improvements.

## References

Every source cited above, in alphabetical order by first author.

- Amaro e Silva, R. and Brito, M. C. (2019). [Spatio-temporal PV forecasting sensitivity to
  modules' tilt and orientation](https://doi.org/10.1016/j.apenergy.2019.113807). *Applied
  Energy*. Read as chapter 5 of Amaro e Silva's open-access doctoral thesis, the published paper
  being paywalled.
- Angus, S., Browell, J., Greenwood, D. and Deakin, M. (2027). [Risk-based dynamic thermal rating in
  distribution transformers via probabilistic
  forecasting](https://doi.org/10.1016/j.epsr.2026.113545). *Electric Power Systems Research*.
- Bernecker, M., Gebhardt, M., Amor, S. B., Wolter, M. and Müsgens, F. (2025). [Quantifying the
  impact of load forecasting accuracy on congestion management in distribution
  grids](https://doi.org/10.1016/j.ijepes.2025.110713). *International Journal of Electrical Power &
  Energy Systems*.
- Bian, Y., Zheng, N., Zheng, Y., Xu, B. and Shi, Y. (2024). [Predicting Strategic Energy Storage
  Behaviors](https://doi.org/10.1109/TSG.2023.3303469). *IEEE Transactions on Smart Grid*.
- Bodnar, C., Rousseau-Rizzi, R., Shankar, N., Merleau, J., Flampouris, S., Candille, G., Antic, S.,
  Miralles, F. and Gupta, J. K. (2025). [A Weather Foundation Model for the Power
  Grid](https://arxiv.org/abs/2509.25268).
- Bollerslev, J., Andersen, P. B., Jensen, T. V., Marinelli, M., Thingvad, A., Calearo, L. and
  Weckesser, T. (2022). [Coincidence Factors for Domestic EV Charging From Driving and Plug-In
  Behavior](https://doi.org/10.1109/TTE.2021.3088275). *IEEE Transactions on Transportation
  Electrification*.
- Bouman, R., Schmeitz, L., Buise, L., Heres, J., Shapovalova, Y. and Heskes, T. (2024). [Acquiring
  Better Load Estimates by Combining Anomaly and Change Point Detection in Power Grid Time-series
  Measurements](https://arxiv.org/abs/2405.16164). *Sustainable Energy, Grids and Networks*.
- Browell, J. and Fasiolo, M. (2021). [Probabilistic Forecasting of Regional Net-load with
  Conditional Extremes and Gridded NWP](https://arxiv.org/abs/2103.10335). *IEEE Transactions on
  Smart Grid*.
- Browell, J., van der Meer, D., Kälvegren, H., Haglund, S., Simioni, E., Bessa, R. J. and Wang, Y.
  (2025). [The hybrid renewable energy forecasting and trading competition
  2024](https://doi.org/10.1016/j.ijforecast.2025.10.005). *International Journal of Forecasting*.
- Buizza, R. and Leutbecher, M. (2015). [The forecast skill
  horizon](https://doi.org/10.1002/qj.2619). *Quarterly Journal of the Royal Meteorological
  Society*.
- Campagne, E., Amara-Ouali, Y., Goude, Y., Zehavi, I. and Kalogeratos, A. (2025). [Graph Neural
  Networks for Electricity Load Forecasting](https://arxiv.org/abs/2507.03690).
- Cordier, G. et al. (2024). [Methods and techniques used to produce electricity forecasts on
  Enedis’ distribution network at a finer grid than the HV/MV
  substation](https://doi.org/10.1049/icp.2024.2058). *IET Conference Proceedings*.
- Dantas, G. and Browell, J. (2026). [Seamless Short‐ to Mid‐Term Probabilistic Wind Power
  Forecasting](https://doi.org/10.1002/we.70079). *Wind Energy*.
- de Vilmarest, J., Browell, J., Fasiolo, M., Goude, Y. and Wintenberger, O. (2024). [Adaptive
  Probabilistic Forecasting of Electricity (Net-)Load](https://doi.org/10.1109/TPWRS.2023.3310280).
  *IEEE Transactions on Power Systems*.
- Deceglie, M. G. et al. (2026). [RdTools](https://doi.org/10.5281/zenodo.1210316). *Zenodo*.
- Doubleday, K., Van Scyoc Hernandez, V. and Hodge, B. M. (2020). [Benchmark probabilistic solar
  forecasts: Characteristics and recommendations](https://doi.org/10.1016/j.solener.2020.05.051).
  *Solar Energy*.
- EA Technology and Western Power Distribution (2019). [Electric Nation Customer Trial Final
  Report](https://eatechnology.com/media/girhcnsc/electric-nation-customer-trial-report.pdf).
- Electricity North West (2018). [ATLAS](https://smarter.energynetworks.org/projects/nia_enwl008/).
- Erdener, B. C., Feng, C., Doubleday, K., Florita, A. and Hodge, B. M. (2022). [A review of
  behind-the-meter solar forecasting](https://doi.org/10.1016/j.rser.2022.112224). *Renewable and
  Sustainable Energy Reviews*.
- Faustine, A., Nunes, N. and Pereira, L. (2025). [Efficiency Through Simplicity: MLP-Based Approach
  for Net-Load Forecasting With Uncertainty Estimates in Low-Voltage Distribution
  Networks](https://doi.org/10.1109/TPWRS.2024.3400123). *IEEE Transactions on Power Systems*.
- Fildes, R. (2020). [Learning from forecasting
  competitions](https://doi.org/10.1016/j.ijforecast.2019.04.012). *International Journal of
  Forecasting*.
- Fox, J., Plecas, M., Neilson, D., Cannon, D. and Parr, J. (2018). [Analysis of local demand trends
  and forecasting through weather correction and benefit to DSO transistion and
  microgrids](https://doi.org/10.34890/134). *CIRED Workshop, Ljubljana*.
- Foygel Barber, R., Candès, E. J., Ramdas, A. and Tibshirani, R. J. (2020). [The limits of
  distribution-free conditional predictive inference](https://doi.org/10.1093/imaiai/iaaa017).
  *Information and Inference: A Journal of the IMA*.
- Fulton, J., Bieker, J., Dudfield, P., Cotton, S., Watts, Z. and Kelly, J. (2024). [Forecasting
  regional PV power in Great Britain with a multi-modal late fusion
  network](https://www.climatechange.ai/papers/iclr2024/46). *ICLR 2024 Workshop on Tackling Climate
  Change with Machine Learning*.
- Gijón, A., Eiraudo, S., Manjavacas, A., Schiera, D. S., Molina-Solana, M. and Gómez-Romero, J.
  (2025). [Integrating Physics and Data-Driven Approaches: An Explainable and Uncertainty-Aware
  Hybrid Model for Wind Turbine Power Prediction](https://arxiv.org/abs/2502.07344). *Computer
  Physics Communications*.
- Gilbert, C., Browell, J. and Stephen, B. (2023). [Probabilistic load forecasting for the low
  voltage network: forecast fusion and daily peaks](https://arxiv.org/abs/2206.11745). *Sustainable
  Energy, Grids and Networks*.
- Gneiting, T. and Ranjan, R. (2011). [Comparing Density Forecasts Using Threshold- and
  Quantile-Weighted Scoring Rules](https://doi.org/10.1198/jbes.2010.08110). *Journal of Business &
  Economic Statistics*.
- Gouveia, A. M. V., Hashmi, M. U., D’hulst, R. and Van Hertem, D. (2026). [Installed PV capacity
  detection on LV substations: Comparison of Data-Driven and Model-Based
  methods](https://doi.org/10.1016/j.ijepes.2026.111848). *International Journal of Electrical Power
  and Energy Systems*.
- Gouveia, A. M. V., Hashmi, M. U., D’hulst, R. and Van Hertem, D. (2026b). [Importance of
  Aggregated DER Installed Capacity in Distribution Networks](https://arxiv.org/abs/2604.13926).
- Haben, S., Ward, J., Vukadinovic Greetham, D., Singleton, C. and Grindrod, P. (2014). [A new error
  measure for forecasts of household-level, high resolution electrical energy
  consumption](https://doi.org/10.1016/j.ijforecast.2013.08.002). *International Journal of
  Forecasting*.
- Haben, S., Giasemidis, G., Ziel, F. and Arora, S. (2019). [Short term load forecasting and the
  effect of temperature at the low voltage level](https://doi.org/10.1016/j.ijforecast.2018.10.007).
  *International Journal of Forecasting*.
- Haben, S., Arora, S., Giasemidis, G., Voss, M. and Greetham, D. V. (2021). [Review of Low Voltage
  Load Forecasting: Methods, Applications, and Recommendations](https://arxiv.org/abs/2106.00006).
  *Applied Energy*.
- Haben, S., Voß, M. and Holderbaum, W. (2023). [Core Concepts and Methods in Load Forecasting: With
  Applications in Distribution Networks](https://doi.org/10.1007/978-3-031-27852-5). *Springer*.
- Hertel, M., Pütz, S., Kolar, J., Schäfer, B., Mikut, R. and Hagenmeyer, V. (2026). [A Benchmark
  for Electrical Load Forecasting Across Grid Levels: Time-Series Transformers Outperform
  Established Methods](https://arxiv.org/abs/2607.15705).
- Hong, T., Pinson, P., Wang, Y., Weron, R., Yang, D. and Zareipour, H. (2020). [Energy Forecasting:
  A Review and Outlook](https://doi.org/10.1109/OAJPE.2020.3029979). *IEEE Open Access Journal of
  Power and Energy*.
- Hüttel, F. B., Rodrigues, F. and Pereira, F. C. (2023). [Mind the Gap: Modelling Difference
  Between Censored and Uncensored Electric Vehicle Charging
  Demand](https://arxiv.org/abs/2301.06418). *Transportation Research Part C: Emerging
  Technologies*.
- Huyghues-Beaufond, N., Tindemans, S., Falugi, P., Sun, M. and Strbac, G. (2020). [Robust and
  automatic data cleansing method for short-term load forecasting of distribution
  feeders](https://doi.org/10.1016/j.apenergy.2019.114405). *Applied Energy*.
- Hyndman, R. J. (2020). [A brief history of forecasting
  competitions](https://doi.org/10.1016/j.ijforecast.2019.03.015). *International Journal of
  Forecasting*.
- Jumper, J. (2024). [Nobel Week interview](https://youtu.be/nNM1QdmFwIs?t=852). Nobel Prize YouTube
  channel, 6 December 2024.
- Jung, B. W., Lee, D. S., Lee, J. W. and Son, S. Y. (2024). [Distribution network voltage
  forecasting based on graph convolutional networks – long short-term
  memory](https://doi.org/10.1049/icp.2024.1900). *IET Conference Proceedings*.
- Kaas, B., Treutlein, M., Gerber, H. B., Neumann, O., Phatthanakhuha, C., Resch, O., Mikut, R. and
  Hagenmeyer, V. (2026). [Probabilistic Low-Voltage Peak Load Forecasting with Time Series
  Foundation Models Evaluated on Application-Oriented Metrics](https://arxiv.org/abs/2607.01966).
- Kim, J.-H., Lee, B.-S. and Kim, C.-H. (2020). [A Study on the Development of Machine-Learning
  Based Load Transfer Detection Algorithm for Distribution Planning](https://doi.org/10.3390/en13174358).
  *Energies*.
- Kim, J.-H., Joung, J.-M. and Lee, B.-S. (2022). [A Study on the Preprocessing Method for Power
  System Applications Based on Polynomial and Standard Patterns](https://doi.org/10.3390/en15041441).
  *Energies*.
- Kim, J.-H. (2025). [Unsupervised Load Transfer Detection Based on Wavelet Change Point
  Analysis and Isolation Forest](https://doi.org/10.5370/KIEE.2025.74.11.1757). *The
  Transactions of The Korean Institute of Electrical Engineers*.
- Kleinebrahm, M. et al. (2026). [Energy-Arena: A Dynamic Benchmark for Operational Energy
  Forecasting](https://arxiv.org/abs/2604.24705). *2026 International Conference on the European
  Energy Market*.
- Kryshtafovych, A., Schwede, T., Topf, M., Fidelis, K. and Moult, J. (2021). [Critical assessment
  of methods of protein structure prediction (CASP) — Round
  XIV](https://doi.org/10.1002/prot.26237). *Proteins*.
- Lerch, S., Thorarinsdottir, T. L., Ravazzolo, F. and Gneiting, T. (2017). [Forecaster’s Dilemma:
  Extreme Events and Forecast Evaluation](https://doi.org/10.1214/16-STS588). *Statistical Science*.
- LF Energy. [OpenSTEF](https://lfenergy.org/projects/openstef/).
- Liander. [Open data](https://www.liander.nl/over-ons/open-data).
- Liu, H., Wang, Y., Wei, C., Li, J. and Lin, Y. (2019). [Two-Stage Short-Term Load Forecasting for
  Power Transformers Under Different Substation Operating
  Conditions](https://doi.org/10.1109/ACCESS.2019.2951422). *IEEE Access*.
- Love, J. et al. (2017). [The addition of heat pump electricity load profiles to GB electricity
  demand: Evidence from a heat pump field trial](https://doi.org/10.1016/j.apenergy.2017.07.026).
  *Applied Energy*.
- Ludwig, N., Arora, S. and Taylor, J. W. (2023). [Probabilistic load forecasting using
  post-processed weather ensemble predictions](https://doi.org/10.1080/01605682.2022.2115411).
  *Journal of the Operational Research Society*.

- Martín, P., Moreno, G., Rodríguez, F. J., Jiménez, J. A. and Fernández, I. (2018). [A Hybrid
  Approach to Short-Term Load Forecasting Aimed at Bad Data Detection in Secondary Substation
  Monitoring Equipment](https://doi.org/10.3390/s18113947). *Sensors*.
- Mayer, M. J. and Gróf, G. (2021). [Extensive comparison of physical models for photovoltaic power
  forecasting](https://doi.org/10.1016/j.apenergy.2020.116239). *Applied Energy*.
- McSweeney, L., Haben, S. and Young, S. (2023). [Data Science Challenges; A Whole Systems Lens for
  Energy Network Solutions](https://doi.org/10.1109/ISGTEUROPE56780.2023.10407541). *2023 IEEE PES
  Innovative Smart Grid Technologies Europe*.
- Meng, B., Loonen, R. and Hensen, J. L. M. (2020). [Data-driven inference of unknown tilt and
  azimuth of distributed PV systems](https://doi.org/10.1016/j.solener.2020.09.077). *Solar Energy*.
- Mesarcik, M., Loke, J., Wildeboer, J. and Lucassen, B. (2025). [Probabilistic day-ahead power
  forecasting in the medium-voltage grid using state space
  models](https://doi.org/10.1049/icp.2025.1968). *IET Conference Proceedings*.
- Messner, J. W., Pinson, P., Browell, J., Bjerregård, M. B. and Schicker, I. (2020). [Evaluation of
  wind power forecasts — An up-to-date view](https://doi.org/10.1002/we.2497). *Wind Energy*.
- Meyer, M., Kaltenpoth, S., Albers, H., Zalipski, K. and Müller, O. (2026). [TS-Arena: A Live
  Forecast Pre-Registration Platform](https://arxiv.org/abs/2512.20761). *Proceedings of the 32nd
  ACM SIGKDD Conference on Knowledge Discovery and Data Mining*.
- Meyers, B., Deceglie, M., Deline, C. and Jordan, D. (2020). [Signal Processing on PV Time-Series
  Data: Robust Degradation Analysis Without Physical
  Models](https://doi.org/10.1109/JPHOTOV.2019.2957646). *IEEE Journal of Photovoltaics*.
- Mitra, P. and Ramavajjala, V. (2023). [Learning to forecast diagnostic parameters using
  pre-trained weather embedding](https://arxiv.org/abs/2312.00290).
- Moriano, J., Rodríguez, F., Martín, P., Jiménez, J. and Vuksanovic, B. (2016). [A New Approach to
  Detection of Systematic Errors in Secondary Substation Monitoring Equipment Based on Short Term
  Load Forecasting](https://doi.org/10.3390/s16010085). *Sensors*.
- National Energy System Operator. [Embedded wind and solar
  forecasts](https://www.neso.energy/data-portal/embedded-wind-and-solar-forecasts).
- National Energy System Operator (2023). [Solar PV Nowcasting
  (NIA2_NGESO002)](https://smarter.energynetworks.org/projects/nia2_ngeso002/).
- National Energy System Operator (2024). [Solar NowCasting innovation project improves solar
  forecasting](https://www.neso.energy/news/solar-nowcasting-innovation-project-improves-solar-forecasting).
- Nespoli, L., Medici, V., Lopatichki, K. and Sossan, F. (2020). [Hierarchical Demand Forecasting
  Benchmark for the Distribution Grid](https://arxiv.org/abs/1910.03976). *Electric Power Systems
  Research*.
- Nguyen, T. N. and Müsgens, F. (2026). [A meta-analysis of solar forecasting based on skill
  score](https://doi.org/10.1063/5.0300682). *Journal of Renewable and Sustainable Energy*.
- Northern Powergrid (2024). [Artificial Forecasting, Alpha
  phase](https://smarter.energynetworks.org/projects/npg_sif_006-1/).
- Northern Powergrid (2024). [Detecting LCTs from Smart Meter Consumption
  Data](https://smarter.energynetworks.org/projects/npg_nia_-49/).
- Northern Powergrid (2024). [IMP/001/911 Code of Practice for the Economic Development of the LV
  System, version
  7.0](https://www.northernpowergrid.com/sites/default/files/assets/IMP001911_0.pdf).
- Northern Powergrid (2025). [Artificial Forecasting, Beta
  phase](https://smarter.energynetworks.org/projects/10145998/).
- Open Climate Fix. [PVNet](https://github.com/openclimatefix/PVNet).
- Ostermann, A. and Haug, T. (2024). [Probabilistic forecast of electric vehicle charging demand:
  analysis of different aggregation levels and energy
  procurement](https://doi.org/10.1186/s42162-024-00319-1). *Energy Informatics*.
- Paredes, G. and Vargas, L. (2017). [Adjustment of discrete load changes in feeder databases for
  improving medium‐term demand forecasting](https://doi.org/10.1049/iet-gtd.2017.0129). *IET
  Generation, Transmission & Distribution*.
- Perry, K. and Muller, M. (2022). [Automated Shift Detection in Sensor-Based PV Power and
  Irradiance Time Series](https://doi.org/10.1109/PVSC48317.2022.9938675). *2022 IEEE 49th
  Photovoltaics Specialists Conference (PVSC)*.
- Pfeifer, P., Tran, J., Fendri, A., Krahl, S., Moser, A. and Verheggen, L. (2021). [Accuracy of
  load and generation forecasts for the operational planning of power distribution
  systems](https://doi.org/10.1049/icp.2021.2177). *IET Conference Proceedings*.
- Pierrot, A. and Pinson, P. (2024). [On Tracking Varying Bounds When Forecasting Bounded Time
  Series](https://doi.org/10.1080/00401706.2024.2350421). *Technometrics*.
- Pinheiro, M. G., Madeira, S. C. and Francisco, A. P. (2023). [Short-term electricity load
  forecasting—A systematic approach from system level to secondary
  substations](https://doi.org/10.1016/j.apenergy.2022.120493). *Applied Energy*.
- Rasp, S. and Lerch, S. (2018). [Neural networks for post-processing ensemble weather
  forecasts](https://arxiv.org/abs/1805.09091). *Monthly Weather Review*.
- Recht, B., Roelofs, R., Schmidt, L. and Shankar, V. (2019). [Do ImageNet Classifiers Generalize to
  ImageNet?](https://arxiv.org/abs/1902.10811) *Proceedings of the 36th International Conference on
  Machine Learning*.
- Richardson, D. S. (2000). [Skill and relative economic value of the ECMWF ensemble prediction
  system](https://doi.org/10.1002/qj.49712656313). *Quarterly Journal of the Royal Meteorological
  Society*.
- Ruhhütl, M., Schmaranz, R. and Dietrichsteiner, T. (2023). [Load and generation forecast on
  substation level](https://doi.org/10.1049/icp.2023.0476). *IET Conference Proceedings*.
- Saint-Drenan, Y.-M., Bofinger, S., Fritz, R., Vogt, S., Good, G. H. and Dobschinski, J. (2015).
  [An empirical approach to parameterizing photovoltaic plants for power forecasting and
  simulation](https://doi.org/10.1016/j.solener.2015.07.024). *Solar Energy*.
- Salinas, D., Flunkert, V., Gasthaus, J. and Januschowski, T. (2020). [DeepAR:
  Probabilistic forecasting with autoregressive recurrent
  networks](https://doi.org/10.1016/j.ijforecast.2019.07.001). *International Journal of
  Forecasting*.
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
- UK Power Networks and PPA Energy and Capula (2014). [Distribution Network Visibility: LCN Fund
  Tier 1 Close Down
  Report](https://www.ofgem.gov.uk/sites/default/files/docs/2014/03/dnv_cdr_version_3.0_270214.pdf).
- Viotti, O., Arnqvist, J. and Olauson, J. (2026). [Estimating Wind‐Power Capacity Time Series From
  Production Data Using a Power Curve Model and Quadratic
  Optimization](https://doi.org/10.1002/we.70136). *Wind Energy*.
- Wang, Y., Zhang, N., Chen, Q., Kirschen, D. S., Li, P. and Xia, Q. (2018). [Data-Driven
  Probabilistic Net Load Forecasting With High Penetration of Behind-the-Meter
  PV](https://doi.org/10.1109/TPWRS.2017.2762599). *IEEE Transactions on Power Systems*.
- Weigel, A. P., Liniger, M. A. and Appenzeller, C. (2007). [The Discrete Brier and Ranked
  Probability Skill Scores](https://doi.org/10.1175/MWR3280.1). *Monthly Weather Review*.
- Western Power Distribution (2017). [Time Series Data
  Quality](https://smarter.energynetworks.org/projects/nia_wpd_011/).
- Western Power Distribution (2021). [Electricity Flexibility and Forecasting System
  (EFFS)](https://smarter.energynetworks.org/projects/wpden03/).
- Willis, H. L., Powell, R. D. and Wall, D. L. (1984). [Load Transfer Coupling Regression
  Curve Fitting for Distribution Load Forecasting](https://doi.org/10.1109/TPAS.1984.318713).
  *IEEE Transactions on Power Apparatus and Systems*.
- Zhang, X. Y., Watkins, C. and Kuenzel, S. (2022). [Multi-quantile recurrent neural network for
  feeder-level probabilistic energy disaggregation considering roof-top solar
  energy](https://doi.org/10.1016/j.engappai.2022.104707). *Engineering Applications of Artificial
  Intelligence*.
