# The current state of the art in energy forecasting

Before we discuss the literature, there is a very important caveat to admit up-front: In 2026, no honest narrative review of the energy forecasting literature can claim to reveal the canonical "state of the art"! That is because (almost) all energy forecasting papers measure performance in different ways, against different datasets. It's like an international football tournament where every team plays by different rules, with different size goals.

Energy forecasting researchers have done great work over the years. But, unfortunately, the literature does not tell us how those approaches compare against each other, especially in messy "real world" energy forecasting scenarios. This isn't anyone's fault; it's a systemic failure. And the industry is already aware of this problem, and people are trying to fix it. But, at the time of writing, the literature cannot yet tell us the current state of the art solutions for the problems that NGED cares about.

Having said that, there have been some valiant attempts to compare multiple forecasting approaches against the same dataset. But none of these attempts directly address the main challenges relevant to Flexpectation. Before we discuss those attempts, we must emphasise two reasons for optimism:

Firstly, whilst we might not know exactly which algorithms provide the best forecasting performance, we do know how to go about finding out. There's no magic. Machine learning is an empirical science, and progress in it comes largely from testing many ideas under identical conditions and measuring carefully — because most ideas fail. John Jumper, who shared the 2024 Nobel Prize in Chemistry for his work on AlphaFold, puts the share of research ideas that fail at around 90%, and treats that rate as an ordinary and necessary feature of doing research rather than as evidence of doing it badly ([Nobel Week interview](https://www.nobelprize.org/prizes/chemistry/2024/jumper/interview/), 6 December 2024, from 14:12). If roughly one idea in ten survives contact with the data, ten attempts is simply the price of one result. So our task is to run hundreds of ML experiments, and then measure performance against the same dataset, using the same performance metrics. This, in turn, requires us to design and build a framework that makes it easy to run hundreds of ML experiments per month. At the time of writing, we have implemented the first version of this framework, and we will continue to evolve the framework over the course of the project.

Secondly - and perhaps most importantly - the fact that the industry doesn't yet know the state of the art is a huge opportunity for the Flexpectation project: We are in a very privileged position where we can try hundreds of ideas, and test the best ideas in the real world. We have a fantastic opportunity to make a significant contribution to the energy forecasting industry by publishing our "leaderboards of ML experiments", and hence help the industry as a whole to better understand how multiple approaches perform.

## What we read

We read fifteen papers in full and drew on four more that were only partly available to us. Where
one of those four appears below, we say so. We also read the published deliverables of six
concurrent GB network projects. The selection was deliberate rather than systematic: a paper earned
its place by bearing on a decision Flexpectation actually faces and by changing something we
believed. Papers may be missing for no better reason than that we did not find them, and the final
section, "What this review excluded, and why", lists what we knowingly left out. A further group of
papers is cited once each, for one specific result, rather than reviewed.

Flexpectation is a Network Innovation Allowance project of £841,733, running from January 2026 to
March 2028. It forecasts 32 time series in the trial area of National Grid Electricity Distribution
(NGED) — 16 primary substations, two grid supply points, two bulk supply points and 12 metered
generators — at half-hourly resolution, 14 days ahead, updated every six hours, with each forecast
expressed as a range of possible loads with a probability attached to each — a 10% chance of
exceeding this load, a 50% chance of exceeding that one — rather than as a single number. From 2027
it scales towards roughly 2,500 series across NGED's network. The 14-day horizon sits at the edge of
what a weather ensemble can supply: [Buizza and Leutbecher (2015)](https://doi.org/10.1002/qj.2619)
put the lead time at which a weather ensemble — the 51 slightly different forecasts the European
Centre for Medium-Range Weather Forecasts (ECMWF) runs from 51 slightly different starting
conditions, whose spread shows how confident the forecast is — stops being any more useful than
quoting the long-run average weather for that day of the year at 16 to 23 days. They measured that
on upper-air variables rather than on the near-surface temperature and irradiance that drive
substation load, for which we would expect a shorter horizon.

Almost every number in this review depends on where in the network it was measured, so it is worth
setting out the voltage levels of a distribution network before any numbers appear:

- **Grid supply point** — where the distribution network meets the transmission system, carrying
  hundreds of thousands of customers. NGED has 52. GB is separately divided into 14 *grid supply
  point groups*, each a whole distribution region containing many grid supply points; Browell and
  Fasiolo (2021), reviewed below, forecast the 14 grid supply point groups, not the 52 individual
  grid supply points.
- **Bulk supply point** — the level below a grid supply point, typically 132 kV to 33 kV. NGED has
  271.
- **Primary substation** — 33 kV to 11 kV, typically a few thousand customers. NGED has 1,161, of
  which 16 are in the trial area.
- **Secondary substation** — 11 kV to 400 V, from tens to a few hundred customers. **Flexpectation
  forecasts no secondary substations**, neither in the 32-series trial area nor in the network-wide
  scale-up from 2027, though several of the studies reviewed below do.
- **Feeder and individual customer** — the bottom of the ladder, and the level at which most of the
  low-voltage forecasting literature works.

## The best published results, and why they cannot be compared

Energy forecasting's own senior figures say that published results in the field cannot be compared
with each other. [Hong et al. (2020)](https://doi.org/10.1109/OAJPE.2020.3029979), a review written
by six of the field's most senior figures, concludes that "most papers can never be replicated,
because the data have never been published". Hong et al. (2020) add that authors sometimes pick the
error measure that favours their own method, that significance tests are seldom run when the
differences between models are small, and that many papers compare a new model only against models
"within the immediate family". [Tawn and Browell (2022)](https://doi.org/10.1016/j.rser.2021.111758)
found eleven wind and solar papers that compared a new model only against other models of the same
type. Hong et al. (2020) name two remedies: publishing the underlying data, and running competitions
in which every team forecasts the same dataset. Both appear in the table below.

Every entry below is best-in-class for the problem its authors set themselves, and almost none can
be compared directly with any other, because the target, the level, the horizon and the weather
assumption differ in nearly every row.

| Source | What they forecast | Level and scale | Horizon | Best result, and what it beat | Weather |
|---|---|---|---|---|---|
| [Browell et al. 2025 (HEFTCom)](https://doi.org/10.1016/j.ijforecast.2025.10.005) | Combined wind and solar output, GB | National: 2 wind and 1 solar fleet | Day-ahead | Winning team scored **22.18 MWh mean pinball loss** against the organisers' quick-start benchmark of 53.58 (note 1) | Real forecasts, live |
| [Kaas et al. 2026](https://arxiv.org/abs/2607.01966) | Net load, Germany | Low-voltage feeder: 200 | 4 days | A general-purpose "foundation" model that had never seen the data beat every purpose-trained model on average error, 3.8 kW against 4.2 kW | Actual weather, after the fact |
| [Hertel et al. 2026](https://arxiv.org/abs/2607.15705) | Load, Germany and Portugal | Transmission, plus 200 low-voltage feeders and 287 individual customers | 4 days | Best model beat a day-type persistence forecast by 59.6% at transmission level, 42.3% at low-voltage feeders, 23.3% at individual customers | Reanalysis and 1–3 h forecasts |
| [Browell & Fasiolo 2021](https://arxiv.org/abs/2103.10335) | Regional net load, GB | Regional: 14 grid supply point groups | Day-ahead | Held the same risk with **up to 24.6% less upward reserve** than a fixed-tail alternative (note 2) | Real forecasts |
| [Pinheiro et al. 2023](https://doi.org/10.1016/j.apenergy.2022.120493) | Load, Portugal | Secondary substation: 96,989 | Day-ahead | 42–47% better than the reference benchmark at system level. **At substation level, beat a naive forecast on 83–87% of network-owned and 66–70% of customer-owned sites** | Real forecasts, 7–8 h old |
| [Gilbert et al. 2023](https://arxiv.org/abs/2206.11745) | Load, GB | Four levels: primary substation down to household | Day-ahead | Combining forecasts gained **0.0–0.4% averaged over all periods**, but **5.7–9.0% when restricted to peaks** | None at all |
| [Bouman et al. 2024](https://arxiv.org/abs/2405.16164) | Switch-event and anomaly detection, Netherlands | Primary substation: 180 | Not a forecast | Annual maximum and minimum load estimates within a 10% margin in 88% and 91% of cases | None |
| [SSEN TRANSITION 2021](https://ssen-innovation.co.uk/transition/) | Net load, Oxfordshire | Primary substation: 13, plus their bulk supply points and 11 kV feeders | 30 min to 10 days | **11 of 13 primary substation models below 10%** mean absolute percentage error when fitted (note 3) | 40-member ICON-EU ensemble to 4 days, then one deterministic forecast to 10 days |
| [Artificial Forecasting (Northern Powergrid)](https://smarter.energynetworks.org/projects/npg_sif_006-1/) | Demand and export at primary substations; net demand at secondary | Primary substation: 551 with export data, 171 modelled; secondary: 729 | Day-ahead to 11 days at primary; week- to month-ahead at secondary | **About 8% lower mean absolute error** of utilisation rate than the network's existing method (note 4) | Real forecasts at primary; none in the published secondary results |

*Weather column:* "real forecasts" means the weather forecast that was genuinely available when the
power forecast was made; "actual weather, after the fact" means observations, or a weather model
re-run after the event, that no forecaster would have had. Why that difference matters is explained
in the next section.

*Terms used in the table.* **Pinball loss** scores a forecast that states a range rather than a
single number, penalising it more heavily for missing on the side it claimed was unlikely. A
**day-type persistence** benchmark predicts today's load from the most recent day of the same type —
last Tuesday for a Tuesday. A **foundation model** is trained on large numbers of unrelated time
series and then applied to a new one it has never seen. **Deterministic** means a single predicted
number with no uncertainty attached.

*Notes on the table.* **1.** HEFTCom's organisers also entered a more competitive reference, which
scored 25.38, and the next two teams scored 23.18 and 24.64. **2.** The 24.6% saving is at the most
extreme tail level Browell and Fasiolo tested, and falls to 3.2% at the least extreme. **3.** The
two SSEN TRANSITION models that missed 10% reached 13.4% and 19.7%, and 94% of its 11 kV feeders
came in below 20%. **4.** Artificial Forecasting also captured 83% of the top 10% of demand values
inside its 5th-to-95th-percentile band, and beat a rolling four-week baseline at all eight of the
near-capacity substations it was evaluated on.

Three further sources sit outside the table, and carry findings rather than comparable scores.
[Haben et al. 2021](https://arxiv.org/abs/2106.00006) reviewed 221 low-voltage forecasting papers
published to 2020, and finding 6 below rests on what they counted. [Shukla & Hong
2024](https://doi.org/10.1049/stg2.12162) reports the BigDEAL competition across three neighbouring
US distribution companies, and finding 1 below rests on what it found.
[Energy-Arena](https://arxiv.org/abs/2604.24705) is a live public leaderboard rather than a
competition — we could not extract the full paper and worked from its abstract and the running
platform, which today carries 24 deterministic challenges across prices, load, wind and solar.

The sharpest illustration that the choice of metric, dataset and horizon decides which model wins
comes from two papers published a fortnight apart, by overlapping groups at the Karlsruhe Institute
of Technology, on the same 200 German low-voltage feeders. Kaas et al. (2026) and Hertel et al.
(2026) name different models as best. Inside Kaas et al. (2026), mean absolute error and an
overload-decision metric name different winners again. Neither disagreement is a mistake: the two
papers test different sets of models at different time resolutions, and the two metrics answer
different questions.

### Three things decide what a headline number means

**A percentage improvement says more about where in the network it was measured than about the
method that produced it.** That is the single most important thing to take from this review, because
it sets what NGED should expect at its primary substations. Finding 2 below gives the measurements.

**Two studies used weather no real forecaster would have had, so their figures are upper bounds
rather than achievable performance.** Kaas et al. (2026) and Hertel et al. (2026) use the weather as
it was known immediately afterwards — short-range forecasts issued one to three hours ahead, or
reanalysis, meaning a weather model re-run after the event with all the observations available. Kaas
et al. (2026) and Hertel et al. (2026) do this deliberately, so that differences between models are
not swamped by weather-forecast error. That is the right choice for their question and the wrong one
for NGED's, because it removes the error that dominates beyond a day or two, which is precisely the
range NGED acts on.

**Averaging over every half-hour hides a gain that only shows up at the daily peak.** An average
over every period is dominated by the quiet ones, and the quiet ones are not why a network buys
flexibility. Gilbert et al. (2023) is the worked example, below.

### Only two kinds of published number transfer to NGED

**A ratio against a baseline transfers, but only if the paper says what the baseline was and which
substations it was averaged over.** Those baselines differ far more than the prose in most papers
suggests — yesterday's value at the same time, the average of the last four weeks, a day-type
persistence rule and the long-run seasonal average all appear among the studies reviewed here, and a
percentage gain against one baseline is not a percentage gain against another. A skill score — how
much less error a forecast has than a stated benchmark, as a percentage — also depends on how many
weather-ensemble members produced it, which almost no paper states. **Errors normalised by something
physical** also transfer: an error expressed as a fraction of a substation's firm capacity or
transformer rating means the same thing at every substation, whereas an error expressed as a
fraction of the load that happened to occur does not. An absolute error in kilowatts or megawatts
tells NGED nothing on its own, because it depends entirely on how big the substation was, and none
of the absolute figures above should be read as a target for this project.

## Six findings recur across the studies we read

These are findings about this literature, not laws of nature: each is what several teams measured on
their own networks, and a network that differs from theirs may well behave differently.

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
individual customers. The model did not get worse; the problem got harder. That pattern is probably
not a fact about forecasting so much as a fact about averaging: a grid supply point aggregates
hundreds of thousands of customers, whose individual quirks cancel out, while a single feeder
aggregates a few dozen, whose quirks do not. Predicting the temperature of a kilogram of air is
easier than predicting the motion of each molecule in it, and for the same reason.

Two consequences follow. We will report accuracy separately for each class of asset — grid supply
points, bulk supply points, primary substations and metered generators — against a stated naive
baseline for each, because a single project-wide accuracy target would mean different things at
different levels. And we will not assume that rising error means falling usefulness. A forecast at a
primary substation may carry a larger percentage error than a forecast at a grid supply point and
still support flexibility procurement just as well, because what NGED needs from the forecast is a
reliable answer to "will this substation exceed its firm capacity?", and that question can be
answered well even when the load itself is hard to predict precisely. Whether decision-usefulness
really is flat across voltage levels is something this project can measure, and we intend to.

**3. In the one study that reported results substation by substation at scale, a substantial
minority of substations were not forecast better by a trained model than by a naive "same time
yesterday" rule.** Pinheiro et al. (2023) found that their model beat a "same time yesterday"
forecast at 83–87% of network-owned secondary substations but at only 66–70% of customer-owned ones.
Those customer-owned sites serve a single customer — one large building or one industrial process —
where load follows decisions no weather model can see. We do not know that NGED's primary
substations will behave the same way, and they may not, because a primary substation aggregates far
more customers than a Portuguese secondary substation does. The warning that a site serving a single
customer may not be forecastable applies with more force to three of the 12 metered generators
Flexpectation forecasts in the trial area — a battery, a gas generator and a biofuel plant — because
those are dispatched on market prices and operator decisions, and no weather forecast contains
either. We expect them to be the hardest series in the trial area, and we will report them
separately rather than pooling them with the wind and solar sites. The risk that some of NGED's 32
series cannot be forecast better than a naive rule is real enough that we will report the fraction
of series beating a naive baseline alongside the average error, rather than reporting the average
alone.

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

Whenever we publish a probabilistic forecast, we will publish how often reality actually fell inside
the range the forecast claimed — its **coverage** — broken down by season, by forecast lead time and
by how heavily loaded the substation was. Breaking the figure down that way is the point: a coverage
figure averaged over a year can read as a healthy 90% while being 99% in the quiet months and 70% at
the winter peaks, and the winter peaks are the only periods NGED buys flexibility for.

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
statistics derived from gridded numerical weather prediction to their model of 14 GB grid supply
point groups. Those spatial statistics helped significantly in two of the 14 regions, hurt
significantly in three, and made no measurable difference in the remaining nine. Weather itself was
worth a great deal to them — adding wind and irradiance cut their pinball loss by 40% overall, and
by 60% in North Scotland against 10% in London — so the question is not whether weather matters but
whether *finer* weather does. Artificial Forecasting obtained postcode-level weather forecasts for
two wind-connected primary substations after their wind-connected models had performed poorly, and
reported that the postcode-level forecasts "did not notably improve model performance", naming
better weather data as a next step. What both results say is that finer weather data does not help
everywhere, so the interesting question is *where* it helps. That question is answerable, and
answering it is part of this project: we expect finer weather data to matter most where a
substation's load is dominated by weather-driven generation or heating, which is where NGED most
needs the forecast to be right.

**Weather has bought less than expected at low voltage in the past.** [Haben et al.
(2019)](https://doi.org/10.1016/j.ijforecast.2018.10.007) tested 100 real low-voltage feeders with
both forecast and observed temperature, and found that temperature had no effect on forecast
accuracy, or a negative one. Haben et al. (2019) used data collected in the early 2010s, and we
expect how much weather matters at a substation to be changing quickly, because the thing that makes
a substation weather-dependent is embedded solar generation and heat pumps, and there are far more
of both on the network now than there were then. A primary substation that was almost
weather-independent ten years ago may be strongly weather-dependent today. That is a prediction,
though, not a measurement, and measuring how much weather now explains at NGED's primary substations
is one of the more useful things this project can report.

**A model trained on none of NGED's data may match a model trained on all of it.** Kaas et al.
(2026) tested Chronos-2, a general-purpose time-series model that had never seen their data, against
models trained specifically on those 200 feeders. Chronos-2 beat every purpose-trained competitor on
mean absolute error, 3.8 kW against 4.2 kW. If our heavily engineered models do not clearly beat an
off-the-shelf model given none of our training data, that is important information about the value
of the whole experimental programme, and we will report it.

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

**[UK Power Networks' NIA_UKPN0104](https://smarter.energynetworks.org/projects/nia_ukpn0104/)**
(2024–2026, £389,444, with Open Climate Fix and Sheffield Solar) infers the capacity of unmetered
solar sitting behind each primary substation from half-hourly substation load and weather, then
forecasts that generation at primary substations. It is the direct predecessor of gap 5 below, and
Open Climate Fix is a partner in both.

**[SSEN FastTrack](https://smarter.energynetworks.org/projects/10166254/)** (Strategic Innovation
Fund, Alpha 2025–2026) is building a probabilistic load forecast substation by substation, rolled up
to a grid supply point view. **[SP Energy Networks'
Predict4Resilience](https://smarter.energynetworks.org/projects/10061710/)** drives a probability
distribution of network faults per district from an ensemble weather forecast, up to seven days
ahead, in an operational control room — the GB precedent for putting ensemble-derived distributions
in front of network operators.

### Northern Powergrid's Artificial Forecasting is further ahead, and sets the bar

One concurrent project matters more than any paper here. Artificial Forecasting is an Ofgem
Strategic Innovation Fund programme, with about £3.9m of grant across its three phases, run by
Northern Powergrid with Faculty, EV.energy and Oaktree Power, the final Beta phase running to
February 2027. Its deliverables are publicly available on the Energy Networks Association's Smarter
Networks Portal, though the Beta deliverables sit under a separate project registration from the
Alpha ones linked above. Artificial Forecasting does much of what Flexpectation does at primary
substations. It also covers secondary substations, which Flexpectation does not. And at the time of
writing it is further ahead than Flexpectation.

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

## Three studies worth a closer look

### [Bouman et al. 2024](https://arxiv.org/abs/2405.16164) — switch-event detection is solved; feeding it into a forecast is not

Bouman et al. (2024) is the most directly useful paper in this review, because it solves the
detection half of a problem in Flexpectation's scope: spotting when a cable fault or planned
maintenance moves part of a network from one substation to another. It leaves the forecasting half
untouched, and the forecasting half is what Flexpectation would add. Working with Alliander on 180
primary substations at 15-minute resolution over roughly a year, the authors detect the step changes
caused when a cable fault or planned maintenance reroutes part of a subgrid to a different
substation — a step up at one, a step down at the other. Events run from a few minutes to several
months.

Three things transfer:

- **They detect on a residual, not on the load itself.** Alliander maintains an independent
  bottom-up estimate of each substation's load, reconstructed from customer telemetry and modelled
  profiles. They fit and rescale that estimate to the measured series, then hunt for step changes in
  the *difference* between the two. Normal daily and seasonal variation largely cancels, leaving a
  much cleaner signal. NGED has no bottom-up estimate of substation load, but Flexpectation produces
  its own forecast, which can serve as the reference series in the same way.

- **They recover a missing sign, which solves a known problem in NGED's trial area.** Some Alliander
  substations measure only absolute current, so reverse flow appears as a rise rather than a sign
  change — the identical defect at ten of NGED's 12 metered generators. Alliander's bottom-up
  estimate is built from measurements that record the direction of flow, so Bouman et al. (2024)
  take the direction from the estimate rather than from the meter. Any reference series that records
  direction independently would work the same way.

- **When their bottom-up estimate fails, the cause is usually wrong topology data**, not a bad
  algorithm — a warning about the network records that any bottom-up estimate of substation load
  depends on.

### [Gilbert, Browell & Stephen 2023](https://arxiv.org/abs/2206.11745) — why an annual average hides what happens at peak

Gilbert et al. (2023) forecast load at four levels of a hypothetical GB distribution hierarchy, from
a primary substation down to individual households, and combine a conventional half-hourly forecast
with a bespoke daily-peak forecast.

The same comparison gives two answers. Averaged over every period, that combination gains 0.0–0.4%
over the conventional forecast alone — indistinguishable from nothing, and a result that would
ordinarily end the investigation. Restricted to the periods containing the daily peak, the same
comparison gives 5.7% at the primary substation, 9.0% at secondary, 8.2% at feeder level and 6.0% at
household level. Combining the two forecasts was always worth having, and we know that only because
Gilbert et al. (2023) reported both the all-period number and the peak-period number.

A second finding bears directly on the choice of metric. At household level during peak periods,
both of their conventional forecasts are worse than a trivial benchmark based only on the time of
day; only their fused forecast beats it. And the ability to predict peak *timing* falls away as you
move down the levels: at the primary substation, peak timing was predicted more than 20% more
accurately than a long-run seasonal average would have managed; at four of the feeders, no better
than that seasonal average at all. Together, the peak-versus-average gap and the collapse in peak
timing are the strongest measured argument in this review for the tail and exceedance metrics
Flexpectation is building.

### [Pinheiro, Madeira & Francisco 2023](https://doi.org/10.1016/j.apenergy.2022.120493) — the closest analogue in a live setting

A production forecasting system at a Portuguese distribution network operator covers 96,989
secondary substations day-ahead, using real weather forecasts with a realistic 7–8 hour delay. It is
the only study in this review running in live production at national scale, and three of the six
findings above rest on it. Two of its lessons shape how we will report: the fraction of substations
beating a naive forecast belongs alongside any pooled average, and expectations for single-customer
sites should be set low from the outset.

One further result from it is worth taking: combining eight copies of the same model, one fitted per
calendar regime — weekday, weekend, public holiday and so on — with the weights updated as new data
arrived, cut system-level root-mean-square error by 24%. It is the cheapest positive result in the
review.

## Gaps we did not find addressed, and where Flexpectation fits

Seven things are not addressed anywhere in the work reviewed above, academic or operational. All
seven bear on what this project is trying to do, whether as a requirement for the trial area or as
research for the network-wide scale-up. Most are questions a research paper has no reason to ask and
a deployed forecasting service has not yet needed to answer, and in several cases the authors and
engineers concerned name the gap themselves.

1. **Weather ensembles as the source of uncertainty, across the full horizon.** GB practice is
   further ahead than the academic literature here, but stops short of a 14-day horizon. [Taylor and
   Buizza (2002)](https://doi.org/10.1109/TPWRS.2002.800906), which we read in part, pushed all 51
   ECMWF members through a load model for England and Wales daily demand at one to ten days ahead in
   2002, and [Ludwig, Arora and Taylor](https://doi.org/10.1080/01605682.2022.2115411) revised that
   approach in 2023, adding a step we will need: raw ensembles are biased and their spread is too
   narrow, so they look more certain than they really are. They must be bias-corrected before the
   load model sees them, or the resulting uncertainty bands are wrong. What we did not find is
   ensemble-driven uncertainty at half-hourly resolution, per substation, across a full 14-day
   horizon — and both Haben et al. (2021) and Ludwig et al. (2023) ask for exactly that in print.
   Haben et al. (2021) put it as a request "to use post-processed weather ensemble predictions to
   generate multi-step probabilistic forecasts of load at different levels of the low-voltage
   hierarchy". Two further things bear on this gap. First, the ensemble itself is being replaced:
   ECMWF's own machine-learned ensemble, [AIFS-ENS](https://doi.org/10.1038/s44387-026-00073-7), has
   been operational since 1 July 2025 with 51 members, 6-hourly to 15 days, and beats the physics
   ensemble on the majority of variables and lead times;
   [GenCast](https://doi.org/10.1038/s41586-024-08252-9) beats it too. Flexpectation runs on the
   physics ensemble today, and whether a machine-learned ensemble forecasts substation load better
   is a question we can answer directly.

2. **Almost every study here optimises average accuracy, but NGED's question is about the top of the
   distribution.** NGED's question is "how likely is load to cross this limit?", not "what is the
   most likely load?". Almost everything in this review optimises average accuracy, and HEFTCom, the
   largest competition here, scores only the 10th to 90th percentiles. Browell and Fasiolo (2021) is
   the one study in this review that models the upper tail explicitly, and what they found is a
   warning rather than a reassurance: they set reserve at the 99.95th percentile — but they also
   find ordinary quantile regression stops being calibrated somewhere around the 1st and 99th
   percentiles, even with five years of half-hourly data across regions far larger than a
   substation. **How far into the tail a forecast of a single substation stays trustworthy is an
   open question, and it is one this project can answer.** Our series are smaller and noisier than
   the regions Browell and Fasiolo worked on, so we expect a narrower reliable range, and a
   parametric tail is likely to be necessary rather than optional. We will measure where ours stops
   and publish the answer, because a network buying flexibility needs to know which percentile it
   can act on.

3. **A decision metric that holds risk constant, priced in pounds, at distribution level.** Most of
   that decision metric exists already, in pieces. Browell and Fasiolo (2021) fix a risk appetite,
   compute the reserve volume each forecast would need to hold it, and compare — the harder half of
   the job, done across whole grid supply point groups rather than at a single substation.
   Artificial Forecasting's Alpha work calculates the extra flexibility volume that forecast error
   would make a network procure: 20,536 kWh implied by a risk-aware forecast against 5,495 kWh
   actually needed, over two eight-day windows at one near-capacity substation. Its Beta phase goes
   further, flagging exceedances of firm capacity from the 95th-percentile bound and scoring true-
   and false-positive rates against that threshold. What is still missing is the price at
   distribution level. Meteorology has priced forecast decisions this way for decades: [Richardson
   (2000)](https://doi.org/10.1002/qj.49712656313) computed the relative economic value of the ECMWF
   ensemble across the whole range of ratios between the cost of acting on a forecast and the loss
   avoided by acting. Richardson's relative-economic-value curve is the right shape for NGED's
   problem, because each substation has its own firm capacity — the load it can carry safely with
   its largest transformer out of service — and its own cost of being wrong, so a single assumed
   cost ratio is the thing to avoid. Every published version of it at a distribution network,
   though, is denominated in energy volumes rather than in money. Artificial Forecasting does put a
   price on its service, but that is a business case for a programme rather than a score that holds
   risk constant and can rank one forecast against another at one substation.

4. **Keeping switching-contaminated history usable.** Detection has been demonstrated at a real
   network operator, by Bouman et al. (2024), described above. Researchers then respond in one of
   two ways, and both alter the load series the model is trained to predict. Most delete the
   affected data: [Huyghues-Beaufond et al.](https://doi.org/10.1016/j.apenergy.2019.114405) detect
   and remove structural breaks across 342 UK medium-voltage feeders. A smaller strand rewrites it
   instead, to an "as if never switched" level: [Paredes and
   Vargas](https://doi.org/10.1049/iet-gtd.2017.0129) do it across 169 real feeders and report
   better medium-term forecasts for it, and Artificial Forecasting does the same in its
   data-preparation pipeline, rescaling a step-change block onto the level of the most recent one
   when that block's median falls outside the most recent block's 10th-to-90th-percentile range, so
   the history is kept rather than dropped. Artificial Forecasting argues for going further on the
   grounds that demand changes of an order of magnitude, mostly caused by network reconfigurations,
   "cannot be directly handled even by powerful nonlinear models like neural networks" — though they
   add that changes that large are rare at their secondary substations. Gilbert et al. name adaptive
   handling of structural breaks as future work. What we did not find is the alternative to
   correcting the load series at all, and it is the question we want to settle: **can a model be
   given switching-contaminated history as it stands, and still use it?** Instead of correcting the
   series, a model could be fed the difference between what a substation actually metered and what a
   model that ignores network topology expected it to meter. That difference is the same quantity
   Bouman et al. (2024) use to detect switch events, but used as a forecast input rather than as a
   detector, so a reading taken while the network was abnormally configured would still carry
   information without anyone having to estimate a level correction first. Later, at the scale of
   NGED's full network, the aim is to reconstruct the demand each substation would have metered
   under its normal running arrangement. A negative result here would still be valuable: evidence
   that switching cannot be recovered from power data alone would strengthen the case for taking
   switching labels from operational systems instead — a route Artificial Forecasting has already
   identified, naming the incorporation of planned-outage records in its post-Beta roadmap.

5. **Where studies separate demand from generation, the generation is metered; nobody separates out
   the rooftop solar and small wind that no meter sees.** This is the task we call *disaggregation*:
   recovering, from a substation's net flow alone, both the half-hourly output of the unmetered
   solar and wind sitting behind that substation and the installed capacity of that unmetered
   generation. It is a different task from estimating how much of a *metered* generator's capacity
   is available today, which is gap 7 below. Where demand and generation are separated at all, the
   generation is usually metered: Artificial Forecasting models gross demand and customer export
   independently at primary substations, which is more than any paper here does. The unmetered solar
   and wind — the rooftop panels and small turbines that appear only as a dent in a substation's net
   flow — have to be estimated from that net flow. This is being worked on now: UK Power Networks'
   NIA_UKPN0104, with Open Climate Fix and Sheffield Solar, infers unmetered solar capacity behind
   each primary substation and forecasts that generation. In the peer-reviewed literature the
   nearest work stops one step short — [Kara et al.](https://doi.org/10.1016/j.segan.2017.11.001)
   and [Li et al.](https://doi.org/10.1109/TPWRS.2020.3035639) recover the solar signal from
   feeder-head and substation measurements without forecasting it, and the one benchmark we found on
   estimating installed capacity is at secondary substations, which is a level below ours; we read
   only its abstract. What remains open at primary substation level is estimating unmetered *wind*
   generation, and putting both unmetered solar and unmetered wind inside a 14-day forecast that
   states its own uncertainty.

6. **No study we read uses which substation neighbours which; topology enters only as an arithmetic
   constraint.** Topology enters this literature in essentially one form: as the summation
   constraint in hierarchical forecast reconciliation. [Nespoli et
   al.](https://arxiv.org/abs/1910.03976) apply it to real secondary substations and cabinets in a
   Swiss distribution grid and gain up to 10% in root-mean-square error at the top level. A
   summation constraint says only that the substations beneath a bulk supply point must add up to
   it. It carries no information about which substation neighbours which, and it stops holding the
   moment the network is switched into an abnormal running arrangement. That is why a summation
   constraint is not enough for Flexpectation. Otherwise, information is shared across substations
   statistically rather than topologically — one of the four models Artificial Forecasting tested at
   secondary substations, a hierarchical Bayesian linear regression, pools its upper-layer
   parameters across six load-profile clusters, though the model they recommended is trained per
   substation — and Gilbert et al. forecast four levels of a hierarchy separately before naming
   exploitation of that hierarchy as future work. SSEN TRANSITION is the exception that shows the
   value: it used the connectivity map throughout. NGED holds a map of which substations and metered
   generators connect to which, which raises a question nobody in this literature has answered:
   **does knowing the shape of the network make the forecast better, or only more consistent?** The
   map makes it possible to forecast a bulk supply point both directly and by summing everything
   beneath it, and to treat the disagreement between the two answers as a check on both. We will
   report whether it improves accuracy as well.

7. **No study estimates, from a generator's revenue meter alone, how much of that generator's
   capacity is actually available today.** This gap is about the 12 *metered* generators
   Flexpectation forecasts in the trial area, each of which has a half-hourly meter of its own; the
   unmetered rooftop solar and small wind of gap 5 are a separate task. We call the amount of
   generation actually available at a metered site its *effective capacity*: the output it could
   produce right now if the weather allowed, as opposed to its nameplate rating. Turbines go out for
   repair, inverters degrade, and sites are curtailed — told by the network operator to generate
   less than they could. A 20 MW wind farm that has been limited to 14 MW for a month is, for
   forecasting purposes, a different wind farm, and a model trained on its nameplate rating cannot
   see the difference. The same goes for a primary substation with a large metered generator
   connected behind it. Estimating effective capacity is standard practice for the owner of an
   individual wind farm or solar plant, and it does not always need the generator's own
   instrumentation. For wind, [Dantas and Browell (2026)](https://doi.org/10.1002/we.70079) forecast
   73 GB wind farms from the ECMWF ensemble and hit our problem exactly: the metered-output database
   they use "does not include information on the farms' available capacity over time", so rather
   than use a nameplate rating they estimate a time series of available capacity for each farm and
   normalise that farm's power by it before modelling. Dantas and Browell (2026) do at an individual
   wind farm what Flexpectation proposes to do at each of its 12 metered generators. How they
   estimate it sits in supplementary material we could not obtain, so we cannot say whether the
   estimate comes from the metered output alone or leans on outage messages as well. They also used
   a data source Flexpectation will not have, excluding curtailed half-hours with published
   bid-acceptance volumes, which exist for transmission-connected wind farms and not for NGED's
   embedded generators. For solar, the estimate is routinely made from the plant's output rather
   than its internals: the open-source [RdTools](https://doi.org/10.5281/zenodo.1210316) estimates
   degradation and soiling from a plant's alternating-current output together with modelled or
   satellite irradiance, and [Mendonça Severiano et al.
   (2026)](https://doi.org/10.1016/j.solener.2026.114382) classify underperformance across 1,089
   systems from inverter data alone — though they catch clipping, when the panels produce more than
   the inverter can pass through, only about half the time, which is a warning for the six
   half-hourly-metered solar farms in the trial area. Artificial Forecasting gets closest, and does
   it at a substation rather than at an individual generator: its Alpha work calibrates each
   substation's forecast installed capacity down to the fraction actually generated over two years,
   and separately found that the National Energy System Operator's national generator-availability
   signal "almost universally substantially improved results" at wind-connected primary substations,
   while a feature tracking connected generation capacity over time did not help, which they put
   down to too few new connections falling inside the training window rather than to the feature
   itself.

    The gap is therefore real but narrower than it first appears, and it states its own research
    question: **can effective capacity be recovered from a metered generator's revenue meter alone,
    with no turbine telemetry, no inverter data and no curtailment record to check the answer
    against?** That is the one version of this nobody has published, and it is the version
    Flexpectation needs, because a half-hourly revenue meter is all NGED holds for these sites. The
    harder version — recovering the capacity of generation that has no meter of its own, from a
    substation's net flow where generation is mixed with demand — is the disaggregation task of gap
    5, and belongs to the network-wide scale-up rather than the trial area. There is also no
    equivalent of RdTools for wind that works from a revenue meter alone, because the wind
    literature assumes turbine telemetry: its authors are the owners who have it. And part of the
    problem is a data question rather than a modelling one, because much distribution-connected
    curtailment in GB is instructed by the network operator under active network management, so for
    those sites the curtailment component of effective capacity is already known inside NGED. Public
    data exists for testing an estimator before it meets NGED's network: Cubico has released the
    [Kelmarsh](https://doi.org/10.5281/zenodo.8252025) and
    [Penmanshiel](https://doi.org/10.5281/zenodo.5946808) wind farm datasets, which carry turbine
    telemetry with alarm and status events *and* the site's own grid meter for the same period, so
    an estimator built from the meter alone can be scored against the turbine records. NGED's
    specification asks us to track effective capacity over time and, optionally, to combine it with
    the forecast into a "prevailing conditions" view; we intend to use effective capacity to
    normalise each metered generator's series before training. The clearest published demonstration
    of why effective capacity matters is incidental: when Hornsea 1's export cable faulted partway
    through the HEFTCom competition, teams that forecast wind and solar separately adapted to the
    step change in available capacity, while those forecasting the combined total struggled and the
    organisers' benchmark, which ignored it, collapsed.

## How ambitious Flexpectation's plan is, and what could go wrong

The seven items above are not a shortlist to choose from. The plan is to attempt all of them
alongside the core forecast, across several families of model:

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
network-wide scale-up from 2027, as do the disaggregation work of gap 5 — separating out unmetered
generation — and forecasting the network as a network. "What this review excluded, and why" explains
why the differentiable-physics strand is the least well supported of the four.

Attempting all seven means running on the order of hundreds of machine-learning experiments a month,
and that is possible only because of engineering already done. One more experiment now costs almost
nothing to run, and the core forecast is already running on the machine-learning operations
framework that makes it cheap. Most of the effort to date has gone into that framework, and each of
its three design choices answers a failure mode named in [Sculley et al.
(2015)](https://papers.nips.cc/paper/5656-hidden-technical-debt-in-machine-learning-systems)'s
account of the hidden costs of running machine learning in production. Every experiment is fully
specified by a config file, and is tracked automatically from raw data through to result. The
non-obvious choice is the third: an experiment runs through the same pipeline that serves production
rather than through a separate research copy of it, which keeps data preparation from becoming what
Sculley et al. call a pipeline jungle — a debt they trace to research and engineering being kept too
far apart. That machinery exists and works today; the leaderboard view over it is still being built.

Flexpectation's plan is riskier than a narrower one would be, and that is worth saying plainly.
Artificial Forecasting chose a focused agenda and delivered it into live operational use, which is
the right way to get a service running and is why its results are the firmest evidence in this
review. Flexpectation is running a live service for a 32-series trial area while attempting a wider
set of open research questions on a smaller budget. Several of the directions listed above will not
work — that is what makes them research directions rather than engineering tasks — and the honest
expectation is that some deliver clearly, some produce a negative result worth publishing, and some
are abandoned. NGED gets three things regardless of how the research goes: a running 14-day
probabilistic forecast for the 32 trial-area series, a published leaderboard of every experiment we
run, and a documented answer on whether substation demand can be forecast better than it is today.

None of that depends on the seven gaps being closed. The core forecast exists and runs today, each
gap is independently useful so one failing does not strand the others, and a failed experiment costs
compute time rather than staff months.

## What this review excluded, and why

**Behind-the-meter solar disaggregation was excluded because most of it works below NGED's level of
aggregation.** Separating a substation's metered flow into demand and the solar generation hidden
inside it is a large and active field, mostly working on United States smart-meter data at
individual customer level. We excluded it as a body, and kept a handful of citations to return to
when the disaggregation work begins. The exclusion covers our reading list rather than the whole
field: work at feeder aggregation and above is real, and gap 5 above names it.

**Network topology detection was excluded because it needs measurements NGED does not have.**
Inferring the network's wiring from high-resolution synchrophasor measurements is well developed,
but those measurements are not available to this project. That exclusion covers neither gap 4, which
detects switching from half-hourly power alone, nor gap 6, which is about using a connectivity map
we already hold rather than inferring one.

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

**The bulk of the low-voltage forecasting literature** is covered through the Haben et al. (2021)
review of 221 papers rather than read individually, and we have not systematically covered
low-voltage work published since it closed in 2020. The same lead author's open-access book-length
treatment of 2023 is the better entry point for anyone following this up.

**CIRED**, the International Conference on Electricity Distribution, is barely represented here, and
it is the venue this audience is most likely to read — it is where European distribution network
operators publish their own operational work, so it is also where a claim of ours is most likely to
be contradicted. We found one directly relevant paper, [Ruhhütl et al.
(2023)](https://doi.org/10.1049/icp.2023.0476), on load and generation forecasting at substation
level at an Austrian network operator. That is a gap in our search rather than in the field, and we
are closing it: we will search the CIRED proceedings systematically before the next milestone, and
report anything that changes the seven gaps above.

## Publishing results that others can compare against

The problem named at the start of this review — that published energy forecasting results cannot be
compared with one another — is one this project is well placed to help with, and others have started
already: HEFTCom and Energy-Arena both compare methods on common data with a common metric, and
Energy-Arena keeps a live public leaderboard. A third approach — recovering a ranking from the
published literature after the fact — shows what the alternative costs: [Nguyen and Müsgens
(2026)](https://doi.org/10.1063/5.0300682) did recover a defensible ranking of solar forecasting
methods from the published literature, by screening 1,447 studies and hand-extracting 4,687 skill
scores from those that reported one, then statistically removing the effect of ten other factors.
Their finding is that ensemble-hybrid models improve on time-series models by 7 to 27 percentage
points of skill score, while many advanced machine-learning methods gave inconsistent gains. A
comparison can therefore be dug out of this literature, but only at that price, and nobody does it
routinely. Publishing comparable results in the first place is much cheaper. Neither HEFTCom nor
Energy-Arena covers distribution-substation load, which is the level NGED acts at, so we intend to
follow their protocols where they apply rather than invent our own. Some substation data is already
public — NGED's [Connected Data Portal](https://connecteddata.nationalgrid.co.uk/), and Northern
Powergrid's [open data
portal](https://northernpowergrid.opendatasoft.com/explore/dataset/primary-operational-metering/),
which publishes half-hourly metering from its primary substations — and publishing the telemetry
behind our own experiments would make the results reproducible by anyone, which is still rare in the
substation literature, where only 52 of the 221 low-voltage papers reviewed used any open dataset at
all. Alongside it we will publish the evaluation protocol, the metric definitions and the code that
computes them. Artificial Forecasting is moving the same way, with substation-level historical
forecasts and model-performance metrics designed into its Open Data Portal release, and a shared
evaluation protocol between two GB networks would be worth more than either alone.

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
  season, by forecast lead time and by how heavily loaded the substation was.** Breaking it down is
  the point, and conformal prediction does not remove the need to: [Foygel Barber et al.
  (2020)](https://doi.org/10.1093/imaiai/iaaa017) prove that a distribution-free guarantee holds
  only on average across all conditions, never separately for the conditions that matter, so a
  conformal forecast can promise 90% coverage overall while failing at the peaks.
- **Each metered generator's series is normalised by its estimated effective capacity** before
  training, and that estimate is tracked as it changes.
- **Negative results are published too**, including whether an off-the-shelf model given none of our
  data matches our own, and whether sustained experimentation stops yielding improvements.
