# The current state of the art in energy forecasting: a summary

This is a short version of a literature review Open Climate Fix carried out for National Grid
Electricity Distribution (NGED), as part of the Flexpectation project. This summary is meant to be
readable on its own. The full review, which cites 101 sources and gives the evidence behind every
claim here, is [published
online](https://openclimatefix.github.io/nged-substation-forecast/background/energy-forecasting-review/),
and is referred to below as "the full review".

## Executive summary

**No honest review of this literature can name a canonical state of the art.** Energy forecasting
papers measure performance in different ways against different datasets, so the literature cannot
rank the approaches it contains. What it does show is that a gradient-boosted tree is the sensible
place to start — the choice NGED's own Electricity Flexibility and Forecasting System reached
independently in 2021 — while giving no evidence of a large, dependable margin for anything more
sophisticated at substation level. We found no study driving a probabilistic substation forecast
from a weather ensemble across a 14-day horizon, none modelling the tails explicitly at substation
level, and none explicitly modelling unmetered generation inside such a forecast.

**One concurrent GB project is further ahead than Flexpectation, and four of Flexpectation's eight
challenges have no counterpart in that project's published material.** Northern Powergrid's
Artificial Forecasting has run operationally through a full winter flexibility procurement cycle,
which is the clearest available evidence that a forecast of this kind changes what a network does.
The four challenges Artificial Forecasting's published material leaves untouched are tracking the
effective capacity of metered generators; forecasting a substation as if it were always in its
normal running arrangement; inferring unmetered solar and wind from a substation's net flow; and
doing the same for heat pumps, chargers, and batteries. Three published results point against parts
of the plan, and we intend to test all three rather than avoid them. Throughout, the value NGED gets
from the forecast sits in both tails of the distribution: the upper tail, where flexibility
procurement holds demand under a limit, and the lower tail, where curtailment holds export under
that same limit.

The caveat above is worth spelling out. In 2026, no honest review of the energy forecasting
literature can claim to reveal the canonical "state of the art"! That is because (almost) all energy
forecasting papers measure performance in different ways, against different datasets. It's like an
international football tournament where every team plays by different rules, with different size
goals.

Energy forecasting researchers have done great work over the years. But, unfortunately, the
literature does not tell us how those approaches compare against each other, especially in messy
"real world" energy forecasting scenarios. This isn't anyone's fault; it's a systemic failure. And
the industry is already aware of this problem, and people are trying to fix it. But, at the time of
writing, the literature cannot yet tell us the current state of the art solutions for the challenges
that NGED cares about.

Having said that, there have been some valiant attempts to compare multiple forecasting approaches
against the same dataset. But none of these attempts directly address the main challenges relevant
to Flexpectation. Before we discuss those attempts, we must emphasise two reasons for optimism:

## Reasons for optimism

Firstly, whilst we might not know exactly which algorithms provide the best forecasting performance,
we *do* know how to *research and develop* a state of the art forecast. There's no magic. Machine
learning is an empirical science, and most research ideas fail. John Jumper, who shared the 2024
Nobel Prize in Chemistry for his work on AlphaFold, puts the share of research ideas that fail at
around 90%, and treats that rate as an ordinary and necessary feature of doing research rather than
as evidence of doing it badly ([Nobel Week interview](https://youtu.be/nNM1QdmFwIs?t=852), 6
December 2024, from 14:12). So progress comes largely from being able to quickly test many ideas
under identical conditions and carefully measure performance. We have built an MLOps framework that
should allow us to test research ideas as efficiently as possible.

Secondly — and perhaps most importantly — the fact that the industry doesn't yet know the state of
the art is a huge opportunity for the Flexpectation project: We are in a very privileged position
where we can try hundreds of ideas, and test the best ideas in the real world. We have a fantastic
opportunity to make a significant contribution to the energy forecasting industry by publishing our
"leaderboards of ML experiments", and hence help the industry as a whole to better understand how
multiple approaches perform.

## AI disclosure

The bulk of the *ideas* in this literature review are "human". The structure of this literature
review is human; the research questions are human; the text was either written manually or drafted
by Claude and heavily reviewed and edited manually.

We used Claude Code as a "research assistant" for this literature review. Claude Code tirelessly
searches the literature, downloads PDFs, creates tables summarising papers, traces citations
forwards and backwards (to find updates on results published a few years ago), adversarially reviews
its own text to check claims against the source PDFs, finds gaps in the literature, writes little
Python scripts to download data published in the literature to confirm results, etc.

We're confident the facts in Claude's "research notes" are accurate because we configured Claude to
check against the downloaded PDFs (rather than half-remembering information embedded in the large
language model's weights), and because we ran on the order of 100 rounds of agentic adversarial
review and hundreds of manual fact checks. (The "literature review" process we developed is written
up as a Claude Code "skill", viewable
[here](https://github.com/openclimatefix/nged-substation-forecast/blob/main/.claude/skills/literature-review/SKILL.md)).

But — to our tastes — Claude struggles to write readable prose. So the text below has been heavily
re-written (and cut down) by hand.

## What the literature says about the eight challenges Flexpectation aims to solve

Flexpectation's specification breaks into eight challenges. This section takes each in turn: what
the challenge is, what the literature says, and what that means for Flexpectation — followed by the
published results the summary rests on, and what those results do not cover. The coverage is uneven.
The first challenge (probabilistic forecasts of net demand at substations) has enough published
results to tabulate, and the second challenge (forecasting metered generators) is the most mature
field on the list. For most of the remaining six we found no published result that could be compared
against anything, so those are described in prose.

**The table below is the whole argument of this section in one screen: for each challenge, the
closest thing already published, and what is missing from it.** The sections that follow give the
evidence behind each row.

| Challenge | Closest published precedent | What this means for Flexpectation |
|---|---|---|
| 1. Probabilistic net-demand forecasts at substations | [Artificial Forecasting](https://smarter.energynetworks.org/projects/npg_sif_006-1/) at 551 primary substations, [Pinheiro et al. (2023)](https://doi.org/10.1016/j.apenergy.2022.120493) at 96,989 Portuguese secondary substations, [SSEN TRANSITION](https://ssen-innovation.co.uk/transition/) at 13 | A gradient-boosted tree (GBT) is a defensible default for Flexpectation version 1, but the literature paints GBTs as a sensible starting point rather than a proven winner |
| 2. Forecasting metered generators | [Dantas and Browell (2026)](https://doi.org/10.1002/we.70079) on 73 wind farms in Great Britain (GB) from the European Centre for Medium-Range Weather Forecasts (ECMWF) ensemble, [HEFTCom](https://doi.org/10.1016/j.ijforecast.2025.10.005)'s day-ahead portfolio forecast, and [Nguyen and Müsgens (2026)](https://doi.org/10.1063/5.0300682)'s meta-analysis of 4,687 skill scores from 188 solar forecasting papers | Gradient-boosted trees fitted separately for each kind of generator are the standard approach, and what won when teams were scored against each other on the same data. A higher-resolution deterministic forecast beat the ensemble at short lead times |
| 3. Estimating the effective capacity of metered generators | A method for each generation technology separately — including two published wind-capacity estimators, one fitting a power curve and one ratcheting a running maximum — most of them working from a revenue meter alone | Flexpectation version 1 needs an estimator that can track effective capacity downwards, which is exactly where the two published wind methods differ |
| 4. Detecting switching events | [Bouman et al. (2024)](https://arxiv.org/abs/2405.16164) at 180 Dutch primary substations, using a second load estimate built from smart meters; a Korean series of four papers on one feeder; [ATLAS](https://smarter.energynetworks.org/projects/nia_enwl008/) on GB substations in 2016 | The one published result scoring both precision and recall reports F1.5 scores (a blend of precision and recall weighted towards recall, 0 for a useless detector and 1 for a perfect one) between about 0.2 and 0.5, from different detectors at different event lengths, and achieved with a second load estimate NGED does not have, so Flexpectation should expect worse rather than better |
| 5. Forecasting a substation as if it were always in its normal running arrangement | Three published responses: leave the level shifts in ([Huyghues-Beaufond et al. (2020)](https://doi.org/10.1016/j.apenergy.2019.114405)), rewrite the history ([Paredes and Vargas (2017)](https://doi.org/10.1049/iet-gtd.2017.0129)), or adapt to the new level ([de Vilmarest et al. (2024)](https://doi.org/10.1109/TPWRS.2023.3310280)) | Every published solution throws information away. In contrast, Flexpectation version 1 makes the abnormal periods an input to the ML model, and masks the abnormal periods in the training target |
| 6. Detecting faulty metering | [Bouman et al. (2024)](https://arxiv.org/abs/2405.16164)'s Dutch dataset, the one public labelled set we found, which merges metering faults and switching into a single class | There is no GB number to beat, so whatever precision and recall Flexpectation publishes becomes the first — cheap to do and worth doing |
| 7. Disaggregating unmetered solar and wind | [Teng et al. (2023)](https://doi.org/10.1016/j.rser.2023.113662) transferring from fully-metered Dutch substations, and [UK Power Networks' Power Flow to Solar Capacity](https://smarter.energynetworks.org/projects/nia_ukpn0104/), this work's direct predecessor | UK Power Networks' Power Flow to Solar Capacity used the same method on the same kind of GB data, and Open Climate Fix delivered that project too, so Flexpectation starts from that method rather than from scratch |
| 8. Disaggregating heat pumps, chargers, and batteries | [Ostermann and Haug (2024)](https://doi.org/10.1186/s42162-024-00319-1) on aggregated charging demand day-ahead | Heat pumps, chargers, and batteries stay inside net demand in Flexpectation version 1 rather than being forecast separately |

### 1. Producing probabilistic forecasts of net demand at substations

#### The challenge

*Net* demand is *gross* demand minus whatever generation sits behind the substation. Flexpectation
version 1 forecasts the 32 series in NGED's trial area, and version 2 extends that to net demand at
every grid supply point, bulk supply point, and primary substation in NGED's licence areas. Our
forecasts will be half-hourly, 14 days ahead, updated every 6 hours, and probabilistic. NGED acts on
the forecast 1 to 10 days ahead, and the question NGED asks of the forecast is "how likely is net
demand to run outside the substation's firm capacity?" rather than "what is the most likely net
demand?". Two costs hang on the answer, and NGED rates the second at least as highly as the first:
what NGED spends procuring flexibility to hold demand under the limit, and what curtailing embedded
generators through Active Network Management costs to hold export under it. Both costs sit in the
tails of the forecast distribution rather than at its centre, so both are bought by the same thing —
extreme quantiles that are calibrated, at both ends. A quantile is a level the forecast says net
demand will stay below a stated fraction of the time, and a calibrated quantile is one the outturn
crosses exactly that often: the level given as the 99th percentile is exceeded 1 time in 100, no
more and no less. Forecasting net demand is the highest priority of the eight challenges, and the
other seven exist mainly to improve our net-demand forecast.

#### What the literature says

**A large literature exists on the topic of forecasting substation load, but very little of what we
read can be compared with the rest of that literature, and we found no papers driving a
probabilistic substation forecast from a weather ensemble across a 14-day horizon.** Two of the
sources this review draws on share a network operator: all four authors of [Mesarcik et al.
(2025)](https://doi.org/10.1049/icp.2025.1968) are at Alliander, the Dutch distribution network
operator, and two of the six authors of [Bouman et al. (2024)](https://arxiv.org/abs/2405.16164) —
cited under challenges 4 and 6 below — are at Alliander as well, the other four being at Radboud
University. Agreement between the two papers is therefore not independent evidence about how a
method carries from one network to another.

| Source | What they forecast | Level and scale | Horizon | Result, and what it was compared against | Weather |
|---|---|---|---|---|---|
| [Kaas et al. (2026)](https://arxiv.org/abs/2607.01966) | Net load, Germany | 200 low-voltage feeders | 4 days | A general-purpose foundation model that had never seen the data beat every purpose-trained model on average error, 3.8 kW against 4.2 kW | 1–3 h forecasts, so effectively after the fact at the 4-day horizon |
| [Hertel et al. (2026)](https://arxiv.org/abs/2607.15705) | Load, Germany and Portugal | Transmission, plus 200 low-voltage feeders and 287 individual customers | 4 days | Best model beat a day-type persistence forecast by 59.6% at transmission level, 42.3% at low-voltage feeders, 23.3% at individual customers | 1–3 h forecasts at the feeders, reanalysis (a modelled reconstruction of past weather) elsewhere |
| [Browell and Fasiolo (2021)](https://arxiv.org/abs/2103.10335) | Regional net load, GB | Regional: 14 grid supply point groups | Day-ahead | Held the same risk with **up to 24.6% less upward reserve** than a fixed-tail alternative, falling to 3.2% at the least extreme risk level tested | Real forecasts |
| [Pinheiro et al. (2023)](https://doi.org/10.1016/j.apenergy.2022.120493) | Load, Portugal | 96,989 secondary substations | Day-ahead | 42–47% better than the reference benchmark at system level. **At substation level, beat a naive forecast on 83–87% of network-owned and 66–70% of customer-owned sites** (the paper's body text and the caption of a figure on the next page give different pairs of numbers for that statistic, so the ranges span both)  | Real forecasts, 7–8 h old |
| [Gilbert et al. (2023)](https://arxiv.org/abs/2206.11745) | Load, GB | 4 levels: primary substation down to household | Day-ahead | Combining forecasts gained **0.0–0.4% averaged over all periods**, but **5.7–9.0% when restricted to peaks** | None at all |
| [SSEN TRANSITION 2021](https://ssen-innovation.co.uk/transition/) | Net load, Oxfordshire | 13 primary substations, plus their bulk supply points and their 33 kV and 11 kV feeders | 30 minutes to 10 days | **11 of 13 primary substation models below 10%** mean absolute percentage error when fitted  | 40-member ICON-EU ensemble to 4 days, then one deterministic forecast to 10 days |
| [Artificial Forecasting (Northern Powergrid)](https://smarter.energynetworks.org/projects/npg_sif_006-1/) | Demand and export at primary substations; active power at secondary | 551 primary substations with export data, 171 modelled; 729 secondary substations | Day-ahead to week-ahead at primary, evaluated to 11 days; week- to month-ahead at secondary | **About 8% lower mean absolute error** of utilisation rate than the network's existing method  | Real forecasts at primary; none in the published secondary results |
| [Ruhhütl et al. (2023)](https://doi.org/10.1049/icp.2023.0476) | Load and generation, Austria | Primary substations, count not stated | Day-ahead | **3–8% mean absolute percentage error** for load, against no baseline the paper states, so not a target; varying with how industrial and how large the supplied area was. Generation is forecast per technology: photovoltaic to **1–5% of installed power**, run-of-river and biomass to **5–15%** mean absolute percentage error. Linear and Gaussian regression preferred over tree regression and a neural network | Real forecasts of global radiation, temperature, and precipitation, from a weather station chosen per substation |
| [Mesarcik et al. (2025)](https://doi.org/10.1049/icp.2025.1968) | Active power in the medium-voltage grid, Netherlands | Trained on 312 Alliander substations over 10 years; tested on 6 chosen for difficult forecasting behaviour | 2 days | **Mean relative mean absolute error 0.07** at the 50th quantile, against 0.08 for a gradient-boosted machine and 0.09 for a linear model — both OpenSTEF models already in production at Alliander. Error scaled by the signal's own 1st and 99th percentiles, not by a rating | Open-Meteo, 4 variables; their model trained on actual weather where the two baselines trained on 1-hour-ahead forecasts |

#### What this means for Flexpectation

**Building Flexpectation version 1 on a GBT (such as XGBoost) is defensible, but the literature
paints GBTs as a sensible default rather than a proven winner**. A GBT builds its forecast from
hundreds of small decision trees, each one fitted to the error the trees before it left behind.
[NGED's own 2021 Electricity Flexibility and Forecasting System (EFFS)
project](https://smarter.energynetworks.org/projects/wpden03/) picked XGBoost, which gave the best
results of the three methods the project tested and was also the easiest to automate, and no study
we read shows a large, dependable margin for anything more sophisticated than XGBoost at substation
level.

**Both network deployments that actually tried boosted trees kept a simpler model instead.**
[Pinheiro et al. (2023)](https://doi.org/10.1016/j.apenergy.2022.120493), running a live system
drawing on 96,989 Portuguese secondary substations, scored 199 MW root-mean-square error at system
level with a tuned gradient-boosted tree against 191 MW for a generalised additive model — the
boosted tree 4% worse — and rejected the boosted tree on the cost of tuning it and on the
interpretability given up with it. [Artificial
Forecasting](https://smarter.energynetworks.org/projects/npg_sif_006-1/) kept the simpler model when
forecasting customer export at primary substations: measured against the Bayesian ridge regression
they went on to adopt (a linear model that shrinks its coefficients and reports uncertainty on
them), boosted trees "helped some substations but harmed others".

**Neither end of the sophistication scale is a safe bet.** [Mesarcik et al.
(2025)](https://doi.org/10.1049/icp.2025.1968) caution about the uncertainty a boosted tree reports
rather than the accuracy it reaches: on the one substation whose calibration they plot, their
gradient-boosted machine's 95th percentile forecast corresponded to the 80th percentile of the
measured data, while a structured state space model and a linear quantile model both tracked the
ideal calibration line closely. [Hertel et al. (2026)](https://arxiv.org/abs/2607.15705) make the
same point from the other end of the sophistication scale, because their purpose-built Transformer
variant — the neural-network architecture, not the electrical kind — lost to a standard
encoder-decoder Transformer on all three of their datasets.

**What did help was refitting the model every month rather than redesigning it.** On both datasets
where [Hertel et al. (2026)](https://arxiv.org/abs/2607.15705) tried refitting, the retrained model
beat the static one. So, for Flexpectation, the literature suggests that the choice of model family
may matter less than the data, the feature engineering, and how often the model is refitted.

**Read those results knowing that when a paper says "XGBoost" it usually means a model with
considerably less feature engineering than what we plan to implement.** [Kaas et al.
(2026)](https://arxiv.org/abs/2607.01966) give their ML model lagged power, weather, time, and
metadata, and nothing beyond that: no clear-sky index, no wind power curve, no monotone constraints.
[Pinheiro et al. (2023)](https://doi.org/10.1016/j.apenergy.2022.120493) ran the one comparison on
equal terms — their GBT and their generalised additive model — a regression that fits a separate
smooth curve for each input and adds the curves together — received the same features, but that
shared feature set was itself short: a linear trend, load lagged 24 hours and 1 week, time of day, 9
day types, the named public holidays, day of year, and temperature interacted with time of day and
with day of year. That shared feature set carried no irradiance and no wind, though the weather they
downloaded held both. So no published head-to-head we found gives a GBT the feature engineering we
plan to implement in Flexpectation version 1.

**None of the numbers in the table above is a target for Flexpectation, because the studies cannot
be compared even with each other.** [Kaas et al. (2026)](https://arxiv.org/abs/2607.01966) and
[Hertel et al. (2026)](https://arxiv.org/abs/2607.15705) name different models as best, even though
they use data from the same 200 low-voltage feeders in Germany. Inside [Kaas et al.
(2026)](https://arxiv.org/abs/2607.01966), mean absolute error and an overload-decision metric name
different winners again. Neither disagreement is a mistake: the two papers test different sets of
models at different time resolutions, and the two metrics answer different questions.

**Accuracy got worse further down the network in every study that forecast more than one voltage
level, but what shrank is the headroom above a naive rule rather than the usefulness of the
forecast.** [Hertel et al. (2026)](https://arxiv.org/abs/2607.15705) ran the same models against a
day-type persistence baseline on three datasets — a German transmission control area, 200 German
low-voltage feeders, and 287 individual Portuguese clients — and the margin over that baseline
shrank from 59.6% to 42.3% to 23.3% as aggregation fell. Their own gloss is that it is easier to
beat a simple approach on highly aggregated data than on volatile feeder- and client-level data: the
model did not get worse, the problem got harder.

**The one study we found reporting results substation by substation at scale shows what the
shrinking headroom costs at an individual site.** [Pinheiro et al.
(2023)](https://doi.org/10.1016/j.apenergy.2022.120493)'s model beat a "same time yesterday"
forecast at 83 to 87% of network-owned secondary substations but at 66 to 70% of customer-owned
ones.

**NGED's primary substations may not behave the same way, because a primary substation aggregates
far more customers than a Portuguese secondary substation does.** A forecast at a primary substation
may also carry a larger percentage error than one at a grid supply point and still support
flexibility procurement and curtailment decisions just as well, because what NGED needs from the
forecast is a reliable answer to "will this substation run outside its firm capacity?". Whether
decision-usefulness really is flat across voltage levels is something this project can measure, and
we intend to.

**On the rest of the Flexpectation specification — a weather ensemble driving substation-level
uncertainty all the way out to 14 days — we found no published result to lean on, for or against,
and two published papers ask for exactly that.** [Haben et al.
(2021)](https://arxiv.org/abs/2106.00006) end their review of 221 low-voltage papers by naming
"post-processed weather ensemble predictions to generate multi-step probabilistic forecasts of load
at different levels of the LV [low-voltage] hierarchy" as an avenue of future research. Of those 221
papers, 3 used a weather *forecast* and none used an *ensemble* of weather forecasts. [Pinheiro et
al. (2023)](https://doi.org/10.1016/j.apenergy.2022.120493), published after that review closed, is
a fourth paper using a real weather forecast, but its inputs are single point forecasts rather than
an ensemble — so even the largest deployment in this review used no weather ensemble.

**[Ludwig et al. (2023)](https://doi.org/10.1080/01605682.2022.2115411) built exactly that, for
Great Britain's national demand 1 to 6 days ahead.** Their multi-step probabilistic load forecast is
driven by a post-processed weather ensemble, from the same 51-member ECMWF ensemble Flexpectation
uses, and they too ask for the method to be pushed down "to different layers of the energy
hierarchy, including the low voltage level".

**Flexpectation's 14-day horizon sits near the edge of what a weather ensemble can reliably
forecast.** [Buizza and Leutbecher (2015)](https://doi.org/10.1002/qj.2619) found that the lead time
beyond which a weather ensemble stops beating a climatological distribution is 16 to 23 days ahead —
measured on upper-air variables, not on the near-surface temperature and irradiance that drive
substation load.

**Almost every substation-load study we found optimises average accuracy, but NGED's question is
about both ends of the distribution, and that is the one place where the literature gives a direct
warning.** [Browell and Fasiolo (2021)](https://arxiv.org/abs/2103.10335) is the only study we found
that models the tails explicitly, and they find that "below 1% and above 99% the forecasts based on
quantile regression only are not calibrated at any GSP [grid supply point] Group. Therefore, these
quantiles are not suitable for use in decision-making" — and that was with 5 years of half-hourly
data, across regions far larger than a substation. Outside those percentiles Browell and Fasiolo
switch to a fitted parametric tail at each end, and Flexpectation plans to follow Browell and
Fasiolo and fit parametric tails rather than reading extreme quantiles straight off the model. The
lower tail is the one curtailment turns on, because a substation runs closest to its export limit
when embedded generation is high and demand is low.

**All the text above is a verdict on Flexpectation version 1.** The three more sophisticated ML
model families we plan to research in 2027 — pre-trained encoders, connectivity-map models, and
differentiable physics (building the known behaviour of a solar panel, a wind turbine, or a building
into the model, so the model has to learn only what the physics cannot supply) — are planned to
*simultaneously* disaggregate *unmetered* generators, infer switching state and demand together,
which the pipelines of separate models in this literature cannot do. The closing section of this
summary sets out the case for the work we plan in Flexpectation version 2.

**The evidence behind those three ML model families is uneven.**

- **Pre-trained models** have the best support of the three, but the measured result is for a
different kind of pre-training from the one we plan: the general-purpose model [Kaas et al.
(2026)](https://arxiv.org/abs/2607.01966) tested was pre-trained on time series, had never been
trained on their data, and still beat every purpose-trained competitor across 200 German
low-voltage feeders.
- **Connectivity-map models** have been measured on NGED's own published data: [Campagne et al.
(2025)](https://arxiv.org/abs/2507.03690) compare eight graph neural network architectures against
feed-forward, persistence, and foundation-model baselines on French regional load and on the GB
distribution networks' open smart-meter feed — about 2 million meters and 50,000 substations across the areas of NGED and Scottish and Southern Electricity Networks (SSEN) — and the graph-aware models won on both. But their graphs are
built from geographic distance or from correlation between series, never from electrical
connectivity, so whether NGED's own connectivity map improves a forecast is still unanswered.
- A search for **differentiable physics applied to substation demand forecasting** produced no
strong result, and we found nobody aggregating building thermal physics up to a substation and
putting it inside a probabilistic forecast, though the ingredients exist separately.

##### Pre-trained encoders

**The case for pre-training an encoder rests on results from computer vision and Earth observation
rather than from energy forecasting.** The idea is to train one model on a very large body of data
until the model can turn a raw input into a compact numerical summary that keeps what matters and
throws the rest away, and then to freeze that model so it never changes again. Every later job reads
the frozen summary instead of the raw input, including jobs nobody had in mind while the model was
training, and each job needs only a small model of its own and a modest amount of its own data. The
expensive learning happens once and is then shared, instead of being repeated from scratch by every
model that needs it.

**Two recent models show how well the arrangement works.** [Siméoni et al.
(2025)](https://arxiv.org/abs/2508.10104) describe DINOv3, a 7-billion-parameter vision model
trained on unlabelled images. Siméoni et al. keep DINOv3 frozen throughout their evaluation and read
every task off its representations, reporting that fine-tuning "is not necessary to obtain strong
performance" on tasks as different as segmentation, depth estimation, and object detection. [Brown
et al. (2025)](https://arxiv.org/abs/2507.22291) describe AlphaEarth Foundations, which encodes
satellite and other Earth-observation data into one 64-byte embedding per 10-metre cell per year,
and report that the embeddings cut error magnitude by about 24% on average against a representative
sample of other featurisation methods, across a broad set of sparse-data mapping evaluations,
without re-training on any of them.

**What transfers to Flexpectation is the arrangement rather than the numbers.** The breadth matters
as much as the freezing: DINOv3 and AlphaEarth Foundations are each a single encoder serving many
different tasks rather than one encoder per task, and that is what Flexpectation plans for its own
weather encoder — the same frozen representation feeding the substation net-demand forecast, the
metered-generator forecasts, and the disaggregation of unmetered generation.

**Neither result promises that a pre-trained encoder beats hand-designed features.** Brown et al.
report that learned featurisations "don't always outperform designed featurization methods in scarce
data regimes", and present AlphaEarth Foundations as the exception on their own evidence: the one
learned featurisation in their comparison that consistently beat the alternatives they tested. The
gradient-boosted tree on hand-designed features stays the baseline the encoders have to beat.

**The encoders Flexpectation plans to pre-train cover weather and time, and possibly a third for
place, and the machinery for the weather encoder has been built separately from any energy
forecast.** We plan to research a neural network that turns the raw ECMWF ensemble into a calibrated
probabilistic weather forecast in physical units, which a substation model then reads, alongside a
time encoder that learns how people use the calendar — e.g. that Christmas is not an ordinary day —
and possibly a space encoder holding the standing geographic context of each substation.

**Both halves of the weather encoder have been built.** [Rasp and Lerch
(2018)](https://arxiv.org/abs/1805.09091) built the first: a neural network that post-processes a
50-member ECMWF ensemble into calibrated probabilistic 2-metre temperature at 537 German stations 48
hours ahead, cutting mean continuous ranked probability score — a single number scoring a whole
forecast distribution against what actually happened, where lower is better — from 1.16 for the raw
ensemble to 0.78, with a learned per-station embedding one of the two components the authors credit
for the gain. [Mitra and Ramavajjala (2023)](https://arxiv.org/abs/2312.00290) built the second:
they freeze a weather autoencoder and train small models on the frozen representation alone, at
accuracy comparable to purpose-built models, though the targets they predict are further weather
variables rather than anything on a network.

**The nearest we found anyone joining the two is one entrant in HEFTCom, a competition to forecast a
GB wind-and-solar portfolio day-ahead.** [Browell et al.
(2025)](https://doi.org/10.1016/j.ijforecast.2025.10.005) report that team Rnt fed embeddings from
their own AI weather models into downstream neural networks and finished third of the ranked
entrants. What we found nobody doing is pre-training a weather encoder against observations and then
reading a substation's probabilistic load forecast off it, or using a differentiable model of a
solar or wind farm to strip out the variance the engineering explains so that the weather encoder
trains on a clean weather signal.

### 2. Forecasting metered generators

#### The challenge

Of the 32 series in the trial area, 12 are individually metered generators — 6 solar farms, 3 wind
farms, a biofuel plant, a battery, and a gas generator — and each needs the same probabilistic,
half-hourly, 14-day forecast as a substation. Solar and wind are driven by weather the ensemble
supplies directly. The battery, the gas generator, and the biofuel plant are probably dispatched on
market prices and operator decisions.

#### What the literature says

**Forecasting wind and solar from a weather forecast is a well-studied area, and one paper matches
Flexpectation's challenge closely.** Nothing we found forecasts a distribution-connected battery or
gas generator inside a net-demand forecast; the closest case for the biofuel plant is a biomass
forecast at Austrian primary substations, [Ruhhütl et al.
(2023)](https://doi.org/10.1049/icp.2023.0476).

#### What this means for Flexpectation

**A warning for a project built on a numerical weather prediction (NWP) ensemble.** [Dantas and
Browell (2026)](https://doi.org/10.1002/we.70079) forecast 73 wind farms in GB — 34 onshore, 39
offshore — from the ECMWF ensemble, seamlessly from 6 to 162 hours (6.75 days) ahead, and two of
their conclusions bear on Flexpectation. Whether weather-forecast error or weather-to-power
conversion error dominates flips with lead time. Weather-to-power uncertainty dominates the short
term, and weather-forecast uncertainty dominates the mid-term. The transition between the two
typically falls 2 to 3 days ahead, arrives earlier for offshore farms than for onshore farms, and
varies dramatically between farms. And a deterministic forecast at higher resolution beat the
ensemble at short lead times.

**Gradient-boosted trees, fitted separately for each kind of generator, is the standard approach in
the literature, and what won when teams were scored against each other on the same data.** [Dantas
and Browell (2026)](https://doi.org/10.1002/we.70079) model the weather-to-power relationship with
quantile regression on gradient-boosted trees, fitting a separate model for each quantile. In
HEFTCom the winning team fitted gradient-boosted trees separately for wind and for solar and
separately for each weather source. Of the top 10 teams, 9 forecast wind and solar separately before
combining the two forecasts. And [Browell et al.
(2025)](https://doi.org/10.1016/j.ijforecast.2025.10.005) conclude that gradient-boosted trees
remain competitive for day-ahead wind and solar forecasting, with performance depending heavily on
implementation. NGED's own EFFS project selected XGBoost when it evaluated model families.

**One result cuts the other way, though team Rnt's route is not an argument against trees.** Rnt
finished third in HEFTCom's forecasting track using no tree-based model at all, feeding embeddings
from machine-learned weather-forecasting models they built in-house into downstream neural networks
that predicted wind and solar generation — a route that rests on building and running a weather
model, not on a different downstream model family.

**The largest meta-analysis of solar forecasting we found puts individual machine-learning models
level with classic statistical ones at the range NGED acts on, and only combinations ahead.**
[Nguyen and Müsgens (2026)](https://doi.org/10.1063/5.0300682) meta-analyse 4,687 skill scores
extracted from 188 solar forecasting papers, fitting a separate regression for each horizon band.
Their baseline class is classic statistical time-series models — the autoregressive integrated
moving average (ARIMA) family, exponential smoothing (ETS, for error, trend, and seasonality), and
multivariate relatives such as autoregressive models with exogenous inputs (ARX) — and every figure
in the table is percentage points of skill score against that baseline. In that table "ensemble"
means a combination of forecasting models, not a weather ensemble.

| Model class | Intra-hour (up to 1 hour) | Intra-day (1 to 6 hours) | Day-ahead (over 6 hours) |
|---|---|---|---|
| Ensemble-hybrid: average several models, and chain one model's output into the next as an input | +12.8 | +21.2 | **+7.0** |
| Pure ensemble: aggregate several models, without the chaining | not significant | −7.0 | **+8.3** |
| Hybrid: the chaining without the aggregating | +8.6 | −19.3 | −11.3 |
| Image-based: sky or satellite imagery | not significant | +10.3 | not significant |
| Individual machine learning, including gradient-boosted trees | −3.1 | not significant | not significant |
| Regression | −11.0 | −5.3 | not significant |
| The weather model's own irradiance field, used directly as the forecast | not significant | −17.4 | **−14.3** |

**Read the model-class table above as the effect of the model class alone, not of the model plus its
data.** Model class and input are separate variables in the same regression, so each model-class
figure is estimated with the inputs held constant, and their "classical time-series" class is not
weather-blind: it explicitly includes autoregressive models with exogenous inputs and vector
autoregressive models, which is where a weather forecast enters a classical model. The comparison is
therefore between a time-series model and a machine-learning model given the same data, and at
day-ahead range the machine-learning model wins nothing — the weather forecast itself is worth far
more than the model wrapped around it, as the next table shows.

**Two limits come with reading the table this way.** Their regression carries no interaction between
model class and input, so it cannot detect whether a machine-learning model exploits a weather
forecast better than an autoregressive model does. That question is the one that matters for
Flexpectation. And only 19% of the 4,687 observations use numerical weather prediction as an input
at all, against 91% that use lagged power, so most of the evidence separating the model classes
comes from models with no weather forecast in them.

**The bottom row is the weather model used raw, and for most of that sample no power curve is
involved.** The class is the numerical weather prediction irradiance field itself — usually global
horizontal irradiance, at most post-processed or averaged across several weather models — used as
the forecast rather than fed as an input to a fitted model. Of the 188 papers in their sample, 118
forecast irradiance rather than photovoltaic plant output. Only 70 papers forecast the output of a
photovoltaic plant, so for most of the sample the weather model's irradiance field is directly
comparable to the irradiance those papers forecast. Their regression separates the model class and
the forecast target as separate variables, so the 14.3-point penalty is estimated with the target
held constant, but the authors never report which targets the numerical-weather-prediction papers
were forecasting. Their own advice is to exhaust the simple models first, because classical
statistical time-series methods "still have very good performance compared to more complex methods
such as individual ML models".

**Most of NGED's metered generators are solar, and the largest meta-analysis of solar forecasting we
found confirms the importance of NWP inputs at the lead times Flexpectation cares about.** Numerical
weather prediction is the largest input effect [Nguyen and Müsgens
(2026)](https://doi.org/10.1063/5.0300682) measure, and the inputs that pay at short range carry the
opposite sign at day-ahead range. Percentage points of skill score again:

| Input | Intra-hour (up to 1 hour) | Intra-day (1 to 6 hours) | Day-ahead (over 6 hours) |
|---|---|---|---|
| Numerical weather prediction | −9.0 | −2.3 | **+11.6** |
| Locally measured weather | not significant | +9.1 | +5.1 |
| Lagged solar power | +5.7 | +8.2 | **−6.4** |
| Data from neighbouring sites | +3.6 | +3.9 | −5.5 |

Each input is a yes-or-no variable rather than a choice between alternatives, so one model can carry
several. Their sample is deterministic forecasting of irradiance or plant output rather than
probabilistic substation net demand.

**For generators, the benefit from better weather-to-power physics is largest at short lead times.**
Differentiable physics (DP) attacks the weather-to-power half of the error, so on [Dantas and
Browell (2026)](https://doi.org/10.1002/we.70079)'s measurement DP has most to offer inside the
first 2 to 3 days of the 1-to-10-day window NGED acts on, and less beyond it, where the weather
forecast itself is the binding constraint.

**Adding a learned residual to a physical generator model is established practice.** [Gijón et al.
(2025)](https://arxiv.org/abs/2502.07344) fit a physics-inspired power model to a wind farm of four
turbines and train a second model on the residual, cutting the physics model's mean absolute
percentage error by 37% and its mean absolute error by 28%, with conformalised quantile regression
supplying the uncertainty. The hybrid gains that margin over the physics model alone; against a
purely data-driven model given the same eight inputs it "essentially matches" rather than beats, so
adding the physics model buys interpretability at no cost in accuracy.

**But Gijón et al. predict power from measured wind rather than forecasting it days ahead.** We
found nobody putting a differentiable model of a generator inside a network's probabilistic
net-demand forecast. On lead time alone, then, the larger differentiable-physics prize for
Flexpectation would be on the demand side rather than the generation side.

**A second reason to try differentiable physics on generators, beyond the accuracy gain above, is to
infer the metadata Flexpectation is not given.** The generation forecasts in this literature are
given metadata about each generator: [Teng et al.
(2023)](https://doi.org/10.1016/j.rser.2023.113662) are given each site's capacity, and HEFTCom's
portfolio was one named 1.2 GW offshore wind farm plus the solar capacity of a region. When an
export-cable fault cut that wind farm's available capacity mid-competition, the winning team clipped
its quantiles to the capacity implied by the outage notices the farm is obliged to publish, while
the organisers' benchmark ignored the fault and, in [Browell et al.
(2025)](https://doi.org/10.1016/j.ijforecast.2025.10.005)'s words, "performed extremely poorly as a
result". NGED's embedded generators publish no outage notices of that kind.

**NGED's Embedded Capacity Register gives a registered capacity for generation of 50 kW and above,
and none of the physics.** The August 2026 edition lists 5,598 connected generators totalling 11,456
MW, of which 4,202 sites and 5,958 MW are solar. But a registered capacity is *contractual* rather
than operational — the export limit is the one "permitted as per the connection agreement" — and the
register carries no panel tilt, panel azimuth, or ratio of direct-current to alternating-current
rating.

**A differentiable model could infer both the operational capacity and the panel orientation of each
generator, and each of those two inferences has been made to work on its own.** [Pierrot and Pinson
(2024)](https://doi.org/10.1080/00401706.2024.2350421) treat a wind farm's capacity as a
time-varying bound fitted jointly with the forecast, and beat probabilistic persistence by 34.2% on
continuous ranked probability score over a 5-month test period, drawn from 14 months of data, at the
Anholt offshore wind farm, though their one clean test of tracking the bound on its own gained
2.43%, and [Meng et al. (2020)](https://doi.org/10.1016/j.solener.2020.09.077) infer the tilt and
azimuth of 13 roof photovoltaic systems in the Netherlands to mean absolute errors of 4.3° and 4.5°,
matching the shape of each system's hourly output against plane-of-array irradiance from a station
up to 195 km away. Because both curves are normalised before matching, their method needs no
nameplate rating. Neither method sits inside a substation's net-demand forecast. Flexpectation would
have to put the method there itself.

**What better orientation metadata is worth to a forecast is a number we have not found in the
literature, so Flexpectation treats it as a hypothesis to test rather than a settled prize.** [Meng
et al. (2020)](https://doi.org/10.1016/j.solener.2020.09.077) and [Saint-Drenan et al.
(2015)](https://doi.org/10.1016/j.solener.2015.07.024) both recover a system's tilt and azimuth from
its metered alternating-current power output paired with an irradiance series measured somewhere
else — a weather station up to 195 km away for Meng et al., the HelioClim-3 satellite database for
Saint-Drenan et al. — and land within a few degrees, but report their accuracy in degrees alone.
Saint-Drenan et al. also found that an azimuth fitted 5° from the true azimuth gave better
simulations than the true value, because the fit balances the systematic error of the physical model
— so accuracy in degrees is the wrong target. What matters is an *effective* tilt and azimuth that
make the forecast right.

**The two cases Flexpectation faces need different machinery, and the dividing line is the limit
Saint-Drenan et al. state.** For a single metered site, fitting tilt, azimuth, and the effective
direct- and alternating-current capacities is the plan — by gradient descent inside the forecast
rather than by grid search, so that the fit stays joint and probabilistic. For unmetered solar
behind a substation their algorithm "performs poorly", because it assumes one orientation per plant
where the series is "the aggregated production of modules with different orientations".
Flexpectation therefore estimates no single orientation per substation: the fleet model represents
the aggregate as a learned mixture of east-, south-, and west-facing basis shapes, with a soft clip
standing in for many differently-sized inverters saturating in turn.

**The trial area's battery, gas generator, and biofuel plant each need a method, and the literature
supplies one to borrow for the battery, none for the gas generator, and a partial one for the
biofuel plant.** For the battery, [Bian et al. (2024)](https://doi.org/10.1109/TSG.2023.3303469)
recover a price-taking storage operator's own optimisation parameters from historical prices and
observed dispatch. We found no method worth borrowing for the gas generator, and what little exists
forecasts a gas plant's own output directly rather than as a component of a substation's net demand.
For the biofuel plant, [Ruhhütl et al. (2023)](https://doi.org/10.1049/icp.2023.0476) forecast
biomass generation behind each Austrian primary substation from the previous day's generation,
scaled to installed power and spread across the day as a constant band, to a mean absolute
percentage error of 5 to 15% — the same shape of problem, though a biomass station burning solid
fuel is not the same plant as a biofuel one.

### 3. Estimating the effective capacity of metered generators

#### The challenge

We call the amount of generation actually available at a metered site its *effective capacity*: the
output it could produce right now if the weather allowed, as opposed to its nameplate rating.
Turbines go out for repair, inverters degrade. A 20 MW wind farm that has been limited to 14 MW for
a month is, for forecasting purposes, a different wind farm, and a model trained on its nameplate
rating cannot see the difference.

#### What the literature says

**A method exists for each generation technology separately, but we found none run across a mixed
fleet of individually metered generators at a distribution network, and the two studies that measure
what estimating capacity is worth downstream measure it for wind alone, at national or single-farm
scale.**

#### What this means for Flexpectation

**Flexpectation version 1 needs an estimator that can track effective capacity downwards, and that
is exactly where the two published wind methods differ.** [Dantas and Browell
(2026)](https://doi.org/10.1002/we.70079) needed available capacity for the same reason we do, and
rather than use a nameplate rating they estimate a time series of available capacity for each farm
from that farm's own metered production, needing no capacity register and no outage messages. The
general shape of that capacity-estimation rule is a running maximum of production, which ratchets
upwards and never comes back down. [Viotti et al. (2026)](https://doi.org/10.1002/we.70136) fit the
most likely capacity time series instead, by quadratic optimisation against a capacity factor
simulated from reanalysis weather and a power curve, and they publish a monotonic variant alongside
a non-monotonic one. The direction of travel is what matters for NGED: a turbine out for repair for
a month makes effective capacity *fall*, and a ratchet cannot follow it down. Flexpectation version
1 will therefore implement estimators that can fall as well as rise.

**The published numbers favour fitting over ratcheting, and the variant Flexpectation needs is the
variant that gave the better forecast.** [Viotti et al. (2026)](https://doi.org/10.1002/we.70136)
say the running maximum "requires monotonically increasing capacity and relies on frequent high wind
events", and report **27.2% lower normalised mean absolute error** than the running maximum at
quantifying capacity after a new wind farm connects.

**The 27.2% is scored by their monotonic variant, which assumes capacity only ever rises.** They
publish a non-monotonic variant alongside it, which can follow capacity down when a turbine goes out
for repair, and that is the version NGED needs. On this test the monotonic variant's error is 31%
below the non-monotonic variant's. But the test only ever adds capacity: it simulates a new wind
farm connecting, so it measures how well each variant spots a step *up*, and says nothing about how
either handles a step down.

**Downstream the ranking reverses.** The non-monotonic variant produced the lowest day-ahead
forecast error, **2.0% below** a model normalised by the running maximum across Sweden as a whole,
and the authors read that 2.0% gap as the non-monotonic variant picking up real changes in available
capacity.

**Two things temper both figures for NGED.** Viotti et al.'s target is a Swedish bidding zone rather
than a single farm, and they report that at 5-minute resolution the running maximum is already a
robust estimate of one farm's installed capacity, so the fitting earns its advantage on hourly,
region-aggregated data. Whichever estimator wins, normalising by effective capacity stays a
hypothesis to test rather than a settled preprocessing step, because no study we found has measured
whether it improves the forecast NGED acts on.

### 4. Detecting switching events

#### The challenge

When a cable fault or planned maintenance moves part of a network from one substation to another,
the load the first substation meters steps down. Each substation that picks up part of that
transferred load records a matching rise, with no change in the underlying demand. The pick-up is
usually shared across two or three neighbouring substations rather than landing on one. NGED's
substations spend roughly a tenth of their operating time in an abnormal running arrangement.
Switching labels exist for the trial area but not for the wider network, so any method meant to
scale to the wider network has to work from power measurements alone.

#### What the literature says

**We found several papers on detecting switching events from metered load, but all these approaches
only consider one substation at a time.** [Bouman et al. (2024)](https://arxiv.org/abs/2405.16164)
detect switching at a real network operator, but detect it in the gap between the substation's own
meter and a second estimate of the same load, built from smart-meter and bulk-customer readings
taken below the substation. A Korean series of four papers detects load transfers on a distribution
feeder from that feeder's own load alone. All four papers are open access, and all four score
against the same nine logged transfers on the Kimhwa distribution feeder in Gangwon province,
measured hourly through 2019.

| Paper | Method | Logged transfers found |
|---|---|---|
| [Kim et al. (2020)](https://doi.org/10.3390/en13174358) | Long short-term memory network, flagging where measured load departs from its prediction | 7 of 9 |
| [Kim et al. (2022)](https://doi.org/10.3390/en15041441) | Polynomial and standard-pattern preprocessing | 7 of 9, and 7 of 7 on a second feeder |
| [Kim (2024)](https://doi.org/10.5370/KIEE.2024.73.11.1873) | A moving average and a moving standard deviation, thresholding the residual of a seasonal-trend decomposition | **8 of 9** |
| [Kim (2025)](https://doi.org/10.5370/KIEE.2025.74.11.1757) | Robust seasonal-trend decomposition, a Haar wavelet transform of the residual, then Pruned Exact Linear Time changepoints, then an isolation forest over each candidate | 7 of 9 |

Every one of those counts is the share of logged events found, and no paper in the series reports a
false-alarm rate. The scores do not track how elaborate the method is: the simplest of the four, a
threshold on a decomposition residual, found the most events, and [Kim
(2025)](https://doi.org/10.5370/KIEE.2025.74.11.1757)'s pipeline — the one closest to what
Flexpectation plans — found 7 of the 9, an average detection rate of 78%.

Electricity North West's [ATLAS](https://smarter.energynetworks.org/projects/nia_enwl008/) project
sorted step changes into erroneous data and network reconfigurations on GB substations in 2016, from
power measurements alone, and published no precision or recall for either rule.

#### What this means for Flexpectation

**Only one published result scores switching detection on both precision and recall, and the scores
it reports are low.** [Bouman et al. (2024)](https://arxiv.org/abs/2405.16164) score their detectors
with the F1.5 score, which blends precision — the share of flagged points that really were switching
— with recall — the share of switched points the detector flagged — weighting recall the more
heavily of the two. An F1.5 score of 1 is a perfect detector and 0 is a useless one, so higher is
better.

**On events shorter than 3 days Bouman et al.'s best detector reaches about 0.2, and on events of 42
days or longer about 0.5.** Those two scores come from different detectors, because no single method
they tried wins across the range. Both figures were achieved on a Dutch network, with the help of a
second load estimate NGED does not have.

**NGED's switches are usually partial and fan out to two or three substations, so we should expect
worse F1.5 scores than 0.2 to 0.5 rather than better.** Do not judge the difficulty from how obvious
a switch looks on a chart. A negative result is worth having here, because evidence that switching
cannot be recovered from power measurements alone would justify extracting switching labels from
NGED's operational systems instead of continuing to infer them.

**The one directly useful paper detects switching but never forecasts, and forecasting is the half
Flexpectation would add.** [Bouman et al. (2024)](https://arxiv.org/abs/2405.16164), working with
the Dutch network operator Alliander, study 180 primary substations at 15-minute resolution over
roughly a year, detecting events that run from a few minutes to several months. Alliander's purpose
is capacity planning: a switch pushes the maximum and minimum load a substation records to the wrong
value, and those two extremes decide whether the substation needs a bigger transformer, so the
detected periods are cut out of the history before the extremes are read off. Flexpectation needs a
forecast that keeps running through a switching event instead.

**Flexpectation will model its own reference series rather than measure one.** Alliander's bottom-up
estimate gives [Bouman et al. (2024)](https://arxiv.org/abs/2405.16164) a second opinion on what
each substation's power should have been. Flexpectation plans to produce that second opinion from
the substation's own meter plus weather and the calendar. The first attempt is classical: a multiple
seasonal-trend decomposition of each series into a trend and daily, weekly, and annual cycles,
leaving a remainder in which a switch shows up as a sustained level shift. The second uses the
project's existing XGBoost machinery, trained with no power-lag features, so that an earlier
switching event cannot contaminate the expected-power estimate. Neither route needs metering from
below the substation.

**Flexpectation also plans to investigate using a signal that Bouman et al.'s
one-substation-at-a-time method cannot see: the power has to go *somewhere*.** [Bouman et al.
(2024)](https://arxiv.org/abs/2405.16164) score each substation against its own history — "the
current analysis considers one year of measurements for one station at a time" — so nothing in their
method asks whether the power that left one substation turned up at another.

**Flexpectation intends to look for both sides of the transfer.** When one substation's metered
power drops, the substations that picked the load up should rise at the same moment, and their rises
should sum to the drop. A step whose rise and drop fail to balance is more likely a meter fault or a
one-off than a switch. That mismatch is where a per-substation detector spends its false positives.
The catch is that an NGED transfer usually fans out across two or three neighbours, so the search
runs over subsets of neighbours rather than over pairs, and the balance holds only approximately.

**We looked for a method that checks both sides, and found none.** The search ran across OpenAlex,
Semantic Scholar, Crossref, arXiv, the works citing [Bouman et al.
(2024)](https://arxiv.org/abs/2405.16164), and the project titles on the Energy Networks
Association's Smarter Networks Portal.

**The closest published precedent is a 1984 regression written for long-range planning.** [Willis et
al. (1984)](https://doi.org/10.1109/TPAS.1984.318713) correct annual peak-load curve fits rather
than detecting an event at a point in time, and their regression needs neither the size nor the
direction of a transfer as an input. The title names a "load transfer coupling" regression, which
suggests the fit couples the substations that exchange load — the feature that would make it the
closest precedent — but we could not obtain the full text to check, and the abstract does not say.

### 5. Forecasting a substation as if it were always in its normal running arrangement

#### The challenge

NGED plans its network against what each substation would carry under its normal running
arrangement, and that same quantity is what Flexpectation has to predict — including for a
substation that has been sitting in an abnormal arrangement for weeks. Forecasting *through* an
abnormal arrangement is a weaker requirement, and not the one NGED has. A model can take lagged
power inputs from inside an abnormal period and stay well-behaved anyway, yet still report what the
substation will carry rather than what the substation would have carried under its normal
arrangement. Predicting that quantity makes the forecasting target something that was never metered,
and leaves the training history contaminated: past readings taken while the network was abnormally
configured describe a different scenario from the scenario being forecast.

#### What the literature says

**Researchers respond in one of three ways:** leaving the level shifts in and paying for them, as
[Huyghues-Beaufond et al. (2020)](https://doi.org/10.1016/j.apenergy.2019.114405) do; rewriting the
history, as [Paredes and Vargas (2017)](https://doi.org/10.1049/iet-gtd.2017.0129) do; or adapting
to the new level, as [de Vilmarest et al. (2024)](https://doi.org/10.1109/TPWRS.2023.3310280) do.

**One published system chose its model on robustness to switching rather than on accuracy.**
[Ruhhütl et al. (2023)](https://doi.org/10.1049/icp.2023.0476) compared linear, tree, Gaussian, and
neural network regressions for day-ahead load at Austrian primary substations, and report that the
Gaussian model "has the lowest MAPE [mean absolute percentage error] of all regression models" but
"is barely able to calculate predictions when there is a major deviation from the normal switching
status", while linear regression "is a little less accurate but is very flexible in terms of
deviations from the switching status".

**Ruhhütl et al. also clean "major deviations of the normal switching status" out of the training
data before fitting, which removes those periods from the training set rather than correcting them
to what the normal arrangement would have carried.** Neither the size of the accuracy sacrifice nor
the size of the switching failure is quantified, so the paper shows that an operator traded accuracy
for switching robustness without saying what the trade cost. We found one substation study that
conditions its forecast on an operating-state label, for a switch of a different kind, and none that
both hands a model the record of when the network was abnormal and refuses to let the model predict
those periods.

#### What this means for Flexpectation

**Every published solution we found throws information away.** Leaving the level shifts in the data
hurts performance, rewriting history erases the level shifts, and adapting to the new level forgets
that a switch happened. Adapting is disqualifying here, because the quantity NGED needs is what the
substation *would* have carried under its normal arrangement. Flexpectation will therefore record
when the network was abnormal and hand that record to the model, which no published method we found
does.

**Rewriting the history is the fallback because it is the only response with a published precedent
behind it.** [Paredes and Vargas (2017)](https://doi.org/10.1049/iet-gtd.2017.0129) rewrite the
history to the level it would have had if the switch had never happened, across 169 real feeders,
and report better medium-term forecasts for it; Artificial Forecasting does the same in its
data-preparation pipeline.

**The fix is a level shift applied to the *older* half of each series.** Paredes and Vargas measure
how far average demand moved across the step and add that difference to every reading before it, and
the variant they recommend uses a separate difference for each hour of the day and each day of the
week rather than one number for the whole series. Paredes and Vargas take the event times from
expert identification rather than from a detector, since detection was not their subject. Adaptive
models are the live alternative — they track a new level once it arrives, including one that arrives
abruptly — but a model that simply adapts to a new load level cannot report what the substation
would have carried under its normal arrangement, which is the quantity NGED needs.

**Flexpectation version 1 feeds the model its switching-contaminated history deliberately, as
information rather than as damage: the abnormal periods become an input, and they stop being a
target.** Instead of correcting the series, the forecasting model can be fed the difference between
what a substation actually metered and what a topology-blind reference model expected it to meter.

**The plan has two parts.** First, label each substation's abnormal running arrangements explicitly
and hand those labels to the model as features, so the model can read its own lagged power inputs
correctly when a lag falls inside an abnormal period. Second, drop the abnormal half-hours from the
training target, so the model is never asked to predict an abnormal arrangement.

**The nearest published precedent for the first half sits inside a substation, where the
never-metered-target problem does not arise.** [Liu et al.
(2019)](https://doi.org/10.1109/ACCESS.2019.2951422) fit a separate regression per substation
operating condition, though their switching moves load between transformers inside one substation,
so the substation total stays metered throughout.

**For the second half, the mechanism has a canonical statement outside energy.** [Salinas et al.
(2020)](https://doi.org/10.1016/j.ijforecast.2019.07.001) state the mechanism for a probabilistic
forecaster, motivated by retail stock-outs, and say they omitted the experiments for it. Searching
OpenAlex, Crossref, and arXiv, we found no load-forecasting study reporting what dropping
contaminated periods from the training target is worth, so Flexpectation will have to measure that
itself.

**Later research will go further and treat the normal-arrangement demand as a latent variable to be
inferred, rather than a series to be repaired first**, through a differentiable-physics model of
each substation with separate photovoltaic, wind, and demand components. Recovering a demand the
meter never saw is mature where demand is censored — airline revenue management calls it
unconstraining, and retail and electric-vehicle-charging work calls it censored-demand recovery, as
in [Hüttel et al. (2023)](https://arxiv.org/abs/2301.06418) — but censoring is one-sided, so the
observed value bounds the latent demand from below, whereas an abnormal running arrangement
substitutes a different set of customers and can read either side of the normal-arrangement demand.
We found no published model that recovers a latent normal-running-arrangement demand for a
distribution substation.

### 6. Detecting faulty metering

#### The challenge

NGED's telemetry carries stuck values that repeat unchanged for hours or days, zeros that mean "no
reading" rather than "no load", physically impossible values, and gaps running from a single
half-hour to several months. Of the 32 series in the trial area, 10 are metered in apparent power
only, so they report magnitude without direction and reverse flow appears as a rise: at one primary
substation the meter bounces off zero on sunny days, when a solar farm behind it exports. A model
trained on uncleaned data learns the fault, and a forecast that fails silently because its recent
history was stuck is worse than one that says it is degraded.

#### What the literature says

**Faulty metering is usually a data-cleaning step mentioned in passing rather than a problem in its
own right.** The only public labelled dataset we found is Dutch. Western Power Distribution, NGED's
predecessor, attempted to recover the direction of flow from a magnitude-only meter, and an
automatic version of that recovery is still open.

#### What this means for Flexpectation

**There is no GB number to beat, so whatever precision and recall we publish becomes the first —
cheap to do and worth doing.** The practical constraint is that the only public labelled data is
Dutch and treats switching and measurement error as a single class: useful for building a detector,
but not for validating the separation between challenges 4 and 6 that cleaning NGED telemetry
actually requires.

**The most useful published method treats faulty metering and switching as one challenge, and its
sign-recovery technique addresses NGED's magnitude-only meters directly.** [Bouman et al.
(2024)](https://arxiv.org/abs/2405.16164) treat measurement errors and switch events as the two
things that must be filtered out before substation measurements can be used, and detect both on the
same residual. Three network-innovation projects in GB tackled faulty metering substantively, one of
them as its whole subject — Electricity North West's ATLAS, UK Power Networks' Distribution Network
Visibility, and NGED's own Time Series Data Quality. None of the three reports how often its checks
are right, and none published its labels.

### 7. Disaggregating unmetered solar and wind from a substation's net flow

#### The challenge

Rooftop panels and small turbines appear only as a dent in a substation's net flow. Recovering both
the half-hourly output of that unmetered generation and its installed capacity, from the net flow
alone, is what we call *disaggregation*. Disaggregation is a different task from estimating how much
of a *metered* generator's capacity is available today, which is challenge 3. Disaggregation is a
stretch goal for the trial area and a requirement for the network-wide scale-up.

#### What the literature says

**Splitting generation out of a substation's net flow has been done where the generation is metered
or its capacity is read from a register.** Inferring the capacity from the net flow instead has also
been done, but at low-voltage substations serving tens of customers rather than at a primary.
Uncertainty and a multi-day horizon each appear in this literature, but never together.

#### What this means for Flexpectation

**No other challenge has a predecessor as close as UK Power Networks' Power Flow to Solar Capacity,
which used the same method on the same kind of GB data and was delivered by Open Climate Fix, the
partner delivering Flexpectation.** The warning is not to read published transfer-learning accuracy
as achievable here: [Teng et al. (2023)](https://doi.org/10.1016/j.rser.2023.113662) are given a
population of fully-metered substations to learn from and are told each site's capacity, whereas
inferring that capacity is half of what NGED needs.

**The direct predecessor of this work is running now in GB.** [UK Power Networks' Power Flow to
Solar Capacity](https://smarter.energynetworks.org/projects/nia_ukpn0104/) (2024 to 2026, £0.4
million), which Open Climate Fix worked on, infers the capacity of unmetered solar sitting behind
each primary substation from half-hourly substation load and weather, then forecasts that
generation. Open Climate Fix is a partner in both Power Flow to Solar Capacity and Flexpectation, so
Flexpectation starts from the Power Flow to Solar Capacity method rather than from scratch.

**The nearest published method splits unmetered wind and solar out of substation measurements, but
needs exactly what NGED lacks.** [Teng et al. (2023)](https://doi.org/10.1016/j.rser.2023.113662)
train on 10 Dutch substations that carry complete renewable metering, then predict solar and wind
power separately at substations with none, from the substation's measured total load, weather,
geospatial position, and each site's known renewable capacity, at 15-minute resolution — a
root-mean-square error of 0.07 against 0.70 for a default transfer-learning model, on a
min-max-scaled target. The paper reads 0.07 as 7%, but does not say what the scaling divides by, so
the figure does not transfer to another dataset.

**Inferring the capacity from the net flow alone has been measured, at a smaller scale than
NGED's.** [Gouveia et al. (2026)](https://doi.org/10.1016/j.ijepes.2026.111848) benchmark
data-driven against model-based estimators of the photovoltaic capacity installed behind a
low-voltage substation, working from the net load and irradiance series a network already holds and
from no register at all. Their substations serve 10 to 100 customers, against the thousands behind a
GB primary.

**Two of their results carry over.** The data-driven estimators matched the physical ones when the
data was clean and beat them clearly when it was noisy, which is the condition NGED's telemetry is
in. And models trained on a Belgian dataset, then applied unseen to American and Australian ones
with only approximate irradiance, stayed under 5% mean absolute percentage error once the linear
models were regularised. What Gouveia et al.'s estimators produce is a capacity figure rather than a
forecast, so the half Flexpectation adds is putting that estimate inside a probabilistic multi-day
forecast.

**GB already has an operational forecast of unmetered generation, but only at national scale and
without uncertainty.** The National Energy System Operator (NESO) publishes [embedded wind and solar
forecasts](https://www.neso.energy/data-portal/embedded-wind-and-solar-forecasts) half-hourly to 14
days ahead, the same resolution and horizon Flexpectation delivers. The forecast is a single number
per half-hour, with no uncertainty attached, and it covers GB as one region rather than substation
by substation.

### 8. Disaggregating other distributed energy resources: heat pumps, electric-vehicle chargers, and batteries

#### The challenge

Heat pumps, electric-vehicle chargers, and price-sensitive domestic batteries change the shape of a
substation's load in ways a model trained on history cannot anticipate, because the number of them
behind any given substation is growing quickly. The stretch goal is to disaggregate and forecast
them separately rather than letting them sit inside net demand.

#### What the literature says

**Heat pumps, chargers, and batteries are the largest gap in the review and the largest deliberate
omission from our search.** In the one study we found that measures charger forecast skill against
aggregation, [Ostermann and Haug (2024)](https://doi.org/10.1186/s42162-024-00319-1), only the site
with more than 100 charge points was significantly better than a naive benchmark, though some models
at one much smaller site also beat it. Heat-pump diversity is untested in the cold weather that
matters, and no diversity factor helps for domestic batteries at all.

#### What this means for Flexpectation

**The realistic Flexpectation version 1 position is that heat pumps, chargers, and batteries stay
inside net demand rather than being forecast separately.** The one measurement we found says a
day-ahead charger forecast only clearly beat a naive benchmark above about 100 charge points, and
forecast uncertainty grows with lead time, so at the 14 days NGED needs that threshold should be
expected to be higher rather than the same. The first deliverable on this strand is reading the
electrification literature properly, not a model.

**Detecting heat pumps, chargers, and batteries and forecasting them are separately hard, and not in
the order we expected.** Northern Powergrid's [smart-meter detection
trial](https://smarter.energynetworks.org/projects/npg_nia_-49/), on 1,500 monitored premises, found
that "EV [electric vehicle] identification at premises level was found to be relatively
straightforward", though "a lack of ground truth, such as registered charging points, precluded
formal validation", and that "aggregation does mask some signals, although EV usage is still clearly
identifiable at feeder and substation level". The same trial found that "the detection of ASHP
[air-source heat pumps] is frustrated by the low levels of adoption (<1% of premises) and
differences in operation (low-slow vs high-fast)". So the spiky, synchronised charging that makes
electric vehicles hard to *forecast* is what makes them easy to *detect* in aggregate; heat pumps
are the reverse.

## How we will know whether each of these worked

The eight challenges above need three different kinds of evaluation, and this literature is far
stronger on the first than on the other two. Forecasting has settled practice we can adopt.
Estimating something nobody measures — an effective capacity, an unmetered solar output — has six
possible substitutes for ground truth, of which this literature uses four. Detecting rare events has
good academic practice and, in GB, no precedent that measured anything at all.

**Standard accuracy measures rewarded flat forecasts that would be of little use for either
flexibility procurement or curtailment decisions, so a peak-aware score belongs alongside a proper
score rather than instead of one.** A forecast that predicts the right peak at the wrong time is
penalised twice by mean absolute error — once for the peak it predicted that did not happen, and
once for the peak that did happen and the forecast missed. A flat, featureless forecast avoids both
penalties. Meteorologists named that effect the double penalty decades ago, and their conclusion
transfers: a score that forgives a peak predicted an hour late is generally no longer a *proper
scoring rule* — a score a forecaster cannot improve by publishing anything other than what they
genuinely believe. The same argument runs at the other end of the distribution: the half-hours of
deepest export are the ones curtailment turns on, and a flat forecast hides those too.

**Two teams independently concluded that mean absolute error was the wrong measure for peaks.**
[Pinheiro et al. (2023)](https://doi.org/10.1016/j.apenergy.2022.120493) adopted a peak-aware error
measure for exactly this reason, and [Artificial
Forecasting](https://smarter.energynetworks.org/projects/npg_sif_006-1/) built a metric over the top
10% of demand values and made it the primary measure for comparing their models, reporting it both
against actual demand and normalised to transformer rating.

**A forecast can state its own uncertainty badly without a single accuracy score revealing it.**
[Kaas et al. (2026)](https://arxiv.org/abs/2607.01966) scored models on 200 German low-voltage
feeders with an overload-decision metric evaluated at each model's 95th percentile for consumer
peaks and its 5th for producer peaks. The two models that came first and second on consumer peaks in
the quantile variant of that metric — Chronos-Bolt, a time-series foundation model, and a
weekly-naive baseline — turned out to have 90% ranges containing the true value only 62% and 58% of
the time across the series as a whole, and 43% and 49% of the time at the consumer peaks themselves.
In [Kaas et al. (2026)](https://arxiv.org/abs/2607.01966)'s results, a model that understates its
uncertainty raises fewer false alarms, so it scores well on a threshold-crossing test while being
exactly the model an operator should not trust near a capacity limit.

**Every forecasting paper we read that describes its split keeps most training data out of the
future of its test data, with one exception, and the training window usually grows rather than
slides, and one length rule is worth adopting outright.** [Pinheiro et al.
(2023)](https://doi.org/10.1016/j.apenergy.2022.120493) held out the whole of 2019 and note that
"one year is the minimum acceptable to test a forecasting model whose target value shows annual
seasonality". Substation load shows exactly that seasonality, so any cross-validation fold — one
train-then-test slice of the history — shorter than a year cannot tell us whether a model handles
both ends of the year, and NGED needs both: winter is when NGED buys flexibility, and summer, when
embedded solar output is highest against the lowest demand, is when export constraints bind and
generators are curtailed.

**Not one of the papers we read addresses the leakage a frequently reissued forecast creates, and
Flexpectation is the most exposed design of the lot.** When a forecast covering 14 days is reissued
every 6 hours, every target half-hour is covered by 56 separate forecasts. Count them as
independent, and a significance test will report a confidence the data does not support. Let a
target half-hour fall on both sides of a train-test boundary, and the test set is contaminated
outright. We will report what we did about the leakage rather than leave it implicit, and we treat
the leakage as an open methodological question rather than a solved one.

**There is no ground truth for an effective capacity or an unmetered solar output, and the papers
that estimate them say so.** This literature uses four substitutes for truth, each of which fails
differently, and does not use two others. The four in use are to hold out sites that are metered and
pretend they are not; to inject a change into real data and see whether the method recovers it; to
compare against an independent tool rather than against truth; and to measure whether the estimate
improves the forecast it was built to improve. The two it does not use are to check an estimate
against physics rather than against an answer, and to use a substation where every feeder and every
embedded generator is metered, purely as validation.

**Flexpectation will run the five substitutes that need no new metering, and treat agreement between
the five as the signal, because no one substitute is trustworthy alone.** The five substitutes are
not five attempts at the same measurement. Every number we publish will say which substitute
produced it. The sixth substitute would anchor all the others, and none of the papers above had one:
a fully metered substation is a field deployment rather than an analysis.

**Detection needs different metrics, and the best-worked example in this review chose them
deliberately.** [Bouman et al. (2024)](https://arxiv.org/abs/2405.16164) score precision (the share
of flagged events that are real), recall (the share of real events that get flagged), and an F-score
combining the two, with β set to 1.5 rather than 1, "to give a higher importance to the recall term,
as the potential impact of a false negative is higher than that of a false positive in power grid
expansion planning". That asymmetry holds for Flexpectation too: a missed switching event silently
corrupts the history a model trains on, whereas a false alarm costs an engineer a look.

**The honest headline from the one paper that measured properly is that detecting switching and
metering faults is hard.** [Bouman et al. (2024)](https://arxiv.org/abs/2405.16164) report F-scores
near 0.2 on the shortest events and around 0.5 on the longest, and conclude that performance "is
relatively low across the board, even on the train data. This indicates that the problem is hard to
learn, though it generalizes fairly well". Any target we set for challenges 4 and 6 should start
from those F-scores rather than from an intuition about how obvious a switching event looks on a
chart.

**None of ATLAS, Distribution Network Visibility, or NGED's own Time Series Data Quality — the three
GB metering-fault projects covered under challenge 6 — offers a number to compare against, a gap we
confirmed by checking each one rather than assumed.** Publishing precision and recall against a
stated label set, with the labels released, would therefore be the first time we know of that a GB
network has done so, and it is the cheapest of this review's commitments to keep.

## What published leaderboards did, and what a single team can borrow from them

**What Flexpectation is building is a leaderboard, not a competition, and the distinction changes
which published lessons apply.** Our leaderboards carry our own experiments, one per class of time
series. Solar farms, wind farms, batteries, and the demand at primary substations each get their own
board, and grid and bulk supply points share a board because their measurements are the same kind of
thing. The leaderboards will be public to view and reproducible, but we are not inviting other teams
to submit entries. Anyone who wants to benchmark against us can rerun the setup for themselves. Not
inviting outside entries means the published lessons about attracting entrants, prize pots, and
qualifying rounds do not apply to us, while the lessons about protocol — what makes a comparison
trustworthy — apply with more force, because a competition gets some of its integrity free from
having rivals who would like to catch each other out, and we will not have any.

**Energy forecasting has run competitions on common data for over a decade, and only one of them
forecast at anything like the level NGED acts on.** The last row of the table is what Flexpectation
is building. The two columns that decide whether a precedent exists are the aggregation level and
whether the leaderboard is still open.

| Leaderboard | What entrants forecast | Aggregation level, set against a primary substation | Take-up | Standing or closed |
|---|---|---|---|---|
| Global Energy Forecasting Competitions 2012, 2014, and 2017 ([Hong et al. (2020)](https://doi.org/10.1109/OAJPE.2020.3029979)) | Hierarchical load, price, wind, and solar, with the data published alongside the papers introducing each competition | Varies, up to national | Hundreds of contestants from more than 60 countries | Closed |
| The second track of GEFCom2017 ([Hyndman (2020)](https://doi.org/10.1016/j.ijforecast.2019.03.015)) | Probabilistic load | 183 delivery-point meters of a US utility — the closest of these to a distribution network | 177 entrants across both tracks | Closed |
| BigDEAL Challenge 2022 ([Shukla and Hong (2024)](https://doi.org/10.1049/stg2.12162)) | The timing of peak demand rather than its size; the final match asked for the magnitude, timing, and shape of daily peak load | Three neighbouring local distribution companies — whole utilities, well above a primary substation | 78 teams from 27 countries | Closed |
| HEFTCom ([Browell et al. (2025)](https://doi.org/10.1016/j.ijforecast.2025.10.005)) | The combined day-ahead output of one GB wind-and-solar portfolio | A single 3.6 GW portfolio: one offshore wind farm plus a regional solar aggregate — the data closest to NGED's | Not stated in what we read | Closed; the competition period was 3 months |
| Three competitions NGED funded with Energy Systems Catapult ([McSweeney et al. (2023)](https://doi.org/10.1109/ISGTEUROPE56780.2023.10407541)) | 1-minute peaks inside half-hourly averages; the daily peak a hidden population of electric-vehicle chargers added; missing values. None was a load forecast | NGED's own grid supply point, bulk supply points, and primary-substation feeders | 37 teams, over 2,500 submissions | Closed between December 2021 and April 2022, though the pages and data are still readable on CodaLab |
| Energy-Arena ([Kleinebrahm et al. (2026)](https://arxiv.org/abs/2604.24705)) | The paper describes deterministic day-ahead tasks; the running platform today carries 24 challenges across prices, load, wind, and solar — 8 scored as point forecasts, 8 as quantiles, 8 as ensembles | Not a distribution network | Not stated in what we read | Standing |
| TS-Arena ([Meyer et al. (2026)](https://arxiv.org/abs/2512.20761)) | 186 live energy series | Not a distribution network | 13 foundation models and 3 statistical baselines run by the platform team, plus outside entries | Standing |
| **Flexpectation's leaderboards** | Net demand at substations, and output at metered generators | One board per class of time series | Public to view and reproducible; outside entries not invited | Standing |

**What we found no example of is a standing leaderboard for substation forecasting** — one that
keeps accepting entries after its competition closes. Two of the three competitions NGED funded sat
at exactly the levels NGED forecasts, which is why the gap is scoped to forecasting rather than to
the voltage level. That is the gap Flexpectation's leaderboards fall into, though the search behind
that statement is ours and we would be glad to be pointed at a counter-example.

**The mechanism that makes a leaderboard trustworthy is time, not policing.** The central idea of
[Meyer et al. (2026)](https://arxiv.org/abs/2512.20761)'s TS-Arena is that a forecast is submitted
before the outturn it will be scored against physically exists, which "makes test-set contamination
impossible by design". HEFTCom made the same argument from experience: because the competition ran
on the real, unknown future, "data leakage, accidental or deliberate, was impossible". A half-hourly
forecasting service is unusually well placed here: every day supplies 48 fresh evaluation points
that can never be reused, and the condition that the answer did not exist when the model was frozen
holds automatically.

**The specific way a single team fools itself is not fabrication but running the baseline badly.**
[Kleinebrahm et al. (2026)](https://arxiv.org/abs/2604.24705) put it as a general problem with
published comparisons: competing methods "are not always implemented or optimized with equal care",
so reported differences "may reflect differences in implementation quality rather than inherent
methodological advantages". [Hong et al. (2020)](https://doi.org/10.1109/OAJPE.2020.3029979) put it
more bluntly, that "sometimes the parameters are manipulated, so that the competing models are being
dominated by the proposed ones". So we run every entry through the same evaluation interface, and
run each baseline from its authors' own code at its authors' recommended defaults — the rule
TS-Arena imposes on itself.

**Run two baselines that bracket the answer, not one.** [Doubleday et al.
(2020)](https://doi.org/10.1016/j.solener.2020.05.051) distinguish the two jobs a benchmark does: a
yardstick, which need not be a good forecast, and what they call a point on the yardstick — a target
for a new method to beat, which "should be close to the state of the art". They recommend carrying
both, so that a new method can be positioned between them rather than merely declared better than
something. That is the shape our leaderboards take: persistence (tomorrow resembles a comparable
recent day) and climatology (tomorrow resembles the historical average for the time of year) as the
naive yardstick, and NGED's incumbent method as the point on the yardstick a new model has to reach.

**Our own leaderboard reuses one fold for both model selection and the published result today, and
we would rather say so than discover the problem later.** The fold that Flexpectation currently
reports serves as both the model-selection set and the reported result, so every hyperparameter
choice and feature ablation is adjudicated on the same 12 months the leaderboard publishes. With
hundreds of experiments planned, the winner's reported skill will be optimistically biased. The
structural fix is a final-test window that no model selection is allowed to touch, and it is
scheduled. Until it lands, three things hold: leaderboard numbers are selection metrics rather than
estimates of future skill, differences smaller than fold-level noise should not drive decisions, and
the number of experiments run against a fold is itself a statistic worth publishing beside the
fold's results.

**Rankings travel better than absolute numbers do.** Where a benchmark has enough data behind it,
the ordering of models survives a change of test set even when the accuracy level does not, and that
decides what a leaderboard should report as its headline. [Fildes
(2020)](https://doi.org/10.1016/j.ijforecast.2019.04.012), reviewing the M4 competition, compared
its daily micro series against a real retail forecasting problem and found the same method scoring
1.665% on one and 11.1% on the other. Fildes's conclusion is a direct endorsement of what
Flexpectation is doing: "each organization needs to organize its own forecasting competition for its
own forecasting problems, and should not rely on even large benchmark data sets", with the published
competition useful for narrowing "the pool of methods to be considered" rather than for predicting
your own error. So a leaderboard should lead with ranks and with margins over a stated baseline, and
treat an absolute skill number as valid only on the distribution it was measured on.

**A finite evaluation window can rank the wrong model first, and several months is not obviously
enough.** [Messner et al. (2020)](https://doi.org/10.1002/we.2497)'s conclusion is the sharpest
warning we found about reading a leaderboard: "evaluation results based on a finite data set are
always subject to some degree of uncertainty and the best ranked forecast does not necessarily have
to be the truly best one. Depending on the actual setup, e.g., in a benchmarking exercise to hire a
forecaster, it should be remembered that even periods of several months may still yield uncertainty
in terms of who the best forecaster truly is." HEFTCom's own competition period was 3 months.

**What a leaderboard without entrants cannot do, we should not claim it does.** Two of the strongest
results in the benchmarks above are unavailable to us. The Critical Assessment of Structure
Prediction (CASP) competition's finding that its field plateaued for 14 years ([Kryshtafovych et al.
(2021)](https://doi.org/10.1002/prot.26237)) is a statement about protein structure prediction only
because dozens of groups were trying independently. A plateau on our leaderboard would be ambiguous
between a hard problem and a team that did not think of the right idea. The M competitions'
conclusions about whole classes of method — that complex methods do not typically beat simpler ones,
that combining methods beats the methods combined ([Hyndman
(2020)](https://doi.org/10.1016/j.ijforecast.2019.03.015)) — describe what many independent people
chose to try, and no single team's leaderboard can support that kind of claim. What our leaderboard
can do is narrower and still worth having: show which approaches beat a stated baseline on NGED's
own data, under one protocol, with the forecasts, the metric definitions, and the code published so
that anyone can check the arithmetic or rerun the comparison themselves.

## Three published results that point against this project's plan

Three results in this literature point against Flexpectation's plan, and we intend to test all three
rather than avoid them.

### Finer-grained weather data has not always paid

[Browell and Fasiolo (2021)](https://arxiv.org/abs/2103.10335) added spatial statistics derived from
gridded numerical weather prediction to their model of 14 grid supply point groups in GB. Those
spatial statistics helped significantly in 2 of the 14 regions, hurt significantly in 3, and made no
measurable difference in the remaining 9. Weather itself was worth a great deal to them — adding
wind and irradiance cut their pinball loss, the equivalent score for a single quantile, by 40%
overall, and by 60% in North Scotland against 10% in London — so the question is not whether weather
matters but whether *finer* weather does. Artificial Forecasting obtained postcode-level weather
forecasts for two wind-connected primary substations after their wind-connected models had performed
poorly, and reported that the postcode-level forecasts "did not notably improve model performance",
naming better weather data as a next step.

### Weather has bought less than expected at low voltage in the past

[Haben et al. (2019)](https://doi.org/10.1016/j.ijforecast.2018.10.007) tested 100 real low-voltage
feeders with both forecast and observed temperature, and found that temperature had no effect on
forecast accuracy, or a negative one. [Haben et al.
(2019)](https://doi.org/10.1016/j.ijforecast.2018.10.007) used data collected in 2014 and 2015. We
expect how much weather matters at a substation to be changing quickly, because embedded solar
generation and heat pumps are what make a substation weather-dependent, and there are far more of
both on the network now than there were then. That expectation is a prediction, though, not a
measurement — and the Scottish primary-substation sensitivities of [Fox et al.
(2018)](https://doi.org/10.34890/134), measured on the 10 years of weather and network data before
its publication and described in the full review, say weather was already moving primary substation
demand well before the mid-2010s.

### A model trained on none of NGED's data may match a model trained on all of it

[Kaas et al. (2026)](https://arxiv.org/abs/2607.01966) tested Chronos-2, a general-purpose
time-series model that had never seen their data, against models trained on the first 160 of their
200 German low-voltage feeders and scored, like Chronos-2, on all 200 feeders. Chronos-2 beat every
purpose-trained competitor on mean absolute error, 3.8 kW against 4.2 kW. Their purpose-trained
models were not heavily engineered, and challenge 1 above found only a modest return to model
sophistication. But a model given a network's whole history, beaten by a model that saw none of that
history, is still important information about the value of any programme of heavy engineering.

## What GB networks have already built

**Nine network projects have already built something close to a piece of Flexpectation, and one of
them is further ahead than Flexpectation is.** The last row of the table is Flexpectation itself, so
the comparison is direct. Where a project's published material does not answer a column, the cell
says so rather than being left blank.

| Project | What it forecasts | Scale | Horizon | Uncertainty published |
|---|---|---|---|---|
| [Artificial Forecasting](https://smarter.energynetworks.org/projects/npg_sif_006-1/) (Northern Powergrid) | Demand and customer export at primary substations; active power at secondary | 551 primary substations with export data, 171 modelled; 729 secondary substations | Day-ahead to 11 days at primary; week- to month-ahead at secondary | Half-hourly, with 5th-to-95th-percentile bands |
| [SSEN TRANSITION](https://ssen-innovation.co.uk/transition/) | Net load, split into demand and generation, then recombined | 13 primary substations, their bulk supply points, and their 33 kV and 11 kV feeders | 30 minutes to 10 days | A 40-member ICON-EU ensemble to 4 days, one deterministic forecast after that |
| [NGED's EFFS](https://smarter.energynetworks.org/projects/wpden03/) | Grid supply points, bulk supply points, primary substation transformers, and generation sites | Network-wide | 1 hour to 6 months | None |
| [UK Power Networks' Power Flow to Solar Capacity](https://smarter.energynetworks.org/projects/nia_ukpn0104/) | The capacity of unmetered solar behind each primary substation, then that solar's generation | Not stated in what we read | Not stated in what we read | Not stated in what we read |
| [SSEN FastTrack](https://smarter.energynetworks.org/projects/10166254/) | How the connections queue, around 180 GW, will load the network | Primary substations up to the grid supply point | A planning horizon rather than an operational one | A probability that a queued connection becomes real load |
| [SP Energy Networks' Predict4Resilience](https://smarter.energynetworks.org/projects/10061710/) | Network faults, not load | Per district | Up to 7 days | A probability distribution driven by a weather ensemble |
| [Fox et al. (2018)](https://doi.org/10.34890/134) (SP Energy Networks) | The effect of weather on past peak demand, not a forward forecast | 13 primary substations in the proof of concept, almost 400 in production | Backwards over 10 years | None |
| [OpenSTEF](https://lfenergy.org/projects/openstef/) (Alliander, the Netherlands) | Net load, with a splitter into solar, wind, and residual parts | Thousands of grid connection points | To 48 hours | Yes; the framework is built for probabilistic forecasting |
| [Cordier et al. (2024)](https://doi.org/10.1049/icp.2024.2058) (Enedis, France) | Consumption and generation at the substation since 2015; the finer-grid method the paper describes covers consumption only | All 2,300 high-voltage-to-medium-voltage substations, extending to 3,678 of the more than 5,000 transformers inside them, and towards 750,000 medium-to-low-voltage substations | Not stated in the paper; the forecasts run at 10- or 30-minute resolution | None stated in the paper |
| **Flexpectation** | Net demand, with unmetered generation inferred | 32 series in the trial area; 52 grid supply points, 271 bulk supply points, and 1,161 primary substations at network scale | 14 days, updated every 6 hours | A 51-member ECMWF ensemble across the whole horizon |

**SSEN's TRANSITION is the closest precedent for Flexpectation's method.** TRANSITION split each
substation's net load — demand minus whatever generation behind that substation happened to produce
— into demand and generation, forecast the two separately, then recombined them. Two things
TRANSITION did not set out to do are what Flexpectation adds. TRANSITION's ensemble covered only the
first 4 days, so from day 4 to day 10 a single deterministic forecast was all it had, whereas NGED
acts out to 14. And TRANSITION was a 13-substation trial rather than a network-wide deployment.
Everything else about TRANSITION's design matches what Flexpectation is building.

**NGED's own Electricity Flexibility and Forecasting System independently selected XGBoost, which
won on accuracy and was also the easiest to automate.** The project compared XGBoost against a long
short-term memory (LSTM) neural network and against ARIMA, and its evaluation report says XGBoost
"provided the best results of the three methods tested, closely followed by LSTM", recommending
XGBoost because it also allows simplified testing of features and can be easily automated. The
report caveats that the LSTM could not be fully explored for want of graphics processing units, and
expects that more testing would have brought the LSTM level with XGBoost rather than past it. That
choice is the same starting point Flexpectation uses.
[EFFS](https://smarter.energynetworks.org/projects/wpden03/) ran from 2018 to 2021 as a Network
Innovation Competition project costing £3.3 million, and its forecasts carried no uncertainty at
all. Adding that uncertainty is the step this project adds. [UK Power Networks' Power Flow to Solar
Capacity](https://smarter.energynetworks.org/projects/nia_ukpn0104/) is the direct predecessor of
Flexpectation's unmetered-solar work, as challenge 7 above sets out.

**Two of the nine projects in the table are outside GB: OpenSTEF in the Netherlands and Enedis in
France.** [OpenSTEF](https://lfenergy.org/projects/openstef/) is also the only operational network
forecasting system in this review whose code can be read rather than inferred from a deliverable.

**Enedis has forecast every one of its high-voltage-to-medium-voltage substations since 2015, and is
now extending the forecast below the substation.** The French distribution network operator covers
all 2,300 of those substations, and is extending the forecast to 3,678 of the more than 5,000
transformers inside those substations, and towards the 750,000 medium-to-low-voltage substations
beyond them ([Cordier et al. (2024)](https://doi.org/10.1049/icp.2024.2058)). Forecasting
operationally at the scale Flexpectation reaches in 2027 has therefore already existed elsewhere for
a decade.

**Fitting a model to each transformer beat the method Enedis runs in production, which shares one
substation forecast out across its transformers by fixed coefficients.** The per-transformer models
scored 6.0% mean absolute percentage error against 9.3% on the day those coefficients were
refreshed, and 8.1% against 13.0% across the whole test period. That second comparison counts only
the transformers whose coefficient then moved by less than 2.5%, and on that comparison 84% of
transformers were more accurate under their own model. Cordier et al. chose both comparisons
deliberately, as the cases where the fixed-coefficient method is "the most relevant and the most
difficult to outperform". Cordier et al. do not say what their percentage error is normalised by,
and report that the complete pipeline has not yet been evaluated end to end.

### Northern Powergrid's Artificial Forecasting is further ahead than Flexpectation

**One concurrent project matters more than any paper here.** [Artificial
Forecasting](https://smarter.energynetworks.org/projects/npg_sif_006-1/) is an Ofgem Strategic
Innovation Fund programme, with about £3.9 million of grant across its three phases, run by Northern
Powergrid with Faculty, EV.energy, and Oaktree Power, the final Beta phase running to February 2027.
The Beta deliverables that the rest of this section draws on sit under a [separate project
registration](https://smarter.energynetworks.org/projects/10145998/) from the Alpha ones. Artificial
Forecasting does much of what Flexpectation does at primary substations, and also covers secondary
substations, which Flexpectation does not. At the time of writing, Artificial Forecasting is further
ahead than Flexpectation.

**Artificial Forecasting has run operationally through a full winter flexibility procurement
cycle.** A forecasting service for primary substations is deployed and has passed the network's
architecture review board, data governance, and information security checks for its current
deployment. It was used operationally by Northern Powergrid's System Forecasting team through a full
winter flexibility procurement cycle to support week-ahead dispatch decisions. It produces
half-hourly probabilistic forecasts with 5th-to-95th-percentile bands, flags forecast exceedances of
firm capacity, and is benchmarked against the network's existing growth-based and persistence
methods and a rolling 4-week baseline. The deliverable states that performance did not materially
degrade on average across the 11-day horizon, without publishing the figures behind that claim.

**Artificial Forecasting's value case puts whole-life net present value at around £60 million for
one network, or £250 million if three further networks adopt Artificial Forecasting.** That value
comes from a 3% reduction in spending on reinforcement — building bigger transformers and cables —
in the current price-control period, rising to 6% in the next, and from a 25% improvement in the
cost-effectiveness of contracted flexibility. None of the three benefit categories in Artificial
Forecasting's benefits assessment is curtailment, even though the forecast covers customer export at
primary substations, so the one published value case in this review puts no money on the export end
that NGED now rate alongside flexibility spend. The project pairs those figures with a direct
caveat: it reports early Beta evidence, from one winter procurement cycle, supporting the
performance assumptions behind the value case, which "remains appropriate, subject to further
validation".

**Artificial Forecasting is independent evidence that short-term substation forecasting is
operationally useful**, that networks will change their procurement process around such a forecast,
and that a benefits case has been made and accepted. Because Artificial Forecasting is public,
operational, and benchmarked against a real incumbent method, Artificial Forecasting is also the
clearest available example of what "working" looks like. Artificial Forecasting's core intellectual
property is to be made available royalty-free to other GB networks, and we would rather build on
that intellectual property than rebuild it — a shared evaluation protocol between two GB networks
would be worth more to both than two separate protocols.

**Flexpectation is nonetheless attempting more than Artificial Forecasting's published material
describes, which is the case for running both.** The two projects overlap on forecasting net demand
at primary substations and on forecasting metered generation. Artificial Forecasting's Beta
registration also claims load disaggregation as an innovation — "a novel approach to forecasting HV
load, separately modelling gross demand and distributed generation" — but the deliverables describe
forecasting two series that are each already measured. The Beta annual progress report produces net
demand "by independently modelling customer export data", the Alpha technical report covers "all 160
substations where both gross demand and customer export data were available", and the Embedded
Capacity Register enters the model as an input feature, listing what is registered rather than
estimating what is not. Flexpectation's challenges 7 and 8 are the different problem of inferring an
unmetered generator's half-hourly output from a substation's net flow, which is blind source
separation. Two more of Flexpectation's challenges do have a counterpart there. The Beta annual
progress report describes automated health checks and dashboards that "highlight substations where
input data is degraded (e.g. faulty sensors, frozen or anomalous values)" and an
extract-transform-load pipeline that "flags frozen/spiky SCADA data before modelling", which is
challenge 6; and the Alpha user research treats planned and unplanned outages as data worth bringing
in and as a reason to widen the error margin, which is a different response to challenge 4's problem
rather than no response.

**Four of Flexpectation's eight challenges have no counterpart we could find in that material:**
tracking the effective capacity of metered generators; forecasting a substation as if it were always
in its normal running arrangement, rather than dropping the periods when it was not; and inferring
unmetered generation from a substation's net flow, solar and wind first and then heat pumps,
chargers, and batteries. Across every Artificial Forecasting deliverable published on the Smarter
Networks Portal — Discovery, Alpha, and Beta, save one file that holds a single blank page —
"abnormal", "unmetered", "blind source" and "source separation" return nothing at all; "capacity"
appears 180 times but never as an effective or derated capacity; and the seven occurrences of a
"switch" stem are generators switching off, switchgear asset types, and switching over a data feed.
Heat pumps and electric vehicles do appear, as drivers of demand growth and as model features rather
than as quantities separated out of a net flow. Flexpectation also delivers 1st and 99th percentiles
where Artificial Forecasting's published bands run from the 5th to the 95th, and the curtailment
decisions NGED describes turn on those outer levels.

## Why we think this ambitious plan can be done

**Measured against the studies we found, the plan sits outside the published literature in five ways
at once.** That gap says more about where our search fell short than about the quality of the work
that fills the rest of the field. No study in this review drives a substation forecast from a
weather ensemble across a 14-day horizon. None models the tails explicitly at substation level; the
one study that models them explicitly at all works on regions far larger than a substation. None
puts unmetered generation inside a probabilistic forecast at substation level over a multi-day
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
level, with one family of model; the few that touch two challenges solve them as a pipeline rather
than together. Pre-training weather and time encoders and then reading a substation's probabilistic
forecast off them would be a full study by that standard, and so would each of the other three
strands. Sizing the four strands as separate studies scopes the work rather than promising an output
— how many of the strands survive contact with the data is exactly what the project has to find out.

**Only the first of those four strands — the heavily-tuned gradient-boosting model — is in scope for
Flexpectation version 1.** The other three strands belong to the network-wide scale-up from 2027, as
does the disaggregation of unmetered generation.

**The main reason for attempting all eight at once is that they may be one challenge rather than
eight.** A switching event, a turbine out for repair, and a stuck meter all surface in the same
place: as a discrepancy between what a substation metered and what the weather and the calendar say
it should have metered. Every study reviewed above that touches more than one of the eight solves
them as a pipeline. In every case one stage's output is frozen before the next stage sees it, so an
error made early cannot be corrected later and the forecast error never gets to tell the capacity
estimator it was wrong.

**So the question we want to answer is whether one model that estimates capacity, switching state,
and demand together beats that pipeline.** NGED's specification leaves room for that combined
approach, asking that these phenomena be handled rather than that they be handled explicitly.

**The first reason for confidence is that experiments are nearly free.** The core forecast already
exists and runs today, on an experiment framework that makes one more experiment cost compute time
rather than staff time. That near-zero cost is what makes it realistic to run on the order of
hundreds of machine-learning experiments a month, and it is the same argument the introduction to
this review makes.

**Several of the four model families will not work.** Expecting that failure is what makes the four
families research directions rather than engineering tasks. The honest expectation is that some
deliver clearly, some produce a negative result worth publishing, and some are abandoned. Both NGED
and this project count a negative result as an outcome: evidence that switching cannot be recovered
from power data alone, for instance, would be worth having, because it would justify extracting
switching labels from operational systems instead of continuing to look.

## References

Every source cited above, in alphabetical order by first author. The full review cites 37 further
sources that this summary does not.

- Bian, Y., Zheng, N., Zheng, Y., Xu, B. and Shi, Y. (2024). [Predicting Strategic Energy Storage
Behaviors](https://doi.org/10.1109/TSG.2023.3303469). *IEEE Transactions on Smart Grid*.
- Bouman, R., Schmeitz, L., Buise, L., Heres, J., Shapovalova, Y. and Heskes, T. (2024). [Acquiring
Better Load Estimates by Combining Anomaly and Change Point Detection in Power Grid Time-series
Measurements](https://arxiv.org/abs/2405.16164). *Sustainable Energy, Grids and Networks*.
- Browell, J. and Fasiolo, M. (2021). [Probabilistic Forecasting of Regional Net-load with
Conditional Extremes and Gridded NWP](https://arxiv.org/abs/2103.10335). *IEEE Transactions on
Smart Grid*.
- Browell, J., van der Meer, D., Kälvegren, H., Haglund, S., Simioni, E., Bessa, R. J. and Wang, Y.
(2025). [The hybrid renewable energy forecasting and trading competition
2024](https://doi.org/10.1016/j.ijforecast.2025.10.005). *International Journal of Forecasting*.
- Brown, C. F. et al. (2025). [AlphaEarth Foundations: An embedding field model for accurate and
efficient global mapping from sparse label data](https://arxiv.org/abs/2507.22291).
- Buizza, R. and Leutbecher, M. (2015). [The forecast skill
horizon](https://doi.org/10.1002/qj.2619). *Quarterly Journal of the Royal Meteorological
Society*.
- Campagne, E., Amara-Ouali, Y., Goude, Y., Zehavi, I. and Kalogeratos, A. (2025). [Graph Neural
Networks for Electricity Load Forecasting](https://arxiv.org/abs/2507.03690).
- Cordier, G. et al. (2024). [Methods and techniques used to produce electricity forecasts on
Enedis’ distribution network at a finer grid than the HV/MV
substation](https://doi.org/10.1049/icp.2024.2058). *CIRED 2024 Vienna Workshop*, in *IET
Conference Proceedings*.
- Dantas, G. and Browell, J. (2026). [Seamless Short‐ to Mid‐Term Probabilistic Wind Power
Forecasting](https://doi.org/10.1002/we.70079). *Wind Energy*.
- de Vilmarest, J., Browell, J., Fasiolo, M., Goude, Y. and Wintenberger, O. (2024). [Adaptive
Probabilistic Forecasting of Electricity (Net-)Load](https://doi.org/10.1109/TPWRS.2023.3310280).
*IEEE Transactions on Power Systems*.
- Doubleday, K., Van Scyoc Hernandez, V. and Hodge, B. M. (2020). [Benchmark probabilistic solar
forecasts: Characteristics and recommendations](https://doi.org/10.1016/j.solener.2020.05.051).
*Solar Energy*.
- Electricity North West (2018). [ATLAS](https://smarter.energynetworks.org/projects/nia_enwl008/).
- Fildes, R. (2020). [Learning from forecasting
competitions](https://doi.org/10.1016/j.ijforecast.2019.04.012). *International Journal of
Forecasting*.
- Fox, J., Plecas, M., Neilson, D., Cannon, D. and Parr, J. (2018). [Analysis of local demand trends
and forecasting through weather correction and benefit to DSO transistion and
microgrids](https://doi.org/10.34890/134). *CIRED Workshop, Ljubljana*.
- Gijón, A., Eiraudo, S., Manjavacas, A., Schiera, D. S., Molina-Solana, M. and Gómez-Romero, J.
(2025). [Integrating Physics and Data-Driven Approaches: An Explainable and Uncertainty-Aware
Hybrid Model for Wind Turbine Power Prediction](https://arxiv.org/abs/2502.07344). *Computer
Physics Communications*.
- Gilbert, C., Browell, J. and Stephen, B. (2023). [Probabilistic load forecasting for the low
voltage network: forecast fusion and daily peaks](https://arxiv.org/abs/2206.11745). *Sustainable
Energy, Grids and Networks*.
- Gouveia, A. M. V., Hashmi, M. U., D’hulst, R. and Van Hertem, D. (2026). [Installed PV capacity
detection on LV substations: Comparison of Data-Driven and Model-Based
methods](https://doi.org/10.1016/j.ijepes.2026.111848). *International Journal of Electrical Power
and Energy Systems*.
- Haben, S., Giasemidis, G., Ziel, F. and Arora, S. (2019). [Short term load forecasting and the
effect of temperature at the low voltage level](https://doi.org/10.1016/j.ijforecast.2018.10.007).
*International Journal of Forecasting*.
- Haben, S., Arora, S., Giasemidis, G., Voss, M. and Greetham, D. V. (2021). [Review of Low Voltage
Load Forecasting: Methods, Applications, and Recommendations](https://arxiv.org/abs/2106.00006).
*Applied Energy*.
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
- Kaas, B., Treutlein, M., Gerber, H. B., Neumann, O., Phatthanakhuha, C., Resch, O., Mikut, R. and
Hagenmeyer, V. (2026). [Probabilistic Low-Voltage Peak Load Forecasting with Time Series
Foundation Models Evaluated on Application-Oriented Metrics](https://arxiv.org/abs/2607.01966).
- Kim, J.-H., Lee, B.-S. and Kim, C.-H. (2020). [A Study on the Development of Machine-Learning
Based Load Transfer Detection Algorithm for Distribution
Planning](https://doi.org/10.3390/en13174358).
*Energies*.
- Kim, J.-H., Joung, J.-M. and Lee, B.-S. (2022). [A Study on the Preprocessing Method for Power
System Applications Based on Polynomial and Standard Patterns](https://doi.org/10.3390/en15041441).
*Energies*.
- Kim, J.-H. (2024). [A Study on the Detection Method of Load Transfer in Distribution System Using
Time Series Decomposition](https://doi.org/10.5370/KIEE.2024.73.11.1873). *The Transactions of The
Korean Institute of Electrical Engineers*.
- Kim, J.-H. (2025). [Unsupervised Load Transfer Detection Based on Wavelet Change Point
Analysis and Isolation Forest](https://doi.org/10.5370/KIEE.2025.74.11.1757). *The
Transactions of The Korean Institute of Electrical Engineers*.
- Kleinebrahm, M. et al. (2026). [Energy-Arena: A Dynamic Benchmark for Operational Energy
Forecasting](https://arxiv.org/abs/2604.24705). *2026 International Conference on the European
Energy Market*.
- Kryshtafovych, A., Schwede, T., Topf, M., Fidelis, K. and Moult, J. (2021). [Critical assessment
of methods of protein structure prediction (CASP) — Round
XIV](https://doi.org/10.1002/prot.26237). *Proteins: Structure, Function, and Bioinformatics*.
- LF Energy. [OpenSTEF](https://lfenergy.org/projects/openstef/).
- Liu, H., Wang, Y., Wei, C., Li, J. and Lin, Y. (2019). [Two-Stage Short-Term Load Forecasting for
Power Transformers Under Different Substation Operating
Conditions](https://doi.org/10.1109/ACCESS.2019.2951422). *IEEE Access*.
- Ludwig, N., Arora, S. and Taylor, J. W. (2023). [Probabilistic load forecasting using
post-processed weather ensemble predictions](https://doi.org/10.1080/01605682.2022.2115411).
*Journal of the Operational Research Society*.
- McSweeney, L., Haben, S. and Young, S. (2023). [Data Science Challenges; A Whole Systems Lens for
Energy Network Solutions](https://doi.org/10.1109/ISGTEUROPE56780.2023.10407541). *2023 IEEE PES
Innovative Smart Grid Technologies Europe*.
- Meng, B., Loonen, R. and Hensen, J. L. M. (2020). [Data-driven inference of unknown tilt and
azimuth of distributed PV systems](https://doi.org/10.1016/j.solener.2020.09.077). *Solar Energy*.
- Mesarcik, M., Loke, J., Wildeboer, J. and Lucassen, B. (2025). [Probabilistic day-ahead power
forecasting in the medium-voltage grid using state space
models](https://doi.org/10.1049/icp.2025.1968). *CIRED 2025*, in *IET Conference Proceedings*. The
version of record is paywalled; we read the authors' own copy, which is titled "…Using Structured
State Space Models".
- Messner, J. W., Pinson, P., Browell, J., Bjerregård, M. B. and Schicker, I. (2020). [Evaluation of
wind power forecasts — An up-to-date view](https://doi.org/10.1002/we.2497). *Wind Energy*.
- Meyer, M., Kaltenpoth, S., Albers, H., Zalipski, K. and Müller, O. (2026). [TS-Arena: A Live
Forecast Pre-Registration Platform](https://arxiv.org/abs/2512.20761). *Proceedings of the 32nd
ACM SIGKDD Conference on Knowledge Discovery and Data Mining*.
- Mitra, P. and Ramavajjala, V. (2023). [Learning to forecast diagnostic parameters using
pre-trained weather embedding](https://arxiv.org/abs/2312.00290).
- National Energy System Operator. [Embedded wind and solar
forecasts](https://www.neso.energy/data-portal/embedded-wind-and-solar-forecasts).
- Nguyen, T. N. and Müsgens, F. (2026). [A meta-analysis of solar forecasting based on skill
score](https://doi.org/10.1063/5.0300682). *Journal of Renewable and Sustainable Energy*.
- Northern Powergrid (2024). [Artificial Forecasting, Alpha
phase](https://smarter.energynetworks.org/projects/npg_sif_006-1/).
- Northern Powergrid (2024). [Detecting LCTs from Smart Meter Consumption
Data](https://smarter.energynetworks.org/projects/npg_nia_-49/).
- Northern Powergrid (2025). [Artificial Forecasting, Beta
phase](https://smarter.energynetworks.org/projects/10145998/).
- Ostermann, A. and Haug, T. (2024). [Probabilistic forecast of electric vehicle charging demand:
analysis of different aggregation levels and energy
procurement](https://doi.org/10.1186/s42162-024-00319-1). *Energy Informatics*.
- Paredes, G. and Vargas, L. (2017). [Adjustment of discrete load changes in feeder databases for
improving medium‐term demand forecasting](https://doi.org/10.1049/iet-gtd.2017.0129). *IET
Generation, Transmission & Distribution*.
- Pierrot, A. and Pinson, P. (2024). [On Tracking Varying Bounds When Forecasting Bounded Time
Series](https://doi.org/10.1080/00401706.2024.2350421). *Technometrics*.
- Pinheiro, M. G., Madeira, S. C. and Francisco, A. P. (2023). [Short-term electricity load
forecasting—A systematic approach from system level to secondary
substations](https://doi.org/10.1016/j.apenergy.2022.120493). *Applied Energy*.
- Rasp, S. and Lerch, S. (2018). [Neural networks for post-processing ensemble weather
forecasts](https://arxiv.org/abs/1805.09091). *Monthly Weather Review*.
- Ruhhütl, M., Schmaranz, R. and Dietrichsteiner, T. (2023). [Load and generation forecast on
substation level](https://doi.org/10.1049/icp.2023.0476). *CIRED 2023, Rome*, in *IET Conference
Proceedings*.
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
- Siméoni, O. et al. (2025). [DINOv3](https://arxiv.org/abs/2508.10104).
- SP Energy Networks (2023).
[Predict4Resilience](https://smarter.energynetworks.org/projects/10061710/).
- Teng, S., Cambier van Nooten, C., van Doorn, J., Ottenbros, A., Huijbregts, M. and Jansen, J.
(2023). [Near real-time predictions of renewable electricity production at substation level via
domain adaptation zero-shot learning in sequence](https://doi.org/10.1016/j.rser.2023.113662).
*Renewable and Sustainable Energy Reviews*.
- UK Power Networks. [Power Flow to Solar Capacity
(NIA_UKPN0104)](https://smarter.energynetworks.org/projects/nia_ukpn0104/).
- Viotti, O., Arnqvist, J. and Olauson, J. (2026). [Estimating Wind‐Power Capacity Time Series From
Production Data Using a Power Curve Model and Quadratic
Optimization](https://doi.org/10.1002/we.70136). *Wind Energy*.
- Western Power Distribution (2021). [Electricity Flexibility and Forecasting System
(EFFS)](https://smarter.energynetworks.org/projects/wpden03/).
- Willis, H. L., Powell, R. D. and Wall, D. L. (1984). [Load Transfer Coupling Regression
Curve Fitting for Distribution Load Forecasting](https://doi.org/10.1109/TPAS.1984.318713).
*IEEE Transactions on Power Apparatus and Systems*.
