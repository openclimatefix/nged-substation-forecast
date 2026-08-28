# The current state of the art in energy forecasting: a summary

This is a short version of a literature review Open Climate Fix carried out for National Grid
Electricity Distribution (NGED), as part of the Flexpectation project, funded through the Network
Innovation Allowance. This summary is readable on its own. The full literature review, which cites
over 100 sources, is
[published online](https://openclimatefix.github.io/nged-substation-forecast/background/energy-forecasting-review/),
and is referred to below as "the full review". This summary is not a strict subset of the full
review: it covers one further challenge, recovering signed power from apparent-power meters, and
cites 24 sources the full review does not.

## Summary

**No honest review of the energy forecasting literature can name a canonical state of the art.**
Energy forecasting papers measure performance in different ways against different datasets, so the
literature cannot rank the approaches it contains — like an international football tournament where
every team plays by different rules, with different size goals. Energy forecasting researchers have
done great work over the years, and the lack of comparability is nobody's fault: it's a systemic
failure, the industry is already aware of it, and people are trying to fix it. We review several
substantial efforts to compare forecasting approaches fairly.

**What the literature *does* show is that the machine learning approach Flexpectation version 1 uses
— a gradient-boosted tree — is a sensible place to start.** The literature provides no conclusive
evidence that anything more sophisticated buys a large, dependable improvement over a
gradient-boosted tree at substation level. NGED's own Electricity Flexibility and Forecasting System
reached the same choice independently in 2019.

**In terms of machine learning research, Flexpectation is ambitious: several of our research ideas
have no precedent in the literature we reviewed.** We found no published model that recovers a
latent normal-running-arrangement demand for a distribution substation; no method that detects a
switching event by checking that the power leaving one substation arrives at its neighbours; no
capacity estimator run across a mixed fleet of individually metered generators at one distribution
network; no paper driving a probabilistic substation forecast from a weather ensemble across a
14-day horizon; none modelling the tails of the distribution explicitly at substation level; none
aggregating building thermal physics up to a substation and putting that physics inside a
probabilistic forecast; and none reading a substation forecast off a pre-trained weather encoder.
Most striking of all, every study we reviewed that touches more than one of the nine challenges
solves them as a pipeline, freezing each stage's output before the next stage sees it — so a mistake
made early can never be put right by what a later stage learns.

**Every one of our research ideas is planned research rather than a result, and research fails.**
Each absence above says that we did not find prior work, not that the approach will succeed, and
some of these ideas will turn out to be worse than the gradient-boosted tree Flexpectation version 1
starts from. A negative result, published clearly, is a real outcome of the project rather than a
failure of it. What makes the ambition worth attempting is that the nine challenges surface in the
same place — as a discrepancy between what a substation metered and what the weather and the
calendar say it should have metered — so one model reasoning about several at once has information
that a serial pipeline throws away. None of that risk falls on the forecast NGED receives: version
1's gradient-boosted tree is the deliverable, and every idea above has to beat it on held-out data
before it goes anywhere near an operational forecast.

**The platform for running those experiments is already built, and built for speed.**
Flexpectation has hundreds of ML ideas to test, so the software platform is designed to
make each experiment quick to run, quick to validate, and directly comparable with every experiment
before it. This makes failure cheap, which is what makes trying hundreds of them realistic.

**Northern Powergrid's Artificial Forecasting project is further ahead than Flexpectation.**
Artificial Forecasting has run operationally through a full winter flexibility procurement cycle,
which is the clearest available evidence that a forecast of this kind changes what a network
operator does.

**The value NGED gets from the forecast sits in both tails of the distribution**: the upper tail,
where flexibility procurement holds demand under a limit, and the lower tail, where curtailment
holds export under that same limit. Yet, most energy forecasting research is focused on the *middle*
of the distribution.

**The literature holds a wealth of knowledge on measuring the performance of power forecasts, and
several traps to avoid.** Mean absolute error rewards flat forecasts that are of little use for
either flexibility or curtailment decisions: a peak predicted an hour late is penalised twice, once
for the peak that did not happen and once for the peak that was missed, and an overly smooth
forecast avoids both penalties. Ranking well on one measure also says little about other measures:
across 200 German low-voltage feeders, the two models that came first and second on consumer peaks
in the quantile version of an overload-decision metric stated their own uncertainty badly, their 90%
ranges containing the true value less than half the time at those consumer peaks.

**Three published results point against parts of Flexpectation's plan, and we intend to test all
three rather than avoid them.** Finer-grained weather has not always improved performance; weather
data has improved performance less than expected at low voltage in the past; and a pre-trained ML
model trained on none of NGED's data may match models trained on all of it.

**Whilst the literature we found does not tell us exactly which algorithms provide the best
forecasting performance, the literature *is* clear on how to *research and develop* a state of the
art forecast.** There's no magic. Machine learning is an empirical science, and most research ideas
fail. John Jumper, who shared the 2024 Nobel Prize in Chemistry for his work on AlphaFold, puts the
share of research ideas that fail at around 90%, and treats that rate as an ordinary and necessary
feature of doing research rather than as evidence of doing it badly ([Nobel Week
interview](https://youtu.be/nNM1QdmFwIs?t=852), 6 December 2024, from 14:12). So progress comes
largely from being able to quickly test many ideas under identical conditions and carefully measure
performance. We have built an MLOps framework that should allow us to test research ideas as
efficiently as possible.

**The fact that the industry doesn't yet know the state of the art is a huge opportunity for the
Flexpectation project.** We are in a very privileged position where we can try hundreds of ideas,
and test the best ideas in the real world. We have an opportunity to make a significant contribution
to the energy forecasting community by publishing leaderboards of ML experiments, and hence help the
industry as a whole to better understand how multiple approaches perform.

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

But — to our tastes — Claude struggles to write readable prose. So the text has been heavily
re-written (and cut down) by hand.

## What the literature says about the nine challenges Flexpectation aims to solve

Flexpectation's specification breaks into nine challenges. This section takes each in turn: what the
challenge is, what the literature says, and what that means for Flexpectation. The coverage is
uneven. The first challenge (probabilistic forecasts of net demand at substations) has a large body
of literature, the second challenge (forecasting metered generators) is the most mature field on the
list, and the eighth challenge (disaggregating unmetered solar and wind) needs the longest
treatment, because the published work sits either side of the aggregation level NGED meters at and
the review borrows from three fields outside energy forecasting.

**The table below is a summary of this entire section: the table describes, for each challenge, the
most relevant papers we found, and the implications for Flexpectation.** The sections that follow
give the evidence behind each row.

| Challenge | Closest published precedent | What this means for Flexpectation |
|---|---|---|
| 1. Probabilistic net-demand forecasts at substations | [Artificial Forecasting](https://smarter.energynetworks.org/projects/npg_sif_006-1/) at 551 primary substations, [Pinheiro et al. (2023)](https://doi.org/10.1016/j.apenergy.2022.120493) at 96,989 Portuguese secondary substations, [Scottish and Southern Electricity Networks' (SSEN) TRANSITION](https://ssen-innovation.co.uk/transition/) at 13 | A gradient-boosted tree (GBT) is a defensible default for Flexpectation version 1, but the literature paints GBTs as a sensible starting point rather than a proven winner |
| 2. Forecasting metered generators | [Dantas and Browell (2026)](https://doi.org/10.1002/we.70079) on 73 wind farms in Great Britain (GB) from the European Centre for Medium-Range Weather Forecasts (ECMWF) ensemble, [HEFTCom](https://doi.org/10.1016/j.ijforecast.2025.10.005)'s day-ahead portfolio forecast, and [Nguyen and Müsgens (2026)](https://doi.org/10.1063/5.0300682)'s meta-analysis of 4,687 skill scores from 188 solar forecasting papers | Gradient-boosted trees fitted separately for each kind of generator are the standard approach, and what won when teams were scored against each other on the same data. A higher-resolution deterministic forecast beat the ensemble at short lead times |
| 3. Estimating the effective capacity of metered generators | [Viotti et al. (2026)](https://doi.org/10.1002/we.70136), fitting a wind farm's capacity against a capacity factor simulated from reanalysis weather, and [Dantas and Browell (2026)](https://doi.org/10.1002/we.70079), ratcheting a running maximum of the farm's own metered production. Every method we found covers one generation technology, and most work from a revenue meter alone | Flexpectation version 1 needs an estimator that can track effective capacity downwards, which is exactly where the two published wind methods differ |
| 4. Detecting switching events | [Bouman et al. (2024)](https://arxiv.org/abs/2405.16164) at 180 Dutch primary substations, using a second load estimate built from smart meters; a Korean series of four papers, three on one feeder and one on two; [ATLAS](https://smarter.energynetworks.org/projects/nia_enwl008/) on GB substations in 2016 | The one published result scoring both precision and recall reports F1.5 scores (a blend of precision and recall weighted towards recall, 0 for a useless detector and 1 for a perfect one) between about 0.2 and 0.5, from different detectors at different event lengths, and achieved with a second load estimate NGED does not have, so Flexpectation should expect worse rather than better |
| 5. Forecasting a substation as if it were always in its normal running arrangement | Three published responses: leave the level shifts in ([Huyghues-Beaufond et al. (2020)](https://doi.org/10.1016/j.apenergy.2019.114405)), rewrite the history ([Paredes and Vargas (2017)](https://doi.org/10.1049/iet-gtd.2017.0129)), or adapt to the new level ([de Vilmarest et al. (2024)](https://doi.org/10.1109/TPWRS.2023.3310280)) | Every published solution throws information away. In contrast, Flexpectation version 1 makes the abnormal periods an input to the ML model, and masks the abnormal periods in the training target |
| 6. Detecting faulty metering | [Bouman et al. (2024)](https://arxiv.org/abs/2405.16164)'s Dutch dataset which merges metering faults and switching into a single class | No GB project publishes labels or an accuracy figure, and Flexpectation is not labelling NGED's telemetry either, so a precision and a recall are out of reach. Flexpectation judges its cleaning rules downstream instead, by whether excluding the periods a rule flags improves the forecast on held-out data |
| 7. Recovering signed power from apparent-power meters | [Bouman et al. (2024)](https://arxiv.org/abs/2405.16164) and Western Power Distribution's 2017 [Time Series Data Quality](https://smarter.energynetworks.org/projects/nia_wpd_011/) both recover the sign from a second measurement of the same power; [SSEN's TRANSITION](https://ssen-innovation.co.uk/transition/) instead uses a meter's own 4-year average net demand together with a model of the generation behind that meter | Flexpectation version 1 forecasts the affected series in apparent power and flags those series to NGED; version 2 puts the magnitude inside a differentiable-physics forward model — the phase-retrieval formulation — and breaks the sign ambiguity with weather and with the persistence of flow direction |
| 8. Disaggregating unmetered solar and wind | [Teng et al. (2023)](https://doi.org/10.1016/j.rser.2023.113662) transferring from fully-metered Dutch substations, and [UK Power Networks' Power Flow to Solar Capacity](https://smarter.energynetworks.org/projects/nia_ukpn0104/), this work's direct predecessor | UK Power Networks' Power Flow to Solar Capacity attacked the same problem on the same kind of GB primary-substation data, and Open Climate Fix delivered that project too |
| 9. Disaggregating heat pumps, chargers, and batteries (stretch goal) | [Ostermann and Haug (2024)](https://doi.org/10.1186/s42162-024-00319-1) on aggregated charging demand day-ahead | Heat pumps, chargers, and batteries stay inside net demand in Flexpectation version 1 rather than being forecast separately |

### 1. Producing probabilistic forecasts of net demand at substations

#### The challenge

*Net* demand is *gross* demand minus whatever generation sits behind the substation. Flexpectation
version 1 forecasts the 20 substations in NGED's trial area, and version 2 extends that to net
demand at every grid supply point, bulk supply point, and primary substation in NGED's licence
areas. Our forecasts will be half-hourly, 14 days ahead, updated every 6 hours, and probabilistic.
NGED acts on the forecast 1 to 10 days ahead, and the question NGED asks of the forecast is "how
likely is net demand to run outside the substation's firm capacity?" rather than "what is the most
likely net demand?". Two costs hang on the answer: what NGED spends procuring flexibility to hold
demand under the substation's capacity, and what curtailing embedded generators costs to hold export
under a substation's export capacity. Both costs sit in the tails of the forecast distribution
rather than at its centre, so the same property of the forecast reduces both — extreme quantiles
that are calibrated, at both ends. A quantile is a level the forecast says net demand will stay
below a stated fraction of the time, and a calibrated quantile is one the outturn crosses exactly
that often: the level given as the 99th percentile is exceeded 1 time in 100, no more and no less.
Forecasting net demand is the highest priority of the nine challenges, and the other eight exist
mainly to improve our net-demand forecast.

#### What the literature says

**Summary:** A large literature exists on the topic of forecasting substation load, but very little of what we
read can be compared with the rest of that literature, and we found no papers driving a
probabilistic substation forecast from a weather ensemble across a 14-day horizon.

##### Papers reviewed

**Nine papers are summarised below.** Each entry below gives what was forecast and at what scale,
the horizon, the result and the baseline the result was measured against, and the weather input.

- **[Kaas et al. (2026)](https://arxiv.org/abs/2607.01966) — net load at 200 low-voltage feeders in
Germany, 4 days ahead.** A general-purpose foundation timeseries model (Chronos-2) that was not
trained on the authors' data beat every purpose-trained model on average error, 3.8 kW against 4.2
kW. Weather: 1–3 h forecasts, so effectively after the fact at the 4-day horizon.
- **[Hertel et al. (2026)](https://arxiv.org/abs/2607.15705) — load in Germany and Portugal, at
transmission level, 200 low-voltage feeders, and 287 individual customers, 4 days ahead.** Their
best model beat a day-type persistence forecast by 59.6% at transmission level, 42.3% at low-voltage
feeders, and 23.3% at individual customers. Weather: 1–3 h forecasts at the feeders, reanalysis (a
modelled reconstruction of past weather) elsewhere.
- **[Browell and Fasiolo (2021)](https://arxiv.org/abs/2103.10335) — regional net load at 14 grid
supply point groups in GB, day-ahead.** Their forecast held the same risk with **up to 24.6% less
upward reserve** than a fixed-tail alternative, falling to 3.2% at the least extreme risk level
tested. Weather: real forecasts.
- **[Pinheiro et al. (2023)](https://doi.org/10.1016/j.apenergy.2022.120493) — load at 96,989
secondary substations in Portugal, day-ahead.** Their forecast was 42–47% better than the reference
benchmark at system level, and **at substation level beat a naive forecast on 83–87% of
network-owned and 66–70% of customer-owned sites** (the paper's body text and the caption of a
figure on the next page give different pairs of numbers for that statistic, so the ranges span
both). Weather: real forecasts, 7–8 h old.
- **[Gilbert et al. (2023)](https://arxiv.org/abs/2206.11745) — load in GB at 4 levels, primary
substation down to household, day-ahead.** Combining forecasts gained **0.0–0.4% averaged over all
periods**, but **5.7–9.0% when restricted to peaks**. Weather: none at all.
- **[SSEN TRANSITION 2021](https://ssen-innovation.co.uk/transition/) — net load in Oxfordshire at 13
primary substations, plus their bulk supply points and their 33 kV and 11 kV feeders, 30 minutes to
10 days ahead.** The project reported **11 of 13 primary substation models below 10%** mean absolute
percentage error when fitted. Weather: 40-member ICON-EU ensemble to 4 days, then one deterministic
forecast to 10 days.
- **[Artificial Forecasting (Northern
Powergrid)](https://smarter.energynetworks.org/projects/npg_sif_006-1/) — demand and export at 551
primary substations with export data, 171 of them modelled, and active power at 729 secondary
substations; day-ahead to week-ahead at primary, evaluated to 11 days, and week- to month-ahead at
secondary.** The published results give **about 8% lower mean absolute error** of utilisation rate
than Northern Powergrid's existing method. Weather: real forecasts at primary; none in the published
secondary results.
- **[Ruhhütl et al. (2023)](https://doi.org/10.1049/icp.2023.0476) — load and generation at Austrian
primary substations, count not stated, day-ahead.** The paper reports **3–8% mean absolute
percentage error** for load, against no baseline the paper states, so not a target; varying with how
industrial and how large the supplied area was. Generation is forecast per technology: photovoltaic
to **1–5% of installed power**, run-of-river and biomass to **5–15%** mean absolute percentage
error. Linear and Gaussian regression were preferred over tree regression and a neural network.
Weather: real forecasts of global radiation, temperature, and precipitation, from a weather station
chosen per substation.
- **[Mesarcik et al. (2025)](https://doi.org/10.1049/icp.2025.1968) — active power in the
medium-voltage grid in the Netherlands, trained on 312 Alliander substations over 10 years and
tested on 6 chosen for difficult forecasting behaviour, 2 days ahead.** Their model reached a **mean
relative mean absolute error of 0.07** at the 50th quantile, against 0.08 for a gradient-boosted
machine and 0.09 for a linear model — both OpenSTEF models already in production at Alliander. Error
scaled by the signal's own 1st and 99th percentiles, not by a rating. Weather: Open-Meteo, 4
variables; their model trained on actual weather where the two baselines trained on 1-hour-ahead
forecasts.

#### What this means for Flexpectation

##### Model family choice

**Building Flexpectation version 1 on a gradient-boosted tree (GBT, such as XGBoost) is defensible,
but the literature paints GBTs as a sensible default rather than a proven winner**. A GBT builds its
forecast from hundreds of small decision trees, each one fitted to the error the trees before it
left behind.
[NGED's own Electricity Flexibility and Forecasting System (EFFS) project](https://smarter.energynetworks.org/projects/wpden03/)
picked XGBoost, which gave the best results of the three methods the project tested and was also
easy to automate, and no study we read shows a large, dependable margin for anything more
sophisticated than XGBoost at substation level.

**Both deployments by network operators that actually tried boosted trees kept a simpler model
instead.** [Pinheiro et al. (2023)](https://doi.org/10.1016/j.apenergy.2022.120493), running a live
system forecasting 96,989 Portuguese secondary substations, scored 199 MW root-mean-square error at
system level with a tuned gradient-boosted tree against 191 MW for a generalised additive model —
the boosted tree 4% worse — and rejected the boosted tree on the effort of tuning it and on the
interpretability given up with it.
[Artificial Forecasting](https://smarter.energynetworks.org/projects/npg_sif_006-1/) kept the
simpler model when forecasting customer export at primary substations: measured against the Bayesian
ridge regression they went on to adopt (a linear model that shrinks its coefficients and reports
uncertainty on them), boosted trees "helped some substations but harmed others".

**Neither end of the sophistication scale is a safe bet.**
[Mesarcik et al. (2025)](https://doi.org/10.1049/icp.2025.1968) caution about the uncertainty a
boosted tree reports rather than the accuracy it reaches: on the one substation whose calibration
they plot, their gradient-boosted machine's 95th percentile forecast corresponded to the 80th
percentile of the measured data, while a structured state space model and a linear quantile model
both tracked the ideal calibration line closely.
[Hertel et al. (2026)](https://arxiv.org/abs/2607.15705) make the same point from the other end of
the sophistication scale, because their purpose-built Transformer variant — the neural-network
architecture, not the electrical kind — lost to a standard encoder-decoder Transformer on all three
of their datasets. [Faustine et al. (2025)](https://doi.org/10.1109/TPWRS.2024.3400123) reach the
same conclusion at Stentaway substation in Plymouth, a primary substation in NGED's own South West
licence area: a multi-layer perceptron trained by quantile regression, the plainest feed-forward
neural network there is, matched or beat N-BEATS, N-HiTS, and a long short-term memory neural
network at Stentaway and at a low-voltage substation on Madeira serving about 100 consumers,
reaching a normalised root-mean-square error of 0.08 and 0.07 day-ahead against each substation's
installed capacity. Every model in that comparison was held to a comparable parameter count rather
than tuned individually, the weather covariates are observed rather than forecast, and the 7-day
figures come from feeding the day-ahead model its own output, so the margins bound how the
architectures rank rather than what Flexpectation should expect.

**What did help was refitting the model every month.** On both datasets where [Hertel et al.
(2026)](https://arxiv.org/abs/2607.15705) tried refitting, the retrained model beat the static one.
So, for Flexpectation, the literature suggests that the choice of model family may matter less than
the data, the feature engineering, and how often the model is refitted.

**Read those results knowing that when a paper says "XGBoost" it usually means a model with
considerably less feature engineering than what we plan to implement.**
[Kaas et al. (2026)](https://arxiv.org/abs/2607.01966) give their ML model lagged power, weather,
time, and six columns describing each low-voltage feeder — among them how many housing units,
industrial and commercial units, and photovoltaic systems the feeder serves — and nothing beyond
that: no clear-sky index, no wind power curve, no monotone constraints.
[Pinheiro et al. (2023)](https://doi.org/10.1016/j.apenergy.2022.120493) ran one of the two
comparisons on equal terms we found — their GBT and their generalised additive model — a regression
that fits a separate smooth curve for each input and adds the curves together — received the same
features, but that shared feature set was itself short: a linear trend, load lagged 24 hours and 1
week, time of day, 9 day types, the named public holidays, day of year, and temperature interacted
with time of day and with day of year. That shared feature set carried no irradiance and no wind.
[Faustine et al. (2025)](https://doi.org/10.1109/TPWRS.2024.3400123) ran the other, giving CatBoost
and a random forest the same lagged net load, irradiance, temperature, and calendar features as
their neural networks, and holding every model to a comparable parameter count rather than tuning
each one; that shared feature set carried no wind. So no published head-to-head we found gives a GBT
the feature engineering we plan to implement in Flexpectation version 1.

##### Limits on the published numbers

**None of the numbers above is a target for Flexpectation, because the studies cannot
be compared even with each other.** [Kaas et al. (2026)](https://arxiv.org/abs/2607.01966) and
[Hertel et al. (2026)](https://arxiv.org/abs/2607.15705) name different models as best, even though
they use data from the same 200 low-voltage feeders in Germany. Inside [Kaas et al.
(2026)](https://arxiv.org/abs/2607.01966), mean absolute error and an overload-decision metric name
different winners again. Neither disagreement is a mistake: the two papers test different sets of
models at different time resolutions, and the two metrics answer different questions.

**Accuracy got worse further down the electricity network in every study that forecast more than one
voltage level, but what shrank is the headroom above a naive rule rather than the usefulness of the
forecast.** [Hertel et al. (2026)](https://arxiv.org/abs/2607.15705) ran the same models against a
day-type persistence baseline on three datasets — a German transmission control area, 200 German
low-voltage feeders, and 287 individual Portuguese clients — and the margin over that baseline
shrank from 59.6% to 42.3% to 23.3% as aggregation fell. Their own gloss is that it is easier to
beat a simple approach on highly aggregated data than on volatile feeder-level and client-level
data.

**The one study we found reporting results substation by substation at scale shows how much skill
the shrinking headroom takes away at an individual site.** [Pinheiro et al.
(2023)](https://doi.org/10.1016/j.apenergy.2022.120493)'s model beat a "same time yesterday"
forecast at 83 to 87% of network-owned secondary substations but at 66 to 70% of customer-owned
ones.

**NGED's primary substations may not behave the same way, because a primary substation aggregates
far more customers than a Portuguese secondary substation does.** A forecast at a primary substation
may also carry a larger percentage error than one at a grid supply point and still support
flexibility procurement and curtailment decisions just as well, because what NGED needs from the
forecast is a reliable answer to "will this substation run outside its firm capacity?". This project
can measure whether decision-usefulness really is flat across voltage levels, and we intend to.

##### Horizon, ensembles, and tails

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

**Almost every substation-load study we found optimises *average* accuracy, but NGED's question is
about both ends of the distribution, and that is the literature gives a direct warning about this.**
[Browell and Fasiolo (2021)](https://arxiv.org/abs/2103.10335) is the only study we found that
models the tails explicitly, and they find that "below 1% and above 99% the forecasts based on
quantile regression only are not calibrated at any GSP [grid supply point] Group. Therefore, these
quantiles are not suitable for use in decision-making" — and that was with 5 years of half-hourly
data, across regions far larger than a substation. Outside those percentiles Browell and Fasiolo
switch to a fitted parametric tail at each end, and Flexpectation plans to follow Browell and
Fasiolo and fit parametric tails rather than reading extreme quantiles straight off the model. The
lower tail is the one curtailment turns on, because a substation runs closest to its export limit
when embedded generation is high and demand is low.

##### Model families for Flexpectation version 2

**All the text above is a verdict on Flexpectation version 1.** The more sophisticated ML model
families we plan to research in 2027 — such as pre-trained encoders, connectivity-map models, and
differentiable physics (explicitly building the known behaviour of a solar panel, a wind turbine, or
a building into the model, so the model has to learn only the physical parameters, not the
equations) — are planned to *simultaneously* reason about multiple sources of variation in
substation power flow, which the pipelines of separate models in this literature cannot do. The
closing section of this summary sets out the case for the work we plan in Flexpectation version 2.

**The evidence behind those three ML model families is uneven.**

- **Pre-trained models** have the best support of the three, but the measured result is for a
different kind of pre-training from the one we plan: the general-purpose model [Kaas et al.
(2026)](https://arxiv.org/abs/2607.01966) tested was pre-trained on time series, had never been
trained on their data, and still beat every purpose-trained competitor across 200 German
low-voltage feeders.
- **Connectivity-map models** have been measured on NGED's own published data: [Campagne et al.
(2025)](https://arxiv.org/abs/2507.03690) compare eight graph neural network architectures against
feed-forward, persistence, and foundation-model baselines on French regional load and on the GB
distribution networks' open smart-meter feed — about 2 million meters and 50,000 substations across the areas of NGED and SSEN — and the graph-aware models won on both. But their graphs are
built from geographic distance or from correlation between series, never from electrical
connectivity, so whether NGED's own connectivity map improves a forecast is still unanswered.
- **Differentiable physics** is established for a generator's own output:
  [Gijón et al. (2025)](https://arxiv.org/abs/2502.07344) fit a turbine model to a wind farm's
  metered production, and [Pierrot and Pinson (2024)](https://doi.org/10.1080/00401706.2024.2350421)
  fit a wind farm's capacity jointly with the forecast, so what would be new for Flexpectation is
  the substation rather than the method. A search for differentiable physics applied to substation
  demand forecasting produced no strong result, and we found nobody aggregating building thermal
  physics up to a substation and putting it inside a probabilistic forecast, though the ingredients
  exist separately.

##### Pre-trained encoders

**The case for pre-training an encoder rests on results from computer vision and Earth observation
rather than from energy forecasting.** The idea is to train one model (an "encoder") on a very large
body of data until the encoder can turn a raw input into a compact numerical summary that keeps what
matters and throws the rest away, and then to freeze the encoder's weights. Every later job reads
the frozen summary instead of the raw input, including jobs nobody had in mind while the encoder was
being trained, and each job needs only a small model of its own and a modest amount of its own data.
The heavy computation happens once, when the encoder is trained, and is then shared, instead of
being repeated from scratch by every model that needs it.

**Two recent models show how well the arrangement works.** [Siméoni et al.
(2025)](https://arxiv.org/abs/2508.10104) describe DINOv3, a 7-billion-parameter vision model
trained on unlabelled images. Siméoni et al. keep DINOv3 frozen throughout their evaluation and read
every task off its representations, reporting that fine-tuning of the encoder "is not necessary to
obtain strong performance" on tasks as different as segmentation, depth estimation, and object
detection. [Brown et al. (2025)](https://arxiv.org/abs/2507.22291) describe AlphaEarth Foundations,
which encodes satellite and other Earth-observation data into one 64-byte embedding per 10-metre
cell per year, and report that the embeddings cut error magnitude by about 24% on average against a
representative sample of other featurisation methods, across a broad set of sparse-data mapping
evaluations, without re-training on any of them.

**Flexpectation can learn from this prior work on pre-training encoders.** The breadth matters as
much as the freezing: DINOv3 and AlphaEarth Foundations are each a single encoder serving many
different tasks rather than one encoder per task, and that is what Flexpectation plans for its own
weather encoder — the same frozen representation feeding the substation net-demand forecast, the
metered-generator forecasts, and the disaggregation of unmetered generation.

**Neither result promises that a pre-trained encoder beats hand-designed features.** Brown et al.
report that learned featurisations "don't always outperform designed featurization methods in scarce
data regimes", and present AlphaEarth Foundations as the exception on their own evidence: the one
learned featurisation in their comparison that consistently beat the alternatives they tested. The
gradient-boosted tree on hand-designed features stays the baseline the encoders have to beat.

**The encoders Flexpectation plans to pre-train cover weather and time, and possibly a third for
place.** We plan to research a neural network that turns the raw ECMWF ensemble into a calibrated
probabilistic weather forecast in physical units, which a substation model then reads, alongside a
time encoder that learns how people use the calendar — e.g. that Christmas is not an ordinary day —
and possibly a space encoder holding the standing geographic context of each substation.

**Both halves of the weather encoder have been built.**
[Rasp and Lerch (2018)](https://arxiv.org/abs/1805.09091) built a neural network that post-processes
a 50-member ECMWF ensemble into calibrated probabilistic 2-metre temperature at 537 German weather
stations 48 hours ahead, cutting mean continuous ranked probability score — a single number scoring
a whole forecast distribution against what actually happened, where lower is better — from 1.16 for
the raw ensemble to 0.78, with a learned per-station embedding one of the two components the authors
credit for the gain. [Mitra and Ramavajjala (2023)](https://arxiv.org/abs/2312.00290) built the
second: they freeze a weather autoencoder and train small models on the frozen representation alone,
at accuracy comparable to purpose-built models, though the targets they predict are further weather
variables rather than anything on an electricity network.

**The nearest we found anyone joining the two is one entrant in HEFTCom, a competition to forecast a
GB wind-and-solar portfolio day-ahead.** [Browell et al.
(2026)](https://doi.org/10.1016/j.ijforecast.2025.10.005) report that team Rnt fed embeddings from
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
gas generator. The closest cases are both at Austrian primary substations, where
[Ruhhütl et al. (2023)](https://doi.org/10.1049/icp.2023.0476) forecast biomass generation from the
previous day's output, and forecast market-dispatched pumped-storage hydro from the generation
schedule its operator is obliged to provide.

#### What this means for Flexpectation

##### Model choice for wind and solar

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
(2026)](https://doi.org/10.1016/j.ijforecast.2025.10.005) conclude that gradient-boosted trees
remain competitive for day-ahead wind and solar forecasting, with performance depending heavily on
implementation. NGED's own EFFS project selected XGBoost when it evaluated model families.

**One result cuts the other way, though team Rnt's route is not an argument against trees.** Rnt
finished third in HEFTCom's forecasting track using no tree-based model at all, feeding embeddings
from machine-learned weather-forecasting models they built in-house into downstream neural networks
that predicted wind and solar generation — a route that rests on building and running a weather
model, not on a different downstream model family.

##### The solar-forecasting meta-analysis

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
forecast better than an autoregressive model does. That question matters for Flexpectation. And only
19% of the 4,687 observations in [Nguyen and Müsgens (2026)](https://doi.org/10.1063/5.0300682) use
numerical weather prediction as an input at all, against 91% that use lagged power, so most of the
evidence separating the model classes comes from models with no weather forecast in them.

**The bottom row is the weather model used raw, and for most of that sample no power curve is
involved.** The class represented by the bottom row in the table above is the numerical weather
prediction irradiance field itself — usually global horizontal irradiance, at most post-processed or
averaged across several weather models — used as the forecast rather than fed as an input to a
fitted model. Of the 188 papers surveyed by
[Nguyen and Müsgens (2026)](https://doi.org/10.1063/5.0300682), 118 forecast irradiance rather than
photovoltaic (PV) power output. Only 70 papers in the survey forecast PV power output, so for most
of the sample the weather model's irradiance field is directly comparable to the irradiance those
papers forecast. The authors' regression separates the model class and the forecast target as
separate variables, so the 14.3-point penalty is estimated with the target held constant, but the
authors never report which targets the numerical-weather-prediction papers were forecasting. Nguyen
and Müsgens's advice is to exhaust the simple models first, because classical statistical
time-series methods "still have very good performance compared to more complex methods such as
individual ML models".

**Most of NGED's metered generators are solar, and the largest meta-analysis of solar forecasting we
found confirms the importance of NWP inputs at the lead times Flexpectation cares about.** Numerical
weather prediction is the largest input effect [Nguyen and Müsgens
(2026)](https://doi.org/10.1063/5.0300682) measure, and the inputs that improve skill at short range
carry the opposite sign at day-ahead range. The table below shows percentage points of skill score
improvement over the classical statistical baseline:

| Input | Intra-hour (up to 1 hour) | Intra-day (1 to 6 hours) | Day-ahead (over 6 hours) |
|---|---|---|---|
| Numerical weather prediction | −9.0 | −2.3 | **+11.6** |
| Locally measured weather | not significant | +9.1 | +5.1 |
| Lagged solar power | +5.7 | +8.2 | **−6.4** |
| Data from neighbouring sites | +3.6 | +3.9 | −5.5 |

Each input is a yes-or-no variable rather than a choice between alternatives, so one model can carry
several. Their sample is deterministic forecasting of irradiance or plant output rather than
probabilistic substation net demand.

##### Differentiable physics for generators

**For generators, the benefit from better weather-to-power physics is largest at short lead times.**
Differentiable physics (DP) attacks the weather-to-power half of the error, so on [Dantas and
Browell (2026)](https://doi.org/10.1002/we.70079)'s measurement DP has most to offer inside the
first 2 to 3 days of the 1-to-10-day window NGED acts on, and less beyond it, where the weather
forecast itself is the largest source of error.

**Adding a learned residual to a physical generator model is established practice, and the physical
model can be fitted to the power data rather than read off a specification sheet.**
[Gijón et al. (2025)](https://arxiv.org/abs/2502.07344) write the actuator-disc equation for a
turbine's power output, `P = ½·Cp·ρ·A·v³`, into a TensorFlow model, and treat the air density ρ and
the area A swept by the blades as known. The power coefficient Cp — the aerodynamic term the
equation does not fix — is estimated from wind speed, pitch angle, and rotor speed by a neural
network whose sigmoid output layer holds Cp below the Betz limit of 0.5926. That neural network is
trained against the measured power of a wind farm of four turbines, so the gradient of the power
error passes back through the physical equation itself. A second neural network is then trained on
the residual, cutting the physics model's mean absolute percentage error by 37% and its mean
absolute error by 28%, with conformalised quantile regression supplying the uncertainty. Gijón et
al. also compare their hybrid model against a purely data-driven model given the same eight inputs,
and report that the hybrid model "essentially matches" the data-driven model rather than beating it,
so adding the physics model made the forecast interpretable without making it less accurate.

**But Gijón et al. predict power from measured wind rather than forecasting it days ahead.** Their
inputs are the turbine's own measurements at the moment being predicted, so their accuracy says how
well a fitted turbine model turns a known wind speed into power, not how well a forecast of that
wind speed turns into a forecast of power days ahead. We found nobody putting a differentiable model
of a generator inside a distribution network's probabilistic net-demand forecast.

##### Inferring engineering parameters

**A second reason to try differentiable physics on generators, beyond the accuracy gain above, is to
infer the engineering parameters NGED does not hold: the capacity a site can actually export today,
a solar array's tilt and azimuth, a turbine's power curve.** The generation forecasts in this
literature are handed those engineering parameters:
[Teng et al. (2023)](https://doi.org/10.1016/j.rser.2023.113662) are given each site's capacity, and
HEFTCom's portfolio was the 1.2 GW Hornsea 1 offshore wind farm plus the solar capacity of a region.
When an export-cable fault cut that wind farm's available capacity mid-competition, the winning team
clipped its quantiles to the capacity implied by the outage notices the farm is obliged to publish,
while the organisers' benchmark ignored the fault and, in
[Browell et al. (2026)](https://doi.org/10.1016/j.ijforecast.2025.10.005)'s words, "performed
extremely poorly as a result". NGED's embedded generators publish no outage notices of that kind.
Estimating each generator's available capacity from its own metered output instead is challenge 3
below, which sets out the published methods in detail.

**NGED's Embedded Capacity Register (ECR) gives a registered capacity for generation of 50 kW and
above, but provides no other engineering parameters.** NGED's August 2026 ECR lists 5,598 connected
generators totalling 11,456 MW, of which 4,202 sites and 5,958 MW are solar. But a registered
capacity is *contractual* rather than operational — the export limit is the one "permitted as per
the connection agreement" — and the register carries no panel tilt, panel azimuth, or ratio of
direct-current to alternating-current rating. Hence Flexpectation plans to infer those engineering
parameters from the power data, using differentiable physics.

**A differentiable model could infer both the operational capacity and the panel orientation of each
generator, and each of those two inferences has been made to work on its own.** [Pierrot and Pinson
(2024)](https://doi.org/10.1080/00401706.2024.2350421) treat a wind farm's capacity as a
time-varying bound fitted jointly with the forecast, and beat probabilistic persistence by 34.2% on
continuous ranked probability score over a 5-month test period, drawn from 14 months of data, at the
Anholt offshore wind farm, though their one clean test of tracking the bound on its own gained
2.43%. [Meng et al. (2020)](https://doi.org/10.1016/j.solener.2020.09.077) infer the tilt and
azimuth of 13 roof photovoltaic systems in the Netherlands to mean absolute errors of 4.3° and 4.5°,
matching the shape of each system's hourly output against plane-of-array irradiance from a weather
station up to 195 km away. Because both curves are normalised before matching, their method needs no
nameplate rating. But Meng et al. do not forecast PV power. They only infer tilt and azimuth.

**We have not found any evidence in the literature to tell us how much a PV power forecast would be
improved by inferring tilt and azimuth, so Flexpectation treats it as a hypothesis to test rather
than a settled prize.** [Meng et al. (2020)](https://doi.org/10.1016/j.solener.2020.09.077) and
[Saint-Drenan et al. (2015)](https://doi.org/10.1016/j.solener.2015.07.024) both recover a system's
tilt and azimuth from its metered alternating-current power output paired with an irradiance series
measured somewhere else — a weather station for Meng et al., the HelioClim-3 satellite database for
Saint-Drenan et al. — and land within a few degrees, but report their accuracy in degrees alone.
Saint-Drenan et al. also found that an azimuth fitted 5° from the true azimuth gave better
simulations than the true value, because the fit balances the systematic error of the physical model
— so accuracy in degrees is the wrong target. What matters is an *effective* tilt and azimuth that
make the forecast right.

**Inferring the engineering parameters of a generator needs different machinery depending on whether
the generator has site-specific metering, compared to disaggregating unmetered generators from
substation power flow.** For a single metered site, fitting tilt, azimuth, and the effective direct-
and alternating-current capacities is the plan for Flexpectation — by gradient descent inside the
forecast rather than by grid search, so that the fit is joint and probabilistic. Challenge 3 below
sets out how that effective capacity would be estimated. For unmetered solar behind a substation,
Saint-Drenan et al.'s algorithm "performs poorly", because it assumes one orientation per plant
where the series is "the aggregated production of modules with different orientations".
Flexpectation therefore plans to implement a fleet model representing the aggregate as a learned
mixture of east-, south-, and west-facing basis shapes, with a soft clip standing in for many
differently-sized inverters saturating in turn. Challenge 8, below, discusses disaggregation in more detail.

##### The battery, gas generator, and biofuel plant

**The trial area's battery, gas generator, and biofuel plant each need a method, and the literature
supplies one to borrow for the battery, two for the gas generator — one needing the operator's
generation schedule, the other never yet fitted to an embedded generator — and a partial method for
the biofuel plant.** For the battery, [Bian et al. (2024)](https://doi.org/10.1109/TSG.2023.3303469)
recover a price-taking storage operator's own optimisation parameters from historical prices and
observed dispatch.

**The closest published case for the gas generator forecasts a market-dispatched plant from the
schedule its operator provides, not from weather or from the plant's own history.**
[Ruhhütl et al. (2023)](https://doi.org/10.1049/icp.2023.0476) call predicting pumped-storage hydro
"almost impossible", because the plant follows continuously changing market prices and the
operator's own strategy, and forecast it instead by linear regression on the generation schedule its
operator is obliged to provide, together with temperature. They report no accuracy figure for that
class of plant, saying only that such plants "depend highly on the accuracy of the provided
schedule", so what the method needs is a schedule rather than a better model.

**The second route models how the gas generator picks its own output, and nobody we found has fitted
such a model to an embedded generator's metered output.** Fitting a model of that shape to an
embedded generator's metered output is what Bian et al. do for storage, and we found nobody doing it
for a gas generator. [Short et al. (2017)](https://doi.org/10.1016/j.apenergy.2016.04.052) model how
a decentralised combined heat and power plant picks its own output, as a mixed-integer linear
program over a piecewise-linear fuel cost, ramp limits, and a start-up cost, maximising profit
against day-ahead and intra-day prices. What could transfer to Flexpectation's
differentiable-physics work is the structure rather than the solver: Short et al. approximate the
fuel cost by three affine pieces chosen "to ensure convexity", so the economic-dispatch half of
their model is a linear program a gradient can pass back through, while the on/off unit-commitment
decisions would have to be smoothed first.

**For the biofuel plant the same paper supplies a partial method.**
[Ruhhütl et al. (2023)](https://doi.org/10.1049/icp.2023.0476) forecast biomass generation behind
each Austrian primary substation from the previous day's generation, scaled to installed power and
spread across the day as a constant band, to a mean absolute percentage error of 5 to 15% — the same
shape of problem, though a biomass station burning solid fuel is not the same plant as a biofuel
generator.

**A GB gas-network project has tested the declared-schedule route on embedded gas generators, and
what stopped it was data rather than modelling.** SGN and Northern Gas Networks'
[Forecaster for Embedded Generation (FEmGE)](https://portal.futureenergynetworks.org.uk/content/projects/NIA2_SGN0081)
2026 Network Innovation Allowance (NIA) project reconstructed gas generators' electricity output
from the Physical Notifications each plant gives the National Energy System Operator (NESO), plus
the balancing bids and offers NESO accepts. Plants that self-dispatch rather than trade through the
Balancing Mechanism were placed out of scope as harder still, and no public record matches a plant's
electricity meter number to its NESO unit identifier, with many small plants sitting inside
aggregated units whose composition is unpublished. Forecasting performance also "reduced
significantly" when transmission-connected plant was excluded, because distribution-connected
generators are a small part of a zonal total — a problem NGED does not have, because NGED meters its
gas generator directly. FEmGE published no accuracy figure, and concluded that more complex
modelling would not improve accuracy without wider access to embedded generators' own data.

### 3. Estimating the effective capacity of metered generators

#### The challenge

We call the amount of generation actually available at a metered site its *effective capacity*: the
power output a generator could produce right now if the weather allowed, as opposed to its nameplate
rating. Turbines go out for repair, inverters degrade. A 20 MW wind farm that has been limited to 14
MW for a month is, for forecasting purposes, a different wind farm, and a model trained on its
nameplate rating cannot see the difference.

#### What the literature says

A method exists for each generation technology separately, but we found none run across a mixed
fleet of individually metered generators at a distribution network operator. Only one study we found
measures what estimating capacity is worth downstream, for wind alone and at the scale of a national
bidding zone; the other estimates a wind farm's capacity but treats the estimate as a known input
rather than measuring what getting it right is worth.

#### What this means for Flexpectation

**Flexpectation version 1 needs an estimator that can track effective capacity downwards, and that
is exactly where the two published wind methods differ.** [Dantas and Browell
(2026)](https://doi.org/10.1002/we.70079) needed available capacity for the same reason we do, and
rather than use a nameplate rating they estimate a time series of available capacity for each farm
from that farm's own metered production, needing no capacity register and no outage messages. The
general shape of that capacity-estimation rule is a running maximum of production, which ratchets
upwards and never comes back down. In contrast, [Viotti et al.
(2026)](https://doi.org/10.1002/we.70136) fit the most likely capacity time series instead, by
quadratic optimisation against a capacity factor simulated from reanalysis weather and a power
curve, and they publish a monotonic variant alongside a non-monotonic one. The direction of travel
is what matters for NGED: a turbine out for repair for a month makes effective capacity *fall*, and
a ratchet cannot follow it down. Flexpectation version 1 will therefore implement estimators that
can fall as well as rise.

**The published numbers favour fitting over ratcheting, on hourly region-aggregated data.**
[Viotti et al. (2026)](https://doi.org/10.1002/we.70136) say that estimating capacity using a
running maximum "requires monotonically increasing capacity and relies on frequent high wind
events". Viotti et al. publish a non-monotonic capacity estimator, which can follow capacity down
when a turbine goes out for repair. The non-monotonic variant produced the lowest day-ahead forecast
error across Sweden as a whole, **2.0% below** a model normalised by the running maximum on mean
absolute error and **2.3% below** it on root-mean-square error, and the authors say the
non-monotonic variant yields the best forecasts "possibly because it captures real changes in
available capacity or corrects seasonal wind-speed biases", while cautioning that the difference in
forecast error may not reflect the quality of the normalisation at all.

**Two caveats temper both figures for NGED.** Viotti et al.'s target is a Swedish bidding zone
rather than a single farm, and they report that at 5-minute resolution the running maximum is
already a robust estimate of one farm's installed capacity, so the fitting shows its advantage on
hourly, region-aggregated data. Whichever estimator wins, normalising by effective capacity stays a
hypothesis to test rather than a settled preprocessing step, because no study we found has measured
whether it improves the forecast NGED acts on.

**A competition on Norwegian wind confirms that normalising by capacity is what practitioners reach
for, and also that the entrants were handed the capacity rather than having to estimate it.** WindAI
asked for the hourly wind power of four Norwegian bidding zones two days ahead, and supplied wind
park metadata alongside the weather and production data; [Authen et al.
(2026)](https://doi.org/10.5617/nmi.13106) report that Statnett scored 5% of its weighted assessment
on "robustness to changes in installed wind power capacity, evolving weather patterns, long-term
climate variability". The two highest-placed teams both predicted capacity factor rather than
absolute production, "to account for maintenance events and future capacity expansions", and both
used Nord Pool unavailability messages to cover planned maintenance and outages. A team given an
honourable mention fitted a physical power curve for each wind park under sequential Bayesian
updating over a sliding window, to absorb "capacity changes or the commissioning of new wind parks",
and initialised a new park's parameters from a prior built on the existing fleet — an answer to the
cold-start problem an estimator faces at a generator that has just connected. Authen et al. decline
to credit any of that with the differences in accuracy, because unavailability messages covered
between 1% and 13% of timestamps depending on the bidding zone, so downtime events "represent only a
limited fraction of the full dataset". Two conclusions follow for Flexpectation. Independent teams
converging on capacity factor as the target is evidence about what practitioners believe rather than
a measurement of what the belief is worth, so the hypothesis in the paragraph above stands
unaltered. And the part WindAI could skip is the part NGED cannot: the Embedded Capacity Register
records the export limit permitted by a site's connection agreement rather than what the site can
generate, so Flexpectation has to estimate the effective capacity that WindAI's entrants were given.

### 4. Detecting switching events

#### The challenge

When a cable fault or planned maintenance moves part of a distribution network from one substation
to another, the load the first substation meters steps down. Each substation that picks up part of
that transferred load records a rise, with no change in the underlying demand. The pick-up is
usually shared across two or three neighbouring substations. NGED's substations spend roughly a
tenth of their operating time in an abnormal running arrangement. Switching labels exist for the
Flexpectation trial area but not for NGED's entire distribution network, so any method meant to
scale beyond the trial area has to work from power measurements alone.

#### What the literature says

**We found several papers on detecting switching events from metered load, but all these approaches
only consider one substation at a time.** [Bouman et al. (2024)](https://arxiv.org/abs/2405.16164)
detect switching at a real network operator, but detect it in the gap between the substation's own
meter and a second estimate of the same load, built from smart-meter and bulk-customer readings
taken below the substation. A Korean series of four papers detects load transfers on a distribution
feeder from that feeder's own load alone. All four Korean papers are open access, and all four score
against the same nine logged transfers on the Kimhwa distribution feeder in Gangwon province,
measured hourly through 2019, and one of the four against a second feeder as well.

| Paper | Method | Logged transfers found |
|---|---|---|
| [Kim et al. (2020)](https://doi.org/10.3390/en13174358) | Long short-term memory (LSTM) neural network, flagging where measured load departs from its prediction | 7 of 9 |
| [Kim et al. (2022)](https://doi.org/10.3390/en15041441) | Polynomial and standard-pattern preprocessing | 7 of 9, and 7 of 7 on a second feeder |
| [Kim (2024)](https://doi.org/10.5370/KIEE.2024.73.11.1873) | A moving average and a moving standard deviation, thresholding the residual of a seasonal-trend decomposition | **8 of 9** |
| [Kim (2025)](https://doi.org/10.5370/KIEE.2025.74.11.1757) | Robust seasonal-trend decomposition, a Haar wavelet transform of the residual, then Pruned Exact Linear Time changepoints, then an isolation forest over each candidate | 7 of 9 |

Every one of those counts is the share of logged events found, and no paper in the series reports a
false-alarm rate. The scores do not track how elaborate the method is: the simplest of the four, a
threshold on a decomposition residual, found the most events, and [Kim
(2025)](https://doi.org/10.5370/KIEE.2025.74.11.1757)'s pipeline — the closest of the four to what
Flexpectation plans — found 7 of the 9, an average detection rate of 78%.

Electricity North West's [ATLAS](https://smarter.energynetworks.org/projects/nia_enwl008/) project
sorted step changes into erroneous data and network reconfigurations on GB substations in 2016, from
power measurements alone, but published no precision or recall for either rule.

#### What this means for Flexpectation

**[Bouman et al. (2024)](https://arxiv.org/abs/2405.16164) detect switching events, but do not
forecast.** Bouman et al., working with the Dutch network operator Alliander, study 180 primary
substations at 15-minute resolution over roughly a year, detecting events that run from a few
minutes to several months. Alliander's purpose is capacity planning: a switch pushes the maximum and
minimum load a substation records to the wrong value, and those two extremes decide whether the
substation needs a bigger transformer, so the detected periods are cut out of the history before the
extremes are read off. In contrast, Flexpectation needs a forecast that keeps running through a
switching event.

**Only one published result we found scores switching detection on both precision and recall, and
its scores are low: about 0.2 on events shorter than 3 days, and about 0.5 on events of 42 days or
longer.** [Bouman et al. (2024)](https://arxiv.org/abs/2405.16164) score their detectors with the
F1.5 score, which blends precision — the share of flagged points that really were switching — with
recall — the share of switched points the detector flagged — weighting recall the more heavily of
the two. An F1.5 score of 1 is a perfect detector and 0 is a useless one, so higher is better. Those
two scores come from different detectors, because no single method they tried wins across the range.
Both figures were achieved on a Dutch distribution network, with the help of a second load estimate
constructed bottom-up from smart meter data.

**Flexpectation will model its own reference time series for each substation.** Alliander's
bottom-up estimate gives [Bouman et al. (2024)](https://arxiv.org/abs/2405.16164) a second opinion
on what each substation's power should have been. Flexpectation plans to produce that second opinion
from the substation's own meter plus weather and the calendar. The first attempt is classical: a
multiple seasonal-trend decomposition of each series into a trend and daily, weekly, and annual
cycles, leaving a remainder in which a switch shows up as a sustained level shift. The second uses
the project's existing XGBoost machinery, trained with no power-lag features, so that an earlier
switching event cannot contaminate the expected-power estimate. Neither route needs metering from
below the substation.

**Flexpectation also plans to investigate using a signal that Bouman et al.'s
one-substation-at-a-time method cannot see: the power has to go *somewhere*.**
[Bouman et al. (2024)](https://arxiv.org/abs/2405.16164) score each substation against its own
history — "the current analysis considers one year of measurements for one station at a time" — so
nothing in their method asks whether the power that left one substation turned up at another. When
one substation's metered power drops, the substations that picked the load up should rise at the
same moment, and their rises should sum to the drop. A step whose rise and drop fail to balance is
more likely a meter fault or a one-off than a switch. That mismatch is where a per-substation
detector spends its false positives. The catch is that an NGED transfer usually fans out across two
or three neighbours, so the search runs over subsets of neighbours rather than over pairs, and the
balance holds only approximately.

**We looked for a method that checks both sides and found none, and the closest published precedent
is a 1984 regression written for long-range planning.** The search ran across OpenAlex, Semantic
Scholar, Crossref, arXiv, the works citing [Bouman et al. (2024)](https://arxiv.org/abs/2405.16164),
and the project titles on the Energy Networks Association's Smarter Networks Portal.
[Willis et al. (1984)](https://doi.org/10.1109/TPAS.1984.318713) correct annual peak-load curve fits
rather than detecting an event at a point in time, and their regression needs neither the size nor
the direction of a transfer as an input. The title names a "load transfer coupling" regression,
which suggests the fit couples the substations that exchange load — the feature that would make it
the closest precedent — but we could not obtain the full text to check, and the abstract does not
say.

**NGED's switches are usually partial and fan out to two or three substations, so we should expect
worse F1.5 scores than Bouman et al.'s 0.2 to 0.5, not better, even with the two additions above.**
Do not judge the difficulty from how obvious a switch looks on a chart. A negative result is worth
having here, because evidence that switching cannot be recovered from power measurements alone would
justify extracting switching labels from NGED's operational systems instead of continuing to infer
them.

### 5. Forecasting a substation as if it were always in its normal running arrangement

#### The challenge

NGED plans its distribution network against what each substation would carry under its normal
running arrangement. As such, Flexpectation aims to forecast substations as if they were always in
their normal running arrangement, including a substation that has been sitting in an abnormal
arrangement for weeks. Predicting the power flow under the normal running arrangement means the
forecasting target goes unmeasured during periods of abnormal running, and leaves the training
history contaminated: past readings taken while the distribution network was abnormally configured
describe a different scenario from the scenario being forecast.

#### What the literature says

**Researchers respond in one of three ways:** leaving the level shifts in, as [Huyghues-Beaufond et
al. (2020)](https://doi.org/10.1016/j.apenergy.2019.114405) do; rewriting the history, as [Paredes
and Vargas (2017)](https://doi.org/10.1049/iet-gtd.2017.0129) do; or adapting to the new level, as
[de Vilmarest et al. (2024)](https://doi.org/10.1109/TPWRS.2023.3310280) do.

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
for switching robustness without saying how much accuracy the trade gave up. We found one substation
study that conditions its forecast on an operating-state label —
[Liu et al. (2019)](https://doi.org/10.1109/ACCESS.2019.2951422), for switching that moves load
between transformers inside a single substation.

#### What this means for Flexpectation

**Every published solution we found throws information away.** Leaving the level shifts in the data
hurts performance, rewriting history erases the level shifts, and adapting to the new level forgets
that a switch happened. Adapting is disqualifying here, because the quantity NGED needs is what the
substation *would* have carried under its normal arrangement. Flexpectation version 1 will therefore
detect the abnormal periods automatically, flag the lagged power inputs that fall inside one, and
drop those periods from the training target — a combination no published method we found uses.

**Rewriting the history is the fallback, because among the three published responses it is the only
one that targets the quantity NGED needs and reports a measured benefit for doing so.**
[Paredes and Vargas (2017)](https://doi.org/10.1049/iet-gtd.2017.0129) rewrite the history to the
level it would have had if the switch had never happened, across 169 real feeders, and report better
medium-term forecasts for it. Northern Powergrid's Artificial Forecasting project rewrites its
history too, in step 6 of the data-preparation pipeline set out in its
[Alpha deliverable](https://smarter.energynetworks.org/projects/npg_sif_006-1/) *WP2-D2 Results
Scope Item 2*, which rescales a block of older readings to align its median with the median of the
most recent block whenever the older block's median falls outside the 10th-to-90th-percentile range
of the recent one. Northern Powergrid hold no readily accessible record of their own distribution
network's configuration changes, so that pipeline hypothesises the timestamps from the load itself
and confirms them with the control room — the position NGED is in outside the trial area.

**The fix is a level shift applied to the *older* half of each series.** Paredes and Vargas measure
how far average demand moved across the step and add that difference to every reading before it, and
the variant they recommend uses a separate difference for each hour of the day and each day of the
week rather than one number for the whole series. Paredes and Vargas take the event times from
expert identification rather than from a detector, since detection was not their subject. Adaptive
models are the live alternative — they track a new level once it arrives, including one that arrives
abruptly — but a model that simply adapts to a new load level cannot report what the substation
would have carried under its normal arrangement, which is the quantity NGED needs.

**Flexpectation version 1 feeds the model its switching-contaminated history deliberately: the
abnormal periods are an input to the ML model but are removed from the training target.** First,
take each substation's abnormal running arrangements from the detector of challenge 4 rather than
from an operational log, and hand those periods to the model as a flag on each lagged power input,
so the model can read a lag that falls inside an abnormal period correctly. Second, drop the
abnormal half-hours from the training target, so the model is never asked to predict an abnormal
arrangement. An alternative worth testing early is to skip the flag and give the model challenge 4's
reference time series alongside the lagged power, leaving the model to notice for itself where a
lagged reading departs from what the reference series expected.

**The nearest published precedent for each half of the plan sits outside the problem NGED has.** For
the first half, [Liu et al. (2019)](https://doi.org/10.1109/ACCESS.2019.2951422) fit a separate
regression per substation operating condition, though their switching moves load between
transformers inside one substation, so the substation total stays metered throughout. For the second
half, [Salinas et al. (2020)](https://doi.org/10.1016/j.ijforecast.2019.07.001) state the mechanism
for a probabilistic forecaster, motivated by retail stock-outs, and say they omitted the experiments
for it. Searching OpenAlex, Crossref, and arXiv, we found no load-forecasting study reporting what
dropping contaminated periods from the training target is worth, so Flexpectation will have to
measure that itself.

**Flexpectation version 2 plans to go further and treat the normal-arrangement demand as a latent
variable to be inferred for every metered substation, rather than a series to be repaired first**,
through a differentiable-physics model of each substation with separate photovoltaic, wind, and
demand components. Recovering a demand the meter never saw is mature where demand is censored —
airline revenue management calls it unconstraining, and retail and electric-vehicle-charging work
calls it censored-demand recovery, as in [Hüttel et al. (2023)](https://arxiv.org/abs/2301.06418) —
but censoring is one-sided, so the observed value bounds the latent demand from below, whereas an
abnormal running arrangement substitutes a different set of customers and can read either side of
the normal-arrangement demand. We found no published model that recovers a latent
normal-running-arrangement demand for a distribution substation.

### 6. Detecting faulty metering

#### The challenge

NGED's telemetry carries stuck values that repeat unchanged for hours or days, zeros that mean "no
reading" rather than "no load", physically impossible values, and gaps running from a single
half-hour to several months. A model trained on uncleaned data learns the fault, and a forecast that
fails silently because the series' recent history was stuck is worse than a forecast that reports
itself degraded.

#### What the literature says

**Faulty metering is usually a data-cleaning step mentioned in passing rather than a research
problem in its own right.** The only public dataset with labelled faults we found is
[Bouman et al. (2024)](https://arxiv.org/abs/2405.16164)'s, published through the Dutch network
operator Alliander's open data portal.

#### What this means for Flexpectation

**The literature offers two shapes of detector — test a reading against a redundant measurement of
the same power, or against a forecast of what that reading should have been — and NGED's
primary-substation telemetry rarely carries the redundant measurement, which leaves the forecast
route.** One family tests a measurement against a physical relationship the measurement has to
satisfy: UK Power Networks' Distribution Network Visibility checked 377 remote terminal units
against the physics their readings have to obey rather than against a forecast, and found 95% of
those units obeyed the expected logic within 15 kVA.
[Bouman et al. (2024)](https://arxiv.org/abs/2405.16164) do the same with a second estimate of a
substation's load, built from smart meters. The other family tests the measurement against a
forecast of what that measurement should have read, the route
[Moriano et al. (2016)](https://doi.org/10.3390/s16010085) and
[Martín et al. (2018)](https://doi.org/10.3390/s18113947) take to find calibration drift in
secondary-substation monitoring equipment.

**The published method that fits NGED's telemetry most closely treats faulty metering and switching
as one challenge, and merging the two faults is exactly what stops the Dutch labels separating
faulty metering from switching.** [Bouman et al. (2024)](https://arxiv.org/abs/2405.16164) treat
measurement errors and switch events as the two contaminants that must be filtered out before
substation measurements can be used, and detect both on the same residual. Detecting both on one
residual is also what merges the two classes in the Dutch labels, so the Dutch dataset can train a
detector but cannot settle whether a flag is a stuck meter or a distribution network reconfiguration
— the separation challenges 4 and 6 exist to make.

**The faults that dominate NGED's telemetry are not the faults the model-based detectors were built
for, and the GB projects that met those faults used threshold rules.** Moriano et al. and Martín et
al. score calibration gain and offset drift plus outliers, injected into clean data rather than
found in the wild, whereas NGED's telemetry carries stuck values, false zeros, and multi-month gaps.
NGED's own Time Series Data Quality searched for zeros, for "non-varying non-zero values, perhaps
indicating a 'stuck' or incorrectly configured sensor", and for gaps, and found metering defects
common rather than exceptional on NGED's own data: 13.8% of analogues in the South West licence area
recording only zeros, and 63% of new solar sites' analogues not commissioned correctly. A detector
built on the assumption that faults are rare is the wrong shape for NGED's telemetry.

**None of the three GB projects reports how often its checks are right.** Electricity North West's
[ATLAS](https://smarter.energynetworks.org/projects/nia_enwl008/), UK Power Networks'
[Distribution Network Visibility](https://www.ofgem.gov.uk/sites/default/files/docs/2014/03/dnv_cdr_version_3.0_270214.pdf),
and NGED's own [Time Series Data Quality](https://smarter.energynetworks.org/projects/nia_wpd_011/)
all tackled faulty metering substantively. None of the three published a figure for how often a
flagged reading really was faulty, nor a label set to measure that against. Distribution Network
Visibility's 95% is the share of units whose readings obeyed the expected logic, not a detection
accuracy, so the GB record tells us what to look for rather than how well the approaches worked.
What Distribution Network Visibility did publish is the shape of the output: a daily health report
ranking units for maintenance. A run of implausible values is a fault to a forecaster and a real
event to a control engineer, and only the purpose of the analysis settles which.

### 7. Recovering signed power from apparent-power meters

#### The challenge

Of the trial area's 16 primary substations, 8 are metered in apparent power (MVA) rather than real
power (MW), as are two of the three wind farms. An apparent-power meter reports a magnitude with no
direction, so when a solar farm behind the substation exports more than the substation's customers
are drawing, the substation's reading rises instead of going negative: the trace "bounces" off zero,
and a midday export reads as a midday peak. NGED report that one of those 10 apparent-power sites
has shown the bounce on sunny days, with two more suspected. The meter is not faulty; the difficulty
is that the quantity NGED needs forecast is signed net demand, while an apparent-power meter reports
the absolute value of signed net demand — and reports even the absolute value only approximately.

#### What the literature says

**A magnitude-only measurement leaves more than one state of the electricity network consistent with
the reading, a result power-system state estimation has worked with since the 1990s.**
[Abur and Expósito (1997)](https://doi.org/10.1109/59.575721) showed that a measurement set
containing current magnitudes can admit multiple solutions, and
[Ju et al. (2018)](https://doi.org/10.1109/TSG.2017.2709463) carry the result into distribution
networks with the remedy: where branch current-magnitude measurements "are the only ones to make the
branch observable" the solution "is not unique", so a current-magnitude measurement can only sharpen
an estimate that other measurements have already pinned down.

**Two of the three published attempts we found rest on a second measurement of the same power.**
[Bouman et al. (2024)](https://arxiv.org/abs/2405.16164)'s Dutch substations carry the same
limitation as NGED's MVA-metered substations, measuring only the absolute current, and Bouman et al.
recover the sign from a bottom-up load estimate built from smart meters, wherever the substation
meter reads non-negative throughout while the bottom-up estimate goes negative. Western Power
Distribution, NGED's predecessor, set out in the 2017
[Time Series Data Quality](https://smarter.energynetworks.org/projects/nia_wpd_011/) NIA project to
"first detect then assign directions to power flows where absent", and piloted a tool reconciling
summed current at a substation's transformers against summed current along its feeders, flipping a
candidate feeder's direction where the two current sums disagreed by more than a threshold. Time
Series Data Quality also counted the circuits at stake on what is now NGED's network: 204 in the
South West licence area and 326 company-wide "experience reverse flows which are not apparent from
the existing analogue values".

**SSEN's TRANSITION, the third attempt and the closest to NGED's position, uses the meter's own
history together with a model of the generation behind the meter.**
[SSEN's TRANSITION](https://ssen-innovation.co.uk/transition/) met feeders metered in amperes only,
where "the direction of the flow cannot be captured by the analogues", and settled the direction in
three steps in the project's Load Forecasting Solution report: take the reading at face value where
modelled generation is too small to have pushed the flow negative; flip the sign where modelled
generation exceeds the average net demand that meter recorded over 4 years; then flip back wherever
the recovered underlying demand comes out greater than net demand. The report gives no accuracy
figure for the direction step, calling the result "a satisfying initial level of automated
computation".

#### What this means for Flexpectation

**Flexpectation version 1 does not attempt the recovery: version 1 forecasts each series in the unit
its own meter reports, real power where the meter is directional and apparent power where the meter
is not, and flags the affected series to NGED alongside the forecast.** Forecasting the bounced
trace is honest about what was measured, and at a substation whose generation never reverses the
flow the apparent-power trace and the signed trace are identical. What forecasting the bounced trace
cannot do is tell NGED whether a peak forecast at midday represents demand approaching the
substation's import capacity or export approaching the substation's export capacity.

**Flexpectation version 2 puts the meter's behaviour inside the model rather than repairing the
series first.** The differentiable-physics forward model reconstructs a substation's signed net flow
from gross demand, metered generation, and unmetered generation, and compares the *magnitude* of the
reconstruction against the apparent-power reading, so the bounce is predicted rather than removed.
Recovering a signal from the magnitude of a transform of that signal is the phase-retrieval problem,
which [Dong et al. (2023)](https://doi.org/10.1109/MSP.2022.3219240) describe as non-convex because
a signal satisfying the magnitude equation is always one of a family of solutions. An apparent-power
meter takes the magnitude half-hour by half-hour, so nothing in the measurement couples one
half-hour's sign to the next, and the family holds one member for every assignment of signs across
the window rather than two. Dong et al.'s uniqueness results do not rescue the problem either: those
results turn on how far the number of measurements exceeds the number of unknowns, and an
apparent-power meter gives exactly one reading per unknown — the ratio at which Dong et al. call
Fourier phase retrieval "fundamentally ill-posed as we only know amplitudes". Dong et al.'s own
prescription for that regime is the one Flexpectation is following, to "leverage a priori
information on the object", so what Flexpectation adds is not the formulation but the information
that breaks the ambiguity — and that information carries the whole weight. The reconstruction's
solar module has to track irradiance, and a prior holds the direction of flow to persist for hours
rather than flickering from one half-hour to the next.

**Apparent power is the magnitude of real power only near unity power factor, and the approximation
is weakest exactly at the bounce the reconstruction is trying to explain.** As real power passes
through zero, reactive power dominates the measured magnitude, so the apparent-power trace has a
soft floor above zero rather than a clean reflection of the signed flow. The reconstruction will
therefore under-fit the bottom of the bounce, and the failure mode to design against is an optimiser
that explains the soft floor with demand that was never there.

### 8. Disaggregating unmetered solar and wind from a substation's net flow

#### The challenge

Rooftop solar panels and small wind turbines appear only as a dent in a substation's net power flow.
Recovering both the half-hourly output of that unmetered generation and its installed capacity, from
the substation's net flow alone, is what we call *disaggregation*. Disaggregation is a different
task from estimating how much of a *metered* generator's capacity is available today, which is
challenge 3.

Distribution network operators (DNOs) do not know exactly how much capacity is installed. Ofgem's
December 2025
[consultation on asset visibility](https://www.ofgem.gov.uk/sites/default/files/2025-12/Enhancing%20asset%20visibility%20-%20Distribution%20Network%20Operator%20Options%20consultation.pdf)
estimates that distribution network operators "are aware of less than half" of the consumer and
distributed energy resources on their distribution networks, a figure Ofgem attributes to the
Department for Energy Security and Net Zero's engagement with the operators rather than to a
measurement, and which that department's own footnote traces to estimates the operators derived from
other datasets, from sales volumes, and from grants processed. Each DNO's **Embedded Capacity
Register** records generation of 50 kW and above and names the primary substation each site sits
behind, but the capacity recorded is the export limit "permitted as per the connection agreement"
rather than what a site can generate. Below 50 kW the register is silent, and that is where most of
the panels are: of the 22,560 MW of solar photovoltaic capacity installed in GB by the end of July
2026, 8,503 MW — 38% of the total — sits in arrays smaller than 50 kW, spread across 2,058,822 of
the 2,068,186 installations, according to the Department for Energy Security and Net Zero's
[solar deployment statistics](https://www.gov.uk/government/statistics/solar-photovoltaics-deployment).
Other registers exist, but none provides a complete picture. The **Renewable Energy Planning
Database** tracks projects through the planning system and starts at 150 kW, a threshold lowered
from 1 MW only in 2021, so smaller projects that cleared planning before 2021 may be absent. The
**Feed-In Tariff** register closed to new applicants on 1 April 2019. And a domestic array reaches
NGED only when the installer notifies NGED, as installers are required to do. None of these
registers records the panel tilt, the panel azimuth, or the ratio of direct-current to
alternating-current rating.

#### What the literature says

**Splitting generation out of a substation's net flow has been done at GB primary substations, but
in the GB projects that have published a result the generation was either metered or its capacity
read from a register, rather than inferred from the net flow.** Northern Powergrid's
[Artificial Forecasting](https://smarter.energynetworks.org/projects/npg_sif_006-1/) models gross
demand and customer export independently at primary substations, but that customer export is
metered, and the baseline Artificial Forecasting measures its customer-export models against is an
extrapolation from Northern Powergrid's own Distribution Future Energy Scenarios, not a capacity
inferred from the net flow. [SSEN's TRANSITION](https://ssen-innovation.co.uk/transition/) split net
load into demand and generation, forecast the two separately, and recombined them; TRANSITION's
rooftop solar is not metered, but TRANSITION read each installation's capacity from a list of
Feed-In Tariff installations. Flexpectation has no register that would carry it as far, for the
reasons set out under "The challenge" above.

**Flexibility Market Asset Registration, the register being built now that sounds as though it
should close this gap, will record the assets that trade flexibility, which is close to the
complement of the arrays this challenge has to find.** Ofgem [appointed Elexon in
2025](https://www.ofgem.gov.uk/decision/decision-flexibility-market-asset-registration) to deliver
Flexibility Market Asset Registration, digital infrastructure due by the third quarter of 2027 that
will collect, store, and share data on assets participating in flexibility markets, aimed first at
assets under 1 MW. Ofgem's [asset visibility
consultation](https://www.ofgem.gov.uk/sites/default/files/2025-12/Enhancing%20asset%20visibility%20-%20Distribution%20Network%20Operator%20Options%20consultation.pdf)
says the register collects data on assets "when they are first registered into a DSO or NESO
flexibility market" — the market of a distribution system operator, or of the National Energy System
Operator — so a rooftop array that never trades flexibility never enters the register, and the
arrays this challenge has to find are the ones nobody has registered anywhere. Where an asset does
enter a flexibility market that NGED itself runs, NGED is the counterparty and already holds the
data; what Flexibility Market Asset Registration adds there is one standardised record across
markets rather than a generator NGED could not previously see.

**The published benchmarks we found of inferring capacity from the net flow work on individually
metered premises, sit at a voltage level below NGED's, or do not say what aggregation they worked
at, and the GB project doing the same at primary substations has not yet published a result.**
[Gouveia et al. (2026)](https://doi.org/10.1016/j.ijepes.2026.111848) benchmark that inference at
low-voltage substations serving 10 to 100 customers rather than at a primary. UK Power Networks'
[Power Flow to Solar Capacity](https://smarter.energynetworks.org/projects/nia_ukpn0104/) project
(with Open Climate Fix) infers solar photovoltaic capacity behind UK Power Networks' primary
substations. [Kanchana et al. (2026)](https://doi.org/10.1016/j.epsr.2026.113279) separate load,
photovoltaic generation, and energy storage from one aggregated net-load series, and report doing so
"without requiring capital-intensive customer-level metering", which is NGED's position exactly. We
hold the publisher's landing page for Kanchana et al. rather than the full text, and that page names
no customer count, no country, no time resolution, and no comparison method, so how far the reported
errors of 8.14% for load, 5.12% for photovoltaic generation, and 11.51% for storage would carry to a
GB primary substation cannot be judged from what we have read. The same page says a generative
adversarial network fills gaps in the load measurements while a variational autoencoder generates
synthetic photovoltaic profiles, and that "observed net-load profiles are assembled to create
validation scenarios", so whether the mixture being separated is one a meter recorded is a question
the page leaves open.

**The one result we found that separated solar from demand at a real distribution substation,
without being told the installed capacity, used that substation's own reactive power.**
[Kara et al. (2018)](https://doi.org/10.1016/j.segan.2017.11.001) estimate the solar generation
downstream of a substation in Riverside, California, from the substation's active and reactive
power, and report a root-mean-square error of 6% of installed capacity across all sky conditions.
The estimator is given neither the installed capacity nor the panel geometry, but it does need one
input NGED would struggle to supply at most substations: the metered output of a second photovoltaic
plant, four miles away on a different feeder and low-pass filtered, standing in for irradiance.
Regressing the load's active power on the measured reactive power is what makes the separation work:
the simpler alternative, assuming the power factor measured at night holds through the day, is
broken by the solar plant's own reactive power consumption, which the preprint version of Kara et
al. found responsible for about 25% of the overestimation at its peak. Two features of the setup
limit how far Kara et al.'s result carries to Flexpectation. The generation behind that substation
is a single 7.5 MW solar site, "the only generation asset located at this substation", rather than a
fleet of differently-oriented rooftops, and the ground truth comes from a second measurement device
at that site's own point of interconnection. Kara et al. also had to detect and compensate
capacitor-bank switching before the reactive power was usable, and their accuracy was still
improving as the sampling rate rose to one sample every five minutes, which puts NGED's half-hourly
data below the rate at which the errors settled. Kara et al. name the amount of photovoltaic
capacity behind the substation, its volatility, and its spatial spread as factors whose effect on
their method they had not studied — the three respects in which a GB primary substation differs most
from their test case.

**Uncertainty and a multi-day horizon each appear in the disaggregation work we found, but not in
the same forecast.** [Zhang et al. (2022)](https://doi.org/10.1016/j.engappai.2022.104707) attach
uncertainty, disaggregating rooftop solar out of net load at grid supply point and feeder level with
a multi-quantile recurrent neural network scored on reliability and sharpness. NESO's
[embedded wind and solar forecasts](https://www.neso.energy/data-portal/embedded-wind-and-solar-forecasts)
are half-hourly to 14 days ahead, as a single number per half-hour with no uncertainty attached. A
survey of behind-the-meter solar forecasting whose 162 sources reach 2021,
[Erdener et al. (2022)](https://doi.org/10.1016/j.rser.2022.112224), judged that "the literature
explicitly focused on uncertainty quantification within BTM [behind-the-meter] systems is immature",
and recommended probabilistic approaches as the way to represent that uncertainty.

**An uncertainty estimate is useful only if the estimate widens where the answer gets worse, and the
one substation-level disaggregation we found that tested for that widening reports the widening
holding — until the generation pattern is unlike anything in the training data.**
[Yi and Wang (2022)](https://arxiv.org/abs/2207.03490) summarise their two journal papers on
disaggregating behind-the-meter solar at substations, and pose the problem as one of *partial
labels*: for some aggregate measurements the operator knows which load types are present, but never
their individual values. Yi and Wang's Bayesian dictionary-learning estimator reaches a total error
rate of 8.97%, against 20.61% and 37.12% for two methods that need fully labelled training data, and
its error weighted so that the estimates the method is unsure about count less — 0.13 to 0.16 —
comes out far below its unweighted root-mean-square error of 5.19 to 6.20, which Yi and Wang read as
showing that the estimates carrying the largest errors are the ones carrying the largest
uncertainty. Where the test period's solar pattern is unlike any pattern in the training data,
however, Yi and Wang report that the true load may fall outside the 99.7% confidence interval — the
failure mode that matters most to Flexpectation, whose substations will carry generation mixes that
no training substation had. The validation runs on 360 generated training samples covering two
industrial loads and one solar generation, not on measurements from a real substation, and carries
no forecast horizon.

**The survey of behind-the-meter solar forecasting by Erdener et al. tabulates net-load
disaggregation studies that run either at individually metered premises or at a whole balancing
area, with no study at the aggregation level of a primary substation.**
[Erdener et al. (2022)](https://doi.org/10.1016/j.rser.2022.112224) tabulate eight studies that
recover photovoltaic capacity, panel tilt, or panel azimuth by disaggregating net load. Seven work
on individually metered premises — two photovoltaic plants in Switzerland, and customer sets of 40,
100, 183, 197, 300, and 1,300 — and the eighth works on the zone of Independent System Operator New
England that covers the state of Maine. A GB primary substation sits between those two aggregation
levels, at a level Erdener et al.'s table does not cover.

**The smart-meter literature on disaggregating rooftop solar is larger than the substation
literature, but the smart-meter work sits at individual premises rather than at a substation, and
leans on a neighbouring-customer comparison NGED cannot make.**
[Cheung et al. (2023)](https://doi.org/10.1109/TSUSC.2022.3192456) use the consumption patterns of
neighbouring customers known to have no panels, which NGED cannot observe, and Cheung et al. are the
one study we found that varies the aggregation count on measured household data: across 5, 10, and
20 Australian customers per aggregated series, their own method's solar mean absolute scaled error
stays between 1.02 and 1.28 — around the average change between consecutive readings — with solar
mean absolute percentage error of 21 to 25%, and Cheung et al. report that both measures stayed
"almost the same as aggregation level varied". The Kara-derived baseline they re-implemented for
comparison degraded instead, from a mean absolute scaled error of 1.47 at 5 customers to 2.20 at 20,
and from 28% to 43% mean absolute percentage error. Results reported elsewhere at an aggregate level
are usually sums of individually metered households rather than a measurement taken at a real
aggregation point, so the smart-meter literature stops far below the thousands of customers behind a
GB primary substation.

**Whether more customers behind a substation makes the estimate easier or harder is unsettled, and
the one study we found that varies the aggregation count on a simulated feeder does not settle the
question.** [Tang et al. (2024)](https://doi.org/10.1016/j.segan.2024.101396) estimate installed
photovoltaic capacity from 24-hour net-load curves for feeders of 20 to 80 London households, and
report "a general trend of increasing RMSE [root-mean-square error] values as the number of
households increases". The rising root-mean-square error is weaker evidence than the trend first
appears: the error is in kilowatts against a total capacity that itself rises with the household
count, the percentage error moves the other way, and the trend reverses sharply between 70 and 80
households. The load and the household count are real, but the solar is simulated at three azimuths,
45°, 0°, and −45°, all of them southerly, so the study has none of the north- and east-west-facing
roofs a real street would carry.

#### What this means for Flexpectation

**UK Power Networks' "Power Flow to Solar Capacity" project is highly relevant to Flexpectation: the
project works on the same kind of GB data, and Open Climate Fix is a partner in both projects.**
[UK Power Networks' Power Flow to Solar Capacity](https://smarter.energynetworks.org/projects/nia_ukpn0104/)
(2024 to 2026, £0.4 million) infers the capacity of unmetered solar sitting behind each primary
substation from half-hourly substation load and weather, then forecasts that generation. Open
Climate Fix is a partner in both Power Flow to Solar Capacity and Flexpectation, so what Power Flow
to Solar Capacity found about inferring solar capacity from GB primary-substation data reaches
Flexpectation directly rather than only through what has been published.

**The nearest published method we found splits unmetered wind and solar out of substation
measurements, but needs each site's installed capacity.**
[Teng et al. (2023)](https://doi.org/10.1016/j.rser.2023.113662) train on 10 Dutch substations that
carry complete renewable metering, then predict solar and wind power separately at substations with
none, from the substation's measured total load, weather, geospatial position, and each site's known
renewable capacity, at 15-minute resolution — a root-mean-square error of 0.07 against 0.70 for a
default transfer-learning model, on a min-max-scaled target. The 0.07 reads as 7% only if the scale
runs from 0 to 1, and Teng et al. do not say what the scaling divides by, so the figure does not
transfer to another dataset. The 0.07 should not be read as achievable here: Teng et al. are told
each site's capacity, whereas inferring the capacity is half of what Flexpectation plans to achieve.

**Inferring the capacity from the net flow alone has been measured, at a smaller scale than
NGED's.** [Gouveia et al. (2026)](https://doi.org/10.1016/j.ijepes.2026.111848) benchmark
data-driven against model-based estimators of the photovoltaic capacity installed behind a
low-voltage substation, working from the net load and irradiance series. Gouveia et al.'s
substations serve 10 to 100 customers, against the thousands behind a GB primary.

**Gouveia et al.'s two transferable results are that data-driven estimators beat model-based
estimators on noisy data, and that a model trained in one country held under 5% mean absolute
percentage error in two others.** The data-driven estimators matched the model-based estimators on
clean data, and beat the model-based estimators clearly on noisy data — the condition NGED's
telemetry is in. And models trained on a Belgian dataset, then applied unseen to American and
Australian datasets with only approximate irradiance, stayed under 5% mean absolute percentage error
once the linear models were regularised. What Gouveia et al.'s estimators produce is a capacity
figure rather than a forecast, so what Flexpectation adds is putting the capacity estimate inside a
probabilistic multi-day forecast, and disaggregating the full shape of the unmetered generation.

**GB already has an operational forecast of unmetered generation, but only at national scale and
without uncertainty.** NESO's
[embedded wind and solar forecasts](https://www.neso.energy/data-portal/embedded-wind-and-solar-forecasts),
described under "What the literature says" above, match the resolution and horizon Flexpectation is
specified to deliver, but cover GB as one region rather than substation by substation.

**Observational cosmology and systems biology have separated superposed signals for decades, and
both give the same warning: a small residual against the measured total is not evidence that the
components were separated correctly.** Fitting a sum of physically parameterised components to one
measurement is routine in observational cosmology, where the technique is called **component
separation**. [Hensley and Bull (2018)](https://doi.org/10.3847/1538-4357/aaa489) show that giving
the *nuisance* component too simple a model biases the component of interest, and that the fit does
not announce the bias: "models that are strongly biased but still yield low χ² values are the most
dangerous". Two consequences follow for Flexpectation. Effort spent making the demand model richer
is justified on separation grounds even where it does not improve the fit to the substation's net
flow, and the diagnostic to watch is the joint distribution over the components rather than the
residual. Working in systems biology,
[Wieland et al. (2021)](https://doi.org/10.1016/j.coisb.2021.03.005) add the matching warning about
uncertainty: confidence intervals read off the curvature at the optimum, which a differentiable
model gives cheaply, are "insensitive to practical non-identifiabilities" and can look reassuringly
finite for a parameter the data do not constrain at all.

**Fitting a differentiable physical forward model to measurements is routine in exploration
geophysics, and that field reports that the order in which the fit admits fine detail decides
whether the fit converges at all.** Full-waveform inversion recovers the properties of the rock
beneath a seismic survey — chiefly the speed at which sound travels through each point of the
subsurface — by simulating the seismograms those properties would produce and adjusting the
properties until the simulation matches the recording, which is the procedure Flexpectation version
2 applies to a substation's net flow. [Virieux and Operto (2009)](https://doi.org/10.1190/1.3238367)
report a failure mode the field calls cycle skipping: because a seismogram oscillates, a starting
model that mis-predicts an arrival by more than half a period leads the optimiser to match the wrong
cycle, and "the so-called cycle-skipping artifacts will lead to convergence toward a local minimum".
The remedy Virieux and Operto report as standard practice is a multi-scale schedule that inverts the
low frequencies first, "because low frequencies are less sensitive to cycle-skipping artifacts",
then admits successively higher frequencies, each stage starting from the model the previous stage
produced. The arithmetic relating frequency to recovered detail belongs to wavefields and does not
carry to a half-hourly power series, but the local-minimum mechanism does: a substation's net flow
is periodic on a daily cycle, so fitting the slowest-varying structure of each component before
admitting half-hourly detail is the transferable precaution.

**Whether two simultaneously fitted components can be told apart is a property of the measurements
rather than of the optimiser, and exploration geophysics tests for that coupling before fitting and
orders the fit around the answer.** [Virieux and Operto (2009)](https://doi.org/10.1190/1.3238367)
report that adding a second class of physical parameter makes the problem more ill-posed, because
"more degrees of freedom are considered in the parameterization" and because "the sensitivity of the
inversion can change significantly from one parameter class to the next", and that "different
parameter classes can be more or less coupled as a function of the aperture angle" — the angle at
which a source and a receiver view the same point in the rock. Coupling of that kind "can be
assessed by plotting the radiation pattern of each parameter class", so the field tests separability
in advance rather than discovering it afterwards. Where the speed of sound and the rock's density
carry the same signature at short apertures, "these two parameters are difficult to reconstruct from
short-offset data", and Virieux and Operto cite a study concluding that the speed of sound and the
rock's absorption "cannot be imaged simultaneously from short-aperture data". The response Virieux
and Operto report is to order the parameter classes rather than run the joint fit for longer: one
study they cite recommends fitting the speed of sound, denoted VP, alone first and fitting the speed
of sound with the absorption jointly second, "because the reliability of the attenuation
reconstruction strongly depends on the accuracy of the starting VP model". One limit rides along: a
seismic survey chooses where to put its sources and its receivers, and the long offsets and wide
apertures that break the coupling are a design choice, whereas Flexpectation takes the telemetry
NGED already collects, so what transfers is the test for coupling and the fitting order rather than
the survey design.

**Hyperspectral unmixing leans for identifiability on each component being observed alone somewhere,
and Flexpectation can test whether that condition holds before fitting a forward model.**
Hyperspectral unmixing, which splits one mixed image pixel into the spectra of the materials
composing it, calls the condition the pure-pixel assumption: for each component there is at least
one observation containing only that component.
[Bioucas-Dias et al. (2012)](https://doi.org/10.1109/JSTARS.2012.2194696) set out a weaker
sufficient condition too, and show what happens when neither holds — on a highly mixed data set with
no observations near the extremes, the fitted simplex comes out smaller than the true one, so the
recovered components are biased rather than merely uncertain. Flexpectation's pure observations are
the half-hours when nature switches one component off — night for solar, calm hours for wind, and
the substations carrying no embedded generation at all — so whether a given substation has those
half-hours is a question the telemetry can answer on its own.

### 9. Disaggregating other distributed energy resources: heat pumps, electric-vehicle chargers, and batteries

#### The challenge

Heat pumps, electric-vehicle chargers, and price-sensitive domestic batteries change the shape of a
substation's load in ways a model trained on history cannot anticipate, because the number installed
behind any given substation is growing quickly. A Flexpectation stretch goal is to disaggregate and
forecast heat pumps, electric-vehicle (EV) chargers, and batteries separately rather than leaving
them inside net demand.

The number of each installed grows fast enough to matter within Flexpectation's own lifetime. Every
figure below is the Holistic Transition pathway of NESO's
[Future Energy Scenarios (FES)](https://www.neso.energy/publications/future-energy-scenarios-fes):

| What | 2024 | 2030 |
|---|---|---|
| Battery-electric cars on the road in GB | 1.4 million | 8.2 million |
| Heat pump stock in GB | 361,000 | 2.3 million |
| Battery storage below 1 MW in GB | 191 MW | 975 MW |
| The same class in the 2024 scenarios, allocated to the four grid supply point groups NGED serves | 49 MW  | 308 MW  |

Electric cars and heat pumps each rise roughly sixfold over those six years. "Below 1 MW" is NESO's
own class boundary, and NESO defines the class as generation and storage under 1 MW that therefore
"includes some larger commercial installations", so the class is wider than the batteries that fall
below the Embedded Capacity Register's 50 kW floor. How much of the 191 MW sits below that floor is
a question the FES scenarios cannot answer. Ofgem's [asset visibility
consultation](https://www.ofgem.gov.uk/sites/default/files/2025-12/Enhancing%20asset%20visibility%20-%20Distribution%20Network%20Operator%20Options%20consultation.pdf)
says the Microgeneration Certification Scheme's (MCS) installation database certifies battery
storage up to 50 kW, and the Department for Energy Security and Net Zero (DESNZ) publishes [counts
drawn from the MCS
database](https://www.gov.uk/government/statistics/mcs-certified-domestic-battery-installation-statistics):
73,987 domestic retrofit battery installations between September 2023, when the series starts, and
March 2026, rising from 24,242 in the 2024/25 financial year to 44,033 in 2025/26, with the 72,459
installations that passed outlier testing holding 666,880 kWh between them. The DESNZ figures cannot
be subtracted from the FES figure of 191 MW, because the DESNZ figures measure stored energy rather
than power, and count domestic retrofits rather than every installation below 50 kW, and accumulate
from September 2023 rather than reporting a stock. The DESNZ series carries no projection, so the
2030 figure has no counterpart outside the FES scenarios.

Scaling the GB figures to NGED is only approximate. NESO's [regional breakdown of the 2024
scenarios](https://www.neso.energy/data-portal/regional-breakdown-fes-data-electricity) allocates
36% of GB's sub-1 MW battery capacity to the four grid supply point groups NGED serves, and holds
that 36% constant across every forecast year, so the regional number is an allocation of the GB
total rather than a forecast of NGED's own area. NGED serves nearly 8 million customers, a little
over a quarter of the GB total.

#### What the literature says

**Heat pumps, EV chargers, and batteries have the fewest directly relevant papers we could find of
the nine challenges.** The one study we found that measures EV charger forecast skill against
aggregation, [Ostermann and Haug (2024)](https://doi.org/10.1186/s42162-024-00319-1), forecasts
charging demand over a 24-hour horizon at 15-minute resolution from 350,000 charging processes at
more than 500 locations across Germany, and repeats the exercise at four aggregation levels: the
individual site, the postal code, the transmission system operator's zone, and the whole portfolio.
Eight machine-learning and deep-learning models are set against a naive benchmark that predicts the
average of the same quarter-hour on the same weekday. Of the five individual sites, only the one
with 145 charge points beat that benchmark by a clear margin; at the site with 8 charge points some
models beat it, and at the sites with 3, 4, and 14 charge points none did. We found no papers
discussing whether domestic batteries responding to a common price signal average out as more are
added. On the topic of heat pumps, we found no measurement of heat-pump diversity in cold weather,

**A targeted literature search for disaggregating heat pumps, chargers, and batteries from
substation measurements found the work split by asset, and found "substation level" used for
aggregations far smaller than a GB primary substation.**
[Gao et al. (2024)](https://doi.org/10.1016/j.apenergy.2024.123361) disaggregate thermostatically
controlled loads — air conditioners, heating and ventilation units, and furnaces — from an
aggregated residential load by contrastive sequence-to-point learning, and generalise the same model
to photovoltaic generation and electric-vehicle charging. The 8.78% Gao et al. report is a mean
absolute percentage error between the estimated thermostatically controlled load and the metered
thermostatically controlled load, so the denominator is the appliance load being recovered rather
than the aggregate the load was pulled out of. Gao et al. give 8.78% as a best case for the
bi-directional model structure and 11.26% as a best case for the unidirectional structure, and name
no baseline either structure is measured against. Gao et al.'s aggregate is the sum of the Pecan
Street dataset's 25 individually metered homes in Austin, Texas, and 25 in New York — two orders of
magnitude below the thousands of customers behind a GB primary substation, and a sum of household
meters rather than a measurement taken at a real substation.
[Ebrahimi et al. (2022)](https://doi.org/10.1109/TII.2021.3118101) split electric-vehicle charging
out of a feeder-head load hour by hour, and needed the charging power and stored energy of 19
vehicles metered live, alongside the hourly energy price and the ambient temperature, to do it. The
feeder-head load Ebrahimi et al. worked on was assembled rather than measured: hourly demand
published by Independent System Operator New England for its Connecticut zone, peaking at about
1,455 kW, added to a simulated charging schedule built from the plug-in and plug-out records of 201
Nissan LEAFs in the My Electric Avenue trial.
[Wang et al. (2022)](https://doi.org/10.1109/TIA.2022.3144244) separate behind-the-meter
photovoltaic generation and battery charging jointly by contextually supervised source separation —
the method family Kara et al. extended for solar under challenge 8, which suggests the battery
problem and the solar problem are the same problem with another component added. Of the Gao et al.
and Wang et al. papers we hold Gao et al.'s abstract, introduction, and dataset description from the
publisher's landing page, and Wang et al.'s abstract, so the Wang et al. citation carries no more
weight than the existence of the work. Both full texts are closed.

**The two heat-pump disaggregation papers we obtained separate a heat pump from the total load of
the single premises the heat pump sits in, and the widest aggregate either paper reports is five
households.** [Brudermueller et al. (2023)](https://doi.org/10.1145/3600100.3623731) estimated the
heat pump's own 15-minute energy in 363 Swiss single-family houses, each fitted with a second meter
on the heat pump and none fitted with photovoltaics, explaining 83% of the variance in that second
meter's readings across households held out of training, against 63% for the better of the two
published baseline algorithms in the comparison.
[Gisiger et al. (2026)](https://doi.org/10.1016/j.egyai.2026.100691) ran the same task over 7,021
Swiss premises with heat pumps through one heating season of 15-minute readings, and summed the
estimated heat-pump load of five of those premises, drawn at random from the dataset and treated as
sharing one transformer, to within 6% of the metered total over an evening peak of 17:00 to 21:00.
Gisiger et al.'s error, normalised by the mean metered heat-pump load, rose from 0.69 on the Swiss
data the model was trained on to 0.78 on Brudermueller et al.'s separate Swiss dataset, and to 1.24
on a German dataset of single-family houses with heat pumps, which Gisiger et al. attribute to
differences in heat pump types, building stock, occupancy patterns, and data collection methods —
measured evidence that a heat-pump model does not survive a change of dataset unaltered, with a
change of country degrading it further still.

#### What this means for Flexpectation

**For Flexpectation version 1: heat pumps, chargers, and batteries stay inside net demand rather
than being forecast separately.** In the one measurement we found, the only site size that clearly
beat a naive benchmark 24 hours ahead was 145 charge points. Forecast uncertainty grows with lead
time, so over the 14 days NGED needs, a site would probably have to be larger than 145 charge points
before a separate charger forecast was worth making.

**Compared to the literature we found, Flexpectation version 2 plans to invert which half of the
problem is learned: the disaggregation methods in this challenge's literature learn each resource's
signature from premises where that resource is separately metered, whereas we plan to use a
differentiable physical model of each distributed energy resource where we write the physics in code
and learn only the parameters.** Gao et al. train on the Pecan Street homes' individually metered
appliances, Brudermueller et al. on 363 Swiss houses each fitted with a second meter on the heat
pump, and Gisiger et al. on 7,021 Swiss premises with heat pumps, while Ebrahimi et al. need the
charging power and stored energy of 19 vehicles metered live. Exogenous inputs already appear in
that work — Ebrahimi et al. take the hourly energy price and the ambient temperature, and Gisiger et
al. find detection easier in colder weeks — but each paper uses those inputs as features feeding a
mapping learned from metered examples, so what the model knows about a heat pump is what heat pumps
looked like in the training set. A differentiable physical model states the relationship between
outdoor temperature and heat-pump electrical demand as an equation instead, and fits the building's
parameters. The measured argument for that inversion sits in this challenge's own literature:
Gisiger et al.'s error, normalised by the mean metered heat-pump load, rose from 0.69 on the data
the model was trained on to 1.24 on a German dataset, which Gisiger et al. attribute to differences
in heat pump types, building stock, occupancy patterns, and data collection methods — differences
that change a learned signature but not the equations behind that signature. The inversion also
supplies the separability an aggregate needs, because each resource answers to a different driver:
outdoor temperature moves the heat pumps, irradiance and panel orientation move the solar, and price
moves the batteries. Whether a given substation's mixture is separable in practice is a property of
the measurements rather than of the optimiser, which is the test challenge 8 borrows from
exploration geophysics and from hyperspectral unmixing. Two limits ride along: the thermal physics
of the thousands of premises behind a primary substation is not one building's physics repeated, and
a differentiable physical model removes the need for submetered training examples at every
substation without removing the need for something to check the answer against.

**The spiky, synchronised charging that makes electric-vehicle load hard to *forecast* is what makes
that load easy to *detect* in aggregate, while heat pumps are hard to detect at all.** Northern
Powergrid's [smart-meter detection trial](https://smarter.energynetworks.org/projects/npg_nia_-49/),
on 1,500 monitored premises, found that "EV [electric vehicle] identification at premises level was
found to be relatively straightforward", though "a lack of ground truth, such as registered charging
points, precluded formal validation", and that "aggregation does mask some signals, although EV
usage is still clearly identifiable at feeder and substation level". The same trial found that "the
detection of ASHP [air-source heat pumps] is frustrated by the low levels of adoption (<1% of
premises) and differences in operation (low-slow vs high-fast)".
[Gisiger et al. (2026)](https://doi.org/10.1016/j.egyai.2026.100691) detected a heat pump at a
single premises from one week of 15-minute readings with a precision of 0.896 by a rule counting
sharp rises in power and 0.953 by a convolutional neural network, and found detection easier in
colder weeks, when heat pumps run more. Gisiger et al. assembled that evaluation set around premises
known to have heat pumps without reporting how many premises without one the set held, so the
precision does not carry to the fewer than one premises in a hundred Northern Powergrid was
searching.

## Evaluating the performance of power forecasts

The nine challenges above need three different kinds of evaluation — scoring a forecast, checking an
estimate of a quantity NGED does not meter, and scoring the detection of a rare event — and the
literature is far stronger on scoring forecasts than on the other two kinds of evaluation. Scoring a
forecast has settled practice Flexpectation can adopt. Checking an estimate of a quantity NGED does
not meter — the effective capacity of a metered generator, the half-hourly output of unmetered
solar, and the direction of flow behind an apparent-power meter — has six possible substitutes for
ground truth, of which the papers we read use four. Scoring the detection of a rare event, such as a
switching event or a metering fault, has good academic practice, and none of the GB projects we
checked published a number to compare against.

**Mean absolute error rewards a flat forecast that would be of little use for either flexibility
procurement or curtailment decisions, so a peak-aware score belongs alongside a proper score rather
than instead of one.** A forecast that predicts the right peak at the wrong time is penalised twice
by mean absolute error — once for the peak it predicted that did not happen, and once for the peak
that did happen and the forecast missed. A flat, featureless forecast avoids both penalties.
Meteorologists named that effect the "double penalty", and the meteorologists' conclusion transfers
to substation forecasting: a score that forgives a peak predicted an hour late is generally no
longer a *proper scoring rule* — a score whose expected value is optimised when the forecaster
publishes the predictive distribution the forecaster actually believes, so that no hedged forecast,
flatter or later-peaking, scores better on average. The same argument runs at the other end of the
distribution: the half-hours of deepest export are the ones curtailment turns on, and a flat
forecast hides the deepest export half-hours too.

**Two teams independently concluded that mean absolute error was the wrong measure for peaks.**
[Pinheiro et al. (2023)](https://doi.org/10.1016/j.apenergy.2022.120493) adopted a peak-aware error
measure for exactly this reason, and
[Artificial Forecasting](https://smarter.energynetworks.org/projects/npg_sif_006-1/) built a metric
over the top 10% of demand values, made that metric its primary measure for comparing models, and
reported the metric both against actual demand and normalised to transformer rating.

**A forecast can state its own uncertainty badly without any accuracy score revealing the fault.**
[Kaas et al. (2026)](https://arxiv.org/abs/2607.01966) scored models on 200 German low-voltage
feeders with an overload-decision metric evaluated at each model's 95th percentile for consumer
peaks and each model's 5th percentile for producer peaks. The two models that came first and second
on consumer peaks in the quantile variant of the overload-decision metric — Chronos-Bolt, a
time-series foundation model, and a weekly-naive baseline — turned out to have 90% ranges containing
the true value only 62% and 58% of the time across the series as a whole, and 43% and 49% of the
time at the consumer peaks themselves. In [Kaas et al. (2026)](https://arxiv.org/abs/2607.01966)'s
results, a model that understates its uncertainty raises fewer false alarms, so it scores well on a
threshold-crossing test while being exactly the model an operator should not trust near a capacity
limit.

**A cross-validation fold shorter than a year cannot show whether a model handles both ends of the
year, which is one length rule worth adopting outright.**
[Pinheiro et al. (2023)](https://doi.org/10.1016/j.apenergy.2022.120493) held out the whole of 2019
and note that "one year is the minimum acceptable to test a forecasting model whose target value
shows annual seasonality". Substation load shows exactly that seasonality, so any cross-validation
fold — one train-then-test slice of the history — shorter than a year cannot tell us whether a model
handles both ends of the year, and NGED needs both: winter is when NGED buys flexibility, and
summer, when embedded solar output is highest against the lowest demand, is when export constraints
bind and generators are curtailed.

**None of the papers we read addresses the leakage a frequently reissued forecast creates, and
Flexpectation reissues its forecast often enough for the leakage to matter.** When a forecast
covering 14 days is reissued every 6 hours, every target half-hour is covered by 56 separate
forecasts. The literature describes two traps: If we were to count the 56 forecasts as independent,
then a significance test will report a confidence the data does not support. If we were to let a
target half-hour fall on both sides of a train-test boundary, then the test set is contaminated
outright. Flexpectation will treat the leakage as an open methodological question rather than a
solved one.

**We have no ground truth for the effective capacity of a metered generator, or an unmetered solar
output.** The papers we read use four substitutes for truth, each of which fails differently. Two
further substitutes appear in none of the papers. The four in use are to hold out sites that are
metered; to inject a change into real data and see whether the method recovers it; to compare
against an independent tool rather than against truth; and to measure whether the estimate improves
the forecast it was built to improve. The two that appear in none of them are to check an estimate
against a physical model, and to use a substation where every feeder and every embedded generator is
metered, purely as validation. No one substitute for ground truth is trustworthy alone, so the best
approach may be to run *multiple* proxy tests report where they disagree. Each test fails in a
different way, so an estimate that survives multiple tests is better supported than an estimate from
the single best substitute.

## Leaderboards of machine learning results

**Flexpectation is building a leaderboard, not a public ML competition, and the distinction changes
which published lessons apply.** Our leaderboards carry our own experiments, one per class of time
series. For example, solar farms, wind farms, batteries, and the demand at primary substations each
get their own leaderboard. The leaderboards are public to view and reproducible, but we are not
inviting other teams to submit entries. Anyone who wants to benchmark against us can rerun the setup
for themselves. Not inviting outside entries means the literature's lessons about attracting
entrants, prize pots, and qualifying rounds do not apply to us, while the lessons about protocol —
what makes a comparison trustworthy — apply with more force, because rival entrants give a
competition some of its integrity by wanting to catch each other out.

**Energy forecasting has run competitions on common data for over a decade, and only the second
track of GEFCom2017 forecast at anything like the level NGED acts on.** The last row of the table
below describes what Flexpectation is building. The two columns that decide whether a precedent
exists are the aggregation level and whether the leaderboard is still open.

| Leaderboard | What entrants forecast | Aggregation level, set against a primary substation | Take-up | Standing or closed |
|---|---|---|---|---|
| Global Energy Forecasting Competitions 2012, 2014, and 2017 ([Hong et al. (2020)](https://doi.org/10.1109/OAJPE.2020.3029979)) | Hierarchical load, price, wind, and solar, with the data published alongside the papers introducing each competition | Varies, up to national | Hundreds of contestants from more than 60 countries | Closed |
| The second track of GEFCom2017 ([Hyndman (2020)](https://doi.org/10.1016/j.ijforecast.2019.03.015)) | Probabilistic load | 183 delivery-point meters of a US utility — the closest of the leaderboards in this table to a distribution network's aggregation level | 177 entrants across both tracks | Closed |
| BigDEAL Challenge 2022 ([Shukla and Hong (2024)](https://doi.org/10.1049/stg2.12162)) | The timing of peak demand rather than its size; the final match asked for the magnitude, timing, and shape of daily peak load | Three neighbouring local distribution companies — whole utilities, well above a primary substation | 78 teams from 27 countries | Closed |
| HEFTCom ([Browell et al. (2026)](https://doi.org/10.1016/j.ijforecast.2025.10.005)) | The combined day-ahead output of one GB wind-and-solar portfolio | A single 3.6 GW portfolio: the 1.2 GW Hornsea 1 offshore wind farm plus a regional solar aggregate — the generation mix closest to NGED's, though at portfolio rather than substation level | Over 170 teams registered, 66 submitted, 24 completed | Closed; the competition period was 3 months |
| Three competitions NGED funded with Energy Systems Catapult ([McSweeney et al. (2023)](https://doi.org/10.1109/ISGTEUROPE56780.2023.10407541)) | 1-minute peaks inside half-hourly averages; the daily peak a hidden population of electric-vehicle chargers added; and missing values. None was a load forecast | NGED's own grid supply point, bulk supply points, and primary-substation feeders | 37 teams, over 2,500 submissions | Closed, though the pages and data are still readable on CodaLab |
| WindAI ([Authen et al. (2026)](https://doi.org/10.5617/nmi.13106)) | Hourly wind power for the whole of a target day two days ahead, submitted daily against an outturn that had not yet happened | Four bidding zones of a transmission network — regions far above a primary substation | 9 teams carry an average score in the competition summary | Closed; the live evaluation ran over 10 working days in autumn 2025 |
| Energy-Arena ([Kleinebrahm et al. (2026)](https://arxiv.org/abs/2604.24705)) | The paper describes deterministic day-ahead tasks; the running platform today carries 24 challenges across prices, load, wind, and solar — 8 scored as point forecasts, 8 as quantiles, and 8 as ensembles | Not a distribution network | Not stated in what we read | Standing |
| TS-Arena ([Meyer et al. (2026)](https://arxiv.org/abs/2512.20761)) | 186 live energy series | Not a distribution network | 13 foundation models and 3 statistical baselines run by the platform team, plus outside entries | Standing |
| Predico ([Elia Group](https://innovation.eliagroup.eu/en/projects/predico-collaborative-forecasting-platform)) | Quarter-hourly probabilistic generation: Belgian solar out to 10 days ahead, and the German wind and solar markets 50Hertz runs day-ahead | National generation totals of two transmission networks | Forecasters join by application; the number taking part is not published | Standing |
| **Flexpectation's leaderboards** | Net demand at substations, and output at metered generators | One board per class of time series | Public to view and reproducible; outside entries not invited | Standing |

**We found no example of a standing leaderboard for substation forecasting** — one that keeps
accepting entries after its competition closes. Two of the three competitions NGED funded sat at
exactly the levels NGED forecasts, which is why the gap is scoped to forecasting rather than to the
voltage level. Flexpectation's leaderboards are meant to fill that gap, though we would be glad to
be pointed at a counter-example.

**WindAI is the closest of these competitions to challenge 3's problem of a generator whose capacity
keeps changing, because robustness to that change was a scored criterion rather than an
afterthought.** Statnett, Norway's transmission system operator, asked entrants for the hourly wind
power of each of four Norwegian bidding zones two days ahead, and [Authen et al.
(2026)](https://doi.org/10.5617/nmi.13106) report a weighted assessment giving 65% to accuracy, 20%
to trustworthiness and explainability, 10% to implementation and presentation, and 5% to "robustness
to changes in installed wind power capacity, evolving weather patterns, long-term climate
variability". The two highest-placed teams both predicted capacity factor rather than absolute
production, which Authen et al. record as a way "to account for maintenance events and future
capacity expansions", and one team given an honourable mention fitted a physical power curve for
each wind park under sequential Bayesian updating to absorb "capacity changes or the commissioning
of new wind parks". Two further results transfer. The top three entries all used gradient-boosted
decision trees, and Authen et al. conclude that the more complex deep-learning architectures'
"additional complexity did not translate into superior performance". And the placings did not follow
the accuracy order — WindSight recorded a lower average root mean square error than Knowit, 216.22
MW against 217.57 MW, and Knowit still took second place — which is what the other 35% of the
assessment is for.

**Predico is a standing leaderboard that pays its entrants, which is the one mechanism
Flexpectation's leaderboards deliberately do without.** Elia Group describes
[Predico](https://innovation.eliagroup.eu/en/projects/predico-collaborative-forecasting-platform) as
"a collaborative forecasting market platform enabling entities with common interests to procure and
sell forecasts", where buyers receive "skill-weighted aggregate market forecasts" and forecasters
are remunerated on performance: the Belgian solar market carries €7,000 a month, split by accuracy
rank and information contribution, with the best forecasters earning about €1,500 to €2,000 a month.
Forecasts are quarter-hourly and probabilistic, given as the 10th, 50th, and 90th percentiles, out
to 10 days ahead for Belgian solar generation and day-ahead for the German wind and solar markets
50Hertz runs. The [platform documentation](https://predico-elia.inesctec.pt/docs/) scores the median
submission by root mean square error and the 10th-to-90th percentile pair by the mean Winkler
interval, and ranks forecasters monthly. Predico forecasts the national generation totals of two
transmission networks rather than anything on a distribution network, and Elia Group describes the
platform as a proof of concept in which participants cannot yet create their own markets, so what
Predico offers Flexpectation is a worked example of a standing, publicly ranked board rather than a
precedent at NGED's aggregation level.

**The mechanism that makes a leaderboard trustworthy is time, not policing.** The central idea of
[Meyer et al. (2026)](https://arxiv.org/abs/2512.20761)'s TS-Arena is that a forecast is submitted
before the outturn it will be scored against physically exists, which "makes test-set contamination
impossible by design". HEFTCom made the same argument from experience: because the competition ran
on the real, unknown future, "data leakage, accidental or deliberate, was impossible". A half-hourly
forecasting service meets that condition easily: every day supplies 48 fresh evaluation points that
can never be reused, and the condition that the answer did not exist when the model was frozen holds
automatically.

**The specific way a single team fools itself is not fabrication but running the baseline badly.**
[Kleinebrahm et al. (2026)](https://arxiv.org/abs/2604.24705) put it as a general problem with
published comparisons: competing methods "are not always implemented or optimized with equal care",
so reported differences "may reflect differences in implementation quality rather than inherent
methodological advantages". [Hong et al. (2020)](https://doi.org/10.1109/OAJPE.2020.3029979) put it
more bluntly, that "sometimes the parameters are manipulated, so that the competing models are being
dominated by the proposed ones". This is a large part of the reason why, in Flexpectation v1, we are
putting effort into *optimising* our XGBoost forecasts before trying more novel approaches.

**Carry two baselines, one below the achievable skill and one at it, rather than a single
baseline.** [Doubleday et al. (2020)](https://doi.org/10.1016/j.solener.2020.05.051) distinguish the
two jobs a benchmark does: a yardstick, which need not be a good forecast, and what they call a
point on the yardstick — a target for a new method to beat, which "should be close to the state of
the art". They recommend carrying both, so that a new method can be positioned between the two
baselines rather than declared better than a single baseline. Flexpectation's leaderboards carry
both kinds of baseline: persistence (tomorrow resembles a comparable recent day) and climatology
(tomorrow resembles the historical average for the time of year) as the naive yardstick, and NGED's
incumbent method as the point on the yardstick a new model has to reach.

**Flexpectation's leaderboard today reuses one fold for both model selection and the published
result, so the winner's reported skill is optimistically biased.** The fold that Flexpectation
currently reports serves as both the model-selection set and the reported result, so every
hyperparameter choice and feature ablation is adjudicated on the same 12 months the leaderboard
publishes. With hundreds of experiments planned, the winner's reported skill will be optimistically
biased. The structural fix is a final-test window that no model selection is allowed to touch, and
that final-test window is scheduled. Until the final-test window lands, three limits hold:
leaderboard numbers are selection metrics rather than estimates of future skill, differences smaller
than fold-level noise should not drive decisions, and the number of experiments run against a fold
is itself a statistic worth publishing beside the fold's results.

**Rankings travel better than absolute numbers do.** Where a benchmark has enough data behind it,
the ordering of models survives a change of test set even when the accuracy level does not, and the
survival of the ordering decides what a leaderboard should report as its headline.
[Fildes (2020)](https://doi.org/10.1016/j.ijforecast.2019.04.012), reviewing the M4 competition,
compared its daily micro series against a real retail forecasting problem and found the same method
scoring 1.665% on the M4 daily micro series and 11.1% on the retail problem. Fildes's conclusion
points the same way Flexpectation is going: "each organization needs to organize its own forecasting
competition for its own forecasting problems, and should not rely on even large benchmark data
sets", with the published competition useful for narrowing "the pool of methods to be considered"
rather than for predicting your own error. So a leaderboard should lead with ranks and with margins
over a stated baseline, and treat an absolute skill number as valid only on the distribution it was
measured on.

**A finite evaluation window can rank the wrong model first, and several months is not obviously
enough.** [Messner et al. (2020)](https://doi.org/10.1002/we.2497)'s conclusion is the sharpest
warning we found about reading a leaderboard: "evaluation results based on a finite data set are
always subject to some degree of uncertainty and the best ranked forecast does not necessarily have
to be the truly best one. Depending on the actual setup, e.g., in a benchmarking exercise to hire a
forecaster, it should be remembered that even periods of several months may still yield uncertainty
in terms of who the best forecaster truly is." HEFTCom's own competition period was 3 months.

**A leaderboard with no outside entrants cannot support two kinds of claim the competitions above
can, and Flexpectation should not make them.** The Critical Assessment of Structure Prediction
(CASP) competition's 14-year plateau in one of its scored categories ([Kryshtafovych et al.
(2021)](https://doi.org/10.1002/prot.26237)) is evidence about the difficulty of protein structure
prediction only because dozens of groups were attacking the problem independently. A plateau on our
leaderboard would be ambiguous between a hard problem and a team that did not think of the right
idea. The first M-competition's conclusions about whole classes of method — that statistically
sophisticated methods do not typically forecast more accurately than simpler ones, which
the M3 competition did not go on to support, and that a combination of several methods forecasts
more accurately, on average, than the individual methods going into the combination ([Hyndman
(2020)](https://doi.org/10.1016/j.ijforecast.2019.03.015)) — describe what many independent people
chose to try, and no single team's leaderboard can support a conclusion about a whole class of
method. What our leaderboard can do is narrower and still worth having: show which approaches beat a
stated baseline on NGED's own data, under one protocol, with the forecasts, the metric definitions,
and the code published so that anyone can check the arithmetic or rerun the comparison themselves.

## Three published results that point against this project's plan

Three results in this literature point against Flexpectation's plan, and we intend to test all three
rather than avoid them.

### Finer-grained weather data has not always improved the forecast

[Browell and Fasiolo (2021)](https://arxiv.org/abs/2103.10335) forecast day-ahead net load — demand
minus embedded generation — half-hourly for each of GB's 14 grid supply point groups over 2014 to
2018, from the European Centre for Medium-Range Weather Forecasts' high-resolution run issued at
midnight and available around 06:00 UTC. The model Browell and Fasiolo were trying to improve
already used the weather across the whole of each region: wind speed and solar irradiance averaged
over the numerical weather prediction cells covering the region, alongside a temperature forecast
taken from the single cell of highest population density. What Browell and Fasiolo added on top was
the *spread* of the weather across those cells — the spatial standard deviation, minimum, and
maximum of the gridded fields. Measured by the Diebold-Mariano test against the same model without
the spread features, adding them improved the pinball score significantly in 2 of the 14 regions,
worsened the pinball score significantly in 3, and made no significant difference in the remaining
9. Browell and Fasiolo report that cross-validation had suggested a small gain which "is not
consistently reproduced on test data and therefore inconclusive", and conclude that gridded
numerical weather prediction "does not appear to add significant value to deterministic and
probabilistic net-load forecasts in the present framework" — while allowing that "it is possible
that other forecasting methods would be able to extract value from this data by constructing
different features".

Weather itself mattered a great deal to their model. Adding the regionally-averaged wind and
irradiance to a model carrying only calendar features and the point temperature cut the pinball
score — the single-quantile equivalent of the continuous ranked probability score — by 40% overall,
by 60% in North Scotland, where embedded wind capacity exceeds peak load, and by 10% in Greater
London, where there is little embedded generation. So the question this result puts to Flexpectation
is not if weather matters but if the spread of the weather across a region does.

Artificial Forecasting obtained postcode-level weather forecasts for two wind-connected primary
substations after the deliverable reported that its wind-connected models had performed poorly, and
found that the postcode-level forecasts "did not notably improve model performance". The deliverable
nonetheless names better weather data as a next step, without saying what would be better than
postcode-level.

### Weather improved low-voltage forecasts less than expected in the past

[Haben et al. (2019)](https://doi.org/10.1016/j.ijforecast.2018.10.007) tested 100 real low-voltage
feeders with both forecast and observed temperature, and found that temperature had no effect on
forecast accuracy, or a negative one.
[Haben et al. (2019)](https://doi.org/10.1016/j.ijforecast.2018.10.007) used data collected in 2014
and 2015. We expect how much weather matters at a substation to be changing quickly, because
embedded solar generation and heat pumps are what make a substation weather-dependent, and there are
far more of both on the distribution network now than there were then. That expectation is a
prediction, though, not a measurement — and the Scottish primary-substation sensitivities of
[Fox et al. (2018)](https://doi.org/10.34890/134), measured on the 10 years of weather and
substation data before its publication and described in the full review, say weather was already
moving primary substation demand well before the mid-2010s.

### A model trained on none of NGED's data may match a model trained on all of it

[Kaas et al. (2026)](https://arxiv.org/abs/2607.01966) tested Chronos-2, a general-purpose
time-series model that had never seen their data, against models trained on the first 160 of their
200 German low-voltage feeders and scored, like Chronos-2, on all 200 feeders. Chronos-2 beat every
purpose-trained competitor on mean absolute error, 3.8 kW against 4.2 kW. The authors describe their
purpose-trained models as lightly engineered, and challenge 1 above found only a modest return to
model sophistication. But a model trained on the feeders' own history, beaten by a model that saw
none of that history, still tells us how much any programme of heavy engineering is likely to
improve accuracy.

## What network operators have already built

**Nine projects run by electricity network operators have already built a forecasting capability
that overlaps Flexpectation's, and one of the nine — Northern Powergrid's Artificial Forecasting —
is deployed where Flexpectation is not, which is why that project gets a subsection of its own at
the end of this section.** The last row of the table is Flexpectation itself, so the comparison is
direct. Where a project's published deliverables do not answer a column, the cell says so rather
than being left blank.

| Project | What the project forecasts | Scale | Horizon | Uncertainty published |
|---|---|---|---|---|
| [Artificial Forecasting](https://smarter.energynetworks.org/projects/npg_sif_006-1/) (Northern Powergrid) | Demand and customer export at primary substations; active power at secondary | 551 primary substations with export data, 171 modelled; 729 secondary substations | Day-ahead to week-ahead at primary, evaluated to 11 days; week- to month-ahead at secondary | Half-hourly, with 5th-to-95th-percentile bands |
| [SSEN TRANSITION](https://ssen-innovation.co.uk/transition/) | Net load, split into demand and generation, then recombined | 13 primary substations, their bulk supply points, and their 33 kV and 11 kV feeders | 30 minutes to 10 days | A 40-member ICON-EU ensemble to 4 days, one deterministic forecast after that |
| [SSEN FastTrack](https://smarter.energynetworks.org/projects/10166254/) | How the connections queue, around 180 GW, will load the distribution network | Primary substations up to the grid supply point | A planning horizon rather than an operational one | A probability that a queued connection becomes real load |
| [NGED's EFFS](https://smarter.energynetworks.org/projects/wpden03/) | Grid supply points, bulk supply points, primary substation transformers, and generation sites | Across NGED's whole distribution network | 1 hour to 6 months | None |
| [UK Power Networks' Power Flow to Solar Capacity](https://smarter.energynetworks.org/projects/nia_ukpn0104/) | The capacity of unmetered solar behind each primary substation, then that solar's generation | Not stated in what we read | Not stated in what we read | Not stated in what we read |
| [SP Energy Networks' Predict4Resilience](https://smarter.energynetworks.org/projects/10061710/) | Electricity network faults, not load | Per district | Up to 7 days | A probability distribution driven by a weather ensemble |
| [Fox et al. (2018)](https://doi.org/10.34890/134) (SP Energy Networks) | The effect of weather on past peak demand, not a forward forecast | 13 primary substations in the proof of concept, almost 400 in production | Backwards over 10 years | None |
| [OpenSTEF](https://lfenergy.org/projects/openstef/) (Alliander, the Netherlands) | Net load, with a splitter into solar, wind, and residual parts | Thousands of grid connection points | To 48 hours | Yes; the framework is built for probabilistic forecasting |
| [Cordier et al. (2024)](https://doi.org/10.1049/icp.2024.2058) (Enedis, France) | Consumption and generation at the substation since 2015; the finer-grid method the paper describes covers consumption, not generation | All 2,300 high-voltage-to-medium-voltage substations, extending to 3,678 of the more than 5,000 transformers inside them, and towards 750,000 medium-to-low-voltage substations | Not stated in the paper; the forecasts run at 10- or 30-minute resolution | None stated in the paper |
| **Flexpectation** | Net demand, with unmetered generation inferred | 32 series in the trial area; 52 grid supply points, 271 bulk supply points, and 1,161 primary substations across NGED's whole distribution network from 2027 | 14 days, updated every 6 hours | A 51-member ECMWF ensemble across the whole horizon |

**SSEN's TRANSITION is the closest precedent we found for Flexpectation's method.** TRANSITION split
each substation's net load — demand minus whatever generation behind that substation happened to
produce — into demand and generation, forecast the two separately, then recombined them.
Flexpectation adds an ensemble that spans the whole 14-day horizon, and deployment across a whole
distribution network; TRANSITION set out to build neither. TRANSITION's ensemble covered the first 4
days, so from day 4 to day 10 a single deterministic forecast was all TRANSITION had, and
Flexpectation's forecast horizon runs to 14 days. And TRANSITION was a 13-substation trial rather
than a deployment across a whole distribution network. The rest of TRANSITION's published design
matches what Flexpectation is building.

**NGED's own Electricity Flexibility and Forecasting System independently selected XGBoost, which
its evaluation reported as the most accurate of the three methods tested and as easy to automate.**
The project compared XGBoost against a long short-term memory (LSTM) neural network and against
ARIMA, and its evaluation report says XGBoost "provided the best results of the three methods
tested, closely followed by LSTM", recommending XGBoost because it also allows simplified testing of
features and can be easily automated. The report caveats that the LSTM could not be fully explored
for want of graphics processing units, and expects that more testing would have brought the LSTM
level with XGBoost rather than past it. Selecting XGBoost is the same starting point Flexpectation
uses. [EFFS](https://smarter.energynetworks.org/projects/wpden03/) ran from 2018 to 2021 as a
Network Innovation Competition project costing £3.3 million, and its forecasts were deterministic.
Publishing uncertainty bands is the step Flexpectation adds.
[UK Power Networks' Power Flow to Solar Capacity](https://smarter.energynetworks.org/projects/nia_ukpn0104/)
is the direct predecessor of Flexpectation's unmetered-solar work, as challenge 8 above sets out.

**Two of the nine projects in the table are outside GB: OpenSTEF in the Netherlands and Enedis in
France.** [OpenSTEF](https://lfenergy.org/projects/openstef/) is also the only operational
forecasting system run by a network operator in this review whose code can be read rather than
inferred from a deliverable.

**Enedis has forecast all 2,300 of its high-voltage-to-medium-voltage substations since 2015, and is
now extending the forecast below the substation**
([Cordier et al. (2024)](https://doi.org/10.1049/icp.2024.2058)). The extension reaches 3,678 of the
more than 5,000 transformers inside those substations, and is heading towards the 750,000
medium-to-low-voltage substations beyond them. Enedis has therefore been forecasting operationally,
at the scale Flexpectation reaches in 2027, for a decade.

**Fitting a model to each transformer beat the method Enedis runs in production, which shares one
substation forecast out across its transformers by fixed coefficients.** The per-transformer models
scored 6.0% mean absolute percentage error against 9.3% on the day those coefficients were
refreshed, and 8.1% against 13.0% across the whole test period. That second comparison counts only
the transformers whose coefficient then moved by less than 2.5%, and on that comparison 84% of
transformers were more accurate under their own model. Cordier et al. chose both comparisons
deliberately, as the cases where the fixed-coefficient method is "the most relevant and the most
difficult to outperform". Cordier et al. do not say what their percentage error is normalised by,
and report that the complete pipeline has not yet been evaluated end to end.

### Northern Powergrid's Artificial Forecasting is already operational, where Flexpectation is not

**Northern Powergrid's Artificial Forecasting is the closest concurrent project we found to
Flexpectation.**
[Artificial Forecasting](https://smarter.energynetworks.org/projects/npg_sif_006-1/) is an Ofgem
Strategic Innovation Fund programme, with about £3.9 million of grant across its three phases, run
by Northern Powergrid with Faculty, EV.energy, and Oaktree Power, the final Beta phase running to
February 2027. The Beta deliverables that the rest of this section draws on sit under a
[separate project registration](https://smarter.energynetworks.org/projects/10145998/) from the
Alpha ones. Artificial Forecasting does much of what Flexpectation does at primary substations, and
also covers secondary substations, which Flexpectation does not. At the time of writing, Artificial
Forecasting is further ahead than Flexpectation.

**Artificial Forecasting has run operationally through a full winter flexibility procurement
cycle.** A forecasting service for primary substations is deployed and has passed Northern
Powergrid's architecture review board, data governance, and information security checks for its
current deployment. Northern Powergrid's System Forecasting team used the service operationally
through a full winter flexibility procurement cycle to support week-ahead dispatch decisions. The
service produces half-hourly probabilistic forecasts with 5th-to-95th-percentile bands, flags
forecast exceedances of firm capacity, and is benchmarked against Northern Powergrid's existing
growth-based and persistence methods and a rolling 4-week baseline. The deliverable states that
performance did not materially degrade on average across the 11-day horizon. The deliverable does
not publish the figures behind that statement.

**Artificial Forecasting's value case puts whole-life net present value at around £60 million for
one distribution network operator, or £250 million if three further operators adopt Artificial
Forecasting.** The net present value comes from a 3% reduction in spending on reinforcement —
building bigger transformers and cables — in the current price-control period, rising to 6% in the
next, and from a 25% improvement in the cost-effectiveness of contracted flexibility. None of the
four benefit categories in Artificial Forecasting's benefits assessment is curtailment. The forecast
covers customer export at primary substations, but the one published value case in this review puts
no money on curtailment, which NGED now values alongside the flexibility it procures. The project
pairs those figures with a direct caveat: it reports early Beta evidence, from one winter
procurement cycle, supporting the performance assumptions behind the value case, which "remains
appropriate, subject to further validation".

**Artificial Forecasting is independent evidence that short-term substation forecasting is
operationally useful**, that a network operator will change its procurement process around a
half-hourly probabilistic substation forecast, and that a benefits case has been made and accepted.
Because Artificial Forecasting is public, operational, and benchmarked against a real incumbent
method, Artificial Forecasting is also the clearest example we found of what "working" looks like.
Artificial Forecasting's core intellectual property is to be made available royalty-free to other GB
distribution network operators, and we would rather build on that intellectual property than rebuild
it — a shared evaluation protocol between two GB distribution network operators would be worth more
to both than two separate protocols.

**Flexpectation is nonetheless attempting more than Artificial Forecasting's published deliverables
describe, which is the case for running both.** The two projects overlap on forecasting net demand
at primary substations and on forecasting metered generation. Artificial Forecasting's Beta
registration also claims load disaggregation as an innovation — "a novel approach to forecasting HV
[high-voltage] load, separately modelling gross demand and distributed generation" — but the
deliverables we read describe forecasting two series that are each already measured. The Beta annual
progress report produces net demand "by independently modelling customer export data", the Alpha
technical report covers "all 160 substations where both gross demand and customer export data were
available", and the Embedded Capacity Register enters the model as an input feature, listing what is
registered rather than estimating what is not. Flexpectation's challenges 8 and 9 are the different
problem of inferring an unmetered generator's half-hourly output from a substation's net flow, which
is blind source separation. Two more of Flexpectation's challenges do have a counterpart in
Artificial Forecasting's deliverables. The Beta annual progress report describes automated health
checks and dashboards that "highlight substations where input data is degraded (e.g. faulty sensors,
frozen or anomalous values)" and an extract-transform-load pipeline that "flags frozen/spiky SCADA
[supervisory control and data acquisition] data before modelling", which is challenge 6; and the
Alpha user research treats planned and unplanned outages as data worth bringing in and as a reason
to widen the error margin, which is a different response to challenge 4's problem rather than no
response.

**Five of Flexpectation's nine challenges have no counterpart we could find in Artificial
Forecasting's published deliverables:** tracking the effective capacity of metered generators;
forecasting a substation as if it were always in its normal running arrangement, rather than
dropping the periods when it was not; recovering signed net demand from an apparent-power meter;
inferring unmetered solar and wind from a substation's net flow; and doing the same for heat pumps,
chargers, and batteries. Across every Artificial Forecasting deliverable published on the Smarter
Networks Portal — Discovery, Alpha, and Beta, save one file that holds a single blank page —
"abnormal", "unmetered", "apparent power", "non-directional", "blind source", and "source
separation" return nothing at all; "capacity" appears 123 times but never as an effective or derated
capacity; and the five occurrences of a "switch" stem are generators switching off, switchgear asset
types, and switching over a data feed. Heat pumps and electric vehicles do appear, as drivers of
demand growth and as model features rather than as quantities separated out of a net flow.
Flexpectation also delivers 1st and 99th percentiles where Artificial Forecasting's published bands
run from the 5th to the 95th, and the curtailment decisions NGED describes turn on those outer
levels.

## Why we think this ambitious plan can be done

**Measured against the studies we found, the plan sits outside the published literature in five ways
at once.** The distance between Flexpectation's plan and the published literature says more about
where our search fell short than about the quality of the work that fills the rest of the field. No
study in this review drives a substation forecast from a weather ensemble across a 14-day horizon.
No study we read models the tails explicitly at substation level; the one study that models them
explicitly at all works on regions far larger than a substation. No study we read puts unmetered
generation inside a probabilistic forecast at substation level over a multi-day horizon, though
unmetered generation, probabilistic forecasting at substation level, and a multi-day horizon each
exist on their own. No study we read tracks the available capacity of a mixed fleet of solar, wind,
and dispatchable generators at one distribution network, or measures whether doing so improves the
forecast. No study we read turns switching-contaminated history at a substation into a useful input
rather than deleting it, rewriting it, or absorbing the accuracy loss of leaving it in; the nearest
precedent, [Liu et al. (2019)](https://doi.org/10.1109/ACCESS.2019.2951422), conditions a forecast
on an operating-state label, but for switching between transformers inside one substation, where the
substation total stays metered throughout. Flexpectation attempts all nine challenges above, across
four families of model:

- a heavily-tuned version of the gradient-boosting approach that won the tabular forecasting competitions reviewed above, and which NGED's own EFFS project independently selected;
- weather and time encoders pre-trained on large datasets, so that a model for one substation can
  start from what has been learned across all of them;
- models that use the connectivity map explicitly;
- differentiable physics — building known physical behaviour directly into the model, so that it has
  to learn only what the physics cannot supply: the response of a solar panel and of a wind turbine
  on the generation side, and the thermal response of buildings on the demand side.
  [Gijón et al. (2025)](https://arxiv.org/abs/2502.07344) fit a model of that kind to a single
  wind farm.

**By the standard of scope in this literature, each of the four strands is a separate piece of
work.** Almost every study reviewed above takes on one of the nine challenges, at one voltage level,
with one family of model; the few that touch more than one almost all solve those challenges as a
pipeline rather than together. Pre-training weather and time encoders and then reading a
substation's probabilistic forecast off them would be a full study by that standard, and so would
each of the other three strands. Sizing the four strands as separate studies scopes the work rather
than promising an output — how many of the strands survive contact with the data is exactly what the
project has to find out.

**Only the heavily-tuned gradient-boosting model, the first of the four strands, is in scope for
Flexpectation version 1.** The other three strands belong to the scale-up across NGED's whole
distribution network from 2027, as does the disaggregation of unmetered generation.

**The main reason for attempting all nine challenges at once is that the nine may be one challenge
rather than nine.** A switching event, a turbine out for repair, and a stuck meter all surface in
the same place: as a discrepancy between what a substation metered and what the weather and the
calendar say it should have metered. Almost every study reviewed above that touches more than one of
the nine challenges solves those challenges as a pipeline, and the exception we found,
[Pierrot and Pinson (2024)](https://doi.org/10.1080/00401706.2024.2350421), fits one wind farm's
time-varying capacity jointly with its probabilistic forecast rather than a substation's several
challenges together. In the pipelines one stage's output is frozen before the next stage sees it, so
an error made early cannot be corrected later and the forecast error never gets to tell the capacity
estimator it was wrong.

**The question we want to answer is whether one model that estimates capacity, switching state, and
demand together beats the serial pipeline every study we read used.** NGED's specification leaves
room for that combined approach, asking that capacity, switching state, and demand be handled,
rather than that each be handled explicitly.

**The first reason for confidence is that one more experiment takes compute time rather than staff
time.** The core forecast already exists and runs today, on an experiment framework that makes one
more experiment cost compute time rather than staff time. That low marginal effort is what makes it
realistic to run on the order of hundreds of machine-learning experiments a month, and it is the
same argument the introduction to this review makes.

**Several of the four model families will not work.** Expecting that failure is what makes the four
families research directions rather than engineering tasks. The honest expectation is that some
deliver clearly, some produce a negative result worth publishing, and some are abandoned. Both NGED
and this project count a negative result as an outcome: evidence that switching cannot be recovered
from power data alone, for instance, would be worth having, because it would justify extracting
switching labels from operational systems instead of continuing to look.

## References

Every source cited above, in alphabetical order by first author. The full review cites 30 sources that this summary does not, and this summary cites 28 that the full review does not.

- Abur, A. and Expósito, A. G. (1997). [Detecting multiple solutions in state estimation in the
presence of current magnitude measurements](https://doi.org/10.1109/59.575721). *IEEE Transactions
on Power Systems*.
- Authen, K., Riemer-Sørensen, S., Michałowska, K., Vedvik, E., Razick, S. and Visoka, K.
(2026). [WindAI: Wind power forecasting in Norway – data competition summary](https://doi.org/10.5617/nmi.13106).
*Nordic Machine Intelligence*.
- Bian, Y., Zheng, N., Zheng, Y., Xu, B. and Shi, Y. (2024). [Predicting Strategic Energy Storage
Behaviors](https://doi.org/10.1109/TSG.2023.3303469). *IEEE Transactions on Smart Grid*.
- Bioucas-Dias, J. M., Plaza, A., Dobigeon, N., Parente, M., Du, Q., Gader, P. and Chanussot, J.
(2012). [Hyperspectral Unmixing Overview: Geometrical, Statistical, and Sparse Regression-Based
Approaches](https://doi.org/10.1109/JSTARS.2012.2194696). *IEEE Journal of Selected Topics in
Applied Earth Observations and Remote Sensing*.
- Bouman, R., Schmeitz, L., Buise, L., Heres, J., Shapovalova, Y. and Heskes, T. (2024). [Acquiring
Better Load Estimates by Combining Anomaly and Change Point Detection in Power Grid Time-series
Measurements](https://arxiv.org/abs/2405.16164). *Sustainable Energy, Grids and Networks*.
- Browell, J. and Fasiolo, M. (2021). [Probabilistic Forecasting of Regional Net-load with
Conditional Extremes and Gridded NWP](https://arxiv.org/abs/2103.10335). *IEEE Transactions on
Smart Grid*.
- Browell, J., van der Meer, D., Kälvegren, H., Haglund, S., Simioni, E., Bessa, R. J. and Wang, Y.
(2026). [The hybrid renewable energy forecasting and trading competition
2024](https://doi.org/10.1016/j.ijforecast.2025.10.005). *International Journal of Forecasting*.
- Brown, C. F. et al. (2025). [AlphaEarth Foundations: An embedding field model for accurate and
efficient global mapping from sparse label data](https://arxiv.org/abs/2507.22291).
- Brudermueller, T., Breer, F. and Staake, T. (2023). [Disaggregation of Heat Pump Load Profiles
From Low-Resolution Smart Meter Data](https://doi.org/10.1145/3600100.3623731). *Proceedings of the
10th ACM International Conference on Systems for Energy-Efficient Buildings, Cities, and
Transportation (BuildSys)*.
- Buizza, R. and Leutbecher, M. (2015). [The forecast skill
horizon](https://doi.org/10.1002/qj.2619). *Quarterly Journal of the Royal Meteorological
Society*.
- Campagne, E., Amara-Ouali, Y., Goude, Y., Zehavi, I. and Kalogeratos, A. (2025). [Graph Neural
Networks for Electricity Load Forecasting](https://arxiv.org/abs/2507.03690).
- Cheung, C. M., Kuppannagari, S. R., Srivastava, A., Kannan, R. and Prasanna, V. K. (2023).
[Behind-the-Meter Solar Generation Disaggregation at Varying Aggregation Levels Using Consumer
Mixture Models](https://doi.org/10.1109/TSUSC.2022.3192456). *IEEE Transactions on Sustainable
Computing*.
- Cordier, G. et al. (2024). [Methods and techniques used to produce electricity forecasts on
Enedis’ distribution network at a finer grid than the HV/MV
substation](https://doi.org/10.1049/icp.2024.2058). *CIRED 2024 Vienna Workshop*, in *IET
Conference Proceedings*.
- Dantas, G. and Browell, J. (2026). [Seamless Short‐ to Mid‐Term Probabilistic Wind Power
Forecasting](https://doi.org/10.1002/we.70079). *Wind Energy*.
- de Vilmarest, J., Browell, J., Fasiolo, M., Goude, Y. and Wintenberger, O. (2024). [Adaptive
Probabilistic Forecasting of Electricity (Net-)Load](https://doi.org/10.1109/TPWRS.2023.3310280).
*IEEE Transactions on Power Systems*.
- Department for Energy Security and Net Zero (2026). [MCS certified domestic battery
installation statistics](https://www.gov.uk/government/statistics/mcs-certified-domestic-battery-installation-statistics).
- Department for Energy Security and Net Zero (2026). [Solar photovoltaics
deployment](https://www.gov.uk/government/statistics/solar-photovoltaics-deployment).
- Dong, J., Valzania, L., Maillard, A., Pham, T., Gigan, S. and Unser, M. (2023). [Phase Retrieval:
From Computational Imaging to Machine Learning: A Tutorial](https://doi.org/10.1109/MSP.2022.3219240).
*IEEE Signal Processing Magazine*.
- Doubleday, K., Van Scyoc Hernandez, V. and Hodge, B. M. (2020). [Benchmark probabilistic solar
forecasts: Characteristics and recommendations](https://doi.org/10.1016/j.solener.2020.05.051).
*Solar Energy*.
- Ebrahimi, M., Rastegar, M. and Arefi, M. M. (2022). [Real-Time Estimation Frameworks for
Feeder-Level Load Disaggregation and PEVs' Charging Behavior Characteristics
Extraction](https://doi.org/10.1109/TII.2021.3118101). *IEEE Transactions on Industrial
Informatics*. Read as the author-posted accepted manuscript.
- Electricity North West (2018). [ATLAS](https://smarter.energynetworks.org/projects/nia_enwl008/).
- Elia Group (2026). [Predico: collaborative forecasting platform](https://innovation.eliagroup.eu/en/projects/predico-collaborative-forecasting-platform).
- Erdener, B. C., Feng, C., Doubleday, K., Florita, A. and Hodge, B.-M. (2022). [A review of
behind-the-meter solar forecasting](https://doi.org/10.1016/j.rser.2022.112224). *Renewable and
Sustainable Energy Reviews*.
- Faustine, A., Nunes, N. J. and Pereira, L. (2025). [Efficiency through Simplicity: MLP-based
Approach for Net-Load Forecasting with Uncertainty Estimates in Low-Voltage Distribution
Networks](https://doi.org/10.1109/TPWRS.2024.3400123). *IEEE Transactions on Power Systems*.
- Fildes, R. (2020). [Learning from forecasting
competitions](https://doi.org/10.1016/j.ijforecast.2019.04.012). *International Journal of
Forecasting*.
- Fox, J., Plecas, M., Neilson, D., Cannon, D. and Parr, J. (2018). [Analysis of local demand trends
and forecasting through weather correction and benefit to DSO transistion and
microgrids](https://doi.org/10.34890/134). *CIRED Workshop, Ljubljana*.
- Gao, A., Zheng, J., Mei, F. and Liu, Y. (2024). [Toward intelligent demand-side energy management
via substation-level flexible load disaggregation](https://doi.org/10.1016/j.apenergy.2024.123361).
*Applied Energy*. Full text not obtained; read as the abstract, highlights, introduction, and
dataset description on the publisher's landing page.
- Gijón, A., Eiraudo, S., Manjavacas, A., Schiera, D. S., Molina-Solana, M. and Gómez-Romero, J.
(2025). [Integrating Physics and Data-Driven Approaches: An Explainable and Uncertainty-Aware
Hybrid Model for Wind Turbine Power Prediction](https://arxiv.org/abs/2502.07344). *Computer
Physics Communications*.
- Gilbert, C., Browell, J. and Stephen, B. (2023). [Probabilistic load forecasting for the low
voltage network: forecast fusion and daily peaks](https://arxiv.org/abs/2206.11745). *Sustainable
Energy, Grids and Networks*.
- Gisiger, O., Melillo, A. and Schuetz, P. (2026). [Heat pump detection and load disaggregation in
low-resolution smart meter data with convolutional neural
networks](https://doi.org/10.1016/j.egyai.2026.100691). *Energy and AI*.
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
- Hensley, B. S. and Bull, P. (2018). [Mitigating Complex Dust Foregrounds in Future Cosmic
Microwave Background Polarization Experiments](https://doi.org/10.3847/1538-4357/aaa489). *The
Astrophysical Journal*.
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
- INESC TEC. [Predico documentation](https://predico-elia.inesctec.pt/docs/).
- Ju, Y., Wu, W., Ge, F., Ma, K., Lin, Y. and Ye, L. (2018). [Fast Decoupled State Estimation for
Distribution Networks Considering Branch Ampere
Measurements](https://doi.org/10.1109/TSG.2017.2709463). *IEEE Transactions on Smart Grid*.
- Jumper, J. (2024). [Nobel Week interview](https://youtu.be/nNM1QdmFwIs?t=852). Nobel Prize YouTube
channel, 6 December 2024.
- Kaas, B., Treutlein, M., Gerber, H. B., Neumann, O., Phatthanakhuha, C., Resch, O., Mikut, R. and
Hagenmeyer, V. (2026). [Probabilistic Low-Voltage Peak Load Forecasting with Time Series
Foundation Models Evaluated on Application-Oriented Metrics](https://arxiv.org/abs/2607.01966).
- Kanchana, W., Singh, J. G. and Ongsakul, W. (2026). [A non-intrusive net-load disaggregation
framework for behind-the-meter DER capacity estimation using a generative adversarial network data
curation](https://doi.org/10.1016/j.epsr.2026.113279). *Electric Power Systems Research*. Full text
not obtained; read as the highlights, abstract, introduction, and section snippets on the
publisher's landing page.
- Kara, E. C., Roberts, C. M., Tabone, M., Alvarez, L., Callaway, D. S. and Stewart, E. M. (2018).
[Disaggregating solar generation from feeder-level
measurements](https://doi.org/10.1016/j.segan.2017.11.001). *Sustainable Energy, Grids and
Networks*. Read as the accepted manuscript of the version of record, and as the preprint
(arXiv:1607.02919, which carries a different title). The preprint's power-factor-based estimator,
the source of the 25% figure cited above, does not appear in the published version.
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
- Martín, P., Moreno, G., Rodríguez, F. J., Jiménez, J. A. and Fernández, I. (2018). [A Hybrid Approach to Short-Term Load Forecasting Aimed at Bad Data Detection in Secondary Substation Monitoring Equipment](https://doi.org/10.3390/s18113947). *Sensors*.
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
- Moriano, J., Rodríguez, F., Martín, P., Jiménez, J. and Vuksanovic, B. (2016). [A New Approach to Detection of Systematic Errors in Secondary Substation Monitoring Equipment Based on Short Term Load Forecasting](https://doi.org/10.3390/s16010085). *Sensors*.
- National Energy System Operator. [Embedded wind and solar
forecasts](https://www.neso.energy/data-portal/embedded-wind-and-solar-forecasts).
- National Energy System Operator (2025). [Future Energy Scenarios 2025](https://www.neso.energy/publications/future-energy-scenarios-fes).
- National Energy System Operator (2025). [Regional breakdown of FES data (electricity)](https://www.neso.energy/data-portal/regional-breakdown-fes-data-electricity).
- Nguyen, T. N. and Müsgens, F. (2026). [A meta-analysis of solar forecasting based on skill
score](https://doi.org/10.1063/5.0300682). *Journal of Renewable and Sustainable Energy*.
- Northern Powergrid (2024). [Artificial Forecasting, Alpha
phase](https://smarter.energynetworks.org/projects/npg_sif_006-1/).
- Northern Powergrid (2024). [Detecting LCTs from Smart Meter Consumption
Data](https://smarter.energynetworks.org/projects/npg_nia_-49/).
- Northern Powergrid (2025). [Artificial Forecasting, Beta
phase](https://smarter.energynetworks.org/projects/10145998/).
- Ofgem (2025). [Decision: flexibility market asset registration](https://www.ofgem.gov.uk/decision/decision-flexibility-market-asset-registration).
- Ofgem (2025). [Enhancing asset visibility: Distribution Network Operator options
consultation](https://www.ofgem.gov.uk/sites/default/files/2025-12/Enhancing%20asset%20visibility%20-%20Distribution%20Network%20Operator%20Options%20consultation.pdf).
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
- SGN and Northern Gas Networks (2026). [Forecaster for Embedded Generation (FEmGE),
NIA2_SGN0081](https://portal.futureenergynetworks.org.uk/content/projects/NIA2_SGN0081).
- Short, M., Crosbie, T., Dawood, M. and Dawood, N. (2017). [Load forecasting and dispatch
optimisation for decentralised co-generation plant with dual energy
storage](https://doi.org/10.1016/j.apenergy.2016.04.052). *Applied Energy*, 186, 304-320.
- Shukla, S. and Hong, T. (2024). [BigDEAL Challenge 2022: Forecasting peak timing of electricity
demand](https://doi.org/10.1049/stg2.12162). *IET Smart Grid*.
- Siméoni, O. et al. (2025). [DINOv3](https://arxiv.org/abs/2508.10104).
- SP Energy Networks (2023).
[Predict4Resilience](https://smarter.energynetworks.org/projects/10061710/).
- Tang, L., Ashtine, M., Hua, W. and Wallom, D. C. H. (2024). [Sensitivity analysis of distributed
photovoltaic system capacity estimation based on artificial neural
network](https://doi.org/10.1016/j.segan.2024.101396). *Sustainable Energy, Grids and Networks*.
- Teng, S., Cambier van Nooten, C., van Doorn, J., Ottenbros, A., Huijbregts, M. and Jansen, J.
(2023). [Near real-time predictions of renewable electricity production at substation level via
domain adaptation zero-shot learning in sequence](https://doi.org/10.1016/j.rser.2023.113662).
*Renewable and Sustainable Energy Reviews*.
- UK Power Networks. [Power Flow to Solar Capacity
(NIA_UKPN0104)](https://smarter.energynetworks.org/projects/nia_ukpn0104/), listed on the portal as
"AI for Visibility and Forecasting of Renewable Generation".
- UK Power Networks and PPA Energy and Capula (2014). [Distribution Network Visibility: LCN Fund
Tier 1 Close Down
Report](https://www.ofgem.gov.uk/sites/default/files/docs/2014/03/dnv_cdr_version_3.0_270214.pdf).
- Viotti, O., Arnqvist, J. and Olauson, J. (2026). [Estimating Wind‐Power Capacity Time Series From
Production Data Using a Power Curve Model and Quadratic
Optimization](https://doi.org/10.1002/we.70136). *Wind Energy*.
- Virieux, J. and Operto, S. (2009). [An overview of full-waveform inversion in exploration
geophysics](https://doi.org/10.1190/1.3238367). *Geophysics*.
- Wang, F., Ge, X., Dong, Z., Yan, J., Li, K., Xu, F., Lu, X., Shen, H. and Tao, P. (2022). [Joint
Energy Disaggregation of Behind-the-Meter PV and Battery Storage: A Contextually Supervised Source
Separation Approach](https://doi.org/10.1109/TIA.2022.3144244). *IEEE Transactions on Industry
Applications*. Abstract only.
- Western Power Distribution (2017). [Time Series Data
Quality](https://smarter.energynetworks.org/projects/nia_wpd_011/).
- Western Power Distribution (2021). [Electricity Flexibility and Forecasting System
(EFFS)](https://smarter.energynetworks.org/projects/wpden03/).
- Wieland, F.-G., Hauber, A. L., Rosenblatt, M., Tönsing, C. and Timmer, J. (2021). [On structural
and practical identifiability](https://doi.org/10.1016/j.coisb.2021.03.005). *Current Opinion in
Systems Biology*.
- Willis, H. L., Powell, R. D. and Wall, D. L. (1984). [Load Transfer Coupling Regression Curve Fitting for Distribution Load Forecasting](https://doi.org/10.1109/TPAS.1984.318713). *IEEE Transactions on Power Apparatus and Systems*.
- Yi, M. and Wang, M. (2022). [Recent Results of Energy Disaggregation with Behind-the-Meter Solar
Generation](https://arxiv.org/abs/2207.03490). *11th Bulk Power Systems Dynamics and Control
Symposium (IREP), Banff*. The authors' own summary of their two *IEEE Transactions on Power Systems*
papers on the same work, both of which are closed.
- Zhang, X. Y., Watkins, C. and Kuenzel, S. (2022). [Multi-quantile recurrent neural network for
feeder-level probabilistic energy disaggregation considering roof-top solar
energy](https://doi.org/10.1016/j.engappai.2022.104707). *Engineering Applications of Artificial
Intelligence*.
