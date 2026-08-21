# The current state of the art in energy forecasting

Before we discuss the literature, there is a very important caveat to admit up-front: In 2026, no honest review of the energy forecasting literature can claim to reveal the canonical "state of the art"! That is because (almost) all energy forecasting papers measure performance in different ways, against different datasets. It's like an international football tournament where every team plays by different rules, with different size goals.

Energy forecasting researchers have done great work over the years. But, unfortunately, the literature does not tell us how those approaches compare against each other, especially in messy "real world" energy forecasting scenarios. This isn't anyone's fault; it's a systemic failure. And the industry is already aware of this problem, and people are trying to fix it. But, at the time of writing, the literature cannot yet tell us the current state of the art solutions for the problems that NGED cares about.

Having said that, there have been some valiant attempts to compare multiple forecasting approaches against the same dataset. But none of these attempts directly address the main challenges relevant to Flexpectation. Before we discuss those attempts, we must emphasise two reasons for optimism:

Firstly, whilst we might not know exactly which algorithms provide the best forecasting performance, we do know how to go about finding out. There's no magic. Machine learning is an empirical science, and progress in it comes largely from testing many ideas under identical conditions and measuring carefully — because most ideas fail. John Jumper, who shared the 2024 Nobel Prize in Chemistry for his work on AlphaFold, puts the share of research ideas that fail at around 90%, and treats that rate as an ordinary and necessary feature of doing research rather than as evidence of doing it badly ([Nobel Week interview](https://www.nobelprize.org/prizes/chemistry/2024/jumper/interview/), 6 December 2024, from 14:12). If roughly one idea in ten survives contact with the data, ten attempts is simply the price of one result. So our task is to run hundreds of ML experiments, and then measure performance against the same dataset, using the same performance metrics. This, in turn, requires us to design and build a framework that makes it easy to run hundreds of ML experiments per month. At the time of writing, we have implemented the first version of this framework, and we will continue to evolve the framework over the course of the project.

Secondly - and perhaps most importantly - the fact that the industry doesn't yet know the state of the art is a huge opportunity for the Flexpectation project: We are in a very privileged position where we can try hundreds of ideas, and test the best ideas in the real world. We have a fantastic opportunity to make a significant contribution to the energy forecasting industry by publishing our "leaderboards of ML experiments", and hence help the industry as a whole to better understand how multiple approaches perform.

## What we read

We read ten papers in full, drew on two more that were only partly available to us, and read in full
the published deliverables of one concurrent UK network project. The bar for inclusion was
deliberately high: a paper had to bear on a decision Flexpectation actually faces
*and* change something we believed. A great deal of good work was left out on that basis, and the
last section of this review says what and why. Two of the sources below we could not read in full,
and everything drawn from them is flagged where it appears.

## The best published results, and why they cannot be compared

The table below gives the best reported result from each source. Every entry is genuinely
best-in-class for the problem its authors set themselves. Almost none of them can be compared
directly with any other, because the target, the horizon, the aggregation level and the weather
assumption differ in nearly every row.

| Source | What they forecast | Horizon | Best result, and what it beat | Weather |
|---|---|---|---|---|
| [Browell et al. 2025 (HEFTCom competition)](https://arxiv.org/abs/2507.01579) | Combined wind and solar output, GB | Day-ahead | Winning team scored 22.18 MWh mean pinball loss against the organisers' benchmark of 53.58, though the benchmark did not adapt to a cable fault and the next teams scored 23.18 and 24.64. Revenue of £88.9m against a £105.2m perfect-decision ceiling | Real forecasts, live |
| [Kaas et al. 2026](https://arxiv.org/abs/2607.01966) | Net load, 200 low-voltage feeders, Germany | 4 days | A general-purpose "foundation" model that had never seen the data beat every purpose-trained model on average error, 3.839 kW against 4.184 kW | Actual weather, after the fact |
| [Hertel et al. 2026](https://arxiv.org/abs/2607.15705) | Load at three grid levels, Germany and Portugal | 4 days | Best model beat a naive forecast by 59.6% at transmission level, 42.3% at low-voltage feeders, 23.3% at individual customers | Actual weather, after the fact |
| [Kleinebrahm et al. 2026 (Energy-Arena)](https://arxiv.org/abs/2604.24705) | Live public leaderboard: prices, load, wind, solar | Day-ahead | No single winner — a continuously updated ranking, which is the point of it | Real forecasts, live |
| [Hong, Xie & Black 2019 (GEFCom2017)](https://doi.org/10.1016/j.ijforecast.2019.02.006) | Hierarchical load, New England | 2–9 weeks | No score available in anything we could read | Real forecasts, live |
| [Shukla & Hong 2024 (BigDEAL competition)](https://doi.org/10.1049/stg2.12162) | Peak load, three US utilities | Rolling months | Winning scores not published. The transferable finding is that rankings on peak size are almost uncorrelated with rankings on peak timing and peak shape, while timing and shape rank together | Mixed |
| [Haben et al. 2021](https://arxiv.org/abs/2106.00006) | Review of 221 low-voltage forecasting papers | — | Of 221 papers, **3** used a weather *forecast* and **none** used a weather ensemble | — |
| [Browell & Fasiolo 2021](https://arxiv.org/abs/2103.10335) | Regional net load, GB | Day-ahead | Their tail model needed **up to 24.6%** less upward reserve than a fixed-tail alternative at the same risk level, the saving depending strongly on how extreme that level is (3.2% at the least extreme tested). Adding wind and irradiance cut error 40% overall — 10% in London, 60% in North Scotland | Real forecasts |
| [Pinheiro et al. 2023](https://doi.org/10.1016/j.apenergy.2022.120493) | Load at 96,989 Portuguese secondary substations | Day-ahead | 42–47% better than the reference benchmark at system level. **At substation level, beat a naive forecast on 82–87% of network-owned and 66–70% of customer-owned sites** (the paper's prose and its figures differ by about four points) | Real forecasts, 7–8 h old |
| [Gilbert et al. 2023](https://arxiv.org/abs/2206.11745) | Load across a four-level GB hierarchy | Day-ahead | Combining forecasts gained **0.0–0.4% averaged over all periods**, but **5.7–9.0% when restricted to peaks** | None at all |
| [Bouman et al. 2024](https://arxiv.org/abs/2405.16164) | Switch-event and anomaly detection, 180 Dutch primary substations | Not a forecast | ~90% of resulting load estimates within a 10% error margin | None |
| **[Artificial Forecasting (Northern Powergrid)](https://smarter.energynetworks.org/projects/npg_sif_006-1/)** | **Demand and export at Northern Powergrid primary substations; net demand at secondary** | **Day- to week-ahead at primary; month-ahead at secondary** | **About 8% lower mean absolute error of utilisation rate; 83% of the top 10% of demand values captured inside its 5th–95th percentile band; better than a rolling four-week baseline at 8 of 8 near-capacity substations** | **Real forecasts at primary; none at secondary** |

### Three things make a result look good with no forecasting skill behind it

**The level of aggregation.** [Hertel et al.](https://arxiv.org/abs/2607.15705) ran the same models
against the same naive benchmark at three levels of the grid and beat it by 59.6% at transmission
and 23.3% at individual customers. The model did not get worse; the problem got harder. A headline
percentage therefore says more about where it was measured than about the method. It is the single
most important thing to take from this review, because it sets what to expect at NGED's primary
substations and customer meters.

**Weather known after the event.** Two of the studies above use the weather as it was known
immediately afterwards — short-range forecasts issued one to three hours ahead, or reanalysis —
rather than the weather that was forecast days out. They do this deliberately, so that differences
between models are not swamped by weather-forecast error, which is the right choice for their
question and the wrong one for ours: it removes the error that dominates beyond a day or two,
precisely the range NGED acts on. Their figures are upper bounds, not achievable performance.

**Averaging over all periods.** [Gilbert et al.](https://arxiv.org/abs/2206.11745)'s forecast
combination looks worthless averaged across every half-hour of the year and clearly worth having at
the daily peak — the same comparison, two answers. A number averaged over 17,520 half-hours is
dominated by the quiet ones, and the quiet ones are not why NGED buys flexibility.

### Which published numbers do transfer

Only two kinds. **Ratios against a stated baseline on a stated population** — which is why [Pinheiro
et al.](https://doi.org/10.1016/j.apenergy.2022.120493)'s finding that only 66–70% of customer-owned
substations beat a naive forecast is the most useful figure in the table. And **errors normalised by
something physical**, such as a substation's firm capacity or transformer rating, rather than by the
load that happened to occur. Absolute errors in kW or MW transfer to nothing, and none of the
absolute figures above should be read as a target for this project.

## What the literature does agree on

Six findings recur across independent studies, and we regard them as robust.

**1. Sophisticated methods beat simple ones by much less than expected.** In a live system covering
96,989 substations, a carefully tuned gradient booster matched a simpler, more interpretable model
on the national series they both forecast — 199 MW against 191 MW in root-mean-square error — and
was set aside on interpretability grounds. The Artificial Forecasting project tested the same class
of model for customer export and did not adopt it: boosted trees "helped some substations but harmed
others", and an interpretable Bayesian Ridge model was chosen instead. Across 729 secondary
substations, a neural network beat an average-of-the-last-four-weeks rule by about one percentage
point, and lost to it outright on the substations with the worst data. The winning entry in the
HEFTCom competition tuned a single setting and left everything else at its default. This is
reassuring rather than disappointing: it means interpretable models remain competitive, and that
effort is better spent on data quality and on evaluation than on model complexity.

**2. Accuracy falls as you move down the voltage levels.** Demand at a grid supply point is smooth
and largely predictable from calendar and weather; demand at a single secondary substation is
dominated by the behaviour of a handful of customers. Accuracy will therefore be reported separately
for each voltage level, against a stated naive baseline at each, because a single project-wide
target would not be interpretable.

**3. A substantial minority of real substations cannot be forecast better than by naive methods.**
Only 66–70% of customer-owned secondary substations in the Portuguese study beat a simple
"same time yesterday" forecast. Those are single-customer sites — one large building or industrial
process — where load is driven by decisions no weather model can see. Reporting the *fraction of
substations that beat a naive baseline*, alongside average error, is the honest way to present
results, and we intend to do so.

**4. Standard accuracy measures quietly reward forecasts that are useless for flexibility.** A
forecast that predicts the right peak at the wrong time is penalised twice — once for the peak that
did not happen, once for the peak it missed. A flat, featureless forecast avoids both penalties. So
conventional error measures systematically favour smooth forecasts over peaky ones, which is exactly
backwards for procuring flexibility against a capacity limit. The authors of the largest deployed
substation-forecasting study chose a peak-aware error measure to score their substation models for
precisely this reason. This finding is the strongest
argument in the review for the tail and exceedance metrics Flexpectation is building.

**5. Stated uncertainty can be badly wrong, and a decision metric alone will not reveal it.** In the
German low-voltage study, on the variant of their overload metric scored at each model's 95th
percentile, the top two models had 90% ranges containing the true value only 62% and 58% of the
time. A model that understates its uncertainty raises fewer false alarms and therefore scores well
on a threshold-crossing test — while being exactly the model an operator should not trust near a
capacity limit. On the point-forecast variant of the same metric the best-calibrated model won
instead, so this is a warning about one scoring choice rather than about decision metrics in
general. Any claim about a probabilistic forecast must be accompanied by evidence that its stated
ranges are honest.

**6. Weather forecasts are barely used at low voltage, and weather ensembles not at all.** Of the
221 low-voltage forecasting papers [Haben et al.](https://arxiv.org/abs/2106.00006) reviewed to
2020, three used a weather forecast and none used an ensemble of them. Northern Powergrid's
published secondary-substation results use no weather at all, because the forecast archive they had
access to did not extend far enough ahead. This is the clearest open gap in the field and the one
Flexpectation is best placed to close.

### An open question this review cannot settle

Finding 1 — that sophisticated methods beat simple ones by less than expected — has two possible
explanations, and nothing we read separates them.

The first is that substation demand has a low ceiling. Load at a single substation is driven by the
decisions of a few hundred customers, much of which is genuinely unpredictable. If that is the whole
story, a simple method already gets close to the ceiling, a sophisticated one has nowhere left to
go, and the modest gains reported across the literature are the correct answer.

The second is that the ceiling has not yet been tested. What this literature calls an "advanced"
method is usually a carefully-constructed statistical model, or an established machine-learning
library applied to a standard feature set. Both are sensible choices, and neither is what a
sustained modern machine-learning effort looks like. AlphaFold reached its result through several
years of a large team running a great many experiments against one fixed, public benchmark. That
route is open to energy forecasting in principle, but it is rare in practice, for structural reasons
rather than any failing of the researchers: a forecasting paper is typically written by a small team
over months rather than years, tests a handful of configurations, and reports on a dataset nobody
else uses — so the field has never accumulated the thousands of comparable attempts that the protein
folding community had before AlphaFold.

This is a hypothesis, and we hold it loosely; the first explanation may well be the right one.
Flexpectation is not resourced like a DeepMind effort either. What it is resourced to do is run
experiments cheaply and in volume against a fixed benchmark, which is the part that matters here,
and that makes the question testable. If the ceiling is real, sustained experimentation will
converge quickly on a small gain over a naive forecast and then stop improving, and we will report
that plainly. If the ceiling has simply not been tested, improvements should keep arriving well past
the point at which a smaller effort would have concluded there were none left. Either answer is
worth publishing, and the second would be worth more to the industry than to this project alone.

## A concurrent UK programme: Northern Powergrid's Artificial Forecasting

One concurrent project matters more than any paper here. Artificial Forecasting is an Ofgem
Strategic Innovation Fund programme, with about £3.9m of grant across its three phases, run by
Northern Powergrid with Faculty, EV.energy and Oaktree
Power, across three phases, with the final Beta phase running to February 2027. Its deliverables are
published openly on the ENA Smarter Networks Portal. It is doing much of what Flexpectation does, at
both primary and secondary substations, and at the time of writing it is further ahead.

**What Artificial Forecasting has achieved.** A forecasting service for primary substations is
deployed, has passed the network's architecture review board, data governance and information
security checks for its current deployment, and was used operationally by Northern Powergrid's
System Forecasting team through the Winter 2025-26 flexibility procurement cycle to support
week-ahead dispatch decisions. It produces half-hourly probabilistic forecasts with
5th-to-95th-percentile bands, flags forecast exceedances of firm capacity, and is benchmarked
against the network's existing growth-based and persistence methods. Performance did not materially
degrade across an 11-day horizon on average. Their own value case puts whole-life net present value
at around £60m for one network, or £250m if three further networks adopt it, driven mainly by a 3%
reduction in reinforcement spend in the current price-control period rising to 6% in the next, and a
25% improvement in the cost-effectiveness of contracted flexibility. Those are estimates made when
the project was proposed, which the Beta phase has not yet revalidated against measured savings.

**Why Artificial Forecasting matters for Flexpectation.** It is independent evidence that short-term
substation forecasting is operationally useful, that networks will actually change their procurement
process around it, and that the benefits case is credible. It also sets a public bar for what
"working" looks like.

Flexpectation is not repeating that work. Artificial Forecasting has shown that operational
substation forecasting is useful and has built the deployment path; the questions Flexpectation
takes on are the ones it has not needed to answer — ensemble-driven uncertainty, the far upper tail,
generation nobody meters, and using switching-affected history rather than discarding it. Their core
intellectual property is available royalty-free, and we would rather build on it than rebuild it.

Because Artificial Forecasting is public, operational and benchmarked against a real incumbent
method, it is also the clearest available picture of where the field currently stops — more
informative on that question than any single paper, because a deployed system has to answer
questions a paper can leave open.

## Four studies worth a closer look

### [Bouman et al. 2024](https://arxiv.org/abs/2405.16164) — switch-event detection at a Dutch network operator

The most directly useful paper in the review, because it takes on half of a problem that is
explicitly in this project's scope. Working with Alliander on 180 primary substations at 15-minute
resolution over roughly a year, the authors detect the step changes caused when a cable fault or
planned maintenance reroutes part of a subgrid to a different substation — a step up at one, a step
down at the other. They note the duration range explicitly: from a few minutes to several months.

Four things transfer:

- **They detect on a residual, not on the load itself.** Alliander maintains an independent bottom-up
  estimate of each substation's load, reconstructed from customer telemetry and modelled profiles.
  They fit and rescale that estimate to the measured series, then hunt for step changes in the
  *difference* between the two. Normal daily and seasonal variation largely cancels, leaving a much
  cleaner signal. We have no bottom-up estimate, but we have our own forecast, which can play the
  same role.
- **They recover a missing sign, which solves a known problem in NGED's trial area.** Some Alliander
  substations measure only absolute current, so reverse flow appears as a rise rather than a sign
  change — the identical defect at ten of NGED's metered sites. Because their reference estimate is
  built from signed measurements, they take the sign from it. Any independently-signed reference
  would do.
- **They stratify evaluation by event length**, into four buckets from "15 minutes to 6 hours" up to
  "42 days or longer", because short anomalies are frequent and long switch events are rare but cover
  most of the affected data. Pooling the two would let the long events dominate any metric.
- **When their bottom-up estimate fails, the cause is usually wrong topology data**, not a bad
  algorithm — a warning about the network records that any disaggregation work depends on.

They achieve roughly 90% of resulting load estimates within a 10% error margin, using deliberately
interpretable methods because reinforcement decisions rest on the output. Their purpose is capacity
planning rather than forecasting, so feeding detected events forward into a demand forecast remains
open — which is the part Flexpectation would contribute. One topology-detection paper cites a survey
of utility experts reporting five to ten switching actions at an urban distribution substation; the
survey's time base is not stated, so we have no dependable external estimate of the event rate.

### [Pinheiro, Madeira & Francisco 2023](https://doi.org/10.1016/j.apenergy.2022.120493) — the closest analogue to Flexpectation

A production forecasting system at a Portuguese distribution network operator, covering 96,989
secondary substations day-ahead, using real weather forecasts with a realistic 7–8 hour delay. It is
the only study in the review operating at our intended scale in a live setting, and its findings are
sobering in a useful way.

At system level the results are excellent: 42–47% better than the standard reference benchmark. At
substation level they are far more modest, and the paper is candid about it — the model beats a
simple "same time yesterday" forecast on 82–87% of network-owned substations but only 66–70% of
customer-owned ones. Findings 1, 3 and 4 above all rest on this study.

Two consequences for Flexpectation: reporting should include the fraction of substations beating a
naive forecast, not just a pooled average; and expectations for single-customer sites should be set
low from the outset.

### [Gilbert, Browell & Stephen 2023](https://arxiv.org/abs/2206.11745) — why an annual average hides what happens at peak

Gilbert et al. forecast load at four levels of a hypothetical GB distribution hierarchy, from a
primary substation down to individual households, and combine a conventional half-hourly forecast
with a bespoke daily-peak forecast.

Averaged over every period of the year, that combination gains 0.0–0.4% over the conventional
forecast alone — indistinguishable from nothing, and a result that would ordinarily end the
investigation. Restricted to the periods containing the daily peak, the same comparison gives 5.7%
at the primary substation, 9.0% at secondary, 8.2% at feeder level and 6.0% at household level. The
technique was always worth having, and we know that only because the authors reported both numbers.

A second finding is more uncomfortable. At household level during peak periods, *both* of their
models are worse than a trivial benchmark based only on the time of day. And the ability to predict
peak *timing* collapses as you disaggregate: better than 20% above the long-run seasonal average at
the primary substation, essentially zero at four of the feeders.

The peak-versus-average gap and the collapse in peak timing together justify the evaluation design
Flexpectation is building — reporting tail and exceedance metrics alongside average error, rather
than treating average error as the headline.

### [Kaas et al. 2026](https://arxiv.org/abs/2607.01966) — uncertainty bands that are narrower than the truth

Two papers from the same institute, on the same 200 low-voltage feeders, published a fortnight
apart, name different models as best — and inside the second of them, average error and an
overload-decision metric name different winners again. Neither disagreement is a mistake. The two
papers test different sets of models at different time resolutions, and the two metrics answer
different questions. Between them they are the clearest available demonstration of the problem set
out at the start of this review: the choice of metric, dataset and horizon decides who wins.

Their uncertainty result is covered in finding 5 above. The lesson worth repeating is that a
confident-looking forecast is not the same as a well-calibrated one, and only a direct check of how
often reality falls inside the stated range will tell the two apart.

## Gaps we did not find addressed, and where Flexpectation fits

Seven things we did not find addressed in the work reviewed above, academic or operational, and all
seven bear directly on what this project is required to deliver. This review was selective, and the
exclusions at the end say what it skipped, so these are gaps in what we read rather than proof that
nobody has tried them. Nor are they criticisms: most are questions a research paper has no reason to
ask and a deployed forecasting service has not yet needed to answer, and in several cases the
authors and engineers concerned name the gap themselves.

1. **Weather ensembles as the source of uncertainty.** Finding 6 gives the literature count.
   Deployed practice is at a similar point: Artificial Forecasting's published Alpha work used a
   commercial point forecast at three cities — Darlington, Leeds and Hull — for its whole network
   area, and no weather at all in its secondary-substation results. Driving forecast uncertainty
   from the spread of 51 physically plausible weather outcomes, over a 14-day horizon with users
   acting one to ten days ahead, is unexplored at distribution level. [Taylor and
   Buizza](https://doi.org/10.1109/TPWRS.2002.800906) did it for national demand in 2002, with the
   same 51 members and the same one- to ten-day range; nobody appears to have brought it down to a
   substation.
2. **The upper tail, not the middle.** NGED's question is "how likely is load to cross this limit?",
   not "what is the most likely load?". Almost everything in this review optimises average accuracy.
   The probabilistic work that does exist reports the 5th to the 95th percentile at best, and the
   two largest competitions stop at the 90th; NGED's delivery quantiles are deliberately weighted
   beyond that, because the decision being made is about the worst plausible case rather than the
   likely one.
3. **A decision metric that holds risk constant, priced in pounds, at distribution level.** Two
   pieces of this exist already. Browell and Fasiolo fix a risk appetite, compute the reserve volume
   each forecast would need to hold it, and compare — which is the harder half of the job, done at
   transmission level. Artificial Forecasting's Alpha work calculates the extra flexibility volume
   that forecast error would make a network procure: 20,536 kWh implied by a risk-aware forecast
   against 5,495 kWh actually needed, over two eight-day windows at one near-capacity substation.
   What nobody combines is a network threshold, a calibrated risk target and a price. Faculty's
   appendix prices a safety margin against under-predicting periods already flagged; they name the
   exceedance a forecast misses entirely as an open item themselves.
4. **Keeping switching-contaminated history usable.** Detection has at least been demonstrated at a
   real network operator: Bouman et al. segment 180 Dutch primary substations well enough that the
   resulting annual maximum and minimum load estimates land within 10% about 90% of the time, though
   their per-timepoint detection scores stay low. Everyone then removes the affected data rather
   than using it. The reference low-voltage benchmark dataset
   ([FeederBW](https://arxiv.org/abs/2602.03521)) excludes feeders with topology changes outright;
   Artificial Forecasting detects the steps and rescales them out, on the stated grounds that steps
   of that size "cannot be directly handled even by powerful nonlinear models"; Gilbert et al. name
   adaptive handling of structural breaks as future work. All three are defensible engineering. What
   nobody has tried is keeping the history. The target is not predicting when a cable will fault —
   nobody can do that. It is to feed recent observations to the model as residuals against a
   switching-independent baseline — the difference between what was measured and what a model that
   ignores topology expected — so a reading taken while the network was abnormally configured still
   carries information. Later, at the scale of NGED's full network, the aim is to reconstruct the
   demand each substation would have metered under its normal running arrangement. A negative result
   here would still be valuable: evidence that switching cannot be recovered from power data alone
   would support the case for extracting switching labels from operational systems instead.
5. **Separating out generation that nobody meters.** Where demand and generation are separated at
   all, the generation is metered: Artificial Forecasting models gross demand and customer export
   independently at primary substations, which is more than any paper here does. The unmetered solar
   and wind — the rooftop panels and small turbines that appear only as a dent in a substation's net
   flow — have to be estimated from that net flow. The one benchmark we found at this level
   estimates installed capacity and stops there; nothing in this review forecasts it. The
   requirement here is at primary substation level.
6. **Forecasting the network as a network.** No result in this review forecasts a substation using
   the network's connectivity. Where information is shared across substations at all — Artificial
   Forecasting pools model parameters across six load-profile clusters — the grouping is statistical
   rather than topological, and Gilbert et al. forecast four levels of a hierarchy separately before
   naming exploitation of that hierarchy as future work. NGED hold a map of which substations and
   metered generators connect to which. That makes it possible to forecast a bulk supply point both
   directly and by summing everything beneath it, and to treat the disagreement between the two
   answers as a check on both. It is information that already exists and is unused.
7. **Tracking how much generation is actually available.** Turbines go out for repair, inverters
   degrade and sites are curtailed. A substation whose 20 MW wind farm has been limited to 14 MW for
   a month is, for forecasting purposes, a different substation, and a model trained on nameplate
   ratings cannot see the difference. Artificial Forecasting gets closest: its Alpha work fed NESO's
   national generator-availability signal into its models and found it "almost universally
   substantially improved results" at wind-connected primary substations, and it separately tried,
   without success, a feature tracking connected generation capacity over time. What we found
   nowhere is a per-substation effective capacity, estimated from that substation's own data and
   tracked as it changes. The requirement is for effective capacity to be tracked over time and,
   optionally, combined with the forecast to give a "prevailing conditions" view; we intend to use
   it to normalise each metered generator's series before training.

**How ambitious Flexpectation's research plan is, and the risk that ambition carries.** The seven
items above are not a shortlist to choose from. The plan is to attempt all of them alongside the
core forecast, across several families of model: a heavily-tuned version of the gradient-boosting
approach that wins most tabular forecasting competitions, weather and time encoders pre-trained on
large datasets, models that use the connectivity map explicitly, and differentiable physics —
building the known behaviour of a solar panel or a wind turbine directly into the model so that it
has to learn only what the physics cannot supply. Physics-informed models for solar generation
exist; we found none applied to substation demand forecasting, though the exclusions section below
explains why we hold that finding loosely.

Attempting all seven means running on the order of hundreds of machine-learning experiments a month,
and that is possible only because of engineering already done. Most of the effort to date has gone
into a machine-learning operations framework built to current industry best practice, whose purpose
is to make one more experiment nearly free: every experiment is fully specified by a config file,
runs through the same pipeline that serves production rather than a separate research copy of it, is
tracked automatically from raw data through to result, and lands in one comparable metrics store.
That machinery exists and works today; the leaderboard view over it is still being built. The plan
is affordable because the marginal experiment is cheap, not because the team is large.

Flexpectation's plan is riskier than a narrower one would be, and that is worth saying plainly.
Artificial Forecasting chose a focused agenda and delivered it into live operational use, which is
the right way to get a service running and is why its results are the firmest evidence in this
review. Flexpectation is funded to do research rather than deployment, and is attempting a wider set
of open questions on a smaller budget. That is a statement about how much of Flexpectation's plan is
still unproven, not a comparison between the two projects. Several of the directions listed above
will not work — that is what makes them research directions rather than engineering tasks — and the
honest expectation is that some deliver clearly, some produce a negative result worth publishing,
and some are abandoned. Two things make that acceptable. Each item is independently useful, so one
failing does not strand the others: switching detection, capacity estimation and disaggregation each
improve the core forecast on their own terms. And the point of being able to run a hundred
experiments is that most of them are allowed to fail.

There is one further contribution, which is about method rather than results. The central problem
identified at the start of this review — that published results cannot be compared — is one this
project is well placed to help with. Others have started already: HEFTCom and Energy-Arena both
compare methods on common data with a common metric, and Energy-Arena keeps a live public
leaderboard. Neither covers distribution-substation load, which is the level NGED acts at, so we
intend to follow their protocols where they apply rather than invent our own. Part of the groundwork
is public too: NGED publish substation data on their
[Connected Data Portal](https://connecteddata.nationalgrid.co.uk/). That alone puts this work ahead
of most of the literature, where only 52 of the 221 low-voltage papers reviewed used any open
dataset at all. What this project can commit to unilaterally is the rest of the apparatus — a
published evaluation protocol, the metric definitions and the code that computes them, and an open
leaderboard carrying every experiment we run. Publishing the telemetry behind those experiments in
turn would make the results reproducible by anyone, which is still rare in the substation
literature. Artificial Forecasting is moving the same way, with substation-level historical
forecasts and model-performance metrics designed into its Open Data Portal release, and a shared
evaluation protocol between two GB networks would be worth more than either alone.

## What we deliberately left out

A selective review is only trustworthy if it says what it excluded.

**Behind-the-meter solar disaggregation** is a large and active field, mostly working on US
smart-meter data at individual customer level. We excluded it as a body because it operates at a
different level of aggregation from ours; we have kept anchor citations for when the disaggregation
stretch goal becomes active.

**Network topology detection** from high-resolution synchrophasor measurements is well developed,
but the measurements it needs are not available to this project.

**General concept-drift detection** was excluded because most of it addresses gradual drift and
model adaptation, whereas our problem is a discrete step change with a known physical cause. The
abrupt-drift and change-point strand of that literature is closer to our problem, and we intend to
read it properly before the switching work begins.

**Differentiable physics applied to substation demand forecasting** produced no strong result.
There is substantial work on physics-informed neural networks for power systems, including models
that map weather to solar output, but we found none applied to forecasting demand at a substation.
Either our search terms were wrong or the intersection is genuinely thin, and we would welcome a
second opinion.

**The bulk of the low-voltage forecasting literature** is covered through the Haben et al. review of
221 papers rather than read individually, which is the appropriate level of detail for work that
closes in 2020. We have not systematically covered low-voltage work published since; where a
specific question arises we go back to individual papers rather than relying on the review.

Finally, two sources we would have liked to use properly. **GEFCom2017** remains paywalled and
everything we know of it is second-hand; nothing in this section rests on it. A **2026 benchmark on
estimating installed solar capacity at low-voltage substations** was available only as an abstract,
and should be read in full before the capacity-estimation work begins.
