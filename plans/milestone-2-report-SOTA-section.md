# The current state of the art in energy forecasting

Before we discuss the literature, there is a very important caveat to admit up-front: In 2026, no honest review of the energy forecasting literature can claim to reveal the canonical "state of the art"! That is because (almost) all energy forecasting papers measure performance in different ways, against different datasets. It's like an international football tournament where every team plays by different rules, with different size goals.

Energy forecasting researchers have done great work over the years. But, unfortunately, the literature does not tell us how those approaches compare against each other, especially in messy "real world" energy forecasting scenarios. This isn't anyone's fault; it's a systemic failure. And the industry is already aware of this problem, and people are trying to fix it. But, at the time of writing, the literature cannot yet tell us the current state of the art solutions for the problems that NGED cares about.

Having said that, there have been some valiant attempts to compare multiple forecasting approaches against the same dataset. But none of these attempts directly address the main challenges relevant to Flexpectation. Before we discuss those attempts, we must emphasise two reasons for optimism:

Firstly, whilst we might not know exactly which algorithms provide the best forecasting performance, we absolutely do know how to develop a state of the art forecasting algorithm. There's no magic. Machine learning is an empirical science: success is largely a function of how many ideas you can test. So our task is "simply" to run hundreds of ML experiments, and then measure performance against the same dataset, using the same performance metrics. This, in turn, requires us to design and build a framework that makes it easy to run hundreds of ML experiments per month. At the time of writing, we have implemented the first version of this framework, and we will continue to evolve the framework over the course of the project.

Secondly - and perhaps most importantly - the fact that the industry doesn't yet know the state of the art is a huge opportunity for the Flexpectation project: We are in a very privileged position where we can try hundreds of ideas, and test the best ideas in the real world. We have a fantastic opportunity to make a significant contribution to the energy forecasting industry by publishing our "leaderboards of ML experiments", and hence help the industry as a whole to better understand how multiple approaches perform.

## What we read

We read ten papers in full, drew on two more that were only partly available to us, and read in full
the published deliverables of one concurrent UK network project. The bar for inclusion was
deliberately high: a paper had to bear on a decision Flexpectation actually faces
*and* change something we believed. A great deal of good work was left out on that basis, and the
last section of this review says what and why.

Two sources are weaker than the rest and are flagged wherever they appear. The GEFCom2017
competition paper remains paywalled, so everything we know about it is second-hand and should be
treated as unverified. A 2026 benchmark on estimating installed solar capacity at low-voltage
substations was available to us only as an abstract.

## The best published results, and why they cannot be compared

The table below gives the best reported result from each source. Every entry is genuinely
best-in-class for the problem its authors set themselves. Not one of them can be compared with any
other.

| Source | What they forecast | Horizon | Best result, and what it beat | Weather |
|---|---|---|---|---|
| Browell et al. 2025 (HEFTCom competition) | Combined wind and solar output, GB | Day-ahead | Winning team scored 22.18 MWh average error against the organisers' benchmark of 53.58. Revenue of £88.9m against a £105.2m perfect-decision ceiling | Real forecasts, live |
| Kaas et al. 2026 | Net load, 200 low-voltage feeders, Germany | 4 days | A general-purpose "foundation" model that had never seen the data beat every purpose-trained model on average error, 3.839 kW against 4.184 kW | Actual weather, after the fact |
| Hertel et al. 2026 | Load at three grid levels, Germany and Portugal | 4 days | Best model beat a naive forecast by 59.6% at transmission level, 42.3% at low-voltage feeders, 23.3% at individual customers | Actual weather, after the fact |
| Kleinebrahm et al. 2026 (Energy-Arena) | Live public leaderboard: prices, load, wind, solar | Day-ahead | No single winner — a continuously updated ranking, which is the point of it | Real forecasts, live |
| Hong, Xie & Black 2019 (GEFCom2017) | Hierarchical load, New England | 2–6 weeks | No score available in anything we could read | Real forecasts, live |
| Shukla & Hong 2024 (BigDEAL competition) | Peak load, three US utilities | Rolling months | Winning scores not published. The transferable finding is that rankings on peak size, peak timing and peak shape barely correlate | Mixed |
| Haben et al. 2021 | Review of 221 low-voltage forecasting papers | — | Of 221 papers, **3** used a weather *forecast* and **none** used a weather ensemble | — |
| Browell & Fasiolo 2021 | Regional net load, GB | Day-ahead | Their tail model needed **24.6%** less upward reserve than the standard alternative at the same risk level. Adding wind and irradiance cut error 40% overall — 10% in London, 60% in North Scotland | Real forecasts |
| Pinheiro et al. 2023 | Load at 96,989 Portuguese secondary substations | Day-ahead | 42–47% better than the reference benchmark at system level. **At substation level, beat a naive forecast on only 83–87% of network-owned and 66–70% of customer-owned sites** | Real forecasts, 7–8 h old |
| Gilbert et al. 2023 | Load across a four-level GB hierarchy | Day-ahead | Combining forecasts gained **0.0–0.4% averaged over all periods**, but **5.7–9.0% when restricted to peaks** | None at all |
| Bouman et al. 2024 | Switch-event and anomaly detection, 180 Dutch primary substations | Not a forecast | ~90% of resulting load estimates within a 10% error margin | None |
| **Artificial Forecasting (Northern Powergrid)** | **Demand and export at GB primary and secondary substations** | **Day- to week-ahead** | **~8% lower error than the network's existing methods; 83% of the highest-demand periods captured inside its uncertainty band; better at 8 of 8 near-capacity substations** | **Real forecasts at primary; none at secondary** |

### Three things make a result look good with no forecasting skill behind it

**The level of aggregation.** Hertel et al. ran the same models against the same naive benchmark at
three levels of the grid and beat it by 59.6% at transmission and 23.3% at individual customers. The
model did not get worse; the problem got harder. A headline percentage therefore says more about
where it was measured than about the method. This is the single most important thing for NGED to
take from this review, because it sets what we should expect at secondary substations.

**Weather known after the event.** Two of the studies above feed their models the weather that
actually happened rather than the weather that was forecast. That removes the error which dominates
forecasts more than a day or two ahead — precisely the range NGED acts on. Results obtained this way
are upper bounds, not achievable performance.

**Averaging over all periods.** Gilbert et al. found that combining forecasts improved accuracy by
0.0–0.4% averaged across every half-hour of the year, which reads as "no effect worth having".
Restricted to the periods containing the daily peak, the same comparison gives 5.7–9.0%. A number
averaged over 17,520 half-hours is dominated by the quiet ones, and the quiet ones are not why NGED
buys flexibility.

### Which published numbers do transfer

Only two kinds. **Ratios against a stated baseline on a stated population** — which is why Pinheiro
et al.'s finding that only 66–70% of customer-owned substations beat a naive forecast is the most
useful figure in the table. And **errors normalised by something physical**, such as a substation's
firm capacity or transformer rating, rather than by the load that happened to occur. Absolute errors
in kW or MW transfer to nothing, and none of the absolute figures above should be read as a target
for this project.

## What the literature does agree on

Six findings recur across independent studies, and we regard them as robust.

**1. Sophisticated methods beat simple ones by much less than expected.** In a live system covering
96,989 substations, a carefully tuned modern machine-learning model lost to a simpler, more
interpretable one on the same inputs. Northern Powergrid tested the same class of model for customer
export and rejected it: it "helped some substations but harmed others". Across 729 secondary
substations, a neural network beat an average-of-the-last-four-weeks rule by about one percentage
point, and lost to it outright on the substations with the worst data. The winning entry in the
HEFTCom competition tuned a single setting and left everything else at its default. This is
reassuring rather than disappointing: it means interpretable models remain competitive, and that
effort is better spent on data quality and on evaluation than on model complexity.

**2. The more local the substation, the less any model can buy you.** This follows from finding 1 and
from the aggregation effect above. Demand at a grid supply point is smooth and largely predictable
from calendar and weather; demand at a single secondary substation is dominated by the behaviour of a
handful of customers. Expectations for forecast accuracy must be set per voltage level, and a single
project-wide accuracy target would be meaningless.

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
backwards for procuring flexibility against a capacity limit. The Portuguese network operator adopted
a peak-aware error measure in production for precisely this reason. This finding is the strongest
argument in the review for the tail and exceedance metrics Flexpectation is building.

**5. Stated uncertainty is frequently wrong, and the usual scoring will not catch it.** In the German
low-voltage study, the two models that won the operational decision metric had uncertainty bands
containing the true value only 62% and 58% of the time, against a nominal 90%. A model that
systematically understates uncertainty raises fewer false alarms and therefore scores well on a
threshold-crossing test — while being exactly the model an operator should not trust near a capacity
limit. Any claim about a probabilistic forecast must be accompanied by evidence that its stated
ranges are honest.

**6. Weather forecasts are barely used at low voltage, and weather ensembles not at all.** Of 221
low-voltage forecasting papers reviewed to 2020, three used a weather forecast and none used an
ensemble of them. Northern Powergrid's published secondary-substation results use no weather at all,
because the forecast archive they had access to did not extend far enough ahead. This is the clearest
open gap in the field and the one Flexpectation is best placed to close.

## A concurrent UK programme: Northern Powergrid's Artificial Forecasting

One concurrent project matters more than any paper here. Artificial Forecasting is a £3.9m Ofgem
Strategic Innovation Fund programme run by Northern Powergrid with Faculty, EV.energy and Oaktree
Power, across three phases, with the final Beta phase running to February 2027. Its deliverables are
published openly on the ENA Smarter Networks Portal. It is doing much of what Flexpectation does, at
both primary and secondary substations, and at the time of writing it is further ahead.

**What Artificial Forecasting has achieved.** A forecasting service for primary substations is
deployed, has passed the network's architecture, data governance and information security review,
and was used operationally by Northern Powergrid's System Forecasting team through the Winter
2025-26 flexibility procurement cycle to support week-ahead dispatch decisions. It produces
half-hourly probabilistic forecasts with 5th-to-95th-percentile bands, flags forecast exceedances of
firm capacity, and is benchmarked against the network's existing growth-based and persistence
methods. Performance did not materially degrade across an 11-day horizon. Their own value case puts
whole-life benefits at around £60m for one network, or £250m if three further networks adopt it,
driven mainly by a 3% reduction in reinforcement spend rising to 6%, and a 25% improvement in the
cost-effectiveness of contracted flexibility.

**Why Artificial Forecasting matters for Flexpectation.** It is independent evidence that short-term
substation forecasting is operationally useful, that networks will actually change their procurement
process around it, and that the benefits case is credible. It also sets a public bar for what
"working" looks like, and their core intellectual property is available to other networks
royalty-free.

Because Artificial Forecasting is public, operational and benchmarked against a real incumbent
method, it is also the clearest available picture of where the field currently stops — more
informative on that question than any single paper, because a deployed system has to answer
questions a paper can leave open. The section below on what nobody has done yet draws on it in
exactly the way it draws on the academic literature.

## Four studies worth a closer look

### Bouman et al. 2024 — switch-event detection at a Dutch network operator

The most directly useful paper in the review, because it solves half of a problem NGED has explicitly
asked us to work on. Working with Alliander on 180 primary substations at 15-minute resolution over
roughly a year, the authors detect the step changes caused when a cable fault or planned maintenance
reroutes part of a subgrid to a different substation — a step up at one, a step down at the other.
They note the duration range explicitly: from a few minutes to several months.

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
open — which is the part Flexpectation would contribute. A separate utility survey puts switching
actions at five to ten per urban distribution substation per year, the only external estimate of the
event rate we found.

### Pinheiro, Madeira & Francisco 2023 — the closest analogue to Flexpectation

A production forecasting system at a Portuguese distribution network operator, covering 96,989
secondary substations day-ahead, using real weather forecasts with a realistic 7–8 hour delay. It is
the only study in the review operating at our intended scale in a live setting, and its findings are
sobering in a useful way.

At system level the results are excellent: 42–47% better than the standard reference benchmark. At
substation level they are far more modest, and the paper is candid about it — the model beats a
simple "same time yesterday" forecast on 83–87% of network-owned substations but only 66–70% of
customer-owned ones. The authors also chose an interpretable model over a tuned modern
machine-learning alternative on the same inputs, on both accuracy and operational grounds, and
adopted a peak-aware error measure in production because standard measures rewarded forecasts that
were too smooth.

Two consequences for Flexpectation: our reporting should include the fraction of substations beating
a naive forecast, not just a pooled average; and our expectations for single-customer sites should
be set low from the outset.

### Gilbert, Browell & Stephen 2023 — why an annual average hides what happens at peak

Gilbert et al. forecast load at four levels of a hypothetical GB distribution hierarchy, from primary
substation down to individual household, and compares an advanced method against a simpler one.

Averaged over every period of the year, the advanced approach gains 0.0–0.4% — indistinguishable from
nothing, and a result that would ordinarily end the investigation. Restricted to the periods
containing the daily peak, the same comparison gives 5.7% at primary substations, 9.0% at secondary,
8.2% at feeder level and 6.0% at household level. The technique was always worth having; the
whole-year average concealed it.

A second finding is more uncomfortable. At household level during peak periods, *both* of their
models are worse than a trivial benchmark based only on the time of day. And the ability to predict
peak *timing* collapses as you disaggregate: better than 20% above seasonal climatology at primary
substations, essentially zero at four of the feeders.

The peak-versus-average gap and the collapse in peak timing together justify the evaluation design
Flexpectation has adopted — reporting tail and exceedance metrics alongside average error, rather
than treating average error as the headline.

### Kaas et al. 2026 — uncertainty bands that are narrower than the truth

Two papers from the same institute, on the same 200 low-voltage feeders, published a fortnight apart,
disagree about which model is best. One scores average error; the other scores a decision about
whether a feeder will exceed its limit. That disagreement is the clearest single demonstration of the
problem set out at the start of this section.

The more important finding is about uncertainty. On the operational decision metric, the two
best-performing models had stated 90% ranges that contained the true value only 62% and 58% of the
time. Because a model that understates its uncertainty raises fewer false alarms, it scores well on a
threshold-crossing test — while being precisely the model an operator should not rely on near a
capacity limit. A confident-looking forecast is not the same as a well-calibrated one, and only a
direct check of how often reality falls inside the stated range will tell the two apart.

## What nobody has done yet, and where Flexpectation fits

Seven things are not done anywhere in the work reviewed above, academic or operational, and all
seven are things NGED has asked this project to try. None of this is a criticism of the work
reviewed. Most of these are questions a research paper has no reason to ask and a deployed
forecasting service has not yet needed to answer, and in several cases the authors and engineers
concerned name the gap themselves.

1. **Weather ensembles as the source of uncertainty.** Three of the 221 low-voltage papers reviewed
   by Haben et al. used a weather forecast at all, and none used an ensemble of them. Deployed
   practice is at a similar point: Artificial Forecasting uses a commercial forecast at three point
   locations for a whole licence area, and no weather at all in its secondary-substation results.
   Driving forecast uncertainty from the spread of 51 physically plausible weather outcomes, over a
   14-day horizon with users acting one to ten days ahead, is genuinely unexplored.
2. **The upper tail, not the middle.** NGED's question is "how likely is load to cross this limit?",
   not "what is the most likely load?". Almost everything in this review optimises average accuracy.
   The probabilistic work that does exist, in the literature and in deployment alike, reports the 5th
   to 95th percentiles; NGED's delivery quantiles are deliberately weighted beyond that, because the
   decision being made is about the worst plausible case rather than the likely one.
3. **A decision metric calibrated to a risk level and priced in pounds.** Several studies score a
   decision rather than an error, and the furthest anyone has taken this is Artificial Forecasting's
   Alpha work, which calculates the extra flexibility volume that forecast error causes a network to
   procure — 20,536 kWh contracted against 5,495 kWh actually needed, at one substation. That is
   further than any paper here goes. What is still missing everywhere is holding risk constant when
   comparing models, converting the answer into pounds rather than kilowatt-hours, and costing the
   opposite error: the exceedance you fail to predict. Faculty name that last one as an open item
   themselves.
4. **Keeping switching-contaminated history usable.** The detection half of this problem is solved:
   Bouman et al. find the step changes reliably at 180 Dutch primary substations. Everyone then
   removes the affected data rather than using it. The reference low-voltage benchmark dataset
   excludes feeders with topology changes outright; Artificial Forecasting detects the steps and
   rescales them out, on the stated grounds that steps of that size "cannot be directly handled even
   by powerful nonlinear models"; Gilbert et al. name adaptive handling of structural breaks as
   future work. All three are defensible engineering. What nobody has tried is keeping the history.
   The target is not predicting when a cable will fault — nobody can do that. It is to feed recent
   observations to the model as residuals against a switching-independent baseline, so a reading
   taken while the network was abnormally configured still carries information, and then to
   reconstruct the demand each substation would have metered under its normal running arrangement.
   NGED has been clear that a negative result here is still valuable: evidence that switching cannot
   be recovered from power data alone would support the case for extracting switching labels from
   operational systems instead.
5. **Separating out generation that nobody meters.** Where demand and generation are separated at
   all, the generation is metered: Artificial Forecasting models gross demand and customer export
   independently at primary substations, which is more than any paper here does. The unmetered
   solar and wind — the rooftop panels and small turbines that appear only as a dent in a
   substation's net flow — has to be estimated from that net flow, and nothing in this review
   attempts it at substation scale. NGED want it at primary level and eventually below.
6. **Forecasting the network as a network.** Every result in this review treats each substation as an
   independent time series, including Gilbert et al.'s, who forecast four levels of a hierarchy
   separately and then name exploiting that hierarchy as future work. NGED hold a map of which
   substations and metered generators connect to which. That makes it possible to forecast a bulk
   supply point both directly and by summing everything beneath it, and to treat the disagreement
   between the two answers as a check on both. It is information that already exists and is unused.
7. **Tracking how much generation is actually available.** Turbines go out for repair, inverters
   degrade and sites are curtailed. A substation whose 20 MW wind farm has been limited to 14 MW for
   a month is, for forecasting purposes, a different substation, and a model trained on nameplate
   ratings cannot see the difference. NGED asked for this **effective capacity** to be tracked over
   time, used to normalise each series before training, and combined with the forecast to give a
   "prevailing conditions" view. We found no published substation-forecasting work that estimates it.

**How ambitious Flexpectation's research plan is, and the risk that ambition carries.** The seven
items above are not a shortlist to choose from. The plan is to attempt all of them alongside the
core forecast, across several families of model: a heavily-tuned version of the gradient-boosting
approach that wins most tabular forecasting competitions, weather and time encoders pre-trained on
large datasets, models that use the connectivity map explicitly, and differentiable physics —
building the known behaviour of a solar panel or a wind turbine directly into the model so that it
has to learn only what the physics cannot supply. We have found no prior application of that last
approach to energy forecasting, though the exclusions section below explains why we hold that
finding loosely.

Attempting all seven means running on the order of hundreds of machine-learning experiments a month,
and that is possible only because of engineering already done. Most of the effort to date has gone
into a machine-learning operations framework built to current industry best practice, whose purpose
is to make one more experiment nearly free: every experiment is fully specified by a config file,
runs through the same pipeline that serves production rather than a separate research copy of it, is
tracked automatically from raw data through to result, and lands on a single leaderboard where it
can be compared with every experiment that came before. That machinery exists and works today. The
research ambition is downstream of it — the plan is affordable because the marginal experiment is
cheap, not because the team is large.

Flexpectation's plan is riskier than a narrower one would be, and that is worth saying plainly.
Artificial Forecasting has taken a tightly-scoped research agenda and delivered it into live
operational use, with £3.9m of Strategic Innovation Fund money across its three phases against
Flexpectation's £841,733 of Network Innovation Allowance funding. Attempting considerably more
research on about a fifth of the budget is not a claim about value for money; it is a statement
about how much of Flexpectation's plan is still unproven. Several of the directions listed above
will not work — that is what makes them research directions rather than engineering tasks — and the
honest expectation is that some deliver clearly, some produce a negative result worth publishing,
and some are abandoned. Two things make that acceptable. Each item is independently useful, so one
failing does not strand the others: switching detection, capacity estimation and disaggregation each
improve the core forecast on their own terms. And the experimentation throughput is exactly what
turns a long list of uncertain ideas into a testable programme rather than a wish list, because the
value of being able to run a hundred experiments is that most of them are allowed to fail.

There is one further contribution, which is about method rather than results. The central problem
identified at the start of this section — that published results cannot be compared — is one this
project is unusually well placed to help fix. NGED intend to publish their substation data down to
primary level, which already puts this work ahead of most of the literature: only 52 of the 221
low-voltage papers reviewed used any open dataset at all. Combined with a published evaluation
protocol and an open leaderboard of every experiment we run, that would make Flexpectation's results
reproducible by others — which almost nothing in this review is. The honest caveat is that we still
choose our own metrics and report our own results, and customer generator data will remain private;
publishing the protocol and the leaderboard is what converts that from a weakness into a
contribution. Artificial Forecasting is already moving the same way, publishing its historical
accuracy alongside its forecasts, and a shared evaluation protocol between two GB networks would be
worth more than either alone.

## What we deliberately left out

A selective review is only trustworthy if it says what it excluded.

**Behind-the-meter solar disaggregation** is a large and active field, mostly working on US
smart-meter data at individual customer level. We excluded it as a body because it operates at a
different level of aggregation from ours; we have kept anchor citations for when the disaggregation
stretch goal becomes active.

**Network topology detection** from high-resolution synchrophasor measurements is well developed but
assumes instrumentation NGED does not have.

**General concept-drift detection** — a large literature on models adapting to gradually changing
conditions — was excluded because our problem is abrupt topology changes rather than gradual drift.

**Differentiable physics applied to energy forecasting** produced no strong result. There is
substantial work on physics-informed neural networks for power systems generally, but we found
nothing applying differentiable physical models to substation demand forecasting. Either our search
terms were wrong or the intersection is genuinely thin, and we would welcome a second opinion.

**The bulk of the low-voltage forecasting literature** is covered through the Haben et al. review of
221 papers rather than read individually, which is the appropriate level of detail for work that
predates the availability of ensemble weather forecasts at this scale.

Finally, two sources we would have liked to use properly. **GEFCom2017** remains paywalled and
everything we know of it is second-hand; nothing in this section rests on it. A **2026 benchmark on
estimating installed solar capacity at low-voltage substations** was available only as an abstract,
and should be read in full before the capacity-estimation work begins.
