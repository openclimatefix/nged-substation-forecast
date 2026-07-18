# Estimating Latent Demand Under Switching Events — Approach & Implementation Roadmap

**Scope.** How to estimate, for each NGED primary substation, its *latent demand under the normal running arrangement* — the demand that would be metered if the network were never reconfigured — given that the network is in fact reconfigured roughly 10% of the time by switching events. Background on what switching events are and why they are hard is at [**Switching Events**](../background/switching-events.md). This document defines the staged modelling plan (v0.6 → v2.5 → v2.6).

> **Status: 🔬 Research / 🚧 Planned.** Epic:
> [#151](https://github.com/openclimatefix/nged-substation-forecast/issues/151) (the v0.6
> detector). None of this is implemented yet. The v0.6 unsupervised
> statistical detector is the nearest-term piece (it feeds the
> [`substation_switching`](delivery-tables.md#table-5-substation_switching) table and the
> training-data mask); the v2.5 / v2.6 mixture models are later research. The post-v2.0 roadmap is
> not yet fully specified, so read "v2.5 / v2.6" as "some time after v2.0". See the
> [roadmap index](index.md) for status conventions and where this fits the overall plan. This is the
> **canonical** treatment of switching events — it supersedes an earlier "switching state-space
> model" sketch (see the retirement note in
> [Net-demand disaggregation](disaggregation.md#handling-abnormal-running-arrangements)).

---

## Part 1 — The graph is a data structure

A **graph** is just **nodes** connected by **edges**. Here the nodes are substations and the edges connect substations that can exchange load (i.e. that can be electrically joined by some switching operation). The graph is the natural way to encode the one fact a per-substation model is blind to: that a switching event is a *cross-substation* phenomenon — load leaving one node reappears at others, so the drop at A and the rises at B and C are the *same power moving*.

It is worth being explicit about what the graph is *not*: it is not a trained model, and nothing is learned along its edges. We use the graph purely as a **data structure** — a fixed map of "who can exchange load with whom" — and run simple, mostly closed-form operations over it. In every stage the graph is used the same humble way: to look up which substations are even *eligible* to exchange load, pruning an otherwise $N \times N$ problem down to each node's handful of real neighbours. The per-stage specifics — the cheap neighbour-subset search in v0.6, the sparsity pattern on the mixing weights in v2.5, and the typed nodes in v2.6 — are described with each version in Part 2 below.

In short: the graph tells us *who can connect to whom*; the simple statistics and the differentiable forward model do the rest. (The project-wide design boundary — the same graph-as-data-structure stance across the disaggregation engine, and what evidence would ever revisit it — is stated once, in [the disaggregation page](disaggregation.md#the-fusion-mechanism).)

> **A note on differentiable physics.** Only the v2.6 stage uses *differentiable physics*: physics-based forward models (e.g. irradiance → PV power) implemented so their latent parameters — capacity, panel orientation, and so on — can be recovered by gradient-based **inversion** (running the forward model backwards to fit observed power). The v0.6 detector and the v2.5 mixture model do not use it. The full treatment lives in [Differentiable Physics](../techniques/differentiable-physics.md), which is the single source of truth for that machinery; we do not re-derive it here.

---

## Part 2 — The staged roadmap

Version numbers are aligned with the main codebase: **v0.6**, then **v2.5**, then **v2.6**. Each stage states its motivation, what it adds, what it misses, and its trade-offs. (Although, note that we haven't fully specified the roadmap _after_ v2, so read "v2.x" as just meaning "some time after v2 is operational"). Escalation between stages must be justified by *measured residual structure at the previous stage*, not by anticipation — this keeps effort matched to demonstrated need.

### Overview of the ladder

```text
v0.6  ── Unsupervised statistical detector on power data.
  │       Flags/masks switching events; VALIDATED against the 32-series logs.
  │       Produces the detection sensitivity floor (a quantified, honest limit).
  │       In-version escalation (v0.6.1), only if sequential matching proves brittle:
  │       the joint edge-flow estimator (detection + attribution in one solve).
  │
v2.5  ── Magnitude-only mixture model. The workhorse.
  │       Reconstructs latent NRA demand per substation. Fully unsupervised.
  │       Faithful to "arbitrary continuous slice, multi-donor" transfer.
  │
v2.6  ── Type-resolved mixture with differentiable PV/wind/demand modules.
          Lets a switching event move proportionally more PV than load.
```

---

### v0.6 — Unsupervised statistical switching-event detector

Issues: [#117](https://github.com/openclimatefix/nged-substation-forecast/issues/117),
[#118](https://github.com/openclimatefix/nged-substation-forecast/issues/118)

**Goal.** Flag periods of abnormal running arrangement using simple statistics on the power time series. **No neural networks, no differentiable physics, no latent-variable inference, no switching-log inputs.**

**Motivation.** Before any reconstruction model, we need to (a) flag/mask switching-affected periods so they stop poisoning forecasting training data; (b) produce an evaluation set to validate heavier models later; and (c) — critically — *quantify how well switching events can be detected from power data at all*, since at scale that is the only signal available.

**Method — three stages, in order:**

#### Stage 1. **Per-series weather/calendar baseline, then detect changepoints on the residual.**

For each substation, form an expected-power baseline that is a function of **exogenous, switching-independent covariates only** — temperature, solar irradiance, recent weather, time-of-day, day-of-week, holidays — fitted across a long history (e.g. an XGBoost or GAM regression). Take the residual (observed − expected). A switching event then shows up as a *sustained level shift* in this residual — not a spike, not a slope. Detect level shifts with a standard mean-shift changepoint method (PELT or binary segmentation with an L2 cost, or CUSUM). **Output:** candidate step times and magnitudes per substation.

**Why the baseline must be weather/calendar-based and *not* a lagged-power baseline.** A tempting cheap baseline is "same half-hour last week." It is unsound here, and disqualified. If last week sat in a switching event and this week is normal (or vice versa), the residual shows a step of the same magnitude and shape as a real event — but with the **sign reversed**, because the contamination is in the *reference*, not the observation. Stage 2's balance attribution would then hunt for donor rises coincident with a source drop that is a baseline artifact, manufacturing phantom events and mis-attributing them. Worse, because switching events can persist for days to months, a lag can land *inside the same ongoing event*, so there is no step at all and a real event is masked entirely. Weather and clock time are unaffected by network topology, so a baseline built only from them cannot be contaminated by switching state — the residual then isolates "power the weather and clock don't explain," which is exactly where a topology change appears, with no comparison period to poison.

**Baseline implementation: reuse the existing XGBoost forecaster with no lag features.** The
production forecaster already consumes most of the covariates the baseline needs — NWP weather,
time-of-day, day-of-week (holiday flags are a
[planned quick win](xgboost-improvements.md#2-uk-holiday-and-calendar-features)) — through the
existing feature pipeline. Configuring it with **no power-lag features** yields the
switching-independent baseline (a lag feature would smuggle the lagged-power contamination above
back in through the side door). Fit with a **quantile (median) objective**, which doubles as the
robust loss required below; fitting additional quantiles (e.g. 10%/90%) gives a per-series,
per-time spread estimate for free — used by the normalisation step below. That quantile fit is
the one piece of genuinely new machinery here: the forecaster has no quantile-objective support
today, so the baseline depends on the
[quantile-objective model family](metrics-and-leaderboard.md#delivering-the-probabilistic-metrics)
landing first.

**Three residual-contamination routes to handle:**

- *Training-set contamination.* If the baseline is fitted on history that itself contains switching events, the fit is biased toward those contaminated periods. Fit **robustly** (quantile or Huber loss), or iteratively: fit → flag large residuals as candidate events → refit excluding them. Because events occupy only ~10% of the time, a robust fit recovers the NRA relationship and the events fall out as residuals. This closes a virtuous loop with the detector itself — detected events feed back to clean the baseline's training data.
- *Persistent events.* A months-long ARA appears as a residual level shift that *stays* shifted, not a transient. The changepoint detector handles this (it catches the onset step), but the baseline must **not** be allowed to slowly adapt and treat the new level as normal. Keep the baseline static (weather/calendar-driven only) over the detection window so a sustained ARA remains visible as a sustained residual offset.
- *Seasonal-maintenance confounding.* Planned switching is not uniform through the year — outages
  cluster in maintenance seasons. A flexible time-of-year covariate fitted on contaminated history
  can therefore absorb systematic ARA effects into "seasonality", and the robust loss does *not*
  fix this, because the contamination is locally dense within the season even though it is only
  ~10% overall. Keep seasonal terms low-flexibility, prefer multiple years of history, and make
  sure the fit → flag → refit loop removes flagged periods from the seasonal fit too.

**Changepoint detection must respect the residual's real statistics.** Textbook mean-shift
detectors (PELT/BinSeg with an L2 cost, CUSUM) assume roughly independent noise with constant
variance. Baseline residuals violate both assumptions. They are **autocorrelated**: an NWP error
or an unmodelled local effect persists for hours to days, and a naive detector fires on that slow
wander as if it were a step — the classic failure mode of changepoint-on-residual pipelines is
hundreds of confident detections that are just weather-model error. And they are
**heteroscedastic**: residual spread differs across substations by orders of magnitude and varies
by time of day (PV-heavy substations are noisiest at midday). Two standard fixes, both required —
detection runs on residuals that have been *normalised* and *whitened*:

- **Normalise: measure surprise in units of each substation's usual wobble.** A 2 MW residual is
  an earthquake at a small rural substation and background noise at a large urban one — and the
  same substation wobbles more at midday than at 4 a.m. So divide each residual by the baseline's
  own estimate of "how wrong am I usually, for this substation, at this kind of moment" (the
  extra quantiles above). Detection then runs on how-unusual-is-this scores rather than raw MW,
  and one threshold scale works fleet-wide.
- **Whiten: subtract the predictable stickiness, keep the genuine news.** Residual errors are
  sticky — if the weather forecast was too cold at 09:00 it is probably still too cold at 09:30 —
  so residuals drift in slow waves rather than arriving as independent coin flips, and a naive
  detector reads each wave as a step. Whitening removes the part of each residual that was
  predictable from the residuals just before it (fit a low-order autoregressive model; keep only
  its surprises, the *innovations*). A genuine switching step is not predictable from the past,
  so it survives whitening; a slow weather-error wave does not. The alternative with the same
  effect: keep the residuals as-is but calibrate the changepoint penalty per series with a block
  bootstrap over believed-clean periods, so the false-alarm rate is controlled under the
  residual's actual stickiness rather than an i.i.d. fiction.

#### Stage 2. **Node-level coincidence / balance attribution.**

A level shift at one substation could be many things (fault, new connection, meter error). What makes it a *switching event* is the conservation fingerprint: coincident, opposite-sign shifts at neighbours that *collectively balance*. Because transfers fan out to 2–3 neighbours, **do not match pairwise.** Instead, for each candidate drop of magnitude $\Delta$ at substation $i$ at time $t$, solve a small constrained attribution: *which subset of $i$'s neighbours show coincident rises ($\approx t$) that sum to $\approx \Delta$?* With a handful of neighbours per primary this is cheap — enumerate subsets, or run a small non-negative least-squares of neighbour rises against the source drop. That candidate neighbour set is a fixed lookup from the network graph (the adjacency of who-can-exchange-load), with no learning over the graph — it is what keeps the search to a handful of substations rather than all $N$. Score by timing coincidence × magnitude-balance agreement. High score → switching event with an identified donor set; low score → "anomaly, unknown cause."

```text
residuals around time t (observed - expected), one row per substation:

 source  i :   ‾‾‾‾‾‾|____________   drop of Δ = 9 MW at t
                     t

 nbr     j :   ______|‾‾‾‾‾‾‾‾‾‾‾‾   rise +5
 nbr     k :   ______|‾‾‾‾‾‾‾‾‾‾‾‾   rise +4
 nbr     m :   ____________________  no change (0)   <- correctly excluded
                     t

 balance test:  +5 (j) + 4 (k) = 9  ==  Δ at i      -> switching event, donors {j,k}
                (pairwise matching would FAIL: neither j nor k alone equals 9)
```

**The neighbourhood-sum test (cheap, powerful corroboration).** Conservation offers a second,
sharper statistic than "the rises sum to the drop": over the candidate set {source + donors}, the
**summed residual should show no step at all** across the event, while every member shows one. So,
once a candidate attribution exists, compute the set's summed residual and require it to be
(approximately — see the tolerance note below) step-free at $t$. This discriminates exactly the
failure mode the per-series view cannot: a **regional weather-model error** steps every nearby
series *and their sum*; a genuine transfer steps the members but leaves the sum flat. It does not
*replace* per-series detection — a flat sum alone cannot say which substations moved or by how
much (it is equally consistent with "no event at all") — it corroborates an attribution after
stage 1 has proposed the pieces. The same statistic, computed around the *logged* events, is also
the very first diagnostic to run, before any detector code exists (see the diagnostic precursor
below). The [joint edge-flow estimator](#v061-the-joint-edge-flow-estimator) goes one step
further: its parameterisation builds this test in, rather than running it as a separate check.

**Calibrate the attribution score against chance.** With ~5 neighbours each carrying noisy
candidate steps, *some* subset will approximately sum to $\Delta$ surprisingly often by luck alone —
subset-sum matching on noisy data is generous. The score therefore needs an explicit null:
estimate, by permutation (run the same subset search at randomly chosen event-free times), how
often a balancing subset of a given quality arises by chance, and only accept attributions that
clear that null. Without this, the event list gets padded with confident-looking coincidences.

**Conservation is only approximate — set the tolerance band accordingly.** Reconfiguration
changes feeding-path lengths, so $I^2R$ losses change; and load served at a slightly different
voltage draws slightly different power (load is mildly voltage-dependent — the effect behind
conservation voltage reduction). Expect the
donor pickups to miss the source drop systematically by a few percent, in either direction. The
balance and neighbourhood-sum tests need a tolerance band, not an equality — and the imbalance
distribution observed on logged events is itself worth reporting.

**Events are intervals, not onsets.** Everything downstream (the ARA mask, the delivered event
table) needs `[start, end]`, so the reversion step must be detected and **paired** with its
onset: an opposite-sign, magnitude-similar step on the same source/donor set. Pairing is not
always clean — restorations can happen in stages; a network can be reconfigured directly from one
ARA into another without passing through NRA; and a fault event has a brief total-loss interim
(during which conservation does *not* hold — the load is simply off) before restoration
completes. Treat unpaired onsets as open intervals (event still in force at the end of the
record) rather than discarding them.

**Filter fleet-wide artifacts before attribution.** Telemetry re-basing, unit changes, or other
data-pipeline shifts on NGED's side produce coincident steps across *many* series at once. Any
step time shared by a large fraction of the fleet is a data artifact, not a switching event, and
must be excluded before the subset search runs — otherwise it manufactures spurious
multi-substation "events".

#### Stage 3. **Composition corroboration (post-attribution, per recipient).**

*Aim.* Stages 1–2 tell us *that* a switching event happened, *when*, and *how much net power* moved to each donor. They do **not** tell us *what kind* of power moved. A slice of the network carries a mix of underlying demand and embedded generation (rooftop PV, small wind), and the meter only ever sees the *net* (demand minus generation). Two transferred slices with the same net magnitude can have completely different make-ups — one might be 8 MW of demand with negligible generation, another might be 11 MW of demand offset by 3 MW of PV, both netting to +8 MW at the donor. The aim of stage 3 is to get a cheap, qualitative read on that make-up: *was the moved slice demand-dominated, PV-dominated, or wind-dominated?* This is corroboration and enrichment, not detection — it does not change whether we flagged the event, but it characterises it.

*Why we want it.* Three uses. (a) **Sanity-checking the attribution:** a leg whose inferred composition is physically implausible (e.g. "pure PV moved at 2 a.m.") is a signal the attribution in stage 2 mis-assigned that donor. (b) **A free preview of v2.6:** the later type-resolved model estimates per-type transfer properly; having a rough independent read here lets us check the heavy model agrees with the cheap one. (c) **Richer event labels:** the delivered event list becomes "source → donors, magnitude *and* rough composition per leg," which is more useful to NGED and to downstream stages.

*Mechanism.* The make-up of a slice is exposed by *when, within the day,* its power moved — because demand, PV, and wind each have a distinct, well-known diurnal signature. After stage 2 has told us donor $j$ picked up some load at event onset, look at the **shape of $j$'s residual step across the hours of the day** (e.g. average the step magnitude by half-hour-of-day over the event's duration):

- a step that appears mainly around **midday and vanishes overnight** → the moved slice was **PV-heavy** (PV only generates in daylight, so a slice rich in PV changes $j$'s net power most when the sun is up);
- a step that is **roughly flat, or tracks the evening demand peak** → **demand-heavy**;
- a step that is **large but uncorrelated with daylight, gusty/variable** → **wind-heavy**.

This is a histogram (step magnitude vs. hour-of-day), not a fitted model — deliberately cheap, matching v0.6's simple-statistics spirit.

*Two preconditions the read-off needs.* First, a **duration floor**: the diurnal histogram only
fills if the event spans several days — for shorter events there is no hour-of-day coverage to
average over, so composition should not be reported below (say) ~3–5 days of event duration.
Second, a **weather-realisation confound**: the histogram measures the slice's shape *under the
weather that actually occurred* — a PV-heavy slice moved during an overcast week reads as
demand-heavy. Both are mitigated the same way: rather than hour-of-day alone, **correlate each
recipient's residual step against the NWP covariates we already hold** (irradiance for PV, wind
speed for wind, temperature and time-of-week for demand). That conditions on the weather that
actually happened instead of assuming a canonical sunny day, and it is still a regression-free
read-off in the v0.6 spirit — a few correlations per leg.

*Order matters — read composition off the recipient, never the source.* The diurnal shape must be measured on **each recipient's individual step**, *after* attribution has identified which donor took which leg. It must **not** be read off the source substation's lumped drop. The reason: a source commonly sheds *different* slices to *different* donors at once — say a PV-heavy slice to donor $j$ and a demand-heavy slice to donor $k$. The source's own residual shows only the *sum* of everything it lost, which blends the two into a meaningless average that matches neither leg. Only the per-recipient steps separate cleanly into "what $j$ got" vs. "what $k$ got." (This is the same source-blends-everything pitfall noted for stage 1, applied to composition rather than magnitude.)

#### Other notes

Issue: [#180](https://github.com/openclimatefix/nged-substation-forecast/issues/180)

**Validation against the 32-series logs (this is the point).** Score the unsupervised detector against the known switching events: detection precision/recall, accuracy of the recovered donor set, error in transferred magnitude, and — most importantly — the **detection sensitivity floor**. The floor is not a single MW number: it is a frontier in **transferred magnitude × event duration**, reported per series relative to that series' residual noise. The duration axis exists because changepoint segmentation has a minimum detectable event length at half-hourly sampling (an event lasting minutes to a few hours appears as a spike or one odd interval, not a step), just as residual noise sets a minimum magnitude. Pair the frontier with the forecast impact of missed small events. *The detector must not consume the logs as input — only as a scoring oracle.* One caveat to carry into the scoring: measured **precision is a lower bound** — if the control-room logs are incomplete (worth asking NGED how complete they believe them to be), some "false positives" will be real, unlogged events.

**Synthetic event injection (the workhorse of tuning and validation).** The logs contain only however many events the trial period happens to contain, which caps the statistical power of any sensitivity estimate. Injection removes that cap: take periods believed clean, move a synthetic slice between real neighbours (subtract a scaled, plausibly-shaped signal from one series and add it across 1–3 others), and measure detection and attribution over a controlled **magnitude × duration grid**. This maps the full sensitivity frontier with arbitrary precision, tunes every threshold and penalty in stages 1–2 *without touching the gold-standard logs*, and works at any scale — including, later, on unlabelled full-fleet data. The real logs then play their proper role: confirming that the synthetic frontier transfers to reality, rather than carrying the whole measurement burden alone.

**Diagnostic precursor.** Before formal changepoints, a near-trivial check: first-difference each residual (steps → spikes) and, over rolling windows, confirm that a source's negative spike coincides with the *summed* rises of a neighbour subset. Equivalently, around each *logged* event, plot the member residuals and the neighbourhood-sum residual: members should step, the sum should stay (approximately) flat. If known-neighbour groups show no such structure, the neighbour list or baseline is wrong — fix that first.

**What it delivers.** A labelled list of detected events (time, source, donor set, per-leg magnitude, rough per-leg composition, score); an ARA mask for the forecasting data; a validation set for later stages; and the quantified sensitivity floor.

**A zeroth-order NRA reconstruction comes for free.** The event table is more than flags: each
event carries an interval, a donor set, and per-leg magnitudes, so a first-cut reconstruction of
the NRA signal is plain arithmetic — over each event interval, add each leg's magnitude back onto
the source and subtract it from its donor (with the
[edge-flow estimator](#v061-the-joint-edge-flow-estimator) this is literally observed minus the
fitted flows). What that restores is each series' mean *level* under NRA. What it cannot restore
is the moved slice's *shape*: the transferred load is live demand with its own diurnal and
seasonal variation, and subtracting a flat block leaves all of that variation sitting in the
"corrected" donor series (and missing from the source), an error that grows with event duration.
Closing that shape gap is exactly what
[v2.5](#v25-magnitude-only-mixture-model-the-workhorse) is for. The subtraction version is still
useful in its own right: it turns the ARA *mask* into an optional *patch* — keep
switching-affected periods in the forecast training data with corrected values rather than
discarding ~10% of the record — and whether the patch beats the hole is cheap to measure on the
synthetic-injection harness.

**What this approach misses / cons.**

- Misses slow/gradual reconfigurations (because changepoint detection algos assume abrupt shift).
- Misses small partial transfers near the noise floor — but these are both the *hardest* and (per the tolerance principle) the *least important* for the forecast, so the failure mode is aligned with priorities. The sensitivity floor makes this limit explicit rather than hidden.
- **Blind to events that straddle the start of the record.** A switching state already in force before the data begins has no observable onset — it is only detectable if and when it *ends*. This is a structural blind spot, not a fixable one, and belongs in the sensitivity-floor reporting alongside the magnitude threshold.
- Struggles with temporally overlapping events on one neighbourhood — the sequential
  detect-then-match structure is the culprit; the joint edge-flow estimator below is the designed
  escalation for exactly this.
- Composition read-off is qualitative only.

**Pros.**

- Trivial to build and debug; transparent; every flag is human-interpretable.
- Runs unsupervised, exactly as the scale regime demands.
- Produces the masking artifact and the honest sensitivity floor that everything downstream relies on.
- De-risks the programme at minimal cost.

#### Residual lag features — a complementary forecaster experiment

The stage-1 baseline enables a cheap experiment on the *production forecaster* that is worth
running as soon as normalised residuals exist, independently of the detector stages 2–3: replace
(or augment) the forecaster's raw power-lag features with **residual lags** — the normalised
"actual − expected" delta at each lag time, for the target substation and, optionally, its
neighbours. The intuition: a raw power lag conflates "what the weather was doing" with "what is
anomalous"; handing the model the anomaly component directly tells it how *normal* each recent
observation is.

What this buys — and why it is plausibly the largest forecast-*accuracy* win available from
switching-awareness:

- **A cleaner lag representation in general.** Separating the expected part from the anomalous
  part of each lagged observation is plausibly useful everywhere, not only during switching
  events.
- **Learned event persistence.** During an ongoing switching event the residual is a sustained
  level shift, so the model can learn "sustained recent residual → carry the offset forward".
  This is the model-learned version of the
  [zeroth-order patch](#v06-unsupervised-statistical-switching-event-detector), applied at
  forecast time instead of to the training data — and for pure forecast accuracy, "the current
  shift will persist" is most of the value switching-handling can deliver, since event *onsets*
  are unpredictable from power data by definition.
- **Neighbour context.** Neighbour residuals (e.g. a neighbourhood-sum feature) expose the
  conservation fingerprint — my residual dropped while my neighbours' rises sum to the same
  amount — which distinguishes "a transfer that will persist and eventually revert" from "a
  permanent change such as load growth or a meter re-base".

**What it does not replace.** This experiment makes the *forecaster* robust to switching; it
produces none of the detector's deliverables. There is no event list or donor attribution (so no
[`substation_switching` table](delivery-tables.md#table-5-substation_switching)), no ARA mask, no
sensitivity floor, and no latent-NRA reconstruction — a good forecast of *metered* power is a
different product from an estimate of what demand would have been under NRA. The two compose
rather than compete: once the detector exists, its outputs (an in-event flag, event age,
attributed magnitude) are themselves natural features for this model. And if the experiment wins
big, that is an argument for re-weighting the *forecast-motivated* part of the switching work —
while the event table, the mask, and the sensitivity floor keep their own justification.

**Caveats to build in from the start:**

- **A residual is not only switching.** It is also NWP error — autocorrelated and
  heteroscedastic, exactly the stage-1 problem. Feed *normalised* residuals (divide by the
  baseline's spread quantiles, which stage 1 fits anyway) rather than raw MW deltas, and include
  a neighbourhood-sum residual so the model can separate a regional NWP bust (the sum steps too)
  from a genuine transfer (the sum stays flat).
- **Fold hygiene.** The baseline is a hindcast fitted on history; in cross-validation, each
  fold's residual features must come from a baseline trained only on that fold's training
  period, otherwise the test period leaks into the features. This makes the experiment pipeline
  more expensive than adding an ordinary feature.
- **No lookahead.** Residual lags must obey the same nullification rule as raw power lags
  (`_nullify_leaky_lags()`), and the expected-power values at past lag times must be hindcast
  from NWP runs that had been published by forecast time. The existing
  freshest-NWP-for-past-target-times join does **not** guarantee this — it selects the freshest
  run per past target time with no publication-time constraint, and is leak-free today only as
  a side effect of daily run cadence — so the residual pipeline must add an explicit
  availability cut, guarded by a leakage test in the spirit of the `_nullify_leaky_lags` tests.
  The natural home for that cut is the planned central NWP analysis-proxy function
  ([#356](https://github.com/openclimatefix/nged-substation-forecast/issues/356)), which unifies
  the dashboard's stitched proxy-analysis query with the feature pipeline's
  freshest-run-per-valid-time join.
- **Cross-series features are new machinery.** Neighbour residuals need the trial-area adjacency
  list (a Part 5 dependency) plus a cross-series join in feature engineering, and a per-series
  feature-naming scheme (aggregated or ranked neighbour features, since each series has its own
  neighbour set).
- **Data volume.** Learning multi-donor conservation implicitly from 32 series with ~10% event
  occupancy asks a lot of a tabular learner; expect the model to learn "persist my own offset"
  easily and neighbour attribution only weakly. The closed-form detector exploits that structure
  directly, which is another reason this complements rather than replaces it.

The natural slot is immediately after implementation step 2 below (the baseline), as a measured
cross-validation experiment; its result also doubles as evidence of the baseline's quality. So
that it gets scheduled alongside the other forecaster experiments, it is also tracked on the
XGBoost quick-wins backlog as
[item 13](xgboost-improvements.md#13-residual-lag-features-from-the-switching-detector-baseline),
where the horizon-band and init-time-anchoring interactions specific to that page are noted.
Before any of that machinery is built, its config-only *single-stage* cousin should run first
as the ablation control:
[aligned lagged-weather features](xgboost-improvements.md#5-aligned-lagged-weather-the-single-stage-ablation-control-for-item-13)
(quick-wins item 5), which lets the booster judge each lag's normality without an explicit
baseline and bounds how much of the anomaly signal a tabular learner extracts unaided.

#### v0.6.1 — the joint edge-flow estimator

> **Status within v0.6:** this section documents the known next move if sequential matching
> (described above) proves brittle. The staged detect-then-match pipeline above remains the day-one
> implementation because it fails legibly. v0.6.1 uses the same priors and the same graph but with
> strictly better structure, and may well turn out to be the right method to get us to v1. The
> machinery it rests on — what a solver is, why convexity matters, and why this is a CVXPY problem
> rather than a PyTorch one — is explained for non-experts in [Convex
> Optimisation](../techniques/convex-optimisation.md).

**The core idea, in plain language.** The three stages above look at each substation on its own,
spot moments where its residual suddenly jumps or drops, and then play matchmaker afterwards:
this substation dropped nine megawatts, those two neighbours rose by five and four, so presumably
they belong together. Detection first, matching second.

The edge-flow estimator flips what we search for. Instead of asking "which substations stepped?",
ask "how much load is flowing across each boundary between neighbouring substations, at every
moment?" Think of the network as a map where each pair of substations that can exchange load has
an invisible pipe between them. Normally every pipe carries nothing. When engineers perform a
switching operation, one or two pipes suddenly carry, say, five megawatts, for a few days, and
then go back to nothing.

Describing the world in terms of pipes rather than steps makes conservation of power automatic.
Whatever flows out of one substation through a pipe necessarily flows into its neighbour — the
same number appears with a minus sign on one side and a plus sign on the other. It is impossible,
in this vocabulary, to describe nine megawatts vanishing from one substation while only eight
appear elsewhere. The balance check that stage 2 performs as a separate verification step
disappears: it is built into the parameterisation.

The estimation question then becomes: given the residuals observed at every substation (stage 1's
baseline is untouched by this proposal), what is the most plausible story about what all the
pipes were carrying? "Plausible" encodes exactly the two priors stated throughout this page:
switching is **rare**, so almost all pipes should carry exactly zero almost all the time; and it
is **abrupt**, so when a pipe does carry something it should be a flat block — switch on, hold
steady, switch off — rather than a gentle wiggle. We write down a score that rewards explaining
the residuals and penalises stories with many active pipes or many changes, and find the single
story with the best score. Because the problem is **convex**, that story is the certified best
one, found deterministically every run — see
[Convex Optimisation](../techniques/convex-optimisation.md) for what that promise means and why
gradient-descent tooling cannot make it.

**Everything falls out of that one answer.** Every stretch where a pipe carries something nonzero
*is* a detected event; the pipe identifies source and donor; the level is the transferred MW; the
stretch's endpoints are the event interval; "any incident pipe active" is the ARA mask.
Overlapping events on one neighbourhood — a listed weakness of the sequential matcher — are
handled natively: they are simply two pipe variables active over overlapping intervals, and the
fit apportions each node's residual across its incident edges because the two events touch
different node subsets. They only become confusable if they hit identical edges at identical
times, at which point no method could separate them and the honest answer is non-identifiability,
not a matcher bug.

**Formally.** This is a **group fused lasso** on signed edge flows $e_{ij}(t)$, with a per-node
anomaly slack $u_i(t)$:

$$
\text{residual}_i(t) \;\approx\; \sum_j \pm\, e_{ij}(t) + u_i(t)
$$

($+$ where flow enters $i$, $-$ where it leaves.)

Each term in that equation, and each penalty applied to it, is a deliberate design choice. The
bullets below unpack them in turn: the two penalties that encode the switching priors, the
anomaly slack that keeps non-switching steps out of the flows, the residuals the fit term
consumes, the approximation the parameterisation makes, and how the penalty weights are set.

- **Fused penalty on grouped increments** → each flow is piecewise-constant with sparse changes.
  The *group* structure ties the increments across edges at each timestep, so a fan-out event —
  one switching action changing two or three edge flows at once — is encouraged to place all its
  changes at one shared changepoint. (Refinement: group per node-neighbourhood rather than
  globally, so unrelated events elsewhere on the network don't share changepoint credit.)
- **Level ($\ell_1$) penalty on the flows** pulls inactive pipes to *exactly* zero — the
  regularise-toward-NRA prior. The exactness matters: "nonzero stretch = detected event" only
  works if inactive pipes read 0.000 MW rather than a trickle of ±0.03 MW, and producing crisp
  zeros is precisely what convex solvers do well and gradient descent does not (see the
  [techniques page](../techniques/convex-optimisation.md#the-corners-are-a-feature-exact-zeros)).
- **The per-node slack $u_i(t)$** (with its own $\ell_1$ penalty) is the escape valve for partnerless
  steps. A new connection, a meter fault, or genuine load growth steps one substation with
  nothing balancing it at neighbours; without $u$, the optimiser's only vocabulary is pipes, so
  it would be *forced* to invent flows. With it, partnerless steps land in $u$ and are reported
  as "anomaly, unknown cause" — exactly as stage 2 would classify them.
- **It consumes stage 1's normalised, whitened residuals** (see stage 1 for the plain-language
  version: measure surprise in units of each substation's usual wobble, after subtracting the
  predictable stickiness of weather-model error). Feeding it raw residuals would inflate phantom
  flows exactly as it inflates phantom changepoints.
- **Flat blocks are an approximation — the same one stage 1 makes.** A real transferred slice is
  live load with its own diurnal shape; a piecewise-constant flow captures its *mean level* and
  leaves the shape in the residual. That is no worse than the mean-shift changepoint detector,
  and stage 3's composition read-off survives unchanged (read the residual around each fitted
  block, per recipient). If injections show the flat-block error matters, the still-convex
  refinement is a piecewise-constant *fraction* of the source's baseline —
  $e_{ij}(t) = f_{ij}(t) \cdot \text{baseline}_j(t)$ — which is also the stepping stone to
  [v2.5](#v25-magnitude-only-mixture-model-the-workhorse), whose
  $\alpha_{ij}(t) \cdot d_j(t)$ is the same idea with the known baseline replaced by a jointly-inferred
  latent.
- **The penalty weights are tuned on the synthetic-injection harness, never on the logs.** Too
  strict smooths real events away; too loose produces confetti of tiny phantom flows. Injection
  sweeps over the magnitude × duration grid set the weights; the logged events stay reserved for
  final scoring — and injection-based tuning is the only kind available at full scale, where no
  logs exist.

**Which of the staged pipeline's failure modes it solves — and which it doesn't.** Several of
the statistical traps that stages 1–2 must handle with explicit machinery simply cannot occur in
this parameterisation:

- *Onset/reversion pairing disappears.* Stage 2 must detect each reversion step and pair it with
  its onset — messy under staged restorations and ARA-to-ARA reconfigurations. Here an event is a
  nonzero *run* of one flow variable, so its start and end are read off directly, and an
  ARA-to-ARA move is just one flow stepping down as another steps up.
- *Regional weather errors can't masquerade as events.* Flows are zero-sum across a neighbourhood
  by construction, so a common-mode residual wave (an NWP bust hitting every nearby series with
  the same sign) is *inexpressible* as edge flows — it falls into the anomaly slack instead. The
  neighbourhood-sum test from stage 2 is not run as a separate check; the parameterisation
  enforces it.
- *The chance-balance null becomes a penalty competition.* Stage 2 needs an explicit permutation
  null because some neighbour subset will approximately balance by luck. Here, for a candidate
  event to enter the solution it must out-compete two rival explanations at once — "stay at zero"
  (the sparsity price) and "call it an anomaly" ($u$) — evaluated jointly over the whole record.
  That calibration is done once, by tuning the penalty weights on the injection harness.
- *The loss/CVR tolerance band is implicit.* Exact conservation lives in the flows; the
  few-percent real-world imbalance (losses change with path length, load is mildly
  voltage-dependent) is absorbed by the fit term and $u$, with no hand-set band.

What it does **not** solve: the dependency on well-behaved residuals (the
normalised-and-whitened bullet above — phantom flows replace phantom changepoints, but only if
that step is skipped), the sensitivity floor (noise and half-hourly sampling still impose a
magnitude × duration limit, measured the same way on injections), the fleet-wide artifact
pre-filter, and stage 3's composition preconditions (duration floor, weather-realisation
confound).

**What it reuses, and why it is not the day-one build.** Everything upstream: the same
weather/calendar baseline, the same normalised/whitened residuals, the same adjacency, and the
same synthetic-injection harness. It replaces only stages 1–2's detect-then-match logic; stage 3
applies unchanged to its output. It is an escalation rather than the starting point because the
staged detector fails *legibly* — every flag traces to a visible step and a named neighbour
subset — while a joint optimiser is harder to inspect when it misbehaves. Build the transparent
version first, keep it as the reference, and adopt the joint estimator where the head-to-head
comparison (on logs + injections) shows it wins. A CVXPY implementation sketch lives in the
implementation-details section directly below.

---

#### Implementation details for v0.6.x (deleted when this ships)

The build order for v0.6, simplest-first, each step delivering something testable before the next
adds complexity. Steps 1–7 are the staged detector; step 8 is the joint edge-flow estimator, built
only if steps 1–7 prove to be too fragile (or to test out convex optimisation as a warm-up for v2!)

1. **Labelled event table + adjacency.** Parse the 32-series switching logs into a tidy table
   (onset, end, source, donor set, magnitude where recorded); obtain the trial-area adjacency
   list from NGED (see Part 5). Nothing downstream is testable without these.
2. **Baseline.** Configure the existing XGBoost forecaster with weather/calendar features only
   (**no power-lag features**), quantile objective (median + spread quantiles), per series.
   Output: normalised residual series. Sanity-check residual autocorrelation and per-series
   spread before proceeding.
3. **Diagnostic precursor — the first detection result, with zero detector code.** Around each
   *logged* event, plot the member residuals and the neighbourhood-sum residual: members should
   step, the sum should stay flat. This validates the baseline, the adjacency list, and the
   conservation premise in one cheap pass. If it fails, fix data before building any detector.
4. **Synthetic-injection harness.** Inject shaped transfers between real neighbours over a
   magnitude × duration grid into believed-clean periods. Built early because every later
   threshold is tuned on it.
5. **Per-series changepoint detection** on whitened/normalised residuals, penalty calibrated by
   block bootstrap; measure the per-series magnitude × duration sensitivity frontier on
   injections.
6. **Attribution.** Neighbour-subset search + balance scoring with the permutation null and
   loss-tolerance band; neighbourhood-sum corroboration; fleet-wide artifact filter;
   onset/reversion pairing into event intervals.
7. **Composition read-off + final validation.** Stage-3 covariate correlations (events above the
   duration floor only); score everything against the logged events (precision reported as a
   lower bound); deliver the event table, ARA mask, and sensitivity frontier.
8. **Escalation (conditional): the joint edge-flow estimator.** Build only if step 6/7 validation
   shows sequential matching is the binding error source (overlapping events, ambiguous
   attributions). Reuses steps 1–5 wholesale; penalty weights tuned on the injection harness;
   adopted only where it beats the staged detector head-to-head. CVXPY sketch below.

The
[residual-lag-features experiment](#residual-lag-features-a-complementary-forecaster-experiment)
hangs off step 2 and can run any time after it, in parallel with steps 3–8 — it does not block,
and is not blocked by, any of them (beyond the adjacency dependency for its neighbour features).

##### Sketch of the edge-flow estimator

Illustrative, untested sketch code, not the implementation. Background on CVXPY and why this problem
is convex: [Convex Optimisation](../techniques/convex-optimisation.md).

```python
import cvxpy as cp
import numpy as np


def fit_edge_flows(R, B, lam_fused, lam_sparse, lam_anomaly, weights=None):
    """Joint convex estimator for switching events (untested sketch).

    Args:
        R: (T, N) normalised, whitened residuals (observed − baseline) for N
            substations over T timesteps; NaNs allowed.
        B: (N, E) signed node–edge incidence matrix over the who-can-exchange-load
            graph (−1/+1 at the two ends of each oriented edge).
        lam_fused: penalty weight on grouped changes over time (events are abrupt).
        lam_sparse: penalty weight on flow levels (regularise toward NRA).
        lam_anomaly: penalty weight on the per-node slack (partnerless steps).
        weights: optional (T, N) fit weights; 0 masks missing data.

    Returns:
        (T, E) fitted edge flows and (T, N) fitted per-node anomalies.
    """
    n_times, n_nodes = R.shape
    n_edges = B.shape[1]

    e = cp.Variable((n_times, n_edges))  # edge flows (the "pipes")
    u = cp.Variable((n_times, n_nodes))  # per-node anomaly slack

    # --- fit term: each node's residual ≈ signed sum of incident flows + anomaly ---
    if weights is None:
        weights = np.where(np.isnan(R), 0.0, 1.0)
    resid = np.nan_to_num(R) - e @ B.T - u
    fit = cp.sum_squares(cp.multiply(np.sqrt(weights), resid))

    # --- group fused penalty: piecewise-constant flows whose changes share timesteps ---
    increments = e[1:, :] - e[:-1, :]
    fused = cp.sum(cp.norm(increments, 2, axis=1))

    # --- level sparsity: pull inactive pipes to exactly zero (NRA prior) ---
    sparse = cp.sum(cp.abs(e))

    # --- anomaly slack: partnerless steps land here, not in invented flows ---
    anomaly = cp.sum(cp.abs(u))

    prob = cp.Problem(
        cp.Minimize(fit + lam_fused * fused + lam_sparse * sparse + lam_anomaly * anomaly)
    )
    prob.solve(solver=cp.CLARABEL)  # or ECOS / SCS
    return e.value, u.value


def extract_events(edge_flows, edge_list, times, tol=1e-4):
    """Turn fitted flows into event records: maximal nonzero runs per edge.

    ``edge_list[k]`` is the oriented pair ``(a, b)`` for edge ``k``, matching the
    incidence matrix (−1 at ``a``, +1 at ``b``): positive flow moves load from
    ``a``'s meter to ``b``'s. ``tol`` is a numerical tolerance for solver precision,
    not a detection threshold — the sparsity penalty does the actual thresholding
    inside the optimisation.
    """
    events = []
    for k, (a, b) in enumerate(edge_list):
        active = np.abs(edge_flows[:, k]) > tol
        # Walk maximal runs of `active`.
        idx = np.flatnonzero(np.diff(np.r_[0, active.astype(int), 0]))
        for start, stop in zip(idx[::2], idx[1::2]):
            level = float(np.median(edge_flows[start:stop, k]))
            source, donor = (a, b) if level > 0 else (b, a)
            events.append({
                "source": source,  # lost the load
                "donor": donor,  # picked it up
                "start": times[start],
                "end": times[stop - 1],
                "magnitude_mw": abs(level),
            })
    return events
```

Notes on the sketch:

- **Scale.** At V1 scale (32 series, half-hourly, months of data) this solves in seconds to
  minutes on a laptop. At V2 scale (~2,500 series), solve per neighbourhood cluster and/or in
  sliding windows rather than one monolithic problem.
- **Group structure.** The `axis=1` group norm ties changepoints across *all* edges per timestep;
  the refinement is grouping per node-neighbourhood, so unrelated events elsewhere on the network
  don't share changepoint credit.
- **Tuning the `lam_*` weights** happens on the synthetic-injection harness (step 4), never on
  the switching logs — see the escalation section above for why.
- **The anomaly slack $u$** carries a plain $\ell_1$ penalty here; if injections show partnerless steps
  are themselves step-like (a new connection is), $u$ can be given its own fused penalty too.

---

### v2.5 — Magnitude-only mixture model (the workhorse)

**Goal.** Reconstruct a latent NRA demand $d_i(t)$ per substation by modelling observed power as a time-varying mixture of each substation's own normal demand and its neighbours'.

**Motivation.** v0.6 already supports a
[zeroth-order NRA reconstruction by subtraction](#v06-unsupervised-statistical-switching-event-detector)
— but that correction is a flat block per event: it restores each series' mean NRA *level* while
leaving the moved slice's time-varying *shape* behind. v2.5 is the simplest model whose
reconstruction carries shape — the correction is a scaled copy of a live, jointly-inferred demand
signal rather than a constant — and it produces latent demand **without modelling anything below
the primary**, sidestepping the (non-existent) feeder-discovery problem entirely. The latent
objects are the substations themselves, which we observe. This is the workhorse: faithful to how
the network actually behaves, and fully unsupervised.

**Method.** Each substation $i$ has a latent normal-demand signal $d_i(t)$. Observed power is a time-varying mixture over the neighbourhood:

$$
\text{observed}_i(t) = \alpha_{ii}(t)\, d_i(t) + \sum_{j \,\in\, \text{neighbours}(i)} \alpha_{ij}(t)\, d_j(t)
$$

- **The neighbourhood is the graph:** $\text{neighbours}(i)$ is exactly $i$'s neighbour set in the network graph, so the edges act as a *sparsity pattern* on the mixing matrix — most $\alpha_{ij}$ are structurally fixed at zero, and only the handful corresponding to real edges are free parameters. This is what makes the model identifiable and cheap rather than an $N \times N$ free-for-all.
- Under NRA: $\alpha_{ii} \approx 1$, $\alpha_{ij} \approx 0$.
- During an ARA: weight shifts from a source onto **one or more** neighbours. Multiple $\alpha_{ij}(t)$ may be active at once for a single source.
- **Conservation = node-level flow balance:** weight leaving $i$ is distributed across a subset of neighbours and must sum to the weight lost at $i$ (approximately mass-preserving over the affected neighbourhood). **Do not** implement this as independent pairwise equal-and-opposite constraints — that is wrong given confirmed 2–3-way fan-out.
- **Priors / regularisation:** $\alpha(t)$ strongly regularised toward the identity (NRA) and **piecewise-constant in time**, because switching events are rare (~10%) and abrupt. A useful by-product: jumps in $\alpha$ are directly interpretable as detected switching events.
- **$d_i(t)$ must itself be modelled, not left free.** If the latent demand were an unconstrained
  value per timestep the model would be hopelessly underdetermined — any observation can be
  explained by moving $d$ instead of $\alpha$. $d_i(t)$ is a weather/calendar-driven model plus a
  smooth residual; in other words, v2.5 embeds the v0.6 baseline inside itself as the latent's
  backbone. The v0.6 work is reused, not discarded.
- **Fitting is bilinear (non-convex) — initialise from v0.6.** The $\alpha \cdot d$ products mean local
  minima are a real risk. Initialise/anchor the $\alpha$ jump times from the v0.6 event list (or the
  edge-flow estimator's fitted flows, which are the same object in additive form) so the
  optimiser starts near the right switching structure. v0.6's output is thus a direct *input* to
  v2.5, not just its validation set.
- **Known degeneracies to guard.** (a) *Scale:* $\alpha_{ii} d_i$ is invariant to rescaling one against
  the other; the identity prior resolves this except for a substation observed under ARA for
  (nearly) its whole record, where v0.6's record-straddling blind spot applies unchanged.
  (b) *Growth vs transfer:* genuine new load at $i$ and a small persistent transfer onto $i$ are
  separated only by conservation (growth has no balancing donor) — keep an explicit per-node
  anomaly/slack term, as in the edge-flow estimator, so partnerless changes are not forced into
  $\alpha$. (c) *Net-zero crossing:* $\alpha$ is a fraction of **net** power, and a fraction of a signal
  near zero moves almost nothing regardless of $\alpha$ — at PV-heavy substations whose net crosses
  zero, the transfer's magnitude information vanishes around the crossing. v2.6's typed
  decomposition removes this by mixing gross components instead.

**Tooling: v2.5 can be built entirely with CVXPY — by alternation, with one honest caveat.** As
written, v2.5 is *not* one convex problem: $\alpha \cdot d$ multiplies two unknowns, which is exactly the
kind of expression CVXPY's DCP check refuses (see
[Convex Optimisation](../techniques/convex-optimisation.md)). But the bilinearity has a special
structure — the problem is convex in each unknown *separately* — and that enables the classic
**alternating** scheme, where every step is a plain CVXPY solve:

- **$\alpha$-step (fix $d$, solve for $\alpha$).** With $d$ treated as known data, the mixture is linear in
  $\alpha$, and the priors (identity-regularised, piecewise-constant in time, node-level balance as
  linear constraints) make this the same group-fused-lasso family as the
  [v0.6.1 edge-flow estimator](#v061-the-joint-edge-flow-estimator) — convex, with the exact
  zeros that let jumps in $\alpha$ read directly as detected events.
- **$d$-step (fix $\alpha$, solve for $d$).** With the routing fixed, the observation model is linear in
  $d$, and "weather/calendar backbone plus smooth residual" is a convex regression.

Iterate the two steps to convergence. There is also a natural **convex warm start**: fixing $d$
to the exogenous baseline makes the whole model one convex problem — it is essentially
[v0.6.1's](#v061-the-joint-edge-flow-estimator) shaped-flow refinement
$e_{ij}(t) = f_{ij}(t) \cdot \text{baseline}_j(t)$ in mixture clothing — so solve that
first, then let alternation release $d$. The caveat to state plainly: convexity now holds per
*step*, not overall. Alternation converges, but to a *local* optimum of the bilinear problem — the
certified-global-optimum promise of the pure-convex world does **not** come back just because
each step uses CVXPY. The v0.6 initialisation (bullet above) and the convex warm start are what
manage that risk. No part of v2.5 needs PyTorch.

**Where `cvxpylayers` enters (and where it doesn't).** The bridge that embeds a convex solve as a
differentiable layer inside a PyTorch model
([the techniques page explains it](../techniques/convex-optimisation.md#the-bridge-welding-cvxpy-into-pytorch))
is *not needed* for v2.5 — alternation above is CVXPY in a loop. It earns its place at **v2.6**,
when $d_i$ decomposes into physics modules (panel trigonometry, power curves, products of
unknowns) that are genuinely non-convex and force PyTorch for the outer model — see the
[v2.6 tooling paragraph](#v26-type-resolved-mixture-with-differentiable-physics-modules) for why
the routing estimation should stay a convex layer even then.

**Why "arbitrary continuous slice" is handled natively.** $\alpha_{ij}(t)$ is a continuous fraction, so "some load, cut anywhere, moved to several donors" is exactly representable. The continuous-fraction form — which earlier looked like a limitation — is in fact *fidelity* to a network where the transferred amount is genuinely continuous and the cut point is free.

**What it adds over v0.6 (including the v0.6.1 edge-flow estimator).**

- **Shape, not just level.** Both v0.6 reconstructions correct an event with a flat block — the
  staged detector's subtraction and v0.6.1's piecewise-constant flows alike — leaving the moved
  slice's diurnal/seasonal variation behind. v2.5's correction $\alpha_{ij}(t) \cdot d_j(t)$ is a
  scaled copy of a live demand signal, so the moved load's variation moves with it. Even v0.6.1's
  shaped-flow refinement $e_{ij}(t) = f_{ij}(t) \cdot \text{baseline}_j(t)$ only borrows shape
  from the *fixed exogenous baseline*; v2.5's shape comes from the jointly-inferred latent
  $d_j(t)$ — the weather/calendar backbone *plus* the smooth residual the baseline cannot see.
- **Joint inference instead of point-estimate commitment.** The subtraction reconstruction
  commits forever to the detector's noisy magnitude estimates (which also absorb the few-percent
  loss/CVR imbalance). v2.5 re-estimates routing and latent demand together, so each refines the
  other and detection-time noise in the magnitudes is smoothed against the demand model.
- Produces the actual latent NRA demand signal $d_i(t)$ as a modelled object, not
  observed-minus-detected-events arithmetic.
- Models routing continuously rather than detecting it after the fact.
- Accommodates multi-donor partial transfer through the summed mixture + node-level balance.

**What it misses / cons.**

- A fractional mixture $\alpha_{ij} d_j$ moves a *scaled copy of neighbour j's whole aggregate demand*. The slice that really moved may have a *different shape* (e.g. unusually PV-heavy). v2.5 can match the step magnitude but carries the wrong shape with it. **Important nuance:** because there is *no stable sub-unit* with a "true" recoverable shape (movable cut points), this is largely **not a fixable limitation** — v2.5's approximation is about as good as the data structurally permits for the *demand* total. v2.6 only partially improves it, and only for the DER component.
- DERs are folded implicitly into $d_i$ (no explicit PV/wind separation yet).

**Pros.**

- Well-posed, identifiable, far easier to fit than any sub-primary model.
- No sub-primary modelling whatsoever; fully unsupervised; scales.
- Degrades gracefully; faithful to continuous, multi-donor, partial transfer.
- The right workhorse: build it, measure residual structure around detected events, escalate only if residuals show systematic shape error.

---

### v2.6 — Type-resolved mixture with differentiable physics modules

**Goal.** Decompose each substation into physically-typed components (demand, PV, wind), each from its own differentiable module, and let each *type* transfer with its own routing weights — so a switching event can move proportionally more PV than load.

**Motivation.** v2.5's residual error is wrong-*shape* transfer. Part of that shape error is *type mix*: the moved slice may carry disproportionate PV or wind relative to the parent's aggregate. Separating the physical types lets the model represent that. The differentiable modules also clean the meter of DERs generally (PV/wind net off demand whether or not switching occurs), which is independently valuable.

**Method.** Decompose each substation into typed components, each from its own differentiable forward module:

$$
d_i(t) = \text{gross\_demand}_i(t) - \text{pv\_metered}_i(t) - \text{pv\_unmetered}_i(t) - \text{wind\_metered}_i(t) - \text{wind\_unmetered}_i(t)
$$

- **Demand:** temperature- and time-of-week-shaped.
- **PV:** irradiance-driven (NWP/satellite) via differentiable panel physics (temperature/spectral correction, inverter clipping); capacity is a latent parameter.
- **Wind:** wind speed via a differentiable power curve.
- **Metered vs unmetered split:** metered modules are tightly constrained by registered capacity/generation and mainly *remove* known generation cleanly; unmetered modules carry the genuine latent inference (latent capacity, latent siting), confined to the smaller residual — a meaningful conditioning win.

Mixing operates **per component-type**, each with its own routing weights:

$$
\text{observed}_i(t) = \sum_j \Big[\, \alpha^{\text{dem}}_{ij}(t)\, \text{gross\_demand}_j(t) - \alpha^{\text{pv}}_{ij}(t)\, \text{pv}_j(t) - \alpha^{\text{wind}}_{ij}(t)\, \text{wind}_j(t) \,\Big]
$$

Each substation is now a small *bundle* of typed nodes rather than one node, and routing happens per type. But the graph stays a plain **data structure**, exactly as in the earlier stages: when the i→j boundary is active, each *type* moves with its own weight — structure plus arithmetic, with nothing learned along the edges.

**Tooling.** This is the stage where PyTorch becomes unavoidable: the typed forward modules are
genuinely non-convex (see [Convex Optimisation](../techniques/convex-optimisation.md) for why),
so the outer model lives in PyTorch per the
[differentiable-physics](../techniques/differentiable-physics.md) plan. The per-type routing
weights $\alpha^{\text{type}}_{ij}(t)$, however, should **remain a differentiable convex layer** inside that
model (via
[`cvxpylayers`](../techniques/convex-optimisation.md#the-bridge-welding-cvxpy-into-pytorch))
rather than becoming free tensors: the layer preserves exact zeros ("nonzero routing = detected
event") and built-in conservation, while gradients flow through the solve to the physics modules.
The full rationale, and the alternation-only path that gets v2.5 built without any PyTorch, is in
the v2.5 tooling note above.

**Prior structure.**

1. **Coupled switching, separate composition.** A reconfiguration is one electrical action — the types do not switch at unrelated times. Introduce one latent **switching indicator per ordered pair**, $s_{ij}(t) \in \{0, 1\}$, piecewise-constant and sparse, governing *whether* the i→j boundary is active; the **type composition** (what fraction of demand/PV/wind rides along) applies only when $s_{ij}(t) = 1$. Types switch *together*; the moved slice can still be disproportionately one type.
2. **Per-pair composition prior — DOWNGRADED.** An earlier design proposed a *learnable per-boundary* prior $\theta_{ij}$ ("the i→j boundary is usually 90% PV"), treating it as a stable feeder fingerprint. **This is downgraded to at most a weak, shared empirical prior, or dropped entirely.** Reason: movable cut points mean a boundary has *no stable composition* — what crosses depends on where the switch was opened this time — and $\theta_{ij}$ could not be fitted at scale anyway, since that requires labels we won't have. Do **not** rely on per-boundary learned composition.
3. **Multi-donor (one-to-many).** As in v2.5, several $s_{ij}(t)$ may be active for one source at once; conservation is node-level flow balance across the donor set, not pairwise.

**What it adds over v2.5.**

- Captures disproportionate-*type* transfer (more PV than load can move).
- Differentiable modules give separately-shaped PV/wind/demand signals, so a step can be attributed to the right type by its temporal signature (midday irradiance-shaped → PV; evening temperature-tracking → load).
- Cleans DERs from the demand estimate generally, not just during switching.

**What it misses / cons.**

- Still moves a *scaled fraction of neighbour j's total PV/wind/demand*, not the specific moved slice's profile. Fixes wrong *type-mix*; does **not** fix wrong *within-type shape*. Given movable cut points, the within-type shape is not stably recoverable anyway, so this residual is largely irreducible and almost certainly second-order for forecasting.
- Larger latent space (three routing matrices): risk the optimiser trades PV- vs wind- vs demand-transfer to fit one step. **Mitigation:** the three types have distinct observable temporal signatures and cannot masquerade as each other if priors lean on irradiance/wind/temperature covariates; regularise each routing weight toward identity and piecewise-constant. When genuinely confounded (a windless, overcast event), accept non-identifiability and let uncertainty reflect it.

**Pros.**

- Physically grounded; the extra freedom is tied to physics, not invented structure.
- Still only primary-level objects; no sub-primary modelling; unsupervised; scales.
- Produces interpretable DER estimates as a by-product.

---

## Part 3 — Considered but rejected: the feeder-block model

An earlier plan included a further stage that modelled the **actual switchable physical units (feeders / load blocks)** explicitly — decomposing each substation into discrete blocks, each routed as a unit. **This stage is retired**, for two independent and decisive reasons:

1. **The unit does not exist.** NGED have been explicit that the network is meshed and run radially with movable cut points; load is a near-continuous distribution splittable almost anywhere. There is no stable, re-identifiable feeder with a persistent identity or composition to discover and route. The model would be trying to recover units that do not persist.
2. **It lives in the unlabelled regime.** This stage is precisely what would run at full scale (~1,161 primary substations), where **no switching labels exist.** Any block model that needed supervision to identify blocks is doomed there twice over.

**Possible deferred successor (a research bet, not a planned step).** If within-type shape error in v2.6 ever proves to materially hurt the forecast, the conceptually correct (but much harder) direction is to model **load as distributed along network arcs with movable cut points**, inferring the cut location rather than assuming discrete blocks. This is a genuine spatial-inference research problem and should be explicitly *deferred*, not chased — and only entertained if v2.6 residuals demonstrate the need.

---

## Part 4 — Cross-cutting implementation requirements

These apply at every stage and are the things most easily got wrong:

- **Fully unsupervised at runtime.** The production model uses **power time series only**. Switching logs are never a runtime input. They exist for the 32-series trial only, and serve exclusively as a **gold-standard test set**.
- **Do not fit pilot-only parameters and rely on them at scale.** Anything learned only on the 16 labelled primaries that cannot be set for the other ~1,145 is forbidden as a production dependency. The *method* generalises; a pilot lookup does not.
- **Conservation is node-level flow balance, everywhere.** One source's lost power is absorbed by a *subset* of neighbours whose pickups sum to it. Never implement conservation as independent pairwise equal-and-opposite matches — confirmed 2–3-way fan-out makes that wrong.
- **Composition is read from recipients, never the source.** Any per-leg composition estimate (v0.6 stage 3) must come from each recipient's individual step *after* attribution; the source's step blends all simultaneous outgoing legs.
- **Partial transfer is the common case.** Expect arbitrary continuous transferred magnitudes down to the noise floor. Quantify the detection sensitivity floor (a magnitude × duration frontier, per series); do not assume a clean event/no-event separation.
- **Every detection statistic gets a null.** Changepoint penalties calibrated per series under the residual's real autocorrelation; attribution scores calibrated against chance-level subset balance by permutation. Uncalibrated thresholds are how phantom events happen.
- **Synthetic event injection is the standard tuning instrument at every stage.** Thresholds, penalties, and sensitivity frontiers are tuned and measured on injected events; the logged events are reserved for final scoring, never for tuning.
- **Routing/switching priors:** regularise toward the identity (NRA) and piecewise-constant in time; switching is rare and abrupt.
- **Metered vs unmetered DER** kept as separate modules from v2.6 onward; metered tightly constrained, unmetered carrying the latent inference.
- **Interpretability artifacts** (detected events, inferred $s_{ij}$, DER estimates) are deliverables in their own right for NGED validation, not just internal state.

---

## Part 5 — Open items / dependencies

- **Switching logs for the 32-series trial** (held; e.g. the DINDER example). Used as the gold-standard validation set for v0.6 and beyond. **Not** available at full scale — this asymmetry drives the whole design.
- **Neighbour/adjacency structure** for the trial substations — which substations can exchange load (needed to define graph edges and the attribution search). Even approximate adjacency helps; note that because cut points move, "adjacency" means "can be electrically connected by some switching," not a fixed feeder map.
- **Confirmed by NGED:** (a) multi-recipient transfer (2–3 donors) is the norm; (b) partial transfers (some, not all, of a substation's load) are the common and harder case; (c) no stable "feeder" unit exists; (d) switching labels exist only for the trial area, not at scale.
- **To ask NGED:** how complete are the control-room switching logs for the trial area? (Determines whether measured detection precision is a tight bound or a loose lower bound.)
