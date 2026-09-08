# Evaluating disaggregation: a multi-pronged protocol

> **Status: 🔬 v2 research.** This applies to the full disaggregation problem (v2) — the plan and
> architecture live on the canonical
> [Net-demand disaggregation](../roadmap/disaggregation.md) roadmap page. The v0.7 capacity
> estimation for metered generators is evaluated separately, with its own
> [head-to-head protocol](../roadmap/capacity-estimation.md#the-head-to-head-protocol) against the
> metered ground truth. See the [roadmap index](../roadmap/index.md) for status conventions.

Substation disaggregation has **no single clean ground truth**: by definition, you are estimating
quantities that are unmetered. Chasing a single objective metric is therefore a trap. The rigorous
approach is a **basket of complementary partial evaluations**, each with different biases, where
agreement across them is the real signal.

The evaluation protocol is not a detail — it is arguably the hardest and most publishable
contribution of the v2 work. The same structural challenge arose in domestic Non-Intrusive Load
Monitoring (NILM), where individual appliance ground truth is similarly hard to obtain. Building a
credible multi-pronged protocol is a natural continuation of that prior work.

---

## Spoke 1: Synthetic aggregation ("the Neural NILM move")

Take individually metered sources (customer-metered solar photovoltaic (PV) and wind, the metered
battery energy storage system (BESS), etc.), sum them into a synthetic "substation," disaggregate,
score against the held-out components. This gives an exact ground truth because you constructed the
aggregate.

**Caveat**: a synthetic clean sum lacks [switching events](../roadmap/switching-events.md), MVA bounce, false
zeros, unmetered load, and correctly-scaled correlated co-movement. It systematically flatters
performance — it measures the model on an easier problem than reality. Always report as "performance
under idealised aggregation," never as real-world skill.

---

## Spoke 2: Partial / spot ground truth (held-out metered sources)

Where a generator is metered behind a primary, pretend it is unmetered, disaggregate it out of the
real net flow, compare to the metered value. Real (not synthetic) ground truth on real substation
data, for the subset where metering happens to exist.

**Caveat**: biased to the sites where metering happens to exist, which may not be representative
of the unmetered fleet.

**Concrete example in the NGED dataset**: Stickney primary's midday peaks correlate with the
separately-metered nearby Leverton solar farm — a ready-made held-out label test.

---

## Spoke 3: Physical-consistency and conservation residuals (label-free)

You do not need labels to detect wrongness. Hard constraint: disaggregated components must sum to
the measured aggregate (Kirchhoff's current law). Softer physical checks: disaggregated PV ≈ 0 at
night and bounded by the clear-sky envelope; disaggregated wind correlates with wind speed, not
irradiance; estimated unmetered-PV capacity is physically plausible given the substation's
geographic footprint.

Violations are detectable errors without any ground truth — a rigorous "wrongness floor" that
discriminates between methods. This spoke is underused in the disaggregation literature: the
[energy-forecasting review](../background/energy-forecasting-review.md) found that using physical
consistency to *score* an estimate, rather than to *shape* the method that produces it, is close to
absent from the papers it read.

---

## Spoke 4: Cross-source corroboration (label-free, indirect)

Where an independent dataset should predict your disaggregated quantity, agreement is evidence. The
main example: estimated unmetered-PV capacity per primary vs. registered PV in the Embedded
Capacity Register (ECR) / Microgeneration Certification Scheme (MCS) for that substation's
geographic catchment (recoverable via the Meter Point Administration Number (MPAN)→substation
mapping).

**Caveat**: the gap between the estimate and the register is partly the unregistered fleet you are
trying to find, so exact agreement is not expected. Gross disagreement in the wrong direction
(estimate < registered) is a detectable error. Weak but real-world triangulation.

---

## Spoke 5: Second-order forecast improvement

Does disaggregating the substation signal before training improve net-demand forecast skill on the
held-out test set? This evaluation has clean ground truth (you are forecasting the future against
observed meter readings).

**Caveat**: this measures the *instrumental value* of disaggregation for forecasting, not
disaggregation fidelity. A disaggregation that is wrong-but-harmless-to-the-forecast scores well;
a correct-but-forecasting-irrelevant disaggregation scores "useless." Report this as a distinct
quantity from disaggregation accuracy. Spoke 5 is arguably the most decision-relevant evaluation
for NGED, because their goal is better forecasts for flexibility procurement.

---

## Spoke 6: Recovery on a fully-instrumented holdout (strongest)

A single substation, even briefly, where every feeder and embedded generator is individually
metered, used purely as validation. One such site anchors the whole evaluation.

---

## Spoke 7: Simulated substations from fitted models

Fit the per-site generation modules ([differentiable physics](differentiable-physics.md)) and the
shared demand-profile basis ([the `BasisLoadNode`](../roadmap/disaggregation.md#node-definitions))
on real telemetry, freeze their parameters, run them forward on real weather, and sum the sites
into a simulated substation. Every quantity the disaggregator must recover — each site's
generation, the residual demand, and each fitted capacity — is written down by construction, and a
whole fleet of simulated substations costs no metering. Unlike Spoke 1, the simulated sum covers
the unmetered fleet and the demand as well as the metered sites.

**Caveat**: this spoke is circular in a way the other six spokes are not. An estimator scored on
data generated by its own model family measures whether the parameters are identifiable, not
whether that model family matches reality, so a disaggregator drawn from the same family recovers
the components it was built to produce. The simulated telemetry is also missing the artefacts
listed under Spoke 1, plus meter noise; a simulator can reproduce the MVA magnitude, but not the
reactive-power floor that lifts it at the bounce. Where an edit can be applied to real telemetry
instead — injection adds to a real series rather than replacing it — prefer the edit to real
telemetry, and reserve the simulated substation for the labels no edit to real data can produce,
such as removing the unmetered fleet entirely. The same construction serves the switching and
capacity problems too; that
[wider v2 idea](../roadmap/index.md#v20-scale-up-future-research) is on the roadmap.

---

## The structural conclusion

A good method scores well across all seven spokes despite their differing biases. A method that
scores well on synthetic aggregation (Spoke 1) but fails physical-consistency checks (Spoke 3) on
real data has overfit to the easy case. The leaderboard columns for disaggregation are not "the
metric" — they are these spokes. Because labels are weak, the protocol must be more carefully
reasoned and transparently caveated than a standard forecasting evaluation: "no clean ground
truth" must not slide into "any evaluation will do."
