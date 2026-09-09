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

**Caveat**: this spoke stops being evidence for any fit that used the register. The capacity work
plans to feed registered capacity in as a
[convex prior](../roadmap/capacity-estimation.md#loss-and-penalties), and once a register is in the
objective, agreement with that register is partly the optimiser doing what it was told.

**The fix is an ablation, not a ban.** Fit once *without* the register prior and score that fit
against the register; ship the fit *with* the prior. The delivered estimate then uses every source
available, while the validation number comes from a fit that never saw the register. Report both,
and keep the no-prior fit as a standing leaderboard column rather than a one-off, or it rots.

**Score the ablation on pattern, not on level.** What the disaggregation recovers is registered
*plus* unregistered capacity, so the estimate should exceed the register and an estimate below it
is a detectable failure. The informative signals are the rank correlation across substations —
does the estimate order catchments the way the register does? — and level agreement restricted to
the subset where the register is near-complete, such as large registered ground-mount, where
little unregistered capacity can hide. Spoke 1 and Spoke 7 read no register at all and stay
independent either way.

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

## Spoke 7: Manual capacity survey from aerial imagery (direct, small-sample)

Count rooftop PV by hand from aerial or high-resolution satellite imagery across a few primary
substations' catchments, and compare the total against the estimated unmetered capacity. This is
the only spoke that measures installed capacity directly, and the only one that is independent of
both the meters and the registers — which is what makes it the check to reach for when the
registers have been used as priors. Rooftop-PV detection from imagery is a well-developed computer
vision task, so a hand-counted pilot can be scaled later if it proves worth it.

**Caveat**: imagery gives panel area, not kilowatts, so the comparison carries an assumed
watts-per-square-metre and misses panels hidden by shading, flat-roof mounting angles, or tree
cover. The imagery's capture date rarely matches the estimate's period. Counting is expensive per
catchment, so the sample is small and chosen rather than random, which makes it a check on
magnitude rather than a statistic.

---

## The structural conclusion

A good method scores well across all seven spokes despite their differing biases. A method that
scores well on synthetic aggregation (Spoke 1) but fails physical-consistency checks (Spoke 3) on
real data has overfit to the easy case. The leaderboard columns for disaggregation are not "the
metric" — they are these spokes. Because labels are weak, the protocol must be more carefully
reasoned and transparently caveated than a standard forecasting evaluation: "no clean ground
truth" must not slide into "any evaluation will do."
