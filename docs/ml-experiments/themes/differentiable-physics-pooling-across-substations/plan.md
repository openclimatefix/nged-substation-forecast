# Differentiable physics: pooling across substations: research plan

**The rows below are the planned and proposed experiments of this theme, in priority order (1 is
first).** Experiment 105, the first of the theme, is finished: pooling helped a new or short-history
substation and was worse at full history.

| Experiment | Question | Priority | Waits for | State |
|---|---|---|---|---|
| Onboarding design (proposed) | Does a pooled model for onboarding and a per-site model for mature substations beat either alone? | 2 | Experiment 105: pooled onboarding scored 4.04 against 14.38 for a per-site model on 4 weeks, and was worse at full history | Proposed |
| Part 4 (proposed) | Does a drift term (a slowly changing demand level), trained from 2019-09, beat the same model on its best window? | 3 | Experiment 112 (finished: an exact variable-projection fit did not improve the pooled model, and its known-answer test failed at realistic noise) and a rerun of experiment 109's gate | Proposed |
| Part 5 (proposed) | Does the MVA observation layer (a model of how a meter that reads apparent power relates to real power) help the MW sites and recover MW at MVA sites? | 4 | Experiment 105 and a better capacity fit | Proposed |
| N9, scale-up | Does the pooled model on every v1 series it can take, including MVA substations, improve full-history skill and capacity identification? | 5 | A rerun of experiment 109's gate, which is outstanding; 3 to 5 GPU hours | Queued |
| BSP as a Kalman observation | Does treating the BSP reading as an extra observation in the Kalman filter improve the Primaries' forecasts? | 6 | Experiment 124 (finished: a BSP term in the joint fit raised clean-forecast error) | Proposed |
| Top-down against bottom-up | Does a direct forecast at a BSP or grid supply point (GSP) beat the sum of the Primary forecasts, and does reconciling the levels help either? | 7 | The closure audit | Proposed |
| Metered generators in the graph | Do metered generators, entered as export-positive nodes at the BSP, make the BSP relation close better? | 8 | The closure audit | Proposed |
| Unmetered wind at higher levels | Can unmetered wind at the BSP and GSP be estimated as the part of the residual that follows the wind forecast? | 9 | The closure audit | Proposed |

**Experiment 109 gives the pooled convex model N for the next experiments to start from, once its
known-answer gate is rerun.** The gate failed under its pre-registered definition, so Part 5
and Part 6 depend on N and on a rerun of the gate with an oracle that estimates every
coefficient and a tuning rule checked on several seeds.

## Planned experiments

<!-- planned-table -->

## N9: scale-up

**The proposal fits the pooled model on every v1 series it can take, including MVA substations
through the Part 5 meter layer.** More substations, and the sunlit rows of MVA substations teaching
the solar term, should improve full-history skill and make capacity better identified. The
experiment runs no onboarding arms. It starts from experiment 109's convex pooling if the gate rerun
supports it, and otherwise from per-site fits with the shared meter layer. It needs 3 to 5 GPU hours.

## Hierarchy: Primaries, BSPs, and GSPs

**[Experiment 124](../../00124-differentiable-physics-h-bsp-closure-audit-and-bsp-hint/result.md)
ran the closure audit and a pilot joint fit.** A BSP reading should equal the sum of its Primaries
minus the metered generators connected there, plus losses and an unmetered remainder. The audit found
that the Primaries and generators close one of the two BSPs up to factors of 0.85 to 1.16, and the
other BSP not at all with the generator history available. The joint fit, with the factors fixed at
1, cut the damage from injected metering errors by 41% but made clean forecasts worse by 2.4 or more
points of capacity. A follow-on arm that fixes the factors at the audit fit stopped at its synthetic
check, where an exactly specified synthetic system also lost clean accuracy at every weight. The loss
therefore comes from adding a BSP term to the Primaries' fit, and not mainly from the factors.
[Experiment 36](../../00036-hierarchy-aware-bsp-gsp-forecasts/result.md) had found that the sum of
Primary forecasts added nothing at the two BSPs.
Three facts shape the design:

- **The remainder is large.** In the data checked, the BSP residual has a daily-mean standard
  deviation larger than the typical switched load, so the audit has to say what the remainder
  contains before any constraint uses it.
- **The GSP relation cannot be closed.** Most of the power at each GSP is unmetered.
- **Reconciliation helps the aggregates most.** [Nespoli et al. (2022)](https://arxiv.org/abs/1910.03976)
  found that reconciling cut the aggregate error by up to 10% and the bottom-level error by under
  1%, and [Brégère and Huard (2022)](https://arxiv.org/abs/2003.00585) found that most of the gain
  came from mixing in other nodes' forecasts, not from the constraint. The expected gain at the
  Primaries is therefore small, and the proposals below aim at the BSP and GSP levels.

**The BSP as a Kalman observation tests whether the closure relation, with its unmetered remainder as
state noise, helps the Primaries' filters.** It runs on the CPU in about 2 hours. Experiment 124 has finished the closure audit.

**Top-down against bottom-up at the BSP and GSP levels asks which direction gives the better
forecast at each level.** The arms are a direct forecast of the BSP, the sum of the Primary
forecasts plus the generators, and a minimum-trace reconciliation of the two, each scored at both
levels. The topology is soft, not a hard sum: the hierarchy enters as a hint that the levels should
roughly agree, because the unmetered remainder makes an exact balance wrong.

**Metered generators enter the graph as export-positive nodes at the BSP.** The Primary series are
the disaggregated demand, which already has the metered generators taken out, and the generators connect
at the BSP busbar. Adding them as nodes should make the BSP relation close better. Their series are
generator data, so any published figure shows them only as an anonymised, capacity-normalised
series, never beside a name.

**Unmetered wind at the BSP and GSP levels is invisible at the Primaries and may show up higher.**
The audit's residual, regressed on the weather forecast's wind speed at the BSP, would give a first
estimate of how much unmetered wind sits there. The estimate only counts if the wind-driven part of
the residual is separate from the solar part, so it waits for the audit.
