# Differentiable physics: pooling across substations: research plan

**The rows below are the planned and proposed experiments of this theme, in priority order (1 is
first).** Experiment 105, the first of the theme, is finished: pooling helped a new or short-history
substation and was worse at full history.

| Experiment | Question | Priority | Waits for | State |
|---|---|---|---|---|
| Partly pooled GAM shapes (proposed) | Do partly pooled GAM shapes (shared heating curve, per-substation offset from full history) beat per-site GAM fits now that PV and switching are explicit terms? | 1 | The explicit PV work, the full-history arm, and the switching terms; experiment 154 supports sharing the heating curve but not offsets learned from 4 weeks | Proposed |
| Onboarding design (proposed) | Does a pooled model for onboarding and a per-site model for mature substations beat either alone? | 2 | Experiment 105: pooled onboarding scored 4.04 against 14.38 for a per-site model on 4 weeks, and was worse at full history | Proposed |
| Part 4 (proposed) | Does a drift term (a slowly changing demand level), trained from 2019-09, beat the same model on its best window? | 3 | Experiment 112 (finished: an exact variable-projection fit did not improve the pooled model, and its known-answer test failed at realistic noise) | Proposed |
| Part 5 (proposed) | Does the MVA observation layer (a model of how a meter that reads apparent power relates to real power) help the MW sites and recover MW at MVA sites? | 4 | Experiment 105 and a better capacity fit | Proposed |
| N9, scale-up | Does the pooled model on every v1 series it can take, including MVA substations, improve full-history skill and capacity identification? | 5 | Nothing; 3 to 5 GPU hours | Queued |
| BSP as a Kalman observation | Does treating the BSP reading as an extra observation in the Kalman filter improve the Primaries' forecasts? | 6 | Experiment 124 (finished: a BSP term in the joint fit raised clean-forecast error) | Proposed |
| Top-down against bottom-up | Does a direct forecast at a BSP or grid supply point (GSP) beat the sum of the Primary forecasts, and does reconciling the levels help either? | 7 | The closure audit | Proposed |
| Metered generators in the graph | Do metered generators, entered as export-positive nodes at the BSP, make the BSP relation close better? | 8 | The closure audit | Proposed |
| Unmetered wind at higher levels | Can unmetered wind at the BSP and GSP be estimated as the part of the residual that follows the wind forecast? | 9 | The closure audit | Proposed |

**Experiment 109 gives the pooled convex model N for the next experiments to start from, and
[experiment 150](../../00150-differentiable-physics-n0-gate-rerun-fair-oracle-five-seeds/result.md)
reran its known-answer gate.** The rerun was `inconclusive`: capacity from 4 weeks was within 25% at
8 of 8 substations on 5 of 5 seeds, the tuning rule was within one grid step of the best penalty on 5
of 5 seeds, and the pooled fit's heating error beat a fair oracle's, but the shrinkage line could not
tell a good fit from a bad one on 4 June weeks. [Experiment
154](../../00154-differentiable-physics-n0-gate-2-heating-error-against-truth/result.md) then scored
the heating error against the injected truth on 4 winter weeks: the pooled fit beat no pooling and
complete pooling on 5 of 5 seeds and recovered the fleet's shared curve as well as separate fits,
but the sign of a substation's own offset was found on only 2 of 5 seeds, by the fair oracle no
more reliably.

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
found that adding the BSP reading to the Primaries' fit raised clean-forecast error at every weight
tried.** The raise was 0.4 to 4.6 points of capacity, although the reading cut the damage from
injected metering errors by about 41%. The closure audit found that the Primaries and generators
close one of the two BSPs up to factors of 0.85 to 1.16, and an exactly specified synthetic system
lost clean accuracy the same way, so the cause is untested.
[Experiment 36](../../00036-hierarchy-aware-bsp-gsp-forecasts/result.md) had found that the sum of
Primary forecasts added nothing at the two BSPs.

**Three facts shape the design:**

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
state noise, helps the Primaries' filters.** It runs on the CPU in about 2 hours.

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
