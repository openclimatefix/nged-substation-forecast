# Differentiable physics: the network: research plan

**The proposed next steps are a topology check on candidate flows, or a smaller level penalty
combined with a check for unlabelled jumps; neither option is planned.** [Experiment
116](../../00116-differentiable-physics-nra-edge-flows-between-primaries/result.md) ended
`inconclusive` because the known-answer gate failed: every setting that found injected transfers
also found flow between substations with no interconnection. [Experiment
131](../../00131-differentiable-physics-nra-edge-flows-drift-term/result.md) added a per-substation
drift term. At equal or higher recall the drift term cuts the flow on unconnected pairs by half to
three quarters, but no setting meets the recall targets and the placebo limit together, and the level
penalty shrinks the flows the fit finds.

## Priority table

| Experiment | Question | Priority | Waits for | State |
|---|---|---|---|---|
| BSP top-down against bottom-up (experiment 161) | Is a bulk supply point forecast better from its metered-MW child Primaries' forecasts plus a learned remainder than as a stand-alone series, with the best-guess GAM-K chain on both sides? | 2 | Nothing; CPU, a few hours | Planned |

## Planned experiments

<!-- planned-table -->
