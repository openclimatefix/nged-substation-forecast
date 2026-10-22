# Differentiable physics: embedded generation: research plan

| Id | Question | Priority | Waits for | State |
|---|---|---|---|---|
| Wind node | A typed wind node beside the flat generator node | Proposed | A flat node that stays off where no generator exists (experiment 110 found it reads 12 to 35 points at all five decision substations) | Not started |
| Dispatchable node | A typed dispatchable-generator node beside the flat generator node | Proposed | A flat node that stays off where no generator exists (experiment 110 found it reads 12 to 35 points at all five decision substations) | Not started |
| Solar node | A typed solar-farm node beside the flat generator node | Proposed | A flat node that stays off where no generator exists (experiment 110 found it reads 12 to 35 points at all five decision substations) | Not started |
| 250, monotone solar network | Does a small monotone neural network between the solar forecast and the demand model beat the hard clip? | High | None | Planned |
| N13, PV orientation mix | Does replacing the single south-facing solar shape with three non-negative capacities (south, east, west) keep the model convex and lower the summer morning and evening error? | Dropped | None. [Experiment 73](../../00073-pvdis-6-orientation-mix-and-dc-ac-ratio/result.md) and [experiment 77](../../00077-pvdis-6b-fitted-solar-shape-in-arm-am/result.md) found the mix unidentified or immaterial, and experiment 114 dropped it | Dropped |

## N13: PV orientation mix

**Experiment 114 dropped N13, so the design below is not queued.** It stays as the design for
any later experiment on solar orientation.

**The proposal gives the convex model three non-negative solar capacities, one each for south, east,
and west, in place of one south-facing unit shape.** The model stays convex, and an east-west array
should get a larger east and west share. The shares count as identified only if they pass the
coverage rule of experiment 113, which asks that 80% intervals cover the injected truth 70 to 90% of
the time. The experiment passes if the shares are identified and the score does not get worse. The run
takes minutes on the CPU. It was first considered for the same bundle as the interaction-term and
heating-threshold experiment, which ran without it as experiment 114.

The later parts of the differentiable-physics series are planned one at a time, from the result of
the part before.

## Planned experiments

<!-- planned-table -->
