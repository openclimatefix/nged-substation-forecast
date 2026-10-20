# Differentiable physics: embedded generation: research plan

| Id | Question | Priority | Waits for | State |
|---|---|---|---|---|
| 00110 | Can the convex model represent the generation side of a Primary substation (MVA meter, flat unmetered generator, export clip) without making its demand forecast worse, and does its solar capacity stay put when the demand terms are stopped from imitating solar? | High | The module pull request (stage 2 of the experiment) | Planned |
| 00113 | Does the convex per-site model identify each hour band's heating slope and the latest PV capacity at realistic noise, and are its 80% month-block bootstrap intervals calibrated? | High | Nothing | Planned |
| Wind node | A typed wind node beside the flat generator node | Proposed | 00110 | Not started |
| Dispatchable node | A typed dispatchable-generator node beside the flat generator node | Proposed | 00110 | Not started |
| Solar node | A typed solar-farm node beside the flat generator node | Proposed | 00110 | Not started |

The later parts of the differentiable-physics series are planned one at a time, from the result of
the part before.

## Planned experiments

<!-- planned-table -->
