# Plan: PVDIS-T, the embedded-solar capacity table (study)

**Question.** How much solar generation sits behind each Primary substation metered in MW in the trial area, per season-year since 2019/20, and how far can each number be trusted?

**Method.** No refit. Read experiment 00040's stored fits (`real_vectors`, `bootstrap`, `unit_vectors`) and experiment 00039's frame. Report the cross-arm capacity C as the capacity, with its 200-draw seven-day-block bootstrap interval. Keep the daily fit M1d / R only as a cross-check column. Recompute the bound B on C and the amplitude it was read at with the 00040 detector, moved into `packages/studies` with tests.

**Size and the five triggers.** Complex, because a published page counts as what gets stored. Changes what gets stored: yes (a page and a table). Touches the production serving path: no. Touches a degradation rule: no. Admits more than one defensible design: yes (which estimator is the headline; how the trend interval is built). Spans code whose callers cannot be named without searching: no. Diff reviews: cut-down (Sonnet), correctness (Opus is replaced by the two science reviews), mutation pass because `packages/studies` changes.

**Planned checks (named before any output).**

1. P1 reproduction: the study's C, T, tau, flags and B_dayL equal 00040's stored `real_scores` to 1e-9.
2. P2 headline estimator: C is the capacity column. The ratio M1d / R over C is reported, with Kirton's separately.
3. P3 trend: log-linear growth per clean Primary with at least 6 season-years, interval from a bootstrap over season-years, compared with the sampling-only interval from the per-draw trajectories.
4. P4 Kirton is left out of the capacity table and carries only the library range.

Everything else (the clear-minus-overcast gap, the evening slope, the M1d seventh-arm re-score) is exploratory.

**Table status rules.** `clean`: unflagged, and the day-long load is detected at least 80% of the time at 0.25 of the capacity (bound amplitude 0.25 or less). `low-power`: unflagged, with bound amplitude 0.5 or more. `flagged`: T above tau. `left-out`: Wrangle T1 and the MVA Primaries. Thresholds are fixed here, before output, and reported with counts.

**Figures.** The nine of `pvdis3_decision.md` section 3, raw data first, plus the C versus M1d / R scatter.

**Reviews.** Fresh Sonnet code review before first run; two Opus science reviews; first-stumble and prose review; figure text reviewed before rendering. No issue exists; the PR body says so.
