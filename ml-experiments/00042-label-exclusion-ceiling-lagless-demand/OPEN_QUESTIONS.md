# Experiment 00042: open questions and choices made on the most literal reading

1. **Fits are not reproducible when workers share the GPU.** A refit of one series on identical
   rows, run at the same time as two other fits (3 workers, 2 threads each, GPU), differed from the
   stored fit by up to 0.036 of capacity in the median (20 rounds) in 2 of 4 repeats. Three fits run
   alone, one after another, were bit-identical. `e00042_fit.py --determinism-check` therefore fails
   under the production settings, and the plan's fallback applies: run the fit with `--refit-all` so
   that every series is refitted in every arm. Decision needed: accept `--refit-all` (about 3 times
   the fits), or fit with one worker, or switch to the CPU.
2. **ERA5-trained rows.** Experiment 00033's `era5_retrained` reused the ENS control member's rows
   (runs from 2024-04-01) and fed ERA5 at the valid time. This experiment follows the plan instead:
   the ERA5 source uses a synthesised run layout (a run every day at 09:00 UTC, 702 half-hourly
   leads) for valid times 2022-01-01 to 2025-06-30, with ERA5 weather at the valid time, and both
   sources are scored on the 51 ENS members. The first 15 days of 2024-04 hold more runs per valid
   time than 00033's rows did. ERA5 features exist from 2019-09 in `era5_features.parquet`.
3. **Trusted rule.** Events with no receiver, and events whose recomputed share is missing (19 of
   the 66 visible events), cannot meet the receiving-load criterion and are not trusted (an earlier
   version let a missing share pass the test). This leaves 8 trusted events: 7 in the ERA5 window
   and 3 in the ENS window. The ENS window sits exactly at the limit of the "fewer than 3" clause,
   so a single further exclusion would move that source's verdict to C4.
4. **Event counts against the plan.** The ERA5 window holds 48 visible events (as the plan says),
   of which 27 (not 26) last longer than 1 day, with 147.4 clipped event-days. The ENS window
   matches (14 events, 9 longer than 1 day, 49.2 event-days).
5. **Overlap convention.** A stamp is the end of its half-hour, so a half-hour is inside an interval
   when the stamp is after `onset_utc` and 30 minutes before the stamp is before `end_utc`.
6. **Random blocks.** Each block starts in the calendar month of the onset of the earliest real
   event that overlaps it, in any year of the window. "Window edges" is read as: the block lies
   wholly inside the window. "Any labelled interval of any visibility" is read as every `real_*`
   row of the labelled set with an onset that names the series. Blocks are the maximal runs of the
   union of a series' clipped intervals, so overlapping events merge into one block (kept). A
   placed block must remove a number of finite readings within 10% of the real block's, widened to
   25% if no placement exists. Where neither works, the closest clear placement in any month is
   used and the console says so. This happens for two series: one whose only matching windows lie
   inside the real event (real block 395 finite readings, random blocks 0), and one whose real
   block has 3 finite readings (random blocks 195). The per-series ratio of finite readings removed
   (random over real) is in the exclusion report.
7. **Minimum detectable gap.** Defined as `1.96 * sqrt(2) * s`, where `s` is the sample standard
   deviation across the 3 model seeds of the no-exclusion mean quantile score outside switching on
   the affected series.
8. **Per-event view.** Read as the mean real-minus-random gap on an affected series in each
   calendar month (of the valid time) of the scoring year in which a visible labelled event names
   that series.
