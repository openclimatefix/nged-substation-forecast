---
id: "00103"
slug: differentiable-physics-1-data-look
title: "Differentiable physics, part 1: which substations, rows, and dates the model can use"
part: 1
theme: Long history and drift
also_relevant_to: [Pooling across substations, "Demand: calendar and weather", Embedded generation]
status: planned
size: >-
  medium, descriptive (no model). Stored: only private files under
  data/ml-experiments/00103-differentiable-physics-1-data-look/. Serving path: not touched.
  Degradation rule: not touched. Designs: one, since each output is a count or a plot. Callers:
  only studies.*, nged_data.storage, and plotting, all known.
hypothesis: >-
  The trial area's power records hold enough clean, demand-only history for the series to go ahead
  as designed: 6 MW Primaries with at least 12 months before July 2025, and 8 MVA Primaries with
  over 3 site-years each of rows with no sunlight.
prediction: >-
  Timestamp: correlation peaks at zero shift before and after 2026-03-26 08:30 UTC. Sites: D5, D9,
  D10, D11, D14, D16 each hold at least 12 months of clean history before 2025-07-01 after cuts;
  D18 holds fewer. Early calibration: cuts D14 before about June 2020 and D15's first fortnight
  (February 2025), no other series. MVA dark rows: each MVA Primary holds 3.2 to 3.4 site-years,
  about 27 in all, with about 27% at 16:00 to 21:59 local time. Generator screen: flags D15, G-a,
  G-b, D1, and D6, and none of the other 15 Primaries; D15's night level has no temperature slope
  (within ±0.1 points per °C). Clip: yes at D5 only. Heating-slope drift: not detectable.
  Lockdown: the weekday daytime departure supports leaving out March 2020 to June 2021.
decision_rule: >-
  Descriptive; each decision has a fixed rule (see "Decisions handed on" in the plan below).
  Verdict `helps` means the data supports Parts 2 and 3 unchanged or with only the predicted cuts;
  `hurts` means a finding forces a redesign (a timestamp offset, fewer than 5 usable long-history
  MW sites, or under 2 dark site-years at more than 2 MVA sites); `inconclusive` means a decision
  could not be made from the plots.
benchmark: none (no model)
row_set_id: none (descriptive; defines dp_mw_v1 and dp_mva_dark_v1 for later parts)
metric: counts, shares, slopes, and correlations
verdict: <helps | hurts | inconclusive>
prod: no
take_home_message: <written after the results>
date: <date>
pr: <pr>
---

# Differentiable physics, part 1: which substations, rows, and dates the model can use

**Plan only; the write-up replaces this page.** The series brief and the Part 1 brief fix the
design; this page records what is committed before any figure is drawn.

## Plan

- **Data.** Cleaned power (`nged_data.storage.scan_cleaned_power`) and the raw Delta table (for
  the availability figure), the 22-series list with units from the metadata table, ERA5
  `temperature_2m`, CAMS `ghi_w_m2` (end-of-hour labels), and the 66 hand-filtered switching
  transfers (`~is_sentinel`, `kind == "real_operation"`, visibility `conserved` or `one_sided`),
  masked from `onset_utc` to `end_utc` at the donor, both sides, and every receiver. Period 2019-09-01
  to 2026-06-30. CPU only, 4 threads.
- **Figures 1 to 11**, in order: availability by month with switching ticks; raw weeks; monthly
  level by hour band; early-calibration check; weekday and weekend profile heatmaps; clip look;
  night power against temperature; correlation of monthly levels; generator screen; timestamp
  convention; MVA dark rows.
- **Printed checks.** Duplicate keys, NaN and null power, stuck runs of 12 or more half-hours,
  zeros outside the `substation_zero` flag, the per-month step check, MVA |S| below zero, and the
  scarce-data census (cold days, bank holidays, Christmas-run days per site and scoring year).
- **Decisions handed on**, each with a rule fixed now: site list and cut dates; early-calibration
  cuts (drop leading months while a band differs by more than 15 points from the same months a year
  later); MVA dark-row counts (a site under 2 site-years leaves the pool); generator screen (flag if
  negative dark share exceeds 1%, or a winter night slope within ±0.1 points per °C with an
  amplitude-to-mean ratio under 0.3; G-a, G-b, and D15 are positive controls); timestamp convention
  (zero shift confirms; any other peak stops the series); clip yes or no (in at least 2 summers the
  slope above 600 W/m² is above −1 point per 100 W/m² while the slope below 500 W/m² is below −4);
  heating-slope drift (first and last winter intervals disjoint in the same direction at 2 or more
  long-history sites, winter 2020-21 left out); lockdown window (months where at least 3 of D5, D10,
  D11, D14 depart from the same month of 2023-24 by over 5 points in the weekday 09:00 to 16:59
  band); switching-mask census.
- **Files handed on** (private): `sites.parquet`, `rows.parquet` (flags `in_pool`, `early_cut`,
  `lockdown`, `switching_mask`, `dark`, `scarce_kind`), `decisions.json`.
- **Review pipeline.** One code review and one numbers review, a prose sweep before any merge.
  The series brief asks Opus for the code and the reviews; this session writes the code itself and
  asks the manager to relay reviews.

## Plan review triage

The Opus plan review made 18 findings. Each was checked against the files on disk before it was
accepted. Where a rule changes the series brief's meaning, the manager has been told rather than the
brief edited.

**Accepted, and changed in the plan:**

- **Lockdown rule (1).** Confirmed: the weekday-daytime level of D5, D10, and D11 drifts by more than
  5 points from 2023-24 in months far from lockdown. The rule now fits, per site, a calendar-month
  effect plus a linear trend on the months outside March 2020 to June 2021, and measures each
  month's departure from that fit. The lockdown window is the contiguous run of months containing
  April 2020 in which at least 3 of the sites with data depart by more than 5 points in the same
  direction (either sign counts).
- **Generator screen (2, 3).** The screen runs on rows after the early-calibration cuts and the
  switching masks. The second branch (flat night slope with a low amplitude-to-mean ratio) is
  dropped as a decision rule, because the mean is near zero for the battery series and none of the
  positive controls sits inside its thresholds; its inputs are still plotted. The negative-dark-share
  branch is the only decision rule, so the MVA Primaries, whose |S| is never negative, go unscreened,
  and the write-up says so. If the screen misses D15, G-a, or G-b, the thresholds are revised and the
  write-up says so.
- **Dark site-years (4).** Counted per window: all history after cuts with the lockdown months kept,
  the same with them removed, and the latest two season-years. The drop rule (under 2 site-years)
  applies to the window with lockdown removed. The 2-year window of Part 3 holds about 1 site-year
  per MVA Primary, so the rule cannot apply there; the manager is asked what Part 3 should use.
- **Site rules (5, 6).** `sites.parquet` reports months of history before each tuning origin
  (2022-07-01, 2023-07-01, 2024-07-01, 2025-07-01) as well as before 2025-07-01, so D9's roughly 10
  months before 2024-07-01 shows. The mapping to `hurts` is reported both ways and passed to the
  manager.
- **Early calibration (7).** The comparison month may fall after June 2026, using power, ERA5, and
  CAMS to their last available month, for this descriptive check only. The cut works in whole
  months, so the "fortnight" prediction becomes "D15's first months".
- **Join key (8).** A power half-hour joins the CAMS hour that contains it: stamps at hh:00 join hh:00
  and stamps at hh:30 join (hh+1):00. The panel build prints zero null irradiance values as the check.
- **Timestamp test (9).** Uses all daylight hours on clear-sky days (clear-sky index above 0.8), reads
  the correlation as negative and most negative at zero shift, reports the change from 0 to ±30
  minutes, and extends the after-correction side to the end of CAMS (2026-09-10) because the figure is
  not a scored statistic.
- **Privacy (10).** The two "battery or biofuel" series appear in no dated figure. Figure 1 shows
  them as G-a and G-b on a day-count axis, and every printed table and figure uses G-a and G-b.
- **Heating-slope drift (11).** Tests the difference between the first and last winters' slopes with
  a bootstrap of the difference, resampling whole months.
- **Flagged rows (12).** Read from `cleaned_power_time_series.delta`, whose `drop_reason` column is
  the only flag; the raw table carries none.
- **Switching mask (13).** The filter is copied into this folder, with experiment 101's convention:
  every half-hour from the one containing `onset_utc` to the one containing `end_utc`, inclusive.
- **Calendar (14, 15).** Days are local (Europe/London). Bank holidays are England and Wales, from the
  `holidays` package. `rows.parquet` holds every half-hour on the grid from the first usable time to
  2026-06-30 with a `present` flag, `season_year`, `day_type`, and the bank-holiday and Christmas-run
  flags.
- **Predictions (16, 17, 18).** The generator screen now also predicts D13 and D17 (bulk supply
  points, not flagged) and D1 and D6 (grid supply points). The run is about 2.4 million unflagged
  half-hours (not 3.5 million), starting 2019-09-17 at most series. The write-up marks which
  predictions were known from earlier rounds and so test nothing.

**Rejected:** none.
