---
id: "00103"
slug: differentiable-physics-1-data-look
title: "Differentiable physics, part 1: which substations, rows, and dates the model can use"
part: 1
theme: Long history and drift
also_relevant_to: [Pooling across substations, "Demand: calendar and weather", Embedded generation]
status: done
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
  D18 holds fewer. Early calibration: cuts D14 before about June 2020 and D15's first fortnight,
  no other series. MVA dark rows: each MVA Primary holds 3.2 to 3.4 site-years, about 27 in all,
  with about 27% at 16:00 to 21:59 local time. Generator screen: flags D15, G-a, G-b, D1, and D6,
  and none of the other 15 Primaries; D15's night level has no temperature slope (within ±0.1
  points per °C). Clip: yes at D5 only. Heating-slope drift: not detectable. Lockdown: the weekday
  daytime departure supports leaving out March 2020 to June 2021 (superseded: the series brief now
  fixes the window at 2020-03-23 to 2021-07-19).
decision_rule: >-
  Descriptive; each decision has a fixed rule, listed in the bullets below. Verdict `helps` means
  the data supports Parts 2 and 3 with no redesign; `hurts` means a finding forces a redesign (a
  timestamp offset, fewer than 5 eligible long-history MW sites at the first tuning origin, or
  under 0.5 dark site-years in Part 3's window at more than 2 MVA sites); `inconclusive` means a
  decision could not be made from the plots.
benchmark: none (no model)
row_set_id: none (descriptive; defines dp_mw_v1 and dp_mva_dark_v1 for later parts)
metric: counts, shares, slopes, and correlations
verdict: helps
prod: no
take_home_message: >-
  The data lets the series go ahead, but three rules written before the data was read need
  changing. Six MW Primaries hold 12 or more months of clean history before July 2025 (D9 holds only
  10 months before the first tuning year), all 8 MVA Primaries hold 2.4 to 2.7 site-years of dark
  rows with the lockdown months removed, and no timestamp error of a whole half-hour is present.
  Three findings change Part 2: nine of the 16 Primaries need leading months cut, the data does not
  isolate the lockdown months from drift (so the window stays fixed in advance), and Part 3's
  2-year window leaves each MVA Primary only about 0.95 dark site-years, which still clears the 0.5
  needed to enter.
date: 2026-10-19
pr: 1348
advice:
  - >-
    Cut the leading months listed in the site table before fitting (D14 nine months, D12 eight, D4
    three, D18 two, and one each at D9, D16, D21, and D22), because their first months sit more than
    15 points from the same months a year later (figure 4).
  - >-
    Leave the fixed lockdown window (2020-03-23 to 2021-07-19) out of every pool except the
    COVID-flag arm, and do not try to detect it: weekday daytime levels at D5, D10, D11, and D14
    drift above a no-lockdown fit through 2021, so a data-defined window would measure drift
    (figure 12).
  - >-
    Admit all 8 MVA Primaries to Part 3: each holds 0.93 to 0.96 dark site-years in the latest two
    season-years, above the 0.5 needed, and 2.4 to 2.7 over all history with the lockdown removed
    (dark site-years bullet).
  - >-
    Count 5 eligible long-history MW sites (D5, D10, D11, D14, D16) at the first tuning origin, 2024-07-01,
    not 6, because D9 holds 10.0 months before it (site table).
  - >-
    Screen generators on rows after the early-calibration cuts and the switching masks: D18's 1.7%
    negative dark share falls to 0.009% once its first two months are cut (generator screen).
  - >-
    Allow for an offset under half an hour in the power stamps: the correlation with CAMS peaks
    between 0 and +30 minutes (parabola vertex +7 to +11 minutes) and a whole half-hour error costs
    over 0.04, so no correction is needed but the convention is not exact (figure 10).
production_status: not-for-production
adopted_in: none
code: ml-experiments/00103-differentiable-physics-1-data-look/
---

# Differentiable physics, part 1: which substations, rows, and dates the model can use

**The data lets the differentiable-physics series go ahead, but three rules written before the data
was read need changing.** Six MW Primary substations hold at least 12 months of clean demand history
before July 2025, and all 8 MVA Primaries hold between 2.4 and 2.7 site-years of dark rows (rows
with no sunlight) once the lockdown months are removed. Three findings change Part 2: nine of the
16 Primaries need leading months cut because their first months do not match later years, the
data does not isolate the lockdown months from drift (so the window stays fixed in advance), and
the 2-year window that Part 3 uses leaves each MVA Primary about 0.95 dark site-years, above the 0.5
needed to enter. Verdict: `helps`, with those three changes.

This experiment fits no model. It looks at the power records of 22 series (16 Primary substations,
four grid or bulk supply points, and two metered generators), at ERA5 temperature and CAMS
irradiance (both estimates of past weather), and at 66 hand-filtered switching events, and decides
which substations, rows, and dates the later parts can use. A Primary substation is the transformer
site that feeds a town or a district, and an MVA Primary reports apparent power instead of net
power, so only its rows without sunlight reflect demand alone.

## Figures

Figures 1 and 2 show the raw data, figures 3 to 8 test the planned modelling choices, figures 9 to 12
test the screens and conventions. The two metered generators appear only in figure 9, as G-a and G-b,
as aggregate statistics with no dates.

![Availability](figure_01_availability.svg)

![Raw weeks](figure_02_raw_weeks.svg)

![Monthly level by hour band](figure_03_monthly_levels.svg)

![Early calibration](figure_04_early_calibration.svg)

![Daily profiles](figure_05_profiles.svg)

![Clip look](figure_06_clip.svg)

![Night power against temperature](figure_07_heating.svg)

![Correlation of monthly levels](figure_08_correlation.svg)

![Generator screen](figure_09_generator_screen.svg)

![Timestamp convention](figure_10_timestamp.svg)

![MVA dark rows](figure_11_mva_dark.svg)

![Lockdown](figure_12_lockdown.svg)

## Sites and dates

- **Roles after the cuts** (`sites.parquet`): `mw_long` for D5, D9, D10, D11, D14, and D16;
  `mw_short` for D18; `generator_case` for D15; `mva_dark` for D2, D3, D4, D8, D12, D19, D21, and
  D22; `excluded` for D1, D6, D13, D17, G-a, and G-b.
- **First usable dates:** D5, D10, D11, D2, D3, D8, and D19 start 2019-09-17; D4 2019-12-01; D21 and
  D22 2019-10-01; D12 2020-05-01; D14 2020-07-01; D16 2021-01-01; D9 2023-09-01; D18 2025-02-01; D15
  2025-10-01 (its cut reflects a level change, not calibration, and D15 is out of the demand pool).
- **Months of history before the first instant of each tuning year** (2022-07, 2023-07, 2024-07,
  2025-07): the 6 long-history sites hold 12 or more months before 2025-07-01 (D9 22.0). Five of
  the six hold 12 or more months before each of the earlier three origins; D9 holds none before
  2023-07 and 10.0 before 2024-07. D18 holds 4.9 months before 2025-07.
- **Dark site-years per MVA Primary:** 2.91 to 3.27 over all history with the lockdown kept (25.5
  in total), 2.36 to 2.68 with the fixed lockdown window removed (20.8 in total), and 0.93 to 0.96
  in the latest two season-years (7.6 in total). The share of dark half-hours at 16:00 to 21:59
  local time is 27.4%, 28.1%, and 27.3% in the three windows. No MVA |S| is below zero.
- **Row sets:** `dp_mw_v1` holds 631,017 in-pool half-hours at the MW sites; `dp_mva_dark_v1` holds
  449,392 dark half-hours at the MVA sites (`rows.parquet`, full half-hourly grid with flags).

## Decisions

- **Early-calibration cuts.** Rule: drop leading months while either band differs by more than 15
  points. Result: D14 nine months, D12 eight, D15 eight, D4 three, D18 two, and one month each at
  D9, D16, D21, and D22 (figure 4). The D9, D16, and D18 cuts remove a part-month of near-zero
  readings after the meter starts. These dates are provisional: the series design session's
  per-series review recommends first usable dates, and Part 1 may only move them later.
- **Generator screen** (negative dark share above 1% only, run after cuts and masks). Flagged: D1
  (73%), D6 (26%), D15 (74%), G-a (62%), G-b (65%). Not flagged: D17 (0.97%, on the limit), D13
  (0%), D18 (0.009% after its cut; 1.7% before it, all in January 2025), and the other 14 Primaries
  (figure 9). The MVA Primaries go unscreened, because |S| is never negative.
- **Timestamp convention.** On clear-sky hours in April to September, the mean correlation across
  the 8 MW Primaries is most negative at zero shift on both sides (before −0.563, after −0.659)
  and falls to −0.556 and −0.646 at +30 minutes. Per site, the peak is at 0 for 1 site before the
  fix and 5 after, at +30 for 6 before and 2 after, and at −30 for D5 on both. A parabola through
  −30, 0, and +30 has its vertex at +10.6 and +6.9 minutes. A whole-half-hour error would cost
  over 0.04 and none is found, so the stamps are left as they are (figure 10).
- **Clip.** Yes at D5 only (4 of 7 summers meet the rule; D14 meets it in 1 of 6).
- **Heating-slope drift.** No. No long-history site has a last-minus-first winter difference whose
  95% interval excludes zero (D9's runs from −0.06 to +0.26 points per °C; the rest are wider).
- **Lockdown window.** Fixed by the series brief at 2020-03-23 to 2021-07-19 (experiment 6's COVID
  indicator) and left out by default; Part 4 has one arm that keeps it with a COVID flag. A window
  detected from the data would not work: after removing a calendar-month effect and a linear trend,
  April 2020 is more than 5 points below the fit at only D11 (−26) and D10 (−7), and the months from
  August 2020 to September 2021 sit mostly 5 to 17 points above it at D5, D10, D11, and D14 (figure
  12), which is drift or growth, not lockdown.
- **Correlation of monthly levels** across D5, D10, D11, D14, and D16 (D9 left out): the mean
  pairwise correlation is 0.53 for the night band (60 months) and 0.43 for the evening band (59
  months), and the first principal component holds 64% and 55% of the variance (figure 8). The
  shared slow factor is therefore worth testing in Part 2.
- **Switching-mask census.** 16,963 half-hours are masked on the 16 Primaries, the most at D5
  (3,682) and the least at D2 (17). The 66 events start in 2022-03, so there are no masks before
  then.

## Predictions

- **Held:** D5 only for clip; heating-slope drift not detectable; the six long-history sites hold
  12 months or more before 2025-07-01 and D18 holds fewer (known in advance from round 3); the
  share of dark half-hours at 16:00 to 21:59 is about 27%; the screen flags D1, D6, D15, G-a, and
  G-b and none of the other 15 Primaries (D17 on the limit).
- **Failed:** timestamp peak at zero on both sides (see above); early calibration cuts no series
  other than D14 and D15 (nine series are cut); each MVA Primary holds 3.2 to 3.4 site-years (D12
  holds 2.91, and the total is 25.5, not about 27); D15's winter night slope is within ±0.1 points
  per °C (it is +0.48); the weekday daytime departure supports leaving out the lockdown months
  (the window is now fixed in advance, and the departures do not isolate it).

## Data validation

- **Duplicates, NaN, and nulls:** no duplicate `(series, time)` keys, no NaN power, and no null CAMS
  irradiance or ERA5 temperature on the panel to 2026-06-30.
- **Stuck runs of 12 or more identical half-hours:** found only at the two metered generators.
- **Zeros outside the `substation_zero` flag:** found only at the two metered generators.
- **Per-month step check:** steps over 3 monthly standard deviations at D11 (2020-04), D12, D18,
  D19, D2, D21, D4, and at D6, whose monthly mean falls by 218 MW in 2023-07. The metered
  generators are not listed with dates.
- **Scarce-data census** (mean per long-history MW site): cold days (daily-mean ERA5 temperature
  below 0 °C) were 3.8, 5.0, and 3.5 in 2023-24, 2024-25, and 2025-26, and 12.0 in the cold block of
  November 2022 to February 2023 (round 3 found 3.7 and 10.2). Bank holidays were 8 in a full year
  and Christmas-run days (24 December to 1 January) were 9.

## Caveats

- ERA5 and CAMS are estimates of past weather and flatter the PV term (experiment 46).
- The switching masks are hindsight labels, and there are none before 2022-03.
- The departures in figure 12 use four sites, and D14 enters only from July 2020.
- The first-month cuts at D9, D16, and D18 rest on very few readings (one night half-hour at D16).
- The code and every printed number are in `report.txt`.

## What the next experiment should answer

Part 2 should use the site table above, with the cut dates once the series design session's review
confirms them, and count 5 eligible long-history MW sites (D5, D10, D11, D14, D16) at the first
tuning origin, so the verdict needs 4 of 5. All 8 MVA Primaries clear the 0.5 dark site-year entry
rule in Part 3's 2-year window. The shared slow factor D2 is worth keeping, given the correlations
above.
