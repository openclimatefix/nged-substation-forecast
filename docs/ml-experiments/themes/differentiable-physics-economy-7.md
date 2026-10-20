# Differentiable physics: Economy 7 and modern equivalents

**This page holds bullet-point notes on what we have found about Economy 7 and its modern
equivalents, written on 2026-10-20.** The notes are a working record, not a reviewed survey. Findings
from our own data are exploratory and come from experiments that were still being reviewed when this
page was written. Claims about tariffs and meters come from the documents linked below, and the ones
we could not confirm in a document we read in full are marked as unverified.

## Why it matters

- Economy 7 is a time-of-use electricity tariff with a cheap seven-hour overnight window, used
  mostly with night storage heaters and immersion water heaters.
- NGED is expected to see the forecast model for each Primary substation for the first time in
  winter. A visible night step in demand that the forecast does not reproduce is the kind of miss an
  operator notices.
- The average gain from modelling the step is small, so it is judged on cold winter nights as well
  as on the overall score.  ## What we see in Primary substation demand
- **The step is real and visible at 13 of the 14 Primaries tested.** In experiment 123, the 14
  Primaries are the six main test Primaries and eight more with long histories, added as a check.
  The pooled estimate over the six main Primaries at a daily mean temperature of 3 °C is 9.9 points
  of effective capacity, with an interval of 6.2 to 13.5. The fitted step at single Primaries runs
  from 0.5 to 33.5 points (median about 13), or 0.05 to 2.67 MW. The one exception, D8, has a step
  of 0.5 points (interval -0.7 to 1.7).
- **Units.** A point is 1% of a series' own 99th percentile of cleaned demand. It is not an
  electrical rating. At the three Primaries where the step first showed (D5, D10, D16), the raw step
  on cold nights is 4.8, 8.2, and 9.5 points (0.27, 0.44, and 0.87 MW). The fitted step at 3 °C is
  larger, at 10.8, 15.4, and 13.4 points.
- **The step starts in the half-hour from 00:30 UTC and ramps over about one half-hour.** It reaches
  7% to 56% of its plateau in the half-hour starting 00:00 and 81% to 119% by 00:30.
- **The step is fixed to UTC, not to the local clock.** On cold British Summer Time nights the
  largest rise sits at the UTC step on 24% to 100% of nights per series, and at the local-clock step
  on at most 11% of nights. This is a post hoc finding. No local-clock mixture is shown, though one
  is not ruled out.
- **No Primary tested shows a second window, meaning a second rise of at least 30% of the main
  one.** The end of the window is not visible, because
  the morning rise overlaps it.
- **The step grows only modestly with cold.** The slope is positive at all 14 series, but the
  cold-dependent part is only 1.1 to 4.7 points at the six main Primaries. Most of the step is a
  flat winter-versus-summer difference.
- **The step changes between winters.** From winter 2021-22 to 2024-25 it falls at five series,
  rises at one, and cannot be told at eight. Some series have fewer winters (D9 has two). Winter
  2025-26 is held back until the base model's pull request merges.
- **The baseline GAM with a Kalman filter leaves most of the step unmodelled.** The baseline leaves
  0.53 to 0.78 of the step at D5, D10, and D16. On cold nights it forecasts below actual from 01:00
  at D5 and D16.
- **The step was easy to miss.** A first test of night-step arms in experiment 121 was not powered,
  because a step in one night half-hour is diluted in a score over the whole night. Scoring the
  01:00 to 03:00 rows on their own detects it.

## What the documents say about the tariff

- **The standard Economy 7 window is 00:30 to 07:30 GMT, seven hours.** Elexon's profiling appendix
  allows regimes up to one hour either side: 00:00 to 07:00, 01:00 to 08:00, 01:30 to 08:30, and
  23:30 to 06:30 ([Elexon
  CP1058a](https://assets.elexon.co.uk/wp-content/uploads/sites/11/2012/02/28171857/cp1058a.pdf)). A
  01:00 GMT start is therefore an allowed regime.
- **Radio teleswitch times are always UTC.** The settlement system converts them to clock time in
  summer only where a per-regime flag, the GMT Indicator, is set ([Elexon, Radio Teleswitch SSC
  Mapping,
  v3.0](https://bscdocscontent-si.elexon.co.uk/pdf/634531b91bf862000722fd97/Radio_Teleswitch_Standard_Settlement_v3.0.pdf)).
- **Which regimes follow GMT all year and which follow BST is unresolved.** Two Elexon documents
  describe the flag in opposite senses ([Elexon, MDD made easy
  v4.0](https://assets.elexon.co.uk/wp-content/uploads/sites/11/2014/02/28164734/mdd_made_easy_v4.0.pdf)).
  The meter timeswitch table that settles it is behind the Elexon portal login.
- **Smart meters are unverified.** The documents we read in full say only that a smart meter's
  switching calendar can be set to match the old radio-teleswitch group. They do not say whether a
  smart meter's window follows BST.
- **A smart meter can hide an Economy 7 window.** Meter timeswitch codes were defined before smart
  meters, so a smart meter cannot be identified from its code ([Elexon BSCP40 Issue
  67](https://www.elexon.co.uk/wp-content/uploads/2016/08/14_265_08_Issue_67_Issue_Report_PUBLIC.pdf)).
- **Of the NGED areas, only the East Midlands appears in the radio-teleswitch mapping.** Its
  seven-hour Economy 7 regimes are all dynamic, with no switching times given. West Midlands, South
  West, and South Wales were not found.
- **A regional window table in an LLM-written summary is unverified, so we do not use it.** It
  claimed split windows in the South West and meters locked to GMT. We found no source for the
  regional rows.

## How many meters, and how the stock is changing

- **NGED's four areas hold 1,141,937 of the 3,249,338 Economy 7 and white-meter points in Great
  Britain (35%), as of November 2022.** The East Midlands has 671,198, the second largest of the 14
  areas. The Midlands has 242,512, the South West 183,654, and South Wales 44,573 ([Ofgem, call for
  input,
  2023](https://www.ofgem.gov.uk/sites/default/files/2023-02/TDCV%202023_Call%20for%20Input.pdf)).
- **Economy 7 households use more electricity.** Ofgem gives 4,748 kWh a year for Economy 7 against
  3,245 kWh for the standard class, and about a 60/40 split between peak and off-peak.
- **The radio teleswitch service is being closed.** Meters fell from over 600,000 in January 2025 to
  under 154,000 in January 2026. Withdrawal went by regional group code from 30 June 2025, paused
  over winter, and restarted on 2 March 2026 ([Ofgem, RTS decision, 5 March
  2026](https://www.ofgem.gov.uk/sites/default/files/2026-03/Ofgem%20Radio%20Teleswitch%20(RTS)%20policy%20consultation%20decision%20and%20update%20on%20phase-out.pdf)).
- **The closure may change the step within our history.** Ofgem says a smart meter or a preset meter
  replaces each radio-teleswitch meter. Whether the new window follows GMT or the local clock is not
  stated. The regional order of withdrawal was not found.

## Modern equivalents, ranked for modelling

- **Intelligent Octopus Go (23:30 to 05:30) is growing fast.** The window comes from Octopus notices
  quoted on third-party pages, so it is unverified. Electric vehicles on the tariff rose from
  156,000 to 278,000 between April 2024 and April 2025 ([Octopus Energy Ltd annual report
  FY25](https://octopusenergy.group/static/documents/OEL_Annual_Report_FY25.pdf)). Extra
  smart-charge slots shift from day to day, so only the fixed window is a daily shape.
- **Octopus Go (00:30 to 05:30) is a fixed window** ([Octopus
  Go](https://octopus.energy/smart/go/)). The window comes from a web-page summary we did not read
  in full, so it is unverified. We found no customer count.
- **Cosy Octopus for heat pumps has three cheap windows** (04:00 to 07:00, 13:00 to 16:00, and 22:00
  to 24:00) and a peak from 16:00 to 19:00 ([Cosy
  Octopus](https://octopus.energy/smart/cosy-octopus/)). These times are also from a web-page
  summary and are unverified. These show as midday and 22:00 changes, not a night step. Heat pumps
  are still few.
- **Clock type for the Octopus tariffs is unverified.** Some forum posts say smart-meter windows
  follow the local clock, but they are anecdotal.
- **Out of scope:** Agile, Tracker, Saving Sessions, and the national Demand Flexibility Service.
  They respond to prices or events, not to a fixed daily window.
- **Radio-teleswitched dynamic tariffs** (Total Heat Total Control and Weathercall) are
  signal-driven and ending, so they are not modelled.  ## What this means for the model
- **The night step is a daily shape fixed in UTC, so the model indexes night shapes by UTC hour.**
  Night shapes are also indexed by local hour, so a local-clock tariff can show up next to the
  UTC-fixed step.
- **The step size is allowed to change over time, per substation.** A slowly changing term, with a
  non-negativity bound, replaces a single fixed coefficient. The UTC share can change slowly too.
- **The learned shape library (experiment 120) is designed to learn a 01:00 shape from the data.**
  Whether it finds the step unaided is a planned test, with an injected step and a placebo first. A
  hand-built night-step arm sets the benchmark.
- **Substation cards show the fitted step.** Each card shows the step in points and MW, its start
  time, its size by winter, and the share that follows UTC against the local clock.
- **Calendar and visible-feature checks.** A forecast-against-actual figure for cold winter nights
  goes on each card. A visible miss on a cold winter night counts against release.
- **The Octopus and Cosy tariffs are not visible at Primary level, so they are not modelled.** A
  check in experiment 123 looked for a rise at the window edges on British Summer Time nights
  (Octopus) and on winter weekdays (Cosy), by year from 2021. The placebo control failed: at
  half-hours with no tariff edge the interval excluded zero in 76% to 79% of series-years, because
  the daily load shape is not flat. The year-on-year change cancels the shape, but the placebo
  false-positive rate was still 21% to 43%.
- **The detection limit is 1.19 points (about 0.06 to 0.22 MW).** Beyond it, with the expected sign,
  only one Primary (D8) shows the Intelligent Octopus Go edges at 23:30 and 05:30, and that Primary
  has no Economy 7 step and a large fall at 16:00, so it is a load change. No Primary shows the
  Octopus Go opening at 00:30. Cosy is not distinguished from other load changes.

## Gaps

- The meter timeswitch table of which regimes follow GMT or BST.
- The smart-meter specification for the clock basis of tariff switching.
- Year-by-year Economy 7 meter counts, and NGED-area counts for Octopus tariffs, electric vehicles,
  and heat pumps.
- Winter 2025-26 data, which shows whether the radio-teleswitch closure changed the step.

## See also

- [Differentiable physics: background](differentiable-physics-background.md)
- [Differentiable physics: literature scan](differentiable-physics-literature.md)
