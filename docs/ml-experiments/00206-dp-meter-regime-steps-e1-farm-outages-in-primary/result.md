---
id: '00206'
slug: dp-meter-regime-steps-e1-farm-outages-in-primary
theme: differentiable-physics-embedded-generation
also_relevant_to:
  - solar-disaggregation
status: planned
size: >-
  simple. Stored: private tables under
  data/ml-experiments/00206-dp-meter-regime-steps-e1-farm-outages-in-primary/ only. Serving path:
  not touched. Degradation rule: not touched. Designs: one defensible design, because the brief fixes
  the test; the farm-to-Primary map is the one free choice and is fixed below before any scoring.
  Callers: this folder's scripts only. Asset graph: not touched.
hypothesis: >-
  When a metered solar farm's meter reads zero for days while the sun is high, the farm's output
  stays in the Primary substation's supplied power, so the Primary reads more solar generation
  (lower net power) during the outage than in the 14 days either side.
prediction: >-
  Written before any score. At least two thirds of the outages at farms whose assigned Primary is
  a decision Primary show a shift of 0.5 to 1.5 times the farm's 99th-percentile AC output, in the
  direction of more solar generation, with a 95% interval that excludes zero. If the sign is the
  other way (the Primary reads less solar generation), the Primary's supplied power includes the
  farm all the time, and a real farm outage then lowers the Primary's generation.
decision_rule: >-
  Supported when at least two thirds of the tested outages at farms assigned to a decision Primary
  (D5, D10, D11, D14, D16) or D9 show the shift above. If fewer do, experiments 207 (E2) onwards of
  the brief are not worth running for the mechanism, and the page says whether the shift is absent,
  or is present in the opposite direction. The placebo rate (below) must be under one third for the
  result to count.
benchmark: >-
  Placebo windows: for each farm, 200 windows of the same length as that farm's median outage,
  drawn from days with no flagged farm half-hour, tested with the identical rule.
row_set_id: >-
  None: no forecast is scored. Rows are the Primary's raw half-hourly supplied power and the farm's
  cleaned series (experiment 179, rows with a null drop_reason are the reference).
metric: >-
  Per outage and Primary: mean sunny-daylight power inside the outage minus the mean over the 14
  days before and the 14 days after, divided by the farm's 99th-percentile AC output (the capacity
  unit of experiment 129), with a 95% day-resampling interval.
code: ml-experiments/00206-dp-meter-regime-steps-e1-farm-outages-in-primary/
---

# E1: do a solar farm's meter outages show up in its Primary's power?

*The plan is committed before any shift is computed. A result section replaces this one when the
experiment finishes.*

**Question.** NGED removes a metered solar farm's output from the supplied power of the Primary
substation it sits under. If a farm's meter fails, the farm keeps generating, the removal stops, and
the Primary appears to gain unmetered solar capacity. This experiment asks whether the farms'
zero-output runs in the clear-sky hours are visible in the supplied power of nearby Primaries.

**Pre-registered rules.** (Farms A to F are anonymised; Primaries are labelled D1 to D22.)

- **Outage:** a run of at least 3 consecutive UTC days on each of which at least 80% of the
  clear-sky half-hours (clear-sky horizontal irradiance above 500 W/m² at the farm, from the
  Ineichen model) are either flagged `zero_in_clear_sky` in experiment 179's cleaned series or have
  no row ("missing"). Runs made mostly of flagged zeros are `zero` outages and runs made mostly of
  missing rows are `missing` outages; the decision uses the `zero` outages, and `missing` outages
  are reported beside them. Farm E counts only from its commissioning date, 2024-10-06.
- **Farm-to-Primary map:** NGED's hierarchy file gives each farm's balancing-grid-supply-point
  parent for five of the six farms but no Primary, so the assigned Primary of a farm is the nearest
  Primary (great-circle distance) that reports in MW and has the same bulk supply point as the farm,
  where known. The Primaries that report in MVA cannot show the sign of a shift and are listed as
  untestable. Every other MW Primary within 10 km of a farm is tested as well, as a non-assigned
  control, because a non-assigned Primary should show no shift.
- **Sunny daylight half-hours:** clear-sky horizontal irradiance above 300 W/m² at the Primary, and
  CAMS irradiance at least 80% of clear-sky.
- **Shift:** for each half-hour of the day, the mean power in the outage minus the mean of the
  14 days before and the 14 days after (days that are themselves in an outage excluded), averaged
  over the half-hours of the day that have at least 3 sunny samples in both windows. An outage is
  tested only if it has at least 100 sunny half-hours inside it and at least 300 in the reference
  windows. The interval resamples days.
- **Shows the shift:** the shift lies between 0.5 and 1.5 farm-capacity units in the direction of
  lower net power, and its 95% interval excludes zero.
- **Reporting:** the farm-to-Primary map (with distances), the number of outages found and tested,
  and the share that show the shift, split by `zero` and `missing`, assigned and non-assigned
  Primaries, and direction. Absolute shifts in MW come beside the ratios.

**Sizing (five triggers).** Stores private tables only; not on the serving path; no degradation
rule; one defensible design; callers are this folder's scripts. Simple, with one Sonnet code review
and one Sonnet write-up review.
