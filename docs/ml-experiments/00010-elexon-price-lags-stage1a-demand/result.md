---
id: "00010"
slug: elexon-price-lags-stage1a-demand
status: planned
size: medium
hypothesis: >-
  Adding the Elexon Market Index Data price from exactly 7 days earlier to the ERA5-trained stage 1a
  demand model lowers its error at the substations that host price-responsive batteries, and does so
  by more than adding the same-lag national demand and wind output, which price follows.
prediction: >-
  Against the experiment 00006 Primary input set, the price columns lower the mean absolute error of
  the median at the listed substation (B2) by at least 0.5 points of capacity, and by less than 0.1
  points at the unlisted series.
decision_rule: >-
  Supported if C1 and C2 are both material at B2: the 98.3% month-resampled interval (95%
  familywise,
  Bonferroni across the three verdict contrasts C1 to C3) excludes zero and the point estimate is at
  least 0.1
  points in the
  direction stated. Refuted if the C1 interval at B2 excludes a gain of 0.3 points. Otherwise
  inconclusive. The selection year decides; the confirmation year is scored once, and if its
  C1 interval at B2 excludes the selection year's point estimate the verdict is inconclusive.
  Before any fit, the half-width of B2's interval is computed from experiment 00006's per-month
  losses; if it exceeds 0.3 points the refuted branch is dropped and the best possible verdict is
  supported or inconclusive. C4, the listed against unlisted comparison, rests on
  two listed series and is reported but cannot change the verdict.
benchmark: >-
  The experiment 00006 Primary input set with recency weights (arm BASE), and a month-by-half-hour
  climatology row.
row_set_id: era5_val_2024_07_to_2025_06_v1
metric: mean absolute error of the median, points of capacity (power over effective capacity, times
100)
verdict: null
advice: []
production_status: null
adopted_in: none
code: ml-experiments/00010-elexon-price-lags-stage1a-demand/
---

# Does the lagged electricity price help the demand model at substations with batteries?

**Status: planned. No fit has run.** The code reuses experiment 00006's fit and report scripts by
copying their patterns.

## Why the price, and why only a lag

**Future prices are hard to predict, so the inputs are prices already published.** A battery that
charges when the price is low and discharges when it is high moves a substation's demand by an
amount the weather and the clock do not explain. The model sees this only if it is given the price.
The price at the same half-hour 7 days earlier is published well before any forecast issue time,
so it exists at forecast time for every lead up to 7 days.

## Price source (step 1, done)

**The price is the `APXMIDP` provider's price in Elexon's Market Index Data.** The data pull is in
`data/ml-experiments/00010-elexon-price-lags-stage1a-demand/prices/` and its `README.md` is the
lineage note. Findings, all from `e00010_fetch_prices.py`:

- **Cost and licence:** the Elexon Insights API needs no key and no payment. The script accepts no
  licence in anyone's name.
- **Coverage:** 134,357 of 136,416 half-hours (98.5%) from 2019-01-01 to 2026-10-12. 36 whole days
  are empty, all between 2019-01-09 and 2021-04-28. 54 more days are partial, 45 of them before
  2021-06. From 2024-04-14 the series is complete.
- **Product mix:** the other provider, `N2EXMIDP`, reports volume 0 in all but 385 of its 134,249
  rows, so only `APXMIDP` is usable. The API does not say which products feed the `APXMIDP` index.
  The plan therefore calls it the EPEX short-term half-hourly index, as `studies/market_downloads`
  does, and does not claim it is the day-ahead price.
- **Zero volume:** 175 `APXMIDP` rows report volume 0 (64 in 2019, 50 in 2021, 18 in 2022, 15 in
  2023, 11 in 2025, 12 in 2026, 5 in 2020). The experiment sets the price on those rows to missing.
- **Values:** -102.92 to 1,983.66 GBP per MWh, the maximum on 2021-09-09. No nulls, and the longest
  run of an unchanged price is 30 half-hours.

**The CONTROL columns need a second pull.** The demand outturn and fuel-mix downloads already in
`data/studies/downloads/market/` cover only 2025-09 to 2026-09, so the same free Insights API
pull is repeated for 2019 to 2026 before any fit, with the same validation.

**Training years with missing prices are not dropped.** XGBoost handles a missing value natively.
`price_lag7d` is missing when its half-hour is missing or has zero volume. The minimum and maximum
are missing when fewer than 40 of the 48 half-hours are available. About 23 half-hours in the
validation and confirmation years have a zero-volume price, so a few rows there carry missing
price columns; the report counts them. No lead beyond 7 days is scored.

## Design

**The base is experiment 00006's Primary input set**: the weather block (ERA5, read at the series'
nearest grid cell), hour and weekday, the COVID indicator, and the days to Christmas inside a 14-day
window, trained with recency weights (half-life 2 years). Folds, XGBoost settings (no column
subsampling), seeds, and the 7-day purge are those of 00006.

| Arm | Adds to BASE |
| --- | --- |
| BASE | nothing |
| PRICE | `price_lag7d` (price at the same half-hour 7 days earlier), `price_min_lag7d` and `price_max_lag7d` (the minimum and maximum over the 48 half-hours ending at the valid time minus 7 days, so nothing post-dates the valid time minus 7 days) |
| CONTROL | three columns with no price in them, matching PRICE's count: GB national demand outturn (Elexon `INDO`) at the same half-hour 7 days earlier, wind generation (Elexon `FUELHH`) at the same half-hour 7 days earlier, and the range of demand minus wind over the 48 half-hours ending 7 days earlier |
| SHUFFLED | the three PRICE columns, all taken from one other week of the same weekday and half-hour, drawn with a fixed seed from within 4 weeks of the true week (a negative control) |

**Weather and timing caveat.** The weather block is ERA5 read at the valid time, as in 00006, so
every score is a ceiling for the same reason 00005's was: ERA5 arrives about 5 days late. The
columns this experiment adds are available at forecast time because
they are at least 7 days old.

**Calendar memorisation control.** Hour and weekday are in BASE (as in 00006), so the price
columns can stand in for the calendar. SHUFFLED keeps the calendar structure and removes the
price's information, and it is the control for that.

## Planned contrasts

All are differences in mean absolute error of the median, in points of capacity, over the
selection year (2024-07 to 2025-06) and then, once, the confirmation year (2025-07 to 2026-06),
with a month-resampled interval at the level a Bonferroni correction over four contrasts sets.

- **C1:** PRICE − BASE.
- **C2:** PRICE − CONTROL (what the price adds beyond national demand and wind).
- **C3:** PRICE − SHUFFLED (what the price adds beyond the calendar structure of the price columns).
- **C4:** the gain PRICE − BASE at the listed series minus the gain at the unlisted series.

## The listed substations, chosen before any fit

**The list comes from NGED's Embedded Capacity Register (August 2026 release), which is
independent of the outcome.** A series is listed if the register's connected storage rows under
the series' name sum to a maximum export capacity of at least 5% of the series' effective capacity.
The 5% threshold is a round number chosen before the sums were run; P4 passed it by 0.1 points,
which is why the plan reports it only beside B2. The register has no commissioning dates, so a
listed battery may have been absent for part of the training years or of the selection year, and
batteries also earn from balancing and frequency response, not only the index price. A gain at B2
is therefore weak evidence of a battery response. The register names the primary, bulk supply point,
and grid supply point
each row connects to. The
result for the 20 series 00006 scored:

- **Listed: B2, a bulk supply point,** with 9.5 MW of connected storage against 47.6 MW of effective
  capacity (20%).
- **Listed: P4, a primary,** with 0.37 MW of connected storage against 7.1 MW (5.1%, just over the
  threshold).
- **Not listed:** the two grid supply points (10.5 MW of connected storage against 384 MW, and 7.4
  against 171 MW, under 5%), the other bulk supply point (0.75 against 55 MW), and the other 15
  primaries (at most 0.19 MW each, against 3 to 18 MW).

**The listed group has two members, one of them with a 0.37 MW battery, so the mechanism cannot be
tested well with this series set.** C4 compares two listed series with the rest. The verdict
therefore rests on C1 and C2 at B2, the only series with a battery that is large against the
series. P4 is reported beside B2. The advice for a later experiment is to repeat the test on a
series set with more battery sites. No series is added after seeing results.

## Resources and sizing

**Sizing, one line per `plan-issue` trigger.** Stores: only private predictions and a price
parquet, no production table. Serving path: not touched. Degradation rule: not touched, but a
production port would need one (price missing means the columns go missing, which XGBoost handles).
More than one defensible design: yes (which price, which lags), so the plan fixes them here.
Code whose callers could not be named without searching: no. This is medium.

**Slot C:** 8 cores, GPU share of 8 GB, one fit at a time. The fits are the 00006 per-series fits
for four arms on two folds. Reviews: Sonnet reviews the plan, the code, and the write-up.

**Earlier advice.** Follows the experiment 00006 advice to use its Primary set and recency weights.
Departs from none of it.
