# Market and other data

**Experiments hold public price, demand, solar, and carbon data, and none of it has yet improved a
forecast of NGED's series by a margin that survived its control.** This page lists each source, what
it is for, when its values become known, and where it sits on disk. The price series are described in
full on the [GB price data page](../../background/gb-price-data-and-ancillary-markets.md) and in the
[data sources page](../../roadmap/data-sources.md#electricity-price-data).

## What each source is for

| Source | Folder under `data/studies/downloads/market/` | What it is | Known when |
|---|---|---|---|
| Imbalance system price | `elexon_system_prices` | Price paid for energy short or long in a half-hour | About 19 minutes after the period ends |
| Market index price | `elexon_mid_apx` | Short-term exchange price, half-hourly | During the half-hour |
| Day-ahead price | `neso_n2ex_day_ahead` | Hourly auction price | The morning before delivery |
| Demand outturn, generation by fuel | `elexon_demand_outturn`, `elexon_fuelhh` | National demand and generation | Demand: about 30 minutes after the half-hour; generation: at `publish_time` |
| Demand and wind forecasts | `elexon_ndf`, `elexon_tsdf`, `elexon_windfor` | National forecasts, every publication kept | At `publish_time` |
| Frequency | `elexon_frequency` | 15-second values and a half-hourly summary | Not checked |
| Bid, offer, and notification data | `elexon_bod`, `elexon_boalf`, `elexon_pn`, `elexon_mels` | Per Balancing Mechanism Unit | Within minutes, for gate-closed periods |
| Carbon intensity | `carbon_intensity` | National forecast and actual, grams per kWh | Forecast up to 2 days ahead |
| PV_Live | `pv_live`, `pv_live_gsp` | Sheffield Solar's estimate of solar generation | Not checked |
| Octopus Agile | `octopus_agile_east_midlands` | Half-hourly import tariff, East Midlands | Not checked |

The "known when" column digests the [publication table](../../roadmap/data-sources.md#when-each-series-is-published)
and the folders' READMEs (docs, disk).

## Timestamps and coverage

- **The market tables mark the start of each period, in UTC, whereas the NGED power feed marks the
  end of each half-hour.** Shift one side by 30 minutes before joining
  ([data sources](../../roadmap/data-sources.md#what-the-data-looks-like); docs).
- **PV_Live labels each half-hour by its end.** The stored `time` column is PV_Live's `datetime_gmt`
  minus 30 minutes (disk).
- **Most market folders cover one year, 2025-09-01 to 2026-09-30.** The two exceptions are
  `pv_live_gsp`, which runs from 2019-01-01 to 2026-10-14 for the Walpole and Bicker Fen grid supply
  points, and `octopus_agile_east_midlands`, which runs from 2025-08-31 to 2026-09-01 (disk).
- **Experiments 177 and 185 keep the system prices for longer spans in their own `public/`
  folders**, under `data/ml-experiments/00185-*/public/system_prices` (disk).
- **The demand flexibility service (DFS) has no page of its own.** The NESO service-requirement
  archive ends on 2026-03-27 and is stored as `dfs_service_requirement.csv` under
  `data/ml-experiments/00177-*/public/` (exp 185; 1,231 days with a record; disk).
- **Each licence's attribution wording applies wherever the data is shown** (docs).
- **Elexon's indicated demand and generation reach at most 42 hours ahead, and no table maps their
  regions to NGED's licence areas**, so neither is a ready input
  ([study](../../studies/indgen-inddem-and-gsp-take.md); docs).

## Price features tried

- **A price at the target half-hour lowered the pooled Primary score by 0.52 points of capacity in the
  selection year and 0.18 in the confirmation year (interval included zero), and a price from
  another week matched about 0.45 of the 0.52** (exp 15; Primaries, two years; provisional).
- **Lagged prices did no better** (exp 10; Primaries and one battery site, two years; provisional).
- **Price changes added to the price level lowered the score by 0.037 and 0.050 points of capacity,
  against 0.039 and 0.047 for changes from another week.** The experiment's verdict was `small
  gain`, and the demand-inputs theme does not adopt it (exp 25; Primaries, two years; provisional).
  See the [demand inputs theme](../themes/demand-inputs/index.md).
- **National system price, DFS events, national demand, wind, solar, and interconnector flow each
  explain under 1% of the day-to-day variation in the best-guess forecast's loss.** Giving the
  Kalman filter the signal in hindsight made the score slightly worse (exp 185; the 37 worst days, one
  year; provisional).
- **Modelling a price-following asset in the history did not improve the forecast at D8 or D14.** The
  ceiling with the true price gained 0.6% at D8 (exp 191; 2 Primaries; provisional).
- **PV_Live lags at the two grid supply points gave no resolved gain** (-0.03 points of capacity;
  exp 27; 2 GSPs; provisional). A PV_Live-based solar forecast gave -0.14 with an interval that
  included zero, and the true PV_Live value gave -0.50, almost all at leads of 7 to 15 days (exp 28;
  2 GSPs; provisional).
- **Add a placebo twin to every price input**, a copy with the right level and the wrong timing
  (exp 10, 15; Primaries; provisional).

Last checked against experiment 183.
