---
review_iteration: 1
reviewer: "scientist"
total_flaws: 4
critical_flaws: 3
---

# ML Rigor Review

## FLAW-001: [Critical] Data Leakage in Dynamic Seasonal Lag for Lead Times > 14 Days
* **File & Line Number:** `packages/xgboost_forecaster/src/xgboost_forecaster/model.py`, lines ~100-115
* **The Theoretical Issue:** The current logic switches to a 14-day lag when the lead time exceeds 7 days. However, ECMWF forecasts extend up to 15 days (360 hours). For any lead time > 14 days (e.g., 14.5 days), the 14-day lag will reference a timestamp that is 0.5 days in the *future* relative to the `init_time`.
* **Concrete Failure Mode:** The model will perfectly predict the target for days 14-15 during backtesting because it is accidentally being fed the actual future power flows. In production, this data will not exist, causing catastrophic failure for week-2 forecasts.
* **Required Architectural Fix:** The Architect must generalize the lag logic. Instead of a hardcoded `if <= 7 else 14` switch, calculate the required lag dynamically: `weeks_ahead = ceil(lead_time / 7_days)` and use `lag = weeks_ahead * 7_days`.

## FLAW-002: [Critical] Lookahead Bias in Multi-NWP `join_asof` Strategy
* **File & Line Number:** `docs/temp/implementation_plan.md`, Section 3
* **The Theoretical Issue:** The proposed plan suggests using `join_asof` on `init_time` to align secondary NWPs (e.g., GFS) with the primary NWP (e.g., ECMWF). This ignores NWP publication delays. A GFS forecast initialized at 12:00z is not actually available until ~15:30z. If ECMWF initializes at 12:00z, joining GFS 12:00z is a lookahead bias.
* **Concrete Failure Mode:** The model will learn to rely on secondary NWP data that won't actually be available in the production environment at the time the forecast is generated.
* **Required Architectural Fix:** The join must account for publication delay. Define an `available_time = init_time + publication_delay` for all NWPs. The `join_asof` must ensure that `secondary_NWP.available_time <= primary_NWP.init_time`.

## FLAW-003: [Critical] Artificial Spikes in NWP Deaccumulation due to Variable Timesteps
* **File & Line Number:** `docs/temp/implementation_plan.md`, Section 2
* **The Theoretical Issue:** The plan proposes using `pl.col(var).diff()` to convert accumulated variables to rates. However, NWP models like ECMWF change their temporal resolution (e.g., 1-hour steps up to 90h, then 3-hour steps). A `.diff()` over a 3-hour step will produce a value 3x larger than a 1-hour step, creating massive artificial spikes in the features.
* **Concrete Failure Mode:** The model will learn spurious relationships where it thinks precipitation/radiation suddenly triples at the 90-hour horizon.
* **Required Architectural Fix:** The differenced values must be divided by the time delta (`valid_time.diff().dt.total_hours()`) to create a normalized rate (e.g., mm/hour or W/m^2). Additionally, the rates must be clipped at 0 (`.clip(lower_bound=0)`) to prevent negative physical values caused by minor numerical artifacts in the NWP output.

## FLAW-004: [Standard] Backtesting Bias Fix Verification
* **File & Line Number:** `docs/temp/implementation_plan.md`, Section 1
* **The Theoretical Issue:** The proposed fix to evaluate by `lead_time` is mathematically sound and correctly addresses the backtesting bias.
* **Concrete Failure Mode:** N/A - The proposed fix is correct.
* **Required Architectural Fix:** Proceed with the implementation as planned, ensuring that `init_time` accurately reflects the forecast generation time.
