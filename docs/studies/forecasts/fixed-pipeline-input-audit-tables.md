# Input-audit evidence for the wind/radiation study

These supplementary tables support the [forecast and lean-input study](fixed-pipeline-lean-features.md).
They select aggregate outputs from the input-audit script without recomputing statistics.
Counts include all 51 weather members and repeated forecasts for overlapping valid times;
they count forecast input rows, rather than distinct observed hours. Shares are fractions.
The domain thresholds are screening diagnostics. A half-hour with a nighttime midpoint can
still include sunlight, so the radiation flags do not prove physical impossibility.

The windchill flag counts inputs with temperature above 10°C or wind speed below 4.8 km/h.
Source-era labels describe archive availability and a May marker at 12 May 2026, 00:00 UTC;
the comparison around the operational cycle change at 06:00 UTC remains deferred. These groups
do not identify a causal upgrade effect. Timing correlations use the control weather member; the
deliberately delayed solar-radiation series checks the timing detector, rather than the
detectability of a forecast-loss change.

## Fold A: combined-pipeline domain screens by existing lead and source-era groups

| Pipeline | Series type | Lead band | Source era | Domain screen | Rows (all weather members) | Supported rows | Flagged rows | Flagged share (fraction of supported rows) |
|---|---|---|---|---|---|---|---|---|
| combined | PV | 72..<264h | 2024-11-13 to 2026-05-11 | longwave_above_1000_screen | 5287680 | 5287680 | 0 | 0.00000000 |
| combined | PV | 72..<264h | 2024-11-13 to 2026-05-11 | longwave_negative | 5287680 | 5287680 | 0 | 0.00000000 |
| combined | PV | 72..<264h | 2024-11-13 to 2026-05-11 | shortwave_above_1500_screen | 5287680 | 5287680 | 0 | 0.00000000 |
| combined | PV | 72..<264h | 2024-11-13 to 2026-05-11 | shortwave_negative | 5287680 | 5287680 | 0 | 0.00000000 |
| combined | PV | 72..<264h | 2024-11-13 to 2026-05-11 | shortwave_night_midpoint_screen | 5287680 | 5287680 | 1323 | 0.00025020 |
| combined | PV | 72..<264h | 2024-11-13 to 2026-05-11 | windchill_outside_domain | 5287680 | 5287680 | 2596251 | 0.49100002 |
| combined | PV | 72..<264h | 2026-05-12 onward | longwave_above_1000_screen | 772956 | 772956 | 0 | 0.00000000 |
| combined | PV | 72..<264h | 2026-05-12 onward | longwave_negative | 772956 | 772956 | 0 | 0.00000000 |
| combined | PV | 72..<264h | 2026-05-12 onward | shortwave_above_1500_screen | 772956 | 772956 | 0 | 0.00000000 |
| combined | PV | 72..<264h | 2026-05-12 onward | shortwave_negative | 772956 | 772956 | 0 | 0.00000000 |
| combined | PV | 72..<264h | 2026-05-12 onward | shortwave_night_midpoint_screen | 772956 | 772956 | 0 | 0.00000000 |
| combined | PV | 72..<264h | 2026-05-12 onward | windchill_outside_domain | 772956 | 772956 | 741678 | 0.95953457 |
| combined | PV | <72h | 2024-11-13 to 2026-05-11 | longwave_above_1000_screen | 1969110 | 1969110 | 0 | 0.00000000 |
| combined | PV | <72h | 2024-11-13 to 2026-05-11 | longwave_negative | 1969110 | 1969110 | 0 | 0.00000000 |
| combined | PV | <72h | 2024-11-13 to 2026-05-11 | shortwave_above_1500_screen | 1969110 | 1969110 | 0 | 0.00000000 |
| combined | PV | <72h | 2024-11-13 to 2026-05-11 | shortwave_negative | 1969110 | 1969110 | 0 | 0.00000000 |
| combined | PV | <72h | 2024-11-13 to 2026-05-11 | shortwave_night_midpoint_screen | 1969110 | 1969110 | 6 | 0.00000305 |
| combined | PV | <72h | 2024-11-13 to 2026-05-11 | windchill_outside_domain | 1969110 | 1969110 | 1075290 | 0.54607919 |
| combined | PV | <72h | 2026-05-12 onward | longwave_above_1000_screen | 315180 | 315180 | 0 | 0.00000000 |
| combined | PV | <72h | 2026-05-12 onward | longwave_negative | 315180 | 315180 | 0 | 0.00000000 |
| combined | PV | <72h | 2026-05-12 onward | shortwave_above_1500_screen | 315180 | 315180 | 0 | 0.00000000 |
| combined | PV | <72h | 2026-05-12 onward | shortwave_negative | 315180 | 315180 | 0 | 0.00000000 |
| combined | PV | <72h | 2026-05-12 onward | shortwave_night_midpoint_screen | 315180 | 315180 | 0 | 0.00000000 |
| combined | PV | <72h | 2026-05-12 onward | windchill_outside_domain | 315180 | 315180 | 277128 | 0.87926899 |
| combined | PV | >=264h | 2024-11-13 to 2026-05-11 | longwave_above_1000_screen | 2409750 | 2409750 | 0 | 0.00000000 |
| combined | PV | >=264h | 2024-11-13 to 2026-05-11 | longwave_negative | 2409750 | 2409750 | 0 | 0.00000000 |
| combined | PV | >=264h | 2024-11-13 to 2026-05-11 | shortwave_above_1500_screen | 2409750 | 2409750 | 0 | 0.00000000 |
| combined | PV | >=264h | 2024-11-13 to 2026-05-11 | shortwave_negative | 2409750 | 2409750 | 0 | 0.00000000 |
| combined | PV | >=264h | 2024-11-13 to 2026-05-11 | shortwave_night_midpoint_screen | 2409750 | 2409750 | 841 | 0.00034900 |
| combined | PV | >=264h | 2024-11-13 to 2026-05-11 | windchill_outside_domain | 2409750 | 2409750 | 1164644 | 0.48330491 |
| combined | PV | >=264h | 2026-05-12 onward | longwave_above_1000_screen | 320994 | 320994 | 0 | 0.00000000 |
| combined | PV | >=264h | 2026-05-12 onward | longwave_negative | 320994 | 320994 | 0 | 0.00000000 |
| combined | PV | >=264h | 2026-05-12 onward | shortwave_above_1500_screen | 320994 | 320994 | 0 | 0.00000000 |
| combined | PV | >=264h | 2026-05-12 onward | shortwave_negative | 320994 | 320994 | 0 | 0.00000000 |
| combined | PV | >=264h | 2026-05-12 onward | shortwave_night_midpoint_screen | 320994 | 320994 | 0 | 0.00000000 |
| combined | PV | >=264h | 2026-05-12 onward | windchill_outside_domain | 320994 | 320994 | 313001 | 0.97509922 |
| combined | Wind | 72..<264h | 2024-11-13 to 2026-05-11 | longwave_above_1000_screen | 2643840 | 2643840 | 0 | 0.00000000 |
| combined | Wind | 72..<264h | 2024-11-13 to 2026-05-11 | longwave_negative | 2643840 | 2643840 | 0 | 0.00000000 |
| combined | Wind | 72..<264h | 2024-11-13 to 2026-05-11 | shortwave_above_1500_screen | 2643840 | 2643840 | 0 | 0.00000000 |
| combined | Wind | 72..<264h | 2024-11-13 to 2026-05-11 | shortwave_negative | 2643840 | 2643840 | 0 | 0.00000000 |
| combined | Wind | 72..<264h | 2024-11-13 to 2026-05-11 | shortwave_night_midpoint_screen | 2643840 | 2643840 | 641 | 0.00024245 |
| combined | Wind | 72..<264h | 2024-11-13 to 2026-05-11 | windchill_outside_domain | 2643840 | 2643840 | 1286015 | 0.48641937 |
| combined | Wind | 72..<264h | 2026-05-12 onward | longwave_above_1000_screen | 386478 | 386478 | 0 | 0.00000000 |
| combined | Wind | 72..<264h | 2026-05-12 onward | longwave_negative | 386478 | 386478 | 0 | 0.00000000 |
| combined | Wind | 72..<264h | 2026-05-12 onward | shortwave_above_1500_screen | 386478 | 386478 | 0 | 0.00000000 |
| combined | Wind | 72..<264h | 2026-05-12 onward | shortwave_negative | 386478 | 386478 | 0 | 0.00000000 |
| combined | Wind | 72..<264h | 2026-05-12 onward | shortwave_night_midpoint_screen | 386478 | 386478 | 0 | 0.00000000 |
| combined | Wind | 72..<264h | 2026-05-12 onward | windchill_outside_domain | 386478 | 386478 | 370323 | 0.95819943 |
| combined | Wind | <72h | 2024-11-13 to 2026-05-11 | longwave_above_1000_screen | 984555 | 984555 | 0 | 0.00000000 |
| combined | Wind | <72h | 2024-11-13 to 2026-05-11 | longwave_negative | 984555 | 984555 | 0 | 0.00000000 |
| combined | Wind | <72h | 2024-11-13 to 2026-05-11 | shortwave_above_1500_screen | 984555 | 984555 | 0 | 0.00000000 |
| combined | Wind | <72h | 2024-11-13 to 2026-05-11 | shortwave_negative | 984555 | 984555 | 0 | 0.00000000 |
| combined | Wind | <72h | 2024-11-13 to 2026-05-11 | shortwave_night_midpoint_screen | 984555 | 984555 | 41 | 0.00004164 |
| combined | Wind | <72h | 2024-11-13 to 2026-05-11 | windchill_outside_domain | 984555 | 984555 | 531172 | 0.53950465 |
| combined | Wind | <72h | 2026-05-12 onward | longwave_above_1000_screen | 157590 | 157590 | 0 | 0.00000000 |
| combined | Wind | <72h | 2026-05-12 onward | longwave_negative | 157590 | 157590 | 0 | 0.00000000 |
| combined | Wind | <72h | 2026-05-12 onward | shortwave_above_1500_screen | 157590 | 157590 | 0 | 0.00000000 |
| combined | Wind | <72h | 2026-05-12 onward | shortwave_negative | 157590 | 157590 | 0 | 0.00000000 |
| combined | Wind | <72h | 2026-05-12 onward | shortwave_night_midpoint_screen | 157590 | 157590 | 0 | 0.00000000 |
| combined | Wind | <72h | 2026-05-12 onward | windchill_outside_domain | 157590 | 157590 | 138263 | 0.87735897 |
| combined | Wind | >=264h | 2024-11-13 to 2026-05-11 | longwave_above_1000_screen | 1204875 | 1204875 | 0 | 0.00000000 |
| combined | Wind | >=264h | 2024-11-13 to 2026-05-11 | longwave_negative | 1204875 | 1204875 | 0 | 0.00000000 |
| combined | Wind | >=264h | 2024-11-13 to 2026-05-11 | shortwave_above_1500_screen | 1204875 | 1204875 | 0 | 0.00000000 |
| combined | Wind | >=264h | 2024-11-13 to 2026-05-11 | shortwave_negative | 1204875 | 1204875 | 0 | 0.00000000 |
| combined | Wind | >=264h | 2024-11-13 to 2026-05-11 | shortwave_night_midpoint_screen | 1204875 | 1204875 | 390 | 0.00032369 |
| combined | Wind | >=264h | 2024-11-13 to 2026-05-11 | windchill_outside_domain | 1204875 | 1204875 | 577132 | 0.47899741 |
| combined | Wind | >=264h | 2026-05-12 onward | longwave_above_1000_screen | 160497 | 160497 | 0 | 0.00000000 |
| combined | Wind | >=264h | 2026-05-12 onward | longwave_negative | 160497 | 160497 | 0 | 0.00000000 |
| combined | Wind | >=264h | 2026-05-12 onward | shortwave_above_1500_screen | 160497 | 160497 | 0 | 0.00000000 |
| combined | Wind | >=264h | 2026-05-12 onward | shortwave_negative | 160497 | 160497 | 0 | 0.00000000 |
| combined | Wind | >=264h | 2026-05-12 onward | shortwave_night_midpoint_screen | 160497 | 160497 | 0 | 0.00000000 |
| combined | Wind | >=264h | 2026-05-12 onward | windchill_outside_domain | 160497 | 160497 | 156954 | 0.97792482 |

## Fold A: precipitation-type missingness and changes from old to combined pipeline

| Pipeline | Series type | Lead band | Source era | Feature | Rows (all weather members) | Old null rows | Combined null rows | Old nonfinite rows | Combined nonfinite rows | Paired finite rows | Changed finite rows | Changed null status rows | Old null share (fraction of rows) | Combined null share (fraction of rows) | Old nonfinite share (fraction of rows) | Combined nonfinite share (fraction of rows) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| combined | PV | 72..<264h | 2024-11-13 to 2026-05-11 | categorical_precipitation_type_surface | 5287680 | 0 | 0 | 0 | 0 | 5287680 | 0 | 0 | 0.00000000 | 0.00000000 | 0.00000000 | 0.00000000 |
| combined | PV | 72..<264h | 2026-05-12 onward | categorical_precipitation_type_surface | 772956 | 0 | 0 | 0 | 0 | 772956 | 0 | 0 | 0.00000000 | 0.00000000 | 0.00000000 | 0.00000000 |
| combined | PV | <72h | 2024-11-13 to 2026-05-11 | categorical_precipitation_type_surface | 1969110 | 0 | 0 | 0 | 0 | 1969110 | 0 | 0 | 0.00000000 | 0.00000000 | 0.00000000 | 0.00000000 |
| combined | PV | <72h | 2026-05-12 onward | categorical_precipitation_type_surface | 315180 | 0 | 0 | 0 | 0 | 315180 | 0 | 0 | 0.00000000 | 0.00000000 | 0.00000000 | 0.00000000 |
| combined | PV | >=264h | 2024-11-13 to 2026-05-11 | categorical_precipitation_type_surface | 2409750 | 0 | 0 | 0 | 0 | 2409750 | 0 | 0 | 0.00000000 | 0.00000000 | 0.00000000 | 0.00000000 |
| combined | PV | >=264h | 2026-05-12 onward | categorical_precipitation_type_surface | 320994 | 0 | 0 | 0 | 0 | 320994 | 0 | 0 | 0.00000000 | 0.00000000 | 0.00000000 | 0.00000000 |
| combined | Wind | 72..<264h | 2024-11-13 to 2026-05-11 | categorical_precipitation_type_surface | 2643840 | 0 | 0 | 0 | 0 | 2643840 | 0 | 0 | 0.00000000 | 0.00000000 | 0.00000000 | 0.00000000 |
| combined | Wind | 72..<264h | 2026-05-12 onward | categorical_precipitation_type_surface | 386478 | 0 | 0 | 0 | 0 | 386478 | 0 | 0 | 0.00000000 | 0.00000000 | 0.00000000 | 0.00000000 |
| combined | Wind | <72h | 2024-11-13 to 2026-05-11 | categorical_precipitation_type_surface | 984555 | 0 | 0 | 0 | 0 | 984555 | 0 | 0 | 0.00000000 | 0.00000000 | 0.00000000 | 0.00000000 |
| combined | Wind | <72h | 2026-05-12 onward | categorical_precipitation_type_surface | 157590 | 0 | 0 | 0 | 0 | 157590 | 0 | 0 | 0.00000000 | 0.00000000 | 0.00000000 | 0.00000000 |
| combined | Wind | >=264h | 2024-11-13 to 2026-05-11 | categorical_precipitation_type_surface | 1204875 | 0 | 0 | 0 | 0 | 1204875 | 0 | 0 | 0.00000000 | 0.00000000 | 0.00000000 | 0.00000000 |
| combined | Wind | >=264h | 2026-05-12 onward | categorical_precipitation_type_surface | 160497 | 0 | 0 | 0 | 0 | 160497 | 0 | 0 | 0.00000000 | 0.00000000 | 0.00000000 | 0.00000000 |

## Fold A: descriptive control-member timing correlations with a one-hour delayed radiation control

| Series type | Construction | Offset (hours) | Equal-series correlation | Supported series | Control-member rows |
|---|---|---|---|---|---|
| PV | one_hour_delayed | -3 | 0.20008989 | 6 | 171473 |
| PV | one_hour_delayed | -2 | 0.40579036 | 6 | 171473 |
| PV | one_hour_delayed | -1 | 0.60336149 | 6 | 171473 |
| PV | one_hour_delayed | 0 | 0.75349706 | 6 | 171473 |
| PV | one_hour_delayed | 1 | 0.81368905 | 6 | 171473 |
| PV | one_hour_delayed | 2 | 0.76316357 | 6 | 171473 |
| PV | one_hour_delayed | 3 | 0.62141091 | 6 | 171473 |
| PV | original | -3 | 0.40579036 | 6 | 171473 |
| PV | original | -2 | 0.60336149 | 6 | 171473 |
| PV | original | -1 | 0.75349706 | 6 | 171473 |
| PV | original | 0 | 0.81368905 | 6 | 171473 |
| PV | original | 1 | 0.76316357 | 6 | 171473 |
| PV | original | 2 | 0.62141091 | 6 | 171473 |
| PV | original | 3 | 0.42986965 | 6 | 171473 |
| Wind | original | -3 | 0.18667080 | 3 | 104607 |
| Wind | original | -2 | 0.20657143 | 3 | 104607 |
| Wind | original | -1 | 0.21912266 | 3 | 104607 |
| Wind | original | 0 | 0.22116096 | 3 | 104607 |
| Wind | original | 1 | 0.21474348 | 3 | 104607 |
| Wind | original | 2 | 0.20074886 | 3 | 104607 |
| Wind | original | 3 | 0.18045151 | 3 | 104607 |

## Fold B (retrospective): combined-pipeline domain screens by existing lead and source-era groups

| Pipeline | Series type | Lead band | Source era | Domain screen | Rows (all weather members) | Supported rows | Flagged rows | Flagged share (fraction of supported rows) |
|---|---|---|---|---|---|---|---|---|
| combined | PV | 72..<264h | 2024-11-13 to 2026-05-11 | longwave_above_1000_screen | 3710556 | 3710556 | 0 | 0.00000000 |
| combined | PV | 72..<264h | 2024-11-13 to 2026-05-11 | longwave_negative | 3710556 | 3710556 | 0 | 0.00000000 |
| combined | PV | 72..<264h | 2024-11-13 to 2026-05-11 | shortwave_above_1500_screen | 3710556 | 3710556 | 0 | 0.00000000 |
| combined | PV | 72..<264h | 2024-11-13 to 2026-05-11 | shortwave_negative | 3710556 | 3710556 | 0 | 0.00000000 |
| combined | PV | 72..<264h | 2024-11-13 to 2026-05-11 | shortwave_night_midpoint_screen | 3710556 | 3710556 | 360 | 0.00009702 |
| combined | PV | 72..<264h | 2024-11-13 to 2026-05-11 | windchill_outside_domain | 3710556 | 3710556 | 1468445 | 0.39574797 |
| combined | PV | 72..<264h | before 2024-11-13 | longwave_above_1000_screen | 3877632 | 3877632 | 0 | 0.00000000 |
| combined | PV | 72..<264h | before 2024-11-13 | longwave_negative | 3877632 | 3877632 | 0 | 0.00000000 |
| combined | PV | 72..<264h | before 2024-11-13 | shortwave_above_1500_screen | 3877632 | 3877632 | 0 | 0.00000000 |
| combined | PV | 72..<264h | before 2024-11-13 | shortwave_negative | 3877632 | 3877632 | 0 | 0.00000000 |
| combined | PV | 72..<264h | before 2024-11-13 | shortwave_night_midpoint_screen | 3877632 | 3877632 | 993 | 0.00025608 |
| combined | PV | 72..<264h | before 2024-11-13 | windchill_outside_domain | 3877632 | 3877632 | 3148101 | 0.81186172 |
| combined | PV | <72h | 2024-11-13 to 2026-05-11 | longwave_above_1000_screen | 1409130 | 1409130 | 0 | 0.00000000 |
| combined | PV | <72h | 2024-11-13 to 2026-05-11 | longwave_negative | 1409130 | 1409130 | 0 | 0.00000000 |
| combined | PV | <72h | 2024-11-13 to 2026-05-11 | shortwave_above_1500_screen | 1409130 | 1409130 | 0 | 0.00000000 |
| combined | PV | <72h | 2024-11-13 to 2026-05-11 | shortwave_negative | 1409130 | 1409130 | 0 | 0.00000000 |
| combined | PV | <72h | 2024-11-13 to 2026-05-11 | shortwave_night_midpoint_screen | 1409130 | 1409130 | 0 | 0.00000000 |
| combined | PV | <72h | 2024-11-13 to 2026-05-11 | windchill_outside_domain | 1409130 | 1409130 | 506927 | 0.35974467 |
| combined | PV | <72h | before 2024-11-13 | longwave_above_1000_screen | 1444014 | 1444014 | 0 | 0.00000000 |
| combined | PV | <72h | before 2024-11-13 | longwave_negative | 1444014 | 1444014 | 0 | 0.00000000 |
| combined | PV | <72h | before 2024-11-13 | shortwave_above_1500_screen | 1444014 | 1444014 | 0 | 0.00000000 |
| combined | PV | <72h | before 2024-11-13 | shortwave_negative | 1444014 | 1444014 | 0 | 0.00000000 |
| combined | PV | <72h | before 2024-11-13 | shortwave_night_midpoint_screen | 1444014 | 1444014 | 0 | 0.00000000 |
| combined | PV | <72h | before 2024-11-13 | windchill_outside_domain | 1444014 | 1444014 | 1172140 | 0.81172343 |
| combined | PV | >=264h | 2024-11-13 to 2026-05-11 | longwave_above_1000_screen | 1659744 | 1659744 | 0 | 0.00000000 |
| combined | PV | >=264h | 2024-11-13 to 2026-05-11 | longwave_negative | 1659744 | 1659744 | 0 | 0.00000000 |
| combined | PV | >=264h | 2024-11-13 to 2026-05-11 | shortwave_above_1500_screen | 1659744 | 1659744 | 0 | 0.00000000 |
| combined | PV | >=264h | 2024-11-13 to 2026-05-11 | shortwave_negative | 1659744 | 1659744 | 0 | 0.00000000 |
| combined | PV | >=264h | 2024-11-13 to 2026-05-11 | shortwave_night_midpoint_screen | 1659744 | 1659744 | 228 | 0.00013737 |
| combined | PV | >=264h | 2024-11-13 to 2026-05-11 | windchill_outside_domain | 1659744 | 1659744 | 663577 | 0.39980684 |
| combined | PV | >=264h | before 2024-11-13 | longwave_above_1000_screen | 1767150 | 1767150 | 0 | 0.00000000 |
| combined | PV | >=264h | before 2024-11-13 | longwave_negative | 1767150 | 1767150 | 0 | 0.00000000 |
| combined | PV | >=264h | before 2024-11-13 | shortwave_above_1500_screen | 1767150 | 1767150 | 0 | 0.00000000 |
| combined | PV | >=264h | before 2024-11-13 | shortwave_negative | 1767150 | 1767150 | 0 | 0.00000000 |
| combined | PV | >=264h | before 2024-11-13 | shortwave_night_midpoint_screen | 1767150 | 1767150 | 672 | 0.00038027 |
| combined | PV | >=264h | before 2024-11-13 | windchill_outside_domain | 1767150 | 1767150 | 1390221 | 0.78670232 |
| combined | Wind | 72..<264h | 2024-11-13 to 2026-05-11 | longwave_above_1000_screen | 1855278 | 1855278 | 0 | 0.00000000 |
| combined | Wind | 72..<264h | 2024-11-13 to 2026-05-11 | longwave_negative | 1855278 | 1855278 | 0 | 0.00000000 |
| combined | Wind | 72..<264h | 2024-11-13 to 2026-05-11 | shortwave_above_1500_screen | 1855278 | 1855278 | 0 | 0.00000000 |
| combined | Wind | 72..<264h | 2024-11-13 to 2026-05-11 | shortwave_negative | 1855278 | 1855278 | 0 | 0.00000000 |
| combined | Wind | 72..<264h | 2024-11-13 to 2026-05-11 | shortwave_night_midpoint_screen | 1855278 | 1855278 | 109 | 0.00005875 |
| combined | Wind | 72..<264h | 2024-11-13 to 2026-05-11 | windchill_outside_domain | 1855278 | 1855278 | 714672 | 0.38521019 |
| combined | Wind | 72..<264h | before 2024-11-13 | longwave_above_1000_screen | 1938816 | 1938816 | 0 | 0.00000000 |
| combined | Wind | 72..<264h | before 2024-11-13 | longwave_negative | 1938816 | 1938816 | 0 | 0.00000000 |
| combined | Wind | 72..<264h | before 2024-11-13 | shortwave_above_1500_screen | 1938816 | 1938816 | 0 | 0.00000000 |
| combined | Wind | 72..<264h | before 2024-11-13 | shortwave_negative | 1938816 | 1938816 | 0 | 0.00000000 |
| combined | Wind | 72..<264h | before 2024-11-13 | shortwave_night_midpoint_screen | 1938816 | 1938816 | 548 | 0.00028265 |
| combined | Wind | 72..<264h | before 2024-11-13 | windchill_outside_domain | 1938816 | 1938816 | 1585589 | 0.81781304 |
| combined | Wind | <72h | 2024-11-13 to 2026-05-11 | longwave_above_1000_screen | 704565 | 704565 | 0 | 0.00000000 |
| combined | Wind | <72h | 2024-11-13 to 2026-05-11 | longwave_negative | 704565 | 704565 | 0 | 0.00000000 |
| combined | Wind | <72h | 2024-11-13 to 2026-05-11 | shortwave_above_1500_screen | 704565 | 704565 | 0 | 0.00000000 |
| combined | Wind | <72h | 2024-11-13 to 2026-05-11 | shortwave_negative | 704565 | 704565 | 0 | 0.00000000 |
| combined | Wind | <72h | 2024-11-13 to 2026-05-11 | shortwave_night_midpoint_screen | 704565 | 704565 | 0 | 0.00000000 |
| combined | Wind | <72h | 2024-11-13 to 2026-05-11 | windchill_outside_domain | 704565 | 704565 | 245083 | 0.34785009 |
| combined | Wind | <72h | before 2024-11-13 | longwave_above_1000_screen | 722007 | 722007 | 0 | 0.00000000 |
| combined | Wind | <72h | before 2024-11-13 | longwave_negative | 722007 | 722007 | 0 | 0.00000000 |
| combined | Wind | <72h | before 2024-11-13 | shortwave_above_1500_screen | 722007 | 722007 | 0 | 0.00000000 |
| combined | Wind | <72h | before 2024-11-13 | shortwave_negative | 722007 | 722007 | 0 | 0.00000000 |
| combined | Wind | <72h | before 2024-11-13 | shortwave_night_midpoint_screen | 722007 | 722007 | 28 | 0.00003878 |
| combined | Wind | <72h | before 2024-11-13 | windchill_outside_domain | 722007 | 722007 | 591491 | 0.81923167 |
| combined | Wind | >=264h | 2024-11-13 to 2026-05-11 | longwave_above_1000_screen | 829872 | 829872 | 0 | 0.00000000 |
| combined | Wind | >=264h | 2024-11-13 to 2026-05-11 | longwave_negative | 829872 | 829872 | 0 | 0.00000000 |
| combined | Wind | >=264h | 2024-11-13 to 2026-05-11 | shortwave_above_1500_screen | 829872 | 829872 | 0 | 0.00000000 |
| combined | Wind | >=264h | 2024-11-13 to 2026-05-11 | shortwave_negative | 829872 | 829872 | 0 | 0.00000000 |
| combined | Wind | >=264h | 2024-11-13 to 2026-05-11 | shortwave_night_midpoint_screen | 829872 | 829872 | 105 | 0.00012653 |
| combined | Wind | >=264h | 2024-11-13 to 2026-05-11 | windchill_outside_domain | 829872 | 829872 | 322385 | 0.38847557 |
| combined | Wind | >=264h | before 2024-11-13 | longwave_above_1000_screen | 883575 | 883575 | 0 | 0.00000000 |
| combined | Wind | >=264h | before 2024-11-13 | longwave_negative | 883575 | 883575 | 0 | 0.00000000 |
| combined | Wind | >=264h | before 2024-11-13 | shortwave_above_1500_screen | 883575 | 883575 | 0 | 0.00000000 |
| combined | Wind | >=264h | before 2024-11-13 | shortwave_negative | 883575 | 883575 | 0 | 0.00000000 |
| combined | Wind | >=264h | before 2024-11-13 | shortwave_night_midpoint_screen | 883575 | 883575 | 269 | 0.00030445 |
| combined | Wind | >=264h | before 2024-11-13 | windchill_outside_domain | 883575 | 883575 | 698910 | 0.79100246 |

## Fold B (retrospective): precipitation-type missingness and changes from old to combined pipeline

| Pipeline | Series type | Lead band | Source era | Feature | Rows (all weather members) | Old null rows | Combined null rows | Old nonfinite rows | Combined nonfinite rows | Paired finite rows | Changed finite rows | Changed null status rows | Old null share (fraction of rows) | Combined null share (fraction of rows) | Old nonfinite share (fraction of rows) | Combined nonfinite share (fraction of rows) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| combined | PV | 72..<264h | 2024-11-13 to 2026-05-11 | categorical_precipitation_type_surface | 3710556 | 0 | 0 | 0 | 0 | 3710556 | 0 | 0 | 0.00000000 | 0.00000000 | 0.00000000 | 0.00000000 |
| combined | PV | 72..<264h | before 2024-11-13 | categorical_precipitation_type_surface | 3877632 | 3877632 | 3877632 | 0 | 0 | 0 | 0 | 0 | 1.00000000 | 1.00000000 | 0.00000000 | 0.00000000 |
| combined | PV | <72h | 2024-11-13 to 2026-05-11 | categorical_precipitation_type_surface | 1409130 | 0 | 0 | 0 | 0 | 1409130 | 0 | 0 | 0.00000000 | 0.00000000 | 0.00000000 | 0.00000000 |
| combined | PV | <72h | before 2024-11-13 | categorical_precipitation_type_surface | 1444014 | 1444014 | 1444014 | 0 | 0 | 0 | 0 | 0 | 1.00000000 | 1.00000000 | 0.00000000 | 0.00000000 |
| combined | PV | >=264h | 2024-11-13 to 2026-05-11 | categorical_precipitation_type_surface | 1659744 | 0 | 0 | 0 | 0 | 1659744 | 0 | 0 | 0.00000000 | 0.00000000 | 0.00000000 | 0.00000000 |
| combined | PV | >=264h | before 2024-11-13 | categorical_precipitation_type_surface | 1767150 | 1767150 | 1767150 | 0 | 0 | 0 | 0 | 0 | 1.00000000 | 1.00000000 | 0.00000000 | 0.00000000 |
| combined | Wind | 72..<264h | 2024-11-13 to 2026-05-11 | categorical_precipitation_type_surface | 1855278 | 0 | 0 | 0 | 0 | 1855278 | 0 | 0 | 0.00000000 | 0.00000000 | 0.00000000 | 0.00000000 |
| combined | Wind | 72..<264h | before 2024-11-13 | categorical_precipitation_type_surface | 1938816 | 1938816 | 1938816 | 0 | 0 | 0 | 0 | 0 | 1.00000000 | 1.00000000 | 0.00000000 | 0.00000000 |
| combined | Wind | <72h | 2024-11-13 to 2026-05-11 | categorical_precipitation_type_surface | 704565 | 0 | 0 | 0 | 0 | 704565 | 0 | 0 | 0.00000000 | 0.00000000 | 0.00000000 | 0.00000000 |
| combined | Wind | <72h | before 2024-11-13 | categorical_precipitation_type_surface | 722007 | 722007 | 722007 | 0 | 0 | 0 | 0 | 0 | 1.00000000 | 1.00000000 | 0.00000000 | 0.00000000 |
| combined | Wind | >=264h | 2024-11-13 to 2026-05-11 | categorical_precipitation_type_surface | 829872 | 0 | 0 | 0 | 0 | 829872 | 0 | 0 | 0.00000000 | 0.00000000 | 0.00000000 | 0.00000000 |
| combined | Wind | >=264h | before 2024-11-13 | categorical_precipitation_type_surface | 883575 | 883575 | 883575 | 0 | 0 | 0 | 0 | 0 | 1.00000000 | 1.00000000 | 0.00000000 | 0.00000000 |

## Fold B (retrospective): descriptive control-member timing correlations with a one-hour delayed radiation control

| Series type | Construction | Offset (hours) | Equal-series correlation | Supported series | Control-member rows |
|---|---|---|---|---|---|
| PV | one_hour_delayed | -3 | 0.32037586 | 6 | 246699 |
| PV | one_hour_delayed | -2 | 0.51603419 | 6 | 246699 |
| PV | one_hour_delayed | -1 | 0.68217742 | 6 | 246699 |
| PV | one_hour_delayed | 0 | 0.79466796 | 6 | 246699 |
| PV | one_hour_delayed | 1 | 0.83124912 | 6 | 246699 |
| PV | one_hour_delayed | 2 | 0.78196913 | 6 | 246699 |
| PV | one_hour_delayed | 3 | 0.65761358 | 6 | 246699 |
| PV | original | -3 | 0.51603419 | 6 | 246699 |
| PV | original | -2 | 0.68217742 | 6 | 246699 |
| PV | original | -1 | 0.79466796 | 6 | 246699 |
| PV | original | 0 | 0.83124912 | 6 | 246699 |
| PV | original | 1 | 0.78196913 | 6 | 246699 |
| PV | original | 2 | 0.65761358 | 6 | 246699 |
| PV | original | 3 | 0.48158485 | 6 | 246699 |
| Wind | original | -3 | 0.15944713 | 3 | 132753 |
| Wind | original | -2 | 0.18699963 | 3 | 132753 |
| Wind | original | -1 | 0.20735377 | 3 | 132753 |
| Wind | original | 0 | 0.21779183 | 3 | 132753 |
| Wind | original | 1 | 0.21684186 | 3 | 132753 |
| Wind | original | 2 | 0.20445725 | 3 | 132753 |
| Wind | original | 3 | 0.18439443 | 3 | 132753 |
