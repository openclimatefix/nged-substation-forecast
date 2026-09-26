# ENS backfill pilot: check report

Checkpoint files read: 23 dates, 2021-04-22 to 2024-03-31.

## (a) Hand decode against ecCodes, and (b) message lengths
ecCodes version 2.49.0.
Messages compared: 317 across 23 dates, each fetched whole.
hand_equals_eccodes: 317 of 317 true, 0 mismatches
Largest absolute difference, hand decode against ecCodes: 0.0
stored_values_equal: 317 of 317 true, 0 mismatches
stored_header_equal: 317 of 317 true, 0 mismatches
length_equal: 317 of 317 true, 0 mismatches
eccodes_names_agree: 317 of 317 true, 0 mismatches
grid_agrees: 317 of 317 true, 0 mismatches
10u messages with a negative reference value across the pilot: 1955
Of the compared messages, 10u with a negative reference value: 62

### (b) The idx chain
2021-04-22: idx gaps or overlaps 0; stored offset or length differs from idx 0
2021-06-02: idx gaps or overlaps 0; stored offset or length differs from idx 0
2021-06-23: idx gaps or overlaps 0; stored offset or length differs from idx 0
2021-07-06: idx gaps or overlaps 0; stored offset or length differs from idx 0
2021-08-17: idx gaps or overlaps 0; stored offset or length differs from idx 0
2021-11-11: idx gaps or overlaps 0; stored offset or length differs from idx 0
2021-12-09: idx gaps or overlaps 0; stored offset or length differs from idx 0
2022-02-18: idx gaps or overlaps 0; stored offset or length differs from idx 0
2022-04-03: idx gaps or overlaps 0; stored offset or length differs from idx 0
2022-04-20: idx gaps or overlaps 0; stored offset or length differs from idx 0
2022-05-06: idx gaps or overlaps 0; stored offset or length differs from idx 0
2022-06-26: idx gaps or overlaps 0; stored offset or length differs from idx 0
2022-07-03: idx gaps or overlaps 0; stored offset or length differs from idx 0
2022-08-08: idx gaps or overlaps 0; stored offset or length differs from idx 0
2022-12-15: idx gaps or overlaps 0; stored offset or length differs from idx 0
2023-04-20: idx gaps or overlaps 0; stored offset or length differs from idx 0
2023-06-26: idx gaps or overlaps 0; stored offset or length differs from idx 0
2023-06-28: idx gaps or overlaps 0; stored offset or length differs from idx 0
2023-07-06: idx gaps or overlaps 0; stored offset or length differs from idx 0
2023-10-01: idx gaps or overlaps 0; stored offset or length differs from idx 0
2023-12-31: idx gaps or overlaps 0; stored offset or length differs from idx 0
2024-01-22: idx gaps or overlaps 0; stored offset or length differs from idx 0
2024-03-31: idx gaps or overlaps 0; stored offset or length differs from idx 0

## (c) Integrity: a second fetch of random ranges
Ranges re-fetched: 20; SHA-256 differs from the first fetch: 0

## (d) Physical sanity
| variable | min | max | mean | non-finite |
|---|---|---|---|---|
| 2t | 258.816 | 307.9 | 284.228 | 0 |
| 2d | 249.73 | 295.746 | 280.927 | 0 |
| 10u | -23.7078 | 41.6591 | 1.8403 | 0 |
| 10v | -30.0813 | 32.3139 | 0.706182 | 0 |
| 100u | -30.7184 | 48.7934 | 2.44824 | 0 |
| 100v | -36.2259 | 39.2057 | 1.08144 | 0 |
| sp | 87840.8 | 104476 | 100865 | 0 |
| msl | 93089.1 | 104271 | 101293 | 0 |
| tp | 0 | 0.466309 | 0.0179452 | 0 |
| strd | 0 | 4.71365e+08 | 1.71468e+08 | 0 |
| ssrd | 0 | 4.25329e+08 | 8.4181e+07 | 0 |
| z500 | 47594 | 58463.8 | 54568.8 | 0 |
2t: mean 284.23 means the values are in KELVIN.
2d: mean 280.93 means the values are in KELVIN.
z500 mean 54569 m2 s-2 is 5564 m of geopotential height.

## (e) De-accumulation
Dynamical.org expects a clamped fraction of 0.08 and an invalid fraction of 0.01.
| date | variable | lead-0 max abs | clamped | invalid | drops in the accumulation |
|---|---|---|---|---|---|
| 2021-04-22 | tp | 0 | 0.0033 | 0.0000 | 0.0033 |
| 2021-04-22 | strd | 0 | 0.0000 | 0.0000 | 0.0000 |
| 2021-04-22 | ssrd | 0 | 0.0099 | 0.0000 | 0.0099 |
| 2021-06-02 | tp | 0 | 0.0061 | 0.0000 | 0.0061 |
| 2021-06-02 | strd | 0 | 0.0000 | 0.0000 | 0.0000 |
| 2021-06-02 | ssrd | 0 | 0.0000 | 0.0000 | 0.0000 |
| 2021-06-23 | tp | 0 | 0.0075 | 0.0000 | 0.0075 |
| 2021-06-23 | strd | 0 | 0.0000 | 0.0000 | 0.0000 |
| 2021-06-23 | ssrd | 0 | 0.0000 | 0.0000 | 0.0000 |
| 2021-07-06 | tp | 0 | 0.0047 | 0.0000 | 0.0047 |
| 2021-07-06 | strd | 0 | 0.0000 | 0.0000 | 0.0000 |
| 2021-07-06 | ssrd | 0 | 0.0000 | 0.0000 | 0.0000 |
| 2021-08-17 | tp | 0 | 0.0056 | 0.0000 | 0.0056 |
| 2021-08-17 | strd | 0 | 0.0000 | 0.0000 | 0.0000 |
| 2021-08-17 | ssrd | 0 | 0.0065 | 0.0000 | 0.0065 |
| 2021-11-11 | tp | 0 | 0.0036 | 0.0000 | 0.0036 |
| 2021-11-11 | strd | 0 | 0.0000 | 0.0000 | 0.0000 |
| 2021-11-11 | ssrd | 0 | 0.0067 | 0.0000 | 0.0067 |
| 2021-12-09 | tp | 0 | 0.0042 | 0.0000 | 0.0042 |
| 2021-12-09 | strd | 0 | 0.0000 | 0.0000 | 0.0000 |
| 2021-12-09 | ssrd | 0 | 0.0044 | 0.0000 | 0.0044 |
| 2022-02-18 | tp | 0 | 0.0008 | 0.0000 | 0.0008 |
| 2022-02-18 | strd | 0 | 0.0000 | 0.0000 | 0.0000 |
| 2022-02-18 | ssrd | 0 | 0.0147 | 0.0000 | 0.0147 |
| 2022-04-03 | tp | 0 | 0.0026 | 0.0000 | 0.0026 |
| 2022-04-03 | strd | 0 | 0.0000 | 0.0000 | 0.0000 |
| 2022-04-03 | ssrd | 0 | 0.0100 | 0.0000 | 0.0100 |
| 2022-04-20 | tp | 0 | 0.0046 | 0.0000 | 0.0046 |
| 2022-04-20 | strd | 0 | 0.0000 | 0.0000 | 0.0000 |
| 2022-04-20 | ssrd | 0 | 0.0030 | 0.0000 | 0.0030 |
| 2022-05-06 | tp | 0 | 0.0034 | 0.0000 | 0.0034 |
| 2022-05-06 | strd | 0 | 0.0000 | 0.0000 | 0.0000 |
| 2022-05-06 | ssrd | 0 | 0.0000 | 0.0000 | 0.0000 |
| 2022-06-26 | tp | 0 | 0.0036 | 0.0000 | 0.0036 |
| 2022-06-26 | strd | 0 | 0.0000 | 0.0000 | 0.0000 |
| 2022-06-26 | ssrd | 0 | 0.0000 | 0.0000 | 0.0000 |
| 2022-07-03 | tp | 0 | 0.0060 | 0.0000 | 0.0060 |
| 2022-07-03 | strd | 0 | 0.0000 | 0.0000 | 0.0000 |
| 2022-07-03 | ssrd | 0 | 0.0000 | 0.0000 | 0.0000 |
| 2022-08-08 | tp | 0 | 0.0023 | 0.0000 | 0.0023 |
| 2022-08-08 | strd | 0 | 0.0000 | 0.0000 | 0.0000 |
| 2022-08-08 | ssrd | 0 | 0.0056 | 0.0000 | 0.0056 |
| 2022-12-15 | tp | 0 | 0.0020 | 0.0000 | 0.0020 |
| 2022-12-15 | strd | 0 | 0.0000 | 0.0000 | 0.0000 |
| 2022-12-15 | ssrd | 0 | 0.0065 | 0.0000 | 0.0065 |
| 2023-04-20 | tp | 0 | 0.0013 | 0.0000 | 0.0013 |
| 2023-04-20 | strd | 0 | 0.0000 | 0.0000 | 0.0000 |
| 2023-04-20 | ssrd | 0 | 0.0093 | 0.0000 | 0.0093 |
| 2023-06-26 | tp | 0 | 0.0053 | 0.0000 | 0.0053 |
| 2023-06-26 | strd | 0 | 0.0000 | 0.0000 | 0.0000 |
| 2023-06-26 | ssrd | 0 | 0.0000 | 0.0000 | 0.0000 |
| 2023-06-28 | tp | 0 | 0.0030 | 0.0000 | 0.0030 |
| 2023-06-28 | strd | 0 | 0.0000 | 0.0000 | 0.0000 |
| 2023-06-28 | ssrd | 0 | 0.0000 | 0.0000 | 0.0000 |
| 2023-07-06 | tp | 0 | 0.0047 | 0.0000 | 0.0047 |
| 2023-07-06 | strd | 0 | 0.0000 | 0.0000 | 0.0000 |
| 2023-07-06 | ssrd | 0 | 0.0000 | 0.0000 | 0.0000 |
| 2023-10-01 | tp | 0 | 0.0048 | 0.0000 | 0.0048 |
| 2023-10-01 | strd | 0 | 0.0000 | 0.0000 | 0.0000 |
| 2023-10-01 | ssrd | 0 | 0.0063 | 0.0000 | 0.0063 |
| 2023-12-31 | tp | 0 | 0.0008 | 0.0000 | 0.0008 |
| 2023-12-31 | strd | 0 | 0.0000 | 0.0000 | 0.0000 |
| 2023-12-31 | ssrd | 0 | 0.0031 | 0.0000 | 0.0031 |
| 2024-01-22 | tp | 0 | 0.0012 | 0.0000 | 0.0012 |
| 2024-01-22 | strd | 0 | 0.0000 | 0.0000 | 0.0000 |
| 2024-01-22 | ssrd | 0 | 0.0144 | 0.0000 | 0.0144 |
| 2024-03-31 | tp | 0 | 0.0054 | 0.0000 | 0.0054 |
| 2024-03-31 | strd | 0 | 0.0000 | 0.0000 | 0.0000 |
| 2024-03-31 | ssrd | 0 | 0.0086 | 0.0000 | 0.0086 |
tp over all dates: clamped fraction 0.0038, invalid fraction 0.0000
strd over all dates: clamped fraction 0.0000, invalid fraction 0.0000
ssrd over all dates: clamped fraction 0.0047, invalid fraction 0.0000

Step change at 144 to 150 h: mean strd rate (W m-2) from the actual elapsed seconds against
a fixed 3 h divisor. The actual-seconds rate should be continuous across the change.
| step (h) | actual seconds | fixed 3 h |
|---|---|---|
| 138 | 323.80 | 323.80 |
| 141 | 320.44 | 320.44 |
| 144 | 317.89 | 317.89 |
| 150 | 316.96 | 633.92 |
| 156 | 317.89 | 635.77 |
| 162 | 317.68 | 635.36 |

## (f) Grid registration
The pipeline uses 48 latitudes from 49.5 to 61.25 and 45 longitudes from -9.0 to 2.0 degrees east on the -180 to 180 axis.
The fetched rows run from 61.25 to 49.0 degrees north.
Pipeline latitudes missing from the fetched rows: []
Pipeline longitudes missing from the fetched columns: []
Section (a) checks the ecCodes latitude and longitude arrays of one message.

## (g) Data-validation checklist
Dates whose step axis equals the 85 expected steps exactly: 23 of 23
Duplicate range hashes within a date (would mean two keys read one message): 0
NaN values in all stored arrays: 0
Mean ssrd rate (W m-2) by valid hour of day, leads 3 to 24 h (period ending):
| valid hour (UTC) | mean | maximum |
|---|---|---|
| 03 | 0.01 | 3.08 |
| 06 | 19.87 | 157.89 |
| 09 | 181.55 | 528.55 |
| 12 | 395.62 | 840.20 |
| 15 | 417.41 | 881.21 |
| 18 | 221.43 | 591.55 |
| 21 | 35.32 | 220.82 |
| 00 | 0.12 | 11.66 |
Largest ssrd rate at valid hours 00 and 03 UTC: 11.662 W m-2

## (h) Totals
Checkpoint files: 23, 316.5 MB on disk.
Messages stored: 23460; range bytes fetched for them: 11.15 GB.
The check took 106 s.
