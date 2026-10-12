# Synthetic radiation resampling check

Invented coordinate: 52°N, 0°E. Reference day: 21 March 2026 (UTC).
Before uses point interpolation; after uses production conservative reconstruction.
Errors compare the average of reconstructed half-hours within each complete native window
with that window's supplied Float32 mean.
The energy centroid is the average half-hour midpoint weighted by radiation energy.
Centroid shifts include all 48 half-hours and are relative to the synthetic reference.
Positive shifts mean later. These checks do not measure forecast skill.

| Field | Native step (h) | Maximum mean error before (W/m²) | After (W/m²) | Centroid shift before (h) | After (h) |
|---|---|---|---|---|---|
| Short-wave | 3 | 104.221438 | 0.000002 | 1.245899 | 0.000434 |
| Short-wave | 6 | 125.526932 | 0.000002 | 2.726965 | 0.003228 |
| Long-wave | 3 | 9.324748 | 0.000000 | 0.045120 | 0.000000 |
| Long-wave | 6 | 12.381561 | 0.000000 | -0.000000 | 0.000000 |
