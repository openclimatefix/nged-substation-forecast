# Synthetic radiation resampling check

Invented coordinate: 52°N, 0°E. Reference day: 21 March 2026 (UTC).
Before uses point interpolation; after uses production interval-aware reconstruction.
Errors compare the average of reconstructed half-hours within each complete native window
with that window's supplied Float32 mean.
The energy centroid is the average half-hour midpoint weighted by radiation energy.
Centroid shifts include all 48 half-hours and are relative to the synthetic reference.
Positive shifts mean later. These checks do not measure forecast skill.
The cloud-transition reference uses 85% of clear sky before noon and 20% afterwards.
Short-wave conserves each native mean. Long-wave evaluates a piecewise cubic Hermite
interpolating polynomial (PCHIP) at half-hour midpoints without enforcing interval energy.
Neither method recovers unknown sub-window changes.

| Field / scenario | Native step (h) | Maximum mean error before (W/m²) | After (W/m²) | Centroid shift before (h) | After (h) |
|---|---|---|---|---|---|
| Short-wave, constant attenuation | 3 | 104.221438 | 0.000002 | 1.245899 | 0.000434 |
| Short-wave, constant attenuation | 6 | 125.526932 | 0.000002 | 2.726965 | 0.003228 |
| Short-wave, sunny morning (85%) / dull afternoon (20%) | 3 | 145.972608 | 0.000001 | 1.113304 | -0.100711 |
| Long-wave | 3 | 9.324748 | 0.978648 | 0.045120 | -0.009471 |
| Long-wave | 6 | 12.381561 | 3.638646 | -0.000000 | -0.035213 |
