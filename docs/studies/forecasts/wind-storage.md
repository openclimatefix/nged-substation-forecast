# Wind component storage measurement

**Storing u/v uses 8.90 % more Parquet bytes than speed/bearing in this 24-run sample.**
Here, u is eastward velocity and v is northward velocity; polar storage holds speed and
meteorological bearing. Adding scalar-mean speeds increases the u/v total by another 17.90 %.
The maintainer chose u/v alone. The measurement
compares storage and spatial cancellation; no forecasting model was trained or scored.

**The contrasts (planned) compare u/v against polar storage, then scalar means against u/v.**
Planned means specified before measurement, rather than proposed future work.
The sample contains the 15th and last day of every month in 2025, all at 00Z, with no unavailable
dates or substitutions. Each forecast contains 85 native steps, 51 members, and 1,671 H3 cells:
7,243,785 rows per forecast and 173,850,840 rows per representation. Seasons group forecasts by
initialization month; forecasts may cross season boundaries. The comparison uses the same
downloaded bytes and H3 weights for all
representations and the actual production `write_nwp`, including 13-bit significand rounding,
sort order, and writer properties. Non-wind compressed-column bytes match exactly in every pair.

**Scalar-gap statistics precede writer rounding and measure spatial cancellation.**
Native rows receive equal weight, rather than weighting 6-hour intervals twice as much as
3-hour intervals. These descriptive sample statistics have no sampling confidence interval and
should not be read as turbine-power losses or forecast-skill changes. The scalar means use
grid-point magnitudes before H3 aggregation; vector magnitudes use aggregated u/v.

Parquet totals include file metadata; Delta-log bytes are separate. MB means 10⁶ bytes.

| Run | Polar MB | Components MB | Components + scalar MB | Components vs polar | Scalar vs components |
|---|---:|---:|---:|---:|---:|
| 2025-01-15 | 156.691 | 171.854 | 203.308 | +9.68% | +18.30% |
| 2025-01-31 | 160.675 | 175.670 | 206.962 | +9.33% | +17.81% |
| 2025-02-15 | 159.015 | 173.487 | 203.496 | +9.10% | +17.30% |
| 2025-02-28 | 159.407 | 172.614 | 205.128 | +8.28% | +18.84% |
| 2025-03-15 | 164.301 | 177.288 | 208.593 | +7.90% | +17.66% |
| 2025-03-31 | 155.347 | 169.111 | 199.966 | +8.86% | +18.25% |
| 2025-04-15 | 166.277 | 178.438 | 209.180 | +7.31% | +17.23% |
| 2025-04-30 | 160.404 | 172.336 | 203.512 | +7.44% | +18.09% |
| 2025-05-15 | 154.936 | 165.225 | 195.571 | +6.64% | +18.37% |
| 2025-05-31 | 157.580 | 172.904 | 202.777 | +9.72% | +17.28% |
| 2025-06-15 | 151.185 | 166.925 | 197.334 | +10.41% | +18.22% |
| 2025-06-30 | 156.712 | 171.037 | 201.532 | +9.14% | +17.83% |
| 2025-07-15 | 157.777 | 172.362 | 203.636 | +9.24% | +18.14% |
| 2025-07-31 | 152.474 | 168.528 | 198.657 | +10.53% | +17.88% |
| 2025-08-15 | 154.787 | 166.967 | 197.632 | +7.87% | +18.37% |
| 2025-08-31 | 155.137 | 170.673 | 201.024 | +10.01% | +17.78% |
| 2025-09-15 | 159.375 | 173.238 | 204.059 | +8.70% | +17.79% |
| 2025-09-30 | 152.744 | 169.412 | 201.065 | +10.91% | +18.68% |
| 2025-10-15 | 159.284 | 172.619 | 205.097 | +8.37% | +18.81% |
| 2025-10-31 | 153.565 | 168.419 | 198.036 | +9.67% | +17.59% |
| 2025-11-15 | 167.997 | 180.338 | 211.120 | +7.35% | +17.07% |
| 2025-11-30 | 159.878 | 175.670 | 206.405 | +9.88% | +17.50% |
| 2025-12-15 | 160.470 | 174.823 | 206.012 | +8.94% | +17.84% |
| 2025-12-31 | 165.823 | 180.188 | 211.009 | +8.66% | +17.10% |
| Pooled | 3801.839 | 4140.126 | 4881.112 | +8.90% | +17.90% |

Season labels use initialization month: DJF is December–February, MAM is March–May,
JJA is June–August, and SON is September–November.

| Initialization season | Components vs polar | Scalar vs components |
|---|---:|---:|
| DJF | +9.00% | +17.86% |
| MAM | +7.97% | +17.80% |
| JJA | +9.53% | +18.03% |
| SON | +9.12% | +17.90% |

Relative gaps divide by vector speed and can grow very large near calm.
Float32 reduction can give negative gaps of a few millionths of a metre per second.
The p90 and p99 columns give the 90th and 99th percentiles, respectively.

| Initialization season | Height | Median gap m/s | Mean gap m/s | p90 gap m/s | p99 gap m/s | Max gap m/s | Median relative gap | p99 relative gap |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| All | 10m | 0.005603 | 0.028963 | 0.062992 | 0.388976 | 12.495455 | 0.001123 | 0.341230 |
| All | 100m | 0.005611 | 0.029196 | 0.059674 | 0.404843 | 15.059071 | 0.000745 | 0.208773 |
| DJF | 10m | 0.005408 | 0.024245 | 0.055894 | 0.294605 | 12.495455 | 0.001008 | 0.246461 |
| DJF | 100m | 0.005257 | 0.023545 | 0.050997 | 0.288733 | 14.726501 | 0.000628 | 0.119385 |
| MAM | 10m | 0.006211 | 0.034742 | 0.075836 | 0.486793 | 8.650116 | 0.001363 | 0.448944 |
| MAM | 100m | 0.006213 | 0.034734 | 0.071415 | 0.504203 | 10.132929 | 0.000916 | 0.287409 |
| JJA | 10m | 0.006165 | 0.034646 | 0.073690 | 0.502721 | 9.927805 | 0.001386 | 0.445549 |
| JJA | 100m | 0.006312 | 0.035810 | 0.072497 | 0.533910 | 12.314890 | 0.000967 | 0.313362 |
| SON | 10m | 0.004801 | 0.022219 | 0.049803 | 0.275305 | 10.586609 | 0.000851 | 0.227443 |
| SON | 100m | 0.004853 | 0.022694 | 0.047500 | 0.287019 | 15.059071 | 0.000569 | 0.125578 |

| Arm | Delta-log bytes |
|---|---:|
| polar | 112217 |
| components | 109905 |
| scalar | 120572 |

## Reproduce the comparison

**The measurement uses frozen experimental schemas, not the current production schema.**
Before creating the frozen worktrees, save `components.patch` and `scalar.patch` from the current
checkout's `studies/wind_storage/variants/` directory outside those worktrees. Create three isolated
worktrees at commit `e3bcb44c`, then cherry-pick `1bef88b2` and `ad2ea712` into each. Keep the polar
worktree unchanged; apply the saved component and scalar patches to their respective worktrees
and commit each variant. Use the same
`uv.lock`, H3 weights, and raw cache for all three. The scripts and detailed instructions are in
`studies/wind_storage/README.md`. Generated reports and full records live under
`data/studies/wind_storage/`; the [summary provenance](../assets/wind-storage-provenance.json)
records hashes of the input manifest, H3 weights, measurement records, reporting script, and
unmodified generated summary. The published tables copy the generated summary.

**Two independent scientific reviews found no material blockers.** The reviews checked paired
row inventories, non-wind byte equality, report provenance, and independent raw-grid reductions,
including an extreme cancellation row. OpenAI Codex performed the reviews; human review is
pending. The measurement does not establish that scalar speed is unhelpful at wind-generation
sites. A matched power-forecast comparison is a separate experiment after archive migration and baseline
preparation.
