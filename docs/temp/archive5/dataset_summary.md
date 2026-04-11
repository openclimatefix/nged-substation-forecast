# Dataset Summary

## Statistical Summary (Substation 110375)

| Statistic | MW | MVAr |
| :--- | :--- | :--- |
| count | 2016.0 | 2016.0 |
| mean | 5.0099 | -1.243072 |
| std | 1.030005 | 0.093507 |
| min | 3.261 | -1.418 |
| 25% | 4.068 | -1.328 |
| 50% | 5.107 | -1.253 |
| 75% | 5.806 | -1.165 |
| max | 7.478 | -1.026 |

## Plot

The plot for substation 110375 can be found at: [substation_110375_mw.html](substation_110375_mw.html)

## Physical Realism Check

- **MW (Active Power):** The values range from 3.261 to 7.478 MW. These seem physically plausible for a primary substation.
- **MVAr (Reactive Power):** The values are negative, which is common for demand-side substations.
- **Timestamps:** The data is at 5-minute intervals (based on the head output). The Patito contract `PowerTimeSeries` requires 30-minute intervals. This might be an issue.

*Audit Note:* The data in `data/NGED/delta/live_primary_flows/` appears to be at 5-minute intervals, while the `PowerTimeSeries` contract requires 30-minute intervals. This discrepancy needs to be addressed in the ingestion pipeline.
