---
review_iteration: 0
reviewer: "scientist"
total_flaws: 1
critical_flaws: 1
---

# ML Rigor Review

## FLAW-001: Potential Lookahead Bias in Rolling Variance Calculation
* **File & Line Number:** `packages/nged_json_data/src/nged_json_data/clean.py`, lines 68-73
* **The Theoretical Issue:** The rolling variance calculation uses a 6-hour window. If this window is not strictly trailing (i.e., `[T-6h, T]`), it may incorporate future data points to determine if a current point is "bad", leading to lookahead bias.
* **Concrete Failure Mode:** The model might learn to ignore data points that are actually valid but appear "bad" due to future volatility, or vice versa, leading to poor generalization on unseen data.
* **Required Architectural Fix:** The Architect must ensure the `rolling` window is strictly trailing (e.g., by ensuring the data is sorted by `end_time` and using the correct `closed` parameter in Polars) and that this cleaning step is applied appropriately to avoid leaking information across the train/test split if the split is temporal.
