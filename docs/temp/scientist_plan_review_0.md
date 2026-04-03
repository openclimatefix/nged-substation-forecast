---
review_iteration: 1
reviewer: "scientist"
total_flaws: 0
critical_flaws: 0
---

# ML Rigor Review

The implementation plan has been reviewed for methodological correctness, data integrity, and ML best practices. The plan is sound and introduces no data leakage or methodological flaws.

## Evaluation of Polars `join_asof` Sorting Concern

The user raised a valid concern regarding the performance cost of explicitly sorting dataframes to fix the Polars `UserWarning` in `_prepare_and_join_nwps`.

**1. Is there a way to ensure the data is already sorted?**
No. The NWP data undergoes several transformations (filtering, feature addition) before the join. More importantly, the raw NWP data is typically sorted by `init_time` (which maps to `available_time`), not by the `by` columns (`valid_time`, `h3_index`). Polars requires the data to be sorted by the `by` columns first, and then the `on` column. Therefore, the data is almost certainly *not* in the correct sort order prior to this step.

**2. Is the sorting step computationally cheap?**
If the data were already sorted, Polars' sorting algorithms (which include fast paths for sorted/partially-sorted data) would make it relatively cheap. However, because the data needs to be re-ordered to group by `valid_time` and `h3_index` primarily, the sort will perform actual computation.

**3. Methodological Necessity (The ML Rigor Perspective)**
This sorting step is **strictly necessary for correctness**. If we attempt to bypass the sort (e.g., by using `.set_sorted()` to trick Polars), `join_asof` will silently produce incorrect matches. In the context of this function, `join_asof` is the critical mechanism that aligns secondary NWP forecasts with the primary forecast without leaking future data (ensuring `other_nwp.available_time <= combined_nwps.available_time`). If the data is unsorted, this temporal guarantee breaks, leading to silent data leakage and invalidating the model's backtesting performance.

**Conclusion:**
The proposed fix in the implementation plan is methodologically correct and robust. Explicitly sorting by `[NwpColumns.VALID_TIME, NwpColumns.H3_INDEX, "available_time"]` is the exact pattern recommended by Polars documentation for `join_asof` with `by` columns. Passing `check_sortedness=False` is appropriate here because Polars currently emits a false-positive warning when `by` groups are used, even when the data is correctly sorted. The performance cost is a necessary trade-off for guaranteed data integrity.

## Other Items
- The transition from `np.datetime64` to `.timestamp()` in the tests is mathematically equivalent and avoids timezone ambiguity.
- The MLflow and Dagster deprecation fixes are standard engineering updates and do not affect the ML methodology.

**Verdict:** Approved. The Architect may proceed with the implementation.
