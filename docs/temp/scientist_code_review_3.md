---
reviewer: "scientist"
total_flaws: 1
critical_flaws: 1
---

# ML Rigor Review

## FLAW-001: Potential Data Leakage in Scaling Parameter Computation
* **File & Line Number:** `packages/dynamical_data/scaling/compute_scaling_params.py`, line 14
* **The Theoretical Issue:** The script `compute_scaling_params.py` scans all parquet files in `data/*.parquet` to compute the min/max values for scaling. If this directory contains test data, the scaling parameters are being computed using information from the test set, which is a form of data leakage.
* **Concrete Failure Mode:** The model will show artificially high accuracy during backtesting but will fail in real-world NESO grid deployment.
* **Required Architectural Fix:** The Architect must redesign the scaling parameter computation pipeline to ensure that min/max values are computed *only* on the training split, and then applied to the validation/test splits.
