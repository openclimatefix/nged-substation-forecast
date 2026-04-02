---
status: "Accepted"
date: "2026-04-02"
author: "Software Architect (gemini-3.1-pro-preview)"
tags: ["geospatial", "nwp", "h3", "anti-meridian"]
---

# ADR-013: Longitude Wrap-Around and Anti-Meridian Handling

## 1. Context & Problem Statement
The NGED substation forecast project uses geospatial data that can span the anti-meridian (180 degrees longitude). When slicing global Numerical Weather Prediction (NWP) datasets or mapping H3 hexagons to a regular grid, naive longitude handling can lead to:
1. **Silent Data Loss**: Points near the anti-meridian (e.g., 179.9 and -179.9) are treated as being on opposite sides of the world, causing joins to fail.
2. **Inefficient Downloads**: Slicing a global dataset from -179 to 179 might trigger a download of the entire global dataset instead of just the small region crossing the anti-meridian.
3. **Incorrect Snapping**: Grid snapping formulas that don't account for the 360-degree circularity of longitude will produce incorrect coordinates.

## 2. Options Considered

### Option A: Naive Slicing and Normalization
* **Description:** Treat longitude as a standard linear coordinate and clip to the dataset's native range.
* **Why it was rejected:** This causes massive, unnecessary data downloads when a small region crosses the anti-meridian (e.g., slicing -179 to 179 for a 2-degree span) and results in silent data loss at the boundaries.

### Option B: Wrap-Around Aware Slicing and Circular Math
* **Description:** Implement gap detection to split slices across the anti-meridian and use modulo operations for grid snapping.
* **Why it was selected:** It provides physical correctness for circular coordinates, prevents performance cliffs during data ingestion, and ensures robustness for any global region.

## 3. Decision
We have implemented robust longitude wrap-around and anti-meridian handling across the geospatial and data ingestion pipelines. This includes normalization to `[-180, 180)`, gap detection in slicing, and a circular snapping formula using modulo 360.

## 4. Consequences
* **Positive:** The pipeline is now robust to anti-meridian crossings, ensuring correct data retrieval and mapping for global datasets.
* **Positive:** Improved performance for regions spanning the 180-degree boundary.
* **Negative/Trade-offs:** The slicing and snapping logic is slightly more complex, requiring careful maintenance and testing.
