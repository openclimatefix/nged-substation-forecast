# Implementation Plan: Move Plot Data to External CSV

## Problem
The Altair plot file (`tests/xgboost_dagster_integration_plot.html`) is still large (116 MB) because the CSV data is embedded directly into the HTML file. This causes slow browser loading times.

## Proposed Solution
1.  **Save Data to External CSV:**
    *   Instead of embedding the CSV data directly into the Altair chart specification, save the CSV data for each substation to a separate file (e.g., `tests/plot_data_substation_<id>.csv`).
2.  **Update Altair Chart:**
    *   Update the Altair chart to use `alt.UrlData` pointing to the generated CSV file.
3.  **Cleanup:**
    *   Ensure the generated CSV files are managed appropriately (e.g., saved in the same directory as the HTML plot).

## Benefits
*   **Faster Loading:** The HTML file will be tiny, and the browser will load the data asynchronously.
*   **Improved Performance:** The browser can handle external CSV files more efficiently than embedded data.
