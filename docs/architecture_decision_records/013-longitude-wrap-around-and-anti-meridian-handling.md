---
status: "Deprecated"
date: "2026-04-04"
author: "Software Architect (gemini-3.1-pro-preview)"
tags: ["geospatial", "nwp", "h3", "anti-meridian"]
---

# ADR-013: Longitude Wrap-Around and Anti-Meridian Handling (DEPRECATED)

## 1. Context & Problem Statement
This ADR previously documented the implementation of robust longitude wrap-around and anti-meridian handling for global NWP datasets.

## 2. Decision
This logic has been removed. The project no longer requires complex anti-meridian wrap-around handling as the data ingestion pipeline has been simplified and modernized.

## 3. Consequences
* **Positive:** Reduced complexity in the geospatial and data ingestion pipelines.
* **Negative/Trade-offs:** None.
