---
status: "Accepted"
date: "2026-04-02"
author: "Architect (gemini-3.1-pro-preview)"
tags: ["data-engineering", "polars", "design-patterns", "architecture"]
---

# ADR-015: Strategy Pattern for CSV Parsing

## 1. Context & Problem Statement
The NGED data ingestion pipeline needs to process various CSV files (e.g., half-hourly power data, metadata, switching events). These files often have different schemas, date formats, and idiosyncrasies. Initially, the parsing logic was hardcoded within the Dagster assets or scattered across utility functions. This made it difficult to add support for new CSV formats or handle variations in existing formats without modifying the core ingestion logic.

## 2. Options Considered

### Option A: Monolithic Parsing Function
* **Description:** A single, large function with multiple `if/else` blocks to handle different file types based on their names or contents.
* **Why it was rejected:** This violates the Open/Closed Principle. Every time a new file format is introduced, the monolithic function must be modified, increasing the risk of introducing bugs and making the code harder to read and maintain.

### Option B: Separate Parsing Functions
* **Description:** Create distinct functions for each file type (e.g., `parse_power_data`, `parse_metadata`).
* **Why it was rejected:** While better than Option A, it still requires the calling code (the Dagster asset) to know which function to call for which file. It doesn't provide a unified interface for parsing, making it harder to process a directory of mixed CSV files generically.

### Option C: Strategy Pattern for Parsing
* **Description:** Define a common interface (or abstract base class) for CSV parsing. Implement specific parsing strategies (classes) for each file type. A factory or context object determines which strategy to use based on the file metadata or contents.

## 3. Decision
We chose **Option C: Strategy Pattern for Parsing**. We implemented specific parser classes (e.g., `PowerDataParser`, `MetadataParser`) that adhere to a common interface. This allows the core ingestion logic to remain agnostic to the specific file formats.

## 4. Consequences
* **Positive:** Highly extensible. Adding support for a new CSV format simply requires creating a new parser class without modifying existing code. It adheres to the Open/Closed Principle and improves code organization.
* **Negative/Trade-offs:** Introduces slightly more boilerplate code (classes and interfaces) compared to simple functions. Requires developers to understand the Strategy pattern.
