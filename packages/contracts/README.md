## Contracts

Defines the "data contracts": the schemas defining the precise shape of each data source, and the
semantics.

## Dependency Isolation

This package is designed to be extremely lightweight. It defines the *shape* of the data using Patito and Polars, but it does **not** contain any ML-specific logic or heavy dependencies like MLflow. This ensures that any component in the system (e.g., a data ingestion script or a dashboard) can import these schemas without bringing in the entire ML stack.

## Design principals

- **Naming of columns**: Prefer snake_case, except for acronyms or SI units. For example, capitalise "DER" (the acronym of distributed energy resource) and use upper case for "MW" (megawatts).
- **Semantic checks**: Checking that a value is within range should be fairly generous. The aim is
to catch physically impossible values, rather than possible-but-unlikely values.
