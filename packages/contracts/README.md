## Contracts

Defines the "data contracts": the schemas defining the precise shape of each data source, and the
semantics.

## Design principals

- **Naming of columns**: Prefer snake_case, except for acronyms or SI units. For example, capitalise "DER" (the acronym of distributed energy resource) and use upper case for "MW" (megawatts).
- **Semantic checks**: Checking that a value is within range should be fairly generous. The aim is
to catch physically impossible values, rather than possible-but-unlikely values.
