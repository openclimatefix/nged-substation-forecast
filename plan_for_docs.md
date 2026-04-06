# Aims

- Start using MkDocs-Material for all documentation.
- Move relevant documentation from README.md, DESIGN.md, and AGENTS.md to docs/
- Use `mkdocstrings` to auto-generate API docs from the python code.
- Implement `.github/workflows/docs.yml` that automatically updates my docs.

# `.github/workflows/docs.yml`

```yaml
name: Publish Documentation

on:
  push:
    branches:
      - main
    paths:
      - 'docs/**'      # Only run if you change the docs
      - 'mkdocs.yml'
      - 'src/**'       # Or if you change the source code (for mkdocstrings)

# This permission is required so the action can push the built HTML to the gh-pages branch
permissions:
  contents: write

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Install uv
        uses: astral-sh/setup-uv@v3
        with:
          enable-cache: true

      - name: Install Python and Dependencies
        run: |
          uv python install 3.12
          uv sync --dev

      - name: Build and Deploy to GitHub Pages
        run: uv run mkdocs gh-deploy --force
```

# Content

- Overview of the tools used by the project (e.g. `uv workspace`, MLflow, Dagster, etc.), and a
  paragraph about each tool. The overview should be readable by a junior dev who might not know the
  tools we're using. For each tool, explain briefly what it is and - crucially - _why_ we're using
  it, and what we're using it for, and how we're using it. (Some of the tools, like MLflow, are huge
  beasts but we're only using _part_ of the functionality. Make clear which part we're using).
- High-level design philosophy. (Mostly from DESIGN.md. Keep it high level. Probably don't need the level of detail found in most of the ADRs. If ADRs contradict then prefer newer ADRs over older ones. Beware that DESIGN.md might be out of date, so rigorously against the actual code)
- High-level overview of the architecture.
- Code style.
    - Adapt from AGENTS.md and my custom agents.
    - Delete `AGENTS.md`
- Configuration
    - Paths
- Guides for specific user groups: How to:
    - Create a forecast.
    - Add a new dataset.
    - Add a new forecasting model.
    - Run a backtest
- Roadmap
    - Move the yet-to-be-implemented ideas from `DESIGN.md` into the roadmap section.
    - Delete `DESIGN.md`
