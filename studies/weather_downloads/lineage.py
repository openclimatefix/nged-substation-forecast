"""One lineage-note format, reused by every product this issue downloads.

One-off throwaway module for
<https://github.com/openclimatefix/nged-substation-forecast/issues/841>. A lineage note is a small
JSON file written alongside each product's data, recording where the data came from, what was
requested, and when — so a later reader of `data/studies/weather/<PRODUCT>/` does not have to
reconstruct the request from the fetch script's git history.
"""

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def write_lineage_note(
    *,
    product_dir: Path,
    source_address: str,
    request_description: str,
    variables: list[str],
    extra: dict[str, Any] | None = None,
    filename: str = "lineage.json",
) -> Path:
    """Write (or overwrite) `<product_dir>/<filename>`.

    Args:
        product_dir: The product's own directory under `data/studies/weather/`.
        source_address: The service the data came from (a base URL, an OPeNDAP catalog address, an
            HTTPS index, or a Zarr store URI). Never a request carrying a coordinate.
        request_description: A human-readable account of what was requested — the model/dataset
            name, the variables, the date range — with no coordinate in it.
        variables: The variable names kept, in their source naming, e.g. `["ssrd", "10si"]`.
        extra: Any further fields worth recording (e.g. measured size, grid spacing used, a note
            about domain coverage gaps).
        filename: Override the default `lineage.json`, for a product directory that holds more than
            one independently-fetched (variable, level) pair — each pair's call must pass a distinct
            filename, or later calls silently overwrite earlier ones' notes.

    Returns:
        The path written.
    """
    product_dir.mkdir(parents=True, exist_ok=True)
    note = {
        "source_address": source_address,
        "request": request_description,
        "variables": variables,
        "retrieved_at_utc": datetime.now(UTC).isoformat(),
    } | (extra or {})
    path = product_dir / filename
    path.write_text(json.dumps(note, indent=2, default=str))
    return path


def write_readme(
    *,
    product_dir: Path,
    product_name: str,
    source_web_page: str,
    script_path: str,
    columns: dict[str, str],
    missing_value_convention: str,
    gotchas: list[str],
    external_docs: dict[str, str],
    lineage_filenames: list[str],
    filename: str = "README.md",
) -> Path:
    """Write (or overwrite) `<product_dir>/<filename>`, a companion to `lineage.json` for humans.

    A lineage note answers "where did this come from and when"; this README answers "how do I read
    it, and how would I get it again" — everything a later reader needs without having to read the
    fetch script itself. See the `data-download` skill for the convention this implements.

    Args:
        product_dir: The product's own directory under `data/studies/weather/`.
        product_name: The product's own name, e.g. "ECMWF IFS HRES 9 km".
        source_web_page: A human-readable web page describing the product or the API that serves it
            — never an API endpoint URL, which belongs in `lineage.json`'s `source_address` instead.
        script_path: Repo-relative path to the script that produced this directory's data, e.g.
            `studies/weather_downloads/fetch_open_meteo_grid.py`.
        columns: Every column name in the written parquet mapped to a one-line description
            including its unit, e.g. `{"shortwave_radiation": "Global horizontal irradiance, W/m2,
            mean over the hour ending at the time column"}`. List every column, not only the value
            columns — a reader who does not already know what `y_index` means cannot use the file.
        missing_value_convention: How a missing reading is represented in the written file — `NaN`,
            Polars null, both, or neither (some products have no missing-value case at all), and
            what causes it (a padded time/step grid, an upstream gap, a value never served).
        gotchas: Known traps a reader would otherwise rediscover the hard way — an upstream data
            defect, a label convention that is easy to get backwards, a range this product's
            aggregated file does not cover, a value that needs clipping or de-averaging before use.
            Each entry is one bullet; keep each to a sentence or two, and point at `lineage.json`'s
            `note` field for the full numbers behind a claim rather than repeating them here.
        external_docs: Further reading mapped to its own URL, e.g. `{"ERA5 documentation":
            "https://..."}` — a product's own technical documentation, a paper describing the
            model, or a page explaining a convention this README only summarises.
        lineage_filenames: The actual filename(s) this product's lineage note(s) were written under
            (the `filename` argument(s) passed to `write_lineage_note` for this product), e.g.
            `["lineage.json"]`, or `["lineage_ASWDIR_S.json", "lineage_ASWDIFD_S.json"]` for a
            product fetched one variable at a time. Interpolated into the README's pointer to the
            lineage note(s) rather than guessed.
        filename: Override the default `README.md`.

    Returns:
        The path written.
    """
    product_dir.mkdir(parents=True, exist_ok=True)
    column_lines = "\n".join(f"- `{name}`: {description}" for name, description in columns.items())
    gotcha_lines = "\n".join(f"- {gotcha}" for gotcha in gotchas) or "- None known."
    docs_lines = (
        "\n".join(f"- [{label}]({url})" for label, url in external_docs.items()) or "- None."
    )
    lineage_lines = ", ".join(f"`{name}`" for name in lineage_filenames)
    readme = f"""# {product_name}

One-off throwaway download for
<https://github.com/openclimatefix/nged-substation-forecast/issues/841>. `product_dir` is
`data/studies/weather/{product_dir.name}/`, several directories below the top-level `data/`
that this repo's `.gitignore` excludes — nothing under it is committed to the repo.

- **Source:** [{source_web_page}]({source_web_page})
- **Re-download with:** `{script_path}` — see that script's own docstring for the exact command
  and any account/licence prerequisite.
- **Full request details, exact date range, and measured row counts/sizes:** {lineage_lines} next
  to this file.

## Columns

{column_lines}

## Missing values

{missing_value_convention}

## Gotchas

{gotcha_lines}

## Further reading

{docs_lines}
"""
    path = product_dir / filename
    path.write_text(readme)
    return path
