"""Select public aggregate tables for publication without recomputing statistics.

The JSON specification has a nonempty `tables` list. Each table has `heading`, `source`,
and `columns`, with optional exact `filters`, `column_display_names`, and `decimal_places`.
The top-level `decimal_places` default is two. Output directories must be new.
"""

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Any, Final

import polars as pl

PUBLIC_BASENAMES: Final[frozenset[str]] = frozenset(
    {"levels.parquet", "contrasts.parquet", "absolute.parquet", "sample_semantics.parquet"}
)
PRIVATE_COLUMNS: Final[frozenset[str]] = frozenset(
    {
        "time_series_id",
        "site_id",
        "site",
        "generator",
        "generator_name",
        "generator_id",
        "time_series_name",
        "latitude",
        "longitude",
        "lat",
        "lon",
        "coordinates",
        "actual",
        "power",
        "power_fcst",
        "ensemble_member",
        "valid_time",
        "power_fcst_init_time",
        "nwp_init_time",
        "time",
    }
)
Scalar = str | int | float | bool | None
ExactFilters = dict[str, Scalar | list[Scalar]]


@dataclass(frozen=True)
class TableSpec:
    """A selection of one reviewed aggregate table and its display formatting."""

    heading: str
    source: Path
    filters: ExactFilters
    columns: tuple[str, ...]
    column_display_names: dict[str, str]
    decimal_places: int


def _sha256(*, path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _precision(*, value: Any) -> int:
    if type(value) is not int or not 0 <= value <= 12:
        raise ValueError("decimal_places must be an integer from zero to twelve")
    return value


def _scalar(*, value: Any) -> bool:
    return (
        value is None
        or type(value) in {str, int, bool}
        or (type(value) is float and math.isfinite(value))
    )


def _specifications(*, document: Any, root: Path) -> list[TableSpec]:
    if not isinstance(document, dict) or not isinstance(document.get("tables"), list):
        raise TypeError("specification must contain a tables list")
    default = _precision(value=document.get("decimal_places", 2))
    output = []
    for item in document["tables"]:
        if not isinstance(item, dict):
            raise TypeError("each table specification must be an object")
        heading, source, columns = item.get("heading"), item.get("source"), item.get("columns")
        if not isinstance(heading, str) or not heading.strip() or "\n" in heading:
            raise ValueError("table heading must be a nonempty single line")
        if not isinstance(source, str) or not source:
            raise ValueError("table source must be a path string")
        if (
            not isinstance(columns, list)
            or not columns
            or any(not isinstance(name, str) or not name for name in columns)
            or len(set(columns)) != len(columns)
        ):
            raise ValueError("columns must contain distinct nonempty column names")
        filters = item.get("filters", {})
        if not isinstance(filters, dict) or any(
            not isinstance(name, str)
            or not (
                (isinstance(value, list) and value and all(_scalar(value=v) for v in value))
                or _scalar(value=value)
            )
            for name, value in filters.items()
        ):
            raise ValueError("filters must map column names to finite scalars or nonempty lists")
        names = item.get("column_display_names", {})
        if not isinstance(names, dict) or any(
            key not in columns or not isinstance(value, str) or not value
            for key, value in names.items()
        ):
            raise ValueError("column_display_names must label requested columns")
        shown = [names.get(name, name) for name in columns]
        if len(set(shown)) != len(shown):
            raise ValueError("displayed column names must be distinct")
        path = Path(source)
        output.append(
            TableSpec(
                heading=heading,
                source=(path if path.is_absolute() else root / path).resolve(),
                filters=filters,
                columns=tuple(columns),
                column_display_names=names,
                decimal_places=_precision(value=item.get("decimal_places", default)),
            )
        )
    if not output:
        raise ValueError("specification must request at least one table")
    return output


def _select(*, specification: TableSpec) -> pl.DataFrame:
    if specification.source.name not in PUBLIC_BASENAMES:
        raise ValueError("source basename is not a public aggregate table")
    frame = pl.read_parquet(specification.source)
    if PRIVATE_COLUMNS.intersection(name.casefold() for name in frame.columns):
        raise ValueError("source contains private identifiers or row-level forecast columns")
    if not set(specification.columns).union(specification.filters) <= set(frame.columns):
        raise ValueError("requested selection or filter column is absent")
    for name, value in specification.filters.items():
        values = value if isinstance(value, list) else [value]
        predicate = pl.lit(False)
        for scalar in values:
            predicate = predicate | (
                pl.col(name).is_null() if scalar is None else pl.col(name) == pl.lit(scalar)
            )
        frame = frame.filter(predicate)
    selected = frame.select(*specification.columns)
    if selected.is_empty():
        raise ValueError("requested table selection is empty")
    for name, dtype in selected.schema.items():
        if dtype.is_float() and (
            selected[name].null_count() or not selected[name].is_finite().all()
        ):
            raise ValueError("selected floating-point values must be finite and non-null")
        if dtype.is_nested() or dtype == pl.Binary:
            raise ValueError("publication cells must be scalar")
    return selected


def _cell(*, value: Any, decimal_places: int) -> str:
    text = f"{value:.{decimal_places}f}" if isinstance(value, float) else str(value)
    return text.replace("&", "&amp;").replace("|", "&#124;").replace("\n", "<br>")


def _markdown(*, frame: pl.DataFrame, specification: TableSpec) -> str:
    names = [specification.column_display_names.get(name, name) for name in frame.columns]
    lines = [
        "## " + specification.heading,
        "",
        "| " + " | ".join(_cell(value=name, decimal_places=0) for name in names) + " |",
        "|" + "---|" * len(names),
    ]
    lines.extend(
        "| "
        + " | ".join(
            _cell(value=value, decimal_places=specification.decimal_places) for value in row
        )
        + " |"
        for row in frame.iter_rows()
    )
    return "\n".join(lines)


def main() -> None:
    """Write a fresh publication extract and hashes of its exact source selections."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    spec_bytes = args.spec.read_bytes()
    specifications = _specifications(
        document=json.loads(spec_bytes), root=args.spec.resolve().parent
    )
    sources = {item.source: _sha256(path=item.source) for item in specifications}
    tables = [
        _markdown(frame=_select(specification=item), specification=item) for item in specifications
    ]
    if any(_sha256(path=path) != digest for path, digest in sources.items()):
        raise ValueError("aggregate source changed while extracting tables")
    args.output.mkdir(parents=True, exist_ok=False)
    publication = args.output / "publication.md"
    publication.write_text("\n\n".join(tables) + "\n")
    (args.output / "spec.json").write_bytes(spec_bytes)
    provenance = {
        "spec_sha256": hashlib.sha256(spec_bytes).hexdigest(),
        "script_sha256": _sha256(path=Path(__file__)),
        "polars_version": version("polars"),
        "sources": {str(path): digest for path, digest in sources.items()},
        "outputs": {
            "publication.md": _sha256(path=publication),
            "spec.json": hashlib.sha256(spec_bytes).hexdigest(),
        },
        "statistics_recomputed": False,
    }
    (args.output / "manifest.json").write_text(json.dumps(provenance, indent=2) + "\n")


if __name__ == "__main__":
    main()
