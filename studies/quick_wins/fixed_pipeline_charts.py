"""Draw reviewed contrast specifications from immutable fixed-pipeline report tables.

Use --text-only first, review the resulting figure-text.md, then render into a different new
directory. Charts read reported numbers and perform no statistical estimation.
"""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, cast

import plotting.ocf_theme as ocf
import polars as pl
import vl_convert as vlc

from studies import charts

INTERVAL_TEXT = (
    "Dot: mean loss difference on matched forecasts. Line: 95% interval from repeated draws "
    "of whole months and one matched training seed."
)

DIRECTION_TEXT = "Negative differences favour the forecast named before minus in each comparison."


def _sha256(*, path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _text(*, spec: dict[str, Any]) -> str:
    lines = ["# Fixed-pipeline figure text", ""]
    table = pl.read_parquet(Path(spec["report"]) / "contrasts.parquet")
    for figure in spec["figures"]:
        lines.extend([f"## Figure {figure['number']}: {figure['title']}", ""])
        lines.extend(figure["subtitle"])
        lines.extend(
            [
                INTERVAL_TEXT,
                spec.get("direction_text", DIRECTION_TEXT),
                charts.PLANNING_NOTES["planned"],
                "",
            ]
        )
        for panel in figure["panels"]:
            values = _rows(table=table, panel=panel)
            lines.extend(
                [
                    panel["title"],
                    panel["axis_title"],
                    panel["zero_label"],
                    "Lower loss",
                    *[row["label"] for row in panel["rows"]],
                    *[
                        f"{row['difference']:+.2f} [{row['lower_95']:+.2f}, {row['upper_95']:+.2f}]"
                        for row in values.iter_rows(named=True)
                    ],
                    "",
                ]
            )
    return "\n".join(lines)


def _rows(*, table: pl.DataFrame, panel: dict[str, Any]) -> pl.DataFrame:
    rows = []
    for wanted in panel["rows"]:
        selected = table.filter(
            pl.col("contrast") == wanted["contrast"],
            pl.col("group") == wanted["group"],
            pl.col("lead_band") == wanted.get("lead_band", "all"),
            pl.col("metric") == panel["metric"],
        )
        if selected.height != 1:
            raise ValueError("each chart row must select exactly one reported contrast")
        found = selected.row(0, named=True)
        if found["seeds"] != 3 or found["months"] < 6:
            raise ValueError("publication charts require three seeds and at least six months")
        if found["kind"] != "planned":
            raise ValueError("this chart specification supports preregistered contrasts only")
        rows.append(
            {
                "label": wanted["label"],
                "family": "weather model",
                "planned": True,
                "difference": found["difference"],
                "lower_95": found["lower95"],
                "upper_95": found["upper95"],
            }
        )
    frame = pl.DataFrame(rows)
    if (
        frame.is_empty()
        or any(frame[column].null_count() for column in ("difference", "lower_95", "upper_95"))
        or frame.select(
            pl.any_horizontal(~pl.col("difference", "lower_95", "upper_95").is_finite()).any()
        ).item()
    ):
        raise ValueError("chart values must be finite and nonempty")
    if frame.filter(pl.col("lower_95") > pl.col("upper_95")).height:
        raise ValueError("chart interval endpoints are reversed")
    return frame


def _precise_labels(*, panel_chart: Any, rows: pl.DataFrame) -> Any:
    """Format displayed bounds directly from reported values, avoiding double rounding."""
    labels = {
        row["label"]: (f"{row['difference']:+.2f} [{row['lower_95']:+.2f}, {row['upper_95']:+.2f}]")
        for row in rows.iter_rows(named=True)
    }
    updated = 0
    for layer in panel_chart.layer:
        text_encoding = getattr(layer.encoding, "text", None)
        if (
            getattr(text_encoding, "field", None) == "text"
            or getattr(text_encoding, "shorthand", None) == "text:N"
        ):
            if not isinstance(layer.data, pl.DataFrame):
                raise TypeError("numeric label layer must retain its source frame")
            layer.data = layer.data.with_columns(
                pl.col("label").replace_strict(labels).alias("text")
            )
            updated += layer.data.height
    if updated != rows.height:
        raise ValueError("every reported contrast must have exactly one numeric label")
    return panel_chart


def main() -> None:
    """Export figure text or render report-backed SVG and PNG into a new directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--text-only", action="store_true")
    parser.add_argument("--reviewed-text", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("chart output exists; use a new immutable directory")
    spec = json.loads(args.spec.read_text())
    text = _text(spec=spec)
    if not args.text_only and (
        args.reviewed_text is None or args.reviewed_text.read_text() != text
    ):
        raise ValueError("rendering requires the exact previously reviewed figure text")
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "figure-text.md").write_text(text)
    provenance = {
        "specification": spec,
        "spec_sha256": _sha256(path=args.spec),
        "script_sha256": _sha256(path=Path(__file__)),
        "text_only": args.text_only,
        "palette": [ocf.DATA_BLUE],
    }
    if not args.text_only:
        source = Path(spec["report"])
        for name in ("contrasts.parquet", "report.md", "provenance.json"):
            provenance[f"report_{name}_sha256"] = _sha256(path=source / name)
        table = pl.read_parquet(source / "contrasts.parquet")
        for figure in spec["figures"]:
            panels = []
            for panel in figure["panels"]:
                if panel["metric"] in {"coverage80", "coverage90"}:
                    raise ValueError("loss-direction charts cannot plot coverage")
                rows = _rows(table=table, panel=panel)
                low = min(0.0, cast(float, rows["lower_95"].min()))
                high = max(0.0, cast(float, rows["upper_95"].max()))
                margin = max((high - low) * 0.2, 0.02)
                panel_chart = charts.interval_panel(
                    rows=rows,
                    x_domain=(low - margin, high + margin),
                    x_title=panel["axis_title"],
                    zero_label=panel["zero_label"],
                    better_label="Lower loss",
                    panel_title=panel["title"],
                    figure_planning="planned",
                    family_key=False,
                    value_labels=True,
                    colour_by_family=True,
                )
                panels.append(_precise_labels(panel_chart=panel_chart, rows=rows))
            chart = charts.figure(
                panels=panels,
                number=figure["number"],
                title=figure["title"],
                subtitle=[
                    *figure["subtitle"],
                    INTERVAL_TEXT,
                    spec.get("direction_text", DIRECTION_TEXT),
                ],
                figure_planning="planned",
            )
            stem = f"fixed_pipeline_figure_{figure['number']}"
            svg = args.output / f"{stem}.svg"
            chart.save(svg)
            subprocess.run(
                [
                    "npx",
                    "svgo@4",
                    "--multipass",
                    "--precision=1",
                    "--final-newline",
                    str(svg),
                ],
                check=True,
                capture_output=True,
            )
            (args.output / f"{stem}.png").write_bytes(vlc.svg_to_png(svg.read_text(), scale=2))
    provenance["outputs_sha256"] = {
        path.name: _sha256(path=path) for path in sorted(args.output.iterdir()) if path.is_file()
    }
    (args.output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")


if __name__ == "__main__":
    main()
