"""Export reviewed text and immutable delivered-coverage charts from a completed report.

Read aggregate coverage means only; do not estimate intervals or reinterpret control knots.
Use --text-only, review figure-text.md, then render with --reviewed-text into a new directory.
"""

import argparse
import hashlib
import json
import math
import subprocess
from importlib.metadata import version
from pathlib import Path
from typing import Any, Final

import altair as alt
import plotting.ocf_theme as ocf
import polars as pl
import vl_convert as vlc

from studies import charts

RECIPES: Final[dict[str, str]] = {
    "production": "Production config",
    "r2_e1_c1_a3w_d3": "Quick-wins quantiles",
}
GROUPS: Final[dict[str, str]] = {"PV": "Solar farms", "Wind": "Wind farms"}
FOLDS: Final[dict[str, str]] = {
    "mid_2025_to_mid_2026": "Fold A: July 2025 to June 2026",
    "backwards_2025_07_to_2026_06": "Fold B: April 2024 to June 2025 (retrospective)",
}
TITLE: Final[str] = "How often do corrected intervals cover outcomes?"
AXIS: Final[str] = "Observed coverage (% of cases)"
TICKS: Final[tuple[int, ...]] = (0, 20, 40, 60, 80, 100)
REPORT_FILES: Final[frozenset[str]] = frozenset(
    {
        "absolute.parquet",
        "contrasts.parquet",
        "synthetic.parquet",
        "sample_semantics.parquet",
        "source_inventory.parquet",
        "report.md",
    }
)
NOTES: Final[tuple[str, ...]] = (
    "Dot: equal-series mean, averaged across three fitting seeds; all leads.",
    "Absolute coverage intervals are not reported; no uncertainty lines are shown.",
    "Nominal targets are descriptive for finite, structured forecast outputs.",
    "Interval endpoints use one-based empirical inverse-CDF ranks ceil(n × q).",
    "Empirical interval ranks have a finite-iid reference, not a calibration guarantee.",
    "Coverage alone cannot rank forecasts; read loss and width alongside it.",
)


def _sha256(*, path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _render_dependencies() -> dict[str, Any]:
    sources = {"studies.charts": charts.__file__, "plotting.ocf_theme": ocf.__file__}
    if any(path is None for path in sources.values()):
        raise ValueError("rendering dependencies must have hashable source files")
    return {
        "source_sha256": {
            name: _sha256(path=Path(path)) for name, path in sources.items() if path is not None
        },
        "versions": {name: version(name) for name in ("altair", "polars", "vl-convert-python")},
    }


def _verify_report(*, root: Path) -> tuple[dict[str, Any], dict[str, str]]:
    manifest_path = root / "manifest.json"
    hashes = {"manifest.json": _sha256(path=manifest_path)}
    manifest = json.loads(manifest_path.read_text())
    files = manifest["files"]
    if set(files) != REPORT_FILES:
        raise ValueError("completed diagnostic output inventory is missing")
    for name, digest in files.items():
        if Path(name).name != name or not (root / name).is_file():
            raise ValueError("report inventory must contain local aggregate files")
        if (root / name).resolve().parent != root.resolve():
            raise ValueError("report inventory escapes the report directory")
        hashes[name] = _sha256(path=root / name)
        if hashes[name] != digest:
            raise ValueError("diagnostic output hash mismatch")
    scorer = Path(__file__).with_name("calibration_diagnostics.py")
    if manifest["script_sha256"] != _sha256(path=scorer):
        raise ValueError("report must come from the reviewed local diagnostic scorer")
    if manifest["fold"] not in FOLDS or manifest["empirical_inverse_cdf"] != "ceil(n*q)-1":
        raise ValueError("unsupported fold or empirical interval semantics")
    for recipe, samples in (("production", 51), ("r2_e1_c1_a3w_d3", 459)):
        seeds = [fit["seed"] for fit in manifest["fits"] if fit["recipe"] == recipe]
        if sorted(seeds) != [0, 1, 2] or manifest["sample_counts"].get(recipe) != samples:
            raise ValueError("required delivered recipe is incomplete or has wrong sample count")
    hashes["scorer"] = _sha256(path=scorer)
    return manifest, hashes


def _rows(*, root: Path) -> pl.DataFrame:
    table = pl.read_parquet(root / "absolute.parquet").filter(
        pl.col("recipe").is_in(list(RECIPES)),
        pl.col("variant") == "delivered",
        pl.col("type").is_in(list(GROUPS)),
        pl.col("lead_band") == "all",
        pl.col("source_era") == "all",
        pl.col("metric").is_in(["coverage80", "coverage90"]),
    )
    rows = []
    for metric in ("coverage80", "coverage90"):
        for group, group_label in GROUPS.items():
            support = []
            for recipe, recipe_label in RECIPES.items():
                selected = table.filter(
                    pl.col("metric") == metric,
                    pl.col("type") == group,
                    pl.col("recipe") == recipe,
                )
                if selected.height != 1:
                    raise ValueError("expected coverage combination must occur exactly once")
                row = selected.row(0, named=True)
                if row["seeds"] != 3 or row["months"] < 6:
                    raise ValueError("chart requires three seeds and at least six months")
                value = row["value_percent"]
                if value is None or not math.isfinite(value) or not 0 <= value <= 100:
                    raise ValueError("coverage must be finite percent between zero and 100")
                counts = tuple(row[name] for name in ("seeds", "series", "months", "rows_per_seed"))
                if any(value is None or value <= 0 for value in counts):
                    raise ValueError("coverage support must be positive")
                support.append(counts)
                rows.append(
                    {"metric": metric, "group": group_label, "model": recipe_label, "value": value}
                )
            if support[0] != support[1]:
                raise ValueError("recipes must have matching aggregate group support")
    return pl.DataFrame(rows)


def _text(*, manifest: dict[str, Any]) -> str:
    lines = [
        "# Delivered coverage figure text",
        "",
        f"Figure 1: {TITLE}",
        "Corrected wind/radiation pipeline; NGED solar and wind farms.",
        FOLDS[manifest["fold"]],
        f"Hyperparameter setting: {manifest['setting']}; device: {manifest['device']}.",
        *NOTES,
        charts.PLANNING_NOTES["exploratory"],
        "Model",
        *RECIPES.values(),
        *GROUPS.values(),
        AXIS,
        "Axis ticks: " + ", ".join(str(tick) for tick in TICKS),
        "80% interval",
        "Nominal target: 80%",
        "90% interval",
        "Nominal target: 90%",
        "",
    ]
    return "\n".join(lines)


def _panel(*, rows: pl.DataFrame, nominal: int) -> alt.LayerChart:
    base = alt.Chart(rows.filter(pl.col("metric") == f"coverage{nominal}"))
    points = base.mark_point(size=100, filled=True, stroke=ocf.BLACK_1, strokeWidth=0.7, aria=False)
    points = points.encode(  # ty: ignore[unresolved-attribute]
        x=alt.X(
            "value:Q",
            title=AXIS,
            scale=alt.Scale(domain=[0, 100]),
            axis=alt.Axis(values=list(TICKS)),
        ),
        y=alt.Y(
            "group:N",
            title=None,
            sort=list(GROUPS.values()),
            axis=alt.Axis(
                labelLimit=charts.LABEL_WIDTH_PX,
                minExtent=charts.LABEL_WIDTH_PX,
                maxExtent=charts.LABEL_WIDTH_PX,
                labelPadding=6,
            ),
        ),
        yOffset=alt.YOffset("model:N", sort=list(RECIPES.values())),
        color=alt.Color(
            "model:N",
            title="Model",
            scale=alt.Scale(domain=list(RECIPES.values()), range=[ocf.DATA_BLUE, ocf.DATA_GREEN]),
        ),
        shape=alt.Shape(
            "model:N",
            title="Model",
            scale=alt.Scale(domain=list(RECIPES.values()), range=["circle", "diamond"]),
        ),
    )
    reference = alt.Chart(pl.DataFrame({"nominal": [nominal]}))
    rule = reference.mark_rule(color=ocf.BLACK_1, strokeDash=[4, 3], aria=False).encode(  # ty: ignore[unresolved-attribute]
        x="nominal:Q"
    )
    label = reference.mark_text(
        align="right",
        dx=-5,
        dy=8,
        baseline="top",
        color=ocf.BLACK_1,
        aria=False,
        text=f"Nominal target: {nominal}%",
    ).encode(x="nominal:Q", y=alt.value(0))  # ty: ignore[unresolved-attribute]
    return (rule + points + label).properties(
        width=charts.PLOT_WIDTH_PX, height=160, title=f"{nominal}% interval"
    )


def _render(*, rows: pl.DataFrame, manifest: dict[str, Any], output: Path) -> None:
    figure = charts.figure(
        panels=[_panel(rows=rows, nominal=nominal) for nominal in (80, 90)],
        number=1,
        title=TITLE,
        subtitle=[
            "Corrected wind/radiation pipeline; NGED solar and wind farms.",
            FOLDS[manifest["fold"]],
            f"Hyperparameter setting: {manifest['setting']}; device: {manifest['device']}.",
            *NOTES,
        ],
        figure_planning="exploratory",
    ).configure_legend(orient="top", direction="horizontal")
    svg = output / "delivered_coverage.svg"
    figure.save(svg)
    subprocess.run(
        ["npx", "svgo@4", "--multipass", "--precision=1", "--final-newline", str(svg)],
        check=True,
        capture_output=True,
    )
    (output / "delivered_coverage.png").write_bytes(vlc.svg_to_png(svg.read_text(), scale=2))


def main() -> None:
    """Verify completed report outputs, export all figure text, and optionally render."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--text-only", action="store_true")
    parser.add_argument("--reviewed-text", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("chart output exists; choose a new immutable directory")
    manifest, hashes = _verify_report(root=args.report)
    rows = _rows(root=args.report)
    text = _text(manifest=manifest)
    if not args.text_only and (
        args.reviewed_text is None or args.reviewed_text.read_text() != text
    ):
        raise ValueError("rendering requires exact previously reviewed figure text")
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "figure-text.md").write_text(text)
    script_hash = _sha256(path=Path(__file__))
    render_dependencies = _render_dependencies()
    if not args.text_only:
        _render(rows=rows, manifest=manifest, output=args.output)
    _, final_hashes = _verify_report(root=args.report)
    if (
        final_hashes != hashes
        or _sha256(path=Path(__file__)) != script_hash
        or _render_dependencies() != render_dependencies
    ):
        raise ValueError("chart inputs or code changed during export")
    provenance = {
        "report_hashes": hashes,
        "script_sha256": script_hash,
        "render_dependencies": render_dependencies,
        "text_only": args.text_only,
        "reviewed_text_sha256": None if args.text_only else _sha256(path=args.reviewed_text),
        "palette": [ocf.DATA_BLUE, ocf.DATA_GREEN],
        "outputs_sha256": {path.name: _sha256(path=path) for path in sorted(args.output.iterdir())},
    }
    (args.output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")


if __name__ == "__main__":
    main()
