"""Export reviewed text, then draw five requested-set gain figures with separate settings.

One explicit spec selects one recipe; both panels include every real input. Saved
gain-share ranges are refit variability, not confidence intervals or feature benefit.
"""

import argparse
import hashlib
import json
import os
import subprocess
from importlib.metadata import version
from pathlib import Path
from typing import Any, Final, cast

import altair as alt
import fixed_pipeline_importance_charts as original
import plotting.ocf_theme as ocf
import polars as pl
import requested_feature_importance as report_code
import vl_convert as vlc

from studies import charts

LABELS: Final[dict[str, str]] = {
    **original.LABELS,
    "downward_short_wave_radiation_flux_surface": "Short-wave GHI",
    "downward_long_wave_radiation_flux_surface": "Downward long-wave radiation",
    "precipitation_surface": "Precipitation rate",
    "sun_east": "Sun: east component",
    "sun_north": "Sun: north component",
    "sun_up": "Sun: up component",
    "clear_sky_irradiance": "Clear-sky GHI (30-min mean)",
    "wind_u_10m": "Eastward wind u at 10 m",
    "wind_v_10m": "Northward wind v at 10 m",
    "wind_u_100m": "Eastward wind u at 100 m",
    "wind_v_100m": "Northward wind v at 100 m",
    "wind_direction_10m_sin": "10 m direction sine",
    "wind_direction_10m_cos": "10 m direction cosine",
    "wind_direction_100m_sin": "100 m direction sine",
    "wind_direction_100m_cos": "100 m direction cosine",
}
SETTING_STYLE: Final[dict[str, tuple[str, str, str]]] = {
    "main": ("Main: blue circles", ocf.DATA_BLUE, "circle"),
    "sensitivity": ("Sensitivity: orange diamonds", ocf.BRAND_ORANGE, "diamond"),
}
COMMON_TEXT: Final[tuple[str, ...]] = (
    (
        "Training gain is the reduction in fitting loss credited to an input when trees split "
        "the training data. These descriptive comparisons were added after the original results. "
        "The descriptive comparisons use per-meter XGBoost squared-error models in "
        "National Grid Electricity Distribution's trial area in Lincolnshire, "
        "with corrected wind and radiation inputs."
    ),
    (
        "Main: 500 trees, depth 6, learning rate 0.05, minimum child weight 1, L2 1. "
        "Sensitivity: 1200 trees, depth 4, learning rate 0.03, minimum child weight 50, L2 5."
    ),
    (
        "Main and sensitivity fits sample 80% of rows and allow all columns. "
        "Retained inputs are duplicated "
        "to width 24; duplicate gain returns to its source before each booster's "
        "shares are normalized to one."
    ),
    (
        "Training: April 2024–June 2025 and July 2025–June 2026, each with three "
        "seeds. Meters have equal weight within each refit; six refits have equal "
        "weight within each setting. Settings stay separate."
    ),
    (
        "All real inputs appear, including zero gain. Dot: mean share. Line: "
        "minimum–maximum across six type-averaged refits, not a confidence interval."
    ),
    (
        "Log axes have a 0.01% floor: smaller shares and endpoints, including zero, "
        "appear at the floor. A floor mark does not establish positive gain; "
        "reported values are unchanged."
    ),
    (
        "Training gain divides credit between correlated inputs and does not "
        "establish held-out forecasting benefit."
    ),
)


def _require(condition: bool, message: str) -> None:
    """Reject invalid aggregate/source invariants without exposing private values."""
    if not condition:
        raise ValueError(message)


def _plain(path: str | Path) -> Path:
    """Require canonical explicit unlinked paths."""
    path = Path(path)
    _require(
        condition=path.is_absolute() and path.resolve() == path,
        message="Noncanonical or linked source",
    )
    return path


def _sha(path: Path) -> str:
    """Hash a canonical source file."""
    return hashlib.sha256(_plain(path=path).read_bytes()).hexdigest()


def _definitions(recipe: str) -> list[str]:
    """Return standalone explanations of the selected real feature representations."""
    kind, real = report_code.RECIPES[recipe]
    lines = [
        (
            f"Target type: {'six solar farms' if kind == 'PV' else 'three wind farms'}; "
            f"{len(real)} real inputs."
        )
    ]
    if kind == "PV":
        lines.append(
            "GHI means global horizontal irradiance, in W/m²; the short-wave input "
            "is downward radiation onto a horizontal surface."
        )
    if "clear_sky_irradiance" in real:
        lines.extend(
            [
                (
                    "Sun east/north/up are dimensionless unit-vector components at the "
                    "midpoint of the half-hour ending at valid time; up remains negative "
                    "below the horizon."
                ),
                (
                    "Clear-sky GHI is the Haurwitz model's global horizontal irradiance "
                    "averaged over that half-hour at 30 one-minute midpoints. "
                    "Clear-sky GHI is an approximation, not measured or tilted-panel irradiance."
                ),
                (
                    "Solar geometry and clear sky use valid time and private coordinates, "
                    "known before issuance; no target observations enter these inputs."
                ),
            ]
        )
    if kind == "Wind":
        lines.append(
            "Wind speeds are in m/s. Raw direction, where present, is the "
            "meteorological bearing the wind comes from, in degrees clockwise from "
            "north."
        )
    if "wind_u_10m" in real:
        lines.append(
            "u/v are reconstructed eastward/northward components in m/s: u = −speed "
            "sin(bearing), v = −speed cos(bearing), from corrected "
            "post-interpolation polar inputs at 10 and 100 m. The reconstructed u/v components "
            "are not native upstream components and add no weather information."
        )
    if "wind_direction_10m_sin" in real:
        lines.append(
            "Dimensionless direction sine/cosine encode the same corrected "
            "post-interpolation meteorological bearings at 10 and 100 m. "
            "The direction sine/cosine inputs replace raw angles, contain no u/v inputs, "
            "and add no weather "
            "information."
        )
    return lines


def _verified_report(root: Path, binding: dict[str, Any]) -> tuple[dict[str, Any], dict[Path, str]]:
    """Verify completed report hashes and exact support without recomputing statistics."""
    _require(
        condition=set(binding) == {"path", "manifest_sha256"},
        message="Unknown report-binding fields",
    )
    manifest_path = root / "manifest.json"
    _require(
        condition=_sha(path=manifest_path) == binding["manifest_sha256"],
        message="Importance manifest changed",
    )
    manifest = json.loads(manifest_path.read_text())
    _require(
        condition=manifest["complete"] is True and len(manifest["refits"]) == 60,
        message="Chart requires the completed sixty-fit gain report",
    )
    _require(
        condition=manifest["script_sha256"] == _sha(path=Path(report_code.__file__).resolve()),
        message="Importance report was written by a different reporting script version",
    )
    expected = {
        (recipe, setting, fold, seed)
        for recipe in report_code.RECIPES
        for setting in report_code.SETTINGS
        for fold in report_code.FOLDS
        for seed in range(3)
    }
    actual = [
        (row["recipe"], row["setting"], row["fold"], row["seed"]) for row in manifest["refits"]
    ]
    _require(
        condition=len(set(actual)) == 60 and set(actual) == expected,
        message="Gain report support matrix differs",
    )
    fingerprints = {manifest_path: binding["manifest_sha256"]}
    _require(
        condition=set(manifest["outputs"])
        == {"importance.parquet", "refit_importance.parquet", "report.md", "spec.json"},
        message="Importance output inventory differs",
    )
    _require(
        condition={path.name for path in root.iterdir()}
        == set(manifest["outputs"]) | {"manifest.json"},
        message="Importance report contains an unknown output",
    )
    for name, expected_hash in manifest["outputs"].items():
        path = root / name
        _require(
            condition=_sha(path=path) == expected_hash, message="Importance output hash changed"
        )
        fingerprints[path] = expected_hash
    return manifest, fingerprints


def _read_table(root: Path, recipe: str) -> pl.DataFrame:
    """Select every real feature in both settings with strict aggregate support."""
    all_rows = pl.read_parquet(root / "importance.parquet")
    _require(
        condition=set(all_rows.columns) == report_code.PUBLIC_COLUMNS,
        message="Importance public schema differs",
    )
    rows = all_rows.filter(pl.col("recipe") == recipe)
    kind, real = report_code.RECIPES[recipe]
    _require(
        condition=rows.null_count().sum_horizontal().item() == 0
        and set(rows["series_type"]) == {kind},
        message="Gain rows are missing or have an unexpected type",
    )
    for setting in ("main", "sensitivity"):
        selected = rows.filter(pl.col("setting") == setting)
        expected_count = 6 if kind == "PV" else 3
        _require(
            condition=selected.height == len(real)
            and set(selected["feature"]) == set(real)
            and selected.filter(
                pl.any_horizontal(
                    ~pl.col("mean_share", "min_share", "max_share").is_finite()
                    | ~pl.col("mean_share", "min_share", "max_share").is_between(0, 1)
                )
                | (pl.col("min_share") > pl.col("mean_share"))
                | (pl.col("mean_share") > pl.col("max_share"))
                | (pl.col("refits") != 6)
                | (pl.col("min_series_per_refit") != expected_count)
                | (pl.col("max_series_per_refit") != expected_count)
            ).is_empty()
            and abs(cast(float, selected["mean_share"].sum()) - 1.0) < 1e-12,
            message="Gain feature coverage, ranges or support differs",
        )
    _require(condition=rows.height == 2 * len(real), message="Unexpected setting rows")
    return rows.with_columns(
        label=pl.col("feature").replace_strict(LABELS),
        mean_percent=pl.col("mean_share") * 100,
        min_percent=pl.col("min_share") * 100,
        max_percent=pl.col("max_share") * 100,
    ).sort("setting", "mean_share", "feature", descending=[False, True, False])


def _panel(rows: pl.DataFrame, setting: str) -> charts.Panel:
    """Draw one setting with fixed labeled logarithmic gain-share marks."""
    title, colour, shape = SETTING_STYLE[setting]
    shown = rows.with_columns(
        display_mean_percent=pl.col("mean_percent").clip(
            lower_bound=original.DISPLAY_FLOOR_PERCENT
        ),
        display_min_percent=pl.col("min_percent").clip(lower_bound=original.DISPLAY_FLOOR_PERCENT),
        display_max_percent=pl.col("max_percent").clip(lower_bound=original.DISPLAY_FLOOR_PERCENT),
    )
    scale = alt.Scale(
        type="log", domain=[original.DISPLAY_FLOOR_PERCENT, 100], nice=False, zero=False
    )
    axis = alt.Axis(
        values=list(original.LOG_TICKS_PERCENT), format=".2~f", labelOverlap=False, labelFlush=False
    )
    base = alt.Chart(shown).encode(
        y=alt.Y(
            "label:N",
            sort=rows["label"].to_list(),
            title=None,
            axis=alt.Axis(
                labelLimit=charts.LABEL_WIDTH_PX,
                minExtent=charts.LABEL_WIDTH_PX,
                maxExtent=charts.LABEL_WIDTH_PX,
                labelPadding=8,
                ticks=False,
                domain=False,
            ),
        )
    )
    interval = cast(alt.Chart, base.mark_rule(color=colour, strokeWidth=2)).encode(
        x=alt.X("display_min_percent:Q", title=original.AXIS_TITLE, scale=scale, axis=axis),
        x2=alt.X2("display_max_percent:Q"),
    )
    dot = cast(alt.Chart, base.mark_point(color=colour, shape=shape, filled=True, size=55)).encode(
        x=alt.X("display_mean_percent:Q", title=original.AXIS_TITLE, scale=scale, axis=axis)
    )
    return alt.layer(interval, dot).properties(
        width=charts.PLOT_WIDTH_PX,
        height=alt.Step(25),
        title=alt.TitleParams(title, anchor="start", fontSize=15),
    )


def _text(table: pl.DataFrame, spec: dict[str, Any]) -> tuple[str, list[str]]:
    """Export all standalone figure words in exact render order."""
    subtitle = [*COMMON_TEXT, *_definitions(recipe=spec["recipe"]), *spec["extra_caption"]]
    lines = [
        "# Requested-set importance figure text",
        "",
        f"Figure {spec['number']}: {spec['title']}",
        *subtitle,
        "",
    ]
    for setting in ("main", "sensitivity"):
        lines.extend(
            [
                SETTING_STYLE[setting][0],
                original.AXIS_TITLE,
                *table.filter(pl.col("setting") == setting)["label"].to_list(),
                "",
            ]
        )
    return "\n".join(lines) + "\n", subtitle


def main() -> None:
    """Export exact standalone text or render one final reviewed two-setting figure."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--text-only", action="store_true")
    parser.add_argument("--reviewed-text", type=Path)
    args = parser.parse_args()
    source, output = _plain(path=args.spec), _plain(path=args.output)
    spec_bytes = source.read_bytes()
    spec = json.loads(spec_bytes)
    _require(
        condition=set(spec)
        == {"preparation_only", "recipe", "number", "title", "extra_caption", "report"}
        and spec["preparation_only"] is False
        and spec["recipe"] in report_code.RECIPES
        and type(spec["number"]) is int
        and 9 <= spec["number"] <= 13
        and isinstance(spec["title"], str)
        and spec["title"].strip()
        and isinstance(spec["extra_caption"], list)
        and all(isinstance(line, str) for line in spec["extra_caption"])
        and not output.exists()
        and output.parent.is_dir(),
        message="Figure spec is not final or output is not new",
    )
    root = _plain(path=spec["report"]["path"])
    _require(
        condition=not output.is_relative_to(root) and not root.is_relative_to(output),
        message="Figure output overlaps its source report",
    )
    manifest, fingerprints = _verified_report(root=root, binding=spec["report"])
    for path in (
        source,
        Path(__file__).resolve(),
        Path(original.__file__).resolve(),
        Path(report_code.__file__).resolve(),
        Path(charts.__file__).resolve(),
        Path(ocf.__file__).resolve(),
    ):
        fingerprints[path] = _sha(path=path)
    _require(
        condition=fingerprints[source] == hashlib.sha256(spec_bytes).hexdigest(),
        message="Figure spec changed during loading",
    )
    table = _read_table(root=root, recipe=spec["recipe"])
    text, subtitle = _text(table=table, spec=spec)
    reviewed_hash = None
    if not args.text_only:
        _require(
            condition=args.reviewed_text is not None
            and _plain(path=args.reviewed_text).read_bytes() == text.encode(),
            message="Rendering requires the exact previously reviewed standalone text",
        )
        reviewed_hash = _sha(path=args.reviewed_text)
        fingerprints[args.reviewed_text] = reviewed_hash
    for path, expected_hash in fingerprints.items():
        _require(
            condition=_sha(path=path) == expected_hash,
            message="Source changed before figure export",
        )
    output.mkdir(mode=0o700, exist_ok=False)
    fd = os.open(output / "figure-text.md", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(text.encode())
    names = ["figure-text.md"]
    if not args.text_only:
        panels = [
            _panel(rows=table.filter(pl.col("setting") == setting), setting=setting)
            for setting in ("main", "sensitivity")
        ]
        chart = charts.figure(
            panels=panels,
            number=str(spec["number"]),
            title=spec["title"],
            subtitle=subtitle,
            figure_planning=None,
        )
        svg = output / f"requested_feature_importance_{spec['number']}.svg"
        chart.save(svg)
        svg.chmod(mode=0o600)
        subprocess.run(
            [
                "npx",
                "--offline",
                "svgo@4",
                "--multipass",
                "--precision=1",
                "--final-newline",
                str(svg),
            ],
            check=True,
            capture_output=True,
        )
        svg.chmod(mode=0o600)
        png = output / f"requested_feature_importance_{spec['number']}.png"
        fd = os.open(png, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "wb") as stream:
            stream.write(vlc.svg_to_png(svg.read_text(), scale=2))
        names.extend([svg.name, png.name])
    for path, expected_hash in fingerprints.items():
        _require(
            condition=_sha(path=path) == expected_hash,
            message="Source changed during figure export",
        )
    repeated_manifest, _ = _verified_report(root=root, binding=spec["report"])
    _require(
        condition=repeated_manifest == manifest,
        message="Report provenance changed during figure export",
    )
    provenance = {
        "complete": True,
        "spec_sha256": hashlib.sha256(spec_bytes).hexdigest(),
        "script_sha256": fingerprints[Path(__file__).resolve()],
        "report_manifest_sha256": spec["report"]["manifest_sha256"],
        "report_output_hashes": manifest["outputs"],
        "text_only": args.text_only,
        "reviewed_text_sha256": reviewed_hash,
        "statistics_recomputed": False,
        "shared_sources_sha256": {
            "old_importance_chart": fingerprints[Path(original.__file__).resolve()],
            "new_gain_report": fingerprints[Path(report_code.__file__).resolve()],
            "charts": fingerprints[Path(charts.__file__).resolve()],
            "theme": fingerprints[Path(ocf.__file__).resolve()],
        },
        "palette": [ocf.DATA_BLUE, ocf.BRAND_ORANGE],
        "dependencies": {name: version(name) for name in ("altair", "polars", "vl-convert-python")},
        "outputs": {name: _sha(path=output / name) for name in names},
    }
    fd = os.open(output / "manifest.json", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as stream:
        stream.write(json.dumps(provenance, indent=2) + "\n")
    print(
        "Completed requested-set importance text export"
        if args.text_only
        else "Completed requested-set importance figure; inspect private PNG before publication"
    )


if __name__ == "__main__":
    try:
        main()
    except (
        ValueError,
        OSError,
        TypeError,
        KeyError,
        RuntimeError,
        AssertionError,
        subprocess.SubprocessError,
        pl.exceptions.PolarsError,
    ):
        raise SystemExit(
            "Requested importance figure failed; partial private output preserved; "
            "no automatic retry"
        ) from None
