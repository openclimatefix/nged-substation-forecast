"""Draw post hoc requested-feature paired losses from immutable aggregate reports.

Export and review standalone text before rendering. Main and sensitivity share
each row's linear axis; this script never computes a statistical estimate.
"""

import argparse
import hashlib
import json
import math
import os
import subprocess
from importlib.metadata import version
from pathlib import Path
from typing import Any, Final, Literal, cast

import altair as alt
import plotting.ocf_theme as ocf
import polars as pl
import vl_convert as vlc

from studies import charts

type FoldType = Literal["fold_a", "fold_b"]
type SettingType = Literal["main", "sensitivity"]
type JsonObject = dict[str, Any]
type Fingerprints = dict[Path, str]

SETTINGS: Final[tuple[SettingType, ...]] = ("main", "sensitivity")
FOLDS: Final[tuple[FoldType, ...]] = ("fold_a", "fold_b")
MONTHS: Final[dict[FoldType, int]] = {"fold_a": 12, "fold_b": 15}
REPORT_IDS: Final[tuple[str, ...]] = tuple(
    f"{fold}_{setting}_equal" for fold in FOLDS for setting in SETTINGS
)
REPORT_FILES: Final[frozenset[str]] = frozenset(
    {"contrasts.parquet", "levels.parquet", "report.md", "provenance.json"}
)
FAMILIES: Final[tuple[tuple[str, str, str, str], ...]] = (
    ("PV", "solar_minimal_requested", "production_cleaned", "PV 2 − full 24"),
    ("PV", "solar_expanded_requested", "production_cleaned", "PV 11 − full 24"),
    ("PV", "solar_expanded_requested", "solar_minimal_requested", "PV 11 − PV 2"),
    ("Wind", "wind_minimal_requested", "production_cleaned", "Wind 5 − full 24"),
    ("Wind", "wind_minimal_requested", "lean_wind", "Wind 5 − old lean 16"),
    ("Wind", "wind_minimal_uv_requested", "wind_minimal_requested", "Wind 9 u/v − Wind 5"),
    (
        "Wind",
        "wind_minimal_circular_requested",
        "wind_minimal_requested",
        "Wind 7 circular − Wind 5",
    ),
)
AXIS_TITLE: Final[str] = "Fair-CRPS difference (points of capacity; left is lower loss)"
ZERO_LABEL: Final[str] = "Zero: equal loss"
LEGEND: Final[tuple[str, ...]] = ("Main: blue circle", "Sensitivity: orange diamond")
CAPTION: Final[tuple[str, ...]] = (
    (
        "Power forecasts at six solar and three wind farms in National Grid Electricity "
        "Distribution's Lincolnshire trial area "
        "each contain 51 weather-driven predictions. Fair continuous ranked probability score "
        "(fair CRPS) measures their distributional error against observed power; lower is better. "
        "The comparisons match the same forecast cases and were added after the original results."
    ),
    (
        "Each row is the first recipe minus the second; negative means lower loss for the first. "
        "Each farm's effective capacity is the 99th percentile of absolute observed power over "
        "its available history, including readings after validation, not a connection rating. "
        "Loss is divided by that fixed scale for scoring only; one point is 1% of capacity."
    ),
    (
        "PV means photovoltaic. Numbers name real inputs: PV 2 uses temperature and short-wave "
        "radiation; PV 11 adds weather and solar geometry. "
        "Wind 5 uses temperature, speeds, and directions at 10/100 m. "
        "Direction is the angle the wind comes from, measured clockwise from north. "
        "Wind 9 adds eastward/northward velocity components (u/v) in m/s at both heights; "
        "Wind 7 replaces each direction angle with its sine and cosine."
    ),
    (
        "Full 24 is the cleaned full-input control; old lean Wind retains 16 inputs. Corrected "
        "wind/radiation processing and per-meter squared-error fitting are shared. All retained "
        "sets duplicate their own inputs to width 24. Wind representations use the same "
        "corrected polar inputs, not native upstream u/v, and add no weather information."
    ),
    (
        "A trains April 2024–June 2025, tests July 2025–June 2026. Retrospective B trains July "
        "2025–June 2026, tests April 2024–June 2025; its later training conditions the comparison."
    ),
    (
        "Main: 500 trees, depth 6, learning rate 0.05, minimum child weight 1, L2 1. "
        "Sensitivity: 1200 trees, depth 4, learning rate 0.03, minimum child weight 50, L2 5. "
        "Main and sensitivity fits sample 80% of rows and allow all columns; "
        "each setting uses three fitting seeds."
    ),
    (
        "Points: reported paired mean differences, with equal meter weight. Lines: unadjusted "
        "95% bootstrap bounds from 2,000 whole-calendar-month draws and one common seed per draw. "
        "The bootstrap bounds do not describe a seed-averaged estimator or uncertainty "
        "when forecasting new meters. "
        "A draw omits a meter if none of its supported months was drawn; "
        "support is in the appendix."
    ),
    (
        "All seven families were chosen post hoc; comparisons share months and are subject to "
        "multiplicity. Differences do not isolate causal effects of individual inputs. "
        "Absolute losses, support, and the separate equal-month t-intervals are in the appendix."
    ),
)


def _require(condition: bool, message: str) -> None:
    """Reject invalid sources without disclosing private values."""
    if not condition:
        raise ValueError(message)


def _keys(value: Any, names: set[str] | frozenset[str]) -> None:
    """Enforce a closed configuration mapping."""
    _require(
        condition=isinstance(value, dict) and set(value) == names,
        message="Unknown or missing configuration fields",
    )


def _plain(path: str | Path) -> Path:
    """Require explicit canonical paths without links."""
    path = Path(path)
    _require(
        condition=path.is_absolute() and path.resolve() == path,
        message="Noncanonical source or output path",
    )
    return path


def _sha(path: Path) -> str:
    """Hash one canonical file without interpreting its contents."""
    return hashlib.sha256(_plain(path=path).read_bytes()).hexdigest()


def _pin(path: Path, expected: str, pins: Fingerprints) -> None:
    """Bind every consumed source to an explicit SHA256."""
    _require(
        condition=isinstance(expected, str)
        and len(expected) == 64
        and all(character in "0123456789abcdef" for character in expected)
        and _sha(path=path) == expected
        and (path not in pins or pins[path] == expected),
        message="Source fingerprint differs",
    )
    pins[path] = expected


def _recheck(pins: Fingerprints) -> None:
    """Reject source changes before output creation and completion."""
    for path, expected in pins.items():
        _require(condition=_sha(path=path) == expected, message="Source changed during export")


def _one(table: pl.DataFrame, filters: JsonObject) -> JsonObject:
    """Select one exact printed aggregate row, with no averaging."""
    selected = table.filter(*(pl.col(name) == value for name, value in filters.items()))
    _require(condition=selected.height == 1, message="Expected exactly one aggregate row")
    return selected.row(0, named=True)


def _source_pins(spec: JsonObject, pins: Fingerprints) -> None:
    """Bind the report producer and actual imported shared chart/theme sources."""
    _keys(value=spec["sources"], names={"reporter", "charts", "theme"})
    actual = {"charts": Path(charts.__file__).resolve(), "theme": Path(ocf.__file__).resolve()}
    for role, record in spec["sources"].items():
        _keys(value=record, names={"path", "sha256"})
        path = _plain(path=record["path"])
        _require(
            condition=(role == "reporter" and path.name == "fixed_pipeline_report.py")
            or (role in actual and path == actual[role]),
            message="Producer or shared source path differs",
        )
        _pin(path=path, expected=record["sha256"], pins=pins)


def _report(
    spec: JsonObject, report_id: str, fold: FoldType, setting: SettingType, pins: Fingerprints
) -> tuple[pl.DataFrame, pl.DataFrame, JsonObject]:
    """Verify the four consumed aggregate artifacts and completed report specification."""
    binding = spec["reports"][report_id]
    _keys(value=binding, names={"path", "files"})
    _keys(value=binding["files"], names=REPORT_FILES)
    root = _plain(path=binding["path"])
    _require(
        condition={path.name for path in root.iterdir()} == REPORT_FILES | {"fit-manifests.json"}
        and all(path.is_file() and path.resolve() == path for path in root.iterdir()),
        message="Unexpected report inventory",
    )
    for name, expected in binding["files"].items():
        _pin(path=root / name, expected=expected, pins=pins)
    provenance = json.loads((root / "provenance.json").read_bytes())
    registered = provenance["specification"]
    _require(
        condition=provenance["script_sha256"] == spec["sources"]["reporter"]["sha256"]
        and provenance["resamples"] == 2000
        and registered["fold"] == fold
        and registered["setting"] == setting
        and registered["padding"] == "equal"
        and not any("source_era" in run for run in registered["runs"]),
        message="Report is not the expected full-era equal-padding scope",
    )
    contrasts = pl.read_parquet(root / "contrasts.parquet")
    levels = pl.read_parquet(root / "levels.parquet")
    return contrasts, levels, registered


def _support(
    levels: pl.DataFrame, label: str, kind: str, fold: FoldType, registered: JsonObject
) -> JsonObject:
    """Verify three explicit seeds and the full type/month aggregate support."""
    seeds = [run["seed"] for run in registered["runs"] if run["label"] == label]
    _require(
        condition=len(seeds) == 3
        and all(type(seed) is int for seed in seeds)
        and set(seeds) == {0, 1, 2},
        message="Report label lacks exactly three matched seeds",
    )
    row = _one(
        table=levels,
        filters={"label": label, "group": kind, "lead_band": "all", "metric": "crps_norm"},
    )
    _require(
        condition=row["series"] == (6 if kind == "PV" else 3)
        and row["seeds"] == 3
        and row["months"] == MONTHS[fold]
        and row["provisional"] is False
        and type(row["rows_per_seed"]) is int
        and row["rows_per_seed"] > 0
        and all(math.isfinite(row[name]) for name in ("value", "lower95", "upper95"))
        and row["lower95"] <= row["upper95"],
        message="Absolute-score support or finite bounds differ",
    )
    return row


def _rows(spec: JsonObject, pins: Fingerprints) -> pl.DataFrame:
    """Copy all fourteen paired rows at both settings from existing reported contrasts."""
    records = []
    for fold in FOLDS:
        for setting in SETTINGS:
            report_id = f"{fold}_{setting}_equal"
            contrasts, levels, registered = _report(
                spec=spec, report_id=report_id, fold=fold, setting=setting, pins=pins
            )
            for kind, left, right, label in FAMILIES:
                contrast = f"{left} minus {right}"
                definitions = [row for row in registered["contrasts"] if row["label"] == contrast]
                _require(
                    condition=len(definitions) == 1
                    and definitions[0]["left"] == left
                    and definitions[0]["right"] == right
                    and definitions[0]["kind"] == "post_hoc_deciding"
                    and definitions[0]["groups"] == [kind]
                    and "all" in definitions[0]["bands"]
                    and "crps_norm" in definitions[0]["metrics"],
                    message="Deciding contrast definition differs",
                )
                row = _one(
                    table=contrasts,
                    filters={
                        "contrast": contrast,
                        "kind": "post_hoc_deciding",
                        "left": left,
                        "right": right,
                        "group": kind,
                        "lead_band": "all",
                        "metric": "crps_norm",
                    },
                )
                a = _support(levels=levels, label=left, kind=kind, fold=fold, registered=registered)
                b = _support(
                    levels=levels, label=right, kind=kind, fold=fold, registered=registered
                )
                _require(
                    condition=row["seeds"] == 3
                    and row["months"] == MONTHS[fold]
                    and row["provisional"] is False
                    and a["rows_per_seed"] == b["rows_per_seed"]
                    and all(
                        math.isfinite(row[name]) for name in ("difference", "lower95", "upper95")
                    )
                    and row["lower95"] <= row["upper95"]
                    and 1
                    <= row["min_supported_months"]
                    <= row["max_supported_months"]
                    <= MONTHS[fold]
                    and 0 <= row["series_with_missing_months"] <= a["series"]
                    and 0 <= row["bootstrap_draws_dropping_series"] <= 2000
                    and 0 <= row["max_dropped_series_per_draw"] <= a["series"],
                    message="Paired contrast bounds or support differ",
                )
                records.append(
                    {
                        "group": kind,
                        "label": f"{label} · {fold[-1].upper()}",
                        "fold": fold,
                        "setting": setting,
                        "difference": row["difference"],
                        "lower95": row["lower95"],
                        "upper95": row["upper95"],
                        "left_loss": a["value"],
                        "right_loss": b["value"],
                        "seeds": row["seeds"],
                        "months": row["months"],
                        "series": a["series"],
                        "rows_per_seed": a["rows_per_seed"],
                        "min_supported_months": row["min_supported_months"],
                        "max_supported_months": row["max_supported_months"],
                        "series_with_missing_months": row["series_with_missing_months"],
                        "bootstrap_draws_dropping_series": row["bootstrap_draws_dropping_series"],
                        "max_dropped_series_per_draw": row["max_dropped_series_per_draw"],
                    }
                )
    frame = pl.DataFrame(records)
    _require(condition=frame.height == 28, message="Figure requires fourteen rows at both settings")
    return frame


def _table(rows: pl.DataFrame) -> str:
    """Print copied aggregate values directly, without statistical estimation."""
    lines = [
        "# Figure 8 source aggregates",
        "",
        "| " + " | ".join(rows.columns) + " |",
        "|" + "---|" * rows.width,
    ]
    lines.extend(
        "| "
        + " | ".join(f"{value:.3f}" if isinstance(value, float) else str(value) for value in row)
        + " |"
        for row in rows.iter_rows()
    )
    return "\n".join(lines) + "\n"


def _text(spec: JsonObject, rows: pl.DataFrame) -> tuple[str, list[str]]:
    """Export every standalone figure word and source aggregate for exact text review."""
    subtitle = [*CAPTION, *spec["extra_caption"]]
    lines = [f"# Figure 8: {spec['title']}", "", *subtitle, *LEGEND, "", AXIS_TITLE, ZERO_LABEL]
    for kind, heading in (("PV", "Solar farms"), ("Wind", "Wind farms")):
        lines.extend(
            [
                "",
                heading,
                *rows.filter(pl.col("group") == kind)["label"]
                .unique(maintain_order=True)
                .to_list(),
            ]
        )
    return "\n".join(lines) + "\n\n" + _table(rows=rows), subtitle


def _panel(rows: pl.DataFrame, title: str, domain: tuple[float, float]) -> charts.Panel:
    """Put both settings and paired bounds on each row's common linear loss axis."""
    shown = rows.with_columns(
        condition=pl.col("setting").replace_strict(dict(zip(SETTINGS, LEGEND, strict=True)))
    )
    colours = alt.Scale(domain=list(LEGEND), range=[ocf.DATA_BLUE, ocf.BRAND_ORANGE])
    scale = alt.Scale(domain=list(domain), nice=False, zero=False)
    legend = alt.Legend(title=None, orient="top", direction="horizontal", labelLimit=300)
    base = alt.Chart(shown).encode(
        y=alt.Y(
            "label:N",
            sort=shown["label"].unique(maintain_order=True).to_list(),
            title=None,
            axis=alt.Axis(
                labelLimit=charts.LABEL_WIDTH_PX,
                minExtent=charts.LABEL_WIDTH_PX,
                maxExtent=charts.LABEL_WIDTH_PX,
                labelPadding=8,
                ticks=False,
                domain=False,
            ),
        ),
        yOffset=alt.YOffset("condition:N", scale=alt.Scale(domain=list(LEGEND), range=[-5, 5])),
        color=alt.Color("condition:N", scale=colours, legend=legend),
    )
    rules = cast(alt.Chart, base.mark_rule(strokeWidth=2)).encode(
        x=alt.X("lower95:Q", title=AXIS_TITLE, scale=scale), x2=alt.X2("upper95:Q")
    )
    points = cast(alt.Chart, base.mark_point(filled=True, size=55)).encode(
        x=alt.X("difference:Q", title=AXIS_TITLE, scale=scale),
        shape=alt.Shape(
            "condition:N",
            scale=alt.Scale(domain=list(LEGEND), range=["circle", "diamond"]),
            legend=legend,
        ),
    )
    zero = (
        alt.Chart(pl.DataFrame({"zero": [0.0]}))
        .mark_rule(color=ocf.BLACK_1, strokeDash=[4, 3])
        .encode(x=alt.X("zero:Q", scale=scale))
    )
    note = (
        alt.Chart(pl.DataFrame({"zero": [0.0], "text": [ZERO_LABEL]}))
        .mark_text(align="left", dx=4, dy=-4, color=ocf.BLACK_1)
        .encode(x=alt.X("zero:Q", scale=scale), y=alt.value(0), text="text:N")
    )
    return alt.layer(zero, rules, points, note).properties(
        width=charts.PLOT_WIDTH_PX,
        height=alt.Step(38),
        title=alt.TitleParams(title, anchor="start"),
    )


def _write(path: Path, content: bytes) -> None:
    """Create one private output without overwriting."""
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(content)


def _arguments() -> argparse.Namespace:
    """Parse explicit immutable input, output and text-review options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--text-only", action="store_true")
    parser.add_argument("--reviewed-text", type=Path)
    return parser.parse_args()


def main() -> None:
    """Export text first or render only its exact reviewed version into an absent root."""
    args = _arguments()
    spec_path, output = _plain(path=args.spec), _plain(path=args.output)
    pins: Fingerprints = {}
    _pin(path=spec_path, expected=_sha(path=spec_path), pins=pins)
    script = Path(__file__).resolve()
    _pin(path=script, expected=_sha(path=script), pins=pins)
    spec = json.loads(spec_path.read_bytes())
    _keys(
        value=spec,
        names={"preparation_only", "number", "title", "extra_caption", "sources", "reports"},
    )
    _require(
        condition=spec["preparation_only"] is False
        and type(spec["number"]) is int
        and spec["number"] == 8
        and isinstance(spec["title"], str)
        and bool(spec["title"].strip())
        and isinstance(spec["extra_caption"], list)
        and all(isinstance(line, str) for line in spec["extra_caption"])
        and not output.exists()
        and output.parent.is_dir(),
        message="Figure is not final or output is not absent",
    )
    _keys(value=spec["reports"], names=set(REPORT_IDS))
    roots = [_plain(path=binding["path"]) for binding in spec["reports"].values()]
    _require(
        condition=len(set(roots)) == 4
        and all(
            not output.is_relative_to(root) and not root.is_relative_to(output) for root in roots
        ),
        message="Report roots repeat or overlap output",
    )
    _source_pins(spec=spec, pins=pins)
    rows = _rows(spec=spec, pins=pins)
    text, subtitle = _text(spec=spec, rows=rows)
    reviewed_hash = None
    if not args.text_only:
        _require(
            condition=args.reviewed_text is not None
            and _plain(path=args.reviewed_text).read_bytes() == text.encode(),
            message="Rendering requires exact previously reviewed figure text",
        )
        reviewed_hash = _sha(path=args.reviewed_text)
        _pin(path=args.reviewed_text, expected=reviewed_hash, pins=pins)
    _recheck(pins=pins)
    output.mkdir(mode=0o700, exist_ok=False)
    _write(path=output / "figure-text.md", content=text.encode())
    _write(path=output / "selected-aggregates.md", content=_table(rows=rows).encode())
    names = ["figure-text.md", "selected-aggregates.md"]
    if not args.text_only:
        low = min(0.0, float(rows["lower95"].min()), float(rows["difference"].min()))
        high = max(0.0, float(rows["upper95"].max()), float(rows["difference"].max()))
        margin = max((high - low) * 0.12, 0.02)
        domain = (low - margin, high + margin)
        figure = charts.figure(
            panels=[
                _panel(rows=rows.filter(pl.col("group") == kind), title=title, domain=domain)
                for kind, title in (("PV", "Solar farms"), ("Wind", "Wind farms"))
            ],
            number=8,
            title=spec["title"],
            subtitle=subtitle,
            figure_planning=None,
        )
        svg = output / "requested_features_figure_8.svg"
        figure.save(svg)
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
        png = output / "requested_features_figure_8.png"
        _write(path=png, content=vlc.svg_to_png(svg.read_text(), scale=2))
        names.extend([svg.name, png.name])
    _recheck(pins=pins)
    for fold in FOLDS:
        for setting in SETTINGS:
            _report(
                spec=spec,
                report_id=f"{fold}_{setting}_equal",
                fold=fold,
                setting=setting,
                pins=pins,
            )
    _recheck(pins=pins)
    provenance = {
        "complete": True,
        "spec_sha256": pins[spec_path],
        "script_sha256": pins[script],
        "text_only": args.text_only,
        "reviewed_text_sha256": reviewed_hash,
        "statistics_recomputed": False,
        "paired_rows_per_setting": 14,
        "source_hashes_by_path_hash": {
            hashlib.sha256(str(path).encode()).hexdigest(): digest for path, digest in pins.items()
        },
        "reports": {report_id: binding["files"] for report_id, binding in spec["reports"].items()},
        "palette": [ocf.DATA_BLUE, ocf.BRAND_ORANGE],
        "dependencies": {name: version(name) for name in ("altair", "polars", "vl-convert-python")},
        "outputs": {name: _sha(path=output / name) for name in names},
    }
    _write(
        path=output / "manifest.json", content=(json.dumps(provenance, indent=2) + "\n").encode()
    )
    print(
        "Completed private Figure 8 text export"
        if args.text_only
        else "Completed private Figure 8; inspect PNG before publication"
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
            "Requested paired figure failed; private partial output preserved; no automatic retry"
        ) from None
