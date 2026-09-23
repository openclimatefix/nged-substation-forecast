"""Draw the anonymised headline chart for the beam/diffuse experiment, Figure 1 of the write-up.

One-off throwaway script for the experiment in
<https://github.com/openclimatefix/nged-substation-forecast/issues/784>. The write-up is
<https://openclimatefix.github.io/nged-substation-forecast/studies/beam-diffuse-split/>.

The chart carries the two findings the experiment has to keep apart, in four panels stacked one
above the other. The top two measure the setup given the weather product's own published beam
against setups given a separation model's estimate of the same beam, which is what the *published
field* buys on top of what global irradiance already implies. The bottom two measure every setup
against the setup given global horizontal irradiance alone, which is what *having* a split buys.
The fitted physical PV model's effects against global irradiance are an order of magnitude larger
than XGBoost's and would flatten them to nothing on a shared scale, so **each panel carries its own
x scale**.

Every difference is in percentage points of capacity, where capacity is each site's 99th
percentile of metered output, and every row label carries the pooled mean absolute error of the two
setups it compares, so a reader can see the error level a difference sits on.

Sites are pooled here and no identifier reaches the chart, because a metered generator's output is
commercially sensitive and this repo is public.

The script only reads the saved per-site summaries and bootstrap intervals, and writes the SVG
straight into the docs assets. Run it with `uv run python studies/beam_diffuse_split/make_chart.py
--results-root <dir> --suffix <suffix>`, where the results directories are named
`beam_diffuse_results_<source><suffix>` and `beam_diffuse_physics_<source><suffix>` under `<dir>`.
Optimise the SVG with `npx svgo@4 --multipass --precision=1 --final-newline` before committing it.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Final, Literal, NamedTuple

import polars as pl
from sources import STUDY_DATA_DIR
from studies.charts import ProductFamily, figure, interval_panel, planning

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
_LOG: Final[logging.Logger] = logging.getLogger("make_chart")

InstrumentType = Literal["xgboost", "physics"]
"""The two instruments: XGBoost, and the fitted physical PV model."""

SourceType = Literal["cams", "open-meteo"]
"""The two irradiance sources the write-up reports, by the key their results directories carry."""

OUTPUT_PATH: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "studies"
    / "assets"
    / "beam_diffuse_split_result.svg"
)
"""Where the write-up's Figure 1 lives."""

SOURCE_NAMES: Final[dict[SourceType, str]] = {"cams": "CAMS", "open-meteo": "ERA5"}
"""Source keys to the product names a reader sees, the satellite retrieval first."""

SOURCE_FAMILIES: Final[dict[SourceType, ProductFamily]] = {
    "cams": "satellite",
    "open-meteo": "reanalysis",
}
"""Each source's product family, which sets its colour."""

UNDRAWN_SOURCES: Final[tuple[str, ...]] = ("cds", "cams_allhours", "ukv", "ukv-live", "icon-d2")
"""Sources with results on disk that the chart deliberately leaves out.

`cds` is the Copernicus route to the same ERA5 fields `open-meteo` serves, checked against each
other by `verify_era5_sources.py`, so drawing both would put one reanalysis on the chart twice
under two names. `cams_allhours` is the reliability-filter sensitivity run, which the write-up
reports in its text. UKV and ICON-D2 belong to the weather-products write-up, not to this one.
Naming the exclusions here is what lets the guard below tell a deliberate omission from a forgotten
one.
"""

ARM_NAMES: Final[dict[str, str]] = {
    "A_global_only": "global only",
    "B_erbs": "Erbs split",
    "B_disc": "DISC split",
    "B_learned": "learned split",
    "C_era5_split": "product's split",
    "D_direct_fraction": "product's beam fraction",
    "P_A_global_only": "global only",
    "P_B_erbs": "Erbs split",
    "P_B_disc": "DISC split",
    "P_C_source_split": "product's split",
}
"""Each setup, by its name in the results, to the words a row label uses for it."""


class Contrast(NamedTuple):
    """One drawn difference: the treatment setup minus the reference setup."""

    treatment: str
    reference: str


class PanelSpec(NamedTuple):
    """One panel of the figure: which instrument, which contrasts, and the x range to draw."""

    instrument: InstrumentType
    contrasts: tuple[Contrast, ...]
    title: str
    x_domain: tuple[float, float]


PLANNED: Final[frozenset[tuple[InstrumentType, Contrast]]] = frozenset(
    {("xgboost", Contrast(treatment="C_era5_split", reference="B_erbs"))}
)
"""The contrasts written into the study plan before any result existed, at the primary setting."""

PANELS: Final[tuple[PanelSpec, ...]] = (
    PanelSpec(
        instrument="xgboost",
        contrasts=(
            Contrast(treatment="C_era5_split", reference="B_erbs"),
            Contrast(treatment="C_era5_split", reference="B_learned"),
        ),
        title="XGBoost: the product's published split against a derived split",
        x_domain=(-0.15, 0.05),
    ),
    PanelSpec(
        instrument="physics",
        contrasts=(Contrast(treatment="P_C_source_split", reference="P_B_erbs"),),
        title="Physical PV model: the published split against Erbs's",
        x_domain=(-0.3, 0.2),
    ),
    PanelSpec(
        instrument="xgboost",
        contrasts=(
            Contrast(treatment="B_erbs", reference="A_global_only"),
            Contrast(treatment="B_disc", reference="A_global_only"),
            Contrast(treatment="B_learned", reference="A_global_only"),
            Contrast(treatment="C_era5_split", reference="A_global_only"),
            Contrast(treatment="D_direct_fraction", reference="A_global_only"),
        ),
        title="XGBoost: every split against global irradiance alone",
        x_domain=(-0.15, 0.0),
    ),
    PanelSpec(
        instrument="physics",
        contrasts=(
            Contrast(treatment="P_B_erbs", reference="P_A_global_only"),
            Contrast(treatment="P_B_disc", reference="P_A_global_only"),
            Contrast(treatment="P_C_source_split", reference="P_A_global_only"),
        ),
        title="Physical PV model: each split against global irradiance",
        x_domain=(-2.0, 0.0),
    ),
)
"""The panels, top to bottom."""

X_TITLE: Final[str] = "Change in mean absolute error (points of capacity, smaller is better)"
"""The axis title every panel shares; `interval_panel` adds which sign is better."""

SUBTITLE: Final[tuple[str, ...]] = (
    "Six PV sites in one 25 km by 23 km box in Lincolnshire, hourly daylight rows, 2019 to 2026.",
    (
        "Each row is one setup's mean absolute error minus another's, pooled over the six sites, "
        "in percentage points of capacity: each site's 99th percentile of output. Each label gives "
        "both setups' own errors."
    ),
    (
        "Dot: estimate. Line: 95% interval from resampling whole months. "
        "Each panel has its own x scale."
    ),
    (
        "CAMS and ERA5 describe weather that has already happened, so this measures what the beam "
        "field carries, not forecast skill."
    ),
)
"""The lines under the title: scope, quantity, marks, and what kind of question this answers."""

PERCENTAGE_POINTS: Final[float] = 100.0


def _results_dir(
    *, root: Path, instrument: InstrumentType, source: SourceType, suffix: str
) -> Path:
    """Return the directory one instrument's run on one source wrote to."""
    stem = "results" if instrument == "xgboost" else "physics"
    return root / f"beam_diffuse_{stem}_{source}{suffix}"


def _raise_on_unlabelled_sources(*, root: Path, suffix: str) -> None:
    """Raise if a results directory names a source no table here names.

    **The failure this exists for is a chart that looks finished with a source missing from it.**
    Drawing iterates `SOURCE_NAMES` rather than the directories on disk, so a source added to the
    experiment and not to the mapping is dropped with no error, no warning, and no gap in the chart
    for a reader to notice. This is R&D code, so it stops rather than degrading.

    Args:
        root: The directory the results directories sit in.
        suffix: The suffix the drawn results directories carry.

    Raises:
        ValueError: If any results directory with this suffix names a source neither table does.
    """
    found = {
        path.name.removeprefix(f"beam_diffuse_{stem}_").removesuffix(suffix)
        for stem in ("results", "physics")
        for path in root.glob(f"beam_diffuse_{stem}_*{suffix}")
        if path.is_dir()
    }
    unlabelled = sorted(found - {*SOURCE_NAMES, *UNDRAWN_SOURCES})
    if unlabelled:
        msg = (
            f"results exist for {unlabelled}, which neither SOURCE_NAMES nor UNDRAWN_SOURCES "
            "names, so they would be left out of the chart without saying so. Add each one to "
            "whichever it belongs in."
        )
        raise ValueError(msg)


def _arm_errors(*, results_dir: Path) -> dict[str, float]:
    """Return each setup's pooled mean absolute error, in percent of capacity.

    Weighted by row count, on the same capped metric and the same weighting as the contrast table
    in `report_results.py`, so the error a label gives and the error the page prints agree.

    Args:
        results_dir: One instrument's run on one source.

    Returns:
        One pooled mean absolute error per setup, keyed by setup name.
    """
    summary = pl.read_parquet(results_dir / "per_site_summary.parquet").filter(
        pl.col("setting") == "primary"
    )
    pooled = summary.group_by("arm").agg(
        mae=(pl.col("mae_capped_fraction_of_capacity") * pl.col("n_rows")).sum()
        / pl.col("n_rows").sum()
        * PERCENTAGE_POINTS
    )
    return dict(zip(pooled["arm"], pooled["mae"], strict=True))


def _panel_rows(*, spec: PanelSpec, root: Path, suffix: str) -> pl.DataFrame:
    """Collect one panel's rows, contrast by contrast, CAMS before ERA5 within each.

    Args:
        spec: The panel.
        root: The directory the results directories sit in.
        suffix: The suffix the drawn results directories carry.

    Returns:
        One row per drawn difference, in points of capacity, with the columns `interval_panel`
        reads.

    Raises:
        ValueError: If a contrast the panel draws is not in the saved intervals exactly once.
    """
    rows = []
    for contrast in spec.contrasts:
        for source, name in SOURCE_NAMES.items():
            results_dir = _results_dir(
                root=root, instrument=spec.instrument, source=source, suffix=suffix
            )
            errors = _arm_errors(results_dir=results_dir)
            found = pl.read_parquet(results_dir / "bootstrap_intervals.parquet").filter(
                (pl.col("setting") == "primary")
                & (pl.col("metric") == "absolute_error_capped_fraction_of_capacity")
                & (pl.col("scope") == "all_sites")
                & (pl.col("treatment") == contrast.treatment)
                & (pl.col("reference") == contrast.reference)
            )
            if found.height != 1:
                msg = f"{contrast} on {source} appears {found.height} times in {results_dir}"
                raise ValueError(msg)
            treatment = f"{ARM_NAMES[contrast.treatment]} ({errors[contrast.treatment]:.2f}%)"
            reference = f"{ARM_NAMES[contrast.reference]} ({errors[contrast.reference]:.2f}%)"
            rows.append(
                {
                    "label": f"{name}: {treatment} − {reference}",
                    "family": SOURCE_FAMILIES[source],
                    "difference": found["difference"].item() * PERCENTAGE_POINTS,
                    "lower_95": found["lower_95"].item() * PERCENTAGE_POINTS,
                    "upper_95": found["upper_95"].item() * PERCENTAGE_POINTS,
                    "planned": (spec.instrument, contrast) in PLANNED,
                }
            )
    return pl.DataFrame(rows)


def main() -> int:
    """Write Figure 1 as an SVG into the docs assets."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-root",
        type=Path,
        default=STUDY_DATA_DIR,
        help="The directory holding the beam_diffuse_results_* and beam_diffuse_physics_* runs.",
    )
    parser.add_argument(
        "--suffix", default="", help="The suffix the drawn results directories carry."
    )
    args = parser.parse_args()
    _raise_on_unlabelled_sources(root=args.results_root, suffix=args.suffix)
    panel_rows = [
        _panel_rows(spec=spec, root=args.results_root, suffix=args.suffix) for spec in PANELS
    ]
    figure_planning = planning(rows=panel_rows)
    panels = [
        interval_panel(
            rows=rows,
            x_domain=spec.x_domain,
            x_title=X_TITLE,
            zero_label="no difference",
            better_label="first setup better",
            panel_title=spec.title,
            family_key=index == 0,
            figure_planning=figure_planning,
        )
        for index, (spec, rows) in enumerate(zip(PANELS, panel_rows, strict=True))
    ]
    chart = figure(
        panels=panels,
        number=1,
        title=(
            "On CAMS the published beam lowers XGBoost's error beyond the Erbs split; "
            "on ERA5 it does not"
        ),
        subtitle=list(SUBTITLE),
        figure_planning=figure_planning,
    )
    chart.save(OUTPUT_PATH)
    _LOG.info("wrote %s", OUTPUT_PATH)
    return 0


if __name__ == "__main__":
    sys.exit(main())
