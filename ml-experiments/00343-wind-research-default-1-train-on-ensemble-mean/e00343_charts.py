"""Draw experiment 00343's figures as SVG. Reads `work/score.json` and `work/final/`.

Reuses experiment 00340's chart functions with this experiment's arms. Figure 1: each arm's score
minus the default's on four row sets. Figures 2 to 4: the power curves (full range, 0 to 8 m/s,
and forecast minus observed median). Afterwards optimise with
`npx svgo@4 --multipass --precision=1 --final-newline`.

    uv run --no-sync python ml-experiments/00343-*/e00343_charts.py   (PYTHONPATH: see common)
"""

import json
from pathlib import Path
from typing import Final

import e00340_charts as charts340  # isort: skip  # ty: ignore[unresolved-import]
import e00340_common as common340  # isort: skip  # ty: ignore[unresolved-import]
import e00343_common as common
import plotting.ocf_theme as ocf

DOCS_DIR: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "ml-experiments"
    / "00343-wind-research-default-1-train-on-ensemble-mean"
)
common340.FINAL_DIR = common.FINAL_DIR
charts340.DOCS_DIR = DOCS_DIR
charts340.LEVEL = "0.9950"
charts340.SCOPE = "3 wind farms, 3 seeds, scored at 1 to 10 days."
charts340.ARM_LABELS = {
    "M": "M: trained on the ensemble mean",
    "MS": "MS: M plus member spread",
    "M0": "M0: M without the offset table",
}
charts340.CURVE_ARMS = {
    "default": charts340.GREY,
    "M": ocf.DATA_BLUE,
    "MS": ocf.DATA_GREEN,
    "M0": ocf.BRAND_ORANGE,
}
charts340.CURVE_LABELS = {"default": "default", "M": "M", "MS": "MS", "M0": "M0"}


def main() -> None:
    """Write the figures."""
    score = json.loads((common.WORK_DIR / "score.json").read_text())
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    charts340.figure_1(score=score).save(DOCS_DIR / "figure_1.svg")
    observed, fc, diff, marks = charts340.curve_frames()
    for zoom, name in ((False, "figure_2.svg"), (True, "figure_3.svg")):
        chart = charts340._curve_panels(
            observed=observed,
            fc=fc,
            marks=marks,
            x_max=8.0 if zoom else 25.0,
            quantile_arms=("default", "M"),
            title=charts340._title(
                text="Power against 100 m wind speed" + (", 0 to 8 m/s" if zoom else ""),
                lines=[
                    (
                        "Black: observed median and 5% to 95% band in bins of ERA5 speed at the "
                        "valid time. Coloured: each arm's median forecast in bins of the "
                        "ensemble-mean speed at forecast time; dashed: the default's and M's 5% "
                        "and 95% quantiles. Grey dotted: cut-in; grey dashed: rated speed; blue "
                        "band: experiment 210 idle band. Scored year, all half-hours, 1 to 10 "
                        "days. W1 to W3 are anonymised farms; the last panel pools them."
                    )
                ],
            ),
        )
        chart.save(DOCS_DIR / name)
    charts340.figure_difference(diff=diff, marks=marks).save(DOCS_DIR / "figure_4.svg")


if __name__ == "__main__":
    main()
