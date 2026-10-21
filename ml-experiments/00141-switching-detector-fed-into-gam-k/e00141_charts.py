"""Charts of experiment 141: each arm's quantile score minus K0's, with 95% intervals."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Final

import altair as alt
import plotting.ocf_theme as ocf
import polars as pl
from e00141_common import WORK_DIR

DOCS: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "docs/ml-experiments/00141-switching-detector-fed-into-gam-k"
)
LABELS: Final[dict[str, str]] = {
    "SKF": "S-KF (causal alarm)",
    "SHE": "S-HE (hindsight end)",
    "SORACLE": "S-ORACLE (true interval)",
    "PLACEBO": "Placebo",
    "VARINF": "Variance inflation",
    "SKF_END": "S-KF, on 48 h longer",
    "SORACLE_END": "S-ORACLE, on 72 h longer",
}
COLOURS: Final[dict[str, str]] = {
    "S-KF (causal alarm)": ocf.BRAND_ORANGE,
    "S-HE (hindsight end)": ocf.DATA_BLUE,
    "S-ORACLE (true interval)": ocf.DATA_PURPLE,
    "Placebo": ocf.DATA_SKY,
    "Variance inflation": ocf.DATA_GREEN,
    "S-KF, on 48 h longer": ocf.BRAND_ORANGE,
    "S-ORACLE, on 72 h longer": ocf.DATA_PURPLE,
}
ROWSETS: Final[dict[str, str]] = {
    "transfer": "Targets inside a simulated transfer",
    "alarm48": "Targets in the 48 hours after a causal alarm",
    "tail": "Targets in the 72 hours after a transfer ends",
}


def main() -> None:
    """Write the headline chart."""
    report = json.loads((WORK_DIR / "report.json").read_text())
    rows = []
    for tag, title in ROWSETS.items():
        for arm, label in LABELS.items():
            q = report[f"{tag}/{arm}"]["qs"]
            rows.append(
                {"rows": title, "arm": label, "mean": q["mean"], "lo": q["lo"], "hi": q["hi"]}
            )
    frame = pl.DataFrame(rows)
    order = list(LABELS.values())
    base = alt.Chart(frame).encode(
        y=alt.Y("arm:N", sort=order, title=None, axis=alt.Axis(labelLimit=200)),
        color=alt.Color(
            "arm:N", scale=alt.Scale(domain=order, range=[COLOURS[a] for a in order]), legend=None
        ),
    )
    layer = (
        base.mark_rule(size=3).encode(  # ty: ignore[unresolved-attribute]
            x=alt.X(
                "lo:Q", title="Quantile score minus K0's (points of capacity; lower is better)"
            ),
            x2="hi:Q",
        )
        + base.mark_point(filled=True, size=90).encode(x="mean:Q")  # ty: ignore[unresolved-attribute]
        + alt.Chart(pl.DataFrame({"x": [0.0]})).mark_rule(color="black").encode(x="x:Q")  # ty: ignore[unresolved-attribute]
    )
    chart = (
        layer.properties(width=380, height=170)
        .facet(
            row=alt.Row(
                "rows:N",
                title=None,
                sort=list(ROWSETS.values()),
                header=alt.Header(labelOrient="top", labelAngle=0, labelAlign="left"),
            )
        )
        .properties(
            title=alt.TitleParams(
                "The true transfer interval helps inside a transfer and hurts after it",
                subtitle=[
                    "Dot: estimate. Line: 95% interval over calendar months. Rule at zero:",
                    "GAM-K's filter with no detector (K0). 44 simulated versions, 6 sites,",
                    "2024-07 to 2026-06. Transfers of 25%+ of the receiving load.",
                ],
            )
        )
    )
    DOCS.mkdir(parents=True, exist_ok=True)
    chart.save(DOCS / "arms_vs_k0.svg")
    print("saved")


if __name__ == "__main__":
    main()
