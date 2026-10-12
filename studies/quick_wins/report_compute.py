"""Print the training time of each arm at 31 series and a linear estimate at 2,500 series.

Reads the `<arm>: trained in N s, predicted and scored in M s` lines that `fit_arms.py` printed
into the study's logs. The 2,500-series figures are a straight multiple of the 31-series
measurement (about 80.6 times) and do not model memory limits or longer histories, which the
printed notes describe.

    uv run python studies/quick_wins/report_compute.py
"""

import re
import statistics
from typing import Final

import arms as arm_definitions
import polars as pl
import quick_wins_inputs as inputs
from report_tables import markdown_table

LINE: Final[re.Pattern[str]] = re.compile(
    r"^(?P<arm>\S+): trained in (?P<train>\d+) s, predicted and scored in (?P<score>\d+) s$"
)
"""One timing line of `fit_arms.py`."""

SERIES_NOW: Final[int] = 31
"""Series fitted by every arm in the study."""

SERIES_V2: Final[int] = 2500
"""Series the version 2 service is expected to forecast."""

LAYOUTS: Final[dict[str, str]] = {
    "production": "one model per series, squared error",
    "c4_absolute_error": "one model per series, absolute error",
    "c2_early_stopping": "one model per series, early stopping",
    "e1_quantile": "one model per series, 9 quantile levels",
    "d2_pooled_by_type": "one model per series type, squared error",
    "d3_pooled_with_site_id": "one model per series type with a site identifier",
    "r2_c4_c2_c1_a3w_d3": "one model per series type, absolute error and early stopping",
    "r2_e1_c1_a3w_d3": "one model per series type, 9 quantile levels",
}
"""The arms whose training time the page quotes, with the layout each one is."""


def _format(*, seconds: float) -> str:
    if seconds < 120:
        return f"{seconds:.0f} s"
    if seconds < 7200:
        return f"{seconds / 60:.0f} min"
    return f"{seconds / 3600:.1f} h"


def main() -> None:
    """Write `compute.md` under the report directory."""
    seconds: dict[str, list[int]] = {}
    for log in sorted((inputs.OUTPUT_ROOT / "logs").glob("*.log")):
        for line in log.read_text().splitlines():
            match = LINE.match(line.strip())
            if match:
                seconds.setdefault(match["arm"], []).append(int(match["train"]))
    scale = SERIES_V2 / SERIES_NOW
    rows = []
    for arm, layout in LAYOUTS.items():
        if arm not in seconds or arm not in arm_definitions.ALL_ARMS:
            continue
        median = statistics.median(seconds[arm])
        rows.append(
            {
                "arm": arm,
                "layout": layout,
                "fits": len(seconds[arm]),
                "training_31_series_gpu": _format(seconds=median),
                "estimate_2500_series_gpu": _format(seconds=median * scale),
            }
        )
    lines = [
        "# Training time at 31 series, and a linear estimate at 2,500 series",
        "",
        (
            "Median over the logged fits of each arm, on one GPU (training only, in seconds of "
            f"wall time). The estimate multiplies by {scale:.1f}."
        ),
        "",
        markdown_table(frame=pl.DataFrame(rows)),
        "",
        (
            "On 8 CPU cores the production recipe (one model per series, squared error) trained "
            "in 141 s, so about 3.2 hours at 2,500 series."
        ),
        (
            "The 31 series hold 9.76 million training rows, so 2,500 series of the same length "
            "would hold about 790 million, which does not fit in 30 GB of GPU memory as one "
            "table per series type."
        ),
        "",
    ]
    path = inputs.OUTPUT_ROOT / "report" / "compute.md"
    path.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
