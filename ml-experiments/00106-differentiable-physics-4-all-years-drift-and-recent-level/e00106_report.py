"""Print experiment 106's numbers: absolute scores, the planned contrasts, and the checks.

Every number the write-up quotes comes from this script. It reads the stored prediction sets
(`<cut>_<arm>`), scores each row with the quantile score (twice the mean pinball loss over the 11
levels, in points of effective capacity) on rows outside the hand-filtered switching masks, and
pairs arms row by row.

- **Planned contrasts** (Bonferroni across the six, so a 99.17% interval resampling whole calendar
  months): C1 to C6, for the scored year and, beside it, the replicate year, with the count of
  decision sites that improve, the estimate without D16, the November-to-March and
  April-to-October estimates, and the same contrast on the mean absolute error of the mean forecast.
- **Exploratory contrasts** at 95%.
- **Checks**: the median forecast minus the mean forecast per site and arm, and tail calibration
  (coverage and pinball loss at q01, q05, q95, and q99).

    uv run python ml-experiments/00106-*/e00106_report.py > docs/ml-experiments/00106-*/report.txt
"""

from dataclasses import dataclass
from functools import cache
from typing import Final

import e00106_common as common
import numpy as np
import polars as pl
from studies.bootstrap import bootstrap_difference_at_level
from studies.dp_gam import QUANTILE_COLUMNS, QUANTILE_LEVELS
from studies.experiment_store import losses_for_bootstrap

SITE_OF_ID: Final[dict[int, str]] = {7: "D9", 8: "D11", 10: "D10", 16: "D14", 17: "D5", 18: "D16"}
DECISION: Final[tuple[str, ...]] = common.DECISION_SITES
N_PLANNED: Final[int] = 6
PLANNED_LEVEL: Final[float] = 100.0 - 5.0 / N_PLANNED
EXPLORATORY_LEVEL: Final[float] = 95.0
N_RESAMPLES: Final[int] = 20000
WINTER_MONTHS: Final[tuple[int, ...]] = (11, 12, 1, 2, 3)
TAIL_LEVELS: Final[tuple[str, ...]] = ("q01", "q05", "q95", "q99")


@dataclass(frozen=True)
class Contrast:
    """A treatment arm minus a reference arm, with the scope and the label that go with it.

    Attributes:
        name: The contrast's label in the report.
        treatment: The treatment set's arm name without the cut label.
        reference: The reference set's arm name without the cut label.
        planned: True for the six pre-registered contrasts.
    """

    name: str
    treatment: str
    reference: str
    planned: bool = False


CONTRASTS: Final[tuple[Contrast, ...]] = (
    Contrast("C1 long history: A1 - W", "a1", "w", planned=True),
    Contrast("C1 beside: A0 - W (years without drift)", "a0", "w"),
    Contrast("C2 assimilation on drift: A3 - A1", "a3", "a1", planned=True),
    Contrast("C3 Kalman against EWMA: A3 - A4", "a3", "a4", planned=True),
    Contrast("C4 drift once filtered: A2 - A3", "a2", "a3", planned=True),
    Contrast("A2 - A0 (filter alone)", "a2", "a0"),
    Contrast("A4_0 - A0 (EWMA alone)", "a4_0", "a0"),
    Contrast("A2 - A4_0", "a2", "a4_0"),
    Contrast("A3 level only - A3", "a3_level", "a3"),
    Contrast("A3 transient only - A3", "a3_anomaly", "a3"),
    Contrast("WK - A3 (best window with the filter)", "wk", "a3"),
    Contrast("A1s - A1 (tuned steps)", "a1s", "a1"),
    Contrast("A1s smooth - A1 smooth (steps, Part 2's drift)", "a1s_smooth", "a1_smooth"),
    Contrast("A1 - A1 smooth (drift step 30 against 3000)", "a1", "a1_smooth"),
    Contrast("A1 shrunk 0.25 - A1", "a1_knot_shrunk25", "a1"),
    Contrast("A1 shrunk 0.50 - A1", "a1_knot_shrunk50", "a1"),
    Contrast("A1 shrunk 0.75 - A1", "a1_knot_shrunk75", "a1"),
    Contrast("A1 tied last knot - A1", "a1_knot_pinned", "a1"),
    Contrast("A1 fitted capacity - A1 pinned", "a1_fc", "a1"),
    Contrast("A1 with COVID flag - A1", "a1_cov", "a1"),
    Contrast("A3 fitted capacity - A3", "a3_fc", "a3"),
)


@cache
def load_arm(*, label: str, arm: str, drop_switching: bool = True) -> pl.DataFrame:
    """Read a stored set with its per-row quantile score and mean absolute error.

    Args:
        label: The cut label, `replicate` or `scored`.
        arm: The arm's name.
        drop_switching: True to leave out the rows inside the switching masks.

    Returns:
        The set with columns `site_name`, `qs`, and `ae` added.
    """
    frame = pl.read_parquet(
        common.DATA_DIR / "predictions" / f"{label}_{arm}" / "predictions.parquet"
    )
    if drop_switching:
        frame = frame.filter(~pl.col("switching_mask"))
    quantiles = frame.select(QUANTILE_COLUMNS).to_numpy().astype(np.float64)
    actual = frame["actual"].to_numpy().astype(np.float64)
    return frame.with_columns(
        qs=pl.Series(common.quantile_score(actual=actual, quantiles=quantiles)),
        ae=(pl.col("actual") - pl.col("mean")).abs(),
        site_name=pl.col("time_series_id").replace_strict(SITE_OF_ID),
    )


def paired(*, label: str, treatment: str, reference: str, sites: tuple[str, ...]) -> pl.DataFrame:
    """Return both arms' rows on the sites, paired, as the frame `studies.bootstrap` reads."""
    arms = {}
    for name in (treatment, reference):
        frame = load_arm(label=label, arm=name).filter(pl.col("site_name").is_in(sites))
        arms[name] = frame.select(
            "time_series_id", "power_fcst_init_time", "valid_time", "seed", "qs", "ae"
        )
    first, second = list(arms.values())
    keys = ["time_series_id", "power_fcst_init_time", "valid_time", "seed"]
    shared = first.select(keys).join(second.select(keys), on=keys, how="inner")
    arms = {name: frame.join(shared, on=keys, how="inner") for name, frame in arms.items()}
    return losses_for_bootstrap(arm_losses=arms)


def estimate(
    *, label: str, contrast: Contrast, sites: tuple[str, ...], months: tuple[int, ...] | None = None
) -> tuple[float, float, int]:
    """Return the mean paired difference in quantile score, in the mean absolute error, and rows.

    Args:
        label: The cut label.
        contrast: The contrast.
        sites: The sites to include.
        months: Calendar months of the valid time to keep, or `None` for all.

    Returns:
        The treatment-minus-reference means of `qs` and `ae`, and the number of rows.
    """
    losses = paired(
        label=label, treatment=contrast.treatment, reference=contrast.reference, sites=sites
    )
    if months is not None:
        valid_month = losses["time"].str.split("|").list.get(1).str.slice(5, 2).cast(pl.Int32)
        losses = losses.filter(valid_month.is_in(months))
    means = losses.group_by("arm").agg(
        pl.col("qs").mean(), pl.col("ae").mean(), pl.len().alias("n")
    )
    treatment = means.filter(pl.col("arm") == contrast.treatment).row(0, named=True)
    reference = means.filter(pl.col("arm") == contrast.reference).row(0, named=True)
    return treatment["qs"] - reference["qs"], treatment["ae"] - reference["ae"], int(treatment["n"])


def interval(*, label: str, contrast: Contrast, level: float) -> tuple[float, float]:
    """Return the month-resampled interval of the contrast on the decision sites."""
    losses = paired(
        label=label, treatment=contrast.treatment, reference=contrast.reference, sites=DECISION
    )
    return bootstrap_difference_at_level(
        losses=losses,
        treatment=contrast.treatment,
        reference=contrast.reference,
        metric="qs",
        level=level,
        n_resamples=N_RESAMPLES,
    )


def sites_improving(*, label: str, contrast: Contrast) -> int:
    """Return how many decision sites have a negative contrast (the treatment scores lower)."""
    count = 0
    for site in DECISION:
        difference, _, _ = estimate(label=label, contrast=contrast, sites=(site,))
        count += int(difference < 0.0)
    return count


def contrast_table(*, planned: bool) -> pl.DataFrame:
    """Return the contrast table: scored-year interval and splits, with the replicate beside."""
    rows = []
    level = PLANNED_LEVEL if planned else EXPLORATORY_LEVEL
    for contrast in CONTRASTS:
        if planned != contrast.planned:
            continue
        row: dict[str, object] = {"contrast": contrast.name}
        for label in ("scored", "replicate"):
            difference, mean_error, _ = estimate(label=label, contrast=contrast, sites=DECISION)
            row[f"{label} QS"] = difference
            row[f"{label} MAE of mean"] = mean_error
            row[f"{label} sites better"] = sites_improving(label=label, contrast=contrast)
            if label == "scored":
                low, high = interval(label=label, contrast=contrast, level=level)
                row[f"{label} low"], row[f"{label} high"] = low, high
                row["without D16"] = estimate(
                    label=label,
                    contrast=contrast,
                    sites=tuple(s for s in DECISION if s != "D16"),
                )[0]
                row["Nov-Mar"] = estimate(
                    label=label, contrast=contrast, sites=DECISION, months=WINTER_MONTHS
                )[0]
                row["Apr-Oct"] = estimate(
                    label=label,
                    contrast=contrast,
                    sites=DECISION,
                    months=tuple(m for m in range(1, 13) if m not in WINTER_MONTHS),
                )[0]
        rows.append(row)
    return pl.DataFrame(rows)


def absolute_table() -> pl.DataFrame:
    """Return every arm's quantile score by site for both years, on rows outside the masks."""
    rows = []
    for label in ("scored", "replicate"):
        for name in sorted(path.name for path in (common.DATA_DIR / "predictions").iterdir()):
            if not name.startswith(f"{label}_") or "_d" in name.removeprefix(f"{label}_")[-3:]:
                continue
            arm = name.removeprefix(f"{label}_")
            frame = load_arm(label=label, arm=arm)
            by_site = frame.group_by("site_name").agg(pl.col("qs").mean(), pl.col("ae").mean())
            decision = frame.filter(pl.col("site_name").is_in(DECISION))
            row: dict[str, object] = {
                "year": label,
                "arm": arm,
                "five sites QS": decision["qs"].mean(),
                "five sites MAE of mean": decision["ae"].mean(),
            }
            rows.append({**row, **dict(by_site.select("site_name", "qs").iter_rows())})
    return pl.DataFrame(rows, infer_schema_length=None).sort("year", "five sites QS")


def median_minus_mean() -> pl.DataFrame:
    """Return the mean of (q50 - mean forecast) per site for the headline arms, in points."""
    rows = []
    for label in ("scored", "replicate"):
        for arm in ("a0", "a1", "w", "a2", "a3", "a4"):
            frame = load_arm(label=label, arm=arm)
            by_site = frame.group_by("site_name").agg((pl.col("q50") - pl.col("mean")).mean())
            rows.append({"year": label, "arm": arm, **dict(by_site.iter_rows())})
    return pl.DataFrame(rows, infer_schema_length=None)


def tail_table() -> pl.DataFrame:
    """Return coverage and pinball loss at the four tail levels for the headline arms."""
    rows = []
    for label in ("scored", "replicate"):
        for arm in ("a0", "a1", "w", "a2", "a3", "a4"):
            frame = load_arm(label=label, arm=arm).filter(pl.col("site_name").is_in(DECISION))
            actual = frame["actual"].to_numpy().astype(np.float64)
            row: dict[str, object] = {"year": label, "arm": arm}
            for column in TAIL_LEVELS:
                level = QUANTILE_LEVELS[QUANTILE_COLUMNS.index(column)]
                predicted = frame[column].to_numpy().astype(np.float64)
                error = actual - predicted
                row[f"{column} coverage"] = float((actual <= predicted).mean())
                row[f"{column} pinball"] = float(
                    np.maximum(level * error, (level - 1.0) * error).mean()
                )
            rows.append(row)
    return pl.DataFrame(rows)


def lead_curve() -> pl.DataFrame:
    """Return the quantile score by days ahead of the Kalman, EWMA, and no-drift Kalman arms."""
    rows = []
    for label in ("scored", "replicate"):
        for arm in ("a2", "a3", "a4"):
            for days in range(1, 8):
                name = arm if days == 1 else f"{arm}_d{days}"
                frame = load_arm(label=label, arm=name).filter(pl.col("site_name").is_in(DECISION))
                level = frame["level"].abs().mean()
                anomaly = frame["anomaly"].abs().mean()
                rows.append(
                    {
                        "year": label,
                        "arm": arm,
                        "days ahead": days,
                        "QS": frame["qs"].mean(),
                        "mean |level|": level,
                        "mean |transient|": anomaly,
                    }
                )
    return pl.DataFrame(rows)


def main() -> None:
    """Print every table of the report."""
    pl.Config.set_tbl_rows(200)
    pl.Config.set_tbl_cols(30)
    pl.Config.set_tbl_width_chars(220)
    pl.Config.set_fmt_str_lengths(70)
    rounded = pl.selectors.float()
    print("== Absolute quantile score by arm, points of capacity (rows outside the masks) ==")
    print(absolute_table().with_columns(rounded.round(3)))
    print(
        f"\n== Planned contrasts ({PLANNED_LEVEL:.2f}% intervals), negative = treatment better =="
    )
    print(contrast_table(planned=True).with_columns(rounded.round(3)))
    print("\n== Exploratory contrasts (95% intervals) ==")
    print(contrast_table(planned=False).with_columns(rounded.round(3)))
    print("\n== Median forecast minus mean forecast, points ==")
    print(median_minus_mean().with_columns(rounded.round(3)))
    print("\n== Tail calibration on the five decision sites ==")
    print(tail_table().with_columns(rounded.round(4)))
    print("\n== Quantile score by days ahead ==")
    print(lead_curve().with_columns(rounded.round(3)))


if __name__ == "__main__":
    main()
