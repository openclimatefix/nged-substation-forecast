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
        sites: The sites the contrast is read on, the five decision sites if not given.
        years: The years the contrast exists in.
    """

    name: str
    treatment: str
    reference: str
    planned: bool = False
    sites: tuple[str, ...] | None = None
    years: tuple[str, ...] = ("scored", "replicate")


CONTRASTS: Final[tuple[Contrast, ...]] = (
    Contrast("C1 long history: A1 - W", "a1", "w", planned=True),
    Contrast("C1 beside: A0 - W (years without drift)", "a0", "w"),
    Contrast("C2 assimilation on drift: A3 - A1", "a3", "a1", planned=True),
    Contrast("C3 Kalman against EWMA: A3 - A4", "a3", "a4", planned=True),
    Contrast("C4 drift once filtered: A2 - A3", "a2", "a3", planned=True),
    Contrast("C5 filter replaces weekly refits: A3 - A1 weekly", "a3", "a1_weekly", planned=True),
    Contrast("C6 DP against lagless XGBoost: A1 - X-best", "a1", "x_best", planned=True),
    Contrast("A2 - A0 (filter alone)", "a2", "a0"),
    Contrast("A4_0 - A0 (EWMA alone)", "a4_0", "a0"),
    Contrast("A2 - A4_0", "a2", "a4_0"),
    Contrast("A3 level only - A3", "a3_level", "a3"),
    Contrast("A3 transient only - A3", "a3_anomaly", "a3"),
    Contrast("WK - A3 (best window with the filter)", "wk", "a3"),
    Contrast("A1s - A1 (tuned steps)", "a1s", "a1"),
    Contrast("A1s smooth - A1 smooth (steps, Part 2's drift)", "a1s_smooth", "a1_smooth"),
    Contrast("A1 - A1 smooth (drift step 30 against 3000)", "a1", "a1_smooth"),
    Contrast("Post hoc C1 with smooth drift: A1 smooth - W smooth", "a1_smooth", "w_smooth"),
    Contrast("Post hoc: A1 smooth - A0", "a1_smooth", "a0"),
    Contrast("A1 shrunk 0.25 - A1", "a1_knot_shrunk25", "a1"),
    Contrast("A1 shrunk 0.50 - A1", "a1_knot_shrunk50", "a1"),
    Contrast("A1 shrunk 0.75 - A1", "a1_knot_shrunk75", "a1"),
    Contrast("A1 tied last knot - A1", "a1_knot_pinned", "a1"),
    Contrast("A1 fitted capacity - A1 pinned", "a1_fc", "a1"),
    Contrast("A1 with COVID flag - A1", "a1_cov", "a1"),
    Contrast("A3 fitted capacity - A3", "a3_fc", "a3"),
    Contrast("A1 - A1 4-weekly refits", "a1", "a1_4weekly"),
    Contrast("A1 weekly - A1 4-weekly", "a1_weekly", "a1_4weekly"),
    Contrast("A3 - A3 4-weekly refits", "a3", "a3_4weekly"),
    Contrast("A3 - A3 weekly refits", "a3", "a3_weekly"),
    Contrast("A3 4-weekly - A1 4-weekly", "a3_4weekly", "a1_4weekly"),
    Contrast("A3 weekly - A1 weekly", "a3_weekly", "a1_weekly"),
    Contrast("A3 delay 0 h - A3 (3.5 h)", "a3_delay0", "a3"),
    Contrast("A3 delay 9 h - A3 (3.5 h)", "a3_delay9", "a3"),
    Contrast("A3 delay 24 h - A3 (3.5 h)", "a3_delay24", "a3"),
    Contrast("A3 - X-best with the filter", "a3", "x_best_k"),
    Contrast("X-best with the filter - X-best", "x_best_k", "x_best"),
    Contrast("X-best zero start - X-best", "x_bestz", "x_best"),
    Contrast("X-best offset, zero start - zero start", "x_bestam", "x_bestz"),
    Contrast("X-best offset, quantile start - X-best", "x_bestamq", "x_best"),
    Contrast("A1 - climatology", "a1", "climatology"),
    Contrast("A1 - manual heuristic", "a1", "manual_heuristic"),
    Contrast("A3 - manual heuristic", "a3", "manual_heuristic"),
    Contrast("A1 - Part 2 G-all+D1", "a1", "p2_g_all_d1", years=("scored",)),
    Contrast("A1 - Part 2 G1", "a1", "p2_g1", years=("scored",)),
    Contrast("lead 24.5-48 h: A3 - lagged XGBoost", "a3_lag24", "l_best_d1"),
    Contrast("lead 24.5-48 h: A1 - lagged XGBoost", "a1_lag24", "l_best_d1"),
    Contrast("lead 144.5-168 h: A3 - lagged XGBoost", "a3_lag144", "l_best_d6"),
    Contrast("lead 144.5-168 h: A1 - lagged XGBoost", "a1_lag144", "l_best_d6"),
    Contrast(
        "D10 only: A1 with Nov 2024 jump masked - A1",
        "a1_d10mask",
        "a1",
        sites=("D10",),
        years=("scored",),
    ),
)

PART_2_DIR: Final = (
    common.ML_EXPERIMENTS_DATA_DIR / "00104-differentiable-physics-2-simple-structural-model"
)
"""Experiment 104's private folder, for its stored G-all+D1 and G1 predictions."""
PART_2_SETS: Final[dict[str, str]] = {"p2_g_all_d1": "g_all_d1_centred", "p2_g1": "g1_centred"}
D10_WINDOWS: Final[dict[str, tuple[str, str]]] = {
    "scored": ("2025-11-17", "2025-12-14"),
    "replicate": ("2024-11-08", "2024-12-01"),
}
"""D10's two unlabelled switching windows, dropped from scoring in one robustness column."""


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
    folder = (
        PART_2_DIR / "predictions" / PART_2_SETS[arm]
        if arm in PART_2_SETS
        else common.DATA_DIR / "predictions" / f"{label}_{arm}"
    )
    frame = pl.read_parquet(folder / "predictions.parquet")
    if drop_switching:
        frame = frame.filter(~pl.col("switching_mask"))
    quantiles = frame.select(QUANTILE_COLUMNS).to_numpy().astype(np.float64)
    actual = frame["actual"].to_numpy().astype(np.float64)
    return frame.with_columns(
        qs=pl.Series(common.quantile_score(actual=actual, quantiles=quantiles)),
        ae=(pl.col("actual") - pl.col("mean")).abs(),
        site_name=pl.col("time_series_id").replace_strict(SITE_OF_ID),
    )


def paired(
    *,
    label: str,
    treatment: str,
    reference: str,
    sites: tuple[str, ...],
    drop_switching: bool = True,
    drop_d10_windows: bool = False,
) -> pl.DataFrame:
    """Return both arms' rows on the sites, paired, as the frame `studies.bootstrap` reads.

    Args:
        label: The cut label.
        treatment: The treatment arm.
        reference: The reference arm.
        sites: The sites to keep.
        drop_switching: True to leave out the rows inside the switching masks.
        drop_d10_windows: True to leave out D10's two unlabelled switching windows.

    Returns:
        The paired loss frame.
    """
    arms = {}
    for name in (treatment, reference):
        frame = load_arm(label=label, arm=name, drop_switching=drop_switching).filter(
            pl.col("site_name").is_in(sites)
        )
        if drop_d10_windows:
            start, end = D10_WINDOWS[label]
            inside = (
                (pl.col("site_name") == "D10")
                & (pl.col("valid_time") > pl.lit(start).str.to_datetime(time_zone="UTC"))
                & (pl.col("valid_time") <= pl.lit(end).str.to_datetime(time_zone="UTC"))
            )
            frame = frame.filter(~inside)
        arms[name] = frame.select(
            "time_series_id", "power_fcst_init_time", "valid_time", "seed", "qs", "ae"
        )
    first, second = list(arms.values())
    keys = ["time_series_id", "power_fcst_init_time", "valid_time", "seed"]
    shared = first.select(keys).join(second.select(keys), on=keys, how="inner")
    arms = {name: frame.join(shared, on=keys, how="inner") for name, frame in arms.items()}
    return losses_for_bootstrap(arm_losses=arms)


def estimate(
    *,
    label: str,
    contrast: Contrast,
    sites: tuple[str, ...],
    months: tuple[int, ...] | None = None,
    drop_switching: bool = True,
    drop_d10_windows: bool = False,
) -> tuple[float, float, int]:
    """Return the mean paired difference in quantile score, in the mean absolute error, and rows.

    Args:
        label: The cut label.
        contrast: The contrast.
        sites: The sites to include.
        months: Calendar months of the valid time to keep, or `None` for all.
        drop_switching: True to leave out the rows inside the switching masks.
        drop_d10_windows: True to leave out D10's two unlabelled switching windows.

    Returns:
        The treatment-minus-reference means of `qs` and `ae`, and the number of rows.
    """
    losses = paired(
        label=label,
        treatment=contrast.treatment,
        reference=contrast.reference,
        sites=sites,
        drop_switching=drop_switching,
        drop_d10_windows=drop_d10_windows,
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
        label=label,
        treatment=contrast.treatment,
        reference=contrast.reference,
        sites=contrast.sites or DECISION,
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
    """Return how many of the contrast's sites have a negative difference (treatment better)."""
    count = 0
    for site in contrast.sites or DECISION:
        difference, _, _ = estimate(label=label, contrast=contrast, sites=(site,))
        count += int(difference < 0.0)
    return count


def _contrast_row(*, contrast: Contrast, level: float) -> dict[str, object]:
    """Return one contrast's row: scored year with intervals and splits, replicate beside it."""
    sites = contrast.sites or DECISION
    row: dict[str, object] = {"contrast": contrast.name}
    for label in contrast.years:
        difference, mean_error, _ = estimate(label=label, contrast=contrast, sites=sites)
        row[f"{label} QS"] = difference
        row[f"{label} MAE of mean"] = mean_error
        row[f"{label} sites better"] = sites_improving(label=label, contrast=contrast)
        row[f"{label} all rows"] = estimate(
            label=label, contrast=contrast, sites=sites, drop_switching=False
        )[0]
        if "D10" in sites and len(sites) > 1:
            row[f"{label} D10 windows dropped"] = estimate(
                label=label, contrast=contrast, sites=sites, drop_d10_windows=True
            )[0]
    if "scored" in contrast.years:
        low, high = interval(label="scored", contrast=contrast, level=level)
        row["scored low"], row["scored high"] = low, high
        if len(sites) > 1:
            row["without D16"] = estimate(
                label="scored", contrast=contrast, sites=tuple(s for s in sites if s != "D16")
            )[0]
            row["Nov-Mar"] = estimate(
                label="scored", contrast=contrast, sites=sites, months=WINTER_MONTHS
            )[0]
            row["Apr-Oct"] = estimate(
                label="scored",
                contrast=contrast,
                sites=sites,
                months=tuple(m for m in range(1, 13) if m not in WINTER_MONTHS),
            )[0]
    return row


def contrast_table(*, planned: bool) -> pl.DataFrame:
    """Return the contrast table: scored-year interval and splits, with the replicate beside."""
    level = PLANNED_LEVEL if planned else EXPLORATORY_LEVEL
    rows = [
        _contrast_row(contrast=contrast, level=level)
        for contrast in CONTRASTS
        if contrast.planned == planned
    ]
    return pl.DataFrame(rows, infer_schema_length=None)


def leave_one_site_out() -> pl.DataFrame:
    """Return each planned contrast with each decision site left out, scored and replicate years."""
    rows = []
    for contrast in CONTRASTS:
        if not contrast.planned:
            continue
        for label in ("scored", "replicate"):
            row: dict[str, object] = {"contrast": contrast.name, "year": label}
            row["all five"] = estimate(label=label, contrast=contrast, sites=DECISION)[0]
            for left_out in DECISION:
                kept = tuple(s for s in DECISION if s != left_out)
                row[f"without {left_out}"] = estimate(label=label, contrast=contrast, sites=kept)[0]
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


def season_by_site() -> pl.DataFrame:
    """Return each planned contrast's November-to-March and April-to-October estimate per site."""
    rows = []
    for contrast in CONTRASTS:
        if not contrast.planned:
            continue
        for site in DECISION:
            winter = estimate(
                label="scored", contrast=contrast, sites=(site,), months=WINTER_MONTHS
            )[0]
            summer = estimate(
                label="scored",
                contrast=contrast,
                sites=(site,),
                months=tuple(m for m in range(1, 13) if m not in WINTER_MONTHS),
            )[0]
            rows.append(
                {
                    "contrast": contrast.name,
                    "site": site,
                    "Nov-Mar": winter,
                    "Apr-Oct": summer,
                    "winter more negative": winter < summer,
                }
            )
    return pl.DataFrame(rows)


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
        for arm in ("a0", "a1", "w", "a2", "a3", "a4", "l_best_d1", "l_best_d6"):
            frame = load_arm(label=label, arm=arm).filter(pl.col("site_name").is_in(DECISION))
            actual = frame["actual"].to_numpy().astype(np.float64)
            row: dict[str, object] = {"year": label, "arm": arm}
            winter = frame.filter(pl.col("valid_time").dt.month().is_in((11, 12, 1, 2)))
            row["q95 coverage Nov-Feb"] = float(
                (winter["actual"].to_numpy() <= winter["q95"].to_numpy()).mean()
            )
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
    """Print every table of the report as comma-separated values."""
    print("== Absolute quantile score by arm, points of capacity (rows outside the masks) ==")
    print(absolute_table().write_csv(float_precision=3))
    print(f"== Planned contrasts ({PLANNED_LEVEL:.2f}% intervals), negative = treatment better ==")
    print(contrast_table(planned=True).write_csv(float_precision=3))
    print("== Exploratory contrasts (95% intervals) ==")
    print(contrast_table(planned=False).write_csv(float_precision=3))
    print("== Planned contrasts with each decision site left out (quantile score) ==")
    print(leave_one_site_out().write_csv(float_precision=3))
    print("== Planned contrasts by season at each site, scored year ==")
    print(season_by_site().write_csv(float_precision=3))
    print("== Median forecast minus mean forecast, points ==")
    print(median_minus_mean().write_csv(float_precision=3))
    print("== Tail calibration on the five decision sites ==")
    print(tail_table().write_csv(float_precision=4))
    print("== Quantile score by days ahead ==")
    print(lead_curve().write_csv(float_precision=3))


if __name__ == "__main__":
    main()
