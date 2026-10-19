"""Print experiment 109's numbers: scores, the planned contrasts, the decision rule, and checks.

Every number the write-up quotes comes from this script. It reads the stored prediction sets
(`<cut>_<arm>`), scores each row with the quantile score (twice the mean pinball loss over the 11
levels, in points of effective capacity) on rows outside the hand-filtered switching masks, and
pairs arms row by row.

- **Planned contrasts** (Bonferroni across the five, so 99% intervals resampling whole calendar
  months): C1 to C5, for the scored year and, beside it, the replicate year, with the count of
  decision sites that improve, the estimate without D16 and without D14, the November-to-March and
  April-to-October estimates, the all-rows estimate, the estimate with D10's two unlabelled
  switching windows dropped, and the same contrast on the mean absolute error of the mean forecast.
- **The decision rule** as the plan states it, applied to those numbers.
- **Exploratory contrasts** at 95%, and the onboarding curve.
- **Checks**: the median forecast minus the mean forecast per site and arm, leave-one-site-out
  estimates, and tail calibration (coverage and pinball loss at q01, q05, q95, and q99).

    uv run python ml-experiments/00109-*/e00109_report.py > docs/ml-experiments/00109-*/report.txt
"""

import json
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Final

import e00109_common as common
import numpy as np
import polars as pl
from studies.bootstrap import bootstrap_difference_at_level
from studies.dp_gam import QUANTILE_COLUMNS, QUANTILE_LEVELS
from studies.experiment_store import losses_for_bootstrap
from studies.sources import ML_EXPERIMENTS_DATA_DIR

DECISION: Final[tuple[str, ...]] = common.DECISION_SITES
N_PLANNED: Final[int] = 5
PLANNED_LEVEL: Final[float] = 100.0 - 5.0 / N_PLANNED
EXPLORATORY_LEVEL: Final[float] = 95.0
N_RESAMPLES: Final[int] = 20000
WINTER_MONTHS: Final[tuple[int, ...]] = (11, 12, 1, 2, 3)
SUMMER_MONTHS: Final[tuple[int, ...]] = tuple(m for m in range(1, 13) if m not in WINTER_MONTHS)
TAIL_LEVELS: Final[tuple[str, ...]] = ("q01", "q05", "q95", "q99")
C1_MARGIN: Final[float] = 0.10
D10_WINDOWS: Final[dict[str, tuple[str, str]]] = {
    "scored": ("2025-11-17", "2025-12-14"),
    "replicate": ("2024-11-08", "2024-12-01"),
}
"""D10's two unlabelled switching windows, dropped from scoring in one robustness row."""

N2L_ARMS: Final[tuple[str, ...]] = (
    "n2l_htenth_otenth",
    "n2l_htenth_osame",
    "n2l_htenth_otenfold",
    "n2l_hsame_otenth",
    "n2l_hsame_otenfold",
    "n2l_htenfold_otenth",
    "n2l_htenfold_osame",
    "n2l_htenfold_otenfold",
)
"""The N-2lambda arms other than the one equal to N: heating and other `lambda` times these."""

EXPERIMENT_105: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00105-differentiable-physics-3-pooled-and-per-site-demand-model"
)
EXPERIMENT_106: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00106-differentiable-physics-4-all-years-drift-and-recent-level"
)
EXTERNAL_SETS: Final[dict[str, tuple[Path, tuple[str, ...]]]] = {
    "p_105": (EXPERIMENT_105 / "predictions" / "p", ("scored",)),
    "s_105": (EXPERIMENT_105 / "predictions" / "s", ("scored",)),
    "l_2y_d1_105": (EXPERIMENT_105 / "predictions" / "l_2y_d1", ("scored",)),
    "climatology": (
        EXPERIMENT_106 / "predictions" / "{label}_climatology",
        ("scored", "replicate"),
    ),
    "manual_heuristic": (
        EXPERIMENT_106 / "predictions" / "{label}_manual_heuristic",
        ("scored", "replicate"),
    ),
}
"""Reference sets stored by earlier experiments, read where they cover the same rows."""


@dataclass(frozen=True)
class Contrast:
    """A treatment arm minus a reference arm, with the scope and the label that go with it.

    Attributes:
        name: The contrast's label in the report.
        treatment: The treatment set's arm name without the cut label.
        reference: The reference set's arm name without the cut label.
        planned: True for the five pre-registered contrasts.
        sites: The sites the contrast is read on, the five decision sites if not given.
        years: The years the contrast exists in.
    """

    name: str
    treatment: str
    reference: str
    planned: bool = False
    sites: tuple[str, ...] | None = None
    years: tuple[str, ...] = ("scored", "replicate")


def x_bar() -> str:
    """Return the C4 bar: whichever lagless XGBoost arm scores better on the tuning year.

    The choice reads the tuning-year sets only, before any scored number.
    """
    scores = {}
    for arm in ("x_2y", "x_2yamq"):
        frame = load_arm(label="tuning", arm=arm)
        scores[arm] = float(frame.filter(pl.col("site_name").is_in(DECISION))["qs"].mean())
    return min(scores, key=lambda arm: scores[arm])


def contrasts() -> tuple[Contrast, ...]:
    """Return every contrast of the plan."""
    bar = x_bar()
    return (
        Contrast("C1 N - G (all rows)", "n", "g", planned=True),
        Contrast("C2 N-4w - G-4w", "n_4w", "g_4w", planned=True),
        Contrast("C3 N-3m - G-3m", "n_3m", "g_3m", planned=True),
        Contrast(f"C4 N - {bar}", "n", bar, planned=True),
        Contrast("C5 N-4w - N-full-4w", "n_4w", "n_full_4w", planned=True),
        Contrast("N-full - N", "n_full", "n"),
        *[Contrast(f"N-2lambda {name.removeprefix('n2l_')} - N", name, "n") for name in N2L_ARMS],
        Contrast("N-4w-pin - G-4w-pin", "n_4w_pin", "g_4w_pin"),
        Contrast("N-4w-cappool - N-4w", "n_4w_cappool", "n_4w"),
        Contrast("N-4w - X-pool-4w", "n_4w", "x_pool_4w"),
        Contrast("N-4w - X-4w", "n_4w", "x_4w"),
        Contrast("N-3m - X-3m", "n_3m", "x_3m"),
        Contrast("N-2w - G-2w", "n_2w", "g_2w"),
        Contrast("N-8w - G-8w", "n_8w", "g_8w"),
        Contrast("N-26w - G-26w", "n_26w", "g_26w"),
        Contrast("N - P (experiment 105)", "n", "p_105", years=("scored",)),
        Contrast("N - S (experiment 105)", "n", "s_105", years=("scored",)),
        Contrast("N - lagged XGBoost d1 (experiment 105)", "n", "l_2y_d1_105", years=("scored",)),
        Contrast("N - climatology", "n", "climatology"),
        Contrast("N - manual heuristic", "n", "manual_heuristic"),
        Contrast("N-win4w - G-win4w", "n_4w", "g_4w", years=("winter",)),
    )


@cache
def site_of_id() -> dict[int, str]:
    """Return the label of each time series id in the feature cache."""
    return {f.table.time_series_id: label for label, f in common.load_cache().sites.items()}


def _folder(*, label: str, arm: str) -> Path:
    if arm in EXTERNAL_SETS:
        template, _ = EXTERNAL_SETS[arm]
        return Path(str(template).format(label=label))
    return common.DATA_DIR / "predictions" / f"{label}_{arm}"


@cache
def load_arm(*, label: str, arm: str, drop_switching: bool = True) -> pl.DataFrame:
    """Read a stored set with its per-row quantile score and absolute error of the mean.

    Args:
        label: The cut label: `tuning`, `replicate`, `scored`, or `winter`.
        arm: The arm's name.
        drop_switching: True to leave out the rows inside the switching masks.

    Returns:
        The set with columns `site_name`, `qs`, and `ae` added.
    """
    frame = pl.read_parquet(_folder(label=label, arm=arm) / "predictions.parquet")
    if arm == "l_2y_d1_105":
        # Lagged XGBoost has its own initialisation time (00:00 UTC the day before the valid
        # day). The pooled model has no lags, so its forecast does not depend on the
        # initialisation: pair on the valid time by giving the reference the 09:00 UTC one.
        frame = frame.with_columns(
            power_fcst_init_time=(
                (pl.col("valid_time") - pl.duration(minutes=30)).dt.truncate("1d")
                - pl.duration(days=1)
                + pl.duration(hours=9)
            )
        )
    if drop_switching:
        frame = frame.filter(~pl.col("switching_mask"))
    quantiles = frame.select(QUANTILE_COLUMNS).to_numpy().astype(np.float64)
    actual = frame["actual"].to_numpy().astype(np.float64)
    mean = pl.col("mean") if "mean" in frame.columns else pl.col("q50")
    return frame.with_columns(
        qs=pl.Series(common.quantile_score(actual=actual, quantiles=quantiles)),
        ae=(pl.col("actual") - mean).abs(),
        site_name=pl.col("time_series_id").replace_strict(site_of_id()),
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
        if drop_d10_windows and label in D10_WINDOWS:
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
    if means.height < 2:
        msg = f"no shared rows for {contrast.name} at {sites}"
        raise ValueError(msg)
    treatment = means.filter(pl.col("arm") == contrast.treatment).row(0, named=True)
    reference = means.filter(pl.col("arm") == contrast.reference).row(0, named=True)
    return treatment["qs"] - reference["qs"], treatment["ae"] - reference["ae"], int(treatment["n"])


def interval(*, label: str, contrast: Contrast, level: float) -> tuple[float, float]:
    """Return the month-resampled interval of the contrast on the contrast's sites."""
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


def sites_improving(*, label: str, contrast: Contrast) -> tuple[int, int]:
    """Return how many sites have a negative difference (treatment better), and how many have rows.

    A site with no shared rows, because the short-history arm needs more usable half-hours than
    the site holds in its window, is left out of both counts.
    """
    better, present = 0, 0
    for site in contrast.sites or DECISION:
        try:
            difference, _, _ = estimate(label=label, contrast=contrast, sites=(site,))
        except ValueError:
            continue
        present += 1
        better += int(difference < 0.0)
    return better, present


def _contrast_row(*, contrast: Contrast, level: float) -> dict[str, object]:
    """Return one contrast's row: each year's estimate, with intervals and splits for the scored."""
    sites = contrast.sites or DECISION
    row: dict[str, object] = {"contrast": contrast.name}
    for label in contrast.years:
        difference, mean_error, n_rows = estimate(label=label, contrast=contrast, sites=sites)
        row[f"{label} QS"] = difference
        row[f"{label} MAE of mean"] = mean_error
        row[f"{label} rows"] = n_rows
        better, present = sites_improving(label=label, contrast=contrast)
        row[f"{label} sites better"], row[f"{label} sites with rows"] = better, present
        row[f"{label} all rows"] = estimate(
            label=label, contrast=contrast, sites=sites, drop_switching=False
        )[0]
        if "D10" in sites and len(sites) > 1:
            row[f"{label} D10 windows dropped"] = estimate(
                label=label, contrast=contrast, sites=sites, drop_d10_windows=True
            )[0]
        low, high = interval(label=label, contrast=contrast, level=level)
        row[f"{label} low"], row[f"{label} high"] = low, high
    label = contrast.years[0]
    if len(sites) > 1:
        for left_out in ("D16", "D14"):
            if left_out in sites:
                row[f"without {left_out}"] = estimate(
                    label=label, contrast=contrast, sites=tuple(s for s in sites if s != left_out)
                )[0]
        row["Nov-Mar"] = estimate(
            label=label, contrast=contrast, sites=sites, months=WINTER_MONTHS
        )[0]
        row["Apr-Oct"] = estimate(
            label=label, contrast=contrast, sites=sites, months=SUMMER_MONTHS
        )[0]
    return row


def contrast_table(*, planned: bool) -> pl.DataFrame:
    """Return the contrast table with intervals at the planned or the exploratory level."""
    level = PLANNED_LEVEL if planned else EXPLORATORY_LEVEL
    rows = []
    for contrast in contrasts():
        if contrast.planned != planned:
            continue
        try:
            rows.append(_contrast_row(contrast=contrast, level=level))
        except (FileNotFoundError, ValueError) as error:
            print(f"# skipped {contrast.name}: {error}")
    return pl.DataFrame(rows, infer_schema_length=None)


def leave_one_site_out() -> pl.DataFrame:
    """Return each planned contrast with each decision site left out, scored and replicate years."""
    rows = []
    for contrast in contrasts():
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
    """Return every arm's quantile score by site for each cut, on rows outside the masks."""
    rows = []
    for label in ("scored", "replicate", "winter", "tuning"):
        folder = common.DATA_DIR / "predictions"
        for name in sorted(path.name for path in folder.iterdir()):
            if not name.startswith(f"{label}_"):
                continue
            arm = name.removeprefix(f"{label}_")
            frame = load_arm(label=label, arm=arm)
            by_site = frame.group_by("site_name").agg(pl.col("qs").mean(), pl.col("ae").mean())
            decision = frame.filter(pl.col("site_name").is_in(DECISION))
            row: dict[str, object] = {
                "cut": label,
                "arm": arm,
                "five sites QS": decision["qs"].mean(),
                "five sites MAE of mean": decision["ae"].mean(),
            }
            rows.append({**row, **dict(by_site.select("site_name", "qs").iter_rows())})
    return pl.DataFrame(rows, infer_schema_length=None).sort("cut", "five sites QS")


def median_minus_mean() -> pl.DataFrame:
    """Return the mean of (q50 - mean forecast) per site for the headline arms, in points."""
    rows = []
    for label in ("scored", "replicate"):
        for arm in ("g", "n", "n_full", "n_4w", "g_4w", "n_3m", "g_3m", "n_full_4w"):
            frame = load_arm(label=label, arm=arm)
            by_site = frame.group_by("site_name").agg((pl.col("q50") - pl.col("mean")).mean())
            rows.append({"year": label, "arm": arm, **dict(by_site.iter_rows())})
    return pl.DataFrame(rows, infer_schema_length=None)


def tail_table() -> pl.DataFrame:
    """Return coverage and pinball loss at the four tail levels for the headline arms."""
    rows = []
    for label in ("scored", "replicate"):
        for arm in ("g", "n", "n_full", "n_4w", "g_4w", "n_3m", "g_3m", "n_full_4w"):
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


def onboarding_curve() -> pl.DataFrame:
    """Return the quantile score of the pooled and per-site arm at each history, both years."""
    rows = []
    for label in ("scored", "replicate"):
        for weeks, n_arm, g_arm in (
            (2, "n_2w", "g_2w"),
            (4, "n_4w", "g_4w"),
            (8, "n_8w", "g_8w"),
            (13, "n_3m", "g_3m"),
            (26, "n_26w", "g_26w"),
        ):
            n_frame = load_arm(label=label, arm=n_arm).filter(pl.col("site_name").is_in(DECISION))
            g_frame = load_arm(label=label, arm=g_arm).filter(pl.col("site_name").is_in(DECISION))
            rows.append(
                {
                    "year": label,
                    "weeks of history": weeks,
                    "N": n_frame["qs"].mean(),
                    "G": g_frame["qs"].mean(),
                    "N minus G": n_frame["qs"].mean() - g_frame["qs"].mean(),
                }
            )
    return pl.DataFrame(rows)


def decision(*, gate_passed: bool) -> dict[str, object]:
    """Apply the plan's decision rule to the planned contrasts.

    Args:
        gate_passed: Whether the known-answer gate counts as passed. When it does not, the verdict
            is capped at `inconclusive`, and the verdict the numbers alone give is named beside it.

    Returns:
        Each rule's inputs and the resulting verdict.
    """
    table = contrast_table(planned=True)

    def row(prefix: str) -> dict[str, object]:
        return table.filter(pl.col("contrast").str.starts_with(prefix)).row(0, named=True)

    c1, c2, c3 = row("C1"), row("C2"), row("C3")

    def supported(entry: dict[str, object]) -> bool:
        return bool(
            entry["scored QS"] < 0.0  # ty: ignore[unsupported-operator]
            and entry["scored sites better"] >= 4  # ty: ignore[unsupported-operator]
            and entry["replicate QS"] < 0.0  # ty: ignore[unsupported-operator]
        )

    def below_zero(entry: dict[str, object]) -> bool:
        return bool(entry["scored QS"] < 0.0)  # ty: ignore[unsupported-operator]

    c1_passes = bool(
        c1["scored high"] <= C1_MARGIN  # ty: ignore[unsupported-operator]
        and c1["replicate QS"] <= C1_MARGIN  # ty: ignore[unsupported-operator]
    )
    c1_refuted = bool(c1["scored low"] > C1_MARGIN)  # ty: ignore[unsupported-operator]
    both_above = bool(
        c2["scored low"] > 0.0  # ty: ignore[unsupported-operator]
        and c3["scored low"] > 0.0  # ty: ignore[unsupported-operator]
    )
    c2_or_c3_supported = supported(c2) or supported(c3)
    if c1_refuted or both_above:
        verdict = "refuted"
    elif c2_or_c3_supported and c1_passes:
        verdict = "supported"
    elif not c1_refuted and (below_zero(c2) or below_zero(c3)):
        verdict = "partly supported"
    else:
        verdict = "inconclusive"
    if not gate_passed:
        verdict = f"inconclusive (capped by the failed gate; the numbers alone give {verdict})"
    return {
        "C1 passes": c1_passes,
        "C1 refuted": c1_refuted,
        "C2 supported": supported(c2),
        "C3 supported": supported(c3),
        "gate counted as passed": gate_passed,
        "verdict": verdict,
    }


def capacity_sources() -> str:
    """Return how the fits of N and N-4w set each site's capacity, one line per cut and site."""
    path = common.WORK_DIR / "capacity_sources.json"
    if not path.exists():
        return "none\n"
    stored = json.loads(path.read_text())
    lines = ["cut,arm,target,site,source"]
    for name in sorted(stored):
        cut, _, arm = name.partition("_")
        if arm not in ("n", "n_4w", "n_3m"):
            continue
        for target, by_site in sorted(stored[name].items()):
            if arm == "n":
                lines.extend(f"{cut},{arm},-,{site},{source}" for site, source in by_site.items())
                break
            lines.append(f"{cut},{arm},{target},{target},{by_site[target]}")
    return "\n".join(lines) + "\n"


def main() -> None:
    """Print every table of the report as comma-separated values."""
    print(f"== C4 bar (better of x_2y and x_2yamq on the tuning year): {x_bar()} ==")
    print("== Absolute quantile score by arm, points of capacity (rows outside the masks) ==")
    print(absolute_table().write_csv(float_precision=3), end="")
    print(f"== Planned contrasts ({PLANNED_LEVEL:.2f}% intervals), negative = treatment better ==")
    print(contrast_table(planned=True).write_csv(float_precision=3), end="")
    print("== Decision rule ==")
    print("The gate failed under the pre-registered definition (runs 1 to 3); a post hoc")
    print("redefinition passes in run 4. The plan counts the gate as failed, so the verdict")
    print("is capped at inconclusive. Beside it, the verdict if the post hoc gate were accepted:")
    print(json.dumps(decision(gate_passed=False), indent=2))
    print(json.dumps(decision(gate_passed=True), indent=2))
    print("== Exploratory contrasts (95% intervals) ==")
    print(contrast_table(planned=False).write_csv(float_precision=3), end="")
    print("== Onboarding curve ==")
    print(onboarding_curve().write_csv(float_precision=3), end="")
    print("== Planned contrasts with each decision site left out (quantile score) ==")
    print(leave_one_site_out().write_csv(float_precision=3), end="")
    print("== Median forecast minus mean forecast, points ==")
    print(median_minus_mean().write_csv(float_precision=3), end="")
    print("== Tail calibration on the five decision sites ==")
    print(tail_table().write_csv(float_precision=4), end="")
    print("== How each fit set its capacities ==")
    print(capacity_sources(), end="")


if __name__ == "__main__":
    main()
