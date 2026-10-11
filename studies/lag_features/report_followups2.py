"""Print every number the round-2 post hoc analyses quote, and save the tables the charts read.

Part of the study in <https://github.com/openclimatefix/nged-substation-forecast/issues/1138>. The
second science review of the draft page accepted a list of reanalyses and four small fits. This
script reads the first run's losses, round 1's losses and (when they exist) round 2's fit losses,
writes `<output-root>/ens_mean/followups2/report_followups2_ens_mean.md` and
`tables_followups2_ens_mean/`, and never writes into an earlier run's files. **Every number is post
hoc**, computed after the first run's results were known, and the report says so.

**The no-fit sections read only saved files and need no GPU:**

1. Positive-control shares corrected for each arm's standing gain.
2. Hindsight monthly rescaling of B0, on the 8 unselected months and on all 18.
3. The one plant-month that dominates, and B0's monthly bias at every plant.
4. The screening months' paired contrasts, with and without that plant-month.
5. Equivalence of each planned contrast against the decision threshold.
6. Climatology from earlier months only, and the look-ahead of the saved climatology.
7. The fingerprint figure's missing rows.
8. AN as coded, the two reports' L1 interval, and the quantile hit rates by month.

**The fit sections** (round 2's four fit blocks) read `losses_followups2_ens_mean.parquet`, and
print that the fits have not run while it does not exist. A report written before the fits is named
`report_followups2_ens_mean_nofit.md`, with `tables_followups2_ens_mean_nofit/`.

Run it with `uv run python studies/lag_features/report_followups2.py`.
"""

import argparse
import logging
import sys
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Final

import lag_arm_columns
import numpy as np
import polars as pl
from build_lag_frame import (
    FULL_SWEEP_LEAD_DAY,
    IFS_LEAD_DAYS,
    LAG_FEATURES_DIR,
    TARGET,
    lag_source_hourly,
    output_paths,
    run_suffix,
    scored_months,
    write_parquet_atomic,
)
from fit_followups import verify_followup_manifest
from fit_followups2 import verify_followup2_manifest
from fit_lag_arms import METRIC, SCREENING_MONTHS, verify_manifest_inputs
from followup2_frames import followup2_dir
from followup_frames import LONG_LEADS, PRODUCT, frame_path
from report_followups import hindsight_losses
from report_lag_features import (
    EXPLORATORY_LEVEL,
    P2_FIRST_MONTH,
    PERCENTAGE_POINTS,
    PLANNED_LEVEL,
    paired_interval,
)
from studies.baselines import climatology
from studies.bootstrap import BOOTSTRAP_SEED, N_BOOTSTRAP_RESAMPLES
from studies.cross_validation import score_prediction
from studies.guards import refuse_to_overwrite

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
_LOG: Final[logging.Logger] = logging.getLogger("report_followups2")

POST_HOC: Final[str] = "Post hoc: written after the first run's results were known."
"""The line under every section heading."""

BORDERLINE_PP: Final[float] = 0.01
"""A 99% bound this close to zero (percentage points), in either bootstrap stream, is called
borderline: the call depends on the random draws."""

ROLE_EIGHT: Final[str] = "the largest-error plant-month of the 8 unselected months"
ROLE_WINDOW: Final[str] = "the largest-error plant-month of the window"
"""How the plant-months removed are named in published text, with no plant letter or month."""

REVIEWED_ARMS: Final[tuple[str, ...]] = ("PC", "W7", "TF", "AN", "L1", "Q30")
"""The arms the plant-month analysis tabulates."""

CONTROL_KINDS: Final[tuple[str, ...]] = ("month", "step")
"""The two positive controls of round 1."""

SHIFT_PERCENTS: Final[tuple[str, ...]] = ("05", "10")
"""The shifts of round 1's controls, as the scope names carry them."""

ORACLE_LEVEL: Final[float] = PLANNED_LEVEL
"""The coverage at which the oracle's own interval must exclude zero for a share to be reported."""

BLEND_SHARE: Final[float] = 0.5
"""The share of the climatology in the 50/50 blends."""

CLIMATOLOGY_LEADS: Final[tuple[int, ...]] = (FULL_SWEEP_LEAD_DAY, *LONG_LEADS)
"""The lead-days the climatology reanalysis scores: 1, 7, 10, and 14."""

FIT_SCOPES: Final[tuple[str, ...]] = (
    "fu2_control_step_pc_s05",
    "fu2_control_step_pc_s10",
    "fu2_lead1",
    "fu2_ifs_lead1",
    "fu2_ifs_lead2",
)
"""The scopes round 2's fits write."""


# --- Helpers -------------------------------------------------------------------------------------


class Aligned:
    """Per-seed, per-row arrays of several named series on the same rows.

    Attributes:
        site: Each row's plant label.
        month: Each row's month label.
        values: Each series' values, shape (n_seeds, n_rows).
    """

    def __init__(self, *, frame: pl.DataFrame, names: list[str]) -> None:
        """Align a wide frame (one column per series) into arrays, dropping rows with a null.

        Args:
            frame: `site`, `time`, `seed`, `month` and one column per name.
            names: The series' column names.

        Raises:
            ValueError: If the seeds do not all hold the same rows after the nulls are dropped.
        """
        complete = frame.drop_nulls(names).sort("seed", "site", "time")
        seeds = sorted(complete["seed"].unique().to_list())
        per_seed = [complete.filter(pl.col("seed") == seed) for seed in seeds]
        keys = per_seed[0].select("site", "time")
        if not all(part.select("site", "time").equals(keys) for part in per_seed[1:]):
            msg = "the seeds do not all hold the same rows"
            raise ValueError(msg)
        self.site: np.ndarray = per_seed[0]["site"].to_numpy()
        self.month: np.ndarray = per_seed[0]["month"].to_numpy()
        self.values: dict[str, np.ndarray] = {
            name: np.stack([part[name].to_numpy() for part in per_seed]) for name in names
        }

    @property
    def n_rows(self) -> int:
        """The number of rows each seed holds."""
        return len(self.site)

    def where(self, *, keep: np.ndarray) -> Aligned:
        """Return the rows a boolean mask keeps.

        Args:
            keep: One boolean per row.

        Returns:
            A new aligned set of the kept rows.
        """
        new = Aligned.__new__(Aligned)
        new.site = self.site[keep]
        new.month = self.month[keep]
        new.values = {name: array[:, keep] for name, array in self.values.items()}
        return new

    def in_months(self, *, months: tuple[str, ...] | list[str]) -> Aligned:
        """Return the rows of some months.

        Args:
            months: The month labels to keep.

        Returns:
            A new aligned set.
        """
        return self.where(keep=np.isin(self.month, list(months)))


def wide(*, losses: pl.DataFrame, names: Mapping[str, str], metric: str = METRIC) -> pl.DataFrame:
    """Pivot per-row losses to one column per arm.

    Args:
        losses: Losses carrying `site`, `time`, `seed`, `month`, `arm` and the metric.
        names: The column name to give each arm kept.
        metric: The loss column.

    Returns:
        `site`, `time`, `seed`, `month` and one column per name, with a row where every arm has one.
    """
    kept = losses.filter(pl.col("arm").is_in(list(names)))
    pivoted = kept.pivot(on="arm", index=["site", "time", "seed", "month"], values=metric)
    return pivoted.rename({arm: name for arm, name in names.items() if arm in pivoted.columns})


def month_bootstrap(
    *,
    data: Aligned,
    statistic: Callable[[dict[str, np.ndarray]], float],
    levels: tuple[float, ...] = (EXPLORATORY_LEVEL,),
    stream: int = 0,
) -> dict[str, float]:
    """Bound a statistic by resampling whole months and a seed, as `studies.bootstrap` does.

    Args:
        data: The aligned series.
        statistic: A function of the resampled series (each shape (rows,)) returning a number.
        levels: The coverages, in percent.
        stream: 0 for the study's bootstrap stream, or another number for a second, independent
            stream, which shows how far a bound near zero depends on the random draws.

    Returns:
        `value` (on every row and seed), and `lower_<level>` and `upper_<level>` for each level.
    """
    months = np.unique(data.month, return_inverse=True)[1]
    rows_by_month = [np.flatnonzero(months == index) for index in range(months.max() + 1)]
    generator = np.random.default_rng(BOOTSTRAP_SEED + stream)
    n_seeds = next(iter(data.values.values())).shape[0]
    resampled = np.empty(N_BOOTSTRAP_RESAMPLES)
    for resample in range(N_BOOTSTRAP_RESAMPLES):
        seed_index = generator.integers(0, n_seeds)
        drawn = generator.integers(0, len(rows_by_month), size=len(rows_by_month))
        rows = np.concatenate([rows_by_month[index] for index in drawn])
        resampled[resample] = statistic({k: v[seed_index, rows] for k, v in data.values.items()})
    result = {"value": float(statistic({k: v.ravel() for k, v in data.values.items()}))}
    for level in levels:
        tail = (100.0 - level) / 2.0
        result[f"lower_{level:g}"] = float(np.nanpercentile(resampled, tail))
        result[f"upper_{level:g}"] = float(np.nanpercentile(resampled, 100.0 - tail))
    return result


def mean_difference(
    *,
    data: Aligned,
    treatment: str,
    reference: str,
    levels: tuple[float, ...] = (95.0,),
    stream: int = 0,
) -> dict[str, float]:
    """Bound the mean of a paired difference between two series.

    Args:
        data: The aligned series.
        treatment: The series compared.
        reference: The series it is compared with.
        levels: The coverages, in percent.
        stream: The bootstrap stream, as `month_bootstrap` takes it.

    Returns:
        `month_bootstrap`'s result for the mean of treatment minus reference.
    """
    return month_bootstrap(
        data=data,
        statistic=lambda d: float((d[treatment] - d[reference]).mean()),
        levels=levels,
        stream=stream,
    )


def _mean(frame: pl.DataFrame) -> float:
    """Return the mean of the loss metric over a frame's rows."""
    return float(frame.select(pl.col(METRIC).mean()).item())


def _pp(value: float | None, *, signed: bool = True) -> str:
    """Format a fraction of capacity as percentage points, or blank for `None`."""
    if value is None or np.isnan(value):
        return ""
    return f"{value * PERCENTAGE_POINTS:{'+' if signed else ''}.3f}"


def _interval(low: float | None, high: float | None) -> str:
    """Format an interval of fractions of capacity as percentage points."""
    return "" if low is None or high is None else f"[{_pp(low)}, {_pp(high)}]"


def _table(*, header: list[str], rows: list[list[str]]) -> list[str]:
    """Format a markdown table."""
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return [*lines, ""]


def read_losses(*, path: Path, scopes: list[str], setting: str = "primary") -> pl.DataFrame:
    """Read the columns the reanalyses use, for some scopes of one setting.

    Args:
        path: A losses parquet file.
        scopes: The scopes to keep.
        setting: The hyperparameter setting.

    Returns:
        The keys, the arm, the scope, the capped absolute error, the signed error and the capacity.
    """
    return (
        pl.scan_parquet(path)
        .filter(pl.col("scope").is_in(scopes) & (pl.col("setting") == setting))
        .select(
            "site",
            "time",
            "month",
            "seed",
            "arm",
            "scope",
            "actual",
            "prediction",
            "effective_capacity_mw",
            "signed_error_capped_mw",
            METRIC,
        )
        .collect()
    )


# --- 1. Corrected positive-control shares --------------------------------------------------------


def corrected_shares_section(
    *, first_path: Path, follow_path: Path, fit_path: Path
) -> tuple[list[str], pl.DataFrame]:
    """Tabulate each arm's recovery of a planted loss, net of its standing gain over B0.

    The share is ((arm minus B0) on the shifted data minus (arm minus B0) on the unshifted data,
    paired by row and seed) divided by (oracle minus B0) on the shifted data. The oracle's own
    standing difference on the unshifted data is zero by construction. The share is blank where the
    oracle's own 99% interval includes zero. Round 1's raw share is printed beside it.

    Args:
        first_path: The first run's losses.
        follow_path: Round 1's losses.
        fit_path: Round 2's losses, which add PC to the step control when the file exists.

    Returns:
        The report lines and a table with one row per (kind, shift, arm).
    """
    scopes = [
        f"fu_control_{kind}_s{percent}" for kind in CONTROL_KINDS for percent in SHIFT_PERCENTS
    ]
    follow = read_losses(path=follow_path, scopes=scopes)
    unshifted_all = read_losses(path=first_path, scopes=[f"lead{FULL_SWEEP_LEAD_DAY}"])
    extra = (
        read_losses(
            path=fit_path,
            scopes=[f"fu2_control_step_pc_s{percent}" for percent in SHIFT_PERCENTS],
        )
        if fit_path.exists()
        else pl.DataFrame()
    )
    records = []
    for kind in CONTROL_KINDS:
        for percent in SHIFT_PERCENTS:
            scoped = follow.filter(pl.col("scope") == f"fu_control_{kind}_s{percent}")
            if kind == "step" and not extra.is_empty():
                scoped = pl.concat(
                    [
                        scoped,
                        extra.filter(
                            (pl.col("scope") == f"fu2_control_step_pc_s{percent}")
                            & (pl.col("arm") == "PC")
                        ),
                    ]
                )
            arms = [a for a in scoped["arm"].unique().sort().to_list() if a not in {"B0", "O"}]
            shifted = wide(losses=scoped, names={a: f"{a}_s" for a in ("B0", "O", *arms)})
            unshifted = wide(losses=unshifted_all, names={a: f"{a}_u" for a in ("B0", *arms)}).drop(
                "month"
            )
            joined = shifted.join(unshifted, on=["site", "time", "seed"], how="inner")
            data = Aligned(
                frame=joined, names=[c for c in joined.columns if c[-2:] in {"_s", "_u"}]
            )
            oracle = month_bootstrap(
                data=data,
                statistic=lambda d: float((d["O_s"] - d["B0_s"]).mean()),
                levels=(ORACLE_LEVEL,),
            )
            oracle_found = oracle[f"upper_{ORACLE_LEVEL:g}"] < 0.0
            for arm in arms:
                share = month_bootstrap(
                    data=data,
                    statistic=lambda d, arm=arm: float(
                        ((d[f"{arm}_s"] - d["B0_s"]) - (d[f"{arm}_u"] - d["B0_u"])).mean()
                        / (d["O_s"] - d["B0_s"]).mean()
                    ),
                )
                raw = float(
                    (data.values[f"{arm}_s"] - data.values["B0_s"]).mean()
                    / (data.values["O_s"] - data.values["B0_s"]).mean()
                )
                standing = float((data.values[f"{arm}_u"] - data.values["B0_u"]).mean())
                records.append(
                    {
                        "kind": kind,
                        "shift": int(percent) / 100,
                        "arm": arm,
                        "share": share["value"] if oracle_found else None,
                        "lower": share["lower_95"] if oracle_found else None,
                        "upper": share["upper_95"] if oracle_found else None,
                        "raw_share": raw,
                        "standing_difference": standing,
                        "oracle_difference": oracle["value"],
                        "oracle_lower": oracle[f"lower_{ORACLE_LEVEL:g}"],
                        "oracle_upper": oracle[f"upper_{ORACLE_LEVEL:g}"],
                        "oracle_found": oracle_found,
                        "n_rows": data.n_rows,
                    }
                )
    table = pl.DataFrame(records)
    lines = [
        (
            "Post hoc. share = ((arm minus B0 on the shifted data) minus (arm minus B0 on the "
            "unshifted data), paired by row and seed) divided by (oracle minus B0 on the shifted "
            "data). The oracle's own standing difference on the unshifted data is zero by "
            "construction. The interval is 95%, from resampling whole months and a seed. The "
            "share is blank where the oracle's own 99% interval includes zero. `raw` is round 1's "
            "share, which counts the arm's standing gain as recovery; `standing` is that gain "
            "(arm minus B0 on the unshifted data, pp)."
        ),
        "",
    ]
    rows = [
        [
            str(r["kind"]),
            f"{r['shift']:.0%}",
            str(r["arm"]),
            "" if r["share"] is None else f"{r['share']:.2f}",
            "" if r["lower"] is None else f"[{r['lower']:.2f}, {r['upper']:.2f}]",
            f"{r['raw_share']:.2f}",
            _pp(r["standing_difference"]),
            f"{_pp(r['oracle_difference'])} {_interval(r['oracle_lower'], r['oracle_upper'])}",
            str(r["n_rows"]),
        ]
        for r in table.iter_rows(named=True)
    ]
    lines += _table(
        header=[
            "Control",
            "Shift",
            "Arm",
            "Corrected share",
            "95% interval",
            "Raw share (round 1)",
            "Standing (pp)",
            "Oracle minus B0 (pp) [99%]",
            "Rows",
        ],
        rows=rows,
    )
    if extra.is_empty():
        lines += ["PC in the plant-specific step control has not been fitted yet.", ""]
    return lines, table


# --- 2. Hindsight scaling ------------------------------------------------------------------------


def hindsight_section(*, first_path: Path, root: Path) -> tuple[list[str], pl.DataFrame]:
    """Tabulate B0 after a perfect monthly rescaling, on the 8 unselected months and on all 18.

    Args:
        first_path: The first run's losses.
        root: The output root, for the lead-day 1 frame.

    Returns:
        The report lines and a table with one row per (window, scaling).
    """
    losses = read_losses(path=first_path, scopes=["lead1"]).filter(pl.col("arm") == "B0")
    frame = pl.read_parquet(
        output_paths(root=root, product=PRODUCT, lead_days=(FULL_SWEEP_LEAD_DAY,))["day1"]
    )
    both = hindsight_losses(losses=losses, frame=frame)
    windows = {
        "8 unselected months": [m for m in scored_months() if m >= P2_FIRST_MONTH],
        "all 18 months": scored_months(),
    }
    records = []
    for window, months in windows.items():
        scoped = both.filter(pl.col("month").is_in(months))
        for scaling in sorted(scoped["arm"].unique().to_list()):
            if scaling == "B0":
                continue
            interval = paired_interval(
                scoped=scoped,
                treatment=scaling,
                reference="B0",
                metric=METRIC,
                level=EXPLORATORY_LEVEL,
            )
            if interval is not None:
                records.append(
                    {
                        "window": window,
                        "scaling": scaling,
                        "difference": interval["difference"],
                        "lower": interval["lower_95"],
                        "upper": interval["upper_95"],
                        "n_months": interval["n_months"],
                        "n_rows": interval["n_rows"],
                    }
                )
    table = pl.DataFrame(records)
    lines = [
        (
            "Post hoc. B0's export-capped forecast after multiplying each plant-month's "
            "predictions by a scale fitted on the same rows. **This is the gain from a perfect "
            "monthly scale, not a ceiling for finer corrections.** No forecast could know the "
            "scale in advance. Intervals are 95%, from resampling whole months and a seed."
        ),
        "",
        *_table(
            header=["Window", "Scaling", "Difference to B0 (pp)", "95% interval (pp)", "Months"],
            rows=[
                [
                    r["window"],
                    r["scaling"],
                    _pp(r["difference"]),
                    _interval(r["lower"], r["upper"]),
                    str(r["n_months"]),
                ]
                for r in table.iter_rows(named=True)
            ],
        ),
    ]
    return lines, table


# --- 3 and 4. The one plant-month, and the screening contrasts -------------------------------


def dominant_plant_month(*, b0: pl.DataFrame, months: list[str]) -> tuple[str, str, float, float]:
    """Pick the plant-month with the largest B0 error among some months.

    Args:
        b0: B0's lead-day 1 losses.
        months: The months of the window.

    Returns:
        The plant, the month, its mean B0 error and the next-largest plant-month's.
    """
    ranked = (
        b0.filter(pl.col("month").is_in(months))
        .group_by("site", "month")
        .agg(error=pl.col(METRIC).mean())
        .sort("error", "site", "month", descending=[True, False, False])
    )
    first, second = ranked.row(0, named=True), ranked.row(1, named=True)
    return first["site"], first["month"], first["error"], second["error"]


def plant_month_section(
    *, lead1: pl.DataFrame, aligned: Aligned
) -> tuple[list[str], dict[str, pl.DataFrame]]:
    """Tabulate each reviewed arm's gain with and without the dominant plant-month.

    Args:
        lead1: The first run's lead-day 1 losses, with B0 and the reviewed arms.
        aligned: The reviewed arms and B0 on the same rows, as `Aligned`.

    Returns:
        The report lines and the tables: the gains, the plant-month differences, and B0's bias.
    """
    b0 = lead1.filter(pl.col("arm") == "B0")
    windows = {
        "8 unselected months": [m for m in scored_months() if m >= P2_FIRST_MONTH],
        "all 18 months": scored_months(),
    }
    gains, cells = [], []
    eight = windows["8 unselected months"]
    eight_site, eight_month, eight_error, eight_next = dominant_plant_month(b0=b0, months=eight)
    lines = [
        (
            "Post hoc. The rule is applied once: the plant-month with the largest mean B0 error "
            "among the 8 unselected months, chosen without reference to any arm. It is removed in "
            "both windows, and the 18-month window also removes its own largest-error "
            "plant-month, alone and together with the first. Intervals are 95% and 99%, from "
            "resampling whole months and a seed. A 99% bound within "
            f"{BORDERLINE_PP:g} points of zero in either of two independent bootstrap streams is "
            "marked borderline at 99%, and the row's call is then neither significant nor not."
        ),
        "",
    ]
    audit = [
        (
            f"- {ROLE_EIGHT}: plant {eight_site}, {eight_month}, B0 error "
            f"{eight_error * PERCENTAGE_POINTS:.2f}% of capacity (the next largest is "
            f"{eight_next * PERCENTAGE_POINTS:.2f}%)."
        )
    ]
    for window, months in windows.items():
        own_site, own_month, own_error, own_next = dominant_plant_month(b0=b0, months=months)
        scoped = aligned.in_months(months=months)
        is_eight = (scoped.site == eight_site) & (scoped.month == eight_month)
        is_own = (scoped.site == own_site) & (scoped.month == own_month)
        variants = [("all rows", scoped)]
        if (own_site, own_month) == (eight_site, eight_month):
            variants.append(
                (f"without {ROLE_EIGHT} (also the window's own)", scoped.where(keep=~is_eight))
            )
        else:
            audit.append(
                f"- {ROLE_WINDOW} ({window}): plant {own_site}, {own_month}, B0 error "
                f"{own_error * PERCENTAGE_POINTS:.2f}% of capacity (the next largest is "
                f"{own_next * PERCENTAGE_POINTS:.2f}%)."
            )
            variants += [
                (f"without {ROLE_EIGHT}", scoped.where(keep=~is_eight)),
                (f"without {ROLE_WINDOW}", scoped.where(keep=~is_own)),
                ("without both", scoped.where(keep=~(is_eight | is_own))),
            ]
        for arm in REVIEWED_ARMS:
            for label, data in variants:
                interval = mean_difference(
                    data=data, treatment=arm, reference="B0", levels=(95.0, 99.0)
                )
                other = mean_difference(
                    data=data, treatment=arm, reference="B0", levels=(99.0,), stream=1
                )
                bounds = [
                    interval["lower_99"],
                    interval["upper_99"],
                    other["lower_99"],
                    other["upper_99"],
                ]
                borderline = any(abs(b) * PERCENTAGE_POINTS < BORDERLINE_PP for b in bounds)
                gains.append(
                    {
                        "window": window,
                        "arm": arm,
                        "rows": label,
                        "difference": interval["value"],
                        "lower_95": interval["lower_95"],
                        "upper_95": interval["upper_95"],
                        "lower_99": interval["lower_99"],
                        "upper_99": interval["upper_99"],
                        "lower_99_stream2": other["lower_99"],
                        "upper_99_stream2": other["upper_99"],
                        "borderline_99": borderline,
                        "n_rows": data.n_rows,
                    }
                )
    gain_table = pl.DataFrame(gains)
    lines += _table(
        header=[
            "Window",
            "Arm",
            "Rows",
            "Difference to B0 (pp)",
            "95% interval (pp)",
            "99% interval (pp)",
            "99% interval, second stream (pp)",
            "99% call",
            "Rows per seed",
        ],
        rows=[
            [
                r["window"],
                r["arm"],
                r["rows"],
                _pp(r["difference"]),
                _interval(r["lower_95"], r["upper_95"]),
                _interval(r["lower_99"], r["upper_99"]),
                _interval(r["lower_99_stream2"], r["upper_99_stream2"]),
                "borderline at 99%" if r["borderline_99"] else "",
                str(r["n_rows"]),
            ]
            for r in gain_table.iter_rows(named=True)
        ],
    )
    lines += [
        "Audit, not for publication (the plant-months removed, by role):",
        "",
        *audit,
        "",
    ]
    # Per plant-month paired differences.
    scoped8 = aligned.in_months(months=windows["8 unselected months"])
    for arm in REVIEWED_ARMS:
        difference = (scoped8.values[arm] - scoped8.values["B0"]).mean(axis=0)
        frame = pl.DataFrame(
            {"site": scoped8.site, "month": scoped8.month, "difference": difference}
        )
        grouped = frame.group_by("site", "month").agg(
            difference=pl.col("difference").mean(), n_rows=pl.len()
        )
        cells += [{"arm": arm, **row} for row in grouped.iter_rows(named=True)]
    cell_table = pl.DataFrame(cells).sort("arm", "site", "month")
    pivot = cell_table.pivot(on="arm", index=["site", "month"], values="difference").sort(
        "month", "site"
    )
    lines += [
        "### Per plant-month paired difference to B0 on the 8 unselected months (pp)",
        "",
        *_table(
            header=["Plant", "Month", *REVIEWED_ARMS],
            rows=[
                [r["site"], r["month"], *(_pp(r[arm]) for arm in REVIEWED_ARMS)]
                for r in pivot.iter_rows(named=True)
            ],
        ),
    ]
    # B0's monthly bias and error at every plant, over the whole record.
    bias = (
        b0.filter(pl.col("month").is_in(scored_months()))
        .group_by("site", "month")
        .agg(
            bias=(pl.col("signed_error_capped_mw") / pl.col("effective_capacity_mw")).mean(),
            error=pl.col(METRIC).mean(),
        )
        .sort("month", "site")
    )
    sites = sorted(bias["site"].unique().to_list())
    for column, title in (
        (
            "bias",
            "B0's mean signed error by plant and month (pp of capacity; positive is over-forecast)",
        ),
        ("error", "B0's mean absolute error by plant and month (pp of capacity)"),
    ):
        matrix = bias.pivot(on="site", index="month", values=column).sort("month")
        lines += [
            f"### {title}",
            "",
            *_table(
                header=["Month", *sites],
                rows=[
                    [r["month"], *(_pp(r[s], signed=column == "bias") for s in sites)]
                    for r in matrix.iter_rows(named=True)
                ],
            ),
        ]
    return lines, {
        "plant_month_gains": gain_table,
        "plant_month_cells": cell_table,
        "b0_bias": bias,
    }


def screening_section(*, lead1: pl.DataFrame) -> tuple[list[str], pl.DataFrame]:
    """Tabulate every fitted arm minus B0 on the screening months, with and without a plant-month.

    Args:
        lead1: The first run's lead-day 1 losses.

    Returns:
        The report lines and a table with one row per (arm, rows).
    """
    months = list(SCREENING_MONTHS)
    scoped = lead1.filter(pl.col("month").is_in(months))
    arms = [a for a in scoped["arm"].unique().sort().to_list() if a != "B0"]
    data = Aligned(
        frame=wide(losses=scoped, names={a: a for a in ("B0", *arms)}), names=["B0", *arms]
    )
    site, month, error, runner_up = dominant_plant_month(
        b0=scoped.filter(pl.col("arm") == "B0"), months=months
    )
    without = data.where(keep=~((data.site == site) & (data.month == month)))
    records = []
    for arm in arms:
        for label, subset in (("all rows", data), (f"without {ROLE_WINDOW}", without)):
            interval = mean_difference(data=subset, treatment=arm, reference="B0")
            records.append(
                {
                    "arm": arm,
                    "rows": label,
                    "difference": interval["value"],
                    "lower": interval["lower_95"],
                    "upper": interval["upper_95"],
                    "n_rows": subset.n_rows,
                }
            )
    table = pl.DataFrame(records)
    lines = [
        (
            "Post hoc. The paired version of the first report's screen: each fitted arm minus B0 "
            "on the 10 screening months, at lead-day 1 and the primary setting, with a 95% "
            "interval from resampling whole months and a seed. The second set of rows removes "
            f"{ROLE_WINDOW} (the one with the largest mean B0 error in the screening months). "
            "The no-fit arms are not in the saved losses. Audit, not for publication: the "
            f"plant-month removed is plant {site}, {month}, at "
            f"{error * PERCENTAGE_POINTS:.2f}% of capacity (the next largest is "
            f"{runner_up * PERCENTAGE_POINTS:.2f}%)."
        ),
        "",
        *_table(
            header=["Arm", "Rows", "Difference to B0 (pp)", "95% interval (pp)", "Rows per seed"],
            rows=[
                [
                    r["arm"],
                    r["rows"],
                    _pp(r["difference"]),
                    _interval(r["lower"], r["upper"]),
                    str(r["n_rows"]),
                ]
                for r in table.sort("rows", "difference").iter_rows(named=True)
            ],
        ),
    ]
    return lines, table


# --- 5. Equivalence against the decision threshold -----------------------------------------------


def equivalence_section(*, root: Path, suffix: str) -> tuple[list[str], pl.DataFrame]:
    """Test each planned contrast's 99% interval against the decision threshold.

    Args:
        root: The output root.
        suffix: The first run's run suffix.

    Returns:
        The report lines and a table with one row per (contrast, setting).
    """
    planned = pl.read_parquet(
        root / PRODUCT / f"tables_{PRODUCT}{suffix}" / "planned.parquet"
    ).with_columns(
        threshold=0.02 * pl.col("reference_value"),
        metric=pl.when(pl.col("contrast") == "P4")
        .then(pl.lit("continuous ranked probability score"))
        .otherwise(pl.lit("mean absolute error")),
    )
    table = planned.with_columns(
        excludes_threshold_gain=pl.col("lower") > -pl.col("threshold"),
        excludes_threshold_loss=pl.col("upper") < pl.col("threshold"),
    ).sort("contrast", "setting")
    lines = [
        (
            "Post hoc. The decision threshold is 2% of the reference arm's own metric. A "
            "contrast excludes a gain that large when the lower end of its 99% interval is above "
            "minus the threshold; it excludes a loss that large when the upper end is below the "
            "threshold. Both reports' planned contrasts are the first run's."
        ),
        "",
        *_table(
            header=[
                "Contrast",
                "Metric",
                "Setting",
                "Reference value (pp)",
                "Threshold, 2% (pp)",
                "Difference (pp)",
                "99% interval (pp)",
                "Excludes a gain that large",
                "Excludes a loss that large",
            ],
            rows=[
                [
                    f"{r['contrast']}: {r['treatment']} minus {r['reference']}",
                    r["metric"],
                    r["setting"],
                    _pp(r["reference_value"], signed=False),
                    _pp(r["threshold"], signed=False),
                    _pp(r["difference"]),
                    _interval(r["lower"], r["upper"]),
                    "yes" if r["excludes_threshold_gain"] else "no",
                    "yes" if r["excludes_threshold_loss"] else "no",
                ]
                for r in table.iter_rows(named=True)
            ],
        ),
    ]
    return lines, table


# --- 6. Climatology from earlier months only ---------------------------------------------------


def past_only_climatology(*, rows: pl.DataFrame, hourly: pl.DataFrame) -> pl.DataFrame:
    """Return each row's median power in the same calendar month and hour of earlier years only.

    A row's calendar month and hour are its `time`'s. The history is the plant's cleaned hourly
    power at the same calendar month and hour in years strictly before the row's year, so no
    month at or after the row's own enters.

    Args:
        rows: `site` and `time`.
        hourly: The cleaned hourly power, with `site`, `time` and `power_mw`.

    Returns:
        `site`, `time` and `past_climatology`, null where no earlier year holds the month and hour.
    """
    keys = {
        "calendar_month": pl.col("time").dt.month(),
        "hour": pl.col("time").dt.hour(),
        "year": pl.col("time").dt.year(),
    }
    history = hourly.with_columns(**keys)
    targets = rows.select("site", "time").with_columns(**keys)
    medians = [
        history.filter(pl.col("year") < year)
        .group_by("site", "calendar_month", "hour")
        .agg(past_climatology=pl.col("power_mw").median())
        .with_columns(year=pl.lit(year, dtype=pl.Int32))
        for year in sorted(targets["year"].unique().to_list())
    ]
    return (
        targets.join(
            pl.concat(medians),
            on=["site", "calendar_month", "hour", "year"],
            how="left",
            maintain_order="left",
        )
        .select("site", "time", "past_climatology")
        .with_columns(pl.col("past_climatology").cast(pl.Float64))
    )


def climatology_section(
    *,
    first_path: Path,
    root: Path,
    hourly: pl.DataFrame,
    keep: dict[int, pl.DataFrame] | None = None,
) -> tuple[list[str], pl.DataFrame]:
    """Score B0, W7 and their 50/50 blends with climatology, past-only and as saved.

    Args:
        first_path: The first run's losses.
        root: The output root.
        hourly: The cleaned hourly power.
        keep: If given, filled with each lead-day's per-row scored losses of every arm scored, for
            a later report that pairs two of them.

    Returns:
        The report lines and a table with one row per (lead-day, arm).
    """
    records = []
    count_rows = []
    for lead in CLIMATOLOGY_LEADS:
        path = (
            output_paths(root=root, product=PRODUCT, lead_days=(lead,))[f"day{lead}"]
            if lead == FULL_SWEEP_LEAD_DAY
            else frame_path(root=root, name=f"lead{lead}")
        )
        frame = pl.read_parquet(path).sort("site", "time")
        saved = read_losses(path=first_path, scopes=[f"lead{lead}"]).filter(
            pl.col("arm").is_in(["B0", "W7"])
        )
        past = past_only_climatology(rows=frame, hourly=hourly)
        if lead == FULL_SWEEP_LEAD_DAY:
            missing = frame.select("site", "time").join(
                past.filter(pl.col("past_climatology").is_null()).select("site", "time"),
                on=["site", "time"],
            )
        existing = pl.DataFrame({"existing": climatology(frame=frame)}).with_columns(
            frame.select("site", "time")
        )
        rows = (
            frame.select(
                "site",
                "time",
                "month",
                "fold",
                "constrained",
                "effective_capacity_mw",
                "cap_mw",
                TARGET,
            )
            .join(past, on=["site", "time"])
            .join(existing, on=["site", "time"])
        )
        usable = rows.drop_nulls(["past_climatology", "existing"])
        count_rows.append(
            {
                "lead": lead,
                "rows": rows.height,
                "with_past": rows["past_climatology"].is_not_null().sum(),
                "with_both": usable.height,
            }
        )
        predictions: dict[str, pl.DataFrame] = {}
        for arm in ("B0", "W7"):
            keyed = (
                saved.filter(pl.col("arm") == arm)
                .select("site", "time", "seed", saved=pl.col("prediction"))
                .join(
                    usable.select("site", "time", "past_climatology", "existing"),
                    on=["site", "time"],
                )
            )
            predictions[arm] = keyed.select("site", "time", "seed", prediction=pl.col("saved"))
            predictions[f"{arm} x past-only climatology"] = keyed.select(
                "site",
                "time",
                "seed",
                prediction=(1 - BLEND_SHARE) * pl.col("saved")
                + BLEND_SHARE * pl.col("past_climatology"),
            )
            predictions[f"{arm} x saved climatology"] = keyed.select(
                "site",
                "time",
                "seed",
                prediction=(1 - BLEND_SHARE) * pl.col("saved") + BLEND_SHARE * pl.col("existing"),
            )
        b0_keys = saved.filter(pl.col("arm") == "B0").select("site", "time", "seed")
        for label, column in (
            ("past-only climatology", "past_climatology"),
            ("saved climatology", "existing"),
        ):
            predictions[label] = b0_keys.join(
                usable.select("site", "time", prediction=pl.col(column)), on=["site", "time"]
            )
        scored_parts = []
        for arm, prediction in predictions.items():
            scored_parts.append(
                score_prediction(rows=usable, prediction=prediction, target=TARGET)
                .select("site", "time", "month", "seed", METRIC)
                .with_columns(arm=pl.lit(arm))
            )
        scored = pl.concat(scored_parts)
        if keep is not None:
            keep[lead] = scored
        for arm in predictions:
            mean_error = _mean(scored.filter(pl.col("arm") == arm))
            interval = (
                paired_interval(
                    scoped=scored.filter(pl.col("arm").is_in([arm, "B0"])),
                    treatment=arm,
                    reference="B0",
                    metric=METRIC,
                    level=EXPLORATORY_LEVEL,
                )
                if arm != "B0"
                else None
            )
            records.append(
                {
                    "lead_day": lead,
                    "arm": arm,
                    "error": mean_error,
                    "difference": None if interval is None else interval["difference"],
                    "lower": None if interval is None else interval["lower_95"],
                    "upper": None if interval is None else interval["upper_95"],
                    "n_rows": usable.height,
                }
            )
    table = pl.DataFrame(records)
    counts = pl.DataFrame(count_rows)
    first_record = hourly.group_by("site").agg(first=pl.col("time").min()).sort("first")
    late_site = first_record.row(-1, named=True)
    missing_total = missing.height
    missing_late = missing.filter(pl.col("site") == late_site["site"]).height
    earlier_years = hourly.select(
        pl.col("time").dt.year().min().alias("first"), pl.col("time").dt.year().max().alias("last")
    ).row(0, named=True)
    most_earlier = int(
        hourly.with_columns(year=pl.col("time").dt.year(), month=pl.col("time").dt.month())
        .group_by("site", "month")
        .agg(years=pl.col("year").n_unique())
        .select(pl.col("years").max())
        .item()
    )
    lines = [
        (
            "Post hoc. The saved climatology baseline is the median over the other folds' months "
            "(later months included), falling back on neighbouring calendar months. The past-only "
            "climatology is the median of the plant's cleaned hourly power at the same calendar "
            "month and hour in years strictly before the row's year, null where no earlier year "
            "holds the month and hour. Every arm is scored on the rows where both exist, so each "
            "lead-day's arms share the same rows. The blends are 50/50 with a saved B0 or W7 "
            "prediction. Intervals are 95%, from resampling whole months and a seed."
        ),
        "",
        (
            "**The two climatologies differ in history depth as well as in look-ahead.** The "
            f"past-only climatology reads the plant's record from {earlier_years['first']} "
            f"(up to {most_earlier - 1} earlier years of the same calendar month), whereas the "
            "saved climatology reads only the 18-month record of the study, so the gap between "
            "them mixes the look-ahead with the depth of history. At lead-day 1, "
            f"{missing_total} of {counts['rows'][0]} rows have no past-only value, and "
            f"{missing_late} of those fall on the plant whose record starts in "
            f"{late_site['first']:%B %Y}."
        ),
        "",
        *_table(
            header=["Lead-day", "Rows in the frame", "Rows with a past-only value", "Rows scored"],
            rows=[
                [str(r["lead"]), str(r["rows"]), str(r["with_past"]), str(r["with_both"])]
                for r in counts.iter_rows(named=True)
            ],
        ),
        *_table(
            header=[
                "Lead-day",
                "Arm",
                "Mean absolute error (pp)",
                "Difference to B0 (pp)",
                "95% interval (pp)",
            ],
            rows=[
                [
                    str(r["lead_day"]),
                    r["arm"],
                    _pp(r["error"], signed=False),
                    _pp(r["difference"]),
                    _interval(r["lower"], r["upper"]),
                ]
                for r in table.iter_rows(named=True)
            ],
        ),
    ]
    return lines, table.join(counts, left_on="lead_day", right_on="lead", how="left")


# --- 7. The fingerprint figure's rows ------------------------------------------------------------


FINGERPRINT_PAIRS: Final[tuple[tuple[str, str], ...]] = (
    ("G-ID", "G-B0"),
    ("G-ID+TF", "G-B0"),
    ("G-FP", "G-B0"),
    ("G-FPnoCK", "G-B0"),
    ("G-L1", "G-B0"),
    ("LOPO G-FP", "LOPO G-B0"),
    ("LOPO G-FPnoCK", "LOPO G-B0"),
    ("LOPO G-L1", "LOPO G-B0"),
    ("G-FP", "G-FPnoCK"),
    ("LOPO G-FP", "LOPO G-FPnoCK"),
)
"""The paired differences of the fingerprint figure. The first eight are each against the shared
model given nothing; the last two compare G-FP with G-FPnoCK, so the clipping ceiling's own effect
is measured directly, for plants in training and for a plant left out."""


def fingerprint_section(*, first_path: Path, follow_path: Path) -> tuple[list[str], pl.DataFrame]:
    """Pair every fingerprint arm with G-B0, from the saved global and leave-one-plant-out fits.

    Args:
        first_path: The first run's losses.
        follow_path: Round 1's losses.

    Returns:
        The report lines and a table with one row per pair and the two arms' own errors.
    """
    first = read_losses(path=first_path, scopes=["global", "lopo"])
    follow = read_losses(path=follow_path, scopes=["fu_global", "fu_lopo"])
    names = {"B0": "G-B0", "L1": "G-L1"}
    parts = []
    for scope, prefix in (("global", ""), ("lopo", "LOPO ")):
        parts.append(
            first.filter(pl.col("scope") == scope).with_columns(
                arm=prefix + pl.col("arm").replace(names)
            )
        )
    for scope, prefix in (("fu_global", ""), ("fu_lopo", "LOPO ")):
        parts.append(
            follow.filter(pl.col("scope") == scope).with_columns(arm=prefix + pl.col("arm"))
        )
    losses = pl.concat(parts)
    records = []
    for treatment, reference in FINGERPRINT_PAIRS:
        scoped = losses.filter(pl.col("arm").is_in([treatment, reference]))
        interval = paired_interval(
            scoped=scoped,
            treatment=treatment,
            reference=reference,
            metric=METRIC,
            level=EXPLORATORY_LEVEL,
        )
        if interval is None:
            continue
        records.append(
            {
                "treatment": treatment,
                "reference": reference,
                "difference": interval["difference"],
                "lower": interval["lower_95"],
                "upper": interval["upper_95"],
                "treatment_error": _mean(scoped.filter(pl.col("arm") == treatment)),
                "reference_error": _mean(scoped.filter(pl.col("arm") == reference)),
                "n_rows": interval["n_rows"],
            }
        )
    table = pl.DataFrame(records)
    lines = [
        (
            "Post hoc. Each fingerprint arm minus the shared model given nothing (G-B0, or LOPO "
            "G-B0 for a plant left out of training), on the fleet-wide folds. Intervals are 95%, "
            "from resampling whole months and a seed."
        ),
        "",
        *_table(
            header=[
                "Arm",
                "Reference",
                "Difference (pp)",
                "95% interval (pp)",
                "Arm's error (pp)",
                "Reference's error (pp)",
            ],
            rows=[
                [
                    r["treatment"],
                    r["reference"],
                    _pp(r["difference"]),
                    _interval(r["lower"], r["upper"]),
                    _pp(r["treatment_error"], signed=False),
                    _pp(r["reference_error"], signed=False),
                ]
                for r in table.iter_rows(named=True)
            ],
        ),
    ]
    return lines, table


# --- 8. Other numbers ----------------------------------------------------------------------------


def an_lines() -> list[str]:
    """Describe AN as coded, from the module's own constants and docstring.

    Returns:
        The report lines.
    """
    doc = lag_arm_columns.analogue_ensemble.__doc__ or ""
    summary = " ".join(doc.split("Args:")[0].split())
    return [
        "AN as coded in `lag_arm_columns.analogue_ensemble`:",
        "",
        f"> {summary}",
        "",
        (
            f"- The window is the last {lag_arm_columns.ANALOGUE_DAYS} whole days before the "
            f"issue day, the same clock hour on each; the {lag_arm_columns.ANALOGUE_COUNT} "
            "candidates whose forecast clear-sky index is closest to the target's are kept; the "
            f"row's columns are null with fewer than {lag_arm_columns.ANALOGUE_MIN} usable "
            "candidates."
        ),
        (
            "- The three columns are `an_mean` (the mean of the kept candidates' power, rescaled "
            "by the target's clear-sky irradiance over the candidate's), `an_csi` (their mean "
            "forecast clear-sky index) and `an_spread` (the standard deviation of the rescaled "
            "power)."
        ),
        (
            "- Only candidates with positive clear-sky irradiance, a measured power and a "
            "forecast irradiance are usable."
        ),
        "",
    ]


def l1_interval_lines(*, first_path: Path, follow_path: Path) -> list[str]:
    """Print L1 minus B0 at the 10% month-level control from both reports' fits, 99% interval.

    Args:
        first_path: The first run's losses.
        follow_path: Round 1's losses.

    Returns:
        The report lines.
    """
    first = read_losses(path=first_path, scopes=["control_s10"])
    follow = read_losses(path=follow_path, scopes=["fu_control_month_s10"])
    rows = []
    for label, scoped, source in (
        ("first run", first, "scope `control_s10`, built by `build_lag_frame.py`"),
        ("round 1", follow, "scope `fu_control_month_s10`, built by `followup_frames.py`"),
    ):
        interval = paired_interval(
            scoped=scoped, treatment="L1", reference="B0", metric=METRIC, level=PLANNED_LEVEL
        )
        if interval is not None:
            rows.append(
                [
                    label,
                    source,
                    f"{interval['difference'] * PERCENTAGE_POINTS:+.4f}",
                    (
                        f"[{interval['lower_95'] * PERCENTAGE_POINTS:+.4f}, "
                        f"{interval['upper_95'] * PERCENTAGE_POINTS:+.4f}]"
                    ),
                ]
            )
    return [
        (
            "L1 minus B0 at the 10% month-level control, 99% interval. The two reports quote "
            "different fits of the same specification: the first run's control frame and its fits "
            "(first report), and round 1's rebuilt control frame and fits (follow-up report). "
            "Each interval is the bootstrap of its own fits."
        ),
        "",
        *_table(header=["Run", "Source", "Difference (pp)", "99% interval (pp)"], rows=rows),
    ]


def hit_rate_lines(*, root: Path) -> tuple[list[str], pl.DataFrame]:
    """Tabulate the share of hours below the 10% quantile and above the 90% quantile by month.

    Args:
        root: The output root.

    Returns:
        The report lines and the table.
    """
    hits = pl.read_parquet(
        followup2_dir(root=root).parent
        / "followups"
        / f"tables_followups_{PRODUCT}"
        / "hit_rates.parquet"
    )
    below = hits.filter(pl.col("level").round(2) == 0.1).select(
        "arm", "month", below_10=pl.col("hit_rate")
    )
    above = hits.filter(pl.col("level").round(2) == 0.9).select(
        "arm", "month", above_90=1.0 - pl.col("hit_rate")
    )
    table = below.join(above, on=["arm", "month"]).sort("month", "arm")
    months = sorted(table["month"].unique().to_list())
    rows = []
    for month in months:
        part = table.filter(pl.col("month") == month)
        cells = []
        for arm in ("B0", "L1"):
            row = part.filter(pl.col("arm") == arm)
            cells += [
                f"{row['below_10'][0] * 100:.1f}%" if not row.is_empty() else "",
                f"{row['above_90'][0] * 100:.1f}%" if not row.is_empty() else "",
            ]
        rows.append([month, *cells])
    lines = [
        (
            "Share of hours with measured power below the forecast's 10% quantile, and above its "
            "90% quantile, by month. A calibrated forecast has 10% in each. The 'all' row pools "
            "the months."
        ),
        "",
        *_table(
            header=["Month", "B0 below 10%", "B0 above 90%", "L1 below 10%", "L1 above 90%"],
            rows=rows,
        ),
    ]
    return lines, table


# --- Round 2's fits ------------------------------------------------------------------------------


def fit_sections(  # noqa: PLR0912
    *, first_path: Path, fit_path: Path
) -> tuple[list[str], dict[str, pl.DataFrame]]:
    """Report the four fit blocks, or say that the fits have not run.

    Args:
        first_path: The first run's losses.
        fit_path: Round 2's losses.

    Returns:
        The report lines and the tables.
    """
    if not fit_path.exists():
        return [
            (
                "The round-2 fits have not been run, so this section is empty. Re-run this "
                "report after `fit_followups2.py`."
            ),
            "",
        ], {}
    fits = read_losses(path=fit_path, scopes=list(FIT_SCOPES))
    first = read_losses(path=first_path, scopes=["lead1", "lead2"])
    lines: list[str] = []
    tables: dict[str, pl.DataFrame] = {}
    # Width-matched nulls.
    lead1 = first.filter(pl.col("scope") == "lead1")
    null_rows = []
    nulls = fits.filter(pl.col("scope") == "fu2_lead1")
    pool = pl.concat([lead1.drop("scope"), nulls.drop("scope")])
    for treatment, reference in (
        ("N2", "B0"),
        ("N2-2", "B0"),
        ("N2-3", "B0"),
        ("W7", "B0"),
        ("W7", "N2-3"),
        ("AN", "N2-3"),
        ("N2-3", "N2"),
        ("PC", "N2-2"),
        ("TF", "N2-2"),
        ("Q30", "N2-2"),
    ):
        interval = paired_interval(
            scoped=pool.filter(pl.col("arm").is_in([treatment, reference])),
            treatment=treatment,
            reference=reference,
            metric=METRIC,
            level=EXPLORATORY_LEVEL,
        )
        if interval is not None:
            null_rows.append(
                {
                    "treatment": treatment,
                    "reference": reference,
                    "difference": interval["difference"],
                    "lower": interval["lower_95"],
                    "upper": interval["upper_95"],
                }
            )
    tables["width_nulls"] = pl.DataFrame(null_rows)
    lines += [
        "### Width-matched nulls at lead-day 1",
        "",
        *_table(
            header=["Treatment", "Reference", "Difference (pp)", "95% interval (pp)"],
            rows=[
                [
                    r["treatment"],
                    r["reference"],
                    _pp(r["difference"]),
                    _interval(r["lower"], r["upper"]),
                ]
                for r in tables["width_nulls"].iter_rows(named=True)
            ],
        ),
    ]
    # Day-of-year test.
    doy_rows = []
    for arm in ("B0", "PC", "W7", "TF"):
        scoped = pl.concat(
            [
                lead1.filter(pl.col("arm") == arm).drop("scope"),
                nulls.filter(pl.col("arm") == f"{arm}-noDOY")
                .drop("scope")
                .with_columns(arm=pl.lit(f"{arm}-noDOY")),
            ]
        )
        interval = paired_interval(
            scoped=scoped,
            treatment=f"{arm}-noDOY",
            reference=arm,
            metric=METRIC,
            level=EXPLORATORY_LEVEL,
        )
        if interval is not None:
            doy_rows.append(
                {
                    "comparison": f"{arm} without day_of_year minus {arm} with it",
                    "difference": interval["difference"],
                    "lower": interval["lower_95"],
                    "upper": interval["upper_95"],
                }
            )
    for arm in ("PC", "W7", "TF"):
        for suffix, source in (("", lead1), ("-noDOY", nulls)):
            scoped = pl.concat(
                [
                    source.filter(pl.col("arm") == f"B0{suffix}").drop("scope"),
                    source.filter(pl.col("arm") == f"{arm}{suffix}").drop("scope"),
                ]
            )
            interval = paired_interval(
                scoped=scoped,
                treatment=f"{arm}{suffix}",
                reference=f"B0{suffix}",
                metric=METRIC,
                level=EXPLORATORY_LEVEL,
            )
            if interval is not None:
                doy_rows.append(
                    {
                        "comparison": f"{arm}{suffix} minus B0{suffix}",
                        "difference": interval["difference"],
                        "lower": interval["lower_95"],
                        "upper": interval["upper_95"],
                    }
                )
    for arm in ("PC", "W7", "TF"):
        with_year = wide(losses=lead1.drop("scope"), names={"B0": "b0_with", arm: "arm_with"})
        without_year = wide(
            losses=nulls.drop("scope"),
            names={"B0-noDOY": "b0_without", f"{arm}-noDOY": "arm_without"},
        ).drop("month")
        joined = with_year.join(without_year, on=["site", "time", "seed"], how="inner")
        data = Aligned(frame=joined, names=["b0_with", "arm_with", "b0_without", "arm_without"])
        interval = month_bootstrap(
            data=data,
            statistic=lambda d: float(
                ((d["arm_with"] - d["b0_with"]) - (d["arm_without"] - d["b0_without"])).mean()
            ),
        )
        doy_rows.append(
            {
                "comparison": (
                    f"({arm} minus B0) minus ({arm}-noDOY minus B0-noDOY), paired by row and seed"
                ),
                "difference": interval["value"],
                "lower": interval["lower_95"],
                "upper": interval["upper_95"],
            }
        )
    tables["day_of_year"] = pl.DataFrame(doy_rows)
    lines += [
        "### Day-of-year test at lead-day 1",
        "",
        *_table(
            header=["Comparison", "Difference (pp)", "95% interval (pp)"],
            rows=[
                [r["comparison"], _pp(r["difference"]), _interval(r["lower"], r["upper"])]
                for r in tables["day_of_year"].iter_rows(named=True)
            ],
        ),
    ]
    # IFS replicate.
    ifs_rows = []
    for lead in IFS_LEAD_DAYS:
        ifs = fits.filter(pl.col("scope") == f"fu2_ifs_lead{lead}")
        if ifs.is_empty():
            continue
        keys = ifs.select("site", "time").unique()
        enst = first.filter(pl.col("scope") == f"lead{lead}").join(keys, on=["site", "time"])
        for product, source in (("IFS HRES", ifs), ("ENS mean", enst)):
            scoped = source.drop("scope")
            interval = paired_interval(
                scoped=scoped,
                treatment="L1",
                reference="B0",
                metric=METRIC,
                level=EXPLORATORY_LEVEL,
            )
            if interval is None:
                continue
            ifs_rows.append(
                {
                    "lead_day": lead,
                    "weather": product,
                    "b0_error": _mean(scoped.filter(pl.col("arm") == "B0")),
                    "l1_error": _mean(scoped.filter(pl.col("arm") == "L1")),
                    "difference": interval["difference"],
                    "lower": interval["lower_95"],
                    "upper": interval["upper_95"],
                    "n_rows": keys.height,
                }
            )
    tables["ifs"] = pl.DataFrame(ifs_rows)
    lines += [
        "### IFS HRES replicate (rows where IFS exists; ENS mean on the same rows)",
        "",
        *_table(
            header=[
                "Lead-day",
                "Weather",
                "B0 error (pp)",
                "L1 error (pp)",
                "L1 minus B0 (pp)",
                "95% interval (pp)",
                "Rows",
            ],
            rows=[
                [
                    str(r["lead_day"]),
                    r["weather"],
                    _pp(r["b0_error"], signed=False),
                    _pp(r["l1_error"], signed=False),
                    _pp(r["difference"]),
                    _interval(r["lower"], r["upper"]),
                    str(r["n_rows"]),
                ]
                for r in tables["ifs"].iter_rows(named=True)
            ],
        ),
    ]
    return lines, tables


# --- The report ----------------------------------------------------------------------------------


def report_text(*, root: Path, tables: Path, smoke: bool) -> str:
    """Build the whole round-2 report and save the tables the charts read.

    Args:
        root: The output root.
        tables: The directory to write the tables into.
        smoke: Whether to report a smoke run's files.

    Returns:
        The report's markdown.
    """
    suffix = run_suffix(smoke=smoke)
    first_path = root / PRODUCT / f"losses_{PRODUCT}{suffix}.parquet"
    follow_path = root / PRODUCT / "followups" / f"losses_followups_{PRODUCT}{suffix}.parquet"
    fit_path = followup2_dir(root=root) / f"losses_followups2_{PRODUCT}{suffix}.parquet"
    out = [
        "# Round-2 follow-ups after the second science review (post hoc)",
        "",
        POST_HOC,
        "",
    ]
    named: dict[str, pl.DataFrame] = {}

    def add(title: str, lines: list[str], **tables_: pl.DataFrame) -> None:
        out.extend([f"## {title}", "", POST_HOC, "", *lines])
        named.update(tables_)

    lines, shares = corrected_shares_section(
        first_path=first_path, follow_path=follow_path, fit_path=fit_path
    )
    add("1. Positive-control shares, corrected for each arm's standing gain", lines, shares=shares)
    lines, hindsight = hindsight_section(first_path=first_path, root=root)
    add("2. Hindsight monthly rescaling of B0", lines, hindsight2=hindsight)

    reviewed = ("B0", *REVIEWED_ARMS)
    lead1 = read_losses(path=first_path, scopes=["lead1"])
    aligned = Aligned(
        frame=wide(losses=lead1, names={a: a for a in reviewed}),
        names=list(reviewed),
    )
    lines, plant_tables = plant_month_section(lead1=lead1, aligned=aligned)
    add("3. The one plant-month that dominates", lines, **plant_tables)
    lines, screening = screening_section(lead1=lead1)
    add("4. Paired contrasts on the screening months", lines, screening_contrasts=screening)
    lines, equivalence = equivalence_section(root=root, suffix=suffix)
    add("5. Equivalence against the decision threshold", lines, equivalence=equivalence)
    lines, clim = climatology_section(first_path=first_path, root=root, hourly=lag_source_hourly())
    add("6. Climatology from earlier months only", lines, climatology_past_only=clim)
    lines, fingerprint = fingerprint_section(first_path=first_path, follow_path=follow_path)
    add("7. The fingerprint figure's rows", lines, fingerprint_pairs=fingerprint)
    lines, hits = hit_rate_lines(root=root)
    add(
        "8. Other numbers the review asked for",
        [
            *an_lines(),
            *l1_interval_lines(first_path=first_path, follow_path=follow_path),
            "### Quantile hit rates by month",
            "",
            *lines,
        ],
        hit_rates_by_month=hits,
    )
    lines, fit_tables = fit_sections(first_path=first_path, fit_path=fit_path)
    add(
        "9 to 12. Round 2's fits (PC in the step control, width-matched nulls, day of year, IFS)",
        lines,
        **fit_tables,
    )
    for name, table in named.items():
        if not table.is_empty():
            write_parquet_atomic(frame=table, path=tables / f"{name}.parquet")
    return "\n".join(out) + "\n"


def main() -> int:
    """Write the round-2 report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=LAG_FEATURES_DIR)
    parser.add_argument("--smoke", action="store_true", help="Report a `--smoke` fit's output.")
    arguments = parser.parse_args()
    root: Path = arguments.output_root
    suffix = run_suffix(smoke=arguments.smoke)
    directory = followup2_dir(root=root)
    fit_path = directory / f"losses_followups2_{PRODUCT}{suffix}.parquet"
    verify_manifest_inputs(root=root, product=PRODUCT, smoke=arguments.smoke)
    verify_followup_manifest(root=root, smoke=arguments.smoke)
    if fit_path.exists():
        verify_followup2_manifest(root=root, smoke=arguments.smoke)
    tag = "" if fit_path.exists() else "_nofit"
    path = directory / f"report_followups2_{PRODUCT}{suffix}{tag}.md"
    final_tables = directory / f"tables_followups2_{PRODUCT}{suffix}{tag}"
    refuse_to_overwrite(paths=[path, final_tables])
    directory.mkdir(parents=True, exist_ok=True)
    partial_tables = final_tables.with_name(final_tables.name + ".partial")
    if partial_tables.exists():
        for old in partial_tables.iterdir():
            old.unlink()
        partial_tables.rmdir()
    partial_tables.mkdir()
    text = report_text(root=root, tables=partial_tables, smoke=arguments.smoke)
    partial_tables.rename(final_tables)
    partial = path.with_name(path.name + ".partial")
    partial.write_text(text)
    partial.replace(path)
    sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
