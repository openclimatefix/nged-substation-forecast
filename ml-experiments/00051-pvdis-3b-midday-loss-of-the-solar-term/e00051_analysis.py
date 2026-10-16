"""Read experiment 51's causes (a) clipping, (b) coefficient, and (c) member noise.

Reads `work/rows.parquet` and `work/arms_long.parquet` (see `e00051_data.py`) and writes
`work/results/*.parquet` plus a printed report. The pre-registered rule is in
`docs/ml-experiments/00051-pvdis-3b-midday-loss-of-the-solar-term/result.md`.

Notation: in points of capacity, `r` is the actual minus the `d1l` median, `shift` is the oracle
shift, and `e = r + shift`, the error left after the oracle shift with coefficient 1 (positive where
the shift over-corrects). `csi` is the CAMS clear-sky index.

    uv run python ml-experiments/00051-*/e00051_analysis.py
"""

from typing import Any, Final

import numpy as np
import polars as pl
from e00051_data import LEVEL_COLUMNS, WORK_DIR

RESULTS_DIR: Final[Any] = WORK_DIR / "results"
LEVELS: Final[np.ndarray] = np.array([0.05, 0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9, 0.95])
ATOM_MASSES: Final[np.ndarray] = np.diff(
    np.concatenate([[0.0], (LEVELS[:-1] + LEVELS[1:]) / 2, [1.0]])
)
KNEES: Final[np.ndarray] = np.arange(0.4, 0.951, 0.025)
RESAMPLES: Final[int] = 10_000
REFIT_RESAMPLES: Final[int] = 1_000
GENERATOR: Final[int] = 20261016
POINTS: Final[float] = 100.0
HOUR_BINS: Final[tuple[tuple[str, int, int], ...]] = (
    ("09-12 UTC", 9, 12),
    ("12-15 UTC", 12, 16),
)
DAY_BANDS: Final[tuple[tuple[str, float, float], ...]] = (
    ("day+1", 24.0, 48.0),
    ("day+2", 48.0, 72.0),
    ("day+3", 72.0, 96.0),
)
SUMMER_HOLIDAY_START: Final[tuple[int, int]] = (7, 22)
SUNNY: Final[float] = 0.7
CLOUDY: Final[float] = 0.3
KEYS: Final[tuple[str, ...]] = ("site", "power_fcst_init_time", "valid_time")


def quantile_score(*, quantiles: np.ndarray, actual: np.ndarray) -> np.ndarray:
    """Return twice the mass-weighted mean pinball loss of the nine levels, per case."""
    difference = actual[:, None] - quantiles
    pinball = np.where(difference >= 0, LEVELS * difference, (LEVELS - 1) * difference)
    return 2.0 * (pinball * ATOM_MASSES).sum(axis=1)


def load_cases() -> pl.DataFrame:
    """Return one row per case with the seed-averaged median and score of each arm, in points.

    Returns:
        The context columns of `rows.parquet` plus `r` (actual minus `d1l` median), `e`, `shift`,
        `<arm>_qs`, `<arm>_q05`, `<arm>_q95`, and the calendar columns the splits use.
    """
    rows = pl.read_parquet(WORK_DIR / "rows.parquet")
    long = pl.read_parquet(WORK_DIR / "arms_long.parquet").join(
        rows.select(*KEYS, "actual"), on=list(KEYS)
    )
    quantiles = long.select(LEVEL_COLUMNS).to_numpy().astype(np.float64)
    long = long.with_columns(
        qs=pl.Series(
            quantile_score(quantiles=quantiles, actual=long["actual"].to_numpy().astype(np.float64))
            * POINTS
        )
    )
    wide = (
        long.group_by("arm", *KEYS)
        .agg(
            qs=pl.col("qs").mean(),
            q50=pl.col("q50").mean(),
            q05=pl.col("q05").mean(),
            q95=pl.col("q95").mean(),
            below05=(pl.col("actual") < pl.col("q05")).mean(),
            below95=(pl.col("actual") < pl.col("q95")).mean(),
        )
        .pivot(on="arm", index=list(KEYS), values=["qs", "q50", "below05", "below95"])
    )
    cases = rows.join(wide, on=list(KEYS), validate="1:1").sort(KEYS)
    local = pl.col("valid_time").dt.convert_time_zone("Europe/London")
    month_day = local.dt.month() * 100 + local.dt.day()
    return cases.with_columns(
        shift=pl.col("oracle_shift") * POINTS,
        ens_mean_shift_points=pl.col("ens_mean_shift") * POINTS,
        r=(pl.col("actual") * POINTS - pl.col("q50_d1l") * POINTS),
        hour=pl.col("valid_time").dt.hour(),
        week=pl.col("power_fcst_init_time").dt.strftime("%G-W%V"),
        valid_week_parity=pl.col("valid_time").dt.week() % 2,
        weekend=local.dt.weekday() >= 6,
        school_holiday=month_day >= SUMMER_HOLIDAY_START[0] * 100 + SUMMER_HOLIDAY_START[1],
        day_band=pl.col("lead_hours").map_elements(
            lambda hours: next(n for n, lo, hi in DAY_BANDS if lo <= hours < hi),
            return_dtype=pl.String,
        ),
    ).with_columns(
        e=pl.col("r") + pl.col("shift"),
        load_group=pl.when(pl.col("weekend") | pl.col("school_holiday"))
        .then(pl.lit("holiday or weekend"))
        .otherwise(pl.lit("term-time weekday")),
    )


def block_bootstrap(*, values: np.ndarray, blocks: np.ndarray) -> tuple[float, float, float, int]:
    """Return the mean of per-row values and a 95% interval resampling whole blocks.

    Args:
        values: One value per row.
        blocks: The block label of each row.

    Returns:
        The mean, the 2.5% and 97.5% bounds, and the number of blocks.
    """
    labels, inverse = np.unique(blocks, return_inverse=True)
    sums = np.bincount(inverse, weights=values)
    counts = np.bincount(inverse).astype(np.float64)
    generator = np.random.default_rng(GENERATOR)
    draws = generator.integers(0, labels.size, size=(RESAMPLES, labels.size))
    means = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
    lower, upper = np.percentile(means, [2.5, 97.5])
    return float(values.mean()), float(lower), float(upper), int(labels.size)


def paired(*, cases: pl.DataFrame, treatment: str, reference: str, metric: str = "qs") -> dict:
    """Return the mean of treatment minus reference with its block interval."""
    values = (cases[f"{metric}_{treatment}"] - cases[f"{metric}_{reference}"]).to_numpy()
    mean, lower, upper, blocks = block_bootstrap(values=values, blocks=cases["week"].to_numpy())
    return {
        "difference": mean,
        "lower": lower,
        "upper": upper,
        "blocks": blocks,
        "rows": cases.height,
    }


def fit_hinge(*, csi: np.ndarray, e: np.ndarray) -> tuple[float, float, float]:
    """Return (knee, intercept, slope above the knee) with the least squared error on the grid."""
    best = (np.inf, KNEES[0], 0.0, 0.0)
    for knee in KNEES:
        x = np.maximum(0.0, csi - knee)
        if (x > 0).sum() < 10:
            continue
        design = np.column_stack([np.ones_like(x), x])
        coefficients, *_ = np.linalg.lstsq(design, e, rcond=None)
        error = float(((design @ coefficients - e) ** 2).sum())
        if error < best[0]:
            best = (error, knee, coefficients[0], coefficients[1])
    return best[1], best[2], best[3]


def fit_line(*, csi: np.ndarray, e: np.ndarray) -> tuple[float, float]:
    """Return (intercept, slope) of the least-squares line of `e` on `csi`."""
    design = np.column_stack([np.ones_like(csi), csi])
    coefficients, *_ = np.linalg.lstsq(design, e, rcond=None)
    return float(coefficients[0]), float(coefficients[1])


def fit_coefficient(*, shift: np.ndarray, r: np.ndarray) -> float:
    """Return the through-origin coefficient k minimising the squared error of r + k * shift."""
    return float(-(shift * r).sum() / (shift**2).sum())


def out_of_fold(*, cases: pl.DataFrame) -> pl.DataFrame:
    """Return per-case absolute errors of the oracle median and of repairs A, L, and B, out of fold.

    Every repair is fitted per series on the cases whose valid-time ISO week has the other parity,
    and scored on the cases with this parity.
    """
    pieces = []
    for site in cases["site"].unique(maintain_order=True):
        part = cases.filter(pl.col("site") == site)
        for parity in (0, 1):
            test = part.filter(pl.col("valid_week_parity") == parity)
            train = part.filter(pl.col("valid_week_parity") != parity)
            knee, intercept, c = fit_hinge(
                csi=train["clear_sky_index"].to_numpy(), e=train["e"].to_numpy()
            )
            b0, b1 = fit_line(csi=train["clear_sky_index"].to_numpy(), e=train["e"].to_numpy())
            csi_test = test["clear_sky_index"].to_numpy()
            e_test = test["e"].to_numpy()
            hinge_prediction = intercept + c * np.maximum(0.0, csi_test - knee)
            line_prediction = b0 + b1 * csi_test
            coefficient_error = np.zeros(test.height)
            train_after_b = np.zeros(train.height)
            test_after_b = np.zeros(test.height)
            for _name, lo, hi in HOUR_BINS:
                train_mask = ((train["hour"] >= lo) & (train["hour"] < hi)).to_numpy()
                test_mask = ((test["hour"] >= lo) & (test["hour"] < hi)).to_numpy()
                k = fit_coefficient(
                    shift=train["shift"].to_numpy()[train_mask], r=train["r"].to_numpy()[train_mask]
                )
                train_after_b[train_mask] = (
                    train["r"].to_numpy()[train_mask] + k * train["shift"].to_numpy()[train_mask]
                )
                test_after_b[test_mask] = (
                    test["r"].to_numpy()[test_mask] + k * test["shift"].to_numpy()[test_mask]
                )
                coefficient_error[test_mask] = np.abs(test_after_b[test_mask])
            # Post hoc: a hinge fitted to what the coefficient repair leaves behind.
            knee_b, intercept_b, c_b = fit_hinge(
                csi=train["clear_sky_index"].to_numpy(), e=train_after_b
            )
            after_b_prediction = intercept_b + c_b * np.maximum(0.0, csi_test - knee_b)
            pieces.append(
                test.with_columns(
                    abs_unrepaired=pl.Series(np.abs(e_test)),
                    abs_hinge=pl.Series(np.abs(e_test - hinge_prediction)),
                    abs_line=pl.Series(np.abs(e_test - line_prediction)),
                    abs_coefficient=pl.Series(coefficient_error),
                    abs_coefficient_hinge=pl.Series(np.abs(test_after_b - after_b_prediction)),
                    abs_d1l=pl.Series(np.abs(test["r"].to_numpy())),
                )
            )
    return pl.concat(pieces)


def slope_with_interval(
    *, cases: pl.DataFrame, rng: np.random.Generator
) -> tuple[float, float, float]:
    """Return the slope of `r` on `shift` with a 95% week-block interval."""
    weeks = cases["week"].unique().sort().to_numpy()
    index_by_week = {week: np.flatnonzero(cases["week"].to_numpy() == week) for week in weeks}
    shift, r = cases["shift"].to_numpy(), cases["r"].to_numpy()

    def slope(idx: np.ndarray) -> float:
        x, y = shift[idx] - shift[idx].mean(), r[idx] - r[idx].mean()
        return float((x * y).sum() / (x**2).sum())

    estimate = slope(np.arange(cases.height))
    draws = []
    for _ in range(REFIT_RESAMPLES):
        picked = rng.choice(weeks, size=weeks.size, replace=True)
        draws.append(slope(np.concatenate([index_by_week[w] for w in picked])))
    lower, upper = np.percentile(draws, [2.5, 97.5])
    return estimate, float(lower), float(upper)


def main() -> None:  # noqa: PLR0912, PLR0915
    """Print and save every read."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    cases = load_cases()
    rng = np.random.default_rng(GENERATOR)
    print("cases (seed-averaged):", cases.height, "weeks:", cases["week"].n_unique())
    with pl.Config(tbl_rows=200, tbl_cols=30, fmt_float="mixed", tbl_width_chars=250):
        # 0. The loss to explain.
        loss_rows = []
        for band, scope in (
            ("day+2", pl.col("day_band") == "day+2"),
            ("day+3", pl.col("day_band") == "day+3"),
            ("day+2 and day+3", pl.col("day_band").is_in(["day+2", "day+3"])),
            ("day+1", pl.col("day_band") == "day+1"),
        ):
            part = cases.filter(scope)
            for name, treatment, reference in (
                ("C6 ens_shift - d0", "d1l_ens_shift", "d0"),
                ("C3 ens_shift - d1l", "d1l_ens_shift", "d1l"),
                ("C4 cams_shift - d1l", "d1l_cams_shift", "d1l"),
                ("ens_shift - ens_mean_shift (c)", "d1l_ens_shift", "d1l_ens_mean_shift"),
                ("ens_mean_shift - d1l", "d1l_ens_mean_shift", "d1l"),
                ("post hoc: fitted ens_shift - ens_shift", "d1l_ens_shift_fitted", "d1l_ens_shift"),
                ("post hoc: fitted ens_shift - d0", "d1l_ens_shift_fitted", "d0"),
            ):
                loss_rows.append(
                    {"band": band, "contrast": name}
                    | paired(cases=part, treatment=treatment, reference=reference)
                )
        loss = pl.DataFrame(loss_rows)
        reproduced = loss.filter(pl.col("contrast") == "C6 ens_shift - d0").filter(
            pl.col("band").is_in(["day+2", "day+3"])
        )
        for band, published in (("day+2", 0.477), ("day+3", 0.172)):
            value = float(reproduced.filter(pl.col("band") == band)["difference"][0])
            if abs(value - published) > 0.005:
                msg = f"C6 at {band} is {value:.3f}, experiment 46 published {published}"
                raise ValueError(msg)
        print("\n0. Quantile-score contrasts (negative favours the treatment), points of capacity")
        print(loss)
        loss.write_parquet(RESULTS_DIR / "contrasts.parquet")

        # (c) Coverage of the sunny lower tail.
        coverage_rows = []
        for band in ("day+1", "day+2", "day+3"):
            part = cases.filter((pl.col("day_band") == band) & (pl.col("clear_sky_index") >= SUNNY))
            for arm in ("d1l", "d1l_ens_shift", "d1l_ens_mean_shift", "d1l_cams_shift"):
                coverage_rows.append(  # noqa: PERF401
                    {
                        "band": band,
                        "arm": arm,
                        "sunny_rows": part.height,
                        "below_q05": float(part[f"below05_{arm}"].to_numpy().mean()),
                    }
                )
        coverage = pl.DataFrame(coverage_rows)
        print("\n(c) Share of actuals below the 0.05 quantile, sunny rows (target 0.05)")
        print(coverage)
        coverage.write_parquet(RESULTS_DIR / "coverage.parquet")

        # (b) Slope of the d1l residual on the oracle shift by month and hour bin.
        slope_rows = []
        for month in (6, 7, 8):
            for name, lo, hi in HOUR_BINS:
                part = cases.filter(
                    (pl.col("local_month") == month) & pl.col("hour").is_between(lo, hi - 1)
                )
                estimate, lower, upper = slope_with_interval(cases=part, rng=rng)
                slope_rows.append(
                    {
                        "month": month,
                        "hours": name,
                        "slope": estimate,
                        "lower": lower,
                        "upper": upper,
                        "rows": part.height,
                    }
                )
        for band, _lo, _hi in DAY_BANDS:
            part = cases.filter(pl.col("day_band") == band)
            estimate, lower, upper = slope_with_interval(cases=part, rng=rng)
            slope_rows.append(
                {
                    "month": 0,
                    "hours": band,
                    "slope": estimate,
                    "lower": lower,
                    "upper": upper,
                    "rows": part.height,
                }
            )
        for site in cases["site"].unique(maintain_order=True):
            part = cases.filter(pl.col("site") == site)
            estimate, lower, upper = slope_with_interval(cases=part, rng=rng)
            slope_rows.append(
                {
                    "month": -1,
                    "hours": site,
                    "slope": estimate,
                    "lower": lower,
                    "upper": upper,
                    "rows": part.height,
                }
            )
        slopes = pl.DataFrame(slope_rows)
        cells = slopes.filter(pl.col("month") > 0)
        print(
            "\n(b) cells with the interval wholly above -0.85:",
            cells.filter(pl.col("lower") > -0.85).height,
            "of",
            cells.height,
        )
        print("\n(b) Slope of the d1l residual on the oracle shift (a coefficient of 1 gives -1)")
        print("    month 0 = by lead day, month -1 = by series")
        print(slopes)
        slopes.write_parquet(RESULTS_DIR / "slopes.parquet")

        # (a) Hinge, per series, on all rows.
        hinge_rows = []
        for site in cases["site"].unique(maintain_order=True):
            part = cases.filter(pl.col("site") == site)
            csi, e = part["clear_sky_index"].to_numpy(), part["e"].to_numpy()
            knee, _intercept, c = fit_hinge(csi=csi, e=e)
            weeks = part["week"].unique().sort().to_numpy()
            index_by_week = {w: np.flatnonzero(part["week"].to_numpy() == w) for w in weeks}
            slopes_draw = []
            knees_draw = []
            for _ in range(REFIT_RESAMPLES):
                picked = rng.choice(weeks, size=weeks.size, replace=True)
                idx = np.concatenate([index_by_week[w] for w in picked])
                k_b, _a_b, c_b = fit_hinge(csi=csi[idx], e=e[idx])
                slopes_draw.append(c_b)
                knees_draw.append(k_b)
            near = part.filter((pl.col("clear_sky_index") - knee).abs() <= 0.025)
            u_knee = float(np.median(near["u_cams"].to_numpy())) if near.height else float("nan")
            hinge_rows.append(
                {
                    "site": site,
                    "coastal": bool(part["coastal"][0]),
                    "knee_csi": knee,
                    "knee_lower": float(np.percentile(knees_draw, 2.5)),
                    "knee_upper": float(np.percentile(knees_draw, 97.5)),
                    "slope_above_knee": c,
                    "slope_lower": float(np.percentile(slopes_draw, 2.5)),
                    "slope_upper": float(np.percentile(slopes_draw, 97.5)),
                    "u_at_knee": u_knee,
                    "implied_dc_ac": 1.0 / u_knee if u_knee > 0 else float("nan"),
                    "rows": part.height,
                }
            )
        hinge = pl.DataFrame(hinge_rows)
        print("\n(a) Hinge fit of e on the clear-sky index, per series (labels anonymised)")
        print(hinge)
        hinge.write_parquet(RESULTS_DIR / "hinge.parquet")

        # Out-of-fold repairs.
        folded = out_of_fold(cases=cases)
        folded.write_parquet(RESULTS_DIR / "out_of_fold.parquet")
        repair_rows = []
        for scope_name, scope in (
            ("all", pl.lit(True)),
            ("day+2 and day+3", pl.col("day_band").is_in(["day+2", "day+3"])),
            ("sunny (csi >= 0.7)", pl.col("clear_sky_index") >= SUNNY),
            ("coastal", pl.col("coastal")),
            ("inland", ~pl.col("coastal")),
            ("term-time weekday", pl.col("load_group") == "term-time weekday"),
            ("holiday or weekend", pl.col("load_group") == "holiday or weekend"),
        ):
            part = folded.filter(scope)
            blocks = part["week"].to_numpy()
            for name, treatment, reference in (
                ("A hinge - unrepaired", "abs_hinge", "abs_unrepaired"),
                ("L line - unrepaired", "abs_line", "abs_unrepaired"),
                ("A hinge - L line", "abs_hinge", "abs_line"),
                ("B coefficient - unrepaired", "abs_coefficient", "abs_unrepaired"),
                ("post hoc: B then hinge - B", "abs_coefficient_hinge", "abs_coefficient"),
                ("unrepaired oracle - d1l", "abs_unrepaired", "abs_d1l"),
            ):
                mean, lower, upper, n_blocks = block_bootstrap(
                    values=(part[treatment] - part[reference]).to_numpy(), blocks=blocks
                )
                repair_rows.append(
                    {
                        "scope": scope_name,
                        "contrast": name,
                        "difference": mean,
                        "lower": lower,
                        "upper": upper,
                        "blocks": n_blocks,
                        "rows": part.height,
                    }
                )
        repairs = pl.DataFrame(repair_rows)
        print("\nOut-of-fold mean absolute error of the median, points; negative favours the first")
        print(repairs)
        repairs.write_parquet(RESULTS_DIR / "repairs.parquet")

        # Splits: hinge slope above each series' knee, refitted per group with the knee fixed.
        split_rows = []
        knees = {r["site"]: r["knee_csi"] for r in hinge_rows}
        for group_column in ("load_group", "coastal"):
            for group in cases[group_column].unique(maintain_order=True):
                part = cases.filter(pl.col(group_column) == group)
                per_site = []
                for site in part["site"].unique():
                    sub = part.filter(pl.col("site") == site)
                    x = np.maximum(0.0, sub["clear_sky_index"].to_numpy() - knees[site])
                    if (x > 0).sum() < 10:
                        continue
                    design = np.column_stack([np.ones_like(x), x])
                    coefficients, *_ = np.linalg.lstsq(design, sub["e"].to_numpy(), rcond=None)
                    per_site.append(float(coefficients[1]))
                split_rows.append(
                    {
                        "split": group_column,
                        "group": str(group),
                        "series": len(per_site),
                        "mean_slope_above_knee": float(np.mean(per_site))
                        if per_site
                        else float("nan"),
                        "rows": part.height,
                    }
                )
        splits = pl.DataFrame(split_rows)
        print("\nSplits: slope above each series' own knee, refitted per group (mean over series)")
        print(splits)
        splits.write_parquet(RESULTS_DIR / "splits.parquet")

        # Temperature at fixed clear-sky index: terciles within each series and 0.1-wide
        # clear-sky-index bin, pooled by sky class, then a slope adjusted for the index.
        temperature_rows = []
        banded = (
            cases.with_columns(
                csi_bin=(pl.col("clear_sky_index") * 10).floor().clip(0, 9),
                sky=pl.when(pl.col("clear_sky_index") >= SUNNY)
                .then(pl.lit("sunny"))
                .when(pl.col("clear_sky_index") < CLOUDY)
                .then(pl.lit("cloudy"))
                .otherwise(pl.lit("middle")),
            )
            .with_columns(
                rank=pl.col("temperature_2m").rank("average").over("site", "csi_bin"),
                size=pl.len().over("site", "csi_bin"),
            )
            .with_columns(
                tercile=((pl.col("rank") - 1) / pl.col("size") * 3)
                .floor()
                .clip(0, 2)
                .cast(pl.Int32),
                e_centred=pl.col("e") - pl.col("e").mean().over("site", "csi_bin"),
            )
        )
        for sky in ("cloudy", "middle", "sunny"):
            for tercile in (0, 1, 2):
                sub = banded.filter((pl.col("sky") == sky) & (pl.col("tercile") == tercile))
                mean, lower, upper, n_blocks = block_bootstrap(
                    values=sub["e_centred"].to_numpy(), blocks=sub["week"].to_numpy()
                )
                temperature_rows.append(
                    {
                        "sky": sky,
                        "temperature_tercile": tercile,
                        "mean_e_within_bin": mean,
                        "lower": lower,
                        "upper": upper,
                        "rows": sub.height,
                        "blocks": n_blocks,
                    }
                )
        temperature = pl.DataFrame(temperature_rows)
        print("\nMean e minus its series-and-index-bin mean, by 2 m temperature tercile within bin")
        print(temperature)
        temperature.write_parquet(RESULTS_DIR / "temperature.parquet")
        sunny = banded.filter(pl.col("sky") == "sunny")
        weeks = sunny["week"].unique().sort().to_numpy()
        index_by_week = {w: np.flatnonzero(sunny["week"].to_numpy() == w) for w in weeks}
        temperature_c = sunny["temperature_2m"].to_numpy()
        e_c = sunny["e_centred"].to_numpy()
        temperature_c = (
            temperature_c
            - sunny.with_columns(m=pl.col("temperature_2m").mean().over("site", "csi_bin"))[
                "m"
            ].to_numpy()
        )

        def temperature_slope(idx: np.ndarray) -> float:
            return float((temperature_c[idx] * e_c[idx]).sum() / (temperature_c[idx] ** 2).sum())

        draws = [
            temperature_slope(
                np.concatenate([index_by_week[w] for w in rng.choice(weeks, weeks.size)])
            )
            for _ in range(REFIT_RESAMPLES)
        ]
        lower, upper = np.percentile(draws, [2.5, 97.5])
        print(
            "Sunny rows: slope of e on temperature within series and index bin (points per K):",
            temperature_slope(np.arange(sunny.height)),
            lower,
            upper,
        )

        # e by clear-sky-index bin, for the first chart and a plain look at the shape.
        binned = (
            cases.with_columns(csi_bin=(pl.col("clear_sky_index") * 10).floor() / 10)
            .group_by("csi_bin")
            .agg(
                e=pl.col("e").mean(),
                r=pl.col("r").mean(),
                shift=pl.col("shift").mean(),
                rows=pl.len(),
            )
            .sort("csi_bin")
        )
        print("\nMean e, r, and shift by clear-sky-index bin")
        print(binned)
    cases.write_parquet(RESULTS_DIR / "cases.parquet")


if __name__ == "__main__":
    main()
