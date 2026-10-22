# ruff: noqa: DTZ001, E501, PLR0915
"""D8's summer look, spike classification, correlation and cooling hinge on experiment 159's panel.

The panel (experiment 130's residual panel, stored by the false-alarm analysis) holds, per
half-hour from 2022-01-01, D8's metered power, the convex demand model's expected power (fitted
on a 24-month window and refitted on the 1st and 16th of each month, so each residual is
out-of-sample up to the next refit), and ERA5 temperature. The temperature is past weather, so
this is a lab shortcut. Writes `d8_results.json` and figures; prints every quoted number.

    OMP_NUM_THREADS=1 systemd-run --user --scope -p MemoryMax=8G -p OOMPolicy=continue \
        uv run --no-sync python ml-experiments/00198-*/e00198_d8.py
"""

import pickle
from datetime import datetime, timedelta
from typing import Final

import altair as alt
import e00198_common as common
import numpy as np
import polars as pl
from plotting import ocf_theme as ocf
from scipy.stats import rankdata

BREAK: Final[datetime] = datetime(2023, 8, 15)
BEFORE_FIRST: Final[datetime] = datetime(2022, 6, 1)
DEPARTURE_MW: Final[float] = 1.5
N_BOOTSTRAP: Final[int] = 20000
WIDTH: Final[int] = 680
SEED: Final[int] = 198


def load_panel() -> pl.DataFrame:
    """Return D8's half-hourly panel as a frame with calendar columns."""
    with common.PANEL_PATH.open("rb") as handle:
        panel = pickle.load(handle)[common.D8_SERIES]
    n = len(panel["power"])
    start = common.PANEL_START.replace(tzinfo=None)
    return pl.DataFrame(
        {
            "t": [start + timedelta(minutes=30 * (i + 1)) for i in range(n)],
            "power": panel["power"],
            "expected": panel["expected"],
            "resid": panel["resid"],
            "temp": panel["temp"],
            "reason": panel["reason"],
        }
    ).with_columns(
        hour=pl.col("t").dt.hour() + pl.col("t").dt.minute() / 60.0,
        month=pl.col("t").dt.month(),
        year=pl.col("t").dt.year(),
        date=pl.col("t").dt.date(),
        doy=pl.col("t").dt.ordinal_day(),
    )


def partial_spearman(*, x: np.ndarray, y: np.ndarray, control: np.ndarray) -> float:
    """Return the Spearman correlation of x and y after removing the rank of `control`."""
    basis = np.stack([np.ones(len(x)), rankdata(control)], axis=1)
    rx, ry = rankdata(x), rankdata(y)
    ex = rx - basis @ np.linalg.lstsq(basis, rx, rcond=None)[0]
    ey = ry - basis @ np.linalg.lstsq(basis, ry, rcond=None)[0]
    return float(np.corrcoef(ex, ey)[0, 1])


def median_fit(*, x: np.ndarray, y: np.ndarray, iterations: int = 60) -> np.ndarray:
    """Return a ridge-regularised median regression by iteratively reweighted least squares."""
    weights = np.ones(len(y))
    coef = np.zeros(x.shape[1])
    for _ in range(iterations):
        gram = (x * weights[:, None]).T @ x + common.RIDGE * np.eye(x.shape[1])
        coef = np.linalg.solve(gram, (x * weights[:, None]).T @ y)
        weights = 1.0 / np.maximum(np.abs(y - x @ coef), 1e-3)
    return coef


def band_design(*, hour: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Return the hinge times a one-hot of the 4-hour band of the hour of day."""
    out = np.zeros((len(h), common.N_BANDS))
    out[np.arange(len(h)), np.minimum((hour // 4).astype(int), common.N_BANDS - 1)] = h
    return out


def bootstrap_mean(*, diff: np.ndarray, block: np.ndarray) -> list[float]:
    """Return the mean and 99% interval of `diff`, resampling 14-day blocks."""
    ids, inverse = np.unique(block, return_inverse=True)
    sums, counts = np.bincount(inverse, weights=diff), np.bincount(inverse).astype(float)
    draws = np.random.default_rng(SEED).integers(0, len(ids), size=(N_BOOTSTRAP, len(ids)))
    means = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
    low, high = np.quantile(means, [0.005, 0.995])
    return [float(diff.mean()), float(low), float(high)]


def look(*, frame: pl.DataFrame) -> dict:
    """Print and return the descriptive numbers of the look-first step and draw its figures."""
    summer = frame.filter(pl.col("month").is_between(6, 9) & pl.col("resid").is_not_nan())
    big = summer.with_columns(
        kind=pl.when(pl.col("resid") <= -DEPARTURE_MW)
        .then(pl.lit("downward"))
        .when(pl.col("resid") >= DEPARTURE_MW)
        .then(pl.lit("upward"))
        .otherwise(pl.lit("none")),
        artefact=pl.col("reason") != "",
    )
    out: dict = {}
    for year in range(2022, 2027):
        sub = big.filter(pl.col("year") == year)
        n = sub.height
        out[str(year)] = {
            "half_hours": n,
            "downward_share": float((sub["kind"] == "downward").mean()),  # ty: ignore[invalid-argument-type]
            "upward_share": float((sub["kind"] == "upward").mean()),  # ty: ignore[invalid-argument-type]
            "artefact_share": float(sub["artefact"].mean()),  # ty: ignore[invalid-argument-type]
            "mean_resid": float(sub["resid"].mean()),  # ty: ignore[invalid-argument-type]
            "upward_max_mw": float(sub.filter(pl.col("kind") == "upward")["resid"].max() or 0.0),  # ty: ignore[invalid-argument-type]
            "downward_min_mw": float(
                sub.filter(pl.col("kind") == "downward")["resid"].min() or 0.0  # ty: ignore[invalid-argument-type]
            ),
        }
    print("Summer (June to September) departures from expected power, by year:")
    for year, row in out.items():
        print(year, {k: round(v, 3) if isinstance(v, float) else v for k, v in row.items()})
    ups = big.filter(pl.col("kind") == "upward")
    downs = big.filter(pl.col("kind") == "downward")
    out["upward_hour_median"] = float(ups["hour"].median() or float("nan"))  # ty: ignore[invalid-argument-type]
    out["downward_hour_median"] = float(downs["hour"].median() or float("nan"))  # ty: ignore[invalid-argument-type]
    out["upward_temp_mean"] = float(ups["temp"].mean() or float("nan"))  # ty: ignore[invalid-argument-type]
    out["downward_temp_mean"] = float(downs["temp"].mean() or float("nan"))  # ty: ignore[invalid-argument-type]
    out["all_summer_temp_mean"] = float(big["temp"].mean())  # ty: ignore[invalid-argument-type]
    print(
        "median hour of upward/downward departures:",
        out["upward_hour_median"],
        out["downward_hour_median"],
    )
    print(
        "mean temperature at upward/downward/all summer half-hours:",
        out["upward_temp_mean"],
        out["downward_temp_mean"],
        out["all_summer_temp_mean"],
    )
    # Figures.
    alt.data_transformers.disable_max_rows()
    daily = (
        summer.group_by("date")
        .agg(
            pl.col("power").max().alias("day_max_power"),
            pl.col("temp").max().alias("day_max_temp"),
            pl.col("year").first().alias("year"),
            pl.col("t").first().alias("t0"),
        )
        .with_columns(
            period=pl.when(pl.col("t0") < BREAK)
            .then(pl.lit("before 2023-08-15"))
            .otherwise(pl.lit("after 2023-08-15"))
        )
        .drop_nulls()
    )
    chart1 = (
        alt.Chart(  # ty: ignore[unresolved-attribute]
            alt.Data(values=daily.select("day_max_temp", "day_max_power", "period").to_dicts())
        )
        .mark_circle(size=28, opacity=0.6)
        .encode(
            x=alt.X("day_max_temp:Q", title="Day's maximum ERA5 temperature (degrees Celsius)"),
            y=alt.Y("day_max_power:Q", title="Day's maximum metered power (MW)"),
            color=alt.Color(
                "period:N",
                title=None,
                scale=alt.Scale(range=[ocf.BRAND_ORANGE, ocf.DATA_BLUE]),
                legend=alt.Legend(orient="top"),
            ),
        )
        .properties(
            width=WIDTH - 90,
            height=260,
            title=alt.TitleParams(
                "D8's hottest days do not draw more power",
                subtitle=["One dot per day, June to September 2022 to 2026."],
                anchor="start",
            ),
        )
    )
    chart1.save(common.FIGURE_DIR / "figure_d8_power_vs_temperature.svg")
    binned = (
        summer.filter(pl.col("hour").is_between(12, 18))
        .with_columns(
            temp_bin=(pl.col("temp") // 2 * 2 + 1),
            period=pl.when(pl.col("t") < BREAK)
            .then(pl.lit("before 2023-08-15"))
            .otherwise(pl.lit("after 2023-08-15")),
        )
        .group_by("temp_bin", "period")
        .agg(pl.col("resid").mean().alias("mean_resid"), pl.len().alias("n"))
        .filter(pl.col("n") >= 40)
    )
    chart2 = (
        alt.Chart(alt.Data(values=binned.to_dicts()))  # ty: ignore[unresolved-attribute]
        .mark_line(point=True, strokeWidth=2)
        .encode(
            x=alt.X("temp_bin:Q", title="ERA5 temperature (degrees Celsius, 2-degree bins)"),
            y=alt.Y("mean_resid:Q", title="Mean metered minus expected power (MW)"),
            color=alt.Color(
                "period:N",
                title=None,
                scale=alt.Scale(range=[ocf.BRAND_ORANGE, ocf.DATA_BLUE]),
                legend=alt.Legend(orient="top"),
            ),
        )
        .properties(
            width=WIDTH - 90,
            height=240,
            title=alt.TitleParams(
                "Afternoon residual against temperature at D8",
                subtitle=[
                    "12:00 to 18:00 UTC, June to September, bins with 40 or more half-hours. A cooling load would rise to the right."
                ],
                anchor="start",
            ),
        )
    )
    (
        chart2
        + alt.Chart(alt.InlineData(values=[{"y": 0}]))  # ty: ignore[unresolved-attribute]
        .mark_rule(color="#888", strokeDash=[4, 3])
        .encode(y="y:Q")
    ).save(common.FIGURE_DIR / "figure_d8_residual_vs_temperature.svg")
    hist = pl.concat(
        [
            ups.with_columns(kind=pl.lit("upward (metered above expected)")),
            downs.with_columns(kind=pl.lit("downward (metered below expected)")),
        ]
    )
    hist = (
        hist.with_columns(hour_bin=(pl.col("hour") // 1).cast(pl.Int32))
        .group_by("hour_bin", "kind")
        .agg(pl.len().alias("n"))
    )
    chart3 = (
        alt.Chart(alt.Data(values=hist.to_dicts()))  # ty: ignore[unresolved-attribute]
        .mark_bar()
        .encode(
            x=alt.X("hour_bin:O", title="Hour of day (UTC)", axis=alt.Axis(values=[0, 6, 12, 18])),
            y=alt.Y("n:Q", title="Half-hours with a departure of 1.5 MW or more"),
            color=alt.Color(
                "kind:N",
                title=None,
                scale=alt.Scale(range=[ocf.BRAND_ORANGE, ocf.DATA_BLUE]),
                legend=alt.Legend(orient="top"),
            ),
            xOffset="kind:N",
        )
        .properties(
            width=WIDTH - 90,
            height=220,
            title=alt.TitleParams(
                "When D8's large summer departures happen",
                subtitle=["June to September 2022 to 2026."],
                anchor="start",
            ),
        )
    )
    chart3.save(common.FIGURE_DIR / "figure_d8_departure_hours.svg")
    return out


def correlation_and_fit(*, frame: pl.DataFrame) -> dict:
    """Return the pre-registered D8 correlation and the cooling-hinge fit and score."""
    out: dict = {}
    hour = frame["hour"].to_numpy()
    date = frame["date"].to_numpy().astype("datetime64[D]").astype(int)
    t = frame["t"]
    month = frame["month"].to_numpy()
    temp = frame["temp"].to_numpy()
    resid = frame["resid"].to_numpy()
    ok = np.isfinite(resid)
    summer = np.isin(month, [6, 7, 8, 9]) & ok
    pre = (t < BREAK).to_numpy() & (t >= BEFORE_FIRST).to_numpy()
    afternoon = (hour >= 12) & (hour < 18)
    day_max = (
        pl.DataFrame({"d": date, "temp": temp}).group_by("d").agg(pl.col("temp").max().alias("dm"))
    )
    warm = np.isin(date, day_max.filter(pl.col("dm") >= 22.0)["d"].to_numpy())
    rows = summer & pre & afternoon & warm
    h20 = np.maximum(0.0, temp - 20.0)
    rho = partial_spearman(x=h20[rows], y=resid[rows], control=frame["doy"].to_numpy()[rows])
    out["prediction_1_partial_spearman"] = rho
    out["prediction_1_rows"] = int(rows.sum())
    print(
        f"D8 partial Spearman (afternoon residual vs hinge tau 20, warm days, before the break): {rho:+.3f} on {int(rows.sum())} half-hours"
    )
    # After the break, for comparison, with the same rule.
    post_rows = summer & ~pre & (t >= BREAK).to_numpy() & afternoon & warm
    out["post_break_partial_spearman"] = partial_spearman(
        x=h20[post_rows], y=resid[post_rows], control=frame["doy"].to_numpy()[post_rows]
    )
    print(
        "D8 same correlation after the break:",
        out["post_break_partial_spearman"],
        int(post_rows.sum()),
    )
    # Fit: summers 2022 and 2023 before the break; score the later summers.
    fit_mask = summer & ((t >= BEFORE_FIRST).to_numpy()) & (t < BREAK).to_numpy()
    score_mask = summer & (t >= BREAK).to_numpy()
    year = frame["year"].to_numpy()
    results = {}
    for tau in common.TAUS:
        h = np.maximum(0.0, temp - tau)
        days_above = len(set(date[fit_mask & (h > 0) & afternoon].tolist()))
        results[tau] = {"fit_days_above": days_above}
    eligible = [
        tau for tau in common.TAUS if results[tau]["fit_days_above"] >= common.MIN_DAYS_ABOVE_TAU
    ]
    print("D8 fit-summer afternoon days above tau:", results)
    if not eligible:
        out["fit"] = {"eligible": False}
        return out
    # Tau by leave-one-summer-out over 2022 and 2023-before-break.
    scores = {}
    for tau in eligible:
        h = np.maximum(0.0, temp - tau)
        x = band_design(hour=hour, h=h)
        diffs = []
        for held in (2022, 2023):
            train = fit_mask & (year != held)
            test = fit_mask & (year == held)
            coef = median_fit(x=x[train], y=resid[train])
            diffs.append(
                float(np.mean(np.abs(resid[test] - x[test] @ coef)) - np.mean(np.abs(resid[test])))
            )
        scores[tau] = float(np.mean(diffs))
    tau = min(scores, key=lambda k: scores[k])
    h = np.maximum(0.0, temp - tau)
    x = band_design(hour=hour, h=h)
    coef = median_fit(x=x[fit_mask], y=resid[fit_mask])
    base = np.abs(resid)
    new = np.abs(resid - x @ coef)
    block = (date - date[score_mask].min()) // 14
    gain = bootstrap_mean(diff=(new - base)[score_mask], block=block[score_mask])
    # Sensitivity: add Gaussian noise of ENS-versus-ERA5 size (std 1.2 C, assumed) to the scored temperature.
    noisy = np.maximum(0.0, temp + np.random.default_rng(SEED).normal(0, 1.2, size=len(temp)) - tau)
    new_noisy = np.abs(resid - band_design(hour=hour, h=noisy) @ coef)
    gain_noisy = bootstrap_mean(diff=(new_noisy - base)[score_mask], block=block[score_mask])
    out["fit"] = {
        "tau": tau,
        "held_out_scores": scores,
        "coef": coef.tolist(),
        "mae_change_mw": gain,
        "mae_change_noisy_mw": gain_noisy,
        "scored_half_hours": int(score_mask.sum()),
        "mae_before_mw": float(base[score_mask].mean()),
    }
    print("D8 cooling hinge:", out["fit"])
    return out


def main() -> None:
    """Run the D8 analysis."""
    common.FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    frame = load_panel()
    results = {"look": look(frame=frame), **correlation_and_fit(frame=frame)}
    common.write_json_atomic(path=common.WORK_DIR / "d8_results.json", content=results)


if __name__ == "__main__":
    main()
