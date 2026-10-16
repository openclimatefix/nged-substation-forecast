"""Score the arms: absolute scores, the planned contrasts with intervals, and the verdict.

Reads the arms `e00052_blend.py` wrote and prints every number the write-up quotes; it also writes
`work/results/results.json`. Scores are in points of capacity (quantiles are fractions of capacity,
multiplied by 100).

    uv run python ml-experiments/00052-more-nwp-products-for-wind-farms/e00052_score.py
"""

import json
from typing import Any, Final

import e00052_common as common
import numpy as np
import polars as pl
from e00052_common import LEVEL_COLUMNS

from studies import bootstrap, experiment_store

RESULTS_DIR = common.WORK_DIR / "results"
ARMS_DIR = common.WORK_DIR / "arms"
KEYS: Final[tuple[str, ...]] = (*common.KEY_COLUMNS, "seed")
POINTS: Final[float] = 100.0
LEVEL_DECISIVE: Final[float] = 95.0
N_RESAMPLES: Final[int] = 50_000
N_RESAMPLES_EXPLORATORY: Final[int] = 10_000
THRESHOLD: Final[float] = 0.15
BANDS: Final[tuple[str, ...]] = tuple(name for name, _, _ in common.LEAD_DAY_BANDS)
PRIMARY_BAND: Final[int] = 1
"""Index of "days 1-2" in `BANDS`."""
STOP_POINTS: Final[float] = 1.0
STOP_MONTHS: Final[tuple[str, ...]] = (
    "2025-07",
    "2025-08",
    "2025-09",
    "2025-10",
    "2025-11",
    "2025-12",
)


def read_arm(*, name: str) -> pl.DataFrame:
    """Return one arm's per-case losses in points of capacity."""
    frame = pl.read_parquet(ARMS_DIR / f"{name}.parquet").sort(*KEYS)
    quantiles = frame.select(LEVEL_COLUMNS).to_numpy().astype(np.float64)
    actual = frame["actual_norm"].to_numpy().astype(np.float64)
    return frame.select(*KEYS, "band", "lead_day").with_columns(
        qs=pl.Series(POINTS * common.quantile_score(quantiles=quantiles, actual=actual)),
        covered=pl.Series(common.covered(quantiles=quantiles, actual=actual).astype(np.float64)),
        exceeds_95=pl.Series((actual > quantiles[:, common.UPPER_INDEX]).astype(np.float64)),
    )


def contrast(
    *,
    arms: dict[str, pl.DataFrame],
    treatment: str,
    reference: str,
    n: int,
    band: int | None = None,
    series: int | None = None,
    months: tuple[str, ...] | None = None,
    level: float = LEVEL_DECISIVE,
) -> dict[str, Any]:
    """Return the quantile-score difference treatment minus reference with its interval."""
    scoped = {}
    for name in (treatment, reference):
        frame = arms[name]
        if band is not None:
            frame = frame.filter(pl.col("band") == band)
        if series is not None:
            frame = frame.filter(pl.col("time_series_id") == series)
        if months is not None:
            frame = frame.filter(
                pl.col("power_fcst_init_time").dt.strftime("%Y-%m").is_in(list(months))
            )
        scoped[name] = frame.select(*KEYS, "qs")
    losses = experiment_store.losses_for_bootstrap(arm_losses=scoped)
    differences, _ = bootstrap.paired_differences(
        losses=losses, treatment=treatment, reference=reference, metric="qs"
    )
    lower, upper = bootstrap.bootstrap_difference_at_level(
        losses=losses,
        treatment=treatment,
        reference=reference,
        metric="qs",
        level=level,
        n_resamples=n,
    )
    return {
        "treatment": treatment,
        "reference": reference,
        "band": "all" if band is None else BANDS[band],
        "series": series,
        "level": level,
        "difference": float(differences.mean()),
        "lower": float(lower),
        "upper": float(upper),
        "cases": int(differences.size),
    }


def verdict(*, row: dict[str, Any]) -> str:
    """Return the pre-registered label for a contrast against the 0.15-point threshold."""
    if row["upper"] < -THRESHOLD:
        return "supported"
    if row["upper"] < 0:
        return "small gain"
    if row["lower"] > 0:
        return "refuted"
    return "inconclusive"


def absolute_table(*, arms: dict[str, pl.DataFrame], sites: dict[int, str]) -> list[dict[str, Any]]:
    """Return each arm's mean quantile score by lead-day band and farm, plus coverage."""
    rows = []
    for name, frame in arms.items():
        rows.append(
            {
                "arm": name,
                "scope": "all",
                "qs": float(frame["qs"].mean()),
                "covered": float(frame["covered"].mean()),
                "exceeds_95": float(frame["exceeds_95"].mean()),
                "cases": frame.height,
            }
        )
        for band, label in enumerate(BANDS):
            part = frame.filter(pl.col("band") == band)
            if part.height:
                rows.append(
                    {
                        "arm": name,
                        "scope": label,
                        "qs": float(part["qs"].mean()),
                        "covered": float(part["covered"].mean()),
                        "exceeds_95": float(part["exceeds_95"].mean()),
                        "cases": part.height,
                    }
                )
        for series, label in sites.items():
            part = frame.filter(pl.col("time_series_id") == series)
            rows.append(
                {
                    "arm": name,
                    "scope": label,
                    "qs": float(part["qs"].mean()),
                    "covered": float(part["covered"].mean()),
                    "exceeds_95": float(part["exceeds_95"].mean()),
                    "cases": part.height,
                }
            )
    return rows


def main() -> None:
    """Print and save every number of the write-up."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    names = sorted(p.stem for p in ARMS_DIR.glob("*.parquet"))
    arms = {name: read_arm(name=name) for name in names}
    sites_frame = common.site_table()
    sites = dict(zip(sites_frame["time_series_id"], sites_frame["site"], strict=True))
    out: dict[str, Any] = {"arms": names}
    out["absolute"] = absolute_table(arms=arms, sites=sites)
    primary = "ens_icon_ifs"
    out["c1"] = contrast(
        arms=arms, treatment=primary, reference="ens", n=N_RESAMPLES, band=PRIMARY_BAND
    )
    out["c1"]["verdict"] = verdict(row=out["c1"])
    out["c2"] = contrast(arms=arms, treatment=primary, reference="ens", n=N_RESAMPLES)
    out["c2"]["verdict"] = verdict(row=out["c2"])
    exploratory = [
        contrast(arms=arms, treatment=name, reference="ens", n=N_RESAMPLES_EXPLORATORY, band=band)
        for name in names
        if name != "ens"
        for band in (None, *range(len(BANDS)))
    ]
    exploratory.extend(
        contrast(
            arms=arms,
            treatment=primary,
            reference="ens",
            n=N_RESAMPLES_EXPLORATORY,
            band=band,
            series=series,
        )
        | {"site": label}
        for series, label in sites.items()
        for band in (None, PRIMARY_BAND)
    )
    out["exploratory"] = exploratory
    out["stop_rule"] = {
        p: contrast(
            arms=arms,
            treatment=f"only_{p}_spread",
            reference="ens",
            n=N_RESAMPLES_EXPLORATORY,
            band=PRIMARY_BAND,
            months=STOP_MONTHS,
            level=95.0,
        )
        for p in common.PRODUCTS
    }
    for row in out["stop_rule"].values():
        row["dropped"] = bool(row["difference"] > STOP_POINTS)
    out["weights"] = {
        path.stem: pl.read_parquet(path).to_dicts()
        for path in common.WORK_DIR.glob("weights_*.parquet")
    }
    out["spread"] = {
        path.stem: pl.read_parquet(path).to_dicts()
        for path in common.WORK_DIR.glob("spread_*.parquet")
    }
    checks = common.WORK_DIR / "product_checks.json"
    out["product_checks"] = json.loads(checks.read_text()) if checks.exists() else {}
    (RESULTS_DIR / "results.json").write_text(json.dumps(out, indent=1, default=str))
    print(f"C1 {out['c1']}\nC2 {out['c2']}")
    print("Absolute (all cases):")
    for row in out["absolute"]:
        if row["scope"] in ("all", *BANDS):
            print(
                f"  {row['arm']:<24}{row['scope']:<10}{row['qs']:.3f}"
                f"  cover {row['covered']:.3f}  >q95 {row['exceeds_95']:.3f}"
            )
    print("Exploratory contrasts vs ens:")
    for row in exploratory:
        print(
            f"  {row['treatment']:<24}{row['band']:<10}{row['series']}"
            f"  {row['difference']:+.3f} [{row['lower']:+.3f}, {row['upper']:+.3f}]"
            f" n={row['cases']}"
        )
    print("Stop rule:")
    for p, row in out["stop_rule"].items():
        print(f"  {p}: {row['difference']:+.3f} dropped={row['dropped']}")


if __name__ == "__main__":
    main()
