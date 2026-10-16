"""Score the arms of experiment 00054: absolute scores, contrasts with intervals, and the verdict.

Reads the arms `e00054_fit.py` wrote and prints every number the write-up quotes; it also writes
`work/results/results.json`. Scores are in points of capacity (quantiles are fractions of capacity,
multiplied by 100). A case's band is set by the hours from the 00 UTC start of the ENS run's date
to its valid time: lead day 0 (under 24 hours) is outside the contrast, and 24 to 54 hours is the
scored range of the contrast C1.

    uv run python ml-experiments/00054-product-trained-xgboost-wind-farms/e00054_score.py
"""

import json
from typing import Any, Final

import e00054_common as common
import numpy as np
import polars as pl
from e00054_common import LEVEL_COLUMNS

from studies import bootstrap, experiment_store

ARMS_DIR = common.WORK_DIR / "arms"
RESULTS_DIR = common.WORK_DIR / "results"
KEYS: Final[tuple[str, ...]] = (*common.KEY_COLUMNS, "seed")
POINTS: Final[float] = 100.0
LEVEL_DECISIVE: Final[float] = 95.0
N_RESAMPLES: Final[int] = 50_000
N_RESAMPLES_EXPLORATORY: Final[int] = 10_000
THRESHOLD: Final[float] = 0.15
BAND_NAMES: Final[tuple[str, str]] = ("lead day 0 (0-24 h)", "days 1-2 (24-54 h)")
SCORED_BAND: Final[int] = 1


def read_arm(*, name: str) -> pl.DataFrame:
    """Return one arm's per-case losses in points of capacity, with the band and month."""
    frame = pl.read_parquet(ARMS_DIR / f"{name}.parquet").sort(*KEYS)
    quantiles = frame.select(LEVEL_COLUMNS).to_numpy().astype(np.float64)
    actual = frame["actual_norm"].to_numpy().astype(np.float64)
    hours = frame.select(
        common.hours_after_run(
            valid_time=pl.col("valid_time"), power_fcst_init_time=pl.col("power_fcst_init_time")
        ).alias("hours")
    )["hours"]
    return frame.select(*KEYS).with_columns(
        band=(hours >= common.SCORED_FROM_HOURS).cast(pl.Int8),
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
    where: pl.Expr | None = None,
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
        if where is not None:
            frame = frame.filter(where)
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
        "band": "all" if band is None else BAND_NAMES[band],
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


def mean_of(series: pl.Series) -> float:
    """Return the mean of a numeric series as a float."""
    return float(series.to_numpy().mean())


def absolute_table(*, arms: dict[str, pl.DataFrame], sites: dict[int, str]) -> list[dict[str, Any]]:
    """Return each arm's mean score, coverage, and above-0.95 share by band and farm."""
    rows = []
    for name, frame in arms.items():
        scopes: list[tuple[str, pl.DataFrame]] = [("all", frame)]
        scopes += [(label, frame.filter(pl.col("band") == b)) for b, label in enumerate(BAND_NAMES)]
        scopes += [
            (
                f"{label} {site}",
                frame.filter((pl.col("time_series_id") == s) & (pl.col("band") == b)),
            )
            for s, site in sites.items()
            for b, label in enumerate(BAND_NAMES)
        ]
        for scope, part in scopes:
            rows.append(
                {
                    "arm": name,
                    "scope": scope,
                    "qs": mean_of(part["qs"]),
                    "covered": mean_of(part["covered"]),
                    "exceeds_95": mean_of(part["exceeds_95"]),
                    "cases": part.height,
                }
            )
    return rows


def exploratory_contrasts(
    *, arms: dict[str, pl.DataFrame], sites: dict[int, str]
) -> list[dict[str, Any]]:
    """Return the exploratory contrasts, each read at 95%."""
    upgrade = pl.col("power_fcst_init_time") >= common.UKV_UPGRADE
    first_month = pl.col("power_fcst_init_time").dt.strftime("%Y-%m") == "2025-07"
    pairs = [
        ("c", "a"),
        ("b_placebo", "a"),
        ("b", "b_placebo"),
        ("b", "ref_best_guess"),
        ("a", "ref_best_guess"),
        ("ref_n1", "a"),
    ]
    out = [
        contrast(arms=arms, treatment=t, reference=r, n=N_RESAMPLES_EXPLORATORY, band=b, level=95.0)
        for t, r in pairs
        for b in (SCORED_BAND, 0, None)
    ]
    for label, where in (
        ("before the UKV upgrade", ~upgrade),
        ("after the UKV upgrade", upgrade),
        ("without July 2025", ~first_month),
    ):
        row = contrast(
            arms=arms,
            treatment="b",
            reference="a",
            n=N_RESAMPLES_EXPLORATORY,
            band=SCORED_BAND,
            where=where,
            level=95.0,
        )
        out.append(row | {"split": label})
    out.extend(
        contrast(
            arms=arms,
            treatment="b",
            reference="a",
            n=N_RESAMPLES_EXPLORATORY,
            band=b,
            series=series,
            level=95.0,
        )
        | {"site": site}
        for series, site in sites.items()
        for b in (SCORED_BAND, 0)
    )
    return out


def main() -> None:
    """Print and save every number of the write-up."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    names = sorted(p.stem for p in ARMS_DIR.glob("*.parquet"))
    arms = {name: read_arm(name=name) for name in names}
    sites_frame = common.site_table()
    sites = dict(zip(sites_frame["time_series_id"], sites_frame["site"], strict=True))
    out: dict[str, Any] = {"arms": names}
    out["absolute"] = absolute_table(arms=arms, sites=sites)
    out["c1"] = contrast(arms=arms, treatment="b", reference="a", n=N_RESAMPLES, band=SCORED_BAND)
    out["c1"]["verdict"] = verdict(row=out["c1"])
    out["exploratory"] = exploratory_contrasts(arms=arms, sites=sites)
    checks = common.WORK_DIR / "ukv_checks.json"
    out["checks"] = json.loads(checks.read_text()) if checks.exists() else {}
    (RESULTS_DIR / "results.json").write_text(json.dumps(out, indent=1, default=str))
    print(f"C1 {out['c1']}")
    print("Absolute scores:")
    for row in out["absolute"]:
        scope = row["scope"]
        for site in sites.values():
            scope = scope.replace(f" {site}", f" [{site}]")
        print(
            f"  {row['arm']:<16}{scope:<32}{row['qs']:.3f}"
            f"  cover {row['covered']:.3f}  >q95 {row['exceeds_95']:.3f}  n={row['cases']}"
        )
    print("Exploratory contrasts (95%):")
    for row in out["exploratory"]:
        tag = row.get("split") or (f"[{row['site']}]" if "site" in row else "")
        print(
            f"  {row['treatment']:<12}minus {row['reference']:<15}{row['band']:<22}{tag:<24}"
            f"{row['difference']:+.3f} [{row['lower']:+.3f}, {row['upper']:+.3f}] n={row['cases']}"
        )
    print("Checks:", json.dumps(out["checks"], default=str))


if __name__ == "__main__":
    main()
