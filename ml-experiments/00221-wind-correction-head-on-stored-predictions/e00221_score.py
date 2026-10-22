# ruff: noqa: E501,PLR0912
"""Score the arms: reproduction of the stored best guess, absolute scores, and planned contrasts.

Reads the arms `e00221_heads.py` wrote, prints every number the write-up quotes, and writes
`work/results/results.json`. Scores are in points of capacity (quantiles are fractions of
capacity, multiplied by 100).

    uv run --no-sync python ml-experiments/00221-wind-correction-head-on-stored-predictions/e00221_score.py
"""

import json
from typing import Any, Final

import e00221_common as common
import numpy as np
import polars as pl
from e00221_common import LEVEL_COLUMNS
from e00221_heads import ARMS_DIR
from e00221_inputs import OUT_PATH

from studies import bootstrap, experiment_store

RESULTS_DIR = common.WORK_DIR / "results"
KEYS: Final[tuple[str, ...]] = (*common.KEY_COLUMNS, "seed")
POINTS: Final[float] = 100.0
LEVEL_DECISIVE: Final[float] = 95.0
N_RESAMPLES: Final[int] = 50_000
N_RESAMPLES_DESCRIPTIVE: Final[int] = 10_000
THRESHOLD: Final[float] = 0.10
BANDS: Final[tuple[str, ...]] = tuple(name for name, _, _ in common.LEAD_DAY_BANDS)
PRIMARY_BAND: Final[int] = 1
SECOND_BAND: Final[int] = 2
STORED_52: Final[dict[str, float]] = {"days 1-2": 6.701, "all": 11.037}
"""Experiment 52's printed best-guess scores on all rows, for the reproduction stop rule."""
ARM_NAMES: Final[tuple[str, ...]] = (
    "A0",
    "A1",
    "A2",
    "A3",
    "A4",
    "A4_mean_only",
    "A5",
    "A1_lofo",
    "A2_lofo",
    "A3_lofo",
)
ATOM_MASSES: Final = common.ATOM_MASSES


def band_index(*, lead_day: np.ndarray) -> np.ndarray:
    """Return the index into `BANDS` of each lead day."""
    out = np.full(lead_day.shape, -1, dtype=np.int64)
    for index, (_, low, high) in enumerate(common.LEAD_DAY_BANDS):
        out[(lead_day >= low) & (lead_day <= high)] = index
    return out


def losses(*, frame: pl.DataFrame) -> pl.DataFrame:
    """Return per-case losses in points of capacity, plus coverage and tail indicators."""
    quantiles = frame.select(LEVEL_COLUMNS).to_numpy().astype(np.float64)
    actual = frame["actual_norm"].to_numpy().astype(np.float64)
    difference = actual[:, None] - quantiles
    pinball = np.where(
        difference >= 0, common.LEVELS * difference, (common.LEVELS - 1) * difference
    )
    mean_forecast = quantiles @ ATOM_MASSES
    return frame.select(*KEYS, "lead_day", "site").with_columns(
        band=pl.Series(band_index(lead_day=frame["lead_day"].to_numpy())),
        qs=pl.Series(POINTS * 2.0 * (pinball * ATOM_MASSES).sum(axis=1)),
        pinball_05=pl.Series(POINTS * pinball[:, common.LOWER_INDEX]),
        pinball_95=pl.Series(POINTS * pinball[:, common.UPPER_INDEX]),
        covered=pl.Series(common.covered(quantiles=quantiles, actual=actual).astype(np.float64)),
        below_05=pl.Series((actual < quantiles[:, common.LOWER_INDEX]).astype(np.float64)),
        above_95=pl.Series((actual > quantiles[:, common.UPPER_INDEX]).astype(np.float64)),
        error_median=pl.Series(POINTS * np.abs(actual - quantiles[:, common.MEDIAN_INDEX])),
        error_mean=pl.Series(POINTS * np.abs(actual - mean_forecast)),
        median_minus_mean=pl.Series(POINTS * (quantiles[:, common.MEDIAN_INDEX] - mean_forecast)),
    )


def read_arm(*, name: str) -> pl.DataFrame:
    """Return one arm's cases with their losses (all seeds)."""
    frames = [pl.read_parquet(path) for path in sorted(ARMS_DIR.glob(f"{name}_seed*.parquet"))]
    if len(frames) != len(common.SEEDS):
        msg = f"arm {name} has {len(frames)} seed files"
        raise ValueError(msg)
    return losses(frame=pl.concat(frames).sort(*KEYS))


def select(
    *, frame: pl.DataFrame, band: int | None = None, site: str | None = None
) -> pl.DataFrame:
    """Return the cases of a lead-day band and a farm."""
    if band is not None:
        frame = frame.filter(pl.col("band") == band)
    if site is not None:
        frame = frame.filter(pl.col("site") == site)
    return frame


def contrast(
    *,
    arms: dict[str, pl.DataFrame],
    treatment: str,
    reference: str,
    metric: str = "qs",
    band: int | None = None,
    site: str | None = None,
    n: int = N_RESAMPLES_DESCRIPTIVE,
    level: float = LEVEL_DECISIVE,
) -> dict[str, Any]:
    """Return the difference treatment minus reference in one metric with its interval."""
    scoped = {
        name: select(frame=arms[name], band=band, site=site).select(*KEYS, metric)
        for name in (treatment, reference)
    }
    pairing = experiment_store.losses_for_bootstrap(arm_losses=scoped)
    differences, _ = bootstrap.paired_differences(
        losses=pairing, treatment=treatment, reference=reference, metric=metric
    )
    lower, upper = bootstrap.bootstrap_difference_at_level(
        losses=pairing,
        treatment=treatment,
        reference=reference,
        metric=metric,
        level=level,
        n_resamples=n,
    )
    return {
        "treatment": treatment,
        "reference": reference,
        "metric": metric,
        "band": "all" if band is None else BANDS[band],
        "site": site,
        "difference": float(differences.mean()),
        "lower": float(lower),
        "upper": float(upper),
        "cases": int(differences.size),
    }


def verdict(*, row: dict[str, Any]) -> str:
    """Return the pre-registered label for C1 against the 0.1-point threshold."""
    if row["upper"] < -THRESHOLD:
        return "supported"
    if row["upper"] < 0:
        return "small gain"
    if row["lower"] > 0:
        return "refuted"
    return "inconclusive"


def absolute_rows(*, arms: dict[str, pl.DataFrame]) -> list[dict[str, Any]]:
    """Return each arm's mean scores by lead-day band, over all leads, and per farm."""
    rows = []
    for name, frame in arms.items():
        scopes: list[tuple[str, str | None, pl.DataFrame]] = [("all", None, frame)]
        scopes += [
            (label, None, frame.filter(pl.col("band") == i)) for i, label in enumerate(BANDS)
        ]
        scopes += [
            (
                f"{label} {site}",
                site,
                frame.filter((pl.col("band") == i) & (pl.col("site") == site)),
            )
            for i, label in enumerate(BANDS)
            if i in (PRIMARY_BAND, SECOND_BAND)
            for site in sorted(frame["site"].unique().to_list())
        ]
        for scope, _, part in scopes:
            if part.height == 0:
                continue
            rows.append(
                {
                    "arm": name,
                    "scope": scope,
                    "cases": part.height,
                    **{
                        column: float(part[column].to_numpy().mean())
                        for column in (
                            "qs",
                            "pinball_05",
                            "pinball_95",
                            "covered",
                            "below_05",
                            "above_95",
                            "error_median",
                            "error_mean",
                        )
                    },
                }
            )
    return rows


def reproduction() -> dict[str, Any]:
    """Recompute the stored best guess on all rows and compare with experiment 52's scores."""
    table = pl.read_parquet(OUT_PATH).sort(*KEYS)
    per_case = losses(frame=table)
    out: dict[str, Any] = {}
    for label, frame in (
        ("days 1-2", per_case.filter(pl.col("band") == PRIMARY_BAND)),
        ("all", per_case),
    ):
        mean = float(frame["qs"].to_numpy().mean())
        out[label] = {
            "recomputed": mean,
            "stored_52": STORED_52[label],
            "gap": mean - STORED_52[label],
        }
    masked = table.filter(~pl.col("masked"))
    unmasked = losses(frame=masked.sort(*KEYS))
    out["unmasked_rows_all_months"] = {
        label: float(unmasked.filter(pl.col("band") == i)["qs"].to_numpy().mean())
        for i, label in enumerate(BANDS)
    }
    out["masked_share"] = float(table["masked"].to_numpy().mean())
    return out


def main() -> None:
    """Print and store the report."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    arms = {name: read_arm(name=name) for name in ARM_NAMES}
    results: dict[str, Any] = {"reproduction": reproduction()}
    results["absolute"] = absolute_rows(arms=arms)
    sites = sorted(arms["A0"]["site"].unique().to_list())
    primary = contrast(
        arms=arms,
        treatment="A2",
        reference="A0",
        band=PRIMARY_BAND,
        n=N_RESAMPLES,
        level=LEVEL_DECISIVE,
    )
    primary["verdict"] = verdict(row=primary)
    results["C1"] = primary
    pairs = {
        "C2": ("A3", "A0"),
        "C3": ("A2", "A3"),
        "C4": ("A1", "A0"),
        "C5": ("A4", "A2"),
        "C6": ("A5", "A0"),
        "A4_vs_A0": ("A4", "A0"),
        "A4_mean_only_vs_A0": ("A4_mean_only", "A0"),
        "A2_lofo_vs_A0": ("A2_lofo", "A0"),
        "A1_lofo_vs_A0": ("A1_lofo", "A0"),
        "A3_lofo_vs_A0": ("A3_lofo", "A0"),
        "A2_lofo_vs_A2": ("A2_lofo", "A2"),
        "A2_lofo_vs_A3_lofo": ("A2_lofo", "A3_lofo"),
    }
    contrasts = []
    for label, (treatment, reference) in pairs.items():
        for band in (None, 0, PRIMARY_BAND, SECOND_BAND, 3):
            row = contrast(arms=arms, treatment=treatment, reference=reference, band=band)
            contrasts.append({"label": label, **row})
    for treatment in ("A2", "A3", "A1", "A2_lofo"):
        for site in sites:
            for band in (PRIMARY_BAND, SECOND_BAND):
                row = contrast(arms=arms, treatment=treatment, reference="A0", band=band, site=site)
                contrasts.append({"label": f"{treatment}_vs_A0_by_site", **row})
    for treatment, reference in (("A2", "A0"), ("A3", "A0"), ("A1", "A0")):
        for metric in ("error_median", "error_mean", "pinball_05", "pinball_95"):
            for band in (PRIMARY_BAND, SECOND_BAND):
                row = contrast(
                    arms=arms, treatment=treatment, reference=reference, metric=metric, band=band
                )
                contrasts.append({"label": f"{treatment}_vs_{reference}_{metric}", **row})
    results["contrasts"] = contrasts
    c2 = next(c for c in contrasts if c["label"] == "C2" and c["band"] == BANDS[PRIMARY_BAND])
    results["blind_share_days_1_2"] = (
        c2["difference"] / primary["difference"] if primary["difference"] < 0 else None
    )
    scored = {
        name: {
            label: float(arms[name].filter(pl.col("band") == i)["qs"].to_numpy().mean())
            for i, label in ((PRIMARY_BAND, BANDS[PRIMARY_BAND]), (SECOND_BAND, BANDS[SECOND_BAND]))
        }
        for name in ("A2", "A3")
    }
    results["stop_rule_blind_beats_weather_both_bands"] = all(
        scored["A3"][label] < scored["A2"][label] for label in scored["A2"]
    )
    results["median_minus_mean_by_site"] = {
        name: {
            site: float(
                arms[name].filter(pl.col("site") == site)["median_minus_mean"].to_numpy().mean()
            )
            for site in sites
        }
        for name in ("A0", "A2", "A3")
    }
    (RESULTS_DIR / "results.json").write_text(json.dumps(results, indent=1))
    print(json.dumps(results["reproduction"], indent=1))
    print(json.dumps(results["C1"], indent=1))
    print("blind share days 1-2:", results["blind_share_days_1_2"])
    print(
        "stop rule, blind beats weather at both bands:",
        results["stop_rule_blind_beats_weather_both_bands"],
    )
    print(
        "\nArm, scope, cases, qs, pinball_05, pinball_95, covered, below_05, above_95, mae_med, mae_mean"
    )
    for row in results["absolute"]:
        if " " in row["scope"] and row["scope"].split()[-1].startswith("W"):
            continue
        print(
            f"{row['arm']:14s} {row['scope']:10s} {row['cases']:7d} {row['qs']:7.3f} "
            f"{row['pinball_05']:6.3f} {row['pinball_95']:6.3f} {row['covered']:.3f} "
            f"{row['below_05']:.3f} {row['above_95']:.3f} {row['error_median']:.3f} "
            f"{row['error_mean']:.3f}"
        )
    print("\nContrasts (treatment minus reference; negative is better)")
    for c in contrasts:
        if c["site"] is None and c["metric"] == "qs" and c["band"] in (BANDS[1], BANDS[2], "all"):
            print(
                f"{c['label']:22s} {c['treatment']:>13s}-{c['reference']:<5s} {c['band']:9s} "
                f"{c['difference']:+.3f} ({c['lower']:+.3f} to {c['upper']:+.3f}) n={c['cases']}"
            )
    print("\nBy site, days 1-2 and 3-6")
    for c in contrasts:
        if c["site"] is not None:
            print(
                f"{c['label']:22s} {c['site']} {c['band']:9s} {c['difference']:+.3f} "
                f"({c['lower']:+.3f} to {c['upper']:+.3f})"
            )
    print("\nTail and mean/median errors")
    for c in contrasts:
        if c["metric"] != "qs" and c["site"] is None:
            print(
                f"{c['label']:34s} {c['band']:9s} {c['difference']:+.3f} "
                f"({c['lower']:+.3f} to {c['upper']:+.3f})"
            )
    print(
        "\nMedian minus mean forecast by site (points):",
        json.dumps(results["median_minus_mean_by_site"]),
    )


if __name__ == "__main__":
    main()
