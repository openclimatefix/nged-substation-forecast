"""Score every arm of experiment 00034 and read the planned contrasts with their intervals.

Reads `work/metrics/<arm>.parquet` (see `e00034_metrics.py`) and writes
`work/results/absolute.parquet` (each arm's mean metrics over its own cases, per group and scope,
with month-block bootstrap intervals on the quantile score) and `work/results/contrasts.parquet`.
The eight planned contrasts C1 to C8 are read at 99.375% (Bonferroni across eight) and at 95%;
every other contrast (`X`, the exploratory and rank-test contrasts) is read at 95% only. A
contrast keeps the cases its two arms share and the seeds both hold.

    uv run python ml-experiments/00034-*/e00034_stats.py
"""

import argparse
from pathlib import Path
from typing import Final

import e00034_common as common
import polars as pl

from studies import bootstrap, experiment_store

METRICS_DIR: Final[Path] = common.WORK_DIR / "metrics"
RESULTS_DIR: Final[Path] = common.WORK_DIR / "results"
MEAN_METRICS: Final[tuple[str, ...]] = ("qs", "mae", "tw_p99", "ex90", "ex95", "cov90")
SCOPES: Final[tuple[tuple[str, float, float], ...]] = (
    ("all leads", 0.0, 10_000.0),
    *common.LEAD_BANDS,
)
N_RESAMPLES: Final[int] = 20_000
PLANNED_LEVEL: Final[float] = 100.0 * (1.0 - 0.05 / 8)
SKIP_ABSOLUTE: Final[frozenset[str]] = frozenset(
    {"production_c", "production_cs", "climatology_c", "climatology_cs", "manual_heuristic_c"}
)
"""Reference arms whose forms add nothing to the page."""
GROUPS: Final[tuple[str, ...]] = ("Primary", "BSP", "GSP")

# id: (group, treatment, reference)
PLANNED: Final[dict[str, tuple[str, str, str]]] = {
    "C1": ("Primary", "monthly_s", "n1_cs"),
    "C2": ("Primary", "monthly_cs", "monthly_s"),
    "C3": ("Primary", "weekly_s", "monthly_s"),
    "C4": ("Primary", "monthly_s", "yearly_s"),
    "C5": ("Primary", "monthly_s", "three_a_year_s"),
    "C6": ("Primary", "monthly_s", "monthly_2y_s"),
    "C7": ("Primary", "weekly_cs", "weekly_s"),
    "C8": ("Primary", "quarterly_s", "monthly_s"),
}
EXPLORATORY: Final[dict[str, tuple[str, str, str]]] = {
    "X_daily_vs_weekly": ("Primary", "daily_s", "weekly_s"),
    "X_daily_vs_monthly": ("Primary", "daily_s", "monthly_s"),
    "X_daily_correction": ("Primary", "daily_cs", "daily_s"),
    "X_warm_daily_vs_weekly": ("Primary", "warm20_daily_s", "warm20_weekly_s"),
    "X_warm_daily_vs_monthly": ("Primary", "warm20_daily_s", "warm20_monthly_s"),
    "X_warm_daily_correction": ("Primary", "warm20_daily_cs", "warm20_daily_s"),
    **{
        f"X_stack_{arm}_{form}": ("Primary", f"{arm}_{form}", "n1_cs")
        for arm in ("yearly", "half_yearly", "quarterly", "monthly", "weekly")
        for form in ("s", "cs")
    },
    "X_half_yearly": ("Primary", "half_yearly_s", "yearly_s"),
    "X_quarterly_vs_half": ("Primary", "quarterly_s", "half_yearly_s"),
    "X_monthly_vs_frozen_22": ("Primary", "monthly", "w22_ens"),
    "X_monthly_vs_n1": ("Primary", "monthly_s", "n1"),
    "X_bsp_monthly_vs_frozen": ("BSP", "monthly_s", "yearly_s"),
    "X_gsp_monthly_vs_frozen": ("GSP", "monthly_s", "yearly_s"),
    "X_bsp_weekly_vs_monthly": ("BSP", "weekly_s", "monthly_s"),
    "X_gsp_weekly_vs_monthly": ("GSP", "weekly_s", "monthly_s"),
    "X_bsp_correction": ("BSP", "monthly_cs", "monthly_s"),
    "X_gsp_correction": ("GSP", "monthly_cs", "monthly_s"),
}
PROXIES: Final[tuple[str, ...]] = (
    "three_a_year",
    "interp",
    "roll12",
    "warm20_monthly",
    "warm50_monthly",
    "warm20_weekly",
    "warm50_weekly",
)
"""Proxy arm names; each is read in its `_cs` form for the rank test (R3, R2) and `_s` for R1."""
ALL_SEEDS: Final[frozenset[str]] = frozenset({"C1", "C2"})
"""The contrasts read on every seed the two arms share; every other contrast uses seed 0."""


def rank_contrasts() -> dict[str, tuple[str, str, str]]:
    """Return the rank-agreement contrasts for the full monthly regime and every proxy.

    R1 is the correction contrast (`_cs` minus `_s`) of each regime. R2 and R3 compare an arm of
    experiments 30 and 24 with the per-series arm of the same regime; the retrained arm names are
    `a7_<regime>_cs` and `type_id_<regime>_cs`.
    """
    out: dict[str, tuple[str, str, str]] = {}
    for regime in ("monthly", *PROXIES):
        out[f"R1_{regime}"] = ("Primary", f"{regime}_cs", f"{regime}_s")
    for regime in ("monthly", "three_a_year", "roll12"):
        out[f"R2_{regime}"] = ("Primary", f"a7_{regime}_cs", f"{regime}_cs")
        out[f"R3_{regime}"] = ("Primary", f"type_id_{regime}_cs", f"{regime}_cs")
    out["R2_frozen"] = ("Primary", "a7_cs", "n1_cs")
    out["R3_frozen"] = ("Primary", "type_id_cs", "own_cs")
    return out


def read_arm(*, group: str, arm: str) -> pl.DataFrame | None:
    """Return one arm's per-case metrics for a group, or `None` when the arm is missing."""
    path = METRICS_DIR / f"{arm}.parquet"
    if not path.exists():
        return None
    return pl.read_parquet(path).filter(pl.col("group") == group)


def contrast(
    *, contrast_id: str, spec: tuple[str, str, str], planned: bool
) -> list[dict[str, object]]:
    """Return one contrast's rows: the quantile score per scope, and the other metrics overall."""
    group, treatment, reference = spec
    frames = {arm: read_arm(group=group, arm=arm) for arm in (treatment, reference)}
    if any(f is None or f.is_empty() for f in frames.values()):
        return []
    seeds = set.intersection(
        *(set(f["seed"].unique().to_list()) for f in frames.values() if f is not None)
    )
    if contrast_id not in ALL_SEEDS:
        seeds &= {0}
    if not seeds:
        return []
    kept = {
        arm: f.filter(pl.col("seed").is_in(sorted(seeds)))
        for arm, f in frames.items()
        if f is not None
    }
    shared = (
        kept[treatment]
        .select(*common.KEYS)
        .unique()
        .join(kept[reference].select(*common.KEYS).unique(), on=list(common.KEYS), how="inner")
    )
    kept = {arm: f.join(shared, on=list(common.KEYS), how="semi") for arm, f in kept.items()}
    rows: list[dict[str, object]] = []
    for scope, low, high in SCOPES:
        scoped = {
            arm: f.filter((pl.col("lead_hours") >= low) & (pl.col("lead_hours") < high))
            for arm, f in kept.items()
        }
        losses = experiment_store.losses_for_bootstrap(arm_losses=scoped)
        metrics = ("qs", "mae", "tw_p99") if scope == "all leads" else ("qs",)
        levels = (
            (("planned", PLANNED_LEVEL), ("95", 95.0))
            if planned and scope == "all leads"
            else (("95", 95.0),)
        )
        for metric in metrics:
            differences, _months = bootstrap.paired_differences(
                losses=losses, treatment=treatment, reference=reference, metric=metric
            )
            for label, level in levels:
                lower, upper = bootstrap.bootstrap_difference_at_level(
                    losses=losses,
                    treatment=treatment,
                    reference=reference,
                    metric=metric,
                    level=level,
                    n_resamples=N_RESAMPLES,
                )
                rows.append(
                    {
                        "id": contrast_id,
                        "group": group,
                        "treatment": treatment,
                        "reference": reference,
                        "scope": scope,
                        "metric": metric,
                        "level_label": label,
                        "level": level,
                        "difference": float(differences.mean()),
                        "lower": lower,
                        "upper": upper,
                        "cases": int(differences.size),
                        "seeds": len(seeds),
                    }
                )
    return rows


def absolute_rows() -> list[dict[str, object]]:
    """Return every arm's mean metrics per group and scope, with intervals on the headline three."""
    rows = []
    for path in sorted(METRICS_DIR.glob("*.parquet")):
        if path.stem in SKIP_ABSOLUTE:
            continue
        frame = pl.read_parquet(path)
        for group in GROUPS:
            part_group = frame.filter(pl.col("group") == group)
            if part_group.is_empty():
                continue
            for scope, low, high in SCOPES:
                part = part_group.filter(
                    (pl.col("lead_hours") >= low) & (pl.col("lead_hours") < high)
                )
                row: dict[str, object] = {
                    "group": group,
                    "arm": path.stem,
                    "scope": scope,
                    "cases": part.height // max(1, part["seed"].n_unique()),
                    "seeds": part["seed"].n_unique(),
                    **{m: float(part[m].mean()) for m in MEAN_METRICS},  # ty: ignore[invalid-argument-type]
                }
                if scope == "all leads":
                    losses = experiment_store.losses_for_bootstrap(arm_losses={path.stem: part})
                    interval = bootstrap.bootstrap_absolute(
                        losses=losses, arm=path.stem, metric="qs"
                    )
                    row["qs_lower"] = interval["lower_95"]
                    row["qs_upper"] = interval["upper_95"]
                rows.append(row)
    return rows


def main() -> None:
    """Write the absolute and contrast tables."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--parts",
        nargs="+",
        choices=("absolute", "planned", "exploratory", "rank"),
        default=["absolute", "planned", "exploratory", "rank"],
    )
    parser.add_argument("--ids", nargs="+", help="read only these contrast ids")
    args = parser.parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if "absolute" in args.parts:
        pl.DataFrame(absolute_rows(), infer_schema_length=None).write_parquet(
            RESULTS_DIR / "absolute.parquet"
        )
    rows = []
    for family, specs, planned in (
        ("planned", PLANNED, True),
        ("exploratory", EXPLORATORY, False),
        ("rank", rank_contrasts(), False),
    ):
        if family not in args.parts:
            continue
        for contrast_id, spec in specs.items():
            if args.ids and contrast_id not in args.ids:
                continue
            rows += [
                {**r, "family": family}
                for r in contrast(contrast_id=contrast_id, spec=spec, planned=planned)
            ]
            print(f"{contrast_id} done", flush=True)
    result = pl.DataFrame(rows, infer_schema_length=None)
    result.write_parquet(
        RESULTS_DIR / f"contrasts_{'_'.join(args.parts)}{'_ids' if args.ids else ''}.parquet"
    )
    with pl.Config(tbl_rows=200, tbl_cols=12, fmt_str_lengths=40):
        print(
            result.filter((pl.col("scope") == "all leads") & (pl.col("metric") == "qs")).select(
                "id", "treatment", "reference", "level_label", "difference", "lower", "upper"
            )
        )


if __name__ == "__main__":
    main()
