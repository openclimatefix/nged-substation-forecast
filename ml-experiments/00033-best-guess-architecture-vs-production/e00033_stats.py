"""Score every arm of experiment 00033 and read the planned contrasts with their intervals.

Reads `work/metrics/<group>/<arm>.parquet` (see `e00033_metrics.py`) and writes
`work/results/absolute.parquet` (each arm's mean metrics and, on the quantile score, MAE of the
median, and twCRPS, month-block bootstrap intervals, over all leads and per lead band) and
`work/results/contrasts.parquet`. Decisive contrasts C1 to C3 are read at 98.33% (Bonferroni across
three); every other contrast is read at 5% divided by the count of its family, and also at 95% for
the lead bands.

    uv run python ml-experiments/00033-*/e00033_stats.py
"""

from pathlib import Path
from typing import Final

import e00033_common as common
import polars as pl

from studies import bootstrap, experiment_store

METRICS_DIR: Final[Path] = common.WORK_DIR / "metrics"
RESULTS_DIR: Final[Path] = common.WORK_DIR / "results"
MEAN_METRICS: Final[tuple[str, ...]] = (
    "qs",
    "mae",
    "tw_p99",
    "tw_p95",
    "brier_p99",
    "brier_p95",
    "ex90",
    "ex95",
    "cov90",
    "hit_p99",
    "alarm_p99",
    "false_alarm_p99",
    "event_p99",
    "hit_p95",
    "alarm_p95",
    "false_alarm_p95",
    "event_p95",
)
INTERVAL_METRICS: Final[tuple[str, ...]] = ("qs", "mae", "tw_p99", "tw_p95")
SCOPES: Final[tuple[tuple[str, float, float], ...]] = (
    ("all leads", 0.0, 10_000.0),
    *common.LEAD_BANDS,
)
N_RESAMPLES_DECISIVE: Final[int] = 20_000
N_RESAMPLES: Final[int] = 20_000
GROUPS: Final[tuple[str, ...]] = ("Primary", "BSP", "GSP", "solar", "wind")

# id: (family, group(s), treatment, reference)
CONTRASTS: Final[dict[str, tuple[str, tuple[str, ...], str, str]]] = {
    "C1": ("decisive", ("Primary",), "n1_cs", "production"),
    "C2": ("decisive", ("solar",), "bg_base_spread", "production"),
    "C3": ("decisive", ("wind",), "bg_full", "production"),
    "C4": ("Primary demand", ("Primary",), "n1_cs", "manual_heuristic"),
    "C5": ("Primary demand", ("Primary",), "n1_cs", "n1"),
    "C6": ("Primary demand", ("Primary",), "n1_cs", "climatology"),
    "C7": ("Primary demand", ("Primary",), "n1_c", "n1"),
    "C8": ("Primary demand", ("Primary",), "production_c", "production"),
    "C9": ("Primary demand", ("Primary",), "a7_c", "n1_c"),
    "C10": ("Primary demand", ("Primary",), "type_id_cs", "own_cs"),
    "C11": ("Primary demand", ("Primary",), "monthly_c", "n1_c"),
    "C12": ("Primary demand", ("Primary",), "monthly", "n1"),
    "C13": ("BSP and GSP", ("BSP", "GSP"), "n1", "production"),
    "C14": ("BSP and GSP", ("BSP", "GSP"), "n1", "manual_heuristic"),
    "C15": ("BSP and GSP", ("BSP", "GSP"), "n1", "climatology"),
    "C16": ("BSP and GSP", ("BSP", "GSP"), "n1_c", "n1"),
    "C17": ("generators", ("solar",), "bg_base_spread", "n1"),
    "C18": ("generators", ("solar",), "bg_base_spread", "manual_heuristic"),
    "C19": ("generators", ("solar",), "bg_base_spread", "climatology"),
    "C20": ("generators", ("wind",), "bg_full", "n1"),
    "C21": ("generators", ("wind",), "bg_full", "manual_heuristic"),
    "C22": ("generators", ("wind",), "bg_full", "climatology"),
    "C23": ("generators", ("solar",), "cams_spread", "bg_base_spread"),
    "C24": ("generators", ("wind",), "capacity_a4", "capacity_a0"),
    "C25": ("generators", ("solar",), "physics_hybrid", "bg_base"),
    "C26": ("generators", ("solar",), "physics_scipy", "bg_base"),
    "C27": ("weather ceiling", ("Primary",), "w22_era5_all_c", "w22_ens_c"),
    "C28": ("weather ceiling", ("solar",), "pw_era5_spread", "bg_base_spread"),
    "C28b": ("weather ceiling", ("solar",), "pw_cams_spread", "cams_spread"),
    "C29": ("weather ceiling", ("wind",), "pw_era5_spread", "bg_full"),
    "C30": ("weather ceiling", ("Primary",), "era5_retrained_c", "n1_refit_c"),
    "C31": ("weather ceiling", ("Primary",), "cams_shortwave_c", "w22_ens_c"),
    "C31b": ("weather ceiling", ("Primary",), "cams_shortwave_c", "w22_era5_all_c"),
    "C32": ("weather ceiling", ("BSP", "GSP"), "w22_era5_all", "w22_ens"),
}
SEED0_ONLY: Final[frozenset[str]] = frozenset({"C11", "C12"})


def read_arm(*, groups: tuple[str, ...], arm: str) -> pl.DataFrame:
    """Return one arm's per-case metrics for one or more groups."""
    return pl.concat([pl.read_parquet(METRICS_DIR / group / f"{arm}.parquet") for group in groups])


def in_scope(*, frame: pl.DataFrame, low: float, high: float) -> pl.DataFrame:
    """Return the rows whose lead is in `[low, high)` hours."""
    return frame.filter((pl.col("lead_hours") >= low) & (pl.col("lead_hours") < high))


def family_sizes() -> dict[str, int]:
    """Return the contrast count of each family."""
    sizes: dict[str, int] = {}
    for family, *_ in CONTRASTS.values():
        sizes[family] = sizes.get(family, 0) + 1
    return sizes


def contrast(
    *, contrast_id: str, spec: tuple[str, tuple[str, ...], str, str], sizes: dict[str, int]
) -> list[dict[str, object]]:
    """Return one contrast's rows: every metric over all leads, the quantile score per band."""
    family, groups, treatment, reference = spec
    frames = {arm: read_arm(groups=groups, arm=arm) for arm in (treatment, reference)}
    seeds = {0} if contrast_id in SEED0_ONLY else set(common.SEEDS)
    for arm, frame in frames.items():
        if not seeds <= set(frame["seed"].unique().to_list()):
            msg = f"{contrast_id}: arm {arm} lacks some of the seeds {sorted(seeds)}"
            raise ValueError(msg)
    frames = {arm: f.filter(pl.col("seed").is_in(sorted(seeds))) for arm, f in frames.items()}
    both = (
        frames[treatment]
        .select(*common.KEYS)
        .unique()
        .join(frames[reference].select(*common.KEYS).unique(), on=list(common.KEYS), how="inner")
    )
    frames = {arm: f.join(both, on=list(common.KEYS), how="semi") for arm, f in frames.items()}
    planned_level = 100.0 * (1.0 - 0.05 / sizes[family])
    rows = []
    for scope, low, high in SCOPES:
        scoped = {arm: in_scope(frame=f, low=low, high=high) for arm, f in frames.items()}
        losses = experiment_store.losses_for_bootstrap(arm_losses=scoped)
        metrics = INTERVAL_METRICS if scope == "all leads" else ("qs",)
        levels = (
            (("planned", planned_level), ("95", 95.0)) if scope == "all leads" else (("95", 95.0),)
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
                    n_resamples=N_RESAMPLES_DECISIVE if family == "decisive" else N_RESAMPLES,
                )
                rows.append(
                    {
                        "id": contrast_id,
                        "family": family,
                        "groups": "+".join(groups),
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
                    }
                )
    return rows


def absolute_rows() -> list[dict[str, object]]:
    """Return every arm's mean metrics per group and scope, with intervals on the headline three."""
    rows = []
    for group in GROUPS:
        for path in sorted((METRICS_DIR / group).glob("*.parquet")):
            if path.stem == "shared_keys":
                continue
            frame = pl.read_parquet(path).filter(pl.col("shared"))
            for scope, low, high in SCOPES:
                part = in_scope(frame=frame, low=low, high=high)
                row: dict[str, object] = {
                    "group": group,
                    "arm": path.stem,
                    "scope": scope,
                    "cases": part.height // max(1, part["seed"].n_unique()),
                    "seeds": part["seed"].n_unique(),
                    **{m: float(part[m].mean()) for m in MEAN_METRICS},
                }
                losses = experiment_store.losses_for_bootstrap(arm_losses={path.stem: part})
                for metric in INTERVAL_METRICS:
                    interval = bootstrap.bootstrap_absolute(
                        losses=losses, arm=path.stem, metric=metric
                    )
                    row[f"{metric}_lower"] = interval["lower_95"]
                    row[f"{metric}_upper"] = interval["upper_95"]
                rows.append(row)
        print(f"absolute: {group} done", flush=True)
    return rows


def main() -> None:
    """Write the absolute and contrast tables."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(absolute_rows()).write_parquet(RESULTS_DIR / "absolute.parquet")
    sizes = family_sizes()
    rows = []
    for contrast_id, spec in CONTRASTS.items():
        rows += contrast(contrast_id=contrast_id, spec=spec, sizes=sizes)
        print(f"{contrast_id} done", flush=True)
    result = pl.DataFrame(rows)
    result.write_parquet(RESULTS_DIR / "contrasts.parquet")
    with pl.Config(tbl_rows=200, tbl_cols=12, fmt_str_lengths=40):
        print(
            result.filter((pl.col("scope") == "all leads") & (pl.col("metric") == "qs")).select(
                "id", "treatment", "reference", "level_label", "difference", "lower", "upper"
            )
        )


if __name__ == "__main__":
    main()
