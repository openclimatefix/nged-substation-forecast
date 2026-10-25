# ruff: noqa: E501, PLR0912
"""Score experiment 345: per-row losses, farm-equal means, month-block intervals at 98.75%.

Solar: per day band and history setting, rows are the scored-year runs that every compared arm
holds with finite quantiles (experiment 310's arms and floors, experiment 318's head, experiment
313's pure network, and this experiment's pooled networks). Seeds of one arm are averaged quantile
by quantile. Wind: experiment 285's scored cases (W1 to W3, kept rows) at 12 to 252 hours, split
into the same day bands, with the ensemble mixtures given experiment 285's spread factor.
`h1` and `h2` are the first and second half of the scored year (the replicate).

    PYTHONPATH=ml-experiments/00285-wind-effective-wind-adapters:ml-experiments/00310-metered-farm-long-lead-scoring \
        uv run --no-sync python ml-experiments/00345-*/e00345_score.py [solar|wind]
"""

import re
import sys
from typing import Final

import e00285_common as c285  # ty: ignore[unresolved-import]
import e00285_models as m285  # ty: ignore[unresolved-import]
import e00285_score as s285  # ty: ignore[unresolved-import]
import e00310_common as c310  # ty: ignore[unresolved-import]
import e00310_score as s310  # ty: ignore[unresolved-import]
import e00345_common as common
import numpy as np
import polars as pl

LO_PCT, HI_PCT = 0.625, 99.375
DRAWS: Final[int] = 10_000
HALVES: Final[dict[str, tuple[str, ...]]] = {
    "h1": ("2025-09", "2025-10", "2025-11", "2025-12", "2026-01", "2026-02"),
    "h2": ("2026-03", "2026-04", "2026-05", "2026-06", "2026-07", "2026-08"),
}
MINE: Final[tuple[str, ...]] = common.SOLAR_ARM_NAMES
# arm label -> (directory holding the files, file prefix, own-history seeds, zero-history seeds)
REFERENCES: Final[dict[str, tuple[object, str]]] = {
    "ref_blend310": (c310.PRED_DIR, "blend"),
    "ref_mdn_plant310": (c310.PRED_DIR, "mdn_ens"),
    "ref_p2": (c310.PRED_DIR, "p2"),
    "ref_gam": (c310.PRED_DIR, "gam"),
    "ref_xgb": (c310.PRED_DIR, "xgb"),
    "ref_mdn_plant318": (common.E318_DIR / "predictions", "default"),
    "ref_pure_lead318": (common.E318_DIR / "predictions", "pure_lead_pooled"),
    "ref_pure313": (common.E313_DIR / "solar_predictions", "pure_mdn"),
}
SINGLE_FAMILIES: Final[tuple[str, ...]] = (
    "ref_p2",
    "ref_gam",
    "ref_xgb",
    "ref_pure313",
    "solar_pooled",
    "both",
)
PLANNED: Final[tuple[tuple[str, str], ...]] = (
    ("solar_pooled", "ref_mdn_plant318"),  # P1
    ("both", "solar_pooled"),  # P2 (solar)
    ("both", "solar_pooled"),  # P3 (zero history uses the same pair)
    ("solar_pooled", "ref_blend310"),  # P4 reads pin99
    ("both", "ref_blend310"),
)
EXTRA: Final[tuple[tuple[str, str], ...]] = (
    ("both", "ref_mdn_plant318"),
    ("solar_pooled", "ref_pure313"),
    ("solar_pooled", "ref_pure_lead318"),
    ("both_plant", "both"),
    ("both_once", "both"),
    ("both_monthly", "both"),
    ("both_shuf_fuel", "both"),
    ("both_shuf_wind", "both"),
    ("both", "ref_pure_lead318"),
    ("ref_mdn_plant318", "ref_blend310"),
    ("ref_mdn_plant318", "floor_climatology"),
    ("ref_mdn_plant318", "floor_heuristic"),
    ("both", "floor_climatology"),
    ("solar_pooled", "floor_climatology"),
    ("ref_blend310", "floor_climatology"),
    ("ref_blend310", "floor_heuristic"),
    ("ref_mdn_plant310", "ref_mdn_plant318"),
)
METRICS: Final[tuple[str, ...]] = (
    "score",
    "pin01",
    "pin05",
    "pin95",
    "pin99",
    "cover01",
    "cover05",
    "cover95",
    "cover99",
    "ae_mean",
    "median_minus_mean",
)


def seed_average(*, files: list, arm_name: str) -> pl.DataFrame:
    """Return one row per (farm, time, run) with the quantiles averaged over the files' seeds."""
    frame = pl.concat(
        [
            pl.read_parquet(f).select(
                *s310.KEYS, "lead_hours", "setting", "band", "actual", *s310.QCOLS
            )
            for f in files
        ]
    ).with_columns(pl.col(s310.QCOLS).cast(pl.Float64), pl.col("lead_hours").cast(pl.Float64))
    return (
        frame.group_by(*s310.KEYS, "setting", "band")
        .agg(
            pl.col("lead_hours").first(),
            pl.col("actual").first(),
            *[pl.col(c).mean() for c in s310.QCOLS],
        )
        .with_columns(arm=pl.lit(arm_name))
    )


def crossing_rate(*, frame: pl.DataFrame) -> float:
    """Return the share of rows whose seed-averaged quantiles are not already sorted."""
    q = frame.select(s310.QCOLS).to_numpy()
    return float((np.diff(q, axis=1) < -1e-9).any(axis=1).mean())


def load_band(*, band: str) -> tuple[pl.DataFrame, list[dict]]:
    """Return per-row losses of every compared arm and setting, and the arms' crossing rates."""
    floors = pl.read_parquet(c310.DATA_DIR / f"floors_{band}.parquet")
    pieces, crossings = [], []
    for setting in c310.SETTINGS:
        parts = []
        mine_dir = common.PRED_DIR / band / setting
        names = sorted({re.sub(r"_s\d+_.*$", "", f.name) for f in mine_dir.glob("*.parquet")})
        for arm in names:
            files = sorted(mine_dir.glob(f"{arm}_s*_*.parquet"))
            parts.append(seed_average(files=files, arm_name=arm))
        for label, (root, prefix) in REFERENCES.items():
            folder = root / band / setting  # ty: ignore[unsupported-operator]
            files = sorted(
                f
                for f in folder.glob(f"{prefix}_s*.parquet")
                if re.fullmatch(rf"{prefix}_s\d+_.*", f.name)
            )
            if files:
                parts.append(seed_average(files=files, arm_name=label))
        columns = parts[0].columns
        parts.append(
            floors.with_columns(
                setting=pl.lit(setting), lead_hours=pl.col("lead_hours").cast(pl.Float64)
            ).select(columns)
        )
        both = pl.concat([p.select(columns) for p in parts])
        for arm, sub in both.group_by("arm"):
            crossings.append(
                {
                    "band": band,
                    "setting": setting,
                    "arm": arm[0],
                    "crossing_rate": crossing_rate(frame=sub),
                }
            )
        finite = both.filter(pl.all_horizontal([pl.col(c).is_finite() for c in s310.QCOLS]))
        arms = finite["arm"].unique().to_list()
        keep = (
            finite.group_by(s310.KEYS)
            .agg(n=pl.col("arm").n_unique())
            .filter(pl.col("n") == len(arms))
            .select(s310.KEYS)
        )
        pieces.append(s310.row_losses(frame=finite.join(keep, on=s310.KEYS, how="inner")))
    rows = pl.concat(pieces).with_columns(
        month=pl.col("init_time").dt.strftime("%Y-%m"), farm=pl.col("time_series_id")
    )
    return rows, crossings


def percent_rows(*, draws: np.ndarray, point: np.ndarray, idx: dict, arm: str, ref: str) -> dict:
    """Return the points and percent change of `arm` against `ref` with intervals."""
    diff = draws[idx[arm]] - draws[idx[ref]]
    change = 100.0 * diff / draws[idx[ref]]
    dlo, dhi = np.nanpercentile(diff, [LO_PCT, HI_PCT])
    lo, hi = np.nanpercentile(change, [LO_PCT, HI_PCT])
    return {
        "points": point[idx[arm]] - point[idx[ref]],
        "points_lo": dlo,
        "points_hi": dhi,
        "percent": 100.0 * (point[idx[arm]] - point[idx[ref]]) / point[idx[ref]],
        "lo": lo,
        "hi": hi,
    }


def score_solar() -> None:
    """Write the solar score and contrast tables."""
    pl.Config.set_tbl_rows(200)
    pl.Config.set_tbl_cols(30)
    scores, contrasts, all_rows, crossings, best_family = [], [], [], [], []
    for band in c310.BAND_ORDER:
        rows, cross = load_band(band=band)
        crossings += cross
        all_rows.append(rows)
        for setting in c310.SETTINGS:
            sub = rows.filter(pl.col("setting") == setting)
            n = sub.group_by("arm").agg(n=pl.len())
            if n["n"].min() != n["n"].max():
                msg = f"arms hold different row counts in {band} {setting}:\n{n.sort('n')}"
                raise ValueError(msg)
            for half, months in [("all", None), *HALVES.items()]:
                part = sub if months is None else sub.filter(pl.col("month").is_in(months))
                for metric in METRICS:
                    sums, counts, arms = s310.farm_sums(rows=part, value=metric)
                    point, draws = s310.boot(sums=sums, counts=counts, seed=345)
                    idx = {a: i for i, a in enumerate(arms)}
                    for a in arms:
                        lo, hi = np.nanpercentile(draws[idx[a]], [LO_PCT, HI_PCT])
                        scores.append(
                            {
                                "band": band,
                                "setting": setting,
                                "half": half,
                                "metric": metric,
                                "arm": a,
                                "value": point[idx[a]],
                                "lo": lo,
                                "hi": hi,
                                "rows": int(n["n"][0]),
                            }
                        )
                    if metric not in ("score", "pin01", "pin99", "ae_mean"):
                        continue
                    families = [a for a in SINGLE_FAMILIES if a in idx]
                    if metric == "score" and families:
                        best = min(families, key=lambda a: point[idx[a]])
                        best_family.append(
                            {"band": band, "setting": setting, "half": half, "best": best}
                        )
                        pairs = [*PLANNED, *EXTRA, ("ref_mdn_plant318", best)]
                    else:
                        pairs = [*PLANNED, *EXTRA]
                    for arm, ref in dict.fromkeys(pairs):
                        if arm not in idx or ref not in idx or arm == ref:
                            continue
                        contrasts.append(
                            {
                                "band": band,
                                "setting": setting,
                                "half": half,
                                "metric": metric,
                                "arm": arm,
                                "reference": ref,
                                "planned": (arm, ref) in PLANNED,
                                **percent_rows(draws=draws, point=point, idx=idx, arm=arm, ref=ref),
                            }
                        )
        print("scored", band, flush=True)
    rows = pl.concat(all_rows)
    rows.write_parquet(common.DATA_DIR / "solar_row_losses.parquet")
    pl.DataFrame(scores).write_csv(common.DATA_DIR / "solar_scores.csv")
    pl.DataFrame(contrasts).write_csv(common.DATA_DIR / "solar_contrasts.csv")
    pl.DataFrame(crossings).write_csv(common.DATA_DIR / "solar_crossing_rates.csv")
    pl.DataFrame(best_family).write_csv(common.DATA_DIR / "solar_best_family.csv")
    farm_contrasts(rows=rows)
    table = pl.DataFrame(scores).filter((pl.col("metric") == "score") & (pl.col("half") == "all"))
    for setting in c310.SETTINGS:
        print(f"--- solar {setting}: quantile score by band")
        print(
            table.filter(pl.col("setting") == setting)
            .pivot(on="arm", index="band", values="value")
            .sort("band")
        )


def farm_contrasts(*, rows: pl.DataFrame) -> None:
    """Write each farm's change in score for the pooling pairs, with month-block intervals."""
    pairs = (
        ("solar_pooled", "ref_mdn_plant318"),
        ("both", "solar_pooled"),
        ("both", "ref_mdn_plant318"),
    )
    out = []
    rng = np.random.default_rng(3450)
    for setting in c310.SETTINGS:
        for band_group, bands in (("all", c310.BAND_ORDER), ("d1", ("d1",)), ("d2", ("d2",))):
            sub = rows.filter((pl.col("setting") == setting) & pl.col("band").is_in(list(bands)))
            months = sorted(sub["month"].unique().to_list())
            weights = rng.multinomial(len(months), np.ones(len(months)) / len(months), size=DRAWS)
            for farm in c310.FARMS:
                farm_rows = sub.filter(pl.col("farm") == farm)
                grouped = farm_rows.group_by("arm", "month").agg(
                    s=pl.col("score").sum(), n=pl.len()
                )
                tot: dict[str, tuple[np.ndarray, np.ndarray]] = {}
                for arm in {a for pair in pairs for a in pair}:
                    g = grouped.filter(pl.col("arm") == arm)
                    s = np.array([g.filter(pl.col("month") == m)["s"].sum() for m in months])
                    n = np.array([g.filter(pl.col("month") == m)["n"].sum() for m in months], float)
                    tot[arm] = (s, n)
                for arm, ref in pairs:
                    (sa, na), (sr, nr) = tot[arm], tot[ref]
                    a_all, r_all = sa.sum() / na.sum(), sr.sum() / nr.sum()
                    a_dr = (weights @ sa) / (weights @ na)
                    r_dr = (weights @ sr) / (weights @ nr)
                    change = 100.0 * (a_dr - r_dr) / r_dr
                    lo, hi = np.nanpercentile(change, [LO_PCT, HI_PCT])
                    out.append(
                        {
                            "setting": setting,
                            "bands": band_group,
                            "farm": farm,
                            "arm": arm,
                            "reference": ref,
                            "points": a_all - r_all,
                            "percent": 100.0 * (a_all - r_all) / r_all,
                            "lo": lo,
                            "hi": hi,
                        }
                    )
    pl.DataFrame(out).write_csv(common.DATA_DIR / "solar_farm_contrasts.csv")


# ---------------------------------------------------------------- wind

WIND_BANDS: Final = c310.BANDS


def wind_day(*, lead_hours: np.ndarray) -> np.ndarray:
    """Return the solar day-band convention for wind leads: day d holds 12+24(d-1) < lead <= 36+24(d-1)."""
    return np.ceil((lead_hours - 12.0) / 24.0).astype(int)


def wind_metrics(*, quantiles: np.ndarray, actual: np.ndarray) -> dict[str, np.ndarray]:
    """Return per-row metrics (points of capacity) of seed-averaged, sorted quantiles."""
    q = np.sort(quantiles.astype(np.float64), axis=1)
    scale = c285.POINTS_PER_FRACTION
    mean_forecast = q[:, 1:10] @ c285.NINE_MASSES
    out = {
        "score": c285.quantile_score(quantiles=q, actual=actual) * scale,
        "ae_mean": np.abs(actual - mean_forecast) * scale,
        "median_minus_mean": (q[:, c285.MEDIAN_INDEX] - mean_forecast) * scale,
    }
    for name, column, index in (
        ("01", "q01", 0),
        ("05", "q05", 1),
        ("95", "q95", 9),
        ("99", "q99", 10),
    ):
        out[f"pin{name}"] = c285.pinball_at(quantiles=q, actual=actual, column=column) * scale
        out[f"cover{name}"] = (actual <= q[:, index]).astype(np.float64)
    return out


def load_mine(*, arm: str, setting: str, weather: str, cases: pl.DataFrame) -> np.ndarray | None:
    """Return the seed-averaged mixtures of one of this experiment's wind arms, or None."""
    seeds = []
    for seed in common.OWN_SEEDS:
        parts = []
        for farm in common.WIND_FARMS:
            path = common.WIND_DIR / f"{arm}__{setting}__{farm}__s{seed}__{weather}.npy"
            if not path.exists():
                return None
            parts.append(np.load(path))
        seeds.append(np.concatenate(parts).astype(np.float64))
    mean = np.mean(seeds, axis=0)
    return c285.apply_spread(cases=cases, mixture=mean) if weather == "ens" else mean


def load_reference_313(*, arm: str, setting: str, weather: str, cases: pl.DataFrame) -> np.ndarray:
    """Return experiment 313's stored pure wind arm, seed-averaged, with the spread factor."""
    seeds_used = (0, 1, 2, 3, 4) if arm == "pure_mdn" else (0, 1, 2)
    raw = np.mean(
        [
            np.concatenate(
                [
                    np.load(
                        common.E313_DIR
                        / "wind"
                        / f"{arm}__{setting}__{farm}__s{seed}__{weather}.npy"
                    )
                    for farm in common.WIND_FARMS
                ]
            ).astype(np.float64)
            for seed in seeds_used
        ],
        axis=0,
    )
    return c285.apply_spread(cases=cases, mixture=raw) if weather == "ens" else raw


def wind_boot(
    *, values: dict[str, np.ndarray], keep: np.ndarray, month: np.ndarray, seed: int
) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    """Return each arm's mean over `keep` rows and its month-block bootstrap draws."""
    months = np.unique(month[keep])
    rng = np.random.default_rng(seed)
    weights = rng.multinomial(len(months), np.ones(len(months)) / len(months), size=DRAWS).astype(
        float
    )
    counts = np.array([(keep & (month == m)).sum() for m in months], dtype=float)
    points, draws = {}, {}
    for arm, v in values.items():
        sums = np.array([v[keep & (month == m)].sum() for m in months])
        points[arm] = float(sums.sum() / counts.sum())
        draws[arm] = (weights @ sums) / (weights @ counts)
    return points, draws


def score_wind() -> None:
    """Write the wind score and contrast tables."""
    cases = s285.ordered_cases()
    specs = m285.build_specs()
    actual = cases["actual_norm"].to_numpy().astype(np.float64)
    month = cases["month"].to_numpy()
    lead = cases["lead_hours"].to_numpy()
    kept = cases["kept"].to_numpy()
    day = wind_day(lead_hours=lead)
    band_masks: dict[str, np.ndarray] = {
        name: kept & np.isin(day, days) & (lead > 12) & (lead <= 252)
        for name, days in WIND_BANDS.items()
    }
    band_masks["all_12_252"] = kept & (lead > 12) & (lead <= 252)
    band_masks["all_to_351"] = kept
    scores, contrasts = [], []
    for weather in ("ens", "era5"):
        for setting in c310.SETTINGS:
            quant: dict[str, np.ndarray] = {}
            for arm in common.WIND_ARM_NAMES:
                q = load_mine(arm=arm, setting=setting, weather=weather, cases=cases)
                if q is not None:
                    quant[arm] = q
            for ref in ("pure_mdn", "pure_xgb"):
                try:
                    quant[f"ref_{ref}313"] = load_reference_313(
                        arm=ref, setting=setting, weather=weather, cases=cases
                    )
                except FileNotFoundError:
                    continue
            if setting == "own_history":
                for name, spec_name in (("ref_adapter285", "A4"), ("ref_core276", "P")):
                    q = s285.load_quantiles(spec=specs[spec_name], setting=weather, cases=cases)
                    if q is not None:
                        quant[name] = q.mean(axis=0)
            values = {arm: wind_metrics(quantiles=q, actual=actual) for arm, q in quant.items()}
            for band, keep in band_masks.items():
                for half, months in [("all", None), *HALVES.items()]:
                    sel = keep if months is None else keep & np.isin(month, months)
                    if not sel.any():
                        continue
                    for metric in METRICS:
                        points, draws = wind_boot(
                            values={a: v[metric] for a, v in values.items()},
                            keep=sel,
                            month=month,
                            seed=3451,
                        )
                        for a in values:
                            lo, hi = np.nanpercentile(draws[a], [LO_PCT, HI_PCT])
                            scores.append(
                                {
                                    "band": band,
                                    "setting": setting,
                                    "weather": weather,
                                    "half": half,
                                    "metric": metric,
                                    "arm": a,
                                    "value": points[a],
                                    "lo": lo,
                                    "hi": hi,
                                    "rows": int(sel.sum()),
                                }
                            )
                        if metric not in ("score", "pin01", "pin99", "ae_mean"):
                            continue
                        families = [
                            a
                            for a in (
                                "wind_pooled",
                                "both",
                                "ref_pure_mdn313",
                                "ref_pure_xgb313",
                                "ref_adapter285",
                            )
                            if a in values
                        ]
                        best = min(families, key=lambda a: points[a]) if families else None
                        pairs = [
                            ("both", "wind_pooled"),
                            ("wind_pooled", "ref_pure_mdn313"),
                            ("both", "ref_pure_mdn313"),
                            ("both_shuf_fuel", "both"),
                            ("both_shuf_wind", "both"),
                            ("ref_adapter285", "ref_pure_mdn313"),
                            ("ref_adapter285", "wind_pooled"),
                            ("wind_pooled", "ref_adapter285"),
                            ("both", "ref_adapter285"),
                        ]
                        if best is not None and metric == "score":
                            pairs.append(("ref_adapter285", best))
                        for arm, ref in dict.fromkeys(pairs):
                            if arm not in draws or ref not in draws or arm == ref:
                                continue
                            diff = draws[arm] - draws[ref]
                            change = 100.0 * diff / draws[ref]
                            contrasts.append(
                                {
                                    "band": band,
                                    "setting": setting,
                                    "weather": weather,
                                    "half": half,
                                    "metric": metric,
                                    "arm": arm,
                                    "reference": ref,
                                    "points": points[arm] - points[ref],
                                    "points_lo": float(np.nanpercentile(diff, LO_PCT)),
                                    "points_hi": float(np.nanpercentile(diff, HI_PCT)),
                                    "percent": 100.0 * (points[arm] - points[ref]) / points[ref],
                                    "lo": float(np.nanpercentile(change, LO_PCT)),
                                    "hi": float(np.nanpercentile(change, HI_PCT)),
                                }
                            )
            print("wind scored", weather, setting, flush=True)
    pl.DataFrame(scores).write_csv(common.DATA_DIR / "wind_scores.csv")
    pl.DataFrame(contrasts).write_csv(common.DATA_DIR / "wind_contrasts.csv")
    table = pl.DataFrame(scores).filter(
        (pl.col("metric") == "score") & (pl.col("half") == "all") & (pl.col("weather") == "ens")
    )
    for setting in c310.SETTINGS:
        print(f"--- wind {setting} (ensemble weather): quantile score by band")
        print(
            table.filter(pl.col("setting") == setting)
            .pivot(on="arm", index="band", values="value")
            .sort("band")
        )


if __name__ == "__main__":
    which = set(sys.argv[1:]) or {"solar", "wind"}
    if "solar" in which:
        score_solar()
    if "wind" in which:
        score_wind()
