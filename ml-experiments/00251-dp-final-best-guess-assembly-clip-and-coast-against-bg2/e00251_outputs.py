# ruff: noqa: E501
"""Write the files the substation-cards session reads: predictions, properties, tables, README.

Output goes to `data/ml-experiments/00251-*/cards/`.

    uv run --no-sync python ml-experiments/00251-*/e00251_outputs.py
"""

import json
from typing import Final

import numpy as np
import polars as pl

import e00251_common as common  # isort: skip

OUT = common.DATA_DIR / "cards"
CARD_ARMS: Final[tuple[str, ...]] = ("arm1", "arm2c")
CLIP_SHARE: Final[dict[str, float]] = {"D5": 0.6}
"""The hand-set clip as a share of the solar capacity held for the substation (D5 only)."""


def write_predictions() -> None:
    """Write `<arm>_scored_<site>.parquet` in the `dp_preds` layout for arm 1 and arm 2c."""
    base = common.ARM_DIR / "scored"
    rows = pl.read_parquet(base / "rows.parquet").select(
        "site", *common.KEYS, "run", "lead_hours", "seed", "actual", "mean"
    )
    folder = OUT / "dp_preds"
    folder.mkdir(parents=True, exist_ok=True)
    for arm in CARD_ARMS:
        q = np.load(base / f"{arm}.npy")
        frame = rows.with_columns([pl.Series(n, q[:, i]) for i, n in enumerate(common.QUANTILES)])
        for site in common.SCORED_SITES:
            frame.filter(pl.col("site") == site).write_parquet(
                folder / f"{arm}_scored_{site}.parquet"
            )


def properties() -> pl.DataFrame:
    """Return per-substation properties: capacity, level, heating response, clip, residual noise."""
    from studies.dp_features import default_input_paths, load_or_build_feature_cache

    cache = load_or_build_feature_cache(paths=default_input_paths(), cache_dir=common.E220_ARMS.parent.parent.parent / "00106-differentiable-physics-4-all-years-drift-and-recent-level" / "cache")  # fmt: skip
    base = common.ARM_DIR / "scored"
    rows = pl.read_parquet(base / "rows.parquet")
    q50 = np.load(base / "arm1.npy")[:, 5]
    rows = rows.with_columns(residual=pl.col("actual") - pl.Series(q50))
    out = []
    for site in common.SCORED_SITES:
        capacity = float(cache.sites[site].table.effective_capacity_mw)
        sub = rows.filter(pl.col("site") == site)
        cold = sub.filter(pl.col("ens_mean_c") < 10.0)
        x = cold["ens_mean_c"].to_numpy().astype(np.float64)
        y = cold["actual"].to_numpy().astype(np.float64)
        slope = float(np.polyfit(x, y, 1)[0])
        short = sub.filter(pl.col("lead_hours") <= 6.0)
        out.append(
            {
                "site": site,
                "effective_capacity_mw": capacity,
                "mean_level_points_of_capacity": float(np.mean(sub["actual"].to_numpy())),
                "heating_response_points_per_degC_below_10C": slope,
                "heating_response_mw_per_degC_below_10C": slope * capacity / 100.0,
                "clip_share_of_solar_capacity": CLIP_SHARE.get(site),
                "kalman_noise_stand_in_residual_sd_lead_le_6h_points": float(
                    np.std(short["residual"].to_numpy(), ddof=1)
                ),
                "residual_sd_all_leads_points": float(np.std(sub["residual"].to_numpy(), ddof=1)),
            }
        )
    return pl.DataFrame(out)


def gap_table() -> pl.DataFrame:
    """Return experiment 205's stored quantile scores after injected gaps, by gap length and rule."""
    scores = json.loads(common.E205_REPORT.read_text())["scores"]
    names = {"n": "no gap (filter, offset, recipe R)", "a": "coast", "b": "coast, widened bands",
             "c2": "frozen model with offset and recipe R", "c": "frozen model alone"}  # fmt: skip
    return pl.DataFrame(
        [
            {
                "gap_hours": int(length),
                "rule": label,
                "quantile_score_all_leads": scores[length]["all"]["qs"][rule],
            }
            for length in sorted(scores, key=int)
            for rule, label in names.items()
        ]
    )


def main() -> None:
    """Write every file and the README."""
    OUT.mkdir(parents=True, exist_ok=True)
    write_predictions()
    props = properties()
    props.write_csv(OUT / "per_substation_properties.csv")
    gap = gap_table()
    gap.write_csv(OUT / "gap_scores_from_experiment_205.csv")
    report = json.loads((common.REPORT_DIR / "report_scored.json").read_text())
    (OUT / "score_tables.json").write_text(json.dumps(
        {k: report[k] for k in ("absolute", "absolute_by_cut", "tails_by_hour", "tails_by_cut", "by_lead_bin")}, indent=2))  # fmt: skip
    (OUT / "contrasts.json").write_text(json.dumps(report["contrasts"], indent=2))
    (OUT / "curtailment_scored.json").write_text(
        (common.REPORT_DIR / "curtailment_scored.json").read_text()
    )
    (OUT / "README.md").write_text(
        """# Experiment 251 outputs for the substation cards

Scored year 2025-07-02 to 2026-06-30, six Primaries (D5, D9, D10, D11, D14, D16), 09:00 UTC inits,
leads 0.5 to 168 h. Quantiles and `actual` are in points of effective capacity.

- `dp_preds/arm1_scored_<site>.parquet`: arm 1, experiment 220's A+H+S (arm D, network head,
  two-number solar summary). Columns as in experiment 169's `dp_preds`, plus `mean`.
- `dp_preds/arm2c_scored_<site>.parquet`: arm 2c (post hoc), arm 1 with D5 replaced by the clipped
  forecast without the network head; the other five substations equal arm 1. Pre-registered winner:
  arm 1 (see the result page).
- `per_substation_properties.csv`: effective capacity, mean level, heating response (OLS slope of
  demand on the ensemble-mean temperature below 10 C), clip share (D5 only), and the spread of arm 1's
  median error. The filter's fitted noise parameters are not stored per site in a file this
  experiment reads, so the last two columns are empirical stand-ins for them, not the fitted values.
- `gap_scores_from_experiment_205.csv`: stored quantile scores after injected reading gaps.
- `score_tables.json`, `contrasts.json`: absolute scores, cuts, tails by hour band, contrasts with
  intervals.
- `curtailment_scored.json`: saving against the manual heuristic at limits of 0.9, 0.75, and 0.5
  times each substation's largest earlier flow.
"""
    )
    print(props)
    print(gap)


if __name__ == "__main__":
    main()
