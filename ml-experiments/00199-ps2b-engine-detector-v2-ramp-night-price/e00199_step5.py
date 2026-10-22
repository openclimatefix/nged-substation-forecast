"""Do the flagged periods at D5, D10, and D11 look like engines? Supporting evidence only.

For each site the frozen experiment-196 detector's flags (all rows, as experiment 196 reported them)
are grouped into flagged days (4 or more flagged half-hours on a UTC date). Reported: the night
share of flagged half-hours against the night share of all rows, and the share of flagged days with
a positive daily spark spread against matched unflagged days (same calendar month and weekend flag),
the method of experiment 186. D8 from 2023-09 to 2025-02, where the engine is visible, is the
positive control.

    OMP_NUM_THREADS=1 uv run --no-sync python ml-experiments/00199-*/e00199_step5.py
"""

import json
from datetime import UTC, datetime
from typing import Any

import numpy as np
import polars as pl
from e00199_common import DATA_DIR, GRID_START, stamp, times
from e00199_pipeline import clean_mask, fit_noise, load_series, run_loop
from e00199_scoring import flagged
from e00199_v2 import load_frozen_fit, night_mask, spread_array

N_DRAWS = 2000
MIN_FLAGGED = 4
SITES = {
    "D8 control 2023-09 to 2025-02": ("D8", "2023-09-01", "2025-03-01"),
    "D8 2025-03 to 2026-06": ("D8", "2025-03-01", "2026-07-01"),
    "D5": ("D5", "2023-06-01", "2026-07-01"),
    "D10": ("D10", "2023-06-01", "2026-07-01"),
    "D11": ("D11", "2023-06-01", "2026-07-01"),
}


def run_site(*, label: str, first: str, last: str) -> dict[str, Any]:
    """Flags, night share, and matched-day spread test for one site and span."""
    d8_noise, params = load_frozen_fit()
    s = load_series(label=label)
    n = len(s.y)
    finite = np.isfinite(s.y) & np.isfinite(s.load)
    noise = d8_noise if label == "D8" else fit_noise(s=s, rows=finite)
    _, post = run_loop(s=s, noise=noise, params=params)
    flag = flagged(smoothed=post.smoothed)
    night = night_mask(label=label, n=n)
    spread, _ = spread_array(n=n)
    a, b = (stamp(datetime.fromisoformat(x).replace(tzinfo=UTC)) for x in (first, last))
    span = np.zeros(n, dtype=bool)
    span[a:b] = True
    use = span & finite
    t = times(n=n)
    frame = pl.DataFrame(
        {
            "day": np.arange(n) // 48,
            "month": t.dt.month().to_numpy(),
            "weekend": (t.dt.weekday().to_numpy() >= 6).astype(int),
            "flag": flag & use,
            "night_flag": flag & use & night,
            "use": use,
            "spread": pl.Series(np.where(use, spread, np.nan)).fill_nan(None),
        }
    )
    days = (
        frame.filter(pl.col("use"))
        .group_by("day")
        .agg(
            month=pl.col("month").first(),
            weekend=pl.col("weekend").first(),
            n_flag=pl.col("flag").sum(),
            n_night_flag=pl.col("night_flag").sum(),
            spread=pl.col("spread").mean(),
            rows=pl.len(),
        )
        .filter(pl.col("rows") >= 40, pl.col("spread").is_not_null())
    )
    out: dict[str, Any] = {
        "night_share_of_rows": float(night[use].mean()),
        "night_share_of_flagged": float(night[use & flag].mean()) if (use & flag).any() else None,
        "flagged_half_hours": int((use & flag).sum()),
        "hour_profile_of_flagged_utc": np.bincount(
            t.dt.hour().to_numpy()[use & flag], minlength=24
        ).tolist(),
    }
    rng = np.random.default_rng(199)
    for name, col in (("any", "n_flag"), ("night", "n_night_flag")):
        hit = days.filter(pl.col(col) >= MIN_FLAGGED)
        pool = days.filter(pl.col("n_flag") == 0)
        res: dict[str, Any] = {"flagged_days": hit.height, "unflagged_days": pool.height}
        if hit.height >= 5 and pool.height > 0:
            groups = {
                key: g["spread"].to_numpy()
                for key, g in pool.partition_by("month", "weekend", as_dict=True).items()
            }
            draws_pos, draws_mean, keep_rows = [], [], []
            for m, w, sp in zip(
                hit["month"].to_list(),
                hit["weekend"].to_list(),
                hit["spread"].to_list(),
                strict=True,
            ):
                if (m, w) in groups and len(groups[(m, w)]) > 0:
                    keep_rows.append((m, w, sp))
            res["flagged_days_matched"] = len(keep_rows)
            obs_pos = np.mean([sp > 0 for _, _, sp in keep_rows])
            obs_mean = np.mean([sp for _, _, sp in keep_rows])
            for _ in range(N_DRAWS):
                pick = rng.integers(len(keep_rows), size=len(keep_rows))
                chosen = [keep_rows[i] for i in pick]
                m_vals = [rng.choice(groups[(m, w)]) for m, w, _ in chosen]
                s_vals = [sp for _, _, sp in chosen]
                draws_pos.append(
                    np.mean([x > 0 for x in s_vals]) - np.mean([x > 0 for x in m_vals])
                )
                draws_mean.append(np.mean(s_vals) - np.mean(m_vals))
            mp = float(np.mean([np.mean(groups[(m, w)] > 0) for m, w, _ in keep_rows]))
            res |= {
                "flagged_positive_share": float(obs_pos),
                "matched_positive_share": mp,
                "diff_positive_points": [
                    float(v) * 100 for v in np.percentile(draws_pos, [2.5, 50, 97.5])
                ],
                "flagged_mean_spread": float(obs_mean),
                "diff_mean_spread": [float(v) for v in np.percentile(draws_mean, [2.5, 50, 97.5])],
            }
        out[name] = res
    _ = GRID_START, clean_mask
    return out


def main() -> None:
    """Print and store the table."""
    results = {}
    for key, (label, first, last) in SITES.items():
        results[key] = run_site(label=label, first=first, last=last)
        r = results[key]
        print(f"== {key}: flagged half-hours {r['flagged_half_hours']}, night share of flagged ")
        print(
            f"   {r['night_share_of_flagged']} against {r['night_share_of_rows']:.2f} of all rows"
        )
        for name in ("any", "night"):
            print(f"   {name}: {json.dumps(r[name], default=float)}")
        print(f"   flagged half-hours by UTC hour: {r['hour_profile_of_flagged_utc']}", flush=True)
    (DATA_DIR / "step5.json").write_text(json.dumps(results, indent=1, default=float))


if __name__ == "__main__":
    main()
