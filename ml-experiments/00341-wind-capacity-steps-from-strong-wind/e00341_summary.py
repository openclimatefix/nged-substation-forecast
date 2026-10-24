# ruff: noqa: E501, PLR0912, PLR0915
"""Experiment 00341, step 4: every number the write-up quotes, printed and saved.

Reads `work/grid_rows.parquet`, `work/arm_e_rows.parquet`, `work/real.json`, and `work/prep.json`.
Writes `work/summary.json` and prints the tables.

    uv run --no-sync python ml-experiments/00341-*/e00341_summary.py
"""

import json
from typing import Final

import e00341_common as common
import numpy as np
import polars as pl

HORIZON_FOUND_DAYS: Final[int] = 60
STEP_BLOCK: Final[int] = 28
EPISODE_BLOCK: Final[int] = 7
N_BOOT: Final[int] = 1000
SEED: Final[int] = 3410
ARMS: Final[tuple[str, ...]] = ("A_p999@2", "A_p999@0", "A_max@0", "B", "C", "D")


def share_and_median(
    *, values: np.ndarray, starts: np.ndarray, limit: int = HORIZON_FOUND_DAYS, ci: bool = False
) -> dict:
    """Return the share found within `limit` days, the median delay, and bootstrap intervals.

    NaN marks a start excluded because the unmodified series already alarmed, and -1 means not
    found. The bootstrap resamples whole calendar months of start dates (30.44-day periods).
    """
    keep = ~np.isnan(values)
    v, s = values[keep], starts[keep]
    if len(v) == 0:
        return {"n": 0}
    delay = np.where(v < 0, np.inf, v)
    plain = {
        "n": len(v),
        "share": float((delay <= limit).mean()),
        "median_delay": float(np.median(delay)),
    }
    if not ci:
        return plain
    months = (s // 30.4375).astype(int)
    uniq = np.unique(months)
    rng = np.random.default_rng(SEED)
    shares, meds = [], []
    for _ in range(N_BOOT):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([np.flatnonzero(months == m) for m in pick])
        shares.append(float((delay[idx] <= limit).mean()))
        meds.append(float(np.median(delay[idx])))
    return {
        **plain,
        "share_ci": [float(np.quantile(shares, 0.025)), float(np.quantile(shares, 0.975))],
        "median_ci": [float(np.quantile(meds, 0.025)), float(np.quantile(meds, 0.975))],
    }


def cell_values(
    frame: pl.DataFrame, *, site: str, cell: str, detector: str, block: int, arm: str
) -> tuple[np.ndarray, np.ndarray]:
    """Return the (values, starts) of one cell."""
    sub = frame.filter(
        (pl.col("site") == site) & (pl.col("cell") == cell) & (pl.col("detector") == detector)
        & (pl.col("block") == block) & (pl.col("arm") == arm)
    ).sort("start")  # fmt: skip
    return sub["value"].to_numpy(), sub["start"].to_numpy()


def main() -> None:
    """Compute and print the summary tables."""
    grid = pl.read_parquet(common.WORK_DIR / "grid_rows.parquet")
    arm_e = pl.read_parquet(common.WORK_DIR / "arm_e_rows.parquet")
    penalty = json.loads((common.WORK_DIR / "penalty.json").read_text())
    out: dict = {"penalty": {"c": penalty["c"], "h": penalty["h"]}}
    fractions = {s: {"t1": 1 / common.N_TURBINES[s], "t2": 2 / common.N_TURBINES[s],
                     "half": 0.5 / common.N_TURBINES[s], "p10": 0.1, "p25": 0.25, "p50": 0.5}
                 for s in common.SITES}  # fmt: skip

    # 1. Steps lasting 90 days: share found within 60 days and median delay, block 28, PELT
    steps: dict = {}
    for site in common.SITES:
        for arm in ARMS:
            for size in ("t1", "half", "p10", "t2", "p25", "p50"):
                if site == "W3" and size == "t2":
                    continue
                for detector in ("pelt", "cusum"):
                    v, s = cell_values(
                        grid,
                        site=site,
                        cell=f"{size}_d90",
                        detector=detector,
                        block=STEP_BLOCK,
                        arm=arm,
                    )
                    steps[f"{site}|{arm}|{size}|{detector}"] = share_and_median(
                        values=v, starts=s, ci=(size in {"p25", "t1"} and detector == "pelt")
                    )
    out["steps_d90_b28"] = steps

    # 2. Smallest step found at 80 % power within 60 days (steps lasting 90 days), per block length
    smallest: dict = {}
    for site in common.SITES:
        for arm in ARMS:
            for block in (7, 14, 28, 56):
                best = None
                for size, frac in sorted(fractions[site].items(), key=lambda kv: kv[1]):
                    if site == "W3" and size == "t2":
                        continue
                    v, s = cell_values(
                        grid, site=site, cell=f"{size}_d90", detector="pelt", block=block, arm=arm
                    )
                    r = share_and_median(values=v, starts=s)
                    if r.get("n", 0) >= 10 and r["share"] >= 0.8:
                        best = frac
                        break
                smallest[f"{site}|{arm}|B{block}"] = best
    out["smallest_step_80pct_60d"] = smallest

    # 3. Episodes: share found by duration, block 7, 25 % and one-turbine
    episodes: dict = {}
    for site in common.SITES:
        for arm in ARMS:
            for size in ("t1", "p25"):
                for duration in (3, 7, 14, 30, 60):
                    v, s = cell_values(
                        grid,
                        site=site,
                        cell=f"{size}_d{duration}",
                        detector="pelt",
                        block=EPISODE_BLOCK,
                        arm=arm,
                    )
                    episodes[f"{site}|{arm}|{size}|d{duration}"] = share_and_median(
                        values=v, starts=s, limit=min(duration + 60, 190)
                    )
    out["episodes_b7"] = episodes

    # 4. False alarms under controls, excess change points per farm over the record
    controls: dict = {}
    for site in common.SITES:
        for arm in ARMS:
            for block in (7, 14, 28):
                for cell in ("null", "swing9", "swing15", "swing25", "drift1"):
                    row = grid.filter(
                        (pl.col("site") == site) & (pl.col("cell") == cell) & (pl.col("block") == block)
                        & (pl.col("arm") == arm) & (pl.col("detector") == "new_pelt")
                    )  # fmt: skip
                    if len(row):
                        controls[f"{site}|{arm}|B{block}|{cell}"] = float(row["value"][0])
    out["controls_new_boundaries"] = controls
    cusum_controls: dict = {}
    for site in common.SITES:
        for arm in ARMS:
            for block in (14, 28):
                for cell in ("null", "swing9", "swing15", "swing25", "drift1"):
                    row = grid.filter(
                        (pl.col("site") == site) & (pl.col("cell") == cell) & (pl.col("block") == block)
                        & (pl.col("arm") == arm) & (pl.col("detector") == "new_cusum")
                    )  # fmt: skip
                    if len(row):
                        cusum_controls[f"{site}|{arm}|B{block}|{cell}"] = float(row["value"][0])
    out["controls_new_cusum_alarms"] = cusum_controls
    bias_frame = pl.read_parquet(common.WORK_DIR / "bias_rows.parquet")
    bias: dict = {}
    for site in common.SITES:
        for arm in ARMS:
            for block in (14, 28):
                v, s = cell_values(
                    bias_frame, site=site, cell="bias-5_d30", detector="pelt", block=block, arm=arm
                )
                ok = ~np.isnan(v)
                bias[f"{site}|{arm}|B{block}"] = {
                    "n": int(ok.sum()),
                    "false_detection_share": float(((v[ok] >= 0) & (v[ok] <= 190)).mean())
                    if ok.any()
                    else None,
                }
    out["bias_minus5_false_detections"] = bias

    # 5. State-space fits: what state was found, by true cell, and the detection share and delay
    hmm = pl.read_parquet(common.WORK_DIR / "hmm_rows.parquet")
    found_class = (
        pl.when(pl.col("arm") == "nameplate").then(pl.lit("nameplate"))
        .when(pl.col("arm").str.starts_with("clip")).then(pl.lit("clip"))
        .otherwise(pl.lit("turbines_out"))
    )  # fmt: skip
    hmm = hmm.with_columns(found=found_class, truth=pl.col("cell").str.replace(r"_d\d+$", ""),
                           duration=pl.col("cell").str.extract(r"_d(\d+)$").cast(pl.Int64))  # fmt: skip
    confusion: dict = {}
    for variant in ("hmm_full", "hmm_lattice"):
        for block in (7, 14):
            for site in common.SITES:
                sub = hmm.filter(
                    (pl.col("detector") == variant)
                    & (pl.col("block") == block)
                    & (pl.col("site") == site)
                    & ~pl.col("value").is_nan()
                )
                table = sub.group_by("truth", "duration", "found").len()
                confusion[f"{variant}|B{block}|{site}"] = table.sort(
                    "truth", "duration", "found"
                ).to_dicts()
    out["hmm_confusion_counts"] = confusion
    delays: dict = {}
    for variant in ("hmm_full", "hmm_lattice"):
        for block in (7, 14):
            for site in common.SITES:
                for truth in sorted(set(hmm["truth"].to_list())):
                    for duration in (3, 7, 14, 30, 60, 90):
                        sub = hmm.filter(
                            (pl.col("detector") == variant)
                            & (pl.col("block") == block)
                            & (pl.col("site") == site)
                            & (pl.col("truth") == truth)
                            & (pl.col("duration") == duration)
                        )
                        v = sub["value"].to_numpy()
                        ok = ~np.isnan(v)
                        fl = sub.filter(~pl.col("value").is_nan())
                        delays[f"{variant}|B{block}|{site}|{truth}|d{duration}"] = {
                            "n": int(ok.sum()),
                            "share_found": float((v[ok] >= 0).mean()) if ok.any() else None,
                            "median_delay_if_found": float(np.median(v[ok][v[ok] >= 0]))
                            if (v[ok] >= 0).any()
                            else None,
                            "found_classes": dict(
                                zip(
                                    *np.unique(fl["found"].to_numpy(), return_counts=True),
                                    strict=True,
                                )
                            )
                            if len(fl)
                            else {},
                            "found_states": dict(
                                zip(
                                    *np.unique(fl["arm"].to_numpy(), return_counts=True),
                                    strict=True,
                                )
                            )
                            if len(fl)
                            else {},
                        }
    out["hmm_detection"] = {
        k: {
            a: (v if not isinstance(v, dict) else {x: int(y) for x, y in v.items()})
            for a, v in d.items()
        }
        for k, d in delays.items()
    }

    # 6. Arm E
    e: dict = {}
    for arm in ("B", "C"):
        for block in (14, 28):
            for cell in sorted(set(arm_e["cell"].to_list())):
                for target in ("W2", "W3"):
                    sub = arm_e.filter(
                        (pl.col("cell") == cell)
                        & (pl.col("target") == target)
                        & (pl.col("block") == block)
                        & (pl.col("arm") == arm)
                    )
                    if not len(sub):
                        continue
                    v = sub.sort("start")["value"].to_numpy()
                    if cell == "null":
                        e[f"{arm}|B{block}|null_calibration_boundaries"] = float(v[0])
                    else:
                        e[f"{arm}|B{block}|{cell}|{target}"] = share_and_median(
                            values=v, starts=sub.sort("start")["start"].to_numpy()
                        )
    out["arm_e"] = e

    (common.WORK_DIR / "summary.json").write_text(json.dumps(out, indent=1))
    print_tables(out)


def print_tables(out: dict) -> None:
    """Print the headline tables."""
    print("penalty", out["penalty"])
    print("\n== steps lasting 90 days, block 28, PELT: share found within 60 days (median delay)")
    for site in common.SITES:
        for arm in ARMS:
            cells = []
            for size in ("t1", "half", "p10", "t2", "p25", "p50"):
                r = out["steps_d90_b28"].get(f"{site}|{arm}|{size}|pelt")
                if r and r.get("n"):
                    cells.append(f"{size}:{r['share']:.2f}({r['median_delay']:.0f},n{r['n']})")
            print(site, arm, " ".join(cells))
    print("\n== smallest step at 80% power within 60 days (steps of 90 days)")
    for k, v in out["smallest_step_80pct_60d"].items():
        if k.split("|")[1] in ("A_p999@2", "A_p999@0", "B", "C", "D"):
            print(k, v)
    print("\n== controls (new boundaries)")
    for k, v in out["controls_new_boundaries"].items():
        if "B28" in k or "B14" in k:
            print(k, v)
    print("\n== CUSUM new alarms under controls, block 28 (swing9 / swing15 / swing25 / drift1)")
    for site in common.SITES:
        for arm in ARMS:
            vals = [
                out["controls_new_cusum_alarms"].get(f"{site}|{arm}|B28|{c}")
                for c in ("swing9", "swing15", "swing25", "drift1")
            ]
            pel = [
                out["controls_new_boundaries"].get(f"{site}|{arm}|B28|{c}")
                for c in ("swing9", "swing15", "swing25", "drift1")
            ]
            print(site, arm, "cusum", vals, "pelt", pel)
    print("\n== bias -5 %")
    for k, v in out["bias_minus5_false_detections"].items():
        print(k, v)
    print("\n== hmm detection (block 7, 14-day and 60-day episodes)")
    for k, v in out["hmm_detection"].items():
        if "|B7|" in k and k.endswith(("|d14", "|d60")) and v["n"]:
            print(k, v["n"], v["share_found"], v["median_delay_if_found"], v["found_classes"])
    print("\n== episodes, 7-day blocks, share found (25% and one-turbine)")
    for site in common.SITES:
        for arm in ARMS:
            for size in ("p25", "t1"):
                cells = [
                    f"d{d}:{out['episodes_b7'][f'{site}|{arm}|{size}|d{d}'].get('share', float('nan')):.2f}"
                    for d in (3, 7, 14, 30, 60)
                ]
                print(site, arm, size, " ".join(cells))
    print_real()
    print("\n== arm E")
    for k, v in out["arm_e"].items():
        print(
            k,
            v
            if not isinstance(v, dict)
            else {
                a: (round(b, 2) if isinstance(b, float) else b)
                for a, b in v.items()
                if a in ("n", "share", "median_delay")
            },
        )


def print_real() -> None:
    """Print the preparation numbers, the real series' change points, and the lattice test."""
    prep = json.loads((common.WORK_DIR / "prep.json").read_text())
    real = json.loads((common.WORK_DIR / "real.json").read_text())
    print(
        "\n== strong set by margin (share of half-hours, share of days with any, mismatch, p999, max, visibility margin)"
    )
    for site in common.SITES:
        print(site, "v_sat", round(prep[site]["v_sat"], 2), "cut_out", round(prep[site]["cut_out"], 2),
              "sigma_v", round(prep[site]["noise"]["sigma_v"], 3), "sigma_p", round(prep[site]["noise"]["sigma_p"], 3))  # fmt: skip
        for m, v in prep[site]["margins"].items():
            print("  margin", m, f"share {v['share_of_half_hours']:.4f}", f"n {v['n_half_hours']}", f"days_any {v['share_of_days_with_any']:.3f}",
                  f"mismatch {v['mismatch_rate_under_50pct']:.3f}", f"p999 {v['p999_observed']:.3f}", f"max {v['max_observed']:.3f}",
                  f"visibility {v['visibility_margin_p999']:.3f}")  # fmt: skip
    print("\n== real series change points")
    for site in common.SITES:
        for key in ("D_B14", "A_p999@0_B28", "A_p999@2_B28", "C_B14", "B_B14"):
            seg = real[site]["segments"].get(key, {})
            if "segments" not in seg:
                print(site, key, "observed blocks", seg.get("n_observed_blocks"))
                continue
            print(
                site,
                key,
                seg["n_observed_blocks"],
                [(x["first_day"], x["last_day"], round(x["level"], 3)) for x in seg["segments"]],
            )
        print(
            site,
            "HMM full B14",
            [(x["state"], x["first_day"], x["last_day"]) for x in real[site]["hmm"]["full_B14"]],
        )
        print(
            site,
            "HMM lattice B14",
            [(x["state"], x["first_day"], x["last_day"]) for x in real[site]["hmm"]["lattice_B14"]],
        )
        print(
            site,
            "HMM full B7",
            [(x["state"], x["first_day"], x["last_day"]) for x in real[site]["hmm"]["full_B7"]],
        )
        print(
            site,
            "idle runs from experiment 210:",
            len(real[site]["outage_runs"]),
            [(r["first"], r["last"]) for r in real[site]["outage_runs"]][:30],
        )
        print(site, "p99 of power over capacity by year", real[site]["p99_by_year_of_capacity"])
        for key, v in real[site]["lattice"].items():
            print(
                site,
                "lattice",
                key,
                "steps",
                v["steps_share"],
                {
                    c: (
                        t["n"],
                        t["within_tenth"],
                        None if t["p_value"] is None else round(t["p_value"], 3),
                    )
                    for c, t in v["lattice"].items()
                },
            )
        print(
            site,
            "capacity state segments",
            [
                (x["first_day"], x["last_day"], x["state"], x["reduced_share"])
                for x in real[site]["capacity_state_segments"]
            ],
        )
    print(
        "arm E real",
        {
            k: [
                (x["first_day"], x["last_day"], round(x["level"], 3)) for x in v.get("segments", [])
            ]
            for k, v in real["arm_e_real"].items()
        },
    )


if __name__ == "__main__":
    main()
