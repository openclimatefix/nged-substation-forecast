# ruff: noqa: PLR0915, PLR0912
"""Run the six diagnostics of experiment 266 on stored forecasts and write `work/report.json`.

Intervals resample whole ISO weeks of forecast initialisation time, 20,000 draws, the same blocks
for every contrast. Nothing is refitted.

    systemd-run --user --scope -p MemoryMax=16G -p OOMPolicy=continue \
        uv run --no-sync python ml-experiments/00266-*/e00266_analyse.py
"""

import pickle

import numpy as np
import polars as pl

import e00266_common as common  # isort: skip

DECISION_SITES = ("D5", "D10", "D11", "D14", "D16")
RNG_DRAWS_REGRESSION = 1000


def blocks_of(*, m: pl.DataFrame) -> np.ndarray:
    """Return the ISO week label of each row's forecast initialisation time."""
    return m["power_fcst_init_time"].dt.strftime("%G-W%V").to_numpy()


def diff_interval(
    *,
    values: np.ndarray,
    mask_a: np.ndarray,
    mask_b: np.ndarray,
    blocks: np.ndarray,
    ratio: bool = False,
) -> dict:
    """Mean over `mask_a` minus (or divided by) mean over `mask_b`, with a week-block interval."""
    rng = np.random.default_rng(0)
    labels, codes = np.unique(blocks, return_inverse=True)
    n = len(labels)
    ok = np.isfinite(values)

    def sums(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        keep = mask & ok
        return (
            np.bincount(codes[keep], weights=values[keep], minlength=n),
            np.bincount(codes[keep], minlength=n).astype(np.float64),
        )

    sa, ca = sums(mask_a)
    sb, cb = sums(mask_b)
    draws = rng.integers(0, n, size=(common.N_RESAMPLES, n))
    with np.errstate(invalid="ignore", divide="ignore"):
        ma = sa[draws].sum(axis=1) / ca[draws].sum(axis=1)
        mb = sb[draws].sum(axis=1) / cb[draws].sum(axis=1)
        res = ma / mb if ratio else ma - mb
        est_a, est_b = sa.sum() / ca.sum(), sb.sum() / cb.sum()
    lo, hi = np.nanpercentile(res, [2.5, 97.5])
    return {
        "est": float(est_a / est_b if ratio else est_a - est_b),
        "lo": float(lo),
        "hi": float(hi),
    }


def load_arm(*, root: str, arm: str) -> pl.DataFrame:
    """Return an older GAM-K arm's stored scored forecasts, all sites."""
    files = sorted(
        common.ML_EXPERIMENTS_DATA_DIR.glob(f"{root}*/work/dp_preds/{arm}_scored_*.parquet")
    )
    return pl.concat(
        [pl.read_parquet(f).with_columns(site=pl.lit(f.stem.rsplit("_", 1)[-1])) for f in files]
    )


def main() -> None:
    """Compute every diagnostic."""
    m = common.read_rows()
    week = blocks_of(m=m)
    month = m["init_month"].to_numpy()
    site = m["site"].to_numpy()
    hour = m["valid_time"].dt.hour().to_numpy()
    lead = m["lead_hours"].to_numpy()
    actual = m["actual"].to_numpy().astype(np.float64)
    unit = m["unit"].to_numpy()
    daylight = (m["clear"].to_numpy() > 20.0) & np.isfinite(m["csi"].to_numpy())
    night = ~daylight
    real = m["realised_csi"].to_numpy()
    ens = m["csi"].to_numpy()
    windows = {
        "AMJ": np.isin(month, common.AMJ),
        "JAS": np.isin(month, common.JAS),
        "NDJFM": np.isin(month, common.NDJFM),
    }
    q = {k: m[f"qs_{k}"].to_numpy() for k in (*common.STACK_ARMS, "manual_heuristic")}
    gap = q["arm1"] - q["bg2"]
    report: dict = {
        "rows": m.height,
        "weeks": {w: len(np.unique(week[mask])) for w, mask in windows.items()},
    }

    # Gate P0.
    report["gate_monthly"] = {
        mo: common.mean_interval(values=gap, mask=month == mo, blocks=week)
        for mo in sorted(set(month))
    }
    # Diagnostic 1: error map.
    d1: dict = {}
    real_bands = {
        "below 0.3": real < 0.3,
        "0.3 to 0.6": (real >= 0.3) & (real < 0.6),
        "0.6 to 0.9": (real >= 0.6) & (real < 0.9),
        "0.9 and above": real >= 0.9,
    }
    ens_bands = {
        "below 0.3": ens < 0.3,
        "0.3 to 0.6": (ens >= 0.3) & (ens < 0.6),
        "0.6 to 0.9": (ens >= 0.6) & (ens < 0.9),
        "0.9 and above": ens >= 0.9,
    }
    for w, wm in windows.items():
        cuts = {"all": np.ones_like(wm), "night": night, "daylight": daylight}
        cuts |= {f"realised csi {k}": daylight & v for k, v in real_bands.items()}
        cuts |= {f"ensemble csi {k}": daylight & v for k, v in ens_bands.items()}
        cuts |= {f"utc hour {h:02d}": daylight & (hour == h) for h in range(5, 20)}
        cuts |= {f"site {s}": np.asarray(site == s) for s in common.SITES}
        cuts |= {f"daylight site {s}": daylight & (site == s) for s in common.SITES}
        d1[w] = {
            name: {
                **common.mean_interval(values=gap, mask=wm & c, blocks=week),
                "share_of_rows": float((wm & c).sum() / wm.sum()),
            }
            for name, c in cuts.items()
        }
        d1[w]["bg2_qs_daylight"] = float(q["bg2"][wm & daylight].mean())
        d1[w]["arm1_qs_daylight"] = float(q["arm1"][wm & daylight].mean())
    amj = windows["AMJ"]
    d1["P1_daylight_over_night"] = diff_interval(
        values=gap, mask_a=amj & daylight, mask_b=amj & night, blocks=week, ratio=True
    )
    d1["P1_daylight_minus_night"] = diff_interval(
        values=gap, mask_a=amj & daylight, mask_b=amj & night, blocks=week
    )
    report["d1"] = d1

    # Diagnostic 4: weather input, one cut for P4.
    delta = np.abs(ens - real)
    agree, disagree = daylight & (delta <= 0.1), daylight & (delta > 0.2)
    middle = daylight & (delta > 0.1) & (delta <= 0.2)
    d4: dict = {
        w: {
            name: {
                **common.mean_interval(values=gap, mask=wm & c, blocks=week),
                "share_of_rows": float((wm & c).sum() / wm.sum()),
            }
            for name, c in (("agree", agree), ("middle", middle), ("disagree", disagree))
        }
        for w, wm in windows.items()
    }
    d4["P4_agree_over_disagree_AMJ"] = diff_interval(
        values=gap, mask_a=amj & agree, mask_b=amj & disagree, blocks=week, ratio=True
    )
    d4["disagree_minus_agree_AMJ"] = diff_interval(
        values=gap, mask_a=amj & disagree, mask_b=amj & agree, blocks=week
    )
    d4["ensemble_minus_realised_csi_mean"] = {
        w: float(np.nanmean((ens - real)[wm & daylight])) for w, wm in windows.items()
    }
    d4["ensemble_minus_realised_csi_mean_clear"] = {
        w: float(np.nanmean((ens - real)[wm & daylight & (real >= 0.8)]))
        for w, wm in windows.items()
    }
    report["d4"] = d4

    # Diagnostic 6: ablations.
    d6 = {}
    for w in ("AMJ", "NDJFM", "JAS"):
        wm = windows[w]
        d6[w] = {
            "arm1_minus_a_s (no head)": common.mean_interval(
                values=q["arm1"] - q["a_s"], mask=wm, blocks=week
            ),
            "arm1_minus_dh (no summary)": common.mean_interval(
                values=q["arm1"] - q["dh"], mask=wm, blocks=week
            ),
            "arm1_minus_d (neither)": common.mean_interval(
                values=q["arm1"] - q["d"], mask=wm, blocks=week
            ),
            "d_minus_bg2": common.mean_interval(values=q["d"] - q["bg2"], mask=wm, blocks=week),
            "a_s_minus_bg2": common.mean_interval(values=q["a_s"] - q["bg2"], mask=wm, blocks=week),
            "dh_minus_bg2": common.mean_interval(values=q["dh"] - q["bg2"], mask=wm, blocks=week),
            "arm1_minus_bg2": common.mean_interval(values=gap, mask=wm, blocks=week),
            "arm1_minus_bg2_daylight": common.mean_interval(
                values=gap, mask=wm & daylight, blocks=week
            ),
            "d_minus_bg2_daylight": common.mean_interval(
                values=q["d"] - q["bg2"], mask=wm & daylight, blocks=week
            ),
            "d_minus_bg2_night": common.mean_interval(
                values=q["d"] - q["bg2"], mask=wm & night, blocks=week
            ),
            "arm1_minus_bg2_night": common.mean_interval(values=gap, mask=wm & night, blocks=week),
        }
    d6["by_site_AMJ"] = {
        s: {
            "arm1_minus_bg2": common.mean_interval(values=gap, mask=amj & (site == s), blocks=week),
            "d_minus_bg2": common.mean_interval(
                values=q["d"] - q["bg2"], mask=amj & (site == s), blocks=week
            ),
            "arm1_minus_a_s": common.mean_interval(
                values=q["arm1"] - q["a_s"], mask=amj & (site == s), blocks=week
            ),
        }
        for s in common.SITES
    }
    report["d6"] = d6

    # Diagnostic 2: older GAM-K arms (experiment 139 and 111) on the same keys.
    keys = ["site", "run", "valid_time"]
    d2: dict = {"row_counts": {}}
    old: dict[str, np.ndarray] = {}
    old_present: dict[str, np.ndarray] = {}
    for root, arm in (
        ("00139", "pin"),
        ("00139", "nopv"),
        ("00139", "asof"),
        ("00139", "hindsight"),
        ("00111", "gamk"),
        ("00111", "gam0"),
        ("00111", "gamk_era5"),
    ):
        d = load_arm(root=root, arm=arm)
        joined = (
            m.select(keys)
            .with_row_index("i")
            .join(d.select(*keys, *common.QUANTILES), on=keys, how="left", validate="1:1")
        )
        have = joined["q50"].is_not_null().to_numpy()
        qq = joined.select(list(common.QUANTILES)).fill_null(0.0).to_numpy()
        score = np.where(have, common.qs_per_row(actual=actual, q=qq), np.nan)
        name = f"{root}_{arm}"
        old[name], old_present[name] = score, have
        d2["row_counts"][name] = {"rows_of_master": int(have.sum()), "of": m.height}
    common_rows = np.logical_and.reduce(
        [old_present[k] for k in ("00139_pin", "00139_nopv", "00111_gamk", "00111_gam0")]
    )
    d2["common_rows"] = int(common_rows.sum())
    d2["pin_qs_five_decision_sites"] = float(
        np.nanmean(old["00139_pin"][np.isin(site, DECISION_SITES)])
    )
    for w, wm in windows.items():
        d2[w] = {}
        for name, s in old.items():
            ok = old_present[name]
            d2[w][f"{name}_minus_bg2"] = common.mean_interval(
                values=s - q["bg2"], mask=wm & ok, blocks=week
            )
        d2[w]["solar_value_nopv_minus_pin_all"] = common.mean_interval(
            values=old["00139_nopv"] - old["00139_pin"], mask=wm, blocks=week
        )
        d2[w]["solar_value_nopv_minus_pin_daylight"] = common.mean_interval(
            values=old["00139_nopv"] - old["00139_pin"], mask=wm & daylight, blocks=week
        )
        d2[w]["solar_value_nopv_minus_pin_sunny"] = common.mean_interval(
            values=old["00139_nopv"] - old["00139_pin"],
            mask=wm & m["sunny"].to_numpy(),
            blocks=week,
        )
        d2[w]["kalman_gamk_minus_gam0"] = common.mean_interval(
            values=old["00111_gamk"] - old["00111_gam0"], mask=wm, blocks=week
        )
        d2[w]["era5_filter_minus_gamk"] = common.mean_interval(
            values=old["00111_gamk_era5"] - old["00111_gamk"], mask=wm, blocks=week
        )
    sv = old["00139_nopv"] - old["00139_pin"]
    d2["P2_solar_value_JAS_minus_AMJ_all"] = diff_interval(
        values=sv, mask_a=windows["JAS"], mask_b=amj, blocks=week
    )
    d2["P2_solar_value_JAS_minus_AMJ_sunny"] = diff_interval(
        values=sv,
        mask_a=windows["JAS"] & m["sunny"].to_numpy(),
        mask_b=amj & m["sunny"].to_numpy(),
        blocks=week,
    )
    d2["by_site_AMJ"] = {
        s: {
            "solar_value_nopv_minus_pin": common.mean_interval(
                values=sv, mask=amj & (site == s), blocks=week
            ),
            "pin_minus_bg2": common.mean_interval(
                values=old["00139_pin"] - q["bg2"], mask=amj & (site == s), blocks=week
            ),
        }
        for s in common.SITES
    }
    report["d2"] = d2

    # Diagnostic 3: capacity held by the pin and the capacity the residual implies.
    held: dict[tuple[str, str], float] = {}
    for s in common.SITES:
        with (common.E139_DIR / "fits" / "pin" / f"{s}.pkl").open("rb") as handle:
            fits = pickle.load(handle)
        cuts = sorted((f.cut, f.pinned.held_points) for f in fits)
        for mo in sorted(set(month)):
            start = np.datetime64(f"{mo}-01")
            usable = [h for c, h in cuts if np.datetime64(c.replace(tzinfo=None), "D") <= start]
            held[(s, mo)] = float(usable[-1]) if usable else float("nan")
    med_d = m["med_d"].to_numpy().astype(np.float64)
    med_arm1 = m["med_arm1"].to_numpy().astype(np.float64)
    clear_mid = daylight & (real > 0.8) & (ens > 0.8) & (unit >= 0.3) & (lead >= 12) & (lead <= 36)

    def slope_interval(*, mask: np.ndarray, med: np.ndarray) -> dict:
        res = (actual - med)[mask]
        x = unit[mask]
        b = week[mask]
        labels, codes = np.unique(b, return_inverse=True)
        n = len(labels)
        if n < 4 or mask.sum() < 50:
            return {
                "slope": float("nan"),
                "lo": float("nan"),
                "hi": float("nan"),
                "n": int(mask.sum()),
            }
        stats = np.stack(
            [
                np.bincount(codes, weights=w, minlength=n)
                for w in (np.ones_like(x), x, x * x, res, x * res)
            ]
        )
        rng = np.random.default_rng(0)
        draws = rng.integers(0, n, size=(RNG_DRAWS_REGRESSION, n))

        def solve(t: np.ndarray) -> np.ndarray:
            c, sx, sxx, sy, sxy = t
            return (c * sxy - sx * sy) / (c * sxx - sx * sx)

        est = solve(stats.sum(axis=1))
        boot = solve(np.stack([stats[:, d].sum(axis=1) for d in draws], axis=1))
        lo, hi = np.nanpercentile(boot, [2.5, 97.5])
        return {"slope": float(est), "lo": float(lo), "hi": float(hi), "n": int(mask.sum())}

    d3: dict = {"by_site_window": {}, "by_site_month": {}}
    for s in common.SITES:
        d3["by_site_window"][s] = {}
        for w, wm in windows.items():
            mask = clear_mid & wm & (site == s)
            cap = float(
                np.nanmean(
                    [
                        held[(s, mo)]
                        for mo in {"AMJ": common.AMJ, "JAS": common.JAS, "NDJFM": common.NDJFM}[w]
                    ]
                )
            )
            for arm_name, med in (("arm_d", med_d), ("arm1", med_arm1)):
                r = slope_interval(mask=mask, med=med)
                corr = {
                    k: -r[k2]
                    for k, k2 in (
                        ("correction_points", "slope"),
                        ("correction_lo", "hi"),
                        ("correction_hi", "lo"),
                    )
                }
                d3["by_site_window"][s].setdefault(w, {})[arm_name] = {
                    **r,
                    **corr,
                    "pinned_points": cap,
                    "correction_pct": corr["correction_points"] / cap * 100.0,
                    "pct_lo": corr["correction_lo"] / cap * 100.0,
                    "pct_hi": corr["correction_hi"] / cap * 100.0,
                }
        d3["by_site_month"][s] = {}
        for mo in sorted(set(month)):
            r = slope_interval(mask=clear_mid & (month == mo) & (site == s), med=med_d)
            cap = held[(s, mo)]
            d3["by_site_month"][s][mo] = {
                "correction_pct": -r["slope"] / cap * 100.0,
                "pinned_points": cap,
                "n": r["n"],
            }
    report["d3"] = d3

    # Diagnostic 5: residual shape at clear midday.
    res1 = actual - med_arm1
    d5: dict = {"by_site_window": {}, "by_month": {}}
    mid = daylight & (real > 0.8) & (ens > 0.8) & (unit >= 0.7)
    morning, afternoon = hour < 12, hour >= 12
    for s in common.SITES:
        d5["by_site_window"][s] = {}
        for w, wm in windows.items():
            base = mid & wm & (site == s)
            mo_ = common.mean_interval(values=res1, mask=base & morning, blocks=week)
            af_ = common.mean_interval(values=res1, mask=base & afternoon, blocks=week)
            denom = (abs(mo_["est"]) + abs(af_["est"])) / 2.0
            d5["by_site_window"][s][w] = {
                "morning": mo_,
                "afternoon": af_,
                "asymmetry": abs(mo_["est"] - af_["est"]) / denom if denom > 0 else float("nan"),
            }
    for mo in sorted(set(month)):
        d5["by_month"][mo] = {
            s: common.mean_interval(
                values=res1, mask=mid & (month == mo) & (site == s), blocks=week
            )
            for s in common.SITES
        }
    d5["all_sites_AMJ"] = {
        "morning": common.mean_interval(values=res1, mask=mid & amj & morning, blocks=week),
        "afternoon": common.mean_interval(values=res1, mask=mid & amj & afternoon, blocks=week),
    }
    d5["all_sites_NDJFM"] = {
        "morning": common.mean_interval(
            values=res1, mask=mid & windows["NDJFM"] & morning, blocks=week
        ),
        "afternoon": common.mean_interval(
            values=res1, mask=mid & windows["NDJFM"] & afternoon, blocks=week
        ),
    }
    # D5 sunny: arm 1 and arm 2c against BG2.
    arm2c = np.load(common.E251_SCORED / "arm2c.npy")
    q2c = common.qs_per_row(actual=actual, q=arm2c)
    sunny = m["sunny"].to_numpy()
    d5["d5_sunny"] = {
        "arm1_minus_bg2_year": common.mean_interval(
            values=gap, mask=sunny & daylight & (site == "D5"), blocks=week
        ),
        "arm2c_minus_bg2_year": common.mean_interval(
            values=q2c - q["bg2"], mask=sunny & daylight & (site == "D5"), blocks=week
        ),
        "arm1_minus_bg2_AMJ": common.mean_interval(
            values=gap, mask=sunny & daylight & (site == "D5") & amj, blocks=week
        ),
        "arm2c_minus_bg2_AMJ": common.mean_interval(
            values=q2c - q["bg2"], mask=sunny & daylight & (site == "D5") & amj, blocks=week
        ),
        "arm1_minus_bg2_NDJFM": common.mean_interval(
            values=gap, mask=sunny & daylight & (site == "D5") & windows["NDJFM"], blocks=week
        ),
    }
    report["d5"] = d5

    # Post hoc: which substations carry the gap, morning against afternoon, and the checks.
    ph: dict = {"daylight_night_by_site_AMJ": {}, "leave_one_site_out_AMJ": {}}
    for s_ in common.SITES:
        ph["daylight_night_by_site_AMJ"][s_] = {
            "daylight": common.mean_interval(
                values=gap, mask=amj & daylight & (site == s_), blocks=week
            ),
            "night": common.mean_interval(values=gap, mask=amj & night & (site == s_), blocks=week),
        }
        ph["leave_one_site_out_AMJ"][s_] = common.mean_interval(
            values=gap, mask=amj & (site != s_), blocks=week
        )
    carriers = np.isin(site, ("D5", "D9", "D10"))
    ph["gap_AMJ_carriers_D5_D9_D10"] = common.mean_interval(
        values=gap, mask=amj & carriers, blocks=week
    )
    ph["gap_AMJ_others"] = common.mean_interval(values=gap, mask=amj & ~carriers, blocks=week)
    ph["morning_minus_afternoon_residual_AMJ"] = diff_interval(
        values=res1, mask_a=mid & amj & morning, mask_b=mid & amj & afternoon, blocks=week
    )
    ph["gap_by_hour_group_AMJ"] = {
        name: common.mean_interval(values=gap, mask=amj & daylight & sel, blocks=week)
        for name, sel in (
            ("08-11 UTC", (hour >= 8) & (hour < 12)),
            ("12-15 UTC", (hour >= 12) & (hour < 16)),
        )
    }
    mae1, mae_b = np.abs(actual - med_arm1), np.abs(actual - m["med_bg2"].to_numpy())
    ph["checks_mae_median_arm1_minus_bg2"] = {
        w: common.mean_interval(values=mae1 - mae_b, mask=wm, blocks=week)
        for w, wm in windows.items()
    }
    mean_d = m["mean_d"].to_numpy().astype(np.float64) if "mean_d" in m.columns else None
    ph["median_minus_mean_forecast_arm_d_by_site"] = (
        {s_: float(np.mean((m["med_d"].to_numpy() - mean_d)[site == s_])) for s_ in common.SITES}
        if mean_d is not None
        else None
    )
    report["post_hoc"] = ph
    common.write_json_atomic(path=common.WORK_DIR / "report.json", content=report)
    print("written", common.WORK_DIR / "report.json")


if __name__ == "__main__":
    main()
