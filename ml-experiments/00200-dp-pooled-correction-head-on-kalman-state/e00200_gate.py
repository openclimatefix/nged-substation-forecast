"""Experiment 200, stage 1: the known-answer gate on the 6 stored substations.

A +3-point step for 8 weeks, and a +3-point shift on Saturdays 17:00 to 21:00 UTC for 8 weeks, are
added to the real errors of one block. The head is trained on clean rows outside the block (purged
by 15 days) and must raise its correction when the injected errors reach its inputs. Recovery is
the mean rise in the correction divided by 3, from the third day (step) or the third Saturday.
Run: `uv run --no-sync python e00200_gate.py [--device cuda]`.
"""

import argparse
import json
from typing import Final

import e00200_data as data_module
import e00200_model as model_module
import numpy as np

SITES: Final[tuple[int, ...]] = (7, 8, 10, 16, 17, 18)
INJECTION: Final[float] = 3.0
SEEDS: Final[tuple[int, ...]] = (0, 1, 2)
PURGE_DAYS: Final[int] = 15
STEP_BLOCK: Final[tuple[str, str]] = ("2025-04-07", "2025-06-02")
SATURDAY_BLOCK: Final[tuple[str, str]] = ("2025-11-03", "2025-12-29")
VALID_BLOCK: Final[tuple[str, str]] = ("2025-01-06", "2025-02-23")


def day(text: str) -> np.datetime64:
    return np.datetime64(text, "us")


def in_block(*, t: np.ndarray, block: tuple[str, str], pad_days: int = 0) -> np.ndarray:
    pad = np.timedelta64(pad_days, "D")
    return (t >= day(block[0]) - pad) & (t < day(block[1]) + pad)


def inject(*, site: data_module.Site, kind: str) -> np.ndarray:
    """Copy of the site's error grid with the known-answer injection added."""
    grid = site.error_grid.copy()
    times = site.t0 + np.arange(len(grid)) * np.timedelta64(30, "m")
    if kind == "step":
        grid[in_block(t=times, block=STEP_BLOCK)] += INJECTION
    else:
        days = times.astype("datetime64[D]")
        weekday = (days.astype(np.int64) + 3) % 7
        minute = ((times - days) // np.timedelta64(1, "m")).astype(np.int64)
        saturday = (weekday == 5) & (minute >= 17 * 60) & (minute < 21 * 60)
        grid[in_block(t=times, block=SATURDAY_BLOCK) & saturday] += INJECTION
    return grid


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    device = parser.parse_args().device
    data_module.WORK_DIR.mkdir(parents=True, exist_ok=True)
    sites = [data_module.load_site(time_series_id=i) for i in SITES]
    clean = [data_module.build_features(site=s, error_grid=s.error_grid) for s in sites]
    n_rows = sum(len(f["target"]) for f in clean)
    print(f"rows (site, run, valid time) at leads 12-48 h: {n_rows}")
    report: dict[str, object] = {"rows": n_rows, "sites": len(sites)}

    for kind, block in (("step", STEP_BLOCK), ("saturday", SATURDAY_BLOCK)):
        injected = [
            data_module.build_features(site=s, error_grid=inject(site=s, kind=kind)) for s in sites
        ]
        xs, ids, ys, offs, masks = [], [], [], [], []
        for k, f in enumerate(clean):
            blocked = in_block(t=f["valid"], block=block, pad_days=PURGE_DAYS) | in_block(
                t=f["init"], block=block, pad_days=PURGE_DAYS
            )
            valid_set = in_block(t=f["init"], block=VALID_BLOCK) & ~blocked
            purge_valid = in_block(t=f["init"], block=VALID_BLOCK, pad_days=PURGE_DAYS) & ~valid_set
            train_set = ~blocked & ~valid_set & ~purge_valid
            xs.append(model_module.design(f=f))
            ids.append(np.full(len(f["target"]), k, dtype=np.int64))
            ys.append(f["target"])
            offs.append(f["q_offsets"])
            masks.append((train_set, valid_set))
        take = lambda which: model_module.Data(  # noqa: E731
            x=np.concatenate([x[m[which]] for x, m in zip(xs, masks, strict=True)]),
            site=np.concatenate([i[m[which]] for i, m in zip(ids, masks, strict=True)]),
            y=np.concatenate([y[m[which]] for y, m in zip(ys, masks, strict=True)]),
            offsets=np.concatenate([o[m[which]] for o, m in zip(offs, masks, strict=True)]),
        )
        train, valid = take(0), take(1)
        print(f"{kind}: train rows {len(train.y)}, validation rows {len(valid.y)}")

        recoveries, r_recoveries, fit_maes = [], [], []
        for seed in SEEDS:
            net = model_module.fit(
                train=train, valid=valid, n_sites=len(sites), seed=seed, device=device
            )
            rise, r_rise, n_scored = [], [], []
            for k, (c, i) in enumerate(zip(clean, injected, strict=True)):
                score = (c["lead"] >= data_module.SCORE_LEAD_MIN) & (
                    c["lead"] <= data_module.SCORE_LEAD_MAX
                )
                if kind == "step":
                    sel = in_block(
                        t=c["valid"],
                        block=(str(day(block[0]) + np.timedelta64(3, "D"))[:10], block[1]),
                    )
                else:
                    days = c["valid"].astype("datetime64[D]")
                    minute = ((c["valid"] - days) // np.timedelta64(1, "m")).astype(np.int64)
                    saturday = (
                        ((days.astype(np.int64) + 3) % 7 == 5)
                        & (minute >= 17 * 60)
                        & (minute < 21 * 60)
                    )
                    third = str(day(block[0]) + np.timedelta64(14, "D"))[:10]
                    sel = saturday & in_block(t=c["valid"], block=(third, block[1]))
                sel = sel & score
                c_clean, _ = model_module.predict(
                    net=net,
                    x=model_module.design(f=c)[sel],
                    site=np.full(sel.sum(), k),
                    device=device,
                )
                c_inj, _ = model_module.predict(
                    net=net,
                    x=model_module.design(f=i)[sel],
                    site=np.full(sel.sum(), k),
                    device=device,
                )
                rise.append(c_inj - c_clean)
                r_rise.append(data_module.R_KAPPA * (i["profile"][sel] - c["profile"][sel]))
                fit_maes.append(
                    float(np.mean(np.abs(c["target"][sel] - c_clean)))
                    - float(np.mean(np.abs(c["target"][sel])))
                )
            recoveries.append(float(np.mean(np.concatenate(rise))) / INJECTION)
            r_recoveries.append(float(np.mean(np.concatenate(r_rise))) / INJECTION)
            print(
                f"  {kind} seed {seed}: head recovery {recoveries[-1]:+.1%}, recipe R {r_recoveries[-1]:+.1%}"
            )
        report[kind] = {
            "head_recovery_by_seed": recoveries,
            "head_recovery_mean": float(np.mean(recoveries)),
            "recipe_r_recovery": float(np.mean(r_recoveries)),
            "train_rows": len(train.y),
            "head_mae_minus_arm_d_mae_clean_block": float(np.mean(fit_maes)),
        }
    (data_module.WORK_DIR / "gate.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
