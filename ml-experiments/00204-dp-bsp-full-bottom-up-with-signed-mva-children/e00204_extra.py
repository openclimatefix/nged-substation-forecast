"""Controls and the error decomposition of experiment 204; writes `work/extra.json`.

1. **Reproduction.** Experiment 161's top-down and partial bottom-up scores on experiment 161's own
   rows (its frames, with its mask), against its stored report.
2. **Known-answer check.** For the five real-power Primaries of experiment 152's injection test,
   the capacity-weighted sum of the signed truth against the sum of arm N's mean (arm (c)) and of
   arm F's latent flow times the power factor (arm (d)), by injection level.
3. **Decomposition.** At each BSP, the standard deviation of the top-down error and of the error of
   the remainder and of the weighted children in arm (d), on the scored rows.

    uv run --no-sync python ml-experiments/00204-*/e00204_extra.py
"""

import json
from typing import Final, cast

import e00204_bu as bu
import e00204_common as common
import e00204_refs as refs
import e00204_score as score
import numpy as np
import polars as pl

KNOWN_SITES: Final[tuple[str, ...]] = ("D5", "D10", "D11", "D14", "D16")
LEVELS: Final[tuple[str, ...]] = ("0.0", "1.0", "2.0")
KEYS: Final[list[str]] = bu.KEYS


def _mean(series: pl.Series) -> float:
    """Return a numeric series' mean as a float."""
    return cast(float, series.mean())


def known_dir(*, site: str, level: str, mode: str):  # noqa: ANN201
    """Return the stored forecasts of the injection test at a site, level, and folding mode."""
    return (
        common.E152_WORK_DIR / "preds" / f"known_{site}_L{level}_{mode}" / f"scored_{site}.parquet"
    )


def known_answer(*, cache) -> dict:  # noqa: ANN001
    """Return the mean absolute error of the summed signed flow for (c) and (d)."""
    out: dict = {}
    for level in LEVELS:
        for mode in ("pf97", "q0"):
            if level == "0.0" and mode == "pf97":
                continue  # experiment 152 stored level 0 under q0 only
            truth = None
            naive = None
            fold = None
            for site in KNOWN_SITES:
                cap = cache.sites[site].table.effective_capacity_mw
                signed = (
                    pl.read_parquet(known_dir(site=site, level=level, mode="signed"))
                    .select(*KEYS, t=pl.col("actual") * cap / 100.0)
                    .rename({"t": f"t_{site}"})
                )
                folded = pl.read_parquet(known_dir(site=site, level=level, mode=mode))
                factor = common.POWER_FACTOR if mode == "pf97" else 1.0
                n = (
                    folded.filter(pl.col("arm") == "N")
                    .select(*KEYS, n=pl.col("mean") * cap / 100.0)
                    .rename({"n": f"n_{site}"})
                )
                f = (
                    folded.filter(pl.col("arm") == "F_fold")
                    .select(*KEYS, f=pl.col("flow") * factor * cap / 100.0)
                    .rename({"f": f"f_{site}"})
                )
                part = signed.join(n, on=KEYS).join(f, on=KEYS)
                truth = part if truth is None else truth.join(part, on=KEYS)
            assert truth is not None
            del naive, fold
            frame = truth.drop_nulls()
            t = np.sum([frame[f"t_{s}"].to_numpy() for s in KNOWN_SITES], axis=0)
            n = np.sum([frame[f"n_{s}"].to_numpy() for s in KNOWN_SITES], axis=0)
            f = np.sum([frame[f"f_{s}"].to_numpy() for s in KNOWN_SITES], axis=0)
            out[f"L{level}_{mode}"] = {
                "rows": frame.height,
                "mae_c_mw": float(np.abs(n - t).mean()),
                "mae_d_mw": float(np.abs(f - t).mean()),
                "bias_c_mw": float((n - t).mean()),
                "bias_d_mw": float((f - t).mean()),
            }
    return out


def reproduction() -> dict:
    """Score experiment 161's arms on experiment 161's own rows (its mask, only its children)."""
    out = {}
    for bsp in common.BSPS:
        frames = {}
        for arm in ("a_td", "b_partial", "b_partial_tdspread"):
            frames[arm] = score.common_rows(
                frames={
                    arm: score.scorable(frame=score.read_arm(arm=arm, bsp=bsp, period="scored"))
                }
            )[arm]
        keys = (
            frames["a_td"]
            .select(score.KEYS)
            .join(frames["b_partial"].select(score.KEYS), on=score.KEYS, how="inner")
        )
        out[bsp] = {}
        for arm, frame in frames.items():
            part = score.with_losses(frame=frame.join(keys, on=score.KEYS, how="inner"))
            out[bsp][arm] = {"rows": part.height, "qs": _mean(part["qs"])}
    return out


def decomposition(*, cache) -> dict:  # noqa: ANN001
    """Return the standard deviation of the top-down error and of arm (d)'s two parts, in points."""
    out = {}
    for bsp in common.BSPS:
        cap = cache.sites[bsp].table.effective_capacity_mw
        weights = common.weights_for(cache=cache, bsp=bsp, variant="fold")
        td = pl.read_parquet(bu.td_path(period="scored", bsp=bsp))
        remainder = pl.read_parquet(
            bu.remainder_path(period="scored", bsp=bsp, variant="fold")
        ).select(*KEYS, mean_r=pl.col("mean"), actual_r=pl.col("actual"))
        frame = td.join(remainder, on=KEYS, how="inner")
        masks = []
        for child in common.BSPS[bsp]:
            part = bu.read_child(child=child, period="scored", mode="fold")
            assert part is not None
            frame = frame.join(part, on=KEYS, how="inner")
            masks.append(pl.col(f"mask_{child}"))
        frame = score.scorable(
            frame=frame.with_columns(
                switching_mask=pl.col("switching_mask") | pl.any_horizontal(*masks)
            )
        )
        children = np.zeros(frame.height)
        for child, weight in weights.items():
            scale = cache.sites[child].table.effective_capacity_mw / cap
            actual = (
                bu.signed_actual(
                    cache=cache, child=child, valid_time=frame["valid_time"], variant="fold"
                )
                if child in common.ALL_MVA
                else frame[f"actual_{child}"].to_numpy()
            )
            children += weight * scale * (actual - frame[f"mean_{child}"].to_numpy())
        error_r = (frame["actual_r"] - frame["mean_r"]).to_numpy().astype(np.float64)
        total = (frame["actual"] - frame["mean"]).to_numpy().astype(np.float64)
        ok = np.isfinite(children) & np.isfinite(error_r)
        out[bsp] = {
            "rows": int(ok.sum()),
            "sd_top_down_error": float(total[ok].std()),
            "sd_remainder_error": float(error_r[ok].std()),
            "sd_children_error": float(children[ok].std()),
            "corr_remainder_children": float(np.corrcoef(error_r[ok], children[ok])[0, 1]),
            "sd_bottom_up_error": float((error_r + children)[ok].std()),
        }
    return out


def references(*, cache) -> dict:  # noqa: ANN001
    """Return climatology's and the manual heuristic's score on the top-down arm's scored rows."""
    out = {}
    for bsp in common.BSPS:
        frames = score.common_rows(
            frames={
                arm: score.scorable(frame=score.read_arm(arm=arm, bsp=bsp, period="scored"))
                for arm in ("a_td", "c", "d", "oracle")
            }
        )
        td = frames["a_td"]
        mh = refs.manual_heuristic(cache=cache, bsp=bsp, frame=td)
        keep = td.join(mh.select(score.KEYS), on=score.KEYS, how="inner")
        out[bsp] = {
            "rows": keep.height,
            "td": _mean(score.with_losses(frame=keep)["qs"]),
            "climatology": _mean(
                score.with_losses(frame=refs.climatology(cache=cache, bsp=bsp, frame=keep))["qs"]
            ),
            "manual_heuristic": _mean(
                score.with_losses(
                    frame=mh.join(keep.select(score.KEYS), on=score.KEYS, how="inner")
                )["qs"]
            ),
        }
    return out


def main() -> None:
    """Run the three parts and write `extra.json`."""
    cache = common.load_cache()
    result = {
        "reproduction": reproduction(),
        "known_answer": known_answer(cache=cache),
        "decomposition": decomposition(cache=cache),
        "references": references(cache=cache),
    }
    common.write_json_atomic(path=common.WORK_DIR / "extra.json", content=result)
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
