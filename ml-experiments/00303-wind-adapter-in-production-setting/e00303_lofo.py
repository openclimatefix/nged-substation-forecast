"""Leave-one-farm-out for P3 and P5 of experiment 00303: a farm with no history.

For each held-out farm, one core and adapter is fitted to the ERA5 rows of the other two farms
pooled (no idle layer, because a farm with no history has no way to say whether it needs one), and
run on the held-out farm's ensemble members. The held-out farm contributes no observed power to any
step: the residual quantiles, the spread factor (one factor for leads under 72 hours and one for
later leads, fitted once on the other farms' hindcast), and the offset table (pooled over the other
two farms by lead day) all come from the other two farms.

- `P3_LOFO`: the ERA5-fitted adapter A4, the fixed spread factor, and the pooled offset table.
- `P5_LOFO`: the same core fine-tuned on the other two farms' hindcast (the adapter also sees lead
  and spread), stopping on their last 90 days. Residual quantiles, the spread factor, and the
  fine-tuned model's early-stopping set are the other farms' last 90 days of cases.

Stores `work/final/P3_LOFO.npy` and `P5_LOFO.npy` (3 seeds, scored cases, 11), in the order of
`e00303_score.ordered_cases`. The pooled no-history pipeline is compared with the per-farm P3 and P5
in the score script, and the idle-layer simplification is part of that contrast.

    systemd-run --user --scope -p MemoryMax=16G -p OOMPolicy=continue \
      uv run --no-sync python e00303_lofo.py
"""

import os
from typing import Final

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_name, "2")
os.environ.setdefault("E00303_TORCH_THREADS", "2")

import e00303_common as common  # noqa: E402
import e00303_core as core_module  # noqa: E402
import e00303_ens_final as ens_final  # noqa: E402
import e00303_fit as fit  # noqa: E402
import e00303_models as models  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

FARMS: Final[tuple[str, ...]] = ens_final.FARMS
SEEDS: Final[tuple[int, ...]] = (0, 1, 2)


def scorable(*, mixture: np.ndarray) -> np.ndarray:
    """Return the cases whose 95th percentile is above the trivial level."""
    return mixture[:, common.UPPER_INDEX] > common.TRIVIAL_LEVEL


def fixed_spread(
    *, mixture: np.ndarray, actual: np.ndarray, lead_hours: np.ndarray
) -> dict[bool, float]:
    """Return the spread factor of the short band (True) and the long band (False)."""
    factors = {}
    for short in (True, False):
        chosen = ((lead_hours < common.SHORT_BAND_HOURS) == short) & scorable(mixture=mixture)
        factors[short] = common.fit_spread_factor(
            quantiles=mixture[chosen], actual=actual[chosen]
        )
    return factors


def apply_fixed_spread(
    *, mixture: np.ndarray, lead_hours: np.ndarray, factors: dict[bool, float]
) -> np.ndarray:
    """Return `mixture` after the short-band and long-band spread factors."""
    out = mixture.copy()
    short = lead_hours < common.SHORT_BAND_HOURS
    for band, factor in factors.items():
        chosen = (short == band) & scorable(mixture=mixture)
        out[chosen] = common.scale_about_median(quantiles=mixture[chosen], factor=factor)
    return out


def pooled_offsets(*, quantiles: np.ndarray, actual: np.ndarray, lead_day: np.ndarray) -> np.ndarray:
    """Return the pinball-optimal offsets per lead day (0 to 15) pooled over the cases."""
    table = np.zeros((ens_final.ens.MAX_LEAD_DAY + 1, 11))
    for day in range(ens_final.ens.MAX_LEAD_DAY + 1):
        chosen = lead_day == day
        if chosen.sum() >= 100:
            residual = actual[chosen, None] - quantiles[chosen]
            table[day] = np.array(
                [np.quantile(residual[:, k], common.LEVELS[k]) for k in range(11)]
            )
    return table


def pooled_train(*, data: list[fit.Hindcast]) -> pl.DataFrame:
    """Return the ERA5 training rows of several farms stacked."""
    return pl.concat([d.train for d in data])


def main() -> None:  # noqa: PLR0915
    """Run both arms for each held-out farm and write the stored quantiles."""
    out_dir = common.WORK_DIR / "final"
    out_dir.mkdir(parents=True, exist_ok=True)
    data = {farm: fit.Hindcast(farm=farm) for farm in FARMS}
    results: dict[str, dict[int, dict[str, np.ndarray]]] = {"P3_LOFO": {}, "P5_LOFO": {}}
    for held_out in FARMS:
        others = [f for f in FARMS if f != held_out]
        train = pooled_train(data=[data[f] for f in others])
        idle_start = core_module.idle_floor_start(
            speed_100m=train["wind_speed_100m"].to_numpy(), target=train["target"].to_numpy()
        )
        for seed in SEEDS:
            core, era5_residuals = fit.fit_era5_core(
                train=train, adapter="net", seed=seed, with_idle_layer=False
            )
            mixtures: dict[str, dict[str, np.ndarray]] = {"P3_LOFO": {}, "P5_LOFO": {}}
            hindcast_mixtures: dict[str, dict[str, np.ndarray]] = {"P3_LOFO": {}, "P5_LOFO": {}}
            for farm in FARMS:
                for set_name, columns in (
                    ("hindcast", data[farm].hind_columns),
                    ("scored", data[farm].scored_columns),
                ):
                    expected = fit.predict_columns(core=core, columns=columns)
                    mixtures["P3_LOFO"][f"{farm}_{set_name}"] = fit.mixture_of_expected(
                        expected=expected, residuals=era5_residuals
                    )
            # P5_LOFO: fine-tune on the other two farms' hindcast
            train_cols, train_y, val_cols, val_y = [], [], [], []
            for farm in others:
                d = data[farm]
                training, validation = d.split(held_out=None)
                tc, ty = d.subset(cases=training, seed=seed)
                vc, vy = d.subset(cases=validation, seed=seed + 100)
                train_cols.append(tc)
                train_y.append(ty)
                val_cols.append(vc)
                val_y.append(vy)
            merge = lambda parts: {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}  # noqa: E731
            tuned = fit.train_core(
                adapter="net_ens",
                start=core,
                idle_start=idle_start,
                with_idle_layer=False,
                train_columns=merge(train_cols),
                train_target=np.concatenate(train_y),
                validation_columns=merge(val_cols),
                validation_target=np.concatenate(val_y),
                seed=seed,
            )
            expected_cache = {}
            for farm in FARMS:
                for set_name, columns in (
                    ("hindcast", data[farm].hind_columns),
                    ("scored", data[farm].scored_columns),
                ):
                    expected_cache[f"{farm}_{set_name}"] = fit.predict_columns(
                        core=tuned, columns=columns
                    )
            # residual quantiles from the other farms' last-90-day cases
            mean_e, obs = [], []
            for farm in others:
                d = data[farm]
                _, validation = d.split(held_out=None)
                member = expected_cache[f"{farm}_hindcast"].reshape(-1, common.MEMBERS)
                mean_e.append(member.mean(axis=1)[validation])
                obs.append(d.hind_actual[validation].astype(np.float64))
            residuals = core_module.ResidualQuantiles.fit(
                expected=np.concatenate(mean_e), observed=np.concatenate(obs)
            )
            for farm in FARMS:
                for set_name in ("hindcast", "scored"):
                    mixtures["P5_LOFO"][f"{farm}_{set_name}"] = fit.mixture_of_expected(
                        expected=expected_cache[f"{farm}_{set_name}"], residuals=residuals
                    )
            for arm in ("P3_LOFO", "P5_LOFO"):
                # spread factor and offsets from the other farms' hindcast cases
                if arm == "P3_LOFO":
                    rows = {f: np.flatnonzero(data[f].usable) for f in others}
                else:
                    rows = {f: data[f].split(held_out=None)[1] for f in others}
                mix = np.concatenate([mixtures[arm][f"{f}_hindcast"][rows[f]] for f in others])
                actual = np.concatenate([data[f].hind_actual[rows[f]] for f in others]).astype(
                    np.float64
                )
                lead = np.concatenate(
                    [
                        (data[f].valid[rows[f]] - data[f].init[rows[f]])
                        / np.timedelta64(1, "h")
                        for f in others
                    ]
                )
                lead_day = np.concatenate(
                    [
                        np.clip(
                            (
                                data[f].valid[rows[f]].astype("datetime64[D]")
                                - data[f].init[rows[f]].astype("datetime64[D]")
                            ).astype(int),
                            0,
                            ens_final.ens.MAX_LEAD_DAY,
                        )
                        for f in others
                    ]
                )
                factors = fixed_spread(mixture=mix, actual=actual, lead_hours=lead)
                held = data[held_out]
                scored_lead = (
                    common.cases_table(sites=held.sites)["lead_hours"].to_numpy().astype(np.float64)
                )
                scored_day = np.clip(
                    common.cases_table(sites=held.sites)["lead_day"].to_numpy(),
                    0,
                    ens_final.ens.MAX_LEAD_DAY,
                )
                adjusted_scored = apply_fixed_spread(
                    mixture=mixtures[arm][f"{held_out}_scored"].astype(np.float64),
                    lead_hours=scored_lead,
                    factors=factors,
                )
                if arm == "P3_LOFO":
                    adjusted_pool = apply_fixed_spread(mixture=mix.astype(np.float64), lead_hours=lead, factors=factors)
                    table = pooled_offsets(quantiles=adjusted_pool, actual=actual, lead_day=lead_day)
                    adjusted_scored = np.sort(
                        np.clip(adjusted_scored + table[scored_day], 0.0, 1.0), axis=1
                    )
                results[arm].setdefault(seed, {})[held_out] = adjusted_scored.astype(np.float32)
                print(held_out, arm, "seed", seed, "spread factors", factors, flush=True)
    for arm, by_seed in results.items():
        stacked = np.stack(
            [np.concatenate([by_seed[s][farm] for farm in FARMS]) for s in SEEDS]
        )
        np.save(out_dir / f"{arm}.npy", stacked)


if __name__ == "__main__":
    models  # noqa: B018
    main()
