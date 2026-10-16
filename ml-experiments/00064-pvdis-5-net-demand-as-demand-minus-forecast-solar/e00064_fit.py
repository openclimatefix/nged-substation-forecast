"""Fit the refitted arms of experiment 00064 and predict the predict-only arms, one job per series.

Each job writes `work/arms_raw/<arm>_seed<s>_<site>.parquet`, atomically, holding the pooled nine
quantiles of the 51-member mixture for every validation run, one block of rows per variant:

- arms `a`, `am`, `a_twin`, `am_twin` (refits; variant `base`);
- arm `b` (refit on gross demand; variants `base`, `member`, `twin`, which are `b`, `b_member`, and
  `b_twin`, all net after subtracting the solar shape from the gross quantiles);
- arm `d1l_shift` (predict-only on experiment 00046's `d1l` boosters, no refit; variants `sf`,
  `sf_member`, `sf_twin`, `sf_temp`).

A refit also saves its booster to `work/boosters/<arm>_seed<s>_<site>.ubj` and reloads it when the
raw file is missing. Each job uses one thread (`QUANTILE_PARAMETERS`), so four run side by side
(see `e00064_run.sh`). The fit reads only rows up to the last training valid time.

    uv run python ml-experiments/00064-*/e00064_fit.py --arm am --seed 0 --site D9
    uv run python ml-experiments/00064-*/e00064_fit.py --arm d1l_shift --seed 0 --site D9
    uv run python ml-experiments/00064-*/e00064_fit.py --smoke [--device cpu]

`--smoke` runs one series (D9), seed 0, 20 rounds, and two validation runs, writes nothing under
`work/arms_raw/` or `work/boosters/`, and asserts the checks listed on `smoke()`.
"""

import argparse
import time
from pathlib import Path
from typing import Final, Literal

import e00064_common as common
import e00064_model as model
import e00064_weights as weights_module
import numpy as np
import polars as pl
import xgboost as xgb

ArmType = Literal["a", "am", "b", "a_twin", "am_twin", "d1l_shift"]
ARMS: Final[tuple[ArmType, ...]] = ("a", "am", "b", "a_twin", "am_twin", "d1l_shift")
SMOKE_ROUNDS: Final[int] = 20
SMOKE_RUNS: Final[int] = 2
SMOKE_SITE: Final[str] = "D9"
SMOKE_TOLERANCE: Final[float] = 1e-6


def reference_temperature(*, site: str) -> float:
    """Return `T_ref`, the training-window mean cell temperature of a series, from the inputs."""
    table = pl.read_parquet(common.INPUTS_DIR / "t_ref.parquet").filter(pl.col("site") == site)
    return float(table["t_ref"][0])


def run_variants(
    *,
    arm: ArmType,
    run: model.RunData,
    ctx: model.SeriesContext,
    booster: xgb.Booster,
    reference: float,
) -> dict[str, np.ndarray]:
    """Return the pooled quantiles (cases, 9) of each variant of one run.

    Args:
        arm: The arm.
        run: The validation run.
        ctx: The series' context.
        booster: The arm's fitted model (`d1l`'s for `d1l_shift`).
        reference: `T_ref` for the temperature derating (used by `d1l_shift`).

    Returns:
        Each variant's pooled quantiles.
    """
    columns = model.d1l_columns()
    if arm in ("a", "a_twin"):
        frame = model.with_solar_term(run=run, ctx=ctx, twin=arm == "a_twin")
        members = model.predict_members(
            booster=booster,
            members=frame,
            columns=[*columns, model.SOLAR_TERM_COLUMN],
            n_members=run.n_members,
        )
        return {"base": common.mixture_quantiles(member_quantiles=members)}
    if arm in ("am", "am_twin"):
        solar = run.solar
        anomaly = solar.anomaly_twin if arm == "am_twin" else solar.anomaly_mean
        margin = model.margin_rows(weight=solar.weight, anomaly=anomaly, scale=ctx.scale)
        members = model.predict_members(
            booster=booster,
            members=run.members,
            columns=columns,
            margin=margin,
            n_members=run.n_members,
        )
        return {"base": common.mixture_quantiles(member_quantiles=members)}
    if arm == "b":
        members = model.predict_members(
            booster=booster,
            members=model.with_gross_lags(run=run, ctx=ctx),
            columns=columns,
            n_members=run.n_members,
        )
        return {
            name: model.added_pool(member_quantiles=members, member_shift=shift)
            for name, shift in model.b_shifts(run=run, ctx=ctx).items()
        }
    members = model.predict_members(
        booster=booster, members=run.members, columns=columns, n_members=run.n_members
    )
    shifts = model.sf_shifts(run=run, ctx=ctx, reference_temperature=reference)
    return {
        name: model.added_pool(member_quantiles=members, member_shift=shift)
        for name, shift in shifts.items()
    }


def predict_arm(
    *,
    arm: ArmType,
    seed: int,
    ctx: model.SeriesContext,
    booster: xgb.Booster,
    weights: pl.DataFrame | None,
    max_runs: int | None,
) -> pl.DataFrame:
    """Predict every validation run of one series and arm, all variants stacked.

    Args:
        arm: The arm.
        seed: The fitting seed, stored in the `seed` column.
        ctx: The series' context.
        booster: The fitted model.
        weights: `model.read_weights()`.
        max_runs: Only the first this many runs when given.

    Returns:
        Columns `KEYS`, `variant`, `seed`, and the nine quantiles.
    """
    reference = reference_temperature(site=ctx.site) if arm == "d1l_shift" else float("nan")
    pieces = []
    for run in model.validation_runs(ctx=ctx, weights=weights, max_runs=max_runs):
        tables = run_variants(arm=arm, run=run, ctx=ctx, booster=booster, reference=reference)
        pieces.extend(
            model.variant_frame(keys=run.keys, quantiles=quantiles, variant=name, seed=seed)
            for name, quantiles in tables.items()
        )
    return pl.concat(pieces)


def obtain_booster(
    *,
    arm: ArmType,
    seed: int,
    ctx: model.SeriesContext,
    weights: pl.DataFrame | None,
    n_rounds: int,
    device: str,
    booster_dir: Path,
) -> xgb.Booster:
    """Load an arm's booster, or fit and save it.

    Args:
        arm: The arm; `d1l_shift` loads experiment 00046's stored `d1l` booster.
        seed: The fitting seed.
        ctx: The series' context.
        weights: `model.read_weights()`, needed by `am` and `am_twin`.
        n_rounds: Boosting rounds.
        device: `cuda` or `cpu`.
        booster_dir: Where this experiment's boosters live.

    Returns:
        The booster.

    Raises:
        FileNotFoundError: If `d1l_shift` finds no stored booster.
    """
    booster = xgb.Booster(params={"nthread": 1, "device": device})
    if arm == "d1l_shift":
        stored = common.E46_BOOSTER_DIR / f"d1l_seed{seed}_{ctx.site}.ubj"
        if not stored.exists():
            msg = f"no stored d1l booster at {stored}"
            raise FileNotFoundError(msg)
        booster.load_model(stored)
        return booster
    path = booster_dir / f"{arm}_seed{seed}_{ctx.site}.ubj"
    if path.exists():
        booster.load_model(path)
        return booster
    rows = model.training_rows(ctx=ctx)
    matrix = model.training_matrix(
        arm=arm, rows=rows, ctx=ctx, columns=model.d1l_columns(), weights=weights
    )
    booster = model.fit_booster(matrix=matrix, seed=seed, n_rounds=n_rounds, device=device)
    model.save_booster(booster=booster, path=path)
    return booster


def run_job(*, arm: ArmType, seed: int, site: str, n_rounds: int, device: str) -> None:
    """Fit (or reload) one arm's booster for a series and seed, and write its raw file."""
    target = common.RAW_ARMS_DIR / f"{arm}_seed{seed}_{site}.parquet"
    if target.exists():
        return
    started = time.monotonic()
    ctx = model.load_context(site=site)
    weights = model.read_weights() if arm in ("am", "am_twin", "d1l_shift") else None
    booster = obtain_booster(
        arm=arm,
        seed=seed,
        ctx=ctx,
        weights=weights,
        n_rounds=n_rounds,
        device=device,
        booster_dir=common.BOOSTER_DIR,
    )
    table = predict_arm(
        arm=arm, seed=seed, ctx=ctx, booster=booster, weights=weights, max_runs=None
    )
    common.write_atomically(frame=table, target=target)
    print(f"{arm} seed {seed} {site}: {time.monotonic() - started:.0f} s", flush=True)


def _require(*, passed: bool, message: str) -> None:
    """Print a check's outcome and raise when it failed."""
    print(("PASS: " if passed else "FAIL: ") + message, flush=True)
    if not passed:
        raise AssertionError(message)


def stored_d1l(*, ctx: model.SeriesContext, seed: int) -> pl.DataFrame:
    """Return experiment 00046's stored raw `d1l` quantiles (variant `base`) for a series."""
    return pl.read_parquet(common.E46_RAW_ARMS_DIR / f"d1l_seed{seed}_{ctx.site}.parquet").filter(
        pl.col("variant") == "base"
    )


def largest(*, values: np.ndarray | pl.Series) -> float:
    """Return the largest absolute value of an array or series."""
    array = values.to_numpy() if isinstance(values, pl.Series) else values
    return float(np.max(np.abs(array)))


def max_difference_to_stored(*, table: pl.DataFrame, stored: pl.DataFrame) -> tuple[int, float]:
    """Return the number of matched rows and the largest absolute quantile difference."""
    both = table.join(stored, on=list(common.KEYS), suffix="_stored", validate="1:1")
    difference = max(largest(values=both[c] - both[f"{c}_stored"]) for c in common.LEVEL_COLUMNS)
    return both.height, difference


def check_margin(
    *,
    ctx: model.SeriesContext,
    rows: pl.DataFrame,
    weights: pl.DataFrame,
    device: str,
) -> None:
    """Checks 2 and 3: the base margin of `am` (shape, content, climatology, zero-tree model)."""
    columns = model.d1l_columns()
    matrix = model.training_matrix(arm="am", rows=rows, ctx=ctx, columns=columns, weights=weights)
    margin = matrix.get_base_margin()
    _require(
        passed=margin.shape == (rows.height, 9),
        message=f"check 2: base margin shape {margin.shape}",
    )
    weight = model.weights_per_row(
        weights=weights,
        site=ctx.site,
        band=rows["band"].to_numpy(),
        month=rows["month"].to_numpy(),
    )
    expected = weight * ctx.scale * rows[model.ANOMALY_COLUMN].to_numpy()
    _require(
        passed=bool(
            np.allclose(margin[:, 4], expected, atol=1e-6) and largest(values=expected) > 0
        ),
        message="check 2: margin equals w * scale * anomaly (and is not all zero)",
    )
    shifted = model.training_matrix(
        arm="am",
        rows=rows.with_columns(clim_cams=pl.col("clim_cams") + 0.3),
        ctx=ctx,
        columns=columns,
        weights=weights,
    ).get_base_margin()
    _require(
        passed=bool(np.array_equal(shifted, margin)),
        message="check 2: adding 0.3 to the CAMS climatology leaves the margin unchanged",
    )
    zero_tree = model.fit_booster(
        matrix=matrix, seed=0, n_rounds=2, device=device, extra_parameters={"eta": 0.0}
    )
    run = next(model.validation_runs(ctx=ctx, weights=weights, max_runs=1))
    run_margin = model.margin_rows(
        weight=run.solar.weight, anomaly=run.solar.anomaly_mean, scale=ctx.scale
    )
    members = model.predict_members(
        booster=zero_tree,
        members=run.members,
        columns=columns,
        margin=run_margin,
        n_members=run.n_members,
    )
    exact = largest(values=members - run_margin[:, None, :])
    _require(
        passed=exact == 0.0 and largest(values=run_margin) > 0,
        message=f"check 3: zero-tree am model returns the margin (max difference {exact})",
    )


def check_shifts(
    *, ctx: model.SeriesContext, zero: model.SeriesContext, weights: pl.DataFrame, device: str
) -> None:
    """Checks 4 and 5: the predict-only shifts on the stored `d1l` booster."""
    stored = stored_d1l(ctx=ctx, seed=0)
    booster = obtain_booster(
        arm="d1l_shift",
        seed=0,
        ctx=ctx,
        weights=weights,
        n_rounds=SMOKE_ROUNDS,
        device=device,
        booster_dir=common.BOOSTER_DIR,
    )
    variants = ("sf", "sf_member", "sf_twin", "sf_temp")
    no_weights = predict_arm(
        arm="d1l_shift", seed=0, ctx=ctx, booster=booster, weights=None, max_runs=SMOKE_RUNS
    )
    for variant in variants:
        part = no_weights.filter(pl.col("variant") == variant)
        count, difference = max_difference_to_stored(table=part, stored=stored)
        _require(
            passed=count == part.height and difference < SMOKE_TOLERANCE,
            message=f"check 4: {variant} with w = 0 equals stored d1l on {count:,} rows "
            f"(max difference {difference:.2e})",
        )
    real = predict_arm(
        arm="d1l_shift", seed=0, ctx=ctx, booster=booster, weights=weights, max_runs=SMOKE_RUNS
    )
    _, moved = max_difference_to_stored(table=real.filter(pl.col("variant") == "sf"), stored=stored)
    _require(
        passed=moved > 1e-4,
        message=f"check 4: sf with the smoke weights moves the quantiles (max {moved:.4f})",
    )
    zero_table = predict_arm(
        arm="d1l_shift", seed=0, ctx=zero, booster=booster, weights=weights, max_runs=SMOKE_RUNS
    )
    for variant in variants:
        count, difference = max_difference_to_stored(
            table=zero_table.filter(pl.col("variant") == variant), stored=stored
        )
        _require(
            passed=count > 0 and difference < SMOKE_TOLERANCE,
            message=f"check 5: {variant} equals stored d1l at zero capacity on {count:,} rows "
            f"(max difference {difference:.2e})",
        )
    zero_run = next(model.validation_runs(ctx=zero, weights=weights, max_runs=1))
    zero_column = model.with_solar_term(run=zero_run, ctx=zero, twin=False)[model.SOLAR_TERM_COLUMN]
    margin = model.margin_rows(
        weight=zero_run.solar.weight, anomaly=zero_run.solar.anomaly_mean, scale=zero.scale
    )
    shifts = model.b_shifts(run=zero_run, ctx=zero)
    _require(
        passed=largest(values=zero_column) == 0.0
        and all(largest(values=v) == 0.0 for v in shifts.values())
        and largest(values=margin) == 0.0,
        message="check 5: a's column, am's margin, and b's shifts are zero at zero capacity",
    )


def check_refits(
    *, ctx: model.SeriesContext, rows: pl.DataFrame, weights: pl.DataFrame, device: str
) -> None:
    """Check 6: every refitted arm trains, predicts finite quantiles, and reports its row count."""
    columns = model.d1l_columns()
    for arm in ("a", "am", "b", "a_twin", "am_twin"):
        arm_started = time.monotonic()
        matrix = model.training_matrix(
            arm=arm, rows=rows, ctx=ctx, columns=columns, weights=weights
        )
        booster = model.fit_booster(matrix=matrix, seed=0, n_rounds=SMOKE_ROUNDS, device=device)
        table = predict_arm(
            arm=arm, seed=0, ctx=ctx, booster=booster, weights=weights, max_runs=SMOKE_RUNS
        )
        quantiles = table.select(common.LEVEL_COLUMNS).to_numpy()
        _require(
            passed=bool(np.isfinite(quantiles).all()) and table.height > 0,
            message=f"check 6: {arm}: {matrix.num_row():,} training rows, "
            f"{table['variant'].n_unique()} variant(s), {table.height:,} prediction rows, "
            f"{time.monotonic() - arm_started:.0f} s",
        )
    gross = model.gross_training_rows(rows=rows, ctx=ctx)
    print(f"arm b keeps {gross.height:,} of {rows.height:,} training rows")


def smoke(*, device: str) -> None:
    """Run the smoke checks on one series, one seed, few rounds, and two validation runs.

    Checks (each prints PASS or raises):

    1. The training window ends at `LAST_TRAINING_VALID` (the loaders raise otherwise).
    2. `am`'s base margin has shape (rows, 9), is set on every inference matrix, and equals
       `w * scale * anomaly`; shifting the CAMS climatology leaves the margin unchanged.
    3. A zero-tree `am` model (learning rate 0) returns exactly its margin.
    4. `sf` with `w = 0` equals experiment 00046's stored `d1l` within 1e-6.
    5. Every `sf`-family arm equals the stored `d1l` within 1e-6 when the capacity is zero, and
       `a`'s input column, `am`'s margin, and `b`'s shifts are zero then.
    6. Every refitted arm predicts finite quantiles of the right shape.

    Args:
        device: `cuda` or `cpu` for the fits and predictions.
    """
    started = time.monotonic()
    ctx = model.load_context(site=SMOKE_SITE)
    zero = model.load_context(site=SMOKE_SITE, capacity_scale=0.0)
    print(f"smoke: {ctx.site}, capacity {ctx.capacity_mw:.3f} MW, scale {ctx.scale:.4f}")
    rows = model.training_rows(ctx=ctx)
    latest = common.latest_time(times=rows["valid_time"])
    _require(
        passed=latest <= common.LAST_TRAINING_VALID,
        message=f"check 1: {rows.height:,} training rows, latest valid time {latest}",
    )
    sums = weights_module.run_job(
        site=SMOKE_SITE, variant="all", n_rounds=SMOKE_ROUNDS, device=device, ctx=ctx
    )
    weights = weights_module.to_slopes(sums=sums).with_columns(
        variant=pl.lit("all"), site=pl.lit(SMOKE_SITE)
    )
    with pl.Config(tbl_rows=20, tbl_cols=20, tbl_width_chars=200):
        print("smoke weights (20 rounds)")
        print(weights.select("band", "season", "slope_oracle", "slope_applied", "n_rows"))
    check_margin(ctx=ctx, rows=rows, weights=weights, device=device)
    check_shifts(ctx=ctx, zero=zero, weights=weights, device=device)
    check_refits(ctx=ctx, rows=rows, weights=weights, device=device)
    print(f"smoke finished in {time.monotonic() - started:.0f} s")


def main() -> None:
    """Run one job, or the smoke checks."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--site", choices=common.SITES)
    parser.add_argument("--rounds", type=int, default=common.N_ROUNDS)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.smoke:
        smoke(device=args.device)
        return
    if args.arm is None or args.seed is None or args.site is None:
        parser.error("give --arm, --seed, and --site, or --smoke")
    run_job(arm=args.arm, seed=args.seed, site=args.site, n_rounds=args.rounds, device=args.device)


if __name__ == "__main__":
    main()
