# ruff: noqa: E501
"""Build every arm's forecast for the three sites and store it as parquet.

Arm 1 is experiment 191's reference. Arms 2 and 2b recalibrate the lower (and upper) quantiles
as-of. Arm 3 is the mixture, falling back to arm 2b where the visibility switch is off. Arm 4
adds hindsight price flags to the mixture (D8 only). The stress runs force the switch on.

    systemd-run --user --scope -p MemoryMax=16G -p OOMPolicy=continue \
        uv run --no-sync python ml-experiments/00264-*/e00264_run.py
"""

import os
from concurrent.futures import ProcessPoolExecutor

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_name] = "1"

import e00264_asof as asof  # noqa: E402
import e00264_common as common  # noqa: E402
import e00264_mixture as mixture  # noqa: E402
import polars as pl  # noqa: E402


def run_site(site: str) -> str:
    """Write every arm for one site; return a one-line summary."""
    reference = common.load_reference(site=site)
    arms: dict[str, pl.DataFrame] = {"arm1": reference}
    arms["arm2"] = asof.recalibrate(reference=reference, sides=("lower",))
    arms["arm2b"] = asof.recalibrate(reference=reference, sides=("lower", "upper"))
    arms["arm3"], log3 = mixture.mixture_arm(reference=reference, fallback=arms["arm2b"], use_price=False, force_visible=False)
    arms["arm3_forced"], log3f = mixture.mixture_arm(reference=reference, fallback=arms["arm2b"], use_price=False, force_visible=True)
    if site == common.ENGINE_SITE:
        arms["arm4"], log4 = mixture.mixture_arm(reference=reference, fallback=arms["arm2b"], use_price=True, force_visible=False)
        common.write_parquet_atomic(path=common.WORK_DIR / "logs" / f"arm4_{site}.parquet", frame=log4)
    for name, frame in arms.items():
        common.write_parquet_atomic(path=common.PREDS_DIR / name / f"{site}.parquet", frame=frame)
    common.write_parquet_atomic(path=common.WORK_DIR / "logs" / f"arm3_{site}.parquet", frame=log3)
    common.write_parquet_atomic(path=common.WORK_DIR / "logs" / f"arm3_forced_{site}.parquet", frame=log3f)
    return f"{site}: {reference.height:,} rows, arms {sorted(arms)}"


def main() -> None:
    """Run the three sites in parallel workers."""
    with ProcessPoolExecutor(max_workers=3) as pool:
        for line in pool.map(run_site, list(common.SITE_IDS)):
            print(line, flush=True)


if __name__ == "__main__":
    main()
