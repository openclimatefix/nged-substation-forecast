"""Post hoc arm: the median plus the as-of empirical residual quantiles (no cells, no mixture).

uv run --no-sync python ml-experiments/00264-*/e00264_post_hoc.py
"""

import e00264_asof as asof
import e00264_common as common


def main() -> None:
    """Write the `arm_emp` forecast for every site."""
    for site in common.SITE_IDS:
        reference = common.load_reference(site=site)
        frame = asof.empirical_residual_quantiles(reference=reference)
        common.write_parquet_atomic(
            path=common.PREDS_DIR / "arm_emp" / f"{site}.parquet", frame=frame
        )
        print(f"{site}: {frame.height:,} rows written")


if __name__ == "__main__":
    main()
