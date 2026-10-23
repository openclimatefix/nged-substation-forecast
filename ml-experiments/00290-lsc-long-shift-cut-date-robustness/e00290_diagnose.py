"""Post hoc: reproduce the cut-position jump of experiment 274 for D3 and nearby end dates.

OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= uv run --no-sync python e00290_diagnose.py
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import e00290_fit as fit
import numpy as np
from studies.dp_ens_weather import replace_site
from studies.dp_features import default_input_paths, load_or_build_feature_cache, naive_utc

ENDS = (
    datetime(2023, 7, 1, tzinfo=UTC),
    datetime(2023, 7, 2, tzinfo=UTC),
    datetime(2023, 6, 20, tzinfo=UTC),
)
OFFSET_HOURS = (-24, 0, 6, 12, 24)


def main() -> None:
    """Print the mean forecast difference over one fixed week, by end date and cut offset."""
    cache = load_or_build_feature_cache(paths=default_input_paths(), cache_dir=fit.CACHE_DIR)
    features = cache.sites["D3"]
    stamps = cache.end_times
    for end in ENDS:
        onset = end - fit.DURATION
        active = (stamps > naive_utc(moment=onset)) & (stamps <= naive_utc(moment=end))
        injected = replace_site(
            cache=cache, site="D3", y=features.y + np.where(active, fit.SIZE * features.y, 0.0)
        )
        first = end + timedelta(days=10)
        index = fit.target_rows(cache=cache, site="D3", cut=first, stop=first + timedelta(days=7))
        for hours in OFFSET_HOURS:
            cut = end + timedelta(hours=hours)
            moved = fit.forecast(cache=injected, site="D3", cut=cut, index=index)
            clean = fit.forecast(cache=cache, site="D3", cut=cut, index=index)
            print(
                f"D3 transfer ending {end:%Y-%m-%d}, cut {hours:+d} h: {np.mean(moved - clean):.2f}"
            )


if __name__ == "__main__":
    main()
