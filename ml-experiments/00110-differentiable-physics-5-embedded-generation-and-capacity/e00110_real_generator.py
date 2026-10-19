"""The real generator case, G-c: a Primary with an embedded generator, summed with its neighbour.

The two series that make up G-c, and which of them carries the generator, are listed in a private
file, `work/g_c_members.json`, with the keys `members` (two D-labels) and `generator_member`.
Nothing here or in the published page names either series next to G-c.

The fit is hindsight: it covers the scored months, so it is labelled so and enters no forecast
score. Four months before the 2025-07-01 cut cannot identify heating or a season, so there is no
forecast arm. The node's level is compared with the mean dark-row export of the series that
carries the generator, which is mostly the generator (it has no temperature slope).

    uv run python ml-experiments/00110-*/e00110_real_generator.py
"""

import json
from dataclasses import replace
from datetime import UTC, datetime
from typing import Final

import e00110_arms as arms
import e00110_common as common
import numpy as np
from studies.dp_features import FeatureCache, SiteFeatures, SiteTable
from studies.dp_gam import NodeSpec, TrainingWindow, fit_gam, generator_path

LABEL: Final[str] = "G-c"
FIT_CUT: Final[datetime] = datetime(2026, 6, 30, 23, 30, tzinfo=UTC)
"""The last half-hour end before the held-out period."""
LEVEL_TOLERANCE: Final[float] = 0.25
MEMBERS_PATH: Final = common.WORK_DIR / "g_c_members.json"


def combined_features(*, cache: FeatureCache, members: tuple[str, str]) -> SiteFeatures:
    """Return the features of the sum of two series, in points of their summed capacity.

    Args:
        cache: The cache.
        members: The two series' D-labels.

    Returns:
        The summed series with the first member's weather and the neighbours' masks combined.
    """
    first, second = (cache.sites[label] for label in members)
    capacity = first.table.effective_capacity_mw + second.table.effective_capacity_mw
    power_mw = (
        first.y * first.table.effective_capacity_mw + second.y * second.table.effective_capacity_mw
    ) / 100.0
    table = SiteTable(
        site=LABEL,
        time_series_id=-1,
        units="MW",
        role="generator_case",
        latitude=first.table.latitude,
        longitude=first.table.longitude,
        effective_capacity_mw=capacity,
        first_usable=max(first.table.first_usable, second.table.first_usable),
    )
    return replace(
        first,
        table=table,
        y=100.0 * power_mw / capacity,
        present=first.present & second.present,
        in_pool=first.in_pool & second.in_pool,
        default_rows=first.default_rows & second.default_rows,
        switching_mask=first.switching_mask | second.switching_mask,
    )


def main() -> None:
    """Fit the node to G-c, compare its level with the dark-row export, and write the result."""
    members = json.loads(MEMBERS_PATH.read_text())
    cache = common.load_cache()
    pair = tuple(members["members"])
    combined = combined_features(cache=cache, members=(pair[0], pair[1]))
    sites = {**cache.sites, LABEL: combined}
    cache = replace(cache, sites=sites)
    tuned = arms.read_tuned()
    part_4 = arms.read_part_4_tuned()
    spec = replace(
        arms.base_specs(part_4=part_4)["b_d1"],
        capacity="fitted",
        node=NodeSpec(**tuned["node"]),
    )
    fit = fit_gam(
        cache=cache,
        site=LABEL,
        specs=[spec],
        cut=FIT_CUT,
        window=TrainingWindow(start=combined.table.first_usable),
        season_years=common.SEASON_YEARS,
    )
    path = generator_path(fit=fit)
    carrier = cache.sites[members["generator_member"]]
    window = (
        carrier.present
        & carrier.dark
        & (cache.end_times > common.to_64(moment=combined.table.first_usable))
        & (cache.end_times <= common.to_64(moment=FIT_CUT))
        & np.isfinite(carrier.y)
    )
    carrier_mw = carrier.y[window] * carrier.table.effective_capacity_mw / 100.0
    reference = float(-carrier_mw.mean() * 100.0 / combined.table.effective_capacity_mw)
    level = float(np.median(path))
    result = {
        "node_level_median": level,
        "node_level_max": float(path.max()),
        "node_level_min": float(path.min()),
        "dark_row_export_points": reference,
        "ratio": level / reference if reference > 0 else None,
        "within_tolerance": bool(reference > 0 and abs(level / reference - 1.0) <= LEVEL_TOLERANCE),
        "step_days": tuned["node"]["step_days"],
        "n_blocks": len(path),
        "path_normalised": (path / combined.table.effective_capacity_mw).tolist(),
    }
    (common.WORK_DIR / "g_c.json").write_text(json.dumps(result, indent=2) + "\n")
    print({k: v for k, v in result.items() if k != "path_normalised"})


if __name__ == "__main__":
    main()
