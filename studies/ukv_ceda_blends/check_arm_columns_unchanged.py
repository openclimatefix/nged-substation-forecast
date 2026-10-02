"""Check that every saved fit's arm columns equal what `fit_aifs.arm_features` returns now.

One-off throwaway script for
<https://github.com/openclimatefix/nged-substation-forecast/issues/1016>. The UKV-CEDA study adds
one branch to `nwp_forecast_comparison._wind_weather_fields`, which every arm's wind columns pass
through. This script proves the branch leaves every existing arm alone: each `*_losses.json` stamp
under `data/studies/nwp_forecast_comparison_*` records the `columns` its fit used, and each arm's
recorded columns must equal `arm_features` for that arm and technology today. It reads the stamps
and fits nothing, writes nothing, and exits non-zero on any difference or on an arm whose columns
`arm_features` cannot resolve.

(`fit_product_blends.py --dry-run` cannot show this, because its `SAME_BUILD_KEYS` leave `columns`
out of the comparison.)

Run it with `uv run python studies/ukv_ceda_blends/check_arm_columns_unchanged.py`.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Final

_STUDY_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_STUDY_DIR.parent / "nwp_forecast_comparison"))
sys.path.insert(0, str(_STUDY_DIR.parent / "beam_diffuse_split"))

import fit_aifs  # noqa: E402
from nwp_forecast_comparison import DomainType, _repo_data_dir  # noqa: E402

STAMP_GLOB: Final[str] = "nwp_forecast_comparison_*/*_losses.json"
"""Under `data/studies/`, every earlier study's stamps."""


def stamp_domain(*, path: Path) -> DomainType:
    """Return the technology a stamp belongs to, read from its file name's first word.

    Args:
        path: A stamp such as `.../wind_single_day1_losses.json`.

    Returns:
        `solar` or `wind`.

    Raises:
        ValueError: If the file name starts with neither.
    """
    first = path.name.split("_", maxsplit=1)[0]
    if first == "solar":
        return "solar"
    if first == "wind":
        return "wind"
    msg = f"{path} does not start with solar or wind"
    raise ValueError(msg)


def check_stamp(*, path: Path) -> tuple[int, list[str]]:
    """Compare one stamp's recorded columns with `arm_features` for every arm it lists.

    Args:
        path: A `*_losses.json` stamp holding a `columns` entry, a JSON object from arm to columns.

    Returns:
        How many arms were compared, and one line for each arm whose columns differ now.
    """
    domain = stamp_domain(path=path)
    recorded: dict[str, list[str]] = json.loads(json.loads(path.read_text())["columns"])
    problems = [
        f"{path.parent.name}/{path.name}: {arm}"
        for arm, columns in recorded.items()
        if list(fit_aifs.arm_features(arm=arm, domain=domain)) != columns
    ]
    return len(recorded), problems


def check_all(*, studies_dir: Path) -> tuple[int, int, list[str]]:
    """Check every earlier study's stamp.

    Args:
        studies_dir: The `data/studies` folder.

    Returns:
        The stamps read, the arms compared, and the lines for every arm that differs.
    """
    paths = sorted(studies_dir.glob(STAMP_GLOB))
    arms = 0
    problems: list[str] = []
    for path in paths:
        compared, found = check_stamp(path=path)
        arms += compared
        problems += found
    return len(paths), arms, problems


def main() -> int:
    """Print the result and return 0 if every recorded arm still resolves to its columns."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--studies-dir", type=Path, default=_repo_data_dir() / "studies")
    args = parser.parse_args()
    stamps, arms, problems = check_all(studies_dir=args.studies_dir)
    sys.stdout.write(f"{stamps} stamps, {arms} arms compared\n")
    for line in problems:
        sys.stdout.write(f"DIFFERS: {line}\n")
    if stamps == 0:
        sys.stdout.write("CHECK FAIL: no stamp found\n")
        return 1
    sys.stdout.write(f"CHECK {'FAIL' if problems else 'PASS'}\n")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
