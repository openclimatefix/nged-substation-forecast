"""Where the effective-capacity study keeps its files, and the loaders every script shares.

Part of the study in <https://github.com/openclimatefix/nged-substation-forecast/issues/1143>.

**A smoke run writes to its own directory and can never be mistaken for a real run.** `claim_root`
writes a `RUN_KIND.txt` into the output root on first use, naming it `smoke` or `real`, and raises if
a later run of the other kind points at the same root. A smoke root is `<study directory>/_smoke`,
and a real run refuses any root whose path holds `_smoke`.
"""

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Final

import numpy as np
from studies.capacity_estimators import CapacityRecord
from studies.capacity_recovery import PlantedEvent
from studies.sources import study_dir_for

STUDY_DIR: Final[Path] = study_dir_for(study="effective_capacity")
"""The study's folder under `data/studies/per_study/`."""

SMOKE_DIR_NAME: Final[str] = "_smoke"
"""The directory under the study's folder that holds smoke runs."""

LAG_DIR: Final[Path] = study_dir_for(study="lag_features") / "ens_mean"
"""The lag-features study's outputs, which this study reads and never writes."""

LAG_FRAME_PATH: Final[Path] = LAG_DIR / "lag_frame_ens_mean_day1.parquet"
"""The 29,445 lead-day 1 rows the forecast contrast scores."""

LAG_FRAME_ROWS: Final[int] = 29445
"""The row count the lag study published."""

LAG_LOSSES_PATH: Final[Path] = LAG_DIR / "losses_ens_mean.parquet"
"""The lag study's per-row losses, from which B0 is reused."""

LAG_FOLLOWUPS2_LOSSES_PATH: Final[Path] = (
    LAG_DIR / "followups2" / "losses_followups2_ens_mean.parquet"
)
"""The lag study's round-2 losses, which hold B0 without day of year."""

OUTPUT_RATIOS_PATH: Final[Path] = LAG_DIR / "followups3" / "output_ratios_ens_mean.parquet"
"""Each plant-month's output over its neighbours'."""

DERATED_PLAN_PATH: Final[Path] = LAG_DIR / "followups3" / "derated_plan.json"
"""Which plant and months the lag study found producing less than its neighbours."""

DRAW_SEEDS: Final[tuple[int, ...]] = tuple(11430 + index for index in range(11))
"""One seed per draw; draw 0 is the tuning draw and draws 1 to 10 are the selection draws."""

CONTROL_SEED: Final[int] = 114300
"""The seed of the positive control's months."""

FALSE_MOVEMENT_SEED: Final[int] = 114301
"""The seed that chooses the quiet plant-months."""

FALSE_MOVEMENTS_PER_PLANT: Final[int] = 4
"""The most quiet plant-months chosen at each plant."""

CONTROL_SIZE: Final[float] = 0.30
"""The positive control's step."""

CONTROL_MONTHS_PER_PLANT: Final[int] = 3
"""How many scored months the positive control plants at each plant."""

DERANGEMENT_SEED: Final[int] = 114302
"""The seed of the negative control's derangement."""

ORIGINAL: Final[str] = "original"
CONTROL: Final[str] = "control"
"""Record identifiers besides `draw00` to `draw10`."""


def draw_id(*, index: int) -> str:
    """Return the record identifier of one draw.

    Args:
        index: 0 for the tuning draw, 1 to 10 for the selection draws.

    Returns:
        `draw00`, `draw01`, and so on.
    """
    return f"draw{index:02d}"


def record_ids() -> list[str]:
    """Return every record identifier, original first."""
    return [ORIGINAL, *(draw_id(index=i) for i in range(len(DRAW_SEEDS))), CONTROL]


def claim_root(*, smoke: bool) -> Path:
    """Return the output root for a run and mark it as real or smoke.

    Args:
        smoke: Whether this is a smoke run.

    Returns:
        The root directory, created if need be.

    Raises:
        RuntimeError: If the root was claimed by the other kind of run, or a real run is pointed at
            a smoke path.
    """
    root = STUDY_DIR / SMOKE_DIR_NAME if smoke else STUDY_DIR
    if not smoke and SMOKE_DIR_NAME in root.parts:
        msg = f"a real run may not write under {SMOKE_DIR_NAME}"
        raise RuntimeError(msg)
    root.mkdir(parents=True, exist_ok=True)
    marker = root / "RUN_KIND.txt"
    kind = "smoke" if smoke else "real"
    if marker.exists():
        if marker.read_text().strip() != kind:
            msg = f"{root} was claimed by a {marker.read_text().strip()} run, not a {kind} run"
            raise RuntimeError(msg)
    else:
        marker.write_text(kind + "\n")
    return root


def sha256_of(*, path: Path) -> str:
    """Return a file's SHA-256 digest.

    Args:
        path: The file.

    Returns:
        The hexadecimal digest.
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inputs_dir(*, root: Path) -> Path:
    """Return the directory of the built inputs."""
    return root / "inputs"


def estimates_dir(*, root: Path) -> Path:
    """Return the directory of the estimators' daily estimates."""
    return root / "estimates"


def save_record(*, path: Path, record: CapacityRecord, power_anm_blind: np.ndarray) -> None:
    """Write a record and the power with the export-capped hours left in.

    Args:
        path: The `.npz` file.
        record: The record.
        power_anm_blind: The power before the export-capped hours were removed.
    """
    temporary = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(
        temporary,
        sites=np.array(record.sites),
        dates=record.dates.astype("datetime64[D]").astype(np.int64),
        power_mw=record.power_mw,
        power_anm_blind=power_anm_blind,
        clear_sky_w_m2=record.clear_sky_w_m2,
        cams_ghi_w_m2=record.cams_ghi_w_m2,
        cams_shape=record.cams_shape,
        clear_shape=record.clear_shape,
        static_p99_mw=record.static_p99_mw,
    )
    temporary.rename(path)


def load_record(*, root: Path, anm_blind: bool = False) -> CapacityRecord:
    """Read the record the build script wrote.

    Args:
        root: The output root.
        anm_blind: Whether to read the power with the export-capped hours left in.

    Returns:
        The record.
    """
    with np.load(inputs_dir(root=root) / "record.npz") as data:
        return CapacityRecord(
            sites=tuple(str(site) for site in data["sites"]),
            dates=data["dates"].astype("datetime64[D]"),
            power_mw=data["power_anm_blind" if anm_blind else "power_mw"],
            clear_sky_w_m2=data["clear_sky_w_m2"],
            cams_ghi_w_m2=data["cams_ghi_w_m2"],
            cams_shape=data["cams_shape"],
            clear_shape=data["clear_shape"],
            static_p99_mw=data["static_p99_mw"],
        )


def event_to_json(*, event: PlantedEvent) -> dict[str, object]:
    """Return an event as a JSON-ready mapping."""
    return {**asdict(event), "hit_days": list(event.hit_days)}


def event_from_json(*, payload: dict[str, object]) -> PlantedEvent:
    """Rebuild an event from `event_to_json`'s result."""
    return PlantedEvent(
        plant=int(payload["plant"]),  # type: ignore[call-overload]
        shape=payload["shape"],  # type: ignore[arg-type]
        size=float(payload["size"]),  # type: ignore[arg-type]
        first_day=int(payload["first_day"]),  # type: ignore[call-overload]
        month_days=int(payload["month_days"]),  # type: ignore[call-overload]
        hit_days=tuple(int(day) for day in payload["hit_days"]),  # type: ignore[attr-defined]
    )


def write_json_atomic(*, path: Path, payload: object) -> None:
    """Write JSON through a temporary file.

    Args:
        path: The destination.
        payload: A JSON-serialisable value.
    """
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.rename(path)


class Events:
    """Everything `build_capacity_inputs.py` decides about the events and the windows."""

    def __init__(self, *, payload: dict[str, object]) -> None:
        """Parse the events file.

        Args:
            payload: The JSON the build script wrote.
        """
        self.draws: list[list[PlantedEvent]] = [
            [event_from_json(payload=e) for e in draw]  # type: ignore[attr-defined]
            for draw in payload["draws"]  # type: ignore[attr-defined]
        ]
        self.control: list[PlantedEvent] = [
            event_from_json(payload=e)
            for e in payload["control"]  # type: ignore[attr-defined]
        ]
        self.reserved: list[PlantedEvent] = [
            event_from_json(payload=e)
            for e in payload["reserved"]  # type: ignore[attr-defined]
        ]
        self.false_movement_windows: list[tuple[int, int, int]] = [
            (int(a), int(b), int(c))
            for a, b, c in payload["false_movement_windows"]  # type: ignore[attr-defined]
        ]
        self.scored_months: list[str] = list(payload["scored_months"])  # type: ignore[call-overload]
        self.low_plant: int = int(payload["low_plant"])  # type: ignore[call-overload]
        self.low_months: list[str] = list(payload["low_months"])  # type: ignore[call-overload]
        self.anm_plant: int = int(payload["anm_plant"])  # type: ignore[call-overload]


def load_events(*, root: Path) -> Events:
    """Read the events file.

    Args:
        root: The output root.

    Returns:
        The events and windows.
    """
    path = inputs_dir(root=root) / "events.json"
    return Events(payload=json.loads(path.read_text()))
