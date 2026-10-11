"""Where the capacity-labeller study keeps its files, and the loaders every script shares.

Part of the study in <https://github.com/openclimatefix/nged-substation-forecast/issues/1147>. The
study reads the first capacity study's built record and events (issue #1143) and never writes them.

**A smoke run writes to its own directory and can never be mistaken for a real run.** `claim_root`
writes a `RUN_KIND.txt` into the output root on first use, naming it `smoke` or `real`, and raises
if a later run of the other kind points at the same root. A smoke root is `<study
directory>/_smoke`, and a real run refuses any root whose path holds `_smoke`. Every directory of
results carries a stamp of the inputs and the code (`check_stamp`), so no stale result is reused.

**The prompt and answer log pairs plant letters with dates, so it lives under `data/` only.**
`log_path` returns a path under the study's folder in `data/`, and nothing writes it elsewhere.
"""

import hashlib
import json
from pathlib import Path
from typing import Any, Final, Literal

import numpy as np
from studies.capacity_estimators import CapacityRecord
from studies.capacity_evidence import CapOverlay, PlantFit
from studies.capacity_recovery import PlantedEvent, plant_losses
from studies.sources import study_dir_for

import studies as studies_package

STUDY_DIR: Final[Path] = study_dir_for(study="llm_capacity_labeller")
"""The study's folder under `data/studies/per_study/`."""

SMOKE_DIR_NAME: Final[str] = "_smoke"
"""The directory under the study's folder that holds smoke runs."""

FIRST_STUDY_BUILT: Final[Path] = study_dir_for(study="effective_capacity") / "built"
"""The first capacity study's built inputs, which this study reads and never writes."""

FRESH_DRAW_SEEDS: Final[tuple[int, int]] = (11441, 11442)
"""The seeds of the two fresh draws of planted losses, never run before this study."""

NULL_SEED: Final[int] = 114400
"""The seed of the synthetic null record's resampled days."""

CONTROL_SIZE: Final[float] = 0.30
"""The positive control's step. Copied from `CONTROL_SIZE` in the first study's
`capacity_paths.py`, which this folder may not import; `positive_control_events` checks that the
first study's events agree."""

TUNING_SEED: Final[int] = 114401
"""The seed of the cross-checks that resample events."""

RecordIdType = Literal["original", "draw01", "draw02", "control", "null"]
"""The records a labeller reads: the original, the two fresh draws, the positive control, and the
synthetic null record."""

RECORD_IDS: Final[tuple[RecordIdType, ...]] = ("original", "draw01", "draw02", "control", "null")
"""The runtime tuple of `RecordIdType`'s values."""


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


PACKAGE_MODULES: Final[tuple[str, ...]] = (
    "capacity_estimators",
    "capacity_convex",
    "capacity_posthoc",
    "capacity_recovery",
    "capacity_seasonal",
    "capacity_evidence",
    "capacity_labeller",
    "capacity_sanity",
    "capacity_llm",
    "labeller_scoring",
)
"""The package modules whose code decides what an estimate, a case file, or a score is."""


def fingerprint(*, extra_files: tuple[Path, ...] = ()) -> str:
    """Hash the first study's inputs and the code that turns them into results.

    Args:
        extra_files: Further files to hash, such as a script's own source.

    Returns:
        A hexadecimal digest that changes if an input file or a source file changes.
    """
    package = Path(studies_package.__file__).parent
    files = [
        *(package / f"{name}.py" for name in PACKAGE_MODULES),
        Path(__file__),
        FIRST_STUDY_BUILT / "record.npz",
        FIRST_STUDY_BUILT / "events.json",
        *extra_files,
    ]
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.name.encode() + sha256_of(path=path).encode())
    return digest.hexdigest()


def check_stamp(*, directory: Path, stamp: str) -> None:
    """Create a directory's stamp file, or check that an existing one matches.

    Args:
        directory: A directory of results.
        stamp: `fingerprint`'s result.

    Raises:
        RuntimeError: If the directory holds results made from other inputs or other code. The
            caller moves the directory to `superseded/`.
    """
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "STAMP.txt"
    if path.exists():
        if path.read_text().strip() != stamp:
            msg = f"{directory} was made from other inputs or code; move it to superseded/"
            raise RuntimeError(msg)
    elif any(directory.iterdir()):
        msg = f"{directory} holds results and no stamp; move it to superseded/"
        raise RuntimeError(msg)
    else:
        path.write_text(stamp + "\n")


def built_dir(*, root: Path) -> Path:
    """Return the directory of this study's built inputs."""
    return root / "built"


def tables_dir(*, root: Path) -> Path:
    """Return the directory of this study's result tables."""
    return root / "tables"


def calls_dir(*, root: Path) -> Path:
    """Return the directory of the per-call cache, under `data/`."""
    return root / "calls"


def log_path(*, root: Path) -> Path:
    """Return the prompt and answer log, under `data/`."""
    return root / "labeller_log.jsonl"


def write_json_atomic(*, path: Path, payload: object) -> None:
    """Write JSON through a temporary file.

    Args:
        path: The destination.
        payload: A JSON-serialisable value.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.rename(path)


def event_to_json(*, event: PlantedEvent) -> dict[str, Any]:
    """Return an event as a JSON-ready mapping."""
    return {
        "plant": event.plant,
        "shape": event.shape,
        "size": event.size,
        "first_day": event.first_day,
        "month_days": event.month_days,
        "hit_days": list(event.hit_days),
    }


def event_from_json(*, payload: dict[str, Any]) -> PlantedEvent:
    """Rebuild an event from `event_to_json`'s result."""
    return PlantedEvent(
        plant=int(payload["plant"]),
        shape=payload["shape"],
        size=float(payload["size"]),
        first_day=int(payload["first_day"]),
        month_days=int(payload["month_days"]),
        hit_days=tuple(int(day) for day in payload["hit_days"]),
    )


def load_first_record(*, anm_blind: bool = False) -> CapacityRecord:
    """Read the first study's record.

    Args:
        anm_blind: Whether to read the power with the export-capped hours left in.

    Returns:
        The record.
    """
    with np.load(FIRST_STUDY_BUILT / "record.npz") as data:
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


class FirstStudyEvents:
    """What the first study's `events.json` fixes: its draws, control, windows, and plants.

    The first draw (`tuning_draw`, seed 11430) tunes this study's Part 1 settings; draws 1 to 10 are
    the first study's selection draws.
    """

    def __init__(self, *, payload: dict[str, Any]) -> None:
        """Parse the events file.

        Args:
            payload: The JSON the first study's build script wrote.
        """
        self.control: list[PlantedEvent] = [event_from_json(payload=e) for e in payload["control"]]
        self.reserved: list[PlantedEvent] = [
            event_from_json(payload=e) for e in payload["reserved"]
        ]
        self.false_movement_windows: list[tuple[int, int, int]] = [
            (int(a), int(b), int(c)) for a, b, c in payload["false_movement_windows"]
        ]
        self.draws: list[list[PlantedEvent]] = [
            [event_from_json(payload=e) for e in draw] for draw in payload["draws"]
        ]
        self.tuning_draw: list[PlantedEvent] = self.draws[0]
        self.scored_months: list[str] = list(payload["scored_months"])
        self.low_plant: int = int(payload["low_plant"])
        self.low_months: list[str] = list(payload["low_months"])
        self.anm_plant: int = int(payload["anm_plant"])


def load_first_events() -> FirstStudyEvents:
    """Read the first study's events file."""
    return FirstStudyEvents(payload=json.loads((FIRST_STUDY_BUILT / "events.json").read_text()))


def positive_control_events(*, first: FirstStudyEvents) -> list[PlantedEvent]:
    """Return the first study's positive control, checked to be steps of `CONTROL_SIZE`.

    Args:
        first: The first study's events.

    Returns:
        The control's events.

    Raises:
        ValueError: If an event is not a step of `CONTROL_SIZE`.
    """
    for event in first.control:
        if event.shape != "step" or abs(event.size - CONTROL_SIZE) > 1e-9:
            msg = "the first study's control is not a step of the size copied here"
            raise ValueError(msg)
    return first.control


def load_plant_fits(*, record: CapacityRecord) -> list[PlantFit]:
    """Read each plant's fitted quantities from the first study's `orientation.json`.

    Args:
        record: The record, for the plant order.

    Returns:
        One fit per plant.
    """
    payload = json.loads((FIRST_STUDY_BUILT / "orientation.json").read_text())
    return [
        PlantFit(
            ac_capacity_mw=float(payload[site]["ac_capacity_mw"]),
            dc_capacity_mw=float(payload[site]["dc_capacity_mw"]),
            tilt_deg=float(payload[site]["tilt_deg"]),
            azimuth_deg=float(payload[site]["azimuth_deg"]),
        )
        for site in record.sites
    ]


def save_cap_overlay(*, root: Path, plant: int, overlay: CapOverlay) -> None:
    """Write the export-cap overlay of the plant that has one.

    Args:
        root: The output root.
        plant: The plant's index.
        overlay: The overlay on the record's hourly grid.
    """
    target = built_dir(root=root)
    target.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        target / "cap_overlay.npz", plant=plant, cap_mw=overlay.cap_mw, binding=overlay.binding
    )


def load_cap_overlays(*, root: Path) -> dict[int, CapOverlay]:
    """Read the export-cap overlays by plant index.

    Args:
        root: The output root.

    Returns:
        A mapping holding the one plant that has an overlay.
    """
    with np.load(built_dir(root=root) / "cap_overlay.npz") as data:
        return {
            int(data["plant"]): CapOverlay(
                cap_mw=data["cap_mw"], binding=data["binding"].astype(bool)
            )
        }


def load_fresh_draws(*, root: Path) -> list[list[PlantedEvent]]:
    """Read the two fresh draws of planted losses.

    Args:
        root: The output root.

    Returns:
        One list of events per fresh draw, in `FRESH_DRAW_SEEDS` order.
    """
    payload = json.loads((built_dir(root=root) / "fresh_draws.json").read_text())
    return [[event_from_json(payload=e) for e in draw] for draw in payload["draws"]]


def record_for(*, root: Path, record_id: RecordIdType) -> CapacityRecord:
    """Build a record by identifier.

    Args:
        root: The output root.
        record_id: `original`, `draw01`, `draw02`, `control`, or `null`.

    Returns:
        The record with its losses planted, or its power replaced for `null`.

    Raises:
        ValueError: If the identifier is unknown.
    """
    original = load_first_record()
    if record_id == "original":
        return original
    if record_id in ("draw01", "draw02"):
        draws = load_fresh_draws(root=root)
        return plant_losses(record=original, events=draws[int(record_id[-1]) - 1])
    if record_id == "control":
        return plant_losses(
            record=original, events=positive_control_events(first=load_first_events())
        )
    if record_id == "null":
        with np.load(built_dir(root=root) / "null_power.npz") as data:
            return original.with_power(power_mw=data["power_mw"].astype(np.float64))
    msg = f"unknown record identifier {record_id}"
    raise ValueError(msg)


def events_of(*, root: Path, record_id: RecordIdType) -> list[PlantedEvent]:
    """Return the planted events of a record.

    Args:
        root: The output root.
        record_id: The record's identifier.

    Returns:
        The events planted in it; empty for `original` and `null`.
    """
    if record_id in ("draw01", "draw02"):
        return load_fresh_draws(root=root)[int(record_id[-1]) - 1]
    if record_id == "control":
        return positive_control_events(first=load_first_events())
    return []
