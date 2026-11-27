"""Check the step-5 bundle inside the production image against its own manifest, then load it.

`build_and_verify_image.sh` pipes this file to `python -I -` inside the built image, with no
network, so the check reads the bundle the image will serve rather than the copy on the host. The
image leaves out the bundle's `fixtures/` folder and its `.pt` weights, which only research's
`reference_check.py` reads, so those entries of the manifest are expected to be absent.

Exits 1 with one line per fault if a file's SHA-256 differs from `manifest.json`, a file the
image needs is missing, the image holds a file the manifest does not list, or an installed
library's version differs from the version research exported the bundle with. Exits non-zero, with
a traceback, if `load_bundle` or `require_servable_bundle` rejects the bundle.
"""

import hashlib
import json
import sys
from importlib import metadata
from pathlib import Path
from typing import Final

from substation_demand_forecaster.bundle import load_bundle, require_servable_bundle

NOT_SHIPPED_PREFIX: Final[str] = "fixtures/"
NOT_SHIPPED_SUFFIX: Final[str] = ".pt"

PINNED_LIBRARIES: Final[tuple[str, ...]] = ("holidays", "pvlib")
"""Libraries whose version changes the input columns, so the image must hold research's version.

`holidays` decides the bank-holiday columns, and `pvlib` the sun columns."""


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def bundle_faults(bundle_dir: Path) -> list[str]:
    """Return one line per file of `bundle_dir` that disagrees with its `manifest.json`."""
    expected: dict[str, str] = json.loads((bundle_dir / "manifest.json").read_text())["sha256"]
    shipped = {
        name: digest
        for name, digest in expected.items()
        if not name.startswith(NOT_SHIPPED_PREFIX) and not name.endswith(NOT_SHIPPED_SUFFIX)
    }
    present = {
        str(path.relative_to(bundle_dir))
        for path in bundle_dir.rglob("*")
        if path.is_file() and path.name != "manifest.json"
    }
    faults = [f"missing: {name}" for name in sorted(shipped.keys() - present)]
    faults += [f"not in the manifest: {name}" for name in sorted(present - expected.keys())]
    faults += [
        f"SHA-256 differs: {name}"
        for name in sorted(shipped.keys() & present)
        if _sha256(bundle_dir / name) != shipped[name]
    ]
    return faults


def _base_version(version: str) -> str:
    """Return a version without its local label: `2.14.1+cpu` gives `2.14.1`."""
    return version.split("+", maxsplit=1)[0]


def library_faults(bundle_dir: Path) -> list[str]:
    """Return one line per installed library whose version differs from the bundle's manifest.

    `holidays` and `pvlib` must match exactly, and the manifest must record both. `torch` is
    compared without its local label, because research's CUDA wheel and the image's CPU wheel of
    one release differ only in that label, and only when the manifest records a `torch` version.
    """
    manifest = json.loads((bundle_dir / "manifest.json").read_text())
    recorded: dict[str, str] = manifest.get("library_versions", {})
    faults = []
    for name in PINNED_LIBRARIES:
        installed = metadata.version(name)
        if name not in recorded:
            faults.append(f"the manifest records no {name} version")
        elif installed != recorded[name]:
            faults.append(f"{name} {installed} is installed, the bundle needs {recorded[name]}")
    if "torch" in recorded:
        installed = _base_version(metadata.version("torch"))
        if installed != _base_version(recorded["torch"]):
            faults.append(f"torch {installed} is installed, the bundle needs {recorded['torch']}")
    return faults


def main(bundle_dir: Path) -> int:
    """Print the bundle's faults and return 1 if it has any, otherwise load it and return 0."""
    faults = bundle_faults(bundle_dir) + library_faults(bundle_dir)
    for fault in faults:
        print(f"  [FAIL] {fault}", file=sys.stderr)
    if faults:
        return 1
    bundle = load_bundle(bundle_dir)
    require_servable_bundle(bundle)
    print(f"  {len(bundle.primaries)} Primaries, {len(bundle.seeds)} seeds, every SHA-256 matches.")
    print(f"  model_release {bundle.model_release}, refit month {bundle.refit_month}")
    print(f"  research commit {bundle.research_commit}")
    return 0


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1])))
