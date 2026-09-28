"""Reading rows of a GRIB edition 1 message that uses 16-bit simple packing, without a GRIB library.

**The staged ECMWF ENS files on Source Cooperative hold one 2 MB message per (member, step,
parameter), and the study only needs a band of rows from each.** This module parses the message
header, checks that the message uses the one packing this decoder understands, and turns a
truncated copy of the message (the header plus the first rows of data) into physical values.
Parsing the header is what lets a caller request a byte range that stops after the last row it
needs, and checking the header is what lets it trust the range it got back.

The layout of a GRIB1 message, as far as this module reads it:

- Indicator section, 8 bytes: `GRIB`, a 24-bit total message length, and the edition (1).
- Product definition section (PDS): the parameter, level, reference time, step, decimal scale
  factor `D`, and (for ECMWF local definition 30) the ensemble member number.
- Grid definition section (GDS), when the PDS flag says one is present: the grid size, the
  corner coordinates, and the scanning mode.
- Binary data section (BDS): an 11-byte header holding the flag byte, the binary scale factor `E`,
  the reference value `R` (an IBM hexadecimal float), and the bits per value, followed by the
  packed integers `X`.

A value decodes as `(R + X * 2^E) / 10^D`. The binary scale factor `E` and the decimal scale
factor `D` are sign-magnitude 16-bit integers, not two's complement, and `R` can be negative.
"""

import struct
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Final

import numpy as np

INDICATOR_SECTION_BYTES: Final[int] = 8
"""The length of the indicator section that opens every GRIB1 message."""

BDS_HEADER_BYTES: Final[int] = 11
"""The length of the fixed part of the binary data section, before the packed values."""

SUPPORTED_BITS_PER_VALUE: Final[int] = 16
"""The only packing width this decoder reads."""

TIME_UNIT_HOURS: Final[int] = 1
"""The GRIB1 code for a forecast time expressed in hours."""

SURFACE_LEVEL_TYPE: Final[int] = 1
"""The GRIB1 code for a level at the ground or water surface."""

ISOBARIC_LEVEL_TYPE: Final[int] = 100
"""The GRIB1 code for a level of constant pressure, whose value is in hectopascals."""

ECMWF_ENSEMBLE_LOCAL_DEFINITION: Final[int] = 30
"""The ECMWF local-definition number whose PDS carries the ensemble member number."""

_MEMBER_OFFSET_IN_PDS: Final[int] = 49
"""Zero-based offset within the PDS of the ensemble member number (octet 50)."""

_MEMBERS_OFFSET_IN_PDS: Final[int] = 50
"""Zero-based offset within the PDS of the ensemble size (octet 51)."""

_LOCAL_DEFINITION_OFFSET_IN_PDS: Final[int] = 40
"""Zero-based offset within the PDS of the local-definition number (octet 41)."""

_STEP_FROM_P1: Final[frozenset[int]] = frozenset({0, 1})
"""Time-range indicators for which octet 19 alone (`P1`) is the forecast step."""

_STEP_FROM_P1_P2_PAIR: Final[int] = 10
"""The time-range indicator for which octets 19 and 20 together are a 16-bit forecast step."""

PARAMETER_NAMES: Final[dict[tuple[int, int], str]] = {
    (128, 134): "sp",
    (128, 167): "2t",
    (128, 168): "2d",
    (128, 165): "10u",
    (128, 166): "10v",
    (228, 246): "100u",
    (228, 247): "100v",
    (128, 228): "tp",
    (128, 175): "strd",
    (128, 169): "ssrd",
    (128, 151): "msl",
    (128, 129): "z",
    (128, 164): "tcc",
    (128, 49): "10fg",
}
"""Short names, keyed by `(table 2 version, indicator of parameter)`.

The keys were read from ECMWF's own ecCodes definitions by setting each short name on a GRIB1
sample message. A parameter that is not listed here (the precipitation type, for one) makes
`parse_header` raise rather than guess.
"""


@dataclass(frozen=True)
class Grib1Header:
    """The fields of a GRIB1 message header that a caller needs to locate and check its data."""

    total_length: int
    """The message length in bytes, from the indicator section."""
    parameter: str
    """The short name of the parameter, from `PARAMETER_NAMES`."""
    level_type: int
    """The GRIB1 level-type code: `SURFACE_LEVEL_TYPE` or `ISOBARIC_LEVEL_TYPE`."""
    level: int
    """The level value: 0 at the surface, hectopascals on an isobaric level."""
    reference_time: datetime
    """The forecast initialisation time, in UTC."""
    step_hours: int
    """The forecast step in hours after `reference_time`."""
    member: int
    """The ensemble member: 0 for the control, 1 to 50 for a perturbed member."""
    ensemble_size: int
    """The number of members in the ensemble, control included."""
    decimal_scale: int
    """The decimal scale factor `D`."""
    binary_scale: int
    """The binary scale factor `E`."""
    reference_value: float
    """The reference value `R`, converted from an IBM float."""
    bits_per_value: int
    """The packed integer width in bits."""
    ni: int
    """The number of points along a row (longitudes)."""
    nj: int
    """The number of rows (latitudes)."""
    first_latitude_degrees: float
    """The latitude of the first row."""
    first_longitude_degrees: float
    """The longitude of the first column, in degrees east."""
    last_latitude_degrees: float
    """The latitude of the last row."""
    last_longitude_degrees: float
    """The longitude of the last column, in degrees east."""
    latitude_increment_degrees: float
    """The spacing between rows."""
    longitude_increment_degrees: float
    """The spacing between columns."""
    data_start: int
    """The byte offset within the message of the first packed value."""

    @property
    def row_bytes(self) -> int:
        """The bytes one row of packed values occupies."""
        return self.ni * self.bits_per_value // 8


def parse_ibm_float32(*, raw: bytes) -> float:
    """Convert four bytes holding an IBM hexadecimal float to a Python float.

    The layout is one sign bit, a 7-bit exponent of 16 with a bias of 64, and a 24-bit mantissa
    that is a fraction of one. Every such value is exactly representable as a 64-bit float, so the
    conversion loses nothing.

    Args:
        raw: Exactly four bytes.

    Returns:
        The value the four bytes encode.

    Raises:
        ValueError: If `raw` is not four bytes long.
    """
    if len(raw) != 4:
        raise ValueError(f"An IBM float is 4 bytes, got {len(raw)}")
    (word,) = struct.unpack(">I", raw)
    sign = -1.0 if word >> 31 else 1.0
    exponent = (word >> 24) & 0x7F
    mantissa = word & 0xFFFFFF
    return sign * mantissa / float(1 << 24) * 16.0 ** (exponent - 64)


def parse_sign_magnitude(*, raw: bytes) -> int:
    """Convert big-endian bytes holding a GRIB sign-magnitude integer to a Python integer.

    The top bit of the first byte is the sign and the remaining bits are the magnitude, so unlike
    two's complement there is a negative zero and `0x8001` is -1.

    Args:
        raw: One or more bytes.

    Returns:
        The signed value.
    """
    value = int.from_bytes(raw, byteorder="big")
    sign_bit = 1 << (8 * len(raw) - 1)
    return -(value & ~sign_bit) if value & sign_bit else value


def _u24(*, raw: bytes) -> int:
    return int.from_bytes(raw, byteorder="big")


@dataclass(frozen=True)
class _Pds:
    parameter: str
    level_type: int
    level: int
    reference_time: datetime
    step_hours: int
    member: int
    ensemble_size: int
    decimal_scale: int


@dataclass(frozen=True)
class _Gds:
    ni: int
    nj: int
    first_latitude: float
    first_longitude: float
    last_latitude: float
    last_longitude: float
    latitude_increment: float
    longitude_increment: float


@dataclass(frozen=True)
class _Bds:
    binary_scale: int
    reference_value: float
    bits_per_value: int


def _parse_pds(*, pds: bytes) -> _Pds:
    """Read the product definition section, refusing what this module does not understand."""
    section_flags = pds[7]
    if not section_flags & 0x80:
        raise ValueError("The PDS says there is no grid definition section")
    if section_flags & 0x40:
        raise ValueError("The PDS says a bitmap section is present, which is not supported")
    parameter_key = (pds[3], pds[8])
    if parameter_key not in PARAMETER_NAMES:
        raise ValueError(f"Unknown parameter: table 2 version {pds[3]}, indicator {pds[8]}")
    level_type = pds[9]
    if level_type not in (SURFACE_LEVEL_TYPE, ISOBARIC_LEVEL_TYPE):
        raise ValueError(f"Unsupported level type {level_type}")
    if pds[17] != TIME_UNIT_HOURS:
        raise ValueError(f"Only a time unit of hours is supported, got code {pds[17]}")
    time_range = pds[20]
    if time_range in _STEP_FROM_P1:
        step_hours = pds[18]
    elif time_range == _STEP_FROM_P1_P2_PAIR:
        step_hours = int.from_bytes(pds[18:20], byteorder="big")
    else:
        raise ValueError(f"Unsupported time range indicator {time_range}")
    local_definition = pds[_LOCAL_DEFINITION_OFFSET_IN_PDS]
    if local_definition != ECMWF_ENSEMBLE_LOCAL_DEFINITION:
        raise ValueError(
            f"Expected ECMWF local definition {ECMWF_ENSEMBLE_LOCAL_DEFINITION}, which carries "
            f"the member number, got {local_definition}"
        )
    return _Pds(
        parameter=PARAMETER_NAMES[parameter_key],
        level_type=level_type,
        level=int.from_bytes(pds[10:12], byteorder="big"),
        reference_time=datetime(
            year=(pds[24] - 1) * 100 + pds[12],
            month=pds[13],
            day=pds[14],
            hour=pds[15],
            minute=pds[16],
            tzinfo=UTC,
        ),
        step_hours=step_hours,
        member=pds[_MEMBER_OFFSET_IN_PDS],
        ensemble_size=pds[_MEMBERS_OFFSET_IN_PDS],
        decimal_scale=parse_sign_magnitude(raw=pds[26:28]),
    )


def _parse_gds(*, gds: bytes) -> _Gds:
    """Read the grid definition section, refusing any grid but the scan-mode-0 regular one."""
    if gds[5] != 0:
        raise ValueError(f"Only a regular latitude-longitude grid is supported, got type {gds[5]}")
    if gds[27] != 0:
        raise ValueError(
            f"Only scanning mode 0 (west to east, north to south) is supported: {gds[27]}"
        )
    return _Gds(
        ni=int.from_bytes(gds[6:8], byteorder="big"),
        nj=int.from_bytes(gds[8:10], byteorder="big"),
        first_latitude=parse_sign_magnitude(raw=gds[10:13]) / 1000,
        first_longitude=parse_sign_magnitude(raw=gds[13:16]) / 1000,
        last_latitude=parse_sign_magnitude(raw=gds[17:20]) / 1000,
        last_longitude=parse_sign_magnitude(raw=gds[20:23]) / 1000,
        longitude_increment=int.from_bytes(gds[23:25], byteorder="big") / 1000,
        latitude_increment=int.from_bytes(gds[25:27], byteorder="big") / 1000,
    )


def _parse_bds(*, bds: bytes, grid: _Gds) -> _Bds:
    """Read the binary data section header, refusing any packing but 16-bit simple packing."""
    flags = bds[3]
    if flags & 0xF0:
        raise ValueError(
            f"The BDS flag byte {flags:#04x} does not say grid-point data with simple packing"
        )
    bits_per_value = bds[10]
    if bits_per_value != SUPPORTED_BITS_PER_VALUE:
        raise ValueError(f"Only {SUPPORTED_BITS_PER_VALUE} bits per value are supported")
    unused_bits = flags & 0x0F
    packed_bits = (int.from_bytes(bds[0:3], byteorder="big") - BDS_HEADER_BYTES) * 8 - unused_bits
    if packed_bits != grid.ni * grid.nj * bits_per_value:
        raise ValueError("The BDS length does not match the number of grid points")
    return _Bds(
        binary_scale=parse_sign_magnitude(raw=bds[4:6]),
        reference_value=parse_ibm_float32(raw=bds[6:10]),
        bits_per_value=bits_per_value,
    )


def parse_header(*, message: bytes) -> Grib1Header:
    """Parse and validate the header of a GRIB1 message, which may be truncated after the header.

    This raises unless the message is edition 1, has a grid definition section and no bitmap
    section, is a grid-point field with simple packing at 16 bits per value, is on a regular
    latitude-longitude grid scanned west to east and north to south, and has a binary data section
    whose length matches the number of grid points.

    Args:
        message: At least the bytes up to the end of the binary data section header. A longer,
            truncated message is fine.

    Returns:
        The parsed header.

    Raises:
        ValueError: If the message is too short, is not the layout described above, or names a
            parameter, level, time unit or step encoding this module does not know.
    """
    if len(message) < INDICATOR_SECTION_BYTES + 3 or message[:4] != b"GRIB":
        raise ValueError("Not a GRIB message: it does not start with 'GRIB'")
    total_length = _u24(raw=message[4:7])
    if total_length & 0x800000:
        raise ValueError("Long-message length coding is not supported")
    if message[7] != 1:
        raise ValueError(f"Only GRIB edition 1 is supported, got edition {message[7]}")

    gds_start = INDICATOR_SECTION_BYTES + _u24(raw=message[8:11])
    if len(message) < gds_start + 3:
        raise ValueError("The message is too short to hold the whole product definition section")
    bds_start = gds_start + _u24(raw=message[gds_start : gds_start + 3])
    if len(message) < bds_start + BDS_HEADER_BYTES:
        raise ValueError("The message is too short to hold the binary data section header")
    product = _parse_pds(pds=message[INDICATOR_SECTION_BYTES:gds_start])
    grid = _parse_gds(gds=message[gds_start:bds_start])
    packing = _parse_bds(bds=message[bds_start : bds_start + BDS_HEADER_BYTES], grid=grid)

    return Grib1Header(
        total_length=total_length,
        parameter=product.parameter,
        level_type=product.level_type,
        level=product.level,
        reference_time=product.reference_time,
        step_hours=product.step_hours,
        member=product.member,
        ensemble_size=product.ensemble_size,
        decimal_scale=product.decimal_scale,
        binary_scale=packing.binary_scale,
        reference_value=packing.reference_value,
        bits_per_value=packing.bits_per_value,
        ni=grid.ni,
        nj=grid.nj,
        first_latitude_degrees=grid.first_latitude,
        first_longitude_degrees=grid.first_longitude,
        last_latitude_degrees=grid.last_latitude,
        last_longitude_degrees=grid.last_longitude,
        latitude_increment_degrees=grid.latitude_increment,
        longitude_increment_degrees=grid.longitude_increment,
        data_start=bds_start + BDS_HEADER_BYTES,
    )


def bytes_needed_for_rows(*, header: Grib1Header, last_row: int) -> int:
    """Return how many bytes from the start of the message hold rows `0` to `last_row`.

    Args:
        header: A parsed header.
        last_row: The zero-based index of the last row wanted, counted from the first row.

    Returns:
        The message prefix length in bytes.
    """
    return header.data_start + (last_row + 1) * header.row_bytes


def unpack_rows(
    *, message: bytes, header: Grib1Header, first_row: int, last_row: int
) -> np.ndarray:
    """Return the packed integers `X` of rows `first_row` to `last_row` inclusive.

    Args:
        message: The message, or a prefix of it that reaches the end of `last_row`.
        header: The header parsed from `message`.
        first_row: The zero-based index of the first row wanted.
        last_row: The zero-based index of the last row wanted.

    Returns:
        An unsigned 16-bit array of shape `(last_row - first_row + 1, header.ni)`.

    Raises:
        ValueError: If the rows are outside the grid, or `message` is too short to hold them.
    """
    if not 0 <= first_row <= last_row < header.nj:
        raise ValueError(f"Rows {first_row} to {last_row} are outside the {header.nj}-row grid")
    end = bytes_needed_for_rows(header=header, last_row=last_row)
    if len(message) < end:
        raise ValueError(f"The message has {len(message)} bytes but rows need {end}")
    start = header.data_start + first_row * header.row_bytes
    packed = np.frombuffer(message, dtype=">u2", count=(end - start) // 2, offset=start)
    return packed.astype(np.uint16).reshape(last_row - first_row + 1, header.ni)


def decode_values(*, packed: np.ndarray, header: Grib1Header) -> np.ndarray:
    """Convert packed integers `X` to physical values, as `(R + X * 2^E) / 10^D` in 64 bits.

    The division by `10^D` is done as a multiplication by a factor built by repeatedly dividing by
    ten, in the order ecCodes does it, so that the result equals ecCodes' to the last bit.

    Args:
        packed: The unsigned integers from `unpack_rows`.
        header: The header of the same message.

    Returns:
        A 64-bit float array shaped like `packed`.
    """
    decimal_factor = 1.0
    for _ in range(abs(header.decimal_scale)):
        decimal_factor = (
            decimal_factor / 10.0 if header.decimal_scale > 0 else decimal_factor * 10.0
        )
    binary_factor = 2.0**header.binary_scale
    return (packed.astype(np.float64) * binary_factor + header.reference_value) * decimal_factor
