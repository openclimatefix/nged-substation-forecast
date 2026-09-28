from datetime import UTC, datetime

import numpy as np
import pytest
from studies.grib1_simple import (
    Grib1Header,
    bytes_needed_for_rows,
    decode_values,
    parse_header,
    parse_ibm_float32,
    parse_sign_magnitude,
    unpack_rows,
)

NI = 4
NJ = 3


def encode_ibm_float32(*, value: float) -> bytes:
    """Encode a float whose mantissa fits 24 bits as an IBM hexadecimal float."""
    if value == 0:
        return bytes(4)
    sign = 0x80 if value < 0 else 0
    magnitude = abs(value)
    exponent = 64
    while magnitude >= 1:
        magnitude /= 16
        exponent += 1
    while magnitude < 1 / 16:
        magnitude *= 16
        exponent -= 1
    mantissa = int(magnitude * (1 << 24))
    assert mantissa / (1 << 24) == magnitude
    return bytes([sign | exponent]) + mantissa.to_bytes(3, byteorder="big")


def encode_sign_magnitude(*, value: int, length: int) -> bytes:
    sign_bit = 1 << (8 * length - 1)
    return (abs(value) | (sign_bit if value < 0 else 0)).to_bytes(length, byteorder="big")


def build_message(
    *,
    packed: np.ndarray,
    reference_value: float = 0.0,
    binary_scale: int = 0,
    decimal_scale: int = 0,
    bits_per_value: int = 16,
    with_bitmap: bool = False,
    time_range: int = 1,
    step: int = 3,
    indicator: int = 167,
    table: int = 128,
    member: int = 7,
) -> bytes:
    pds = bytearray(52)
    pds[0:3] = (52).to_bytes(3, byteorder="big")
    pds[3] = table
    pds[7] = 0x80 | (0x40 if with_bitmap else 0)
    pds[8] = indicator
    pds[9] = 1
    pds[12:17] = bytes([23, 6, 26, 0, 0])
    pds[17] = 1
    if time_range == 10:
        pds[18:20] = step.to_bytes(2, byteorder="big")
    else:
        pds[18] = step
    pds[20] = time_range
    pds[24] = 21
    pds[26:28] = encode_sign_magnitude(value=decimal_scale, length=2)
    pds[40] = 30
    pds[49] = member
    pds[50] = 51

    gds = bytearray(32)
    gds[0:3] = (32).to_bytes(3, byteorder="big")
    gds[6:8] = NI.to_bytes(2, byteorder="big")
    gds[8:10] = NJ.to_bytes(2, byteorder="big")
    gds[10:13] = encode_sign_magnitude(value=90000, length=3)
    gds[13:16] = encode_sign_magnitude(value=0, length=3)
    gds[17:20] = encode_sign_magnitude(value=89500, length=3)
    gds[20:23] = encode_sign_magnitude(value=359750, length=3)
    gds[23:25] = (250).to_bytes(2, byteorder="big")
    gds[25:27] = (250).to_bytes(2, byteorder="big")

    data = packed.astype(">u2").tobytes()
    padded = data + bytes(len(data) % 2)
    bds = bytearray(11)
    bds[0:3] = (11 + len(padded)).to_bytes(3, byteorder="big")
    bds[3] = 8 * (len(padded) - len(data))
    bds[4:6] = encode_sign_magnitude(value=binary_scale, length=2)
    bds[6:10] = encode_ibm_float32(value=reference_value)
    bds[10] = bits_per_value

    body = bytes(pds) + bytes(gds) + bytes(bds) + padded
    total = 8 + len(body) + 4
    return b"GRIB" + total.to_bytes(3, byteorder="big") + b"\x01" + body + b"7777"


PACKED = np.arange(NI * NJ, dtype=np.uint16).reshape(NJ, NI) * 1000


def test_ibm_float_round_trips_a_negative_value():
    assert parse_ibm_float32(raw=encode_ibm_float32(value=-118.625)) == -118.625


def test_ibm_float_decodes_a_known_word():
    # 0x42640000 is 100.0: exponent 0x42 - 64 = 2, mantissa 0x640000 / 2**24 = 0.390625.
    assert parse_ibm_float32(raw=bytes.fromhex("42640000")) == 100.0


def test_ibm_float_of_zero_is_zero():
    assert parse_ibm_float32(raw=bytes(4)) == 0.0


def test_ibm_float_needs_four_bytes():
    with pytest.raises(ValueError, match="4 bytes"):
        parse_ibm_float32(raw=bytes(3))


def test_sign_magnitude_is_not_twos_complement():
    assert parse_sign_magnitude(raw=bytes.fromhex("8001")) == -1
    assert parse_sign_magnitude(raw=bytes.fromhex("0005")) == 5
    assert parse_sign_magnitude(raw=bytes.fromhex("800f")) == -15


def test_header_fields_are_read_from_the_right_octets():
    header = parse_header(message=build_message(packed=PACKED, reference_value=-3.0, step=9))

    assert header == Grib1Header(
        total_length=len(build_message(packed=PACKED, reference_value=-3.0, step=9)),
        parameter="2t",
        level_type=1,
        level=0,
        reference_time=datetime(2023, 6, 26, tzinfo=UTC),
        step_hours=9,
        member=7,
        ensemble_size=51,
        decimal_scale=0,
        binary_scale=0,
        reference_value=-3.0,
        bits_per_value=16,
        ni=NI,
        nj=NJ,
        first_latitude_degrees=90.0,
        first_longitude_degrees=0.0,
        last_latitude_degrees=89.5,
        last_longitude_degrees=359.75,
        latitude_increment_degrees=0.25,
        longitude_increment_degrees=0.25,
        data_start=8 + 52 + 32 + 11,
    )


def test_a_step_above_255_is_read_from_both_step_octets():
    header = parse_header(message=build_message(packed=PACKED, time_range=10, step=360))

    assert header.step_hours == 360


def test_a_header_is_readable_from_a_message_cut_after_the_bds_header():
    message = build_message(packed=PACKED)

    header = parse_header(message=message[: 8 + 52 + 32 + 11])

    assert header.parameter == "2t"


def test_decode_uses_a_negative_reference_value_and_negative_binary_scale():
    message = build_message(packed=PACKED, reference_value=-2.5, binary_scale=-2)
    header = parse_header(message=message)

    values = decode_values(
        packed=unpack_rows(message=message, header=header, first_row=0, last_row=NJ - 1),
        header=header,
    )

    assert values.tolist() == (-2.5 + PACKED * 0.25).tolist()


def test_decode_applies_a_positive_decimal_scale():
    message = build_message(packed=PACKED, reference_value=-3.0, binary_scale=1, decimal_scale=1)
    header = parse_header(message=message)

    values = decode_values(
        packed=unpack_rows(message=message, header=header, first_row=0, last_row=NJ - 1),
        header=header,
    )

    assert values.tolist() == ((PACKED * 2.0 - 3.0) * 0.1).tolist()
    assert values[0, 1] == pytest.approx(199.7)


def test_decode_applies_a_negative_decimal_scale():
    message = build_message(packed=PACKED, decimal_scale=-2)
    header = parse_header(message=message)

    values = decode_values(
        packed=unpack_rows(message=message, header=header, first_row=0, last_row=NJ - 1),
        header=header,
    )

    assert values.tolist() == (PACKED * 100.0).tolist()


def test_unpack_rows_returns_only_the_requested_rows():
    message = build_message(packed=PACKED)
    header = parse_header(message=message)

    rows = unpack_rows(message=message, header=header, first_row=1, last_row=2)

    assert rows.tolist() == PACKED[1:3].tolist()


def test_unpack_rows_works_on_a_message_cut_after_the_last_wanted_row():
    message = build_message(packed=PACKED)
    header = parse_header(message=message)
    cut = message[: bytes_needed_for_rows(header=header, last_row=1)]

    rows = unpack_rows(message=cut, header=header, first_row=0, last_row=1)

    assert rows.tolist() == PACKED[:2].tolist()


def test_unpack_rows_refuses_a_message_cut_short_of_the_rows():
    message = build_message(packed=PACKED)
    header = parse_header(message=message)
    cut = message[: bytes_needed_for_rows(header=header, last_row=1)]

    with pytest.raises(ValueError, match="rows need"):
        unpack_rows(message=cut, header=header, first_row=0, last_row=2)


def test_unpack_rows_refuses_rows_outside_the_grid():
    message = build_message(packed=PACKED)
    header = parse_header(message=message)

    with pytest.raises(ValueError, match="outside"):
        unpack_rows(message=message, header=header, first_row=0, last_row=NJ)


def test_a_bitmap_is_refused():
    with pytest.raises(ValueError, match="bitmap"):
        parse_header(message=build_message(packed=PACKED, with_bitmap=True))


def test_a_width_other_than_16_bits_is_refused():
    with pytest.raises(ValueError, match="16 bits per value"):
        parse_header(message=build_message(packed=PACKED, bits_per_value=12))


def test_an_unknown_parameter_is_refused():
    with pytest.raises(ValueError, match="Unknown parameter"):
        parse_header(message=build_message(packed=PACKED, indicator=1))


def test_a_time_range_that_is_not_understood_is_refused():
    with pytest.raises(ValueError, match="time range"):
        parse_header(message=build_message(packed=PACKED, time_range=4))


def test_a_message_that_is_not_grib_is_refused():
    with pytest.raises(ValueError, match="GRIB"):
        parse_header(message=b"\x00" * 200)


def test_a_bds_length_that_disagrees_with_the_grid_is_refused():
    message = bytearray(build_message(packed=PACKED))
    bds_length_offset = 8 + 52 + 32
    message[bds_length_offset : bds_length_offset + 3] = (11 + 4).to_bytes(3, byteorder="big")

    with pytest.raises(ValueError, match="number of grid points"):
        parse_header(message=bytes(message))
