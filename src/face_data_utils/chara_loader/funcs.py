import struct

from msgpack import Unpacker, packb, unpackb


def load_length(data_stream, struct_type):
    length = struct.unpack(struct_type, data_stream.read(struct.calcsize(struct_type)))[
        0
    ]
    return data_stream.read(length)


def load_string(data_stream):
    length = 0
    i = 0
    while True:
        serial = struct.unpack("B", data_stream.read(struct.calcsize("B")))[0]
        length |= (0b01111111 & serial) << 7 * i
        if serial >> 7 != 1:
            break
        i += 1
    data = data_stream.read(length)
    return data


def load_type(data_stream, struct_type):
    return struct.unpack(struct_type, data_stream.read(struct.calcsize(struct_type)))[0]


def write_string(data_stream, value):
    length_bytes = b""
    length = len(value)
    while True:
        serial = length & 0b1111111
        if length >> 7 != 0:
            length = length >> 7
            length_bytes += struct.pack("b", 0b10000000 | serial)
        else:
            length_bytes += struct.pack("b", serial)
            break
    data_stream.write(length_bytes)
    data_stream.write(value)


def msg_unpack(data):
    return unpackb(data, raw=False, strict_map_key=False)


def msg_pack(data):
    serialized = packb(data, use_single_float=True, use_bin_type=True)
    return serialized, len(serialized)


def splice_map(orig: bytes, overrides: dict) -> bytes:
    """Re-emit a top-level msgpack map, replacing/appending only ``overrides`` keys and copying
    every other key/value's ORIGINAL bytes verbatim.

    A naive unpack->repack of the card's KKEx (ExtensibleSaveFormat plugin data) minimizes integer
    widths (e.g. int32 ``15`` -> positive fixint). MessagePack-C# deserializes those typeless dicts
    by wire format, so a fixint becomes a boxed ``byte`` instead of ``int`` and a plugin's
    ``(int)data[key]`` cast throws ``InvalidCastException`` (observed in AdvIKPlugin). Preserving the
    untouched plugins' original bytes avoids that entirely; only the keys we actually change are repacked.
    """
    b0 = orig[0]
    if 0x80 <= b0 <= 0x8f:
        count, hlen = b0 & 0x0f, 1
    elif b0 == 0xde:
        count, hlen = struct.unpack(">H", orig[1:3])[0], 3
    elif b0 == 0xdf:
        count, hlen = struct.unpack(">I", orig[1:5])[0], 5
    else:
        raise ValueError("KKEx is not a msgpack map (first byte 0x%02x)" % b0)

    body = orig[hlen:]
    unp = Unpacker(raw=False, strict_map_key=False)
    unp.feed(body)
    seen, entries = set(), []  # entries: (key_bytes, value_bytes)
    for _ in range(count):
        k0 = unp.tell(); key = unp.unpack(); v0 = unp.tell(); unp.unpack(); v1 = unp.tell()
        seen.add(key)
        value = packb(overrides[key], use_single_float=True, use_bin_type=True) \
            if key in overrides else body[v0:v1]
        entries.append((body[k0:v0], value))
    for k, v in overrides.items():  # append keys not already present (e.g. a fresh KKABMPlugin.ABMData)
        if k not in seen:
            entries.append((packb(k, use_bin_type=True),
                            packb(v, use_single_float=True, use_bin_type=True)))

    n = len(entries)
    if n <= 0x0f:
        header = struct.pack("B", 0x80 | n)
    elif n <= 0xffff:
        header = b"\xde" + struct.pack(">H", n)
    else:
        header = b"\xdf" + struct.pack(">I", n)
    return header + b"".join(kb + vb for kb, vb in entries)


def get_png_length(png_data, orig=0):
    idx = orig
    assert png_data[idx : idx + 8] == b"\x89\x50\x4e\x47\x0d\x0a\x1a\x0a"

    idx += 8
    while True:
        chunk_len = struct.unpack(">I", png_data[idx : idx + 4])[0]
        chunk_type = png_data[idx + 4 : idx + 8].decode()
        idx += chunk_len + 12
        if chunk_type == "IEND":
            break
    return idx - orig


def get_png(data_stream):
    origin_pos = data_stream.tell()
    assert data_stream.read(8) == b"\x89\x50\x4e\x47\x0d\x0a\x1a\x0a"
    while True:
        length = load_type(data_stream, ">I")
        chunk_type = data_stream.read(4)
        data_stream.read(length + 4)
        if chunk_type == b"IEND":
            break
    end_pos = data_stream.tell()
    data_stream.seek(origin_pos)
    png_data = data_stream.read(end_pos - origin_pos)
    return png_data
