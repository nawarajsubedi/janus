"""Minimal decoder for the AWS event-stream framing used by ConverseStream."""

import json
import struct
from typing import Any


def parse_frames(buf: bytearray) -> list[tuple[str, dict[str, Any]]]:
    """Pop complete frames off `buf`; return (event-type, json payload) pairs.

    Frame: total_len, headers_len, prelude_crc (4 bytes each) | headers | payload | crc.
    Bedrock headers are all type 7 (string): name_len, name, 0x07, value_len(2), value.
    Exception frames are reported with event-type "error".
    """
    events = []
    while len(buf) >= 12:
        total, hlen = struct.unpack(">II", buf[:8])
        if len(buf) < total:
            break
        headers, i = {}, 12
        while i < 12 + hlen:
            nlen = buf[i]
            name = bytes(buf[i + 1 : i + 1 + nlen]).decode()
            (vlen,) = struct.unpack(">H", buf[i + 2 + nlen : i + 4 + nlen])
            headers[name] = bytes(buf[i + 4 + nlen : i + 4 + nlen + vlen]).decode()
            i += 4 + nlen + vlen
        payload = bytes(buf[12 + hlen : total - 4])
        kind = headers.get(":event-type", "error") if headers.get(":message-type") == "event" else "error"
        events.append((kind, json.loads(payload) if payload else {}))
        del buf[:total]
    return events
