#!/usr/bin/env python3
"""Extract a ZIP member that is itself an uncompressed ZIP, without copying it."""

from __future__ import annotations

import argparse
import io
import struct
import zipfile
from pathlib import Path


class BoundedFile(io.RawIOBase):
    def __init__(self, path: Path, start: int, length: int) -> None:
        self._handle = path.open("rb")
        self._start = start
        self._length = length
        self._position = 0

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self._position

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            position = offset
        elif whence == io.SEEK_CUR:
            position = self._position + offset
        elif whence == io.SEEK_END:
            position = self._length + offset
        else:
            raise ValueError(f"Unsupported whence: {whence}")
        if position < 0:
            raise ValueError("Negative seek position")
        self._position = min(position, self._length)
        return self._position

    def readinto(self, buffer: bytearray) -> int:
        remaining = self._length - self._position
        count = min(len(buffer), remaining)
        if count <= 0:
            return 0
        self._handle.seek(self._start + self._position)
        data = self._handle.read(count)
        buffer[: len(data)] = data
        self._position += len(data)
        return len(data)

    def close(self) -> None:
        self._handle.close()
        super().close()


def stored_member_extent(outer_path: Path, member_name: str) -> tuple[int, int]:
    with zipfile.ZipFile(outer_path) as outer:
        info = outer.getinfo(member_name)
        if info.compress_type != zipfile.ZIP_STORED:
            raise ValueError("Nested member must be stored without compression")
        header_offset = info.header_offset
        length = info.file_size

    with outer_path.open("rb") as handle:
        handle.seek(header_offset)
        header = handle.read(30)
    signature, = struct.unpack_from("<I", header, 0)
    if signature != 0x04034B50:
        raise ValueError("Invalid local ZIP header")
    filename_length, extra_length = struct.unpack_from("<HH", header, 26)
    return header_offset + 30 + filename_length + extra_length, length


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outer", type=Path, required=True)
    parser.add_argument("--member", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    start, length = stored_member_extent(args.outer, args.member)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with BoundedFile(args.outer, start, length) as bounded:
        with zipfile.ZipFile(bounded) as nested:
            bad = nested.testzip()
            if bad is not None:
                raise ValueError(f"Nested CRC failure: {bad}")
            nested.extractall(args.output_dir)
            print(f"Extracted {len(nested.infolist())} entries")


if __name__ == "__main__":
    main()

