"""
Simple range coder (arithmetic coding) with a stable rANS-like API.

This replaces the previous lzma shim so benchmarks exercise a real entropy coder.
"""
from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
import struct
from typing import List, Tuple

import numpy as np


MAGIC = b"RC01"
TOP = 1 << 24
MASK32 = (1 << 32) - 1
MAX_TOTAL = 1 << 15


@dataclass
class FreqTable:
    cum: List[int]  # cum[0]=0, cum[-1]=total
    total: int

    @classmethod
    def from_freqs(cls, freqs: List[int]) -> "FreqTable":
        cum = [0]
        total = 0
        for f in freqs:
            if f <= 0:
                raise ValueError("freqs must be positive")
            total += int(f)
            cum.append(total)
        return cls(cum=cum, total=total)

    def sym_to_range(self, sym: int) -> Tuple[int, int]:
        return self.cum[sym], self.cum[sym + 1]

    def range_to_sym(self, x: int) -> int:
        return bisect_right(self.cum, x) - 1


class RangeEncoder:
    def __init__(self) -> None:
        self.low = 0
        self.range = 0xFFFFFFFF
        self.out = bytearray()

    def _normalize(self) -> None:
        while self.range < TOP:
            self.out.append((self.low >> 24) & 0xFF)
            self.low = (self.low << 8) & MASK32
            self.range = (self.range << 8) & MASK32

    def encode(self, sym: int, ft: FreqTable) -> None:
        self._normalize()
        r = self.range // ft.total
        lo, hi = ft.sym_to_range(sym)
        self.low = (self.low + r * lo) & MASK32
        self.range = (r * (hi - lo)) & MASK32

    def finish(self) -> bytes:
        for _ in range(4):
            self.out.append((self.low >> 24) & 0xFF)
            self.low = (self.low << 8) & MASK32
        return bytes(self.out)


class RangeDecoder:
    def __init__(self, data: bytes) -> None:
        self.data = data
        self.pos = 0
        self.low = 0
        self.range = 0xFFFFFFFF
        self.code = 0
        for _ in range(4):
            self.code = ((self.code << 8) | self._read_byte()) & MASK32

    def _read_byte(self) -> int:
        if self.pos >= len(self.data):
            return 0
        b = self.data[self.pos]
        self.pos += 1
        return b

    def _normalize(self) -> None:
        while self.range < TOP:
            self.code = ((self.code << 8) | self._read_byte()) & MASK32
            self.low = (self.low << 8) & MASK32
            self.range = (self.range << 8) & MASK32

    def decode(self, ft: FreqTable) -> int:
        self._normalize()
        r = self.range // ft.total
        x = ((self.code - self.low) // r) if r else 0
        sym = ft.range_to_sym(int(x))
        lo, hi = ft.sym_to_range(sym)
        self.low = (self.low + r * lo) & MASK32
        self.range = (r * (hi - lo)) & MASK32
        return sym


def _build_freqs(data: np.ndarray, alphabet: int) -> List[int]:
    counts = np.bincount(data, minlength=alphabet).astype(np.int64)
    counts += 1  # smoothing keeps all symbols decodable
    total = int(counts.sum())
    if total <= MAX_TOTAL:
        return counts.astype(int).tolist()

    scale = MAX_TOTAL / total
    freqs = np.maximum(1, np.floor(counts * scale)).astype(np.int64)
    diff = MAX_TOTAL - int(freqs.sum())
    if diff > 0:
        freqs[:diff] += 1
    elif diff < 0:
        idx = np.where(freqs > 1)[0]
        take = min(-diff, len(idx))
        freqs[idx[:take]] -= 1
    return freqs.astype(int).tolist()


def encode(data: np.ndarray, alphabet: int = 256) -> bytes:
    arr = np.asarray(data, dtype=np.uint8).ravel()
    if arr.size == 0:
        header = MAGIC + struct.pack("<H", alphabet) + b""
        return header
    used = int(arr.max()) + 1
    alphabet = max(int(alphabet), used)
    if alphabet > 256:
        raise ValueError("alphabet must be <= 256")

    freqs = _build_freqs(arr, alphabet)
    ft = FreqTable.from_freqs(freqs)
    enc = RangeEncoder()
    for sym in arr:
        enc.encode(int(sym), ft)
    payload = enc.finish()

    header = bytearray()
    header += MAGIC
    header += struct.pack("<H", alphabet)
    for f in freqs:
        header += struct.pack("<I", int(f))
    return bytes(header) + payload


def decode(payload: bytes, length: int, alphabet: int = 256) -> np.ndarray:
    if length <= 0:
        return np.array([], dtype=np.uint8)
    if len(payload) < len(MAGIC) + 2:
        raise ValueError("payload too small")
    magic = payload[:4]
    if magic != MAGIC:
        raise ValueError("bad payload magic")
    header_pos = 4
    stored_alphabet = struct.unpack_from("<H", payload, header_pos)[0]
    header_pos += 2
    alphabet = stored_alphabet or alphabet
    if alphabet > 256:
        raise ValueError("alphabet must be <= 256")
    needed = header_pos + 4 * alphabet
    if len(payload) < needed:
        raise ValueError("payload missing frequency table")

    freqs = []
    for i in range(alphabet):
        f = struct.unpack_from("<I", payload, header_pos + 4 * i)[0]
        if f == 0:
            raise ValueError("zero frequency in payload")
        freqs.append(int(f))
    header_pos = needed

    ft = FreqTable.from_freqs(freqs)
    dec = RangeDecoder(payload[header_pos:])
    out = np.empty(length, dtype=np.uint8)
    for i in range(length):
        out[i] = dec.decode(ft)
    return out
