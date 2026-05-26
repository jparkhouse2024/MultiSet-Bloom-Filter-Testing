from __future__ import annotations

import hashlib
from typing import Iterator, Union


HashableItem = Union[str, bytes, bytearray, memoryview, int]


def _to_bytes(item: HashableItem) -> bytes:
    if isinstance(item, bytes):
        return item
    if isinstance(item, bytearray):
        return bytes(item)
    if isinstance(item, memoryview):
        return item.tobytes()
    return str(item).encode("utf-8")


def stable_hashes(
    item: HashableItem,
    m: int,
    k: int,
    namespace: HashableItem = b"",
) -> Iterator[int]:
    """
    Shared double-hashing routine for every Bloom-style structure.
    """
    if m <= 0:
        raise ValueError("m must be positive")
    if k < 0:
        raise ValueError("k must be non-negative")

    data = _to_bytes(item)
    ns = _to_bytes(namespace)

    if ns:
        data = ns + b"\0" + data

    h1 = int.from_bytes(hashlib.blake2b(data, digest_size=16).digest(), "big")
    h2 = int.from_bytes(hashlib.sha256(data).digest(), "big")
    step = 1 + (h2 % (m - 1)) if m > 1 else 0

    for i in range(k):
        yield (h1 + i * step) % m


class BloomFilter:
    """
    Standard Bloom filter backed by the repository-wide hash helper.
    """

    def __init__(
        self,
        m_bits: int,
        k_hashes: int,
        namespace: HashableItem = b"",
    ):
        if m_bits <= 0:
            raise ValueError("m_bits must be positive")
        if k_hashes < 0:
            raise ValueError("k_hashes must be non-negative")

        self.m = m_bits
        self.k = k_hashes
        self.namespace = namespace
        self.bits = bytearray((m_bits + 7) // 8)

    def _hashes(self, item: HashableItem) -> Iterator[int]:
        return stable_hashes(item, self.m, self.k, self.namespace)

    def _set_bit(self, idx: int):
        self.bits[idx // 8] |= 1 << (idx % 8)

    def _get_bit(self, idx: int) -> bool:
        return bool((self.bits[idx // 8] >> (idx % 8)) & 1)

    def add(self, item: HashableItem):
        for h in self._hashes(item):
            self._set_bit(h)

    def contains(self, item: HashableItem) -> bool:
        return all(self._get_bit(h) for h in self._hashes(item))

    def bit_positions(self) -> Iterator[int]:
        for byte_idx, value in enumerate(self.bits):
            while value:
                lsb = value & -value
                bit_idx = lsb.bit_length() - 1
                yield byte_idx * 8 + bit_idx
                value ^= lsb
