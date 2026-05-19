import hashlib
import math
from collections import deque


class BloomFilter:
    """
    Standard Bloom filter using double hashing.
    """

    def __init__(self, m_bits: int, k_hashes: int):
        self.m = m_bits
        self.k = k_hashes
        self.bits = bytearray((m_bits + 7) // 8)

    def _hashes(self, item: bytes):
        h1 = int(hashlib.blake2b(item, digest_size=16).hexdigest(), 16)
        h2 = int(hashlib.sha256(item).hexdigest(), 16)

        for i in range(self.k):
            yield (h1 + i * h2) % self.m

    def _set_bit(self, idx: int):
        self.bits[idx // 8] |= 1 << (idx % 8)

    def _get_bit(self, idx: int) -> bool:
        return (self.bits[idx // 8] >> (idx % 8)) & 1

    def add(self, item: bytes):
        for h in self._hashes(item):
            self._set_bit(h)

    def contains(self, item: bytes) -> bool:
        return all(self._get_bit(h) for h in self._hashes(item))