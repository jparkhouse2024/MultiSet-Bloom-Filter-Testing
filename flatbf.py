"""
FlatBloofi packs many per-set Bloom filters into transposed bit slices.

"""

from __future__ import annotations

from typing import Dict, List, Optional

from BloomFilter import BloomFilter, stable_hashes


# ============================================================
# Flat-Bloofi Block
# ============================================================


class FlatBloofiBlock:
    """
    Stores up to 64 Bloom filters packed together.

    layout[i] is a 64-bit integer where bit j
    corresponds to Bloom filter j's bit i.
    """

    CAPACITY = 64

    def __init__(self, m: int):
        self.m = m

        # One 64-bit integer per Bloom filter bit position
        self.layout = [0] * m

        # Bitmask of occupied slots
        self.used_mask = 0

        # slot -> bloom filter id
        self.slot_to_id: List[Optional[str]] = [None] * 64

        # bloom filter id -> slot
        self.id_to_slot: Dict[str, int] = {}

    def is_full(self) -> bool:
        return self.used_mask == (1 << 64) - 1

    def empty(self) -> bool:
        return self.used_mask == 0

    def _find_free_slot(self) -> Optional[int]:
        for i in range(64):
            if ((self.used_mask >> i) & 1) == 0:
                return i
        return None

    # --------------------------------------------------------
    # Insert
    # --------------------------------------------------------

    def insert(self, bf_id: str, bf: BloomFilter):
        if self.is_full():
            raise ValueError("FlatBloofiBlock full")

        if bf_id in self.id_to_slot:
            raise ValueError("Duplicate Bloom filter id")

        slot = self._find_free_slot()

        if slot is None:
            raise RuntimeError("No free slot")

        self.used_mask |= (1 << slot)

        self.slot_to_id[slot] = bf_id
        self.id_to_slot[bf_id] = slot

        for pos in bf.bit_positions():
            self.layout[pos] |= (1 << slot)

    # --------------------------------------------------------
    # Query
    # --------------------------------------------------------

    def query(self, hashes: List[int]) -> List[str]:
        if not hashes:
            return []

        result_mask = self.used_mask

        for h in hashes:
            result_mask &= self.layout[h]

            if result_mask == 0:
                return []

        matches = []

        x = result_mask

        while x:
            lsb = x & -x
            slot = lsb.bit_length() - 1

            bf_id = self.slot_to_id[slot]

            if bf_id is not None:
                matches.append(bf_id)

            x ^= lsb

        return matches


# ============================================================
# Flat-Bloofi Index
# ============================================================


class FlatBloofi:
    def __init__(self, m: int, k: int):
        self.m = m
        self.k = k

        self.blocks: List[FlatBloofiBlock] = []

        # Store original Bloom filters
        self.filters: Dict[str, BloomFilter] = {}

    # --------------------------------------------------------
    # Hash helper
    # --------------------------------------------------------

    def _hashes(self, item: str) -> List[int]:
        return list(stable_hashes(item, self.m, self.k))

    # --------------------------------------------------------
    # Insert
    # --------------------------------------------------------

    def insert(self, bf_id: str, bf: BloomFilter):
        if bf_id in self.filters:
            raise ValueError("Duplicate Bloom filter id")

        block = None

        for b in self.blocks:
            if not b.is_full():
                block = b
                break

        if block is None:
            block = FlatBloofiBlock(self.m)
            self.blocks.append(block)

        block.insert(bf_id, bf)

        self.filters[bf_id] = bf

    # --------------------------------------------------------
    # Query
    # --------------------------------------------------------

    def query(self, item: str) -> List[str]:
        hashes = self._hashes(item)

        matches = []

        for block in self.blocks:
            matches.extend(block.query(hashes))

        return matches


# ============================================================
# Helper
# ============================================================


def build_filter(m: int, k: int, items: List[str]) -> BloomFilter:
    bf = BloomFilter(m, k)

    for item in items:
        bf.add(item)

    return bf


# ============================================================
# Example
# ============================================================


if __name__ == "__main__":
    M = 25600
    K = 4

    index = FlatBloofi(M, K)

    bf1 = build_filter(M, K, ["alice", "bob", "charlie"])
    bf2 = build_filter(M, K, ["dog", "cat", "mouse"])
    bf3 = build_filter(M, K, ["apple", "banana", "pear"])

    index.insert("bf1", bf1)
    index.insert("bf2", bf2)
    index.insert("fruits", bf3)

    print(index.query("alice"))
    print(index.query("cat"))
    print(index.query("banana"))
    print(index.query("unknown"))
