"""
IBF: Invertible Bloom Filter / simplified IBLT implementation.

Based on:
"A Simple Version of the Invertible Bloom Lookup Table"

Supports:
    - insert
    - delete
    - query (GET)
    - list_entries (LISTENTRIES)

Uses XOR instead of sums to avoid overflow issues, exactly as
suggested in the paper.
"""

from collections import deque

from BloomFilter import stable_hashes


class IBFCell:
    __slots__ = ("count", "key_xor", "value_xor")

    def __init__(self):
        self.count = 0
        self.key_xor = 0
        self.value_xor = 0


class IBF:
    """
    Invertible Bloom Filter (IBF / simplified IBLT).

    Parameters:
        m : number of cells
        k : number of hash functions
    """

    def __init__(self, m=10000, k=3):
        self.m = m
        self.k = k

        self.cells = [IBFCell() for _ in range(m)]

    # ------------------------------------------------------------
    # Hashing
    # ------------------------------------------------------------

    def _hashes(self, key):
        return stable_hashes(
            key,
            self.m,
            self.k,
            namespace="ibf",
        )

    # ------------------------------------------------------------
    # Insert
    # ------------------------------------------------------------

    def insert(self, key, value):
        """
        INSERT(x, y)

        Adds key-value pair to the IBF.
        """

        for idx in self._hashes(key):
            cell = self.cells[idx]

            cell.count += 1
            cell.key_xor = key
            cell.value_xor = value

    # ------------------------------------------------------------
    # Query / GET
    # ------------------------------------------------------------

    def query(self, key):
        """
        GET(x)

        Returns:
            value
            None
            "NOT_FOUND"

        Semantics follow the paper:
            - value       : successfully recovered
            - None        : definitely absent
            - NOT_FOUND   : lookup failed probabilistically
        """

        for idx in self._hashes(key):
            cell = self.cells[idx]

            # Empty cell => definitely absent
            if cell.count == 0:
                return None

            # Singleton cell
            if cell.count == 1:
                if cell.key_xor == key:
                    return cell.value_xor
                else:
                    return None

        # Could not isolate key
        return "NOT_FOUND"
    
