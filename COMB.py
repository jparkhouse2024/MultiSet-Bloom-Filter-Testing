"""
COMB: Combinatorial Bloom Filter
Reference implementation based on:

"Fast Dynamic Multiple-Set Membership Testing
Using Combinatorial Bloom Filters"

This implementation follows the same style and structure as the
KBF implementation.

Features:
    - multiple-set membership testing
    - deterministic decoding (no misclassification)
    - constant-weight group encodings
    - insert
    - query

Terminology:
    group_id == set identifier

Return values for query():
    group_id
    None          -> definitely not present
    "FAILURE"     -> classification failure
"""

import itertools
from math import comb


from BloomFilter import stable_hashes


# ============================================================
# Constant Weight Code Generator
# ============================================================

class COMBCodebook:
    """
    Assigns each group id a unique constant-weight binary code.

    Codes are represented as tuples of hash-set indices.
    Example:
        weight=3
        code=(0,2,5)

    means:
        hash sets 0, 2, and 5 are active.
    """

    def __init__(self, num_hash_sets, weight):
        self.l = num_hash_sets
        self.w = weight

        self.group_to_code = {}
        self.code_to_group = {}

        self.available_codes = itertools.combinations(
            range(num_hash_sets),
            weight,
        )

    def assign_group(self, group_id):
        if group_id in self.group_to_code:
            return self.group_to_code[group_id]

        try:
            code = next(self.available_codes)
        except StopIteration:
            raise ValueError("No more available COMB codes")

        self.group_to_code[group_id] = code
        self.code_to_group[code] = group_id

        return code

    def get_code(self, group_id):
        return self.group_to_code[group_id]

    def decode_code(self, code):
        return self.code_to_group.get(tuple(sorted(code)))

    @property
    def capacity(self):
        return comb(self.l, self.w)


# ============================================================
# COMB Cell
# ============================================================

class COMBCell:
    __slots__ = ("bit",)

    def __init__(self):
        self.bit = 0


# ============================================================
# COMB
# ============================================================

class COMB:
    """
    Combinatorial Bloom Filter.

    Parameters
    ----------
    m : int
        Number of bits in Bloom filter.

    l : int
        Number of hash sets.

    k : int
        Number of hashes per hash set.

    w : int
        Constant weight of group encodings.

    Notes
    -----
    - Total number of hash functions = l * k
    - Insert cost = w * k hashes
    - Query cost = l * k hashes
    """

    def __init__(self, m=100000, l=8, k=4, w=3):
        if w > l:
            raise ValueError("w must be <= l")

        self.m = m
        self.l = l
        self.k = k
        self.w = w

        self.array = bytearray(m)   # or bytearray(m)

        self.codebook = COMBCodebook(l, w)

    # ========================================================
    # Hashing
    # ========================================================

    def _hashes_for_set(self, key, hash_set_id):
        """
        Returns the k hash positions for one hash set.
        """

        namespace = f"comb_{hash_set_id}"

        return stable_hashes(
            key,
            self.m,
            self.k,
            namespace=namespace,
        )

    # ========================================================
    # Insert
    # ========================================================

    def insert(self, key, group_id):
        """
        Insert key into a group.
        """

        code = self.codebook.assign_group(group_id)

        for hash_set_id in code:
            for idx in self._hashes_for_set(key, hash_set_id):
                self.array[idx] = 1

    # ========================================================
    # Query Hash Set
    # ========================================================

    def _query_hash_set(self, key, hash_set_id):
        """
        Returns True if all bits in the hash set are 1.
        """

        for idx in self._hashes_for_set(key, hash_set_id):
            if self.array[idx] == 0:
                return False

        return True

    # ========================================================
    # Query
    # ========================================================

    def query(self, key):
        """
        Returns:
            group_id
            None
            "FAILURE"

        Logic:
            - If fewer than w hash sets match:
                  definitely absent

            - If exactly w hash sets match:
                  decode directly

            - If more than w hash sets match:
                  classification failure
        """

        active_sets = []

        for hash_set_id in range(self.l):
            if self._query_hash_set(key, hash_set_id):
                active_sets.append(hash_set_id)

        num_active = len(active_sets)

        # definitely absent
        if num_active < self.w:
            return None

        # exact code match
        if num_active == self.w:
            return self.codebook.decode_code(tuple(active_sets))

        # too many positives
        return "FAILURE"

    

if __name__ == "__main__":

    # ========================================================
    # Create COMB
    # ========================================================

    comb_filter = COMB(
        m=50000,   # bloom filter size
        l=8,       # number of hash sets
        k=4,       # hashes per hash set
        w=3,       # constant-weight code size
    )


    # ========================================================
    # Insert elements
    # ========================================================


    animals = {
        "lion": "mammal",
        "tiger": "mammal",
        "eagle": "bird",
        "penguin": "bird",
        "cobra": "reptile",
        "alligator": "reptile",
    }

    for animal, group in animals.items():
        comb_filter.insert(animal, group)

    # ========================================================
    # Query inserted elements
    # ========================================================

    print("\n=== Inserted Elements ===")

    for animal in animals:
        result = comb_filter.query(animal)
        print(f"{animal:12} -> {result}")

    # ========================================================
    # Query missing elements
    # ========================================================

    print("\n=== Missing Elements ===")

    missing = [
        "wolf",
        "shark",
        "falcon",
        "dragon",
    ]

    for animal in missing:
        result = comb_filter.query(animal)
        print(f"{animal:12} -> {result}")

   