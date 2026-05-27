"""
EC-BF (Error Correcting Code Bloom Filter)

This implementation keeps the defining EC-BF idea: each set is assigned a
Hamming-distance codeword, and an element is classified by recovering the
codeword bits that appear present.

Example:
    bf = ECBloomFilter(L=4, m=1000, K=2)
    bf.insert("alice", 0)
    bf.query("alice")     # (True, 0)
    bf.query("mallory")   # (False, None)

Benchmark bits:
    EC-BF stores ordinary Bloom-filter bit arrays. In the benchmark, "bits per
    element-set pair" is the approximate Python memory footprint of those
    filters and codebook metadata divided by (inserted elements * number of
    sets). It is a normalized space cost, not a literal bit per pair.
"""

from BloomFilter import BloomFilter
from hamming_Codes import HammingCodes


class ECBloomFilter:
    """
    Compact EC-BF using one Bloom filter for every (code position, bit value).

    The previous trie-shaped implementation used many edge filters plus leaf
    filters. This representation keeps the same Hamming-code feature with only
    2 * code_length filters. Querying forms the possible codeword bits and then
    intersects them with the Hamming codebook.

    Parameters
    ----------
    L : int
        Number of sets.
    m : int
        Number of bits in each per-position/per-bit Bloom filter.
    K : int
        Number of hash functions in each Bloom filter.
    """

    def __init__(self, L, m=200000, K=3):
        self.L = L
        self.m = m
        self.K = K

        self.c = HammingCodes.build_hamming_distance_3_codes(L)
        self.c_inv = {code: set_id for set_id, code in self.c.items()}
        self.k_prime = len(next(iter(self.c.values())))

        self.position_filters = [
            (
                BloomFilter(
                    self.m,
                    self.K,
                    namespace=f"ecbf:{idx}:0",
                ),
                BloomFilter(
                    self.m,
                    self.K,
                    namespace=f"ecbf:{idx}:1",
                ),
            )
            for idx in range(self.k_prime)
        ]

        # Valid code prefixes let query prune impossible paths without storing
        # a full trie of Bloom filters.
        self.prefix_codes = {""}

        for code in self.c.values():
            for idx in range(1, len(code) + 1):
                self.prefix_codes.add(code[:idx])

        # Compatibility handle for existing benchmark/debug code.
        self.PT = self.prefix_codes

    def iter_position_filters(self, position):
        zero_filter, one_filter = self.position_filters[position]
        return (("0", zero_filter), ("1", one_filter))

    def _virtual_key(self, position: int, bit: str, key) -> str:
        return f"{position}:{bit}:{key}"

    def insert(self, x, set_id):
        if set_id not in self.c:
            raise ValueError("invalid set id")

        code = self.c[set_id]

        for idx, bit in enumerate(code):
            vkey = self._virtual_key(idx, bit, x)
            filter_idx = 1 if bit == "1" else 0
            self.position_filters[idx][filter_idx].add(vkey)

    def query(self, x):
        """
        Returns:
            (False, None)  absent
            (True, set_id) unique classification
            (True, None)   classification failure
        """
        prefixes = [""]

        for idx in range(self.k_prime):
            possible_bits = []

            for bit, bf in self.iter_position_filters(idx):
                vkey = self._virtual_key(idx, bit, x)

                if bf.contains(vkey):
                    possible_bits.append(bit)

            if not possible_bits:
                return (False, None)

            next_prefixes = []

            for prefix in prefixes:
                for bit in possible_bits:
                    candidate_prefix = prefix + bit

                    if candidate_prefix in self.prefix_codes:
                        next_prefixes.append(candidate_prefix)

            if not next_prefixes:
                return (False, None)

            prefixes = next_prefixes

        candidates = [
            prefix
            for prefix in prefixes
            if prefix in self.c_inv
        ]

        if len(candidates) == 0:
            return (False, None)

        if len(candidates) == 1:
            return (True, self.c_inv[candidates[0]])

        return (True, None)


# ------------------------------------------------------------
# Example Usage
# ------------------------------------------------------------

if __name__ == "__main__":
    bf = ECBloomFilter(
        m=1000,
        L=4,
        K=2,
    )

    data = [
        (["apple", "banana", "orange"], 0),
        (["dog", "cat", "mouse"], 1),
        (["red", "blue", "green"], 2),
        (["car", "train", "plane"], 3),
    ]

    for items, set_id in data:
        for x in items:
            bf.insert(x, set_id)

    tests = [
        "apple",
        "banana",
        "orange",
        "dog",
        "cat",
        "mouse",
        "red",
        "blue",
        "green",
        "plane",
        "not_present1",
        "not_present2",
    ]

    for x in tests:
        print(f"{x:15} -> {bf.query(x)}")
