"""
EC-BF (Error Correcting Code Bloom Filter)
Reference implementation based on:

"Amplifying Multi-Set Bloom Filter Classification Accuracy
Using Error-Correcting Codes"

This implementation focuses on clarity rather than optimization.

Features:
- Multiple sets
- Arbitrary binary set encodings
- Prefix-tree decoding
- Optional repeated hash amplification (K parameter)
- No false negatives
- No misclassification:
    query(x) either returns:
        (False, None)
        (True, correct_set)
        (True, None)   # Failure-to-Classify

Author: ChatGPT
"""

import hashlib
from collections import defaultdict
from hamming_Codes import hammoing_codes
from BloomFilter import BloomFilter


# ============================================================
# Utility Hash Function
# ============================================================

def deterministic_hash(value: str, seed: str, m: int) -> int:
    """
    Deterministic hash into range [0, m).
    """
    h = hashlib.sha256((seed + value).encode()).hexdigest()
    return int(h, 16) % m



# ============================================================
# Compact Array Trie
# ============================================================

class ArrayPrefixTree:
    """
    Compact binary trie using array representation.

    Node indexing:
        left  child = 2*i + 1
        right child = 2*i + 2
    """

    def __init__(self, height):

        self.height = height

        # Total nodes in complete binary tree
        self.size = (1 << (height + 1)) - 1

        # Whether node exists
        self.exists = bytearray(self.size)

        # Leaf payload
        # -1 means not a leaf
        self.leaf_code = [-1] * self.size

        # Root exists
        self.exists[0] = 1

    def insert(self, code):

        node = 0

        for bit in code:

            if bit == "0":
                node = 2 * node + 1
            else:
                node = 2 * node + 2

            self.exists[node] = 1

        self.leaf_code[node] = code

# ============================================================
# Prefix Tree
# ============================================================

# class TrieNode:
#     def __init__(self):
#         self.children = {}
#         self.is_leaf = False
#         self.code = None


# class PrefixTree:
#     def __init__(self):
#         self.root = TrieNode()

#     def insert(self, code: str):
#         node = self.root

#         for bit in code:
#             if bit not in node.children:
#                 node.children[bit] = TrieNode()

#             node = node.children[bit]

#         node.is_leaf = True
#         node.code = code


# ============================================================
# EC-BF
# ============================================================

class ECBloomFilter:
    """
    EC-BF with repeated hash amplification.

    Parameters:
    -----------
    m : int
        Size of bit array.

    code_map : dict[int -> str]
        Maps set IDs to binary codewords.

    K : int
        Number of repeated hash functions per bit.
    """

    def __init__(self, L, m=200000, K=3):

        self.m = m
        self.K = K

        # Bit array
        self.A = bytearray(m)

        # Set ID -> code
        r = hammoing_codes.choose_r(L)
        codes = hammoing_codes.build_hamming_distance_3_codes(L, r)
        self.c = codes

        # Code -> set ID
        self.c_inv = {v: k for k, v in codes.items()}

        # Code length
        self.k_prime = len(next(iter(codes.values())))

        # Prefix tree
        self.PT = ArrayPrefixTree(self.k_prime)

        for code in codes.values():
            self.PT.insert(code)


    # ========================================================
    # Internal Hash
    # ========================================================

    def _hash(self, x, i, bit, j):
        """
        H_i,j^bit(x)
        """
        seed = f"{i}-{bit}-{j}"
        return deterministic_hash(str(x), seed, self.m)

    # ========================================================
    # Insert
    # ========================================================

    def insert(self, x, set_id):
        """
        Insert element x from set set_id.
        """

        code = self.c[set_id]

        for i, bit in enumerate(code):

            for j in range(self.K):

                idx = self._hash(x, i, bit, j)

                self.A[idx] = 1

    # ========================================================
    # Query
    # ========================================================

    def query(self, x):

        current_nodes = [0]

        for i in range(self.k_prime):

            # ------------------------------------
            # Evaluate B[0] and B[1]
            # ------------------------------------

            B = {}

            for bit in ["0", "1"]:

                ok = True

                for j in range(self.K):

                    idx = self._hash(x, i, bit, j)

                    if self.A[idx] == 0:
                        ok = False
                        break

                B[bit] = ok

            # ------------------------------------
            # Expand tree nodes
            # ------------------------------------

            next_nodes = []

            for node in current_nodes:

                # bit = 0
                if B["0"]:

                    child = 2 * node + 1

                    if self.PT.exists[child]:
                        next_nodes.append(child)

                # bit = 1
                if B["1"]:

                    child = 2 * node + 2

                    if self.PT.exists[child]:
                        next_nodes.append(child)

            current_nodes = next_nodes

            # Early termination
            if not current_nodes:
                return (False, None)

        # ----------------------------------------
        # Extract candidates
        # ----------------------------------------

        candidates = []

        for node in current_nodes:

            code = self.PT.leaf_code[node]

            if code != -1:
                candidates.append(code)

        # No candidates
        if len(candidates) == 0:
            return (False, None)

        # Unique candidate
        if len(candidates) == 1:

            code = candidates[0]

            return (True, self.c_inv[code])

        # Multiple candidates
        return (True, None)


# ============================================================
# Example Usage
# ============================================================

if __name__ == "__main__":

    """
    Example:

    4 sets with Hamming-distance-2 codewords.
    """

    bf = ECBloomFilter(
        m=1000,
        L=4,
        K=2,
    )

    # --------------------------------------------------------
    # Insert Elements
    # --------------------------------------------------------

    set0 = ["apple", "banana", "orange"]
    set1 = ["dog", "cat", "mouse"]
    set2 = ["red", "blue", "green"]
    set3 = ["car", "train", "plane"]

    for x in set0:
        bf.insert(x, 0)

    for x in set1:
        bf.insert(x, 1)

    for x in set2:
        bf.insert(x, 2)

    for x in set3:
        bf.insert(x, 3)
    
    print(bf.A)

    # --------------------------------------------------------
    # Queries
    # --------------------------------------------------------

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
        "not_present3",
        "not_present4",
        "not_present5",
        "not_present6",
        "not_present7",
        "not_present8",
        "not_present9",
        "not_present10"
    ]

    print()

    for x in tests:

        result = bf.query(x)

        print(f"{x:15} -> {result}")
