import hashlib
import math
from collections import deque
from BloomFilter import BloomFilter


class BloomNode:
    """
    Node in the BT tree.
    """

    def __init__(self, depth, path_id="", is_leaf=False):
        self.depth = depth
        self.path_id = path_id
        self.is_leaf = is_leaf

        self.edge_filters = []
        self.children = []

        self.group_id = None
        self.leaf_filter = None


class Bloom_Tree:
    """
    Bloom Tree implementation.
    """

    def __init__(
        self,
        num_groups: int,
        d: int = 2,
        bits_per_filter: int =  4000,
        k_internal: int = None,
        k_leaf: int =7,
    ):
        self.g = num_groups
        self.d = d

        self.height = math.ceil(math.log(num_groups, d))

        if k_internal is None:
            k_internal = max(1, round(math.log2(d)))

        self.bits_per_filter = bits_per_filter
        self.k_internal = k_internal
        self.k_leaf = k_leaf

        self.root = self._build_tree()

    # ------------------------------------------------------------
    # Tree Construction
    # ------------------------------------------------------------

    def _build_tree(self):
        root = BloomNode(depth=0, path_id="")

        leaf_counter = [0]

        def build(node, depth):
            if depth == self.height:
                node.is_leaf = True

                if leaf_counter[0] < self.g:
                    node.group_id = leaf_counter[0]
                    leaf_counter[0] += 1

                node.leaf_filter = BloomFilter(
                    self.bits_per_filter,
                    self.k_leaf,
                )

                return

            for edge_idx in range(self.d):

                bf = BloomFilter(
                    self.bits_per_filter,
                    self.k_internal,
                )

                node.edge_filters.append(bf)

                child = BloomNode(
                    depth=depth + 1,
                    path_id=node.path_id + str(edge_idx),
                )

                node.children.append(child)

                build(child, depth + 1)

        build(root, 0)

        return root

    # ------------------------------------------------------------
    # Virtual Key
    # ------------------------------------------------------------

    def _virtual_key(self, path_id: str, key: str) -> bytes:
        return f"{path_id}:{key}".encode()

    # ------------------------------------------------------------
    # Convert Group ID -> Tree Path
    # ------------------------------------------------------------

    def _group_to_path(self, group_id: int):

        digits = []

        x = group_id

        for _ in range(self.height):
            digits.append(x % self.d)
            x //= self.d

        digits.reverse()

        while len(digits) < self.height:
            digits.insert(0, 0)

        return digits

    # ------------------------------------------------------------
    # Insert
    # ------------------------------------------------------------

    def insert(self, key: str, group_id: int):

        if not (0 <= group_id < self.g):
            raise ValueError("invalid group id")

        path = self._group_to_path(group_id)

        node = self.root

        for edge in path:

            vkey = self._virtual_key(node.path_id, key)

            node.edge_filters[edge].add(vkey)

            node = node.children[edge]

        leaf_key = self._virtual_key(node.path_id, key)

        node.leaf_filter.add(leaf_key)

    # ------------------------------------------------------------
    # Query
    # ------------------------------------------------------------

    def query(self, key: str):
        """
        Returns:
            group_id
            None
            "CLASSIFICATION_FAILURE"
        """

        matches = []

        def dfs(node):

            if node.is_leaf:

                if node.group_id is None:
                    return

                vkey = self._virtual_key(node.path_id, key)

                if node.leaf_filter.contains(vkey):
                    matches.append(node.group_id)

                return

            vkey = self._virtual_key(node.path_id, key)

            for edge_idx, bf in enumerate(node.edge_filters):

                if bf.contains(vkey):
                    dfs(node.children[edge_idx])

        dfs(self.root)

        if len(matches) == 0:
            return None

        if len(matches) == 1:
            return matches[0]

        return "CLASSIFICATION_FAILURE"


# ------------------------------------------------------------
# Example Usage
# ------------------------------------------------------------

if __name__ == "__main__":

    BT = Bloom_Tree(
        num_groups=8,
        d=2,
        bits_per_filter=20,
        k_internal=2,
        k_leaf=2,
    )

    data = [
        ("alice", 0),
        ("bob", 1),
        ("charlie", 2),
        ("david", 3),
        ("eve", 4),
    ]

    for key, group in data:
        BT.insert(key, group)

    print(BT.query("alice"))
    print(BT.query("bob"))
    print(BT.query("charlie"))

    print(BT.query("mallory"))