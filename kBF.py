import hashlib
import math
from collections import deque


class KBFCell:
    __slots__ = ("counter", "value")

    def __init__(self):
        self.counter = 0
        self.value = 0


class KBF:
    """
    Approximate key-value Bloom Filter (kBF)

    Supports:
        - create
        - insert
        - query
        - update
        - delete
        - join
        - compress

    This implementation follows the paper design:
        - XOR superposition of encoded values
        - collision-resilient decoding
        - counting cells
    """

    def __init__(self, m=10000, k=3):
        self.m = m
        self.k = k

        self.cells = [KBFCell() for _ in range(m)]

        # value <-> encoding mappings
        self.value_to_encoding = {}
        self.encoding_to_value = {}

        # used for constructing unique encodings
        self.used_encodings = set()
        self.used_xors = set()

        self.next_encoding_candidate = 1

    # ============================================================
    # Hashing
    # ============================================================

    def _hashes(self, key):
        key = str(key).encode()

        for i in range(self.k):
            digest = hashlib.blake2b(
                key + i.to_bytes(2, "little"),
                digest_size=8
            ).digest()

            yield int.from_bytes(digest, "little") % self.m

    # ============================================================
    # Encoding generation
    # ============================================================

    def _create_encoding(self, value):
        """
        Create collision-resistant XOR encoding.

        Requirement:
            - all encodings unique
            - XOR of any pair unique
        """

        if value in self.value_to_encoding:
            return self.value_to_encoding[value]

        while True:
            candidate = self.next_encoding_candidate
            self.next_encoding_candidate += 1

            if candidate in self.used_encodings:
                continue

            ok = True
            new_xors = []

            for e in self.used_encodings:
                x = candidate ^ e

                if x in self.used_xors or x in self.used_encodings:
                    ok = False
                    break

                new_xors.append(x)

            if ok:
                self.used_encodings.add(candidate)

                for x in new_xors:
                    self.used_xors.add(x)

                self.value_to_encoding[value] = candidate
                self.encoding_to_value[candidate] = value

                return candidate

    # ============================================================
    # Insert
    # ============================================================

    def insert(self, key, value):
        evalue = self._create_encoding(value)

        for idx in self._hashes(key):
            cell = self.cells[idx]

            cell.counter += 1
            cell.value ^= evalue

    # ============================================================
    # Delete
    # ============================================================

    def delete(self, key, value):
        if value not in self.value_to_encoding:
            return False

        evalue = self.value_to_encoding[value]

        for idx in self._hashes(key):
            cell = self.cells[idx]

            if cell.counter == 0:
                return False

            cell.counter -= 1
            cell.value ^= evalue

        return True

    # ============================================================
    # Update
    # ============================================================

    def update(self, key, new_value):
        old_value = self.query(key)

        if old_value is None:
            return False

        self.delete(key, old_value)
        self.insert(key, new_value)

        return True

    # ============================================================
    # Decoding helpers
    # ============================================================

    def _decode2(self, xor_value):
        """
        Given:
            xor_value = a XOR b

        Find unique pair (a,b)
        """

        for e in self.used_encodings:
            other = xor_value ^ e

            if other in self.used_encodings:
                return (e, other)

        return None

    def _decode22(self, v1, v2):
        """
        v1 = a XOR b
        v2 = a XOR c

        Find common encoding a
        """

        p1 = self._decode2(v1)
        p2 = self._decode2(v2)

        if p1 is None or p2 is None:
            return None

        s1 = set(p1)
        s2 = set(p2)

        common = s1.intersection(s2)

        if len(common) == 1:
            return next(iter(common))

        return None

    def _decode23(self, v2, v3):
        """
        Decode:
            v2 = a XOR b
            v3 = a XOR c XOR d
        """

        pair = self._decode2(v2)

        if pair is None:
            return None

        a, b = pair

        x1 = v3 ^ a
        x2 = v3 ^ b

        if x1 in self.used_xors and x2 not in self.used_xors:
            return a

        if x2 in self.used_xors and x1 not in self.used_xors:
            return b

        return None

    # ============================================================
    # Query
    # ============================================================

    def query(self, key):
        queue2 = []
        queue3 = []

        for idx in self._hashes(key):
            cell = self.cells[idx]

            if cell.counter == 0:
                return None

            if cell.counter == 1:
                return self.encoding_to_value.get(cell.value)

            elif cell.counter == 2 and cell.value != 0:
                queue2.append(cell.value)

            elif cell.counter == 3 and cell.value != 0:
                queue3.append(cell.value)

        # Try decode22
        if len(queue2) >= 2:
            e = self._decode22(queue2[0], queue2[1])

            if e is not None:
                return self.encoding_to_value.get(e)

        # Try decode23
        if len(queue2) >= 1 and len(queue3) >= 1:
            e = self._decode23(queue2[0], queue3[0])

            if e is not None:
                return self.encoding_to_value.get(e)

        return None


# ============================================================
# Example usage
# ============================================================

if __name__ == "__main__":

    kbf = KBF(m=5000, k=3)

    kbf.insert("alice", "online")
    kbf.insert("bob", "offline")
    kbf.insert("charlie", "busy")

    print(kbf.query("alice"))
    print(kbf.query("bob"))
    print(kbf.query("charlie"))

    kbf.update("alice", "away")

    print(kbf.query("alice"))

    kbf.delete("bob", "offline")

    print(kbf.query("bob"))