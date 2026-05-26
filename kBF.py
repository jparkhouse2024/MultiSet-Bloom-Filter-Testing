"""
kBF: approximate key-value Bloom filter using XOR-superposed value encodings.

"""

import itertools
from functools import reduce
from operator import xor

from BloomFilter import stable_hashes


class KBFSequence:
    """
    Value encoding manager for kBF.

    Encodings are chosen so individual codes and pairwise XORs are distinct.
    The decoder can then reason about collided cells in the same broad shape as
    BhBF's Bh-sequence decoder.
    """

    def __init__(self):
        self.value_to_encoding = {}
        self.encoding_to_value = {}
        self.used_encodings = set()
        self.used_xors = set()
        self.next_encoding_candidate = 1

    def encode(self, value):
        if value in self.value_to_encoding:
            return self.value_to_encoding[value]

        while True:
            candidate = self.next_encoding_candidate
            self.next_encoding_candidate += 1

            if candidate in self.used_encodings:
                continue

            ok = True
            new_xors = []

            for encoding in self.used_encodings:
                pair_xor = candidate ^ encoding

                if (
                    pair_xor in self.used_xors
                    or pair_xor in self.used_encodings
                ):
                    ok = False
                    break

                new_xors.append(pair_xor)

            if not ok:
                continue

            self.used_encodings.add(candidate)
            self.used_xors.update(new_xors)
            self.value_to_encoding[value] = candidate
            self.encoding_to_value[candidate] = value

            return candidate

    def has_value(self, value):
        return value in self.value_to_encoding

    def decode_value(self, encoding):
        return self.encoding_to_value[encoding]


class KBFDecoder:
    """
    Decodes XOR-superposed cell values up to h colliding encodings.
    """

    def __init__(self, sequence, h):
        self.sequence = sequence
        self.h = h

    def decode(self, count, xor_value):
        if count == 0:
            return []

        if count > self.h:
            return None

        if count == 1:
            if xor_value in self.sequence.encoding_to_value:
                return [xor_value]
            return None

        matches = []
        encodings = list(self.sequence.used_encodings)

        for combo in itertools.combinations(encodings, count):
            combo_xor = reduce(xor, combo, 0)

            if combo_xor == xor_value:
                matches.append(combo)

                if len(matches) > 1:
                    return None

        if not matches:
            return None

        return list(matches[0])

    def is_valid_remainder(self, count, xor_value):
        return self.decode(count, xor_value) is not None


class KBFCell:
    __slots__ = ("counter", "value")

    def __init__(self):
        self.counter = 0
        self.value = 0


class KBF:
    """
    Approximate key-value Bloom Filter (kBF).

    Supports:
        - insert
        - query
        - update
        - delete
    """

    def __init__(self, m=10000, k=3, h=3):
        self.m = m
        self.k = k
        self.h = h

        self.sequence = KBFSequence()
        self.decoder = KBFDecoder(self.sequence, h)
        self.cells = [KBFCell() for _ in range(m)]

    # ------------------------------------------------------------
    # Hashing
    # ------------------------------------------------------------

    def _hashes(self, key):
        return stable_hashes(key, self.m, self.k, namespace="kbf")

    # ------------------------------------------------------------
    # Insert
    # ------------------------------------------------------------

    def insert(self, key, value):
        encoded_value = self.sequence.encode(value)

        for idx in self._hashes(key):
            cell = self.cells[idx]
            cell.counter += 1
            cell.value ^= encoded_value

    # ------------------------------------------------------------
    # Delete
    # ------------------------------------------------------------

    def delete(self, key, value):
        if not self.sequence.has_value(value):
            return False

        indices = list(self._hashes(key))

        if any(self.cells[idx].counter == 0 for idx in indices):
            return False

        encoded_value = self.sequence.encode(value)

        for idx in indices:
            cell = self.cells[idx]
            cell.counter -= 1
            cell.value ^= encoded_value

        return True

    # ------------------------------------------------------------
    # Update
    # ------------------------------------------------------------

    def update(self, key, new_value):
        old_value = self.query(key)

        if old_value is None or old_value == "UNKNOWN":
            return False

        self.delete(key, old_value)
        self.insert(key, new_value)

        return True

    # ------------------------------------------------------------
    # Decode Cell
    # ------------------------------------------------------------

    def _decode_cell(self, cell):
        return self.decoder.decode(cell.counter, cell.value)

    # ------------------------------------------------------------
    # Query
    # ------------------------------------------------------------

    def query(self, key):
        """
        Returns:
            value
            None
            "UNKNOWN"
        """
        indices = list(self._hashes(key))
        queried_cells = [self.cells[i] for i in indices]
        queried_cells.sort(key=lambda c: c.counter)

        common_encodings = None
        invalid_cells = []

        for cell in queried_cells:
            if cell.counter == 0:
                return None

            decoded = self._decode_cell(cell)

            if decoded is None:
                invalid_cells.append(cell)
                continue

            decoded_set = set(decoded)

            if common_encodings is None:
                common_encodings = decoded_set
            else:
                common_encodings &= decoded_set

            if len(common_encodings) == 0:
                return None

        if common_encodings is None:
            return "UNKNOWN"

        if len(common_encodings) == 1:
            encoding = next(iter(common_encodings))
            return self.sequence.decode_value(encoding)

        candidates = []

        for candidate in common_encodings:
            valid_candidate = True

            for cell in invalid_cells:
                remaining_count = cell.counter - 1
                remaining_value = cell.value ^ candidate

                if remaining_count == 0:
                    if remaining_value != 0:
                        valid_candidate = False
                        break

                    continue

                if (
                    remaining_count <= self.h
                    and not self.decoder.is_valid_remainder(
                        remaining_count,
                        remaining_value,
                    )
                ):
                    valid_candidate = False
                    break

            if valid_candidate:
                candidates.append(candidate)

        if len(candidates) == 1:
            return self.sequence.decode_value(candidates[0])

        return "UNKNOWN"


# ------------------------------------------------------------
# Example usage
# ------------------------------------------------------------

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
