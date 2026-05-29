"""
BhBF: multi-set membership using B_h sequence sums.

Example:
    bhbf = BhBF(num_sets=4, m=1000, k=3, h=3)
    bhbf.insert("alice", 0)
    bhbf.query("alice")    # 0
    bhbf.query("mallory")  # None

Benchmark bits:
    BhBF does not store simple Bloom bits. Each hashed cell stores a count and
    a sum of B_h set codes. The benchmark still reports "bits per element-set
    pair" as total approximate memory for all cells, sequence tables, and
    metadata divided by (inserted elements * number of sets). This lets the
    counted integer representation be compared against bit-array filters.
"""

import itertools

from BloomFilter import stable_hashes


def generate_bh_sequence(num_codes, h=3, method="powers"):
    """
    Construct a B_h sequence.

    A B_h sequence requires every sum of h elements, with repetitions
    allowed, to be unique.

    The default powers construction uses 1, (h + 1), (h + 1)^2, ... . Any sum
    of at most h sequence elements is then a base-(h + 1) number with digits
    in [0, h], so there are no carries and each bounded multiset sum is unique.

    method="greedy" is available for compact small sequences, but it gets slow
    for larger num_codes.
    """
    if num_codes < 0:
        raise ValueError("num_codes must be non-negative")
    if h <= 0:
        raise ValueError("h must be positive")
    if method not in {"powers", "greedy"}:
        raise ValueError("method must be 'powers' or 'greedy'")

    if method == "powers":
        base = h + 1

        return [base ** i for i in range(num_codes)]

    codes = []
    sums_by_count = {0: {0}}

    for count in range(1, h + 1):
        sums_by_count[count] = set()

    candidate = 1

    while len(codes) < num_codes:
        new_sums_by_count = {}
        valid = True

        for count in range(1, h + 1):
            new_sums = set()

            for candidate_repetitions in range(1, count + 1):
                remaining = count - candidate_repetitions

                for existing_sum in sums_by_count[remaining]:
                    total = (
                        candidate_repetitions * candidate
                        + existing_sum
                    )

                    if (
                        total in sums_by_count[count]
                        or total in new_sums
                    ):
                        valid = False
                        break

                    new_sums.add(total)

                if not valid:
                    break

            if not valid:
                break

            new_sums_by_count[count] = new_sums

        if valid:
            codes.append(candidate)

            for count, new_sums in new_sums_by_count.items():
                sums_by_count[count].update(new_sums)

        candidate += 1

    return codes


def is_bh_sequence(codes, h=3):
    """
    Verify the B_h uniqueness property for counts 1..h.
    """
    if h <= 0:
        raise ValueError("h must be positive")

    for count in range(1, h + 1):
        seen = set()

        for combo in itertools.combinations_with_replacement(
            codes,
            count,
        ):
            total = sum(combo)

            if total in seen:
                return False

            seen.add(total)

    return True


class BhSequence:
    """
    Simple Bh-sequence manager.
    """

    def __init__(self, codes):

        self.codes = codes

        self.code_to_set = {}
        self.set_to_code = {}

    def assign_set_ids(self, num_sets):

        if num_sets > len(self.codes):
            raise ValueError("Not enough Bh codes")

        for i in range(num_sets):
            self.code_to_set[self.codes[i]] = i
            self.set_to_code[i] = self.codes[i]

    def encode(self, set_id):

        return self.set_to_code[set_id]

    def decode_set(self, code):

        return self.code_to_set[code]


class BhDecoder:
    """
    Precomputes all sums up to h elements, keyed by collision count.

    Used for O(1)-style decoding.
    """

    def __init__(self, codes, h):

        self.h = h
        self.codes = codes

        # count -> sum -> tuple(codes)
        self.sum_table = {r: {} for r in range(1, h + 1)}

        for r in range(1, h + 1):

            for combo in itertools.combinations_with_replacement(
                codes,
                r,
            ):

                s = sum(combo)

                self.sum_table[r][s] = combo

    def decode(self, total_sum, count):

        return list(self.sum_table.get(count, {}).get(total_sum, []))

    def is_valid_sum(self, total_sum, count):

        return total_sum in self.sum_table.get(count, {})


class BhBFCell:

    def __init__(self):

        self.count = 0
        self.sum = 0


class BhBF:
    """
    BhBF implementation from the paper.

    Supports:
        - insert
        - query
        - delete
        - update
    """

    def __init__(
        self,
        num_sets,
        m=1024,
        k=3,
        h=3,
        bh_codes=None,
    ):

        self.num_sets = num_sets
        self.m = m
        self.k = k
        self.h = h

        if bh_codes is None:
            bh_codes = generate_bh_sequence(num_sets, h)

        if num_sets > len(bh_codes):
            raise ValueError("Not enough Bh codes")
        if not is_bh_sequence(bh_codes[:num_sets], h):
            raise ValueError("bh_codes must be a valid B_h sequence")

        self.sequence = BhSequence(bh_codes)

        self.sequence.assign_set_ids(num_sets)

        self.decoder = BhDecoder(
            bh_codes[:num_sets],
            h,
        )

        self.cells = [[0]*2 for _ in range(m)]

    # ------------------------------------------------------------
    # Hashing
    # ------------------------------------------------------------

    def _hashes(self, element):
        return stable_hashes(element, self.m, self.k, namespace="bhbf")

    # ------------------------------------------------------------
    # Insert
    # ------------------------------------------------------------

    def insert(self, element, set_id):

        code = self.sequence.encode(set_id)

        for idx in self._hashes(element):

            self.cells[idx][0] += 1
            self.cells[idx][1] += code

    # ------------------------------------------------------------
    # Delete
    # ------------------------------------------------------------

    def delete(self, element, set_id):

        code = self.sequence.encode(set_id)

        for idx in self._hashes(element):

            self.cells[idx][0]-= 1
            self.cells[idx][1] -= code

    # ------------------------------------------------------------
    # Update
    # ------------------------------------------------------------

    def update(self, element, old_set, new_set):

        old_code = self.sequence.encode(old_set)
        new_code = self.sequence.encode(new_set)

        for idx in self._hashes(element):

            self.cells[idx][0] -= old_code
            self.cells[idx][1] += new_code

    # ------------------------------------------------------------
    # Decode Cell
    # ------------------------------------------------------------

    def _decode_cell(self, cell):

        if cell[0] == 0:
            return []

        if cell[0] <= self.h:

            return self.decoder.decode(cell[1], cell[0])

        return None

    # ------------------------------------------------------------
    # Query
    # ------------------------------------------------------------

    def query(self, element):
        """
        Returns:
            set_id
            None
            "UNKNOWN"
        """

        indices = list(self._hashes(element))

        queried_cells = [self.cells[i] for i in indices]

        # Strategy 1:
        # sort by count ascending

        queried_cells.sort(key=lambda c: c[0])

        common_codes = None

        invalid_cells = []

        # --------------------------------------------------------
        # Process valid cells
        # --------------------------------------------------------

        for cell in queried_cells:

            if cell[0] == 0:
                return None

            decoded = self._decode_cell(cell)

            # invalid cell
            if decoded is None:

                invalid_cells.append(cell)
                continue

            decoded_set = set(decoded)

            if common_codes is None:
                common_codes = decoded_set
            else:
                common_codes &= decoded_set

            if len(common_codes) == 0:
                return None

        if common_codes is None:
            return "UNKNOWN"

        # --------------------------------------------------------
        # Unique answer
        # --------------------------------------------------------

        if len(common_codes) == 1:

            code = next(iter(common_codes))

            return self.sequence.decode_set(code)

        # --------------------------------------------------------
        # Strategy 2:
        # Try resolving UNKNOWN with invalid cells
        # --------------------------------------------------------

        candidates = []

        for candidate in common_codes:

            valid_candidate = True

            for cell in invalid_cells:

                # Only process h+1 cells
                if cell[0] != self.h + 1:
                    continue

                remaining = cell[1] - candidate
                remaining_count = cell[0] - 1

                if not self.decoder.is_valid_sum(
                    remaining,
                    remaining_count,
                ):

                    valid_candidate = False
                    break

            if valid_candidate:
                candidates.append(candidate)

        if len(candidates) == 1:

            return self.sequence.decode_set(candidates[0])

        return "UNKNOWN"



# ------------------------------------------------------------
# Example Usage
# ------------------------------------------------------------

if __name__ == "__main__":

    bhbf = BhBF(
        num_sets=4,
        m=32,
        k=3,
        h=3,
    )

    # --------------------------------------------------------
    # Insert
    # --------------------------------------------------------

    bhbf.insert("e1", 0)
    bhbf.insert("e2", 1)
    bhbf.insert("e3", 2)
    bhbf.insert("e4", 3)
    bhbf.insert("e5", 2)

    # --------------------------------------------------------
    # Query
    # --------------------------------------------------------

    print("e1 ->", bhbf.query("e1"))
    print("e2 ->", bhbf.query("e2"))
    print("e3 ->", bhbf.query("e3"))
    print("e5 ->", bhbf.query("e5"))

    print("unknown ->", bhbf.query("unknown"))

    # --------------------------------------------------------
    # Update
    # --------------------------------------------------------

    bhbf.update("e1", 0, 2)

    print("e1 after update ->", bhbf.query("e1"))

    # --------------------------------------------------------
    # Delete
    # --------------------------------------------------------

    bhbf.delete("e2", 1)

    print("e2 after delete ->", bhbf.query("e2"))
