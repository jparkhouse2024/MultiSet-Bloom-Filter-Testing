import hashlib
import itertools


class BhSequence:
    """
    Simple Bh-sequence manager.

    This implementation uses a manually supplied Bh sequence.
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
    Precomputes all sums up to h elements.

    Used for O(1)-style decoding.
    """

    def __init__(self, codes, h):

        self.h = h
        self.codes = codes

        # sum -> tuple(codes)
        self.sum_table = {}

        for r in range(1, h + 1):

            for combo in itertools.combinations_with_replacement(
                codes,
                r,
            ):

                s = sum(combo)

                self.sum_table[s] = combo

    def decode(self, total_sum):

        return list(self.sum_table.get(total_sum, []))

    def is_valid_sum(self, total_sum):

        return total_sum in self.sum_table


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

            # Example B3 sequence from the paper
            bh_codes = [
                1,
                22,
                55,
                72,
                130,
                200,
                350,
                500,
            ]

        if num_sets > len(bh_codes):
            raise ValueError("Not enough Bh codes")

        self.sequence = BhSequence(bh_codes)

        self.sequence.assign_set_ids(num_sets)

        self.decoder = BhDecoder(
            bh_codes[:num_sets],
            h,
        )

        self.cells = [BhBFCell() for _ in range(m)]

    # ------------------------------------------------------------
    # Hashing
    # ------------------------------------------------------------

    def _hashes(self, element):

        data = str(element).encode()

        h1 = int(
            hashlib.blake2b(
                data,
                digest_size=16,
            ).hexdigest(),
            16,
        )

        h2 = int(
            hashlib.sha256(data).hexdigest(),
            16,
        )

        for i in range(self.k):

            yield (h1 + i * h2) % self.m

    # ------------------------------------------------------------
    # Insert
    # ------------------------------------------------------------

    def insert(self, element, set_id):

        code = self.sequence.encode(set_id)

        for idx in self._hashes(element):

            self.cells[idx].count += 1
            self.cells[idx].sum += code

    # ------------------------------------------------------------
    # Delete
    # ------------------------------------------------------------

    def delete(self, element, set_id):

        code = self.sequence.encode(set_id)

        for idx in self._hashes(element):

            self.cells[idx].count -= 1
            self.cells[idx].sum -= code

    # ------------------------------------------------------------
    # Update
    # ------------------------------------------------------------

    def update(self, element, old_set, new_set):

        old_code = self.sequence.encode(old_set)
        new_code = self.sequence.encode(new_set)

        for idx in self._hashes(element):

            self.cells[idx].sum -= old_code
            self.cells[idx].sum += new_code

    # ------------------------------------------------------------
    # Decode Cell
    # ------------------------------------------------------------

    def _decode_cell(self, cell):

        if cell.count == 0:
            return []

        if cell.count <= self.h:

            return self.decoder.decode(cell.sum)

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

        queried_cells.sort(key=lambda c: c.count)

        common_codes = None

        invalid_cells = []

        # --------------------------------------------------------
        # Process valid cells
        # --------------------------------------------------------

        for cell in queried_cells:

            if cell.count == 0:
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
                if cell.count != self.h + 1:
                    continue

                remaining = cell.sum - candidate

                if not self.decoder.is_valid_sum(remaining):

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
