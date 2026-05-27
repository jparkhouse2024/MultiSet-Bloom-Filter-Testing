import itertools


class HammingCodes:
    """
    Binary Hamming code construction.

    Produces codewords with minimum Hamming distance 3.

    Uses the classical linear Hamming code:
        [2^m - 1, 2^m - m - 1, 3]

    where:
        n = codeword length
        k = message length
        d = minimum distance = 3
    """

    @staticmethod
    def hamming_distance(a: str, b: str) -> int:
        return sum(x != y for x, y in zip(a, b))

    @staticmethod
    def required_m(num_codewords: int) -> int:
        """
        Find smallest m such that:
            2^(2^m - m - 1) >= num_codewords
        """

        m = 2

        while True:
            k = (1 << m) - m - 1
            if (1 << k) >= num_codewords:
                return m
            m += 1

    @staticmethod
    def parity_positions(m: int):
        """
        Return parity bit positions (1-indexed).
        """
        return {1 << i for i in range(m)}

    @staticmethod
    def encode_message(message: int, m: int) -> str:
        """
        Encode integer message into a Hamming codeword.
        """

        n = (1 << m) - 1
        parity_pos = HammingCodes.parity_positions(m)

        code = [0] * (n + 1)  # 1-indexed

        # --------------------------------------------------
        # Fill data bits
        # --------------------------------------------------

        data_positions = [
            pos for pos in range(1, n + 1)
            if pos not in parity_pos
        ]

        for bit_idx, pos in enumerate(data_positions):
            code[pos] = (message >> bit_idx) & 1

        # --------------------------------------------------
        # Compute parity bits
        # --------------------------------------------------

        for p in parity_pos:
            parity = 0

            for pos in range(1, n + 1):
                if pos & p:
                    parity ^= code[pos]

            code[p] = parity

        return "".join(str(code[pos]) for pos in range(1, n + 1))

    @staticmethod
    def build_hamming_distance_3_codes(num_codes: int):
        """
        Build dictionary:
            {id -> codeword}

        with guaranteed minimum Hamming distance 3.
        """

        m = HammingCodes.required_m(num_codes)

        n = (1 << m) - 1
        k = n - m

        max_codes = 1 << k

        if num_codes > max_codes:
            raise ValueError(
                f"Cannot generate {num_codes} codewords "
                f"with m={m}."
            )

        codes = {}

        for i in range(num_codes):
            codes[i] = HammingCodes.encode_message(i, m)

        return codes


# ============================================================
# Example usage
# ============================================================

if __name__ == "__main__":

    num_sets = 300

    codes = HammingCodes.build_hamming_distance_3_codes(num_sets)

    vals = list(codes.values())

    print("Generated", len(vals), "codes")
    print("Code length:", len(vals[0]))

    # --------------------------------------------------
    # Verify minimum distance
    # --------------------------------------------------

    D = {}

    min_dist = float("inf")

    for i in range(len(vals)):
        for j in range(i + 1, len(vals)):

            d = HammingCodes.hamming_distance(vals[i], vals[j])

            min_dist = min(min_dist, d)

            D[d] = D.get(d, 0) + 1

    print("Minimum distance:", min_dist)
    print("Distance histogram:", D)