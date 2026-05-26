import itertools
import functools

class hammoing_codes():
    def __init__(self):
        self.n = 0


    def hamming_distance(a: str, b: str) -> int:
        return sum(x != y for x, y in zip(a, b))


    def is_valid_set(codewords, new_code):
        """
        Check if new_code maintains Hamming distance >= 3
        from all existing codewords.
        """
        for c in codewords:
            if hammoing_codes.hamming_distance(c, new_code) < 3:
                return False
        return True


    def generate_hamming_codewords(r: int):
        """
        Generate all non-zero binary vectors of length r
        (Hamming space basis candidates).
        """
        return [
            "".join(bits)
            for bits in itertools.product("01", repeat=r)
            if any(bits)
        ]


    def codebook_balance_score(codes, r: int):
        if not codes:
            return (0, 0, 0)

        counts = [
            sum(code[idx] == "1" for code in codes)
            for idx in range(r)
        ]
        size = len(codes)
        max_bucket = max(max(count, size - count) for count in counts)
        imbalance = sum(abs((2 * count) - size) for count in counts)
        weight_spread = sum(abs(code.count("1") * 2 - r) for code in codes)

        return (max_bucket, imbalance, weight_spread)


    @functools.lru_cache(maxsize=None)
    def _build_hamming_distance_3_codewords(n: int, r: int = 6):
        """
        Build a tuple of n codewords with Hamming distance >= 3.

        Parameters:
        -----------
        n : int
            number of codewords needed
        r : int
            bit-length of base space (increase if n is large)

        Returns:
        --------
        tuple[str, ...]
        """

        candidates = hammoing_codes.generate_hamming_codewords(r)

        # EC-BF stores one Bloom filter for each code position and bit value.
        # Balanced columns keep those filters close to the same load, which
        # improves the space/correctness tradeoff for a fixed number of bits.
        candidates.sort(
            key=lambda code: (
                abs((2 * code.count("1")) - r),
                code,
            )
        )

        if n <= 64 and len(candidates) <= 4096:
            start_limit = 128
        elif n <= 128:
            start_limit = 16
        else:
            start_limit = 1

        starts = candidates[:min(len(candidates), start_limit)]
        best_codes = None
        best_score = None

        for start in starts:
            codes = [start]
            remaining = [code for code in candidates if code != start]

            while len(codes) < n:
                valid = [
                    code for code in remaining
                    if hammoing_codes.is_valid_set(codes, code)
                ]

                if not valid:
                    break

                valid.sort(
                    key=lambda code: (
                        hammoing_codes.codebook_balance_score(
                            codes + [code],
                            r,
                        ),
                        code,
                    )
                )
                chosen = valid[0]
                codes.append(chosen)
                remaining.remove(chosen)

            if len(codes) != n:
                continue

            score = (
                hammoing_codes.codebook_balance_score(codes, r),
                tuple(codes),
            )

            if best_score is None or score < best_score:
                best_score = score
                best_codes = codes

        if best_codes is None:
            raise ValueError(
                f"Could not generate {n} codes. "
                f"Increase r (currently {r})."
            )

        return tuple(best_codes)


    def build_hamming_distance_3_codes(n: int, r: int = 6):
        """
        Build dictionary of n balanced codewords with Hamming distance >= 3.
        """
        codes = hammoing_codes._build_hamming_distance_3_codewords(n, r)
        return {i: code for i, code in enumerate(codes)}

    def choose_r(n: int) -> int:
        """
        Choose smallest r such that:
            n (r + 1) <= 2^r
        """
        r = 1
        while True:
            if n * (r + 1) <= (1 << r):
                return r+1
            r += 1

# ============================================================
# Example usage
# ============================================================

if __name__ == "__main__":
    n = 300
    r = hammoing_codes.choose_r(n)
    codes = hammoing_codes.build_hamming_distance_3_codes(n, r)

    for k, v in codes.items():
        print(k, v)

    print("\nPairwise distance check:")

    vals = list(codes.values())

    D = dict()

    for i in range(len(vals)):
        for j in range(i + 1, len(vals)):
            d = hammoing_codes.hamming_distance(vals[i], vals[j])
            if d in D:
                D[d]+=1
            else:
                D[d]=1
    print(D)
