import itertools

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


    def build_hamming_distance_3_codes(n: int, r: int = 6):
        """
        Build dictionary of n codewords with Hamming distance >= 3.

        Parameters:
        -----------
        n : int
            number of codewords needed
        r : int
            bit-length of base space (increase if n is large)

        Returns:
        --------
        dict[int, str]
        """

        candidates = hammoing_codes.generate_hamming_codewords(r)

        # Sort longer / higher-weight vectors first (better separation)
        candidates.sort(key=lambda x: -x.count("1"))

        codes = []

        for c in candidates:
            if hammoing_codes.is_valid_set(codes, c):
                codes.append(c)
            if len(codes) == n:
                break

        if len(codes) < n:
            raise ValueError(
                f"Could only generate {len(codes)} codes. "
                f"Increase r (currently {r})."
            )

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