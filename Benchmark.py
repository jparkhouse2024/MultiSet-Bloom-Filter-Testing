import time
import random
import string
import sys
from collections import defaultdict

# ============================================================
# Import Datastructres
# ============================================================

from ECBF import ECBloomFilter
from BloomTree import Bloom_Tree
from BhBF import BhBF
from kBF import KBF
from flatbf import FlatBloofi, BloomFilter, FlatBloofiBlock


# ============================================================
# Utilities
# ============================================================

def random_string(n=10):
    return ''.join(random.choice(string.ascii_lowercase) for _ in range(n))


def deep_size(obj, seen=None):
    """
    Approximate memory usage (recursive).
    """
    if seen is None:
        seen = set()

    obj_id = id(obj)
    if obj_id in seen:
        return 0
    seen.add(obj_id)

    size = sys.getsizeof(obj)

    if isinstance(obj, dict):
        size += sum(deep_size(k, seen) + deep_size(v, seen) for k, v in obj.items())

    elif hasattr(obj, '__dict__'):
        size += deep_size(obj.__dict__, seen)

    elif isinstance(obj, (list, tuple, set)):
        size += sum(deep_size(i, seen) for i in obj)

    return size


# ============================================================
# Dataset Generator
# ============================================================

def generate_dataset(L=10, items_per_set=200):
    """
    Returns:
        sets = [S1, ..., SL]
        labels = dict item -> set_id
    """
    sets = []
    labels = {}

    for i in range(L):
        S = []
        for _ in range(items_per_set):
            x = random_string(30)
            S.append(x)
            labels[x] = i
        sets.append(S)

    return sets, labels


def generate_queries(sets, labels, num_queries=2000, p_in=0.7):
    """
    Mix of positive and negative queries.
    """
    all_items = [x for S in sets for x in S]
    queries = []

    for _ in range(num_queries):
        if random.random() < p_in:
            x = random.choice(all_items)
        else:
            x = random_string(12)

        queries.append(x)

    return queries


# ============================================================
# Benchmark Runner
# ============================================================

class BenchmarkResult:
    def __init__(self, name):
        self.name = name
        self.build_time = 0
        self.query_time = 0
        self.space_bytes = 0
        self.accuracy = 0
        self.false_negative = 0
        self.false_positive = 0


# ============================================================
# EC-BF Adapter
# ============================================================

def run_ecbf(ECBFClass, L, sets, queries, labels):
    bf = ECBFClass( L=L)

    # --------------------------------------
    # Build ground truth
    # --------------------------------------

    start = time.perf_counter()
    for i, S in enumerate(sets):
        for x in S:
            bf.insert(x, i)
    build_time = time.perf_counter() - start

    # --------------------------------------
    # memory
    # --------------------------------------
    space = deep_size(bf)
    print(deep_size(bf.PT))
    # print(deep_size(bf.A))

    correct = 0
    fp = 0
    fn = 0
    failure = 0

    start = time.perf_counter()

    for x in queries:
        res = bf.query(x)
        
        if x in labels:  
            true_set = labels[x]
        else:
            true_set=None

        # prediction decoding
        if res == (False, None):
            pred = None
        elif res == (True, None):
            pred = -1
        else:
            pred = res[1]

        # ----------------------------------
        # scoring
        # ----------------------------------
        if pred == true_set:
            correct += 1
        else:
            if pred is None and true_set is not None:
                fn += 1
            elif pred is not None and true_set is None:
                fp += 1
            elif pred == -1:
                failure += 1

    query_time = time.perf_counter() - start

    return build_time, query_time, space, correct, fp, fn


# ============================================================
# Bloom Tree Adapter
# ============================================================

def run_bloomTree(Bloom_Tree, sets, queries, labels):
    bf_index = Bloom_Tree(num_groups=len(sets), d=2, bits_per_filter=2048)

    start = time.perf_counter()

    for i, S in enumerate(sets):
        for x in S:
            bf_index.insert(x, i)

    build_time = time.perf_counter() - start

    space = deep_size(bf_index)

    correct = 0
    fp = fn = 0

    start = time.perf_counter()

    for x in queries:
        res = bf_index.query(x)

        if x in labels:  
            true_set = labels[x]
        else:
            true_set=None

        pred = res

        if pred == true_set:
            correct += 1
        elif pred is None:
            fn += 1
        elif pred == "CLASSIFICATION_FAILURE":
            fn += 1
        else:
            fp += 1

    query_time = time.perf_counter() - start

    return build_time, query_time, space, correct, fp, fn


# ============================================================
# BhBF Adapter
# ============================================================

def run_bhbf(BhBFClass, sets, queries):
    bf = BhBFClass(num_sets=len(sets), m=2000, k=3)

    start = time.perf_counter()

    for i, S in enumerate(sets):
        for x in S:
            bf.insert(x, i)

    build_time = time.perf_counter() - start
    space = deep_size(bf)

    correct = 0
    fp = fn = 0

    start = time.perf_counter()

    for x in queries:
        res = bf.query(x)

        true_set = None
        for i, S in enumerate(sets):
            if x in S:
                true_set = i
                break

        pred = res

        if pred == true_set:
            correct += 1
        elif pred is None:
            fn += 1
        else:
            fp += 1

    query_time = time.perf_counter() - start

    return build_time, query_time, space, correct, fp, fn


# ============================================================
# KBF Adapter (IMPORTANT: treat value = set_id)
# ============================================================

def run_kbf(KBFClass, sets, queries):
    kbf = KBFClass(m=5000, k=3)

    start = time.perf_counter()

    for i, S in enumerate(sets):
        for x in S:
            kbf.insert(x, str(i))

    build_time = time.perf_counter() - start
    space = deep_size(kbf)

    correct = 0
    fp = fn = 0

    start = time.perf_counter()

    for x in queries:
        res = kbf.query(x)

        true_set = None
        for i, S in enumerate(sets):
            if x in S:
                true_set = str(i)
                break

        if res == true_set:
            correct += 1
        elif res is None:
            fn += 1
        else:
            fp += 1

    query_time = time.perf_counter() - start

    return build_time, query_time, space, correct, fp, fn


# ============================================================
# FlatBloofi Adapter
# ============================================================

def run_flatbloofi(FlatBloofiClass, sets, queries):
    index = FlatBloofiClass(m=256, k=3)

    start = time.perf_counter()

    for i, S in enumerate(sets):
        bf = BloomFilter(256, 3)
        for x in S:
            bf.add(x)
        index.insert(str(i), bf)

    build_time = time.perf_counter() - start
    space = deep_size(index)

    correct = 0
    fp = fn = 0

    start = time.perf_counter()

    for x in queries:
        hashes = index._hashes(x)
        res = index.query(hashes)

        true_sets = set()
        for i, S in enumerate(sets):
            if x in S:
                true_sets.add(str(i))

        pred = set(res)

        if pred == true_sets:
            correct += 1
        elif not pred:
            fn += 1
        else:
            fp += 1

    query_time = time.perf_counter() - start

    return build_time, query_time, space, correct, fp, fn


# ============================================================
# Main Experiment
# ============================================================

def run_all(
    ECBFClass,
    Bloom_Tree,
    BhBFClass,
    KBFClass,
    FlatBloofiClass, L, set_size
):
    sets, labels = generate_dataset(L, items_per_set=set_size)
    queries = generate_queries(sets, labels)

    results = []

    # EC-BF
    r = run_ecbf(ECBFClass, L, sets, queries, labels)
    results.append(("EC-BF", r))

    # Bloofi
    r = run_bloomTree(Bloom_Tree, sets, queries, labels)
    results.append(("Bloom_Tree", r))

    # BhBF
    r = run_bhbf(BhBFClass, sets, queries)
    results.append(("BhBF", r))

    # KBF
    r = run_kbf(KBFClass, sets, queries)
    results.append(("KBF", r))

    # FlatBloofi
    r = run_flatbloofi(FlatBloofiClass, sets, queries)
    results.append(("FlatBloofi", r))

    print("\n================ RESULTS ================\n")

    for name, (bt, qt, mem, correct, fp, fn) in results:
        total = len(queries)

        print(f"{name}")
        print(f"  Build time: {bt:.4f}s")
        print(f"  Query time: {qt:.4f}s")
        print(f"  Memory (approx): {mem / 1e6:.2f} MB")
        print(f"  Accuracy: {correct / total:.4f}")
        print(f"  FP: {fp}, FN: {fn}")
        print("----------------------------------------")


# ============================================================
# Usage
# ============================================================

if __name__ == "__main__":

    run_all(
    ECBloomFilter,
    Bloom_Tree,
    BhBF,
    KBF,
    FlatBloofi, 5, 100
)
    pass