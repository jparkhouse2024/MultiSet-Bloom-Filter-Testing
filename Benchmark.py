import time
import random
import string
import sys
import math
import os

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib-cache")

# ============================================================
# Import Datastructres
# ============================================================

from ECBF import ECBloomFilter
from BloomTree import Bloom_Tree
from BhBF import BhBF
from kBF import KBF
from flatbf import FlatBloofi, BloomFilter, FlatBloofiBlock
from hamming_Codes import HammingCodes

# Conceptual storage for one independently seeded hash function. The code uses
# double hashing, but the papers count hash families as part of the structure;
# this lets bits-per-pair include that seed/function metadata.
HASH_FUNCTION_BYTES = 16


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


def hash_family_bytes(num_hashes):
    return num_hashes * HASH_FUNCTION_BYTES


def ecbf_hash_bytes(bf):
    if hasattr(bf, "position_filters"):
        total_hashes = sum(
            len(filters) * bf.K
            for filters in bf.position_filters
        )

        return hash_family_bytes(total_hashes)

    total_hashes = 0

    def walk(node):
        nonlocal total_hashes

        total_hashes += len(node.edge_filters) * bf.K

        if node.leaf_filter is not None:
            total_hashes += bf.K

        for child in node.children.values():
            walk(child)

    walk(bf.root)

    return hash_family_bytes(total_hashes)


def bloom_tree_hash_bytes(tree):
    total_hashes = 0

    def walk(node):
        nonlocal total_hashes

        total_hashes += len(node.edge_filters) * tree.k_internal

        if node.leaf_filter is not None:
            total_hashes += tree.k_leaf

        for child in node.children:
            walk(child)

    walk(tree.root)

    return hash_family_bytes(total_hashes)


def bloom_params(n_items, target_fp):
    """
    Standard Bloom filter parameter selection.
    """
    if n_items <= 0:
        return 1, 1
    if not (0 < target_fp < 1):
        raise ValueError("target_fp must be between 0 and 1")

    m = math.ceil(
        -(n_items * math.log(target_fp)) / (math.log(2) ** 2)
    )
    k = max(1, round((m / n_items) * math.log(2)))

    return m, k


def bloom_m_for_fixed_k(n_items, target_fp, k):
    """
    Pick m for a fixed number of hashes using the Bloom FP approximation.
    """
    if n_items <= 0:
        return 1
    if not (0 < target_fp < 1):
        raise ValueError("target_fp must be between 0 and 1")
    if k <= 0:
        raise ValueError("k must be positive")

    denominator = math.log(1 - (target_fp ** (1 / k)))
    return math.ceil(-(k * n_items) / denominator)


def hashes_for_m(n_items, m):
    if n_items <= 0:
        return 1

    return max(1, round((m / n_items) * math.log(2)))


def bloom_tree_max_edge_groups(num_groups, d):
    height = math.ceil(math.log(num_groups, d)) if num_groups > 1 else 1
    edge_counts = {}

    for group_id in range(num_groups):
        digits = []
        x = group_id

        for _ in range(height):
            digits.append(x % d)
            x //= d

        path = tuple(reversed(digits))
        prefix = ()

        for edge in path:
            edge_key = (prefix, edge)
            edge_counts[edge_key] = edge_counts.get(edge_key, 0) + 1
            prefix = prefix + (edge,)

    return max(edge_counts.values(), default=1), height


def ecbf_max_position_sets(num_sets):
    codes = HammingCodes.build_hamming_distance_3_codes(num_sets)
    bucket_counts = {}

    for code in codes.values():
        for idx, bit in enumerate(code):
            bucket_key = (idx, bit)
            bucket_counts[bucket_key] = bucket_counts.get(bucket_key, 0) + 1

    return max(bucket_counts.values(), default=1), len(next(iter(codes.values())))


def tuned_parameters(num_sets, items_per_set, target_fp=0.01, d=2, space_factor=1.0):
    """
    Derive benchmark parameters from the dataset shape.

    target_fp is the intended query-level false-positive budget. Structures
    with multiple Bloom probes split that budget across filters by union bound.
    """
    total_items = num_sets * items_per_set

    bt_max_groups, bt_height = bloom_tree_max_edge_groups(num_sets, d)
    bt_filter_items = max(items_per_set, bt_max_groups * items_per_set)
    bt_filter_fp = target_fp / (bt_height + 1)
    bt_m, bt_k_internal = bloom_params(bt_filter_items, bt_filter_fp)
    bt_m = max(1, math.ceil(bt_m * space_factor))
    bt_k_internal = hashes_for_m(bt_filter_items, bt_m)
    bt_k_leaf = hashes_for_m(items_per_set, bt_m)

    ecbf_max_sets, ecbf_code_len = ecbf_max_position_sets(num_sets)
    ecbf_filter_items = max(items_per_set, ecbf_max_sets * items_per_set)
    ecbf_filter_fp = target_fp / (ecbf_code_len + 1)
    ecbf_m, ecbf_k = bloom_params(ecbf_filter_items, ecbf_filter_fp)
    ecbf_m = max(1, math.ceil(ecbf_m * space_factor))
    ecbf_k = hashes_for_m(ecbf_filter_items, ecbf_m)

    _, optimal_cell_k = bloom_params(total_items, target_fp)
    cell_h = 3
    cell_k = min(optimal_cell_k, cell_h)
    cell_m = bloom_m_for_fixed_k(total_items, target_fp, cell_k)
    cell_m = max(1, math.ceil(cell_m * space_factor))
    cell_k = min(cell_h, hashes_for_m(total_items, cell_m))

    flat_filter_fp = target_fp / max(1, num_sets)
    flat_m, flat_k = bloom_params(items_per_set, flat_filter_fp)
    flat_m = max(1, math.ceil(flat_m * space_factor))
    flat_k = hashes_for_m(items_per_set, flat_m)

    return {
        "EC-BF": {
            "L": num_sets,
            "m": ecbf_m,
            "K": ecbf_k,
            "space_factor": space_factor,
            "capacity_basis": ecbf_filter_items,
            "per_filter_fp": ecbf_filter_fp,
        },
        "Bloom_Tree": {
            "num_groups": num_sets,
            "d": d,
            "bits_per_filter": bt_m,
            "k_internal": bt_k_internal,
            "k_leaf": bt_k_leaf,
            "space_factor": space_factor,
            "capacity_basis": bt_filter_items,
            "per_filter_fp": bt_filter_fp,
        },
        "BhBF": {
            "num_sets": num_sets,
            "m": cell_m,
            "k": cell_k,
            "h": cell_h,
            "space_factor": space_factor,
            "capacity_basis": total_items,
        },
        "KBF": {
            "m": cell_m,
            "k": cell_k,
            "h": cell_h,
            "space_factor": space_factor,
            "capacity_basis": total_items,
        },
        "FlatBloofi": {
            "m": flat_m,
            "k": flat_k,
            "space_factor": space_factor,
            "capacity_basis": items_per_set,
            "per_filter_fp": flat_filter_fp,
        },
    }


def format_params(params):
    display = []

    for key, value in params.items():
        if key in {"capacity_basis", "per_filter_fp", "L", "num_sets", "num_groups"}:
            continue

        display.append(f"{key}={value}")

    if "capacity_basis" in params:
        display.append(f"n_basis={params['capacity_basis']}")

    if "per_filter_fp" in params:
        display.append(f"p_filter={params['per_filter_fp']:.4g}")

    return ", ".join(display)


# ============================================================
# Dataset Generator
# ============================================================

def generate_dataset(L=10, items_per_set=200, string_length=3000):
    """
    Returns:
        sets = [S1, ..., SL]
        labels = dict item -> set_id
    """
    sets = []
    labels = {}
    used = set()

    for i in range(L):
        S = []
        while len(S) < items_per_set:
            x = random_string(string_length)
            if x in used:
                continue

            used.add(x)
            S.append(x)
            labels[x] = i

        sets.append(S)

    return sets, labels


def generate_queries(sets, labels, num_queries=2000, p_in=0.7, string_length=3000):
    """
    Mix of positive and negative queries.
    """
    all_items = [x for S in sets for x in S]
    queries = []

    for _ in range(num_queries):
        if random.random() < p_in:
            x = random.choice(all_items)
        else:
            x = random_string(string_length)

            while x in labels:
                x = random_string(string_length)

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
        self.total = 0
        self.correct = 0
        self.member_total = 0
        self.member_correct = 0
        self.none_total = 0
        self.none_correct = 0
        self.false_negative = 0
        self.false_positive = 0
        self.classification_failure = 0
        self.misclassification = 0
        self.memory_accesses = 0

    @property
    def accuracy(self):
        return self.correct / self.total if self.total else 0

    @property
    def member_accuracy(self):
        return (
            self.member_correct / self.member_total
            if self.member_total
            else 0
        )

    @property
    def none_accuracy(self):
        return self.none_correct / self.none_total if self.none_total else 0

    @property
    def false_positive_rate(self):
        return (
            self.false_positive / self.none_total
            if self.none_total
            else 0
        )

    @property
    def false_negative_rate(self):
        return (
            self.false_negative / self.member_total
            if self.member_total
            else 0
        )

    @property
    def query_us(self):
        return (self.query_time / self.total) * 1e6 if self.total else 0

    @property
    def avg_memory_accesses(self):
        return self.memory_accesses / self.total if self.total else 0


def bloom_contains_count(bf, item):
    accesses = 0

    for h in bf._hashes(item):
        accesses += 1

        if not bf._get_bit(h):
            return False, accesses

    return True, accesses


def ecbf_query_count(bf, key):
    if hasattr(bf, "position_filters"):
        prefixes = [""]
        accesses = 0

        for idx in range(bf.k_prime):
            possible_bits = []

            for bit, bloom_filter in bf.iter_position_filters(idx):
                vkey = bf._virtual_key(idx, bit, key)
                contains, count = bloom_contains_count(bloom_filter, vkey)
                accesses += count

                if contains:
                    possible_bits.append(bit)

            if not possible_bits:
                return (False, None), accesses

            next_prefixes = []

            for prefix in prefixes:
                for bit in possible_bits:
                    candidate_prefix = prefix + bit

                    if candidate_prefix in bf.prefix_codes:
                        next_prefixes.append(candidate_prefix)

            if not next_prefixes:
                return (False, None), accesses

            prefixes = next_prefixes

        candidates = [
            prefix
            for prefix in prefixes
            if prefix in bf.c_inv
        ]

        if len(candidates) == 0:
            return (False, None), accesses

        if len(candidates) == 1:
            return (True, bf.c_inv[candidates[0]]), accesses

        return (True, None), accesses

    matches = []
    accesses = 0

    def dfs(node):
        nonlocal accesses

        if node.is_leaf:
            leaf_key = bf._virtual_key(node.path_id, key)
            contains, count = bloom_contains_count(node.leaf_filter, leaf_key)
            accesses += count

            if contains:
                matches.append(node.set_id)

            return

        vkey = bf._virtual_key(node.path_id, key)

        for bit, edge_filter in node.edge_filters.items():
            contains, count = bloom_contains_count(edge_filter, vkey)
            accesses += count

            if contains:
                dfs(node.children[bit])

    dfs(bf.root)

    if len(matches) == 0:
        return (False, None), accesses

    if len(matches) == 1:
        return (True, matches[0]), accesses

    return (True, None), accesses


def bloom_tree_query_count(tree, key):
    matches = []
    accesses = 0

    def dfs(node):
        nonlocal accesses

        if node.is_leaf:
            if node.group_id is None:
                return

            vkey = tree._virtual_key(node.path_id, key)
            contains, count = bloom_contains_count(node.leaf_filter, vkey)
            accesses += count

            if contains:
                matches.append(node.group_id)

            return

        vkey = tree._virtual_key(node.path_id, key)

        for edge_idx, edge_filter in enumerate(node.edge_filters):
            contains, count = bloom_contains_count(edge_filter, vkey)
            accesses += count

            if contains:
                dfs(node.children[edge_idx])

    dfs(tree.root)

    if len(matches) == 0:
        return None, accesses

    if len(matches) == 1:
        return matches[0], accesses

    return "CLASSIFICATION_FAILURE", accesses


def score_single_label(result, prediction, true_set):
    result.total += 1

    is_member_query = true_set is not None
    is_failure = prediction in {"CLASSIFICATION_FAILURE", "UNKNOWN", -1}

    if is_member_query:
        result.member_total += 1

        if prediction == true_set:
            result.correct += 1
            result.member_correct += 1
        elif prediction is None:
            result.false_negative += 1
        elif is_failure:
            result.classification_failure += 1
        else:
            result.misclassification += 1

        return

    result.none_total += 1

    if prediction is None:
        result.correct += 1
        result.none_correct += 1
    else:
        result.false_positive += 1


def score_set_label(result, prediction, true_set):
    result.total += 1

    expected = set() if true_set is None else {str(true_set)}
    predicted = set(prediction)

    if true_set is not None:
        result.member_total += 1

        if predicted == expected:
            result.correct += 1
            result.member_correct += 1
        elif not predicted:
            result.false_negative += 1
        else:
            result.misclassification += 1

        return

    result.none_total += 1

    if not predicted:
        result.correct += 1
        result.none_correct += 1
    else:
        result.false_positive += 1


# ============================================================
# EC-BF Adapter
# ============================================================

def run_ecbf(ECBFClass, params, sets, queries, labels):
    result = BenchmarkResult("EC-BF")
    bf = ECBFClass(
        L=params["L"],
        m=params["m"],
        K=params["K"],
    )

    # --------------------------------------
    # Build ground truth
    # --------------------------------------

    start = time.perf_counter()
    for i, S in enumerate(sets):
        for x in S:
            bf.insert(x, i)
    result.build_time = time.perf_counter() - start

    # --------------------------------------
    # memory
    # --------------------------------------
    result.space_bytes = deep_size(bf) + ecbf_hash_bytes(bf)

    start = time.perf_counter()

    for x in queries:
        res, accesses = ecbf_query_count(bf, x)
        result.memory_accesses += accesses
        true_set = labels.get(x)

        # prediction decoding
        if res == (False, None):
            pred = None
        elif res == (True, None):
            pred = -1
        else:
            pred = res[1]

        score_single_label(result, pred, true_set)

    result.query_time = time.perf_counter() - start

    return result


# ============================================================
# Bloom Tree Adapter
# ============================================================

def run_bloomTree(Bloom_Tree, params, sets, queries, labels):
    result = BenchmarkResult("Bloom_Tree")
    bf_index = Bloom_Tree(
        num_groups=params["num_groups"],
        d=params["d"],
        bits_per_filter=params["bits_per_filter"],
        k_internal=params["k_internal"],
        k_leaf=params["k_leaf"],
    )

    start = time.perf_counter()

    for i, S in enumerate(sets):
        for x in S:
            bf_index.insert(x, i)

    result.build_time = time.perf_counter() - start
    result.space_bytes = deep_size(bf_index) + bloom_tree_hash_bytes(bf_index)

    start = time.perf_counter()

    for x in queries:
        res, accesses = bloom_tree_query_count(bf_index, x)
        result.memory_accesses += accesses
        true_set = labels.get(x)
        score_single_label(result, res, true_set)

    result.query_time = time.perf_counter() - start

    return result


# ============================================================
# BhBF Adapter
# ============================================================

def run_bhbf(BhBFClass, params, sets, queries, labels):
    result = BenchmarkResult("BhBF")
    bf = BhBFClass(
        num_sets=params["num_sets"],
        m=params["m"],
        k=params["k"],
        h=params["h"],
    )

    start = time.perf_counter()

    for i, S in enumerate(sets):
        for x in S:
            bf.insert(x, i)

    result.build_time = time.perf_counter() - start
    result.space_bytes = deep_size(bf) + hash_family_bytes(params["k"])

    start = time.perf_counter()

    for x in queries:
        res = bf.query(x)
        result.memory_accesses += params["k"]
        true_set = labels.get(x)
        score_single_label(result, res, true_set)

    result.query_time = time.perf_counter() - start

    return result


# ============================================================
# KBF Adapter (IMPORTANT: treat value = set_id)
# ============================================================

def run_kbf(KBFClass, params, sets, queries, labels):
    result = BenchmarkResult("KBF")
    kbf = KBFClass(
        m=params["m"],
        k=params["k"],
        h=params["h"],
    )

    start = time.perf_counter()

    for i, S in enumerate(sets):
        for x in S:
            kbf.insert(x, str(i))

    result.build_time = time.perf_counter() - start
    result.space_bytes = deep_size(kbf) + hash_family_bytes(params["k"])

    start = time.perf_counter()

    for x in queries:
        res = kbf.query(x)
        result.memory_accesses += params["k"]

        true_set = labels.get(x)

        if true_set is not None:
            true_set = str(true_set)

        score_single_label(result, res, true_set)

    result.query_time = time.perf_counter() - start

    return result


# ============================================================
# FlatBloofi Adapter
# ============================================================

def run_flatbloofi(FlatBloofiClass, params, sets, queries, labels):
    result = BenchmarkResult("FlatBloofi")
    index = FlatBloofiClass(m=params["m"], k=params["k"])

    start = time.perf_counter()

    for i, S in enumerate(sets):
        bf = BloomFilter(params["m"], params["k"])
        for x in S:
            bf.add(x)
        index.insert(str(i), bf)

    result.build_time = time.perf_counter() - start
    result.space_bytes = deep_size(index) + hash_family_bytes(params["k"])

    start = time.perf_counter()

    for x in queries:
        res = index.query(x)
        result.memory_accesses += params["k"] * max(1, len(index.blocks))
        true_set = labels.get(x)
        score_set_label(result, res, true_set)

    result.query_time = time.perf_counter() - start

    return result


# ============================================================
# Main Experiment
# ============================================================

def run_suite(
    ECBFClass,
    Bloom_Tree,
    BhBFClass,
    KBFClass,
    FlatBloofiClass,
    L,
    set_size,
    target_fp=0.01,
    space_factor=1.0,
    seed=0,
    num_queries=2000,
    p_in=0.7,
):
    random.seed(seed)
    sets, labels = generate_dataset(L, items_per_set=set_size)
    queries = generate_queries(
        sets,
        labels,
        num_queries=num_queries,
        p_in=p_in,
    )
    params = tuned_parameters(
        L,
        set_size,
        target_fp=target_fp,
        space_factor=space_factor,
    )

    results = []

    r = run_ecbf(ECBFClass, params["EC-BF"], sets, queries, labels)
    results.append((r, params["EC-BF"]))

    r = run_bloomTree(Bloom_Tree, params["Bloom_Tree"], sets, queries, labels)
    results.append((r, params["Bloom_Tree"]))

    r = run_bhbf(BhBFClass, params["BhBF"], sets, queries, labels)
    results.append((r, params["BhBF"]))

    r = run_kbf(KBFClass, params["KBF"], sets, queries, labels)
    results.append((r, params["KBF"]))

    r = run_flatbloofi(
        FlatBloofiClass,
        params["FlatBloofi"],
        sets,
        queries,
        labels,
    )
    results.append((r, params["FlatBloofi"]))

    return results, len(queries)


def print_results(results, L, set_size, target_fp, num_queries, seed):
    print("\n================ RESULTS ================\n")
    print(
        f"Dataset: L={L}, set_size={set_size}, "
        f"target_fp={target_fp:.4g}, queries={num_queries}, seed={seed}"
    )
    print()

    for result, used_params in results:
        print(f"{result.name}")
        print(f"  Params: {format_params(used_params)}")
        print(f"  Build time: {result.build_time:.4f}s")
        print(f"  Query time: {result.query_time:.4f}s")
        print(f"  Memory (approx): {result.space_bytes / 1e6:.2f} MB")
        print(f"  Overall correctness: {result.accuracy:.4f}")
        print(f"  False positive rate: {result.false_positive_rate:.4f}")
        print(f"  False negative rate: {result.false_negative_rate:.4f}")
        print(f"  Query speed: {result.query_us:.2f} us/query")
        print(f"  Memory access overhead: {result.avg_memory_accesses:.2f} reads/query")
        print(
            f"  Member correctness: {result.member_accuracy:.4f} "
            f"({result.member_correct}/{result.member_total})"
        )
        print(
            f"  None correctness: {result.none_accuracy:.4f} "
            f"({result.none_correct}/{result.none_total})"
        )
        print(
            f"  FP: {result.false_positive}, "
            f"FN: {result.false_negative}, "
            f"Failures: {result.classification_failure}, "
            f"Misclassifications: {result.misclassification}"
        )
        print("----------------------------------------")


def _svg_x(value, min_log, max_log, left, plot_width):
    if max_log == min_log:
        return left + plot_width / 2

    return left + ((math.log10(value) - min_log) / (max_log - min_log)) * plot_width


def _svg_y(value, y_min, y_max, top, plot_height):
    clipped = min(y_max, max(y_min, value))
    return top + ((y_max - clipped) / (y_max - y_min)) * plot_height


def write_correctness_space_svg(series, metric, title, output_path):
    width = 920
    height = 580
    left = 92
    right = 238
    top = 58
    bottom = 82
    plot_width = width - left - right
    plot_height = height - top - bottom
    y_min = 0.90
    y_max = 1.0
    colors = {
        "EC-BF": "#2563eb",
        "Bloom_Tree": "#16a34a",
        "BhBF": "#dc2626",
        "KBF": "#9333ea",
        "FlatBloofi": "#d97706",
    }

    all_spaces = [
        max(1e-9, point["space_mb"])
        for points in series.values()
        for point in points
    ]
    min_log = math.log10(min(all_spaces))
    max_log = math.log10(max(all_spaces))

    lines = [
        '<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        (
            f'<text x="{width / 2}" y="28" text-anchor="middle" '
            'font-family="Arial" font-size="20" font-weight="700">'
            f'{title}</text>'
        ),
    ]

    # Grid and y labels.
    for i in range(6):
        y_value = y_min + i * ((y_max - y_min) / 5)
        y = _svg_y(y_value, y_min, y_max, top, plot_height)
        lines.append(
            f'<line x1="{left}" y1="{y:.2f}" '
            f'x2="{left + plot_width}" y2="{y:.2f}" '
            'stroke="#e5e7eb"/>'
        )
        lines.append(
            f'<text x="{left - 12}" y="{y + 4:.2f}" '
            'text-anchor="end" font-family="Arial" font-size="12">'
            f'{y_value * 100:.0f}%</text>'
        )

    # X labels use endpoints and midpoint in log space.
    for i in range(3):
        log_value = min_log + i * ((max_log - min_log) / 2)
        value = 10 ** log_value
        x = _svg_x(value, min_log, max_log, left, plot_width)
        lines.append(
            f'<line x1="{x:.2f}" y1="{top}" '
            f'x2="{x:.2f}" y2="{top + plot_height}" '
            'stroke="#f3f4f6"/>'
        )
        lines.append(
            f'<text x="{x:.2f}" y="{top + plot_height + 24}" '
            'text-anchor="middle" font-family="Arial" font-size="12">'
            f'{value:.3g}</text>'
        )

    lines.extend(
        [
            (
                f'<line x1="{left}" y1="{top + plot_height}" '
                f'x2="{left + plot_width}" y2="{top + plot_height}" '
                'stroke="#111827" stroke-width="1.4"/>'
            ),
            (
                f'<line x1="{left}" y1="{top}" '
                f'x2="{left}" y2="{top + plot_height}" '
                'stroke="#111827" stroke-width="1.4"/>'
            ),
            (
                f'<text x="{left + plot_width / 2}" y="{height - 24}" '
                'text-anchor="middle" font-family="Arial" font-size="14">'
                'Approx memory used, MB (log scale)</text>'
            ),
            (
                f'<text x="22" y="{top + plot_height / 2}" '
                'text-anchor="middle" font-family="Arial" font-size="14" '
                'transform="rotate(-90 22 '
                f'{top + plot_height / 2})">Correctness</text>'
            ),
        ]
    )

    legend_y = top + 12

    for idx, (name, points) in enumerate(series.items()):
        color = colors.get(name, "#374151")
        sorted_points = sorted(points, key=lambda p: p["space_mb"])
        coords = []

        for point in sorted_points:
            x = _svg_x(
                max(1e-9, point["space_mb"]),
                min_log,
                max_log,
                left,
                plot_width,
            )
            y = _svg_y(point[metric], y_min, y_max, top, plot_height)
            coords.append((x, y, point))

        if len(coords) > 1:
            path = " ".join(f"{x:.2f},{y:.2f}" for x, y, _ in coords)
            lines.append(
                f'<polyline points="{path}" fill="none" '
                f'stroke="{color}" stroke-width="2.5"/>'
            )

        for x, y, point in coords:
            label = f'{point[metric] * 100:.2f}% @ p={point["target_fp"]:.3g}'
            lines.append(
                f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4.5" '
                f'fill="{color}"><title>{name}: {label}</title></circle>'
            )

        ly = legend_y + idx * 24
        lx = left + plot_width + 28
        lines.append(
            f'<line x1="{lx}" y1="{ly}" x2="{lx + 22}" y2="{ly}" '
            f'stroke="{color}" stroke-width="3"/>'
        )
        lines.append(
            f'<text x="{lx + 30}" y="{ly + 4}" '
            'font-family="Arial" font-size="13">'
            f'{name}</text>'
        )

    lines.append("</svg>")

    with open(output_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))


def generate_diagrams(
    ECBFClass,
    Bloom_Tree,
    BhBFClass,
    KBFClass,
    FlatBloofiClass,
    L,
    set_size,
    seed=0,
    output_dir="diagrams",
    target_fp=0.2,
    space_factors=None,
    num_queries=10000,
    p_in=0.5,
):
    if space_factors is None:
        space_factors = [
            0.003,
            0.005,
            0.008,
            0.01,
            0.015,
            0.02,
            0.03,
            0.05,
            0.08,
            0.12,
            0.2,
            0.35,
            0.6,
            0.8,
            1.0,
            1.2,
            1.5,
            2.0,
            3.0,
            5.0,
        ]

    os.makedirs(output_dir, exist_ok=True)
    series = {}
    total_pairs = L * set_size

    for space_factor in space_factors:
        results, _ = run_suite(
            ECBFClass,
            Bloom_Tree,
            BhBFClass,
            KBFClass,
            FlatBloofiClass,
            L,
            set_size,
            target_fp=target_fp,
            space_factor=space_factor,
            seed=seed,
            num_queries=num_queries,
            p_in=p_in,
        )

        for result, _ in results:
            series.setdefault(result.name, []).append(
                {
                    "target_fp": target_fp,
                    "space_factor": space_factor,
                    "space_mb": result.space_bytes / 1e6,
                    "bits_per_pair": (result.space_bytes * 8) / total_pairs,
                    "accuracy": result.accuracy,
                    "member_accuracy": result.member_accuracy,
                    "none_accuracy": result.none_accuracy,
                    "false_positive_rate": result.false_positive_rate,
                    "false_negative_rate": result.false_negative_rate,
                    "query_us": result.query_us,
                    "memory_access_overhead": result.avg_memory_accesses,
                }
            )

    outputs = [
        (
            "accuracy",
            "Correctness Rate",
            "correctness_rate",
            "Correctness rate",
            False,
        ),
        (
            "false_positive_rate",
            "False Positive Rate",
            "false_positive_rate",
            "False positive rate",
            False,
        ),
        (
            "false_negative_rate",
            "False Negative Rate",
            "false_negative_rate",
            "False negative rate",
            False,
        ),
        (
            "query_us",
            "Query Speed",
            "query_speed",
            "Query time (microseconds/query)",
            True,
        ),
        (
            "memory_access_overhead",
            "Memory Access Overhead",
            "memory_access_overhead",
            "Approx memory reads/query",
            True,
        ),
    ]

    written = []

    for metric, title, basename, xlabel, log_x in outputs:
        paths = write_metric_plot(
            series,
            metric,
            title,
            xlabel,
            os.path.join(output_dir, basename),
            log_x=log_x,
        )
        written.extend(paths)

    print("\nSweep stress summary:")

    for name, points in series.items():
        min_accuracy = min(point["accuracy"] for point in points)
        max_fpr = max(point["false_positive_rate"] for point in points)
        max_fnr = max(point["false_negative_rate"] for point in points)
        min_bits = min(point["bits_per_pair"] for point in points)
        max_bits = max(point["bits_per_pair"] for point in points)
        print(
            f"  {name}: min correctness={min_accuracy:.4f}, "
            f"max FPR={max_fpr:.4f}, max FNR={max_fnr:.4f}, "
            f"bits/pair range={min_bits:.2f}-{max_bits:.2f}"
        )

    return written


def write_metric_plot(series, metric, title, xlabel, output_base, log_x=False):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {
        "EC-BF": "#2563eb",
        "Bloom_Tree": "#16a34a",
        "BhBF": "#dc2626",
        "KBF": "#9333ea",
        "FlatBloofi": "#d97706",
    }
    markers = {
        "EC-BF": "o",
        "Bloom_Tree": "s",
        "BhBF": "^",
        "KBF": "D",
        "FlatBloofi": "P",
    }

    fig, ax = plt.subplots(figsize=(9.5, 6.2), dpi=150)

    for name, points in series.items():
        valid_points = [
            point for point in points
            if point[metric] > 0 or not log_x
        ]
        valid_points.sort(key=lambda point: point[metric])

        x_values = [point[metric] for point in valid_points]
        y_values = [point["bits_per_pair"] for point in valid_points]

        ax.plot(
            x_values,
            y_values,
            marker=markers.get(name, "o"),
            color=colors.get(name, None),
            linewidth=1.8,
            markersize=5,
            label=name,
        )

    ax.set_title(f"{title} vs Space", fontsize=14, weight="bold")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Bits per element-set pair")
    ax.set_yscale("log")

    if metric == "accuracy":
        ax.set_xlim(0.90, 1.001)
    elif metric == "false_positive_rate":
        ax.set_xlim(0, 0.2)
    elif metric == "false_negative_rate":
        ax.set_xlim(left=0)
    elif log_x:
        ax.set_xscale("log")

    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="best", frameon=True)
    fig.tight_layout()

    png_path = f"{output_base}.png"
    pdf_path = f"{output_base}.pdf"
    fig.savefig(png_path)
    fig.savefig(pdf_path)
    plt.close(fig)

    return [png_path, pdf_path]


def run_all(
    ECBFClass,
    Bloom_Tree,
    BhBFClass,
    KBFClass,
    FlatBloofiClass,
    L,
    set_size,
    target_fp=0.01,
    space_factor=1.0,
    seed=0,
    num_queries=2000,
    p_in=0.7,
):
    results, num_queries = run_suite(
        ECBFClass,
        Bloom_Tree,
        BhBFClass,
        KBFClass,
        FlatBloofiClass,
        L,
        set_size,
        target_fp=target_fp,
        space_factor=space_factor,
        seed=seed,
        num_queries=num_queries,
        p_in=p_in,
    )
    print_results(results, L, set_size, target_fp, num_queries, seed)

    return results


# ============================================================
# Usage
# ============================================================

if __name__ == "__main__":
    L = 10
    SET_SIZE = 200
    NUM_QUERIES = 1000
    P_IN = 0.5
    SEED = 0

    run_all(
        ECBloomFilter,
        Bloom_Tree,
        BhBF,
        KBF,
        FlatBloofi,
        L,
        SET_SIZE,
        target_fp=0.2,
        space_factor=1.5,
        seed=SEED,
        num_queries=NUM_QUERIES,
        p_in=P_IN,
    )

    diagram_paths = generate_diagrams(
        ECBloomFilter,
        Bloom_Tree,
        BhBF,
        KBF,
        FlatBloofi,
        L,
        SET_SIZE,
        seed=SEED,
        num_queries=NUM_QUERIES,
        p_in=P_IN,
    )

    print("\nGenerated diagrams:")

    for path in diagram_paths:
        print(f"  {path}")
