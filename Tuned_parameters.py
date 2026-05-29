import math 
from hamming_Codes import HammingCodes
from math import comb as choose

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



def ecbf_max_position_sets(num_sets):
    codes = HammingCodes.build_hamming_distance_3_codes(num_sets)
    bucket_counts = {}

    for code in codes.values():
        for idx, bit in enumerate(code):
            bucket_key = (idx, bit)
            bucket_counts[bucket_key] = bucket_counts.get(bucket_key, 0) + 1

    return max(bucket_counts.values(), default=1), len(next(iter(codes.values())))

class Parameters:

    def tuned_parameters(num_sets, items_per_set, target_fp=0.01, d=2, space_factor=1.0):
        """
        Derive benchmark parameters from the dataset shape.

        target_fp is the intended query-level false-positive budget. Structures
        with multiple Bloom probes split that budget across filters by union bound.
        """
        total_items = num_sets * items_per_set


        '''
        Bloom Tree Paramater Description: Each individual Bloom Filter has it's own 
        m and k based on the level in the tree. We do assume however that the sets 
        have an even distribution. 

        The internal bloom filters will have k=1 (if d=2) while the leafs will be 
        more loaded

        The goal is to have a 50/50 chance of hashing to a 1 or a 0 for any given hash 
        so we can use 

        m = n*k / ln(2)

        We have to manually change the space inside the BloomTree class size 
        each filter size is custom. 

        Changes: The only changes we can test is by changing the space_factor. 
        The hash function numbers are preset based on the math from the paper. 
        '''
        target_uc = .01

        bt_l = math.ceil(math.log(num_sets, d))
        bt_k_internal = max(1, math.ceil(math.log2(d)))
        bt_k_leaf = max(1,math.ceil(math.log2((bt_l * (d - 1)) / (target_uc * d))))

        # From:
        # k = ln(2) * m / n
        #
        # => m = n*k / ln(2)

        bt_internal_n = num_sets * items_per_set
        bt_leaf_n = items_per_set

        bt_m_internal = math.ceil(bt_internal_n * bt_k_internal / math.log(2))
        bt_m_leaf = math.ceil(bt_leaf_n * bt_k_leaf / math.log(2))


        '''
        ECBF Paramater Description: Just like Bloom Tree, each Bloom Filter is 
        tailored to the estimated size of the set. Unlike Bloom Tree, the number 
        of hash functions for internal and leaf nodes is the same: K. So we 
        only need to specify K to change 

        For now, we can set this to 6 since there is no optimal setting. 

        We also want The goal is to have a 50/50 chance of hashing to a 1 or a 0
        for any given hash so we can use 
        
        m = n*k / ln(2)

        But since we also have a space_factor, we use the space_factor in the function. 

        Change: K and space_factor
        '''

        ecbf_K=6


        '''
        COMB:   We set w to 3, l to the lowest value that can accomidate the
        number of sets.  
        Furthermore, we set k=4

        Since each element inserts k*w elements, we take m with relation to k*w,
        but start with a 50/50 0/1 ratio. 

        Change: w to 3 or 4. K, space_factor
        '''

        comb_w = 3
        comb_l = comb_w
        while choose(comb_l, comb_w) < num_sets:
            comb_l += 1
        
        # target per-hash-set FP
        p_taget= target_fp / comb_l

 
        comb_m = math.ceil(-(total_items * comb_w * math.log(p_taget))
                      / (math.log(2) ** 2)*space_factor)

        # optimal k
        comb_k = max(1, math.ceil((comb_m / (total_items * comb_w)) * math.log(2)))


        '''
        IBF, KBF, PhBF:   These each use the same principle of k hash functions 
        into a single array, thus we use the same k and m for each starting with 
        a 50/50 0/1 ratio. 

        Change: space_factor
        '''

        _, optimal_cell_k = bloom_params(total_items, target_fp)
        cell_h = 3
        cell_k = 5
        cell_m = int(total_items*cell_k/ math.log(2)*space_factor)


        flat_filter_fp = target_fp / max(1, num_sets)
        flat_m, flat_k = bloom_params(items_per_set, flat_filter_fp)
        flat_m = int(flat_m*space_factor)
        # flat_m, flat_k = bloom_params(items_per_set, flat_filter_fp)
        # flat_m = max(1, math.ceil(flat_m * space_factor))
        # flat_k = hashes_for_m(items_per_set, flat_m)

        result =  {
            "EC-BF": {
                "L": num_sets,
                "K": ecbf_K,
                "space_factor": space_factor,
                "total_items": total_items,
            },
            "Bloom_Tree": {
                "num_groups": num_sets,
                "d": d,
                "bits_per_filter_internal": bt_m_internal,
                "bits_per_filter_leaf": bt_m_leaf,
                "k_internal": bt_k_internal,
                "k_leaf": bt_k_leaf,
                "space_factor": space_factor,
                "total_items": total_items,
            },
            "BhBF": {
                "num_sets": num_sets,
                "m": cell_m,
                "k": cell_k,
                "h": cell_h,
                "space_factor": space_factor,
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
            },
            "IBF": {
                "m": cell_m,
                "k": cell_k,
            },
            "COMB": {
                "m": comb_m,
                "l": comb_l,
                "k": comb_k,
                "w": comb_w,
            },
        }
        print(result)
        return result









