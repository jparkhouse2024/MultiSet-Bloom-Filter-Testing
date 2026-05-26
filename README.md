# MultiSet Bloom Filter Testing

This repository benchmarks several approximate multi-set membership data
structures against the same generated workload:

- EC-BF
- BloomTree
- BhBF
- kBF
- FlatBloofi

The benchmark generates disjoint sets, mixes member and non-member queries, and
reports correctness, member correctness, non-member correctness, false positive
rate, false negative rate, query speed, memory access overhead, and normalized
space cost.

## Current Benchmark Model

Run the full benchmark and regenerate plots with:

```bash
python3 Benchmark.py
```

The generated figures are written to `diagrams/` as PNG and PDF files. The
main comparison plots use bits per element-set pair on the vertical axis.
Correctness plots focus on the mostly-correct region from 90% to 100%, and the
false-positive plot focuses on rates from 0 to 0.2.

## Implementation Notes

The Bloom-style structures now share the same deterministic hash helper from
`BloomFilter.py`, so differences in benchmark results are less likely to come
from unrelated hashing choices.

EC-BF keeps the defining Hamming-code idea but uses compact per-position,
per-bit Bloom filters rather than a trie of Bloom filters. Its Hamming codebook
is balanced so each code position has similar zero/one load, improving the
space/correctness tradeoff.

BloomTree and FlatBloofi both use tree or packed Bloom-filter layouts. BloomTree
uses independently namespaced edge filters, while FlatBloofi packs per-set
Bloom filters into transposed bit slices.

BhBF and kBF both use XOR-style encoded cell values with count-aware decoding.
kBF uses value encodings for approximate key-value lookup, while BhBF uses the
Bh sequence for multi-set classification.

## Diagrams

The benchmark regenerates:

- `diagrams/correctness_rate.pdf`
- `diagrams/false_positive_rate.pdf`
- `diagrams/false_negative_rate.pdf`
- `diagrams/query_speed.pdf`
- `diagrams/memory_access_overhead.pdf`

PNG versions are generated alongside the PDFs.
