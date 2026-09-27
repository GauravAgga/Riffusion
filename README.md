# RankWeave

A lightweight, framework-agnostic library for fusing multiple search rankings into a single, better-ranked result list — built for modern hybrid search and RAG systems.

> **Note:** This project is **not** related to the [Riffusion music-generation AI](https://github.com/riffusion/riffusion-hobby). The PyPI package is published as `rankweave` to avoid naming conflicts.

![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)

## Why RankWeave?

Hybrid retrieval pipelines often combine multiple retrievers — for example, BM25 keyword search and dense vector search. Each retriever returns scores on incompatible scales, making direct score combination unreliable.

**Reciprocal Rank Fusion (RRF)** solves this by using only rank positions, not raw scores. For each document `d` that appears in one or more ranked lists:

```
RRF(d) = Σ  1 / (k + rank_i(d))
```

Documents that rank highly across multiple retrievers rise to the top. No score normalization required.

**Weighted RRF** keeps the same rank-based approach but applies a per-ranking weight:

```
WRRF(d) = Σ  w_i / (k + rank_i(d))
```

When scores *are* meaningful, **Weighted Score Fusion** combines (optionally min-max normalized) scores with per-ranking weights:

```
WSF(d) = Σ  w_i · s'_i(d)
```



## Features

- **RRF fusion** — merge rankings from heterogeneous retrievers using rank positions only
- **Weighted RRF** — same rank-based fusion with required per-list weights
- **Weighted score fusion** — combine retriever scores with per-list weights and optional min-max normalization
- **Framework-agnostic** — works with dict payloads from Elasticsearch, OpenSearch, pgvector, LangChain, LlamaIndex, or any custom retriever
- **Flexible document IDs** — use a field name or a callable extractor
- **Zero runtime dependencies**
- **Pluggable architecture** — designed for additional fusion algorithms

**Planned:** additional score-based fusion variants.

## Installation

```bash
pip install rankweave
```

For local development:

```bash
git clone https://github.com/GauravAgga/Riffusion.git
cd Riffusion
pip install -e ".[dev]"
```

**Requirements:** Python 3.10+

## Quick Start



### Reciprocal Rank Fusion (RRF)

Fuse BM25 and vector search results when score scales are incompatible and all retrievers should count equally:

```python
from rankweave import fuse

bm25_results = [
    {"id": "doc-a", "text": "Introduction to machine learning"},
    {"id": "doc-b", "text": "Deep learning fundamentals"},
    {"id": "doc-c", "text": "Natural language processing"},
]

vector_results = [
    {"id": "doc-b", "text": "Deep learning fundamentals"},
    {"id": "doc-a", "text": "Introduction to machine learning"},
    {"id": "doc-d", "text": "Transformer architectures"},
]

fused = fuse([bm25_results, vector_results], method="rrf", k=60, top_k=10)

for result in fused:
    print(result.document_id, result.score)
```



### Weighted Reciprocal Rank Fusion

When ranks are trustworthy but one retriever should count more, use weighted RRF. Weights are **required**, must each satisfy `0 < w_i <= 1`, and must sum to `1`:

```python
from rankweave import fuse

bm25_results = [
    {"id": "doc-a", "text": "Introduction to machine learning"},
    {"id": "doc-b", "text": "Deep learning fundamentals"},
]

vector_results = [
    {"id": "doc-b", "text": "Deep learning fundamentals"},
    {"id": "doc-c", "text": "Transformer architectures"},
]

fused = fuse(
    [bm25_results, vector_results],
    method="weighted_rrf",
    weights=[0.3, 0.7],
    k=60,
    top_k=10,
)

for result in fused:
    print(result.document_id, result.score)
```



### Weighted Score Fusion

When retriever scores are meaningful, fuse them with weights (scores are min-max normalized per list by default):

```python
from rankweave import fuse

bm25_results = [
    {"id": "doc-a", "score": 12.4, "text": "Introduction to machine learning"},
    {"id": "doc-b", "score": 8.1, "text": "Deep learning fundamentals"},
]

vector_results = [
    {"id": "doc-b", "score": 0.92, "text": "Deep learning fundamentals"},
    {"id": "doc-c", "score": 0.71, "text": "Transformer architectures"},
]

fused = fuse(
    [bm25_results, vector_results],
    method="weighted_score",
    score_field="score",
    weights=[0.3, 0.7],
    top_k=10,
)

for result in fused:
    print(result.document_id, result.score)
```



### Custom document IDs

```python
results = fuse(
    [product_ranking],
    id_field="product_id",
)

results = fuse(
    [product_ranking],
    id_field=lambda item: item["product"]["id"],
)
```



## API Reference



### `fuse(rankings, *, method, id_field, score_field, weights, k, normalize, top_k)`


| Parameter     | Default      | Description                                                                                                                                          |
| ------------- | ------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------- |
| `rankings`    | *(required)* | Iterable of ranked document lists. Order within each list represents rank (best first).                                                              |
| `method`      | `"rrf"`      | Fusion algorithm: `"rrf"`, `"weighted_rrf"`, or `"weighted_score"`.                                                                                  |
| `id_field`    | `"id"`       | Field name or callable used to identify documents across lists.                                                                                      |
| `score_field` | `None`       | Score field or callable. **Required** for `"weighted_score"`.                                                                                        |
| `weights`     | `None`       | Per-ranking weights. **Required** for `"weighted_rrf"` (`0 < w_i <= 1`, must sum to 1). Optional for `"weighted_score"`. Not supported by plain RRF. |
| `k`           | `60`         | RRF / weighted RRF smoothing constant.                                                                                                               |
| `normalize`   | `True`       | For `"weighted_score"`: min-max normalize each ranking to `[0, 1]` before weighting. Set `False` for raw weighted CombSUM.                           |
| `top_k`       | `None`       | Maximum number of results to return.                                                                                                                 |


**Returns:** A list of `FusedResult` objects, each with:

- `document` — the original document payload
- `document_id` — the document identifier
- `score` — the fused ranking score



## Architecture

```
rankweave/
├── api/          # Public fuse() entry point
├── core/         # Data models and FusionAlgorithm protocol
└── algorithms/   # Fusion implementations (RRF, weighted RRF, weighted score, …)
```



## Choosing an Algorithm



### Reciprocal Rank Fusion (RRF)

**Good fit:**

- Hybrid search (keyword + semantic) with incompatible score scales
- Multi-index RAG with different retrievers
- Ensemble retrieval where all sources should count equally and you trust ranks more than raw scores

**Not ideal:**

- When you need to weight one retriever more heavily than another — use weighted RRF
- When calibrated relevance scores matter more than rank position — use weighted score fusion



### Weighted Reciprocal Rank Fusion

**Good fit:**

- Same situations as RRF, but one retriever should count more (for example, `weights=[0.3, 0.7]`)
- You trust ranks more than raw scores, yet still want source emphasis

**Not ideal:**

- All retrievers should count equally — plain RRF is simpler
- Raw scores are meaningful and comparable (after optional normalization) — prefer weighted score fusion



### Weighted Score Fusion

**Good fit:**

- Retrievers produce meaningful scores you want to combine
- You want to emphasize one source (for example, `weights=[0.3, 0.7]`)
- Scores need a common scale — keep `normalize=True` (default) for min-max per list

**Not ideal:**

- Scores are on wildly different, untrusted scales and ranks are more reliable — prefer RRF or weighted RRF
- You only have ordered lists without scores



## Development

```bash
git clone https://github.com/GauravAgga/Riffusion.git
cd Riffusion
pip install -e ".[dev]"
pytest tests/ -v
```



## Contributing

Contributions are welcome. To add a new fusion algorithm:

1. Implement the `FusionAlgorithm` protocol in `src/rankweave/core/algorithm.py`
2. Register it in `src/rankweave/api/fusion.py`
3. Add tests in `tests/`

Please open an issue before large changes, and ensure all tests pass before submitting a pull request.

## Roadmap

- [ ] Publish to PyPI
- [x] Weighted score fusion
- [x] Weighted RRF
- [ ] CI with GitHub Actions
- [ ] Integration examples (LangChain, LlamaIndex)



## License

Licensed under the [Apache License 2.0](LICENSE).



