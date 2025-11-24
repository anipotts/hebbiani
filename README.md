# Hebbiani

> A local, research-grade playground for **hybrid retrieval**, **cross-encoder reranking**, and **late-interaction token matching** — inspired by Hebbia’s Matrix, built to learn how real production retrieval systems work.

---

## 1. Motivation

**Hebbiani** is a personal, open-source attempt to recreate (in miniature) the kinds of ranking systems used by tools like **Hebbia Matrix**:

- Hybrid retrieval (BM25 + dense embeddings)
- Cross-encoder rerankers for high-precision top‑K
- Late-interaction / token-level matching for long‑tail, nuanced queries
- Dynamic orchestration that picks _which_ reranker to use, _how many_ candidates to score, and _how_ to balance **latency vs accuracy vs cost**

The goals of this project:

1. **Learn by doing**: Implement the core ideas behind modern enterprise RAG / retrieval systems, not just call a hosted API.
2. **Experiment like a researcher**: Run ablations and benchmarks (BM25 vs dense vs cross-encoder vs late interaction) on real document corpora.
3. **Stand out as a candidate**: Have a concrete repo that shows familiarity with the same concepts and trade-offs real SWE/ML engineers at Hebbia (and similar companies) think about daily.

This is **not** meant to be a clone of Hebbia or Matrix; it’s a small, opinionated lab where you can:

- Load your own document corpus (PDFs, text reports, contracts)
- Ask complex questions
- Watch how different retrieval/ranking strategies behave
- Measure quality and latency trade-offs

---

## 2. High-Level Overview

At a high level, Hebbiani is a **multi-stage ranking pipeline**:

1. **Ingest** documents → chunk them → store metadata.
2. **Index** the chunks with:

   - a **sparse index** (BM25)
   - a **dense index** (bi-encoder embeddings + FAISS)

3. **Retrieve** candidates via:

   - BM25
   - dense vector search
   - a hybrid combination of both

4. **Rerank** those candidates using one or more of:

   - **Cross-encoder reranker** (query + document encoded jointly)
   - **Late-interaction style scorer** (token-level similarity aggregation)
   - (optional) **LLM reranker stub** for later extension

5. **Orchestrate** which reranker(s) to use based on:

   - query type (short keyword vs long natural-language question)
   - cost budget (low/medium/high)
   - experimental configuration

6. **Evaluate & log** metrics like nDCG@k, MRR, precision@k, along with latency breakdowns for each stage.

The whole thing runs locally on a MacBook (CPU by default, GPU if available).

---

## 3. Learning Objectives

Hebbiani is designed so that building and extending it will teach you:

- The **differences between**:

  - BM25 / sparse retrieval
  - Bi-encoder dense retrieval
  - Cross-encoder reranking
  - Late-interaction ranking (ColBERT-style ideas)

- How to build a **hybrid search pipeline** that uses multiple signals.
- How to implement a **dynamic orchestrator** that switches strategies based on query complexity and cost.
- How to **benchmark** ranking systems using classical IR metrics.
- How to make real **engineering trade-offs** between latency, quality, and resource usage.

By the end, you should be able to walk into a system-design or ML-for-search interview and comfortably discuss:

- Why retrieval is usually **two-stage (retrieve → rerank)**.
- Why cross-encoders are **more accurate but slower**.
- Why late-interaction models are powerful for **long-tail, nuanced queries**.
- How to design and reason about **production-like retrieval stacks**.

---

## 4. Architecture

### 4.1 Components

- **Document Store (SQLite)**

  - Stores chunks with `doc_id`, `title`, `chunk_index`, `text`, and optional labels.

- **BM25 Index**

  - Built from tokenized chunk text using `rank-bm25` or similar.

- **Dense Embedding Index (FAISS)**

  - Uses a bi-encoder from `sentence-transformers` (e.g. `all-MiniLM-L6-v2`).

- **Cross-Encoder Reranker**

  - Uses a cross-encoder from `sentence-transformers` (e.g. `cross-encoder/ms-marco-MiniLM-L-6-v2`).

- **Late-Interaction Scorer (Toy ColBERT-ish)**

  - Computes token-level similarities between query and document embeddings and aggregates them.

- **Orchestrator**

  - Analyzes query type and picks which retrieval + reranking combo to run.

- **Experiment Runner**

  - Runs predefined experiments, logs metrics, and produces comparison tables.

- **Interface(s)**

  - Minimal CLI for ad-hoc querying.
  - Optional FastAPI endpoint for local playground usage.

### 4.2 Data Flow

```text
            ┌─────────────────────┐
            │   Raw Documents     │
            │  (PDF / TXT / etc)  │
            └─────────┬───────────┘
                      │  Ingestion + Chunking
                      v
           ┌───────────────────────┐
           │    SQLite: chunks     │
           │  (id, doc_id, text)   │
           └─────────┬─────────────┘
        Build BM25   │    Build Embeddings + FAISS
                      v
        ┌──────────────────┐    ┌──────────────────┐
        │   BM25 Index     │    │  FAISS Index     │
        └────────┬─────────┘    └────────┬─────────┘
                 │  initial retrieval     │
                 └────────┬──────────────┘
                          v
                   ┌────────────┐
                   │ Candidates │ (top-K)
                   └─────┬──────┘
                         │
           ┌──────────────┴─────────────────┐
           v                                v
┌───────────────────────┐        ┌───────────────────────┐
│ Cross-Encoder Rerank  │        │ Late-Interaction      │
│  (query, doc) → score │        │ token-level scoring   │
└───────────────────────┘        └───────────────────────┘
           └──────────────┬─────────────────┘
                          v
                   ┌────────────┐
                   │  Final Top │
                   │   Results  │
                   └────────────┘
```

---

## 5. Repository Structure (Planned)

```text
hebbiani/
  README.md
  pyproject.toml / requirements.txt
  hebbiani/
    __init__.py
    config/
      default.yaml
    data/
      db.sqlite3            # created at runtime
      indexes/
        bm25.pkl
        faiss.index
        embeddings.npy
        ids.npy
    ingest/
      ingest.py             # load & chunk documents
    index/
      build_indexes.py
      bm25_index.py
      dense_index.py
    retrieval/
      orchestrator.py       # hybrid retrieval + reranking selection
      cross_encoder.py
      late_interaction.py
    eval/
      metrics.py            # nDCG, MRR, etc.
      experiments.py        # experiment runner
    interfaces/
      cli.py
      api.py                # optional FastAPI endpoint
  notebooks/
    01_quickstart.ipynb
    02_cross_encoder_vs_dense.ipynb
    03_late_interaction_experiments.ipynb
  data/
    raw_docs/               # user-provided PDFs / TXTs
    labels/                 # optional relevance judgments for eval
```

This is aspirational — you don’t need everything on day one. The project can start with just `ingest.py`, `build_indexes.py`, `orchestrator.py`, and `cli.py`, then grow into this structure.

---

## 6. Setup & Getting Started

### 6.1 Prerequisites

- Python 3.10+
- macOS (tested on MacBook Pro; should also work on Linux)
- `virtualenv` or `uv`/`poetry` if you prefer.

### 6.2 Install

```bash
git clone https://github.com/<your-username>/hebbiani.git
cd hebbiani
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Dependencies (rough sketch):

- `sentence-transformers`
- `faiss-cpu`
- `rank-bm25`
- `pypdf`
- `numpy`, `scipy`
- `fastapi`, `uvicorn` (optional for API)

### 6.3 Ingest Documents

Place some PDFs or TXTs into `data/raw_docs/`:

```bash
mkdir -p data/raw_docs
cp ~/Documents/some_reports/*.pdf data/raw_docs/
```

Run ingestion:

```bash
python -m hebbiani.ingest.ingest
```

This will:

- Extract text
- Chunk into overlapping segments
- Store chunks in `data/db.sqlite3`

### 6.4 Build Indexes

```bash
python -m hebbiani.index.build_indexes
```

This will:

- Train BM25 over chunk tokens
- Encode all chunks with the bi-encoder
- Build a FAISS index
- Save everything to `data/indexes/`

### 6.5 Query via CLI

```bash
python -m hebbiani.interfaces.cli
```

Example interaction:

```text
Query (blank to quit): what are the main termination conditions in these contracts?

[1] score=0.842
...chunk text...
------------------------------------------------------------
[2] score=0.808
...chunk text...
------------------------------------------------------------
...
```

Under the hood, the CLI will call the orchestrator, which will:

- Analyze the query type
- Run hybrid retrieval (BM25 + dense)
- Rerank with the cross-encoder (and optionally late-interaction)

---

## 7. Core Concepts Implemented

### 7.1 BM25 (Sparse Retrieval)

- Classical inverted index-based retrieval.
- Great for:

  - strict keyword matching
  - exact term presence (e.g., legal clause names, tickers)

- Weak for:

  - semantic similarity
  - paraphrased queries

Hebbiani uses BM25 to:

- Provide strong lexical grounding for queries.
- Complement dense retrieval in hybrid scoring.

### 7.2 Bi-Encoder Dense Retrieval

- Sentence-transformer bi-encoder (e.g., `all-MiniLM-L6-v2`).
- Encodes query and document separately into vectors.
- Similarity = dot product or cosine similarity.

Pros:

- Very fast at scale with FAISS.
- Good semantic recall.

Cons:

- Limited token-level interaction (query and doc don’t see each other during encoding).

### 7.3 Cross-Encoder Reranking

- Cross-encoder model takes `(query, document)` as a single input sequence.
- Can attend across both texts simultaneously.

Pros:

- Very accurate relevance scoring.
- Great for top‑10 selection in legal/finance-style queries.

Cons:

- Much slower than bi-encoders.
- Only used on a small candidate set (e.g., top‑100 from BM25 + dense).

In Hebbiani, the cross-encoder is the **primary high-precision reranker**.

### 7.4 Late-Interaction Token Matching (Toy ColBERT-Style)

- Instead of producing a single embedding per document, we:

  - compute embeddings per token (or sub-token chunk)
  - for each query token, find the max similarity over document tokens
  - aggregate these maxima into a relevance score

This approximates late-interaction models like ColBERT:

- Better at capturing phrase-level and token-level matches.
- Strong for long-tail, nuanced queries.

In Hebbiani, this is implemented as a **research-focused reranker** to:

- Compare dense global embeddings vs token-wise matching.
- See how it changes ranking and metrics.

---

## 8. Orchestrator & Dynamic Strategy Selection

A key idea of Hebbiani is to not use a single fixed strategy, but to **choose a retrieval/rerank pipeline based on the query and cost budget**.

### 8.1 Query Analysis

A simple rule-based analyzer might compute:

- Query length (short vs long)
- Presence of wh-words (`what`, `why`, `how`, etc.)
- Presence of obvious keywords (tickers, clause names, section numbers)

Based on this, it classifies a query as e.g.:

- `keyword_lookup`
- `short_question`
- `complex_question`

### 8.2 Cost Budgets

The orchestrator exposes cost presets like:

- `low` → BM25 + dense only, no cross-encoder
- `medium` → hybrid retrieval + cross-encoder on fewer candidates
- `high` → hybrid retrieval + cross-encoder + late-interaction reranker (or deeper cascades)

### 8.3 Example Strategy Table

| Query Type       | Cost Budget | Retrieval    | Rerankers                        |
| ---------------- | ----------: | ------------ | -------------------------------- |
| keyword_lookup   |         low | BM25 only    | none                             |
| keyword_lookup   |      medium | BM25 + dense | cross-encoder on top‑30          |
| short_question   |      medium | BM25 + dense | cross-encoder on top‑50          |
| complex_question |        high | BM25 + dense | cross-encoder → late-interaction |

This mimics how a production system might adapt based on **user intent + resource constraints**.

---

## 9. Evaluation & Experiments

A central goal is to treat Hebbiani as a **mini research environment**.

### 9.1 Metrics

Implement standard ranking metrics:

- **nDCG@k** (normalized discounted cumulative gain)
- **MRR@k** (mean reciprocal rank)
- **Precision@k**

Given a small labeled dataset of queries and relevant chunk IDs, you can:

- Compare pipelines: `BM25`, `dense`, `BM25+dense`, `BM25+dense+cross-encoder`, `BM25+dense+cross-encoder+late-interaction`.
- Track latency for each stage.

### 9.2 Example Experiments

1. **Cross-Encoder Ablation**

   - Setup: Evaluate `dense-only` vs `dense+cross-encoder`.
   - Expectation: cross-encoder improves nDCG/MRR on complex queries at the cost of latency.

2. **Late-Interaction Ablation**

   - Setup: Compare `cross-encoder-only` vs `cross-encoder+late-interaction` on long-tail legal/finance queries.
   - Goal: understand whether token-level matching helps on subtle phrase differences.

3. **BM25 vs Dense vs Hybrid**

   - Setup: Evaluate each retrieval strategy.
   - Observation: BM25 is strong for exact terms, dense for semantic paraphrases, hybrid typically best overall.

4. **Candidate Pool Size Study**

   - Vary the number of candidates passed from retrieval → reranker (e.g., 20, 50, 100).
   - See how quality vs latency changes.

Each experiment can be defined as a small config and run via `experiments.py`.

---

## 10. Possible Extensions / Roadmap

Some ideas to extend Hebbiani over time:

1. **Dataset Integrations**

   - Add loaders for public corpora (e.g., subsets of legal, finance, or QA datasets).

2. **LLM Reranker Plugin**

   - Abstract a reranker interface and optionally plug in an external LLM reranker (e.g., OpenAI API) for comparison.

3. **UI Layer**

   - Small Next.js frontend that talks to the FastAPI backend.
   - Show:

     - retrieved chunks
     - scores per reranker
     - which pipeline path was used

4. **Multi-Agent Orchestration**

   - Add a simple agent layer:

     - Planner: decides which pipeline to use
     - Retriever: runs orchestrator
     - Analyzer: summarizes top chunks
     - Critic: checks consistency across top chunks

5. **Multi-Modal Experiments**

   - Extend ingestion to handle images+captions, tables, etc.

6. **Index Introspection Tools**

   - Visualize nearest neighbors for a given query.
   - Explore embeddings and BM25 scores side-by-side.

---

## 11. How to Talk About Hebbiani in Interviews

This project is explicitly designed to map to **real interview talking points**. Building it gives you honest, non-hand-wavy things to say like:

> I built a local multi-stage ranking system inspired by Hebbia’s Matrix. It combines BM25, dense retrieval, cross-encoder reranking, and a late-interaction token matcher behind a dynamic orchestrator that chooses retrieval and reranking strategies based on query complexity and a cost budget. I used standard IR metrics to run ablations comparing different pipelines and latency/quality trade-offs.

You can credibly discuss:

- Why production systems use **two-stage retrieval**.
- Why **cross-encoders** are reserved for top‑K reranking.
- How **late-interaction models** approximate token-level attention while remaining scalable.
- How to design an **orchestrator** that treats cost/latency as first-class concerns.
- How you measured and compared different pipelines.

This repo is meant to be a living notebook of your understanding of modern retrieval and ranking — and a very visible proof that you’re not just calling APIs, but actually understanding the architecture beneath them.

---

## 12. Status

This README is an architectural and learning blueprint. The implementation can be built incrementally:

1. Minimal pipeline: ingestion → BM25 + dense → cross-encoder → CLI.
2. Add evaluation + metrics.
3. Add late-interaction scorer.
4. Add orchestrator logic & experiment runner.
5. (Optional) Add API + UI.

Even at stage (1)–(2), Hebbiani is already a strong portfolio project. The later stages turn it into a serious, research-flavored playground that mirrors what cutting-edge AI search teams work on day to day.
