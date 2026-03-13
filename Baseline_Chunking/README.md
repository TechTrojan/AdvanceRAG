# 🔍 Advanced RAG Optimization — Hybrid Retrieval + CrossEncoder Reranking

This repository contains an advanced Retrieval-Augmented Generation (RAG) experiment focused on improving retrieval quality using:

* ✅ Hybrid Search (FAISS + BM25)
* ✅ Weighted Ensemble Retrieval
* ✅ CrossEncoder-based Reranking

Branch: `hybrid_retrieval`
Core Logic: `Baseline_Chunking/src/DataRetriever.py`

---

# 🎯 Objective

Improve RAG performance for structured financial documents by:

1. Increasing recall using hybrid search
2. Improving ranking quality using reranking
3. Reducing hallucination risk
4. Enhancing answer relevance

---

# 🏗 Architecture Overview

### Step 1 — Vector Retrieval (Semantic Search)

* FAISS vector store
* SentenceTransformer embeddings
* Captures semantic similarity

### Step 2 — Keyword Retrieval

* BM25Retriever
* Strong exact-match handling
* Effective for numeric-heavy documents

### Step 3 — Hybrid Fusion

* EnsembleRetriever
* Weighted merging of FAISS + BM25
* Balances semantic and lexical relevance

### Step 4 — CrossEncoder Reranking

* CrossEncoderReranker
* Re-ranks top retrieved chunks
* Improves ranking precision
* Reduces irrelevant context

![Architecture Diagram](/Baseline_Chunking/asset/images/Architect_hybrid_rerank.png)

---

# ⚙️ Retrieval Pipeline (DataRetriever.py)

High-level logic:

```text
Query
  ↓
FAISS Retrieval
  ↓
BM25 Retrieval
  ↓
Ensemble Fusion
  ↓
CrossEncoder Reranking
  ↓
Top-K Context
  ↓
LLM Answer Generation
```

This layered retrieval strategy ensures:

* High recall (hybrid search)
* High precision (reranking)
* Better grounding

---

# 🧠 Why Hybrid + Reranking?

### Hybrid Search Alone:

✔ Improves recall
⚠ May introduce some noisy results

### Reranking Layer:

✔ Filters noise
✔ Improves ordering quality
✔ Boosts context precision
✔ Improves answer alignment

Together, they create a more production-grade retrieval stack.

---

## 📊 Evaluation Results — Hybrid + Reranking

| Metric | Score |
|--------|-------|
| Context Adherence | 90.48% |
| Context Precision | 37.14% |
| Answer Relevance | 77.06% |
| Groundedness | 80.95% |

---

## 🔎 Key Improvements

### ✅ Highest Context Adherence (90.48%)
The model relies more consistently on retrieved context rather than generating unsupported information.

### ✅ Highest Answer Relevance (77.06%)
Combining semantic similarity (FAISS) with keyword matching (BM25), then applying CrossEncoder reranking ensures the most relevant chunks appear first.

### ✅ Highest Groundedness (80.95%)
Factual claims are more consistently traceable to retrieved context, reducing hallucination risk and improving numeric extraction reliability.

---

## ⚖ Observed Tradeoff

Context Precision decreased (37.14%) due to increased recall from hybrid retrieval. 

This is expected in hybrid systems:
- More documents retrieved
- Broader coverage
- Slight increase in irrelevant candidates

However, groundedness and answer relevance improved — which is more critical for enterprise RAG systems.


# 🔬 Key Design Decisions

| Component            | Purpose                     |
| -------------------- | --------------------------- |
| FAISS                | Semantic similarity         |
| BM25                 | Exact keyword matching      |
| EnsembleRetriever    | Weighted hybrid fusion      |
| CrossEncoderReranker | Precision-focused reranking |
| Top-K Filtering      | Control context size        |

---

# 🚀 Why This Matters for Agents

In Agentic AI systems:

* Retrieval directly impacts reasoning quality
* Poor retrieval → flawed tool calls
* Noisy context → hallucinated intermediate steps

Hybrid + reranking enables:

* Cleaner context for reasoning
* Better multi-step planning
* More reliable financial fact extraction
* Higher trust in production systems

---

# 📂 Repository Structure

```
Baseline_Chunking/
└── src/
    ├── DataRetriever.py      ← Hybrid + Reranking logic
    ├── Ingestion.py
    ├── Evaluation.py
    └── ...
```

---

# 🛠 Technologies Used

* LangChain
* FAISS
* BM25Retriever
* EnsembleRetriever
* CrossEncoderReranker
* OpenAI LLM
* SentenceTransformers

---

# 🔗 Repository

Hybrid retrieval branch:

https://github.com/TechTrojan/AdvanceRAG/tree/hybrid_retrieval


---

## 👨‍💻 Author

Experiment conducted to explore production-grade retrieval strategies for structured enterprise documents using Hybrid Search and Reranking.

---
