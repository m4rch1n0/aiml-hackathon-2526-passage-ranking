# AIML Hackathon 2526 — Passage Ranking Challenge

> **1st place** on the [Kaggle competition](https://www.kaggle.com/competitions/aiml-a-a-2025-26-end-of-year-hackathon) with a final **MAP@10 of 0.7268667247**, organized for the *Applicazioni Informatiche del Machine Learning* course at Sapienza University of Rome (a.y. 2025/26).

A two-stage neural passage ranking system on a sampled subset of MS MARCO. First-stage dense retrieval with a Bi-Encoder, second-stage reranking with an unweighted ensemble of three Cross-Encoders fused via Reciprocal Rank Fusion (RRF).

---

## Authors - Team **Pareto**

| Author | GitHub |
|---|---|
| Lorenzo Cerovaz | [@CerovazS](https://github.com/CerovazS) |
| Federico Forner | [@Fede2717](https://github.com/Fede2717) |
| Marco Galletti | [@m4rch1n0](https://github.com/m4rch1n0) |

All three authors contributed equally to design, implementation, experiments, and writing.

---

## Final Standings

We finished **1st with MAP@10 = 0.7268667247**.

The result is most striking when compared to the rest of the top 10:

|  | MAP@10 |
|---|---:|
| Our gap to the 2nd place | **0.00701** |
| Average gap between consecutive teams from 2nd to 10th | 0.00048 |
| Total MAP@10 spread covering all 9 places from 2nd to 10th | 0.00386 |

**Our single-step gap from 1st to 2nd is approximately 14.5× larger than the average gap between consecutive teams below us** — and, in absolute terms, **roughly 1.8× larger than the entire spread separating the 2nd from the 10th place team combined**. On a metric where the rest of the top 10 was tightly clustered within ~0.004 MAP@10, our solution stood apart by a margin nearly twice that width.

---

## Approach

We used a **two-stage retrieval pipeline**, the standard architecture for modern neural IR. The design balances efficiency (cheap Bi-Encoder over the full collection) and effectiveness (expensive Cross-Encoders only on candidates):

```
              ┌──────────────────────┐
   Query ──▶  │  Stage 1: Retrieval  │ ──▶  300 candidate passages
              │  Dual Encoder + FAISS│
              └──────────────────────┘
                         │
                         ▼
              ┌──────────────────────┐
              │  Stage 2: Reranking  │
              │  3× Cross-Encoders   │ ──▶  Top-10 ranked passages
              │  + RRF ensemble      │
              └──────────────────────┘
```

### Stage 1 — Dense Retrieval (Bi-Encoder)

- **Model:** `sentence-transformers/all-mpnet-base-v2` (768-dim embeddings)
- **Index:** FAISS `IndexFlatIP` (inner product on L2-normalized vectors)
- **Pooling:** mean pooling over token embeddings, attention-mask weighted
- **Cutoff:** top-K = **300** candidates per query

We selected K=300 by analyzing **Oracle Recall** — the fraction of queries for which the gold passage appears in the top-K. At K=300, Oracle Recall ≈ **0.99**: nearly all relevant passages are retrieved, and increasing K further only adds compute without coverage gain.

### Stage 2 — Cross-Encoder Ensemble

Three Cross-Encoders score each (query, candidate) pair jointly:

```
[CLS] query [SEP] passage [SEP] → encoder → relevance score
```

| Model | Validation MAP@10 |
|---|---:|
| `jinaai/jina-reranker-v2-base-multilingual` | 0.696 |
| `BAAI/bge-reranker-v2-m3` | 0.690 |
| `cross-encoder/ms-marco-MiniLM-L-12-v2` | 0.680 |
| ~~`castorini/monot5-base-msmarco-10k`~~ | 0.626 (excluded) |

MonoT5 underperformed and was dropped. BM25 (lexical baseline, MAP ≈ 0.45) was also too weak to contribute.

### Ensemble — Reciprocal Rank Fusion (RRF)

We fuse the three rerankers via RRF. For document *d* ranked at position *rᵢ(d)* by model *i*:

$$\text{RRF}(d) = \sum_{i=1}^{N} w_i \cdot \frac{1}{k + r_i(d)}$$

**Final configuration:** unweighted (*wᵢ* = 1) with smoothing **k = 1**.

---

## Key Finding: Simpler Generalizes Better

Our most counterintuitive result. We performed a grid search over: model subsets, weighted vs. unweighted RRF, smoothing *k* ∈ {0, 1, 2, 5, 10, …, 100}, and BM25 inclusion.

| Configuration | Models | Local Val MAP@10 | Public LB MAP@10 |
|---|:---:|---:|---:|
| Weighted (∝ MAP) + BM25 | 4 | **0.7052** | 0.7219 |
| **Unweighted Top-3 (final)** | 3 | 0.7032 | **0.7233** |

The weighted+BM25 configuration achieved a higher local validation score yet underperformed on the public leaderboard. The unweighted Top-3 configuration generalized better despite a slightly lower validation score. **Weight tuning on the validation set was overfitting**; equal votes across three strong, complementary rerankers preserved diversity and ultimately won.

The optimal *k = 1* heavily emphasizes top ranks: position 1 gets 1/2, position 10 gets 1/11 (a 5.5× spread). Higher *k* flattens the distribution and dilutes the signal.

---

## Reproducibility

The entire pipeline was executed on Kaggle using a free T4 GPU.

1. Open a new Kaggle notebook with **GPU enabled**.
2. Attach the competition data (`msmarco_sampled` dataset).
3. Upload [`notebook.ipynb`](./notebook.ipynb) and run all cells top-to-bottom.
4. Download `submission.csv` and submit.

Approximate runtime: **~10 hours** on a single T4 (the majority of which is spent on Cross-Encoder reranking of 300 candidates × 4912 queries × 3 models).

The notebook is self-contained: it installs dependencies, downloads the dataset, builds the FAISS index, runs the ensemble reranking, and writes a valid Kaggle submission file. Running it locally requires only installing the packages listed in [`requirements.txt`](./requirements.txt) and pointing the data paths at a local copy of the competition data.

---

## Repository Layout

```
.
├── README.md           # this file
├── LICENSE             # MIT
├── paper.pdf           # 2-page write-up of the method and experiments
├── notebook.ipynb      # the Kaggle submission notebook
├── requirements.txt    # pinned dependencies
└── .gitignore
```

The dataset is **not** included in this repository (MS MARCO sampled, distributed by the course organizers via the Kaggle competition).

---

## Paper

A 2-page write-up covering the method, ablation tables, and the simpler-generalizes-better finding is available in [`paper.pdf`](./paper.pdf).

> Cerovaz L., Forner F., Galletti M. *Two-Stage Neural Passage Ranking: An Ensemble Approach for MS MARCO.* AIML Hackathon 2526, Sapienza University of Rome, May 2026.

---

## Acknowledgments

We thank the *Applicazioni Informatiche del Machine Learning* course staff at Sapienza University of Rome for organizing the competition and curating the MS MARCO sampled split, and the open-source authors of MPNet, BGE, Jina, and MiniLM for releasing strong pretrained models.

---

## License

MIT — see [`LICENSE`](./LICENSE).
