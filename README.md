# Grouped Value Attention (GVA)

**Efficient KV Caching via On-Demand Key Reconstruction**

Grouped Value Attention (GVA) is a KV-cache architecture that stores
grouped **values** and reconstructs content keys with a learned linear
map:

\[ K = VM \]

At inference time, the map can be absorbed into the query:

\[ q(VM)\^T = (qM^T)V^T \]

so the full content-key cache does not need to be materialized.

## Why GVA?

Standard GQA caches both keys and values. GVA keeps the value cache and
uses it directly as the persistent content representation, reducing the
cache footprint.

With decoupled RoPE, the intended persistent cache is:

\[ N\_{`\mathrm{GVA}`{=tex}} = T G d_h + T d_r \]

compared with:

\[ N\_{`\mathrm{GQA}`{=tex}} = 2T G d_h \]

For the configurations studied, this corresponds to approximately
**45--47% fewer persistent cache scalars** than matched GQA.

## Architecture

GVA builds on grouped-query attention:

-   Query heads are divided into `G` value groups.
-   Each group stores a shared value representation.
-   Each query head has a learned reconstruction map `M`.
-   Content keys are represented implicitly as `V M`.
-   At decode time, `M` is absorbed into the query, avoiding
    materialization of content keys.
-   A small shared decoupled RoPE channel preserves positional
    information.

### Decode path

``` text
hidden state
     │
     ├──► Q ──► split into content / RoPE
     │             │
     │             └──► q̃ = q_content Mᵀ
     │
     ├──► V ──► grouped value cache
     │
     └──► positional projection ──► shared RoPE key cache

content scores = q̃ Vᵀ
position scores = q_rope k_ropeᵀ

attention = softmax(content + position)
```

## Results

Experiments were performed with decoder-only Transformers at
approximately **350M parameters**, trained from scratch on a **30B-token
FineWeb-Edu sample**.

Zero-shot accuracy was evaluated on HellaSwag, WinoGrande, OpenBookQA,
ARC-Easy, and ARC-Challenge.

  Method                          Avg. Accuracy
  ----------------------------- ---------------
  GQA                                     44.36
  MLA                                     43.88
  GVA baseline                            43.91
  GVA, scale-matched + Q-norm         **44.41**
  GVA + DRoPE (`d_r=24`)                  43.30
  GVA + DRoPE (`d_r=16`)                  44.18

The `d_r=16` decoupled-RoPE configuration is within **0.18 percentage
points** of GQA while using a substantially smaller intended cache
representation.

## Important: What this paper does *not* claim yet

The current work establishes the cache representation and small-scale
model-quality results, but **does not yet report end-to-end inference
speedups**.

Custom decoding kernels are under development/evaluation. The paper does
not currently provide:

-   fused decode throughput
-   tokens/sec benchmarks
-   peak serving-memory measurements
-   batch-capacity measurements
-   large-model scaling experiments
-   longer-context evaluations

Therefore, the reported **45--47% reduction is a representation-level
cache reduction**, not a measured 45--47% reduction in serving memory or
latency.

## Repository

This repository contains the research implementation and experiments for
Grouped Value Attention.

> **Status:** Research prototype / preprint. The inference-kernel and
> broader scaling evaluation are ongoing.

## Citation

``` bibtex
@article{tripathi2026gva,
  title={Grouped Value Attention: Efficient KV Caching via On-Demand Key Reconstruction},
  author={Tripathi, Vishesh and Kumar, Abhay and Khan, Ramsha},
  year={2026}
}
```

## References

-   Vaswani et al., *Attention Is All You Need*, NeurIPS 2017.
-   Shazeer, *Fast Transformer Decoding: One Write-Head is All You
    Need*, 2019.
-   Ainslie et al., *GQA: Training Generalized Multi-Query Transformer
    Models from Multi-Head Checkpoints*, EMNLP 2023.
-   DeepSeek-AI, *DeepSeek-V2: A Strong, Economical, and Efficient
    Mixture-of-Experts Language Model*, 2024.
