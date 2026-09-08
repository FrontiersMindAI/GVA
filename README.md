# Grouped Value Attention (GVA)

The KV cache is a primary bottleneck for Transformer decoding: its memory footprint and cache-read traffic grow with sequence length. Grouped-query attention (GQA) reduces this cost by sharing key–value heads, but still stores both a key and a value at every step. We introduce Grouped Value Attention (GVA), which stores grouped values and reconstructs content keys with a learned linear map. At inference, the map can be absorbed into the query, eliminating the need to materialize content keys in the intended decode path. A small shared decoupled RoPE channel retains positional information through a separately cached positional key. For the configurations studied, this representation reduces persistent cache scalars by approximately 45–47% relative to matched GQA. At the 350M-parameter scale with 30B FineWeb-Edu tokens, the 16-dimensional positional variant reaches 44.18 average accuracy across five tasks, compared with 44.36 for GQA and 43.88 for MLA. These results demonstrate near-GQA benchmark accuracy with a more compact cache representation. To translate this compact representation into faster autoregressive inference, we have developed custom decoding kernels and are currently evaluating their end-to-end inference performance with an open-source release planned soon.



## Citation

``` bibtex
@article{tripathi2026gva,
  title={Grouped Value Attention: Efficient KV Caching via On-Demand Key Reconstruction},
  author={Tripathi, Vishesh and Kumar, Abhay and Khan, Ramsha},
  year={2026}
}
```

