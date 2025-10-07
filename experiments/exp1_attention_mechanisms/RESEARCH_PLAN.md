Goal is to release LLM for $100

1. Make sure experiments are easily run on more compute
2. Keep them up to date? Run repeatedly with improvements?
3. Should I implement these architectures into main files or exps?

We will test:
1. standard attention
2. DeepSeek MLA
3. DeepSeek Sparse Attention
4. Stanadard + Gated Delta Net Attention (Qwen3-Next, FLA)
- Test different mixtures
5. GDN + MLA?
6. Mamba2 or other FLA with MLA or GQA?


Why does Qwen3-Next do this:
1. Adopt the output gating mechanism from our prior work [2] to reduce low-rank issues in attention.
2. Increase the dimension per attention head from 128 to 256.
3. Apply rotary position encoding only to the first 25% of position dimensions — improving extrapolation to longer sequences.


# MoE





Note:
- The efficiency or throughput improvement depends highly on the implementation.





Do we need to try different random seeds? That would 3x number of experiments.

# Understanding Transformer
1. "Attention mechanism dynamically forms short-term, contextual associations,
while FFNs encode persistent, long-term associative memories distilled during training." - Zhong, S. et al. (2025). Understanding Transformer from the Perspective of Associative Memory. ByteDance Seed.