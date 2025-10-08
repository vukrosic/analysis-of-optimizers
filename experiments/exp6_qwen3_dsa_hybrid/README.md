# Experiment 6: Qwen3-Next Attention Variants

## Research Questions

1. Can replacing full attention with DeepSeek Sparse Attention (DSA) improve the efficiency and performance of a hybrid attention architecture that combines full attention and Gated DeltaNet (GDN)?

2. Which combination of attention mechanisms across layers produces the best efficiency-performance tradeoff: (1) Full Attention + GDN, (2) DSA + GDN, (3) DSA only, or (4) Full Attention only?

## Hypothesis

We hypothesize that:
1. **DSA-Only variant** will achieve better computational efficiency than the baseline by reducing quadratic complexity across all attention layers, but may sacrifice some model quality by replacing the specialized GDN mechanism.
2. **Hybrid variant** will provide the best balance, applying DSA to full attention layers (where sparsity is most beneficial) while preserving GDN's linear complexity and specialized design for linear attention layers.
3. The hybrid approach will demonstrate that different attention mechanisms (full attention, DSA, and GDN) can be strategically combined within a single architecture to optimize for both efficiency and performance.

## Experiment Design

Compare 3 attention configurations:

1. **Baseline**: Standard Qwen3-Next (full attention + gated deltanet)
2. **DSA-Only**: All layers use DeepSeek Sparse Attention  
3. **Hybrid**: DSA for full attention, gated deltanet for linear attention

## Usage

```bash
# Test models work
python test_models.py

# Run experiment (trains all 3 variants)
python run_experiment.py

# Visualize results
python visualize_results.py
```

## Files

- `config.py` - 3 configs (SMALL/MEDIUM/LARGE)
- `models.py` - 3 model variants (imports from existing code)
- `run_experiment.py` - Training script
- `test_models.py` - Quick test
- `visualize_results.py` - Results plots

