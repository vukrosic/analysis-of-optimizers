# Experiment 6: Qwen3-Next Attention Variants

## Research Questions

1. Can replacing full attention with DeepSeek Sparse Attention (DSA) improve the efficiency and performance of a hybrid attention architecture that combines full attention and Gated DeltaNet (GDN)?

- Yes.

2. Which combination of attention mechanisms across layers produces the best efficiency-performance tradeoff: (1) Full Attention + GDN, (2) DSA + GDN, (3) DSA only, or (4) Full Attention only?

## Experiment Design

Compare 8 attention layer patterns across a 4-layer LLM architecture:

### Original Patterns (Full Attention + Linear Attention)
1. **Sandwich**: L → F → F → L
2. **Alternating**: F → L → F → L
3. **Linear First**: L → L → F → F
4. **Full First**: F → F → L → L

### DSA Patterns (DeepSeek Sparse Attention + Linear Attention)
5. **DSA Sandwich**: L → D → D → L
6. **DSA Alternating**: D → L → D → L
7. **DSA Linear First**: L → L → D → D
8. **DSA Full First**: D → D → L → L

**Legend**: L = Linear Attention (Gated DeltaNet), F = Full Attention, D = DeepSeek Sparse Attention

## Usage

```bash
pip install -r requirements.txt

# Run comprehensive experiment (trains all 8 patterns, ~4 minutes)
python experiments/exp6_qwen3_dsa_hybrid/run_experiment.py
```

## Files

- `run_experiment.py` - Main experiment script (trains and compares all 8 patterns)
- `models.py` - Enhanced model supporting all 3 attention types (F, L, D)
- `config.py` - Configuration options (SMALL/MEDIUM/LARGE)
- `visualize_results.py` - Results visualization
- `results/` - Experiment outputs (JSON results + PNG plots)

