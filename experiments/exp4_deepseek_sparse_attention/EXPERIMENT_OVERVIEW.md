# Experiment 4: DeepSeek Sparse vs Classic Attention - Quick Overview

## What This Experiment Does

Compares **DeepSeek Sparse Attention** (from DeepSeek-V3 paper) vs **Classic Dense Attention** across different sequence lengths (64, 128, 256).

## Files Structure (Minimal & Clean)

```
exp4_deepseek_sparse_attention/
├── run_experiment.py        # Main experiment script (run this!)
├── exp4_models.py           # Model definitions (sparse & classic)
├── sparse_attention.py      # DeepSeek sparse attention implementation
├── config.py                # Configuration helpers
├── README.md                # Documentation
└── results/                 # All results saved here
    ├── sequence_length_comparison.png  # Main comparison plot ⭐
    ├── summary.json                    # Numerical summary
    ├── RESULTS_SUMMARY.md              # Results explained
    ├── seq_64/                         # Results for L=64
    ├── seq_128/                        # Results for L=128
    └── seq_256/                        # Results for L=256
```

## Running the Experiment

```bash
cd experiments/exp4_deepseek_sparse_attention
python run_experiment.py
```

Takes ~10-15 minutes on GPU.

## Quick Results

| Metric           | Classic (Dense) | Sparse (DSA) | Winner      |
|------------------|-----------------|--------------|-------------|
| Best Val Loss    | 7.05            | **2.49**     | ✅ Sparse   |
| Best Accuracy    | 11.8%           | **61.0%**    | ✅ Sparse   |
| Speed (sec/step) | 0.063           | 0.067        | ~Similar    |

**Key Finding**: DeepSeek sparse attention dramatically outperforms classic attention while maintaining similar training speed.

## What's Implemented from the Paper

1. **Lightning Indexer** - Lightweight attention indexer with few heads
2. **Top-k Token Selection** - Selects most relevant tokens (50% sparsity)
3. **Sparse Attention** - Attends only to selected tokens

All based on DeepSeek-V3.2-Exp paper specifications.

## Customization

Edit `run_experiment.py`:
- `SEQUENCE_LENGTHS = [64, 128, 256]` - Which sequence lengths to test
- `BASE_CONFIG` - Model size, training steps, etc.

## Key Difference from Classic Attention

- **Classic**: Full O(L²) attention matrix computation
- **Sparse**: O(Lk) where k < L, using learned token selection

This reduces computation while improving performance!

