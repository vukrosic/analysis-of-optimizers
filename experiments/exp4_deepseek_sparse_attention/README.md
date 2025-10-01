# Experiment 4: DeepSeek Sparse vs Classic Attention

## Overview

This experiment compares **DeepSeek Sparse Attention** (from the DeepSeek-V3 paper) with **Classic Dense Attention** across different sequence lengths.

## What's Being Compared

1. **Classic Attention**: Standard dense multi-head attention (O(L²) complexity)
2. **Sparse Attention (DSA)**: DeepSeek's sparse attention with lightning indexer (O(Lk) complexity)

## Key Features

- **Lightning Indexer**: Selects top-k most relevant tokens using lightweight indexer heads
- **Top-k Selection**: Only attends to ~50% of tokens (reduces computation)
- **Sequence Length Scaling**: Tests how both approaches scale with longer sequences

## Running the Experiment

```bash
cd experiments/exp4_deepseek_sparse_attention
python run_experiment.py
```

## Results

Results are saved to `results/`:
- `sequence_length_comparison.png` - Main comparison plot across all sequence lengths
- `summary.json` - Numerical results summary
- `seq_*/` - Detailed results for each sequence length

## Configuration

Edit `SEQUENCE_LENGTHS` and `BASE_CONFIG` in `run_experiment.py` to customize:
- Sequence lengths to test (default: 64, 128, 256)
- Model size, training steps, etc.

## Files

- `run_experiment.py` - Main experiment script
- `exp4_models.py` - Model definitions (sparse and classic)
- `sparse_attention.py` - DeepSeek sparse attention implementation
- `config.py` - Configuration helpers

