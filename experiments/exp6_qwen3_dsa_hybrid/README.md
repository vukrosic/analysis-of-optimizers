# Experiment 6: Qwen3-Next Attention Variants

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

