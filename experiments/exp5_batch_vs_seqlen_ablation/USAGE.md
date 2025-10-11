# MoE Ablation Study - Quick Guide

## 🎯 Simple Approach (Recommended)

### 1. List available configs
```bash
python run_ablation.py              # Shows all available configs
```

### 2. Run a config
```bash
python run_ablation.py large_batch_lr003
```

### Available configs:
```bash
# LR = 0.01 configs
python run_ablation.py large_batch_lr001    # batch=104, seqlen=256
python run_ablation.py long_seq_lr001       # batch=6, seqlen=4096
python run_ablation.py balanced_lr001       # batch=26, seqlen=1024

# LR = 0.02 configs  
python run_ablation.py large_batch_lr002
python run_ablation.py long_seq_lr002
python run_ablation.py balanced_lr002

# LR = 0.03 configs
python run_ablation.py large_batch_lr003
python run_ablation.py long_seq_lr003
python run_ablation.py balanced_lr003

# LR = 0.04 configs
python run_ablation.py large_batch_lr004
python run_ablation.py long_seq_lr004
python run_ablation.py balanced_lr004

# Extended 1000-step runs (LR=0.03)
python run_ablation.py large_batch_lr003_1000steps
python run_ablation.py long_seq_lr003_1000steps
python run_ablation.py balanced_lr003_1000steps
```

### 3. Create custom config
Edit `configs_ablation.py` and add your config:
```python
MY_CUSTOM = AblationConfig(
    name="my_custom",
    batch_size=32,      # ← Edit to max out GPU
    seq_len=384,        # ← Adjust based on memory
    lr=0.03,
    grad_accum=2,
    max_steps=100
)
```

Then add it to the `CONFIGS` registry at the bottom:
```python
CONFIGS = {
    # ... existing configs ...
    'my_custom': MY_CUSTOM,  # ← Add this line
}
```

Run it:
```bash
python run_ablation.py my_custom
```

## 📊 Plot Results
```bash
python plot_part1.py        # Part 1: LR sweep results
python plot_part2.py        # Part 2: Extended training results
python plot_val_vs_time_tokens.py  # Validation curves
```

## 💡 Workflow for Finding Max Memory

1. **Start with a small config:**
   ```bash
   python run_ablation.py large_batch_lr003
   ```

2. **Create your own config in `configs_ablation.py`:**
   - Copy one of the existing configs
   - Increase `batch_size` or `seq_len`
   - Add it to the `CONFIGS` dictionary

3. **Test your config:**
   ```bash
   python run_ablation.py my_custom
   ```

4. **Check memory usage in output** (shows peak memory)

5. **Repeat until OOM, then back off slightly**

## 🔧 Model Config

Edit `config.py` to change MoE settings:
- `num_experts`: Number of experts (default: 8)
- `expert_top_k`: Experts per token (default: 2)
- `load_balancing_weight`: Load balancing loss weight (default: 0.01)

---

## Old CLI Method (still works)

<details>
<summary>Click to expand</summary>

```bash
python ablation_batch_vs_seqlen.py --batch 32 --seqlen 384 --lr 0.015
```

All args: `--batch`, `--seqlen`, `--lr`, `--grad-accum`, `--steps`, `--name`

</details>
