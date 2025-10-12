# Experiment 7: Gated DeltaNet Training (H100)

## Experimental Plan

**Phase 1: LR Ablation (200 steps)**
- 3 architectures: Full DeltaNet, Full Transformer, Hybrid Sparse (17% attn)
- 3 learning rates: 5e-4, 1e-3, 2e-3
- Total: 9 experiments

**Phase 2: Full Training (1000 steps)**
- Train each architecture with optimal LR from Phase 1
- Track: val loss, accuracy, perplexity

**Phase 3: Benchmarking**
- HellaSwag, ARC Challenge

---

## Commands

**LR Ablation (Phase 1)**
```bash
python run_lr_ablation_h100.py
```

**Train individual architecture**
```bash
python run_experiment.py --experiment h100_deltanet
python run_experiment.py --experiment h100_transformer
python run_experiment.py --experiment h100_hybrid_sparse
python run_experiment.py --experiment h100_hybrid_alternating
python run_experiment.py --experiment h100_hybrid_late
```

**Resume/extend training**
```bash
python run_experiment.py --experiment h100_deltanet --resume checkpoints_h100_deltanet/best_model.pt
python run_experiment.py --experiment h100_deltanet --resume checkpoints_h100_deltanet/best_model.pt --extend-steps 5000
```

**Compare & benchmark**
```bash
python compare_experiments.py
python ../../benchmarks/arc_challenge.py --checkpoint checkpoints_h100_deltanet/best_model.pt
```

## Architectures

- **h100_deltanet**: Pure DeltaNet (O(n))
- **h100_transformer**: Pure attention (O(n²))
- **h100_hybrid_sparse**: 4 attn layers [5,11,17,23] (17%)
- **h100_hybrid_alternating**: Every other layer attn (50%)
- **h100_hybrid_late**: Last 8 layers attn (33%)

## Config (H100)
1536d × 24L × 24H | seq=2048 | bs=48 | lr=1e-3
