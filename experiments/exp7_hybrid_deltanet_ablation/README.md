# Experiment 7: Hybrid DeltaNet Architecture Ablation (H100)

## Overview
Comprehensive ablation study testing 13 architectures across the full spectrum of DeltaNet/Attention mixtures (0% to 100% attention) to find the optimal hybrid ratio for language modeling.

## Experimental Phases

**Phase 1: LR Ablation (200 steps)**
- 3 architectures: Full DeltaNet, Full Transformer, Hybrid Sparse (17% attn)
- 3 learning rates: 5e-4, 1e-3, 2e-3
- Total: 9 experiments
- **Results**: DeltaNet prefers 1e-3, Hybrids/Transformer prefer 2e-3

**Phase 2: Architecture Mixture Ablation**
- 13 architectures: 0%, 8%, 17%, 25%, 33%, 42%, 50%, 58%, 67%, 75%, 83%, 92%, 100% attention
- Each trained with optimal LR from Phase 1
- **Goal**: Identify the sweet spot for DeltaNet/Attention mixture

**Phase 3: Full Training (1000 steps)**
- Train winner architecture with optimal hyperparameters

**Phase 4: Benchmarking**
- HellaSwag, ARC Challenge evaluation

---

## Commands

**LR Ablation (Phase 1)**
```bash
python run_lr_ablation_h100.py
```

**Architecture Mixture Ablation (Phase 2)**
```bash
python run_full_architecture_comparison.py
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

## Architectures Tested

### Pure Architectures
- **h100_deltanet (0%)**: Pure DeltaNet - O(n) complexity
- **h100_transformer (100%)**: Pure attention - O(n²) complexity

### Hybrid Architectures (DeltaNet + Attention Mix)
- **h100_hybrid_8 (8%)**: 1/12 layers attention (last layer only)
- **h100_hybrid_sparse (17%)**: 2/12 layers attention [5, 11]
- **h100_hybrid_25 (25%)**: 3/12 layers attention [4, 8, 11]
- **h100_hybrid_late (33%)**: 4/12 layers attention [8, 9, 10, 11]
- **h100_hybrid_42 (42%)**: 5/12 layers attention [2, 4, 6, 8, 11]
- **h100_hybrid_alternating (50%)**: 6/12 layers attention (every other)
- **h100_hybrid_58 (58%)**: 7/12 layers attention
- **h100_hybrid_67 (67%)**: 8/12 layers attention
- **h100_hybrid_75 (75%)**: 9/12 layers attention
- **h100_hybrid_83 (83%)**: 10/12 layers attention
- **h100_hybrid_92 (92%)**: 11/12 layers attention (all but first)

## Model Configuration (H100)
- **Base**: 768d × 12L × 12H (~188M-302M params depending on architecture)
- **Sequence length**: 2048
- **Batch size**: 48
- **Learning rates**: 
  - Pure DeltaNet: 1e-3
  - Hybrids/Transformer: 2e-3

## Key Findings
- Pure DeltaNet prefers lower LR (1e-3)
- Any attention prefers higher LR (2e-3)
- Hybrid Sparse (17%) won initial ablation - testing full spectrum to confirm optimality
