# Analysis of Optimizers - Bachelor's Thesis Project

**Vuk Rosić**

## About This Thesis

This repository contains the research and experiments for my master's thesis, which compares the **Muon optimizer** and **Adam optimizer** for training large language models and neural networks. The thesis examines novel optimizers' design philosophy and breaks down their behavior through systematic ablations.

### Research Objectives

Through methodical hyperparameter optimization across more than 45 tests, this research:

- **Identifies optimal configurations** for both Muon and Adam optimizers
- **Reveals fundamental design principles** about the interaction between learning rates, momentum, and second-order curvature information
- **Analyzes Newton-Schulz iteration efficiency** for computational optimization
- **Develops key design concepts** beyond empirical comparison, highlighting the significance of gradient orthogonalization and unique scheduling needs

### Key Findings

**Muon Optimizer's Optimal Configuration:**
- Momentum: 0.9
- Weight decay: 0.2
- Learning rate schedule: Cosine decay

**Newton-Schulz Iteration Analysis:**
- 3 steps yield **40% computational savings** while maintaining quality similar to 5 iterations

**Main Conclusion:**
The findings demonstrate Muon's superiority as an optimizer for MoE (Mixture of Experts) transformer training, providing quicker convergence and increased robustness compared to Adam.

## Primary Experiment: exp9 - Muon vs Adam

The core of this thesis is **Experiment 9**, which provides a comprehensive comparison between Muon and Adam optimizers. This experiment includes:

- Extensive hyperparameter sweeps for both optimizers
- Performance analysis across multiple metrics (loss, convergence speed, robustness)
- Computational efficiency comparisons
- Ablation studies on key optimizer components

📁 **Main experiment location**: `experiments/exp9_muon_vs_adam/`

## Quick Start

```bash
pip install -r requirements.txt

# Run the main Muon vs Adam comparison
python train_moe.py
```

## Repository Structure

- **`experiments/`** - All experimental results, with exp9 being the primary thesis experiment
- **`experiments/exp9_muon_vs_adam/`** - Main thesis experiment comparing Muon and Adam
- **`experiments/ai_research_paper.tex`** - Thesis LaTeX document
- **`models/`** - Model architectures (MoE transformers)
- **`training/`** - Training scripts and optimizer implementations
- **`configs/`** - Configuration files for experiments

## Other Experiments

While this repository's primary focus is the Muon vs Adam comparison (exp9), it also contains additional experiments from earlier research stages:

### [Exp7: Hybrid DeltaNet Architecture Ablation](experiments/exp7_hybrid_deltanet_ablation/)
- Comprehensive ablation of 13 architectures finding optimal attention layer distribution

### [Exp6: Gated DeltaNet Training](experiments/exp6_gated_deltanet_training/)
- Learning rate ablation study for Gated DeltaNet architecture

### [Exp5: Batch Size vs Sequence Length](experiments/exp5_batch_vs_seqlen_ablation/)
- Balanced approach analysis for batch size and sequence length optimization

### [Exp4: AMP vs FP32 on T4](experiments/exp4_amp_fp32_t4/)
- Mixed precision vs full precision performance comparison

### [Exp3: PLASA + GDN Hybrid](experiments/exp3_plasa_gdn_hybrid/)
- Per-layer adaptive sparse attention analysis

### [Exp1: DSA + GDN Hybrid](experiments/exp1_dsa_gdn_hybrid/)
- DeepSeek Sparse Attention with Gated DeltaNet hybrid testing

## Research Questions

1. How do Muon and Adam optimizers compare for training large language models?
2. What are the optimal hyperparameter configurations for each optimizer?
3. How does gradient orthogonalization affect training dynamics?
4. What is the computational trade-off of Newton-Schulz iterations in Muon?
5. Which optimizer provides better convergence and robustness for MoE transformers?

## Citation

If you use this work in your research, please cite:

```bibtex
@mastersthesis{rosic2025muon,
  author = {Rosić, Vuk},
  title = {Comparative Analysis of Muon and Adam Optimizers for Training Large Language Models},
  school = {Óbuda University},
  year = {2025}
}
```

## Contact

**Vuk Rosić**
- GitHub: [@vukrosic](https://github.com/vukrosic)
- YouTube: [Vuk Rosić Channel](https://www.youtube.com/channel/UC7XJj9pv_11a11FUxCMz15g)

---

*This research is part of my bachelor's thesis work on optimizer design for large language models.*