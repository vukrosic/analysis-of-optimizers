# Experiment 4 Documentation Index

## 📚 Available Documentation

### Quick Start
- **[QUICKSTART.txt](QUICKSTART.txt)** - Fast introduction, how to run
- **[README.md](README.md)** - Project overview and basics

### Detailed Guides  
- **[TUTORIAL.md](TUTORIAL.md)** ⭐ - Complete tutorial from basics to advanced
  - What is sparse attention?
  - How the experiment works
  - How to run and customize
  - Troubleshooting and extensions
  
- **[RESULTS_AND_ANALYSIS.md](RESULTS_AND_ANALYSIS.md)** 📊 - Full results analysis
  - Detailed performance comparison
  - Why sparse attention works better
  - Lessons learned (including the seed bug!)
  - Interpretation guidelines

### Quick Reference
- **[EXPERIMENT_OVERVIEW.md](EXPERIMENT_OVERVIEW.md)** - High-level summary

### Results
- **[results/sequence_length_comparison.png](results/sequence_length_comparison.png)** - Main visualization
- **[results/summary.json](results/summary.json)** - Numerical results
- **[results/seq_*/](results/)** - Per-sequence detailed results

## 🗺️ Reading Guide

**I'm completely new**:
1. Start with QUICKSTART.txt (5 min)
2. Run the experiment (10-15 min)
3. Read TUTORIAL.md (30 min)

**I want to understand the results**:
1. Read RESULTS_AND_ANALYSIS.md
2. Look at sequence_length_comparison.png
3. Check summary.json

**I want to modify the experiment**:
1. Read TUTORIAL.md Part 6 (Customization)
2. Check the code comments in run_experiment.py
3. See sparse_attention.py for implementation details

**I want the research findings**:
1. RESULTS_AND_ANALYSIS.md (Sections 2-4)
2. The comparison plot
3. DeepSeek paper (in repo root)

## 📝 Key Files

```
exp4_deepseek_sparse_attention/
├── Documentation/
│   ├── INDEX.md (you are here)
│   ├── TUTORIAL.md ⭐ 
│   ├── RESULTS_AND_ANALYSIS.md 📊
│   ├── QUICKSTART.txt 🚀
│   ├── README.md
│   └── EXPERIMENT_OVERVIEW.md
│
├── Code/
│   ├── run_experiment.py
│   ├── exp4_models.py
│   ├── sparse_attention.py
│   └── config.py
│
└── Results/
    ├── sequence_length_comparison.png
    ├── summary.json
    └── seq_*/
```

## 💡 Quick Answers

**What did we find?**  
Sparse attention learns ~300% better than classic attention, especially for longer sequences!

**Is it faster?**  
About the same speed (~0.06s per step), but would scale better for very long sequences.

**How does it work?**  
Lightning indexer selects top-k relevant tokens, then attention only uses those.

**Can I replicate it?**  
Yes! Just run `python run_experiment.py` (takes ~15 min on GPU).

**What's the catch?**  
Small-scale experiment. Results might vary with different data/hyperparameters.

---

*Last updated: October 2025 (with fixed seed experiment)*
