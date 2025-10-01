# Experiment 4: DeepSeek Sparse Attention (DSA) Implementation & Comparison

## Overview
This experiment implements and evaluates **DeepSeek Sparse Attention (DSA)** as described in the DeepSeek-V3.2-Exp paper. We compare the sparse attention mechanism with classic dense attention using the GLM4 MoE architecture.

## DeepSeek Sparse Attention (DSA)

### Core Components

#### 1. Lightning Indexer
The lightning indexer computes index scores between query token h_t and preceding tokens h_s:

```
I_{t,s} = Σ_{j=1}^{H_I} w_{t,j}^I · ReLU(q_{t,j}^I · k_s^I)
```

Where:
- `H_I`: Number of indexer heads (typically 4-8)
- `q_{t,j}^I`: Query from indexer head j for token t
- `k_s^I`: Key from preceding token s
- `w_{t,j}^I`: Weight parameter for head j
- `ReLU`: Activation function (for throughput efficiency)

**Key Features**:
- Small number of heads (4-8 vs 8-32 for main attention)
- Can be implemented in FP8 for efficiency
- Lightweight computational cost

#### 2. Fine-Grained Token Selection
- Selects top-k tokens based on index scores I_{t,s}
- Default k=2048 for 128K context (from paper)
- Reduces attention complexity from O(L²) to O(Lk)

#### 3. Sparse Attention Mechanism
```
u_t = Attention(h_t, {c_s | I_{t,s} ∈ Top-k(I_{t,:})})
```

Attention is computed only on the selected sparse key-value entries.

### Training Strategy

#### Stage 1: Dense Warm-up (1000 steps)
- Keep dense attention, freeze main model
- Train only the lightning indexer
- Align indexer with main attention distribution using KL divergence:
  ```
  L_I = Σ_t KL(p_{t,:} || Softmax(I_{t,:}))
  ```

#### Stage 2: Sparse Training
- Enable token selection mechanism
- Train both indexer and main model
- Indexer loss focuses on selected tokens:
  ```
  L_I = Σ_t KL(p_{t,S_t} || Softmax(I_{t,S_t}))
  ```

## Model Architectures

### 1. DeepSeek Sparse Attention Model (DSA)
- **Attention**: DeepSeek Sparse Attention with lightning indexer
- **MoE**: GLM4 MoE (4 experts, top-2 routing)
- **Architecture**: Same as Exp 3 base model
- **Key Parameter**: top_k = 2048 (adjustable)

### 2. Classic DeepSeek Attention Model (Baseline)
- **Attention**: Standard dense multi-head attention
- **MoE**: GLM4 MoE (4 experts, top-2 routing)
- **Architecture**: Same as Exp 3 base model
- **Baseline**: For comparison

## Usage

### Quick Start (Full Experiment)
```bash
cd experiments/exp4_deepseek_sparse_attention
python run_experiment.py
```

### Individual Components

#### Train Sparse Attention Model
```bash
python train_sparse.py
```

#### Train Classic Attention Model (Baseline)
```bash
python train_classic.py
```

#### Compare Results
```bash
python compare_results.py
```

## Configuration

### Model Configuration
```python
# Base model config (from Exp 3 optimal)
d_model = 256
n_heads = 8
n_layers = 6
d_ff = 512
num_experts = 4
expert_top_k = 2

# DSA specific config
indexer_heads = 4        # Number of lightning indexer heads
sparse_top_k = 512      # Number of tokens to select (for 128 seq len)
indexer_dim = 64        # Indexer dimension
```

### Training Configuration
```python
# Warmup stage (indexer only)
warmup_steps = 200
warmup_lr = 1e-3

# Sparse training stage
max_steps = 5000
learning_rate = 3e-3    # From Exp 3 optimal
batch_size = 16
```

## Evaluation Metrics

### 1. Performance Metrics
- **Validation Loss**: Language modeling loss
- **Validation Accuracy**: Token prediction accuracy
- **Perplexity**: Model confidence measure

### 2. Efficiency Metrics
- **Training Time**: Wall-clock time per epoch
- **Memory Usage**: GPU memory consumption
- **FLOPs**: Computational cost
- **Tokens/Second**: Throughput

### 3. Attention Analysis
- **Sparsity Pattern**: Visualization of selected tokens
- **Indexer Alignment**: KL divergence between indexer and main attention
- **Token Selection**: Distribution of selected tokens

## Expected Results

### Performance Comparison
| Metric | Classic Attention | Sparse Attention (DSA) | Improvement |
|--------|------------------|----------------------|-------------|
| Val Loss | ~0.06 | ~0.06 | Similar |
| Val Accuracy | ~98.7% | ~98.5% | -0.2% |
| Perplexity | ~1.06 | ~1.06 | Similar |

### Efficiency Comparison (128K context equivalent)
| Metric | Classic Attention | Sparse Attention (DSA) | Speedup |
|--------|------------------|----------------------|---------|
| Training Time | 1x | 0.6-0.7x | 1.4-1.7x |
| Memory Usage | 1x | 0.7-0.8x | 1.2-1.4x |
| Inference FLOPs | O(L²) | O(Lk) | 62x @ 128K |

**Note**: Actual speedup depends on sequence length. Longer sequences show greater benefits.

## File Structure
```
exp4_deepseek_sparse_attention/
├── README.md                      # This file
├── sparse_attention.py            # DSA implementation
├── models.py                      # Model definitions (sparse & classic)
├── train_sparse.py               # Train sparse attention model
├── train_classic.py              # Train classic attention model (baseline)
├── run_experiment.py             # Run full comparison experiment
├── compare_results.py            # Analyze and compare results
├── visualize_attention.py        # Visualize attention patterns
├── results/                      # Generated results
│   ├── sparse/                  # Sparse attention results
│   │   ├── training_results.json
│   │   ├── training_curves.png
│   │   └── final_model.pt
│   ├── classic/                 # Classic attention results
│   │   ├── training_results.json
│   │   ├── training_curves.png
│   │   └── final_model.pt
│   └── comparison/              # Comparison results
│       ├── comparison_metrics.json
│       ├── efficiency_comparison.png
│       ├── performance_comparison.png
│       └── attention_patterns.png
└── configs/
    ├── sparse_config.py         # Sparse attention config
    └── classic_config.py        # Classic attention config
```

## Key Findings (from DeepSeek-V3.2-Exp Paper)

### 1. Performance
- **Minimal degradation**: < 1% performance drop vs dense attention
- **Stable training**: RL curves closely aligned with baseline
- **Generalization**: Works well across diverse tasks

### 2. Efficiency
- **Long-context speedup**: Significant gains for sequences > 32K
- **Memory savings**: ~30% reduction in attention memory
- **Scalability**: Benefits increase with sequence length

### 3. Implementation
- **Lightning indexer**: FP8 implementation for efficiency
- **Top-k selection**: Hardware-friendly sparse pattern
- **ReLU activation**: Chosen for throughput considerations

## Advanced Features

### 1. Attention Pattern Visualization
```bash
python visualize_attention.py --checkpoint results/sparse/final_model.pt
```
Generates heatmaps showing:
- Selected token patterns
- Indexer score distribution
- Comparison with dense attention

### 2. Efficiency Profiling
```bash
python profile_efficiency.py --model sparse --sequence_lengths 128,256,512,1024
```
Measures:
- Time per forward pass
- Memory consumption
- FLOPs count
- Throughput

### 3. Ablation Studies
```bash
python ablation_study.py
```
Tests:
- Different indexer head counts
- Various top-k values
- Alternative activation functions
- Different indexer dimensions

## Hardware Requirements
- **GPU**: NVIDIA GPU with 8+ GB VRAM
- **RAM**: 16+ GB system RAM
- **Storage**: 2+ GB for models and results

## Troubleshooting

### Common Issues
1. **CUDA OOM**: Reduce `sparse_top_k` or `batch_size`
2. **Slow training**: Enable mixed precision with `use_amp=True`
3. **Poor indexer alignment**: Increase `warmup_steps`
4. **Sparse pattern artifacts**: Adjust `indexer_heads` or learning rate

### Performance Tips
1. **Sequence length matters**: Longer sequences = greater speedup
2. **Top-k selection**: Balance between performance and efficiency
3. **Indexer capacity**: More heads = better alignment but slower
4. **Warmup is critical**: Proper indexer initialization is essential

## Comparison with Other Experiments

| Experiment | Model | Attention | Key Focus |
|-----------|-------|-----------|-----------|
| Exp 1 | Multiple | Standard | Architecture comparison |
| Exp 2 | DeepSeek+MLP | Standard | LR optimization |
| Exp 3 | DeepSeek+MoE | Standard | LR+Expert optimization |
| **Exp 4** | **DeepSeek+MoE** | **Sparse (DSA)** | **Sparse attention efficiency** |

## Future Work
1. **Longer contexts**: Test on 32K, 64K, 128K sequences
2. **FP8 implementation**: Optimize indexer with FP8
3. **Kernel optimization**: Custom CUDA kernels for top-k selection
4. **Cross-attention DSA**: Apply to encoder-decoder models
5. **Dynamic top-k**: Adaptive selection based on token importance

## References
- DeepSeek-V3.2-Exp Paper: "Boosting Long-Context Efficiency with DeepSeek Sparse Attention"
- Native Sparse Attention (Yuan et al., 2025)
- MLA Architecture (DeepSeek-V2)
- MQA (Shazeer, 2019)

## Citation
If you use this implementation, please cite:
```bibtex
@article{deepseek2025v32,
  title={DeepSeek-V3.2-Exp: Boosting Long-Context Efficiency with DeepSeek Sparse Attention},
  author={DeepSeek-AI},
  journal={arXiv preprint},
  year={2025}
}
```
