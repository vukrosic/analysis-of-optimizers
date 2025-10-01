# Experiment 5: DeepSeek MHLA with/without Sparse Attention

**Testing whether sparse token selection improves DeepSeek's already-efficient Multi-Head Latent Attention (MHLA).**

---

## 📋 Table of Contents

1. [Quick Start](#-quick-start)
2. [What's Being Compared](#-whats-being-compared)
3. [Key Differences from Experiment 4](#-key-differences-from-experiment-4)
4. [Architecture Details](#-architecture-details)
5. [How to Run](#-how-to-run)
6. [Expected Results](#-expected-results)
7. [Understanding the Code](#-understanding-the-code)
8. [Customization](#-customization)

---

## 🚀 Quick Start

```bash
cd experiments/exp5_mhla_sparse_attention
python run_experiment.py
```

**What you'll get**: Comparison of DeepSeek MHLA (dense) vs DeepSeek MHLA + Sparse Attention across sequence lengths 64, 128, 256.

**Time**: ~20-25 minutes on GPU

---

## 🎯 What's Being Compared

### Baseline: DeepSeek Multi-Head Latent Attention (Dense)

**Architecture**:
- ✅ **Multi-Head Latent Attention (MHLA)** with KV compression
- ✅ **LoRA-style projections** (compress → decompress)
- ✅ **RoPE positional encoding**
- ✅ **Shared latent space** for efficient KV cache
- ❌ **NO sparse token selection** (attends to all tokens)

**Characteristics**:
- Already efficient via latent compression
- O(L²) attention complexity
- Reduced KV cache size
- Standard for DeepSeek models

### Experimental: DeepSeek MHLA + Sparse Attention

**Architecture**:
- ✅ **Multi-Head Latent Attention (MHLA)** with KV compression
- ✅ **LoRA-style projections** (compress → decompress)
- ✅ **RoPE positional encoding**
- ✅ **Shared latent space** for efficient KV cache
- ✅ **Lightning Indexer** for token relevance scoring
- ✅ **Top-k token selection** (sparse attention mask)

**Characteristics**:
- MHLA efficiency + sparse selection
- O(Lk) attention complexity where k < L
- Further reduced computation
- Tests if sparsity helps already-efficient attention

---

## 🔑 Key Differences from Experiment 4

| Aspect | Experiment 4 | Experiment 5 (This) |
|--------|-------------|---------------------|
| **Baseline** | Standard Multi-Head Attention | DeepSeek MHLA (with latent compression) |
| **Attention Type** | QKV linear projections | Compressed KV with LoRA-style up/down projections |
| **KV Cache** | Full dimensional | Compressed to latent space |
| **Positional Encoding** | Simple RoPE | DeepSeek's RoPE with nope/rope split |
| **Question** | Does sparse help standard attention? | Does sparse help already-efficient MHLA? |

**Why This Matters**: Experiment 4 showed sparse helps standard attention. But does it help when attention is already optimized via latent compression? This experiment answers that.

---

## 🏗️ Architecture Details

### Multi-Head Latent Attention (MHLA)

MHLA reduces memory by compressing KV states:

```python
# 1. Compress Keys/Values to latent space
compressed_kv = W^DKV @ hidden_states  # Down-projection
compressed_kv = LayerNorm(compressed_kv)

# 2. Decompress for each head
kv = W^UK @ compressed_kv  # Up-projection
k_nope, v = split(kv)

# 3. Add positional component
k_pe = RoPE(W^KR @ hidden_states)
key = concat([k_nope, k_pe])

# 4. Standard attention
attn_output = softmax(Q @ K^T / √d) @ V
```

**Benefits**:
- KV cache: `batch × seq_len × latent_dim` (smaller!)
- Shared latent space across heads
- Efficient for long contexts

### Sparse MHLA (Our Addition)

We add Lightning Indexer before attention:

```python
# 1-3. Same MHLA compression/decompression as above

# 4. Lightning Indexer (NEW)
index_scores = LightningIndexer(hidden_states)  # [B, L, L]

# 5. Top-k Selection (NEW)
top_k_mask = TopKSelector(index_scores, k=L/2)  # Select 50%

# 6. Sparse Attention (NEW)
sparse_mask = create_mask(top_k_mask)  # -inf for non-selected
attn_output = softmax(Q @ K^T / √d + sparse_mask) @ V
```

**Additional Benefits**:
- Further reduced computation (50% tokens)
- Forced attention selectivity
- Potential regularization effect

---

## 🏃 How to Run

### Basic Usage

```bash
python run_experiment.py
```

### What Happens

1. **Load Data**: TinyStories dataset tokenized for each sequence length
2. **Train Baseline**: DeepSeek MHLA (dense) for 1000 steps
3. **Train Sparse**: DeepSeek MHLA + Sparse for 1000 steps
4. **Compare**: Generate plots and metrics
5. **Save Results**: JSON files + visualization

### Output Structure

```
results/
├── sequence_length_comparison.png    # Main visualization
├── summary.json                      # Numerical comparison
└── seq_*/
    ├── baseline_results.json         # Dense MHLA results
    └── sparse_results.json           # Sparse MHLA results
```

---

## 📊 Expected Results

### Hypothesis

**Sparse attention may provide smaller improvements than Experiment 4** because:
1. MHLA is already efficient via compression
2. Latent space may already capture relevant information
3. Additional sparsity might be redundant

**OR sparse attention could still help** because:
1. Forced selectivity acts as regularization
2. Complementary to latent compression
3. Reduces attention dilution

### Metrics to Watch

- **Loss**: Does sparse achieve lower loss?
- **Accuracy**: Is there a performance gap?
- **Training Time**: Overhead of indexer vs compute savings
- **Sequence Length Trends**: Does benefit increase with length?

---

## 🔍 Understanding the Code

### File Structure

```
exp5_mhla_sparse_attention/
├── sparse_mhla_attention.py    # Lightning Indexer + Sparse MHLA wrapper
├── exp5_models.py              # Full model definitions (baseline & sparse)
├── run_experiment.py           # Training script
├── README.md                   # This file
└── results/                    # Experiment outputs
```

### Key Classes

#### 1. `LightningIndexer` (sparse_mhla_attention.py)

Computes token relevance scores:

```python
I_{t,s} = Σ_j w_{t,j} · ReLU(q_{t,j}^I · k_s^I)
```

- Input: Hidden states `[batch, seq_len, d_model]`
- Output: Index scores `[batch, seq_len, seq_len]`
- Parameters: ~40K for typical config

#### 2. `DeepSeekSparseMLHA` (sparse_mhla_attention.py)

Wraps DeepseekV3Attention with sparse selection:

```python
class DeepSeekSparseMLHA(nn.Module):
    def __init__(self, config, layer_idx, sparse_top_k=512):
        self.mhla = DeepseekV3Attention(config, layer_idx)  # Standard MHLA
        self.indexer = LightningIndexer(...)                # Token scorer
        self.selector = TopKTokenSelector(...)              # Top-k picker
```

#### 3. Model Classes (exp5_models.py)

- `BaselineMLHAMoELLM`: Dense MHLA + MoE
- `SparseMLHAMoELLM`: Sparse MHLA + MoE

Both share same architecture except for sparse attention component.

---

## 🎨 Customization

### Change Sparsity Level

```python
# In run_experiment.py, line ~215
config['sparse_top_k'] = int(seq_len * 0.25)  # 75% sparse
config['sparse_top_k'] = int(seq_len * 0.75)  # 25% sparse
```

### Adjust MHLA Compression

```python
# In BASE_CONFIG
BASE_CONFIG = {
    'kv_lora_rank': 32,   # More compression (smaller)
    'kv_lora_rank': 128,  # Less compression (larger)
    'q_lora_rank': 64,    # Also compress queries
}
```

### Test Different Sequence Lengths

```python
SEQUENCE_LENGTHS = [32, 64, 128, 256, 512]
```

### Change Model Size

```python
BASE_CONFIG = {
    'd_model': 512,      # Larger model
    'n_heads': 16,       # More attention heads
    'n_layers': 8,       # Deeper
}
```

### Adjust Training

```python
BASE_CONFIG = {
    'steps': 2000,       # Longer training
    'learning_rate': 1e-3,  # Different LR
    'batch_size': 32,    # Larger batches
}
```

---

## 🔬 Research Questions

This experiment helps answer:

1. **Does sparse attention help already-efficient architectures?**
   - MHLA already compresses KV cache
   - Does additional sparsity provide benefits?

2. **Is sparsity complementary to latent compression?**
   - Do they work together or overlap?
   - Which provides more benefit?

3. **Does the combination scale better?**
   - Do benefits increase with sequence length?
   - Is combined approach best for long contexts?

---

## 📊 Interpreting Results

### Scenario 1: Sparse Provides Large Improvements

**Interpretation**: Sparse attention and latent compression are **complementary**
- Latent compression: Reduces memory
- Sparse selection: Improves attention quality
- **Implication**: Use both for long-context models

### Scenario 2: Sparse Provides Small Improvements

**Interpretation**: Latent compression already captures most benefits
- MHLA's efficiency sufficient
- Sparse adds marginal value
- **Implication**: Latent compression alone may be enough

### Scenario 3: Sparse Hurts Performance

**Interpretation**: Excessive compression/sparsity
- Both mechanisms remove information
- Combined effect too aggressive
- **Implication**: Balance needed between efficiency and capacity

---

## 🔗 Related Experiments

- **Experiment 1**: Ablation of DeepSeek components
- **Experiment 3**: MHLA + MoE optimization
- **Experiment 4**: Sparse attention on standard MHA (comparison baseline)

---

## 📚 References

- **DeepSeek-V3 Paper**: Multi-Head Latent Attention architecture
- **DeepSeek-V3.2-Exp Paper**: Lightning Indexer and sparse attention
- **Implementation**: `deepseek_modeling.py` (MHLA), `sparse_mhla_attention.py` (sparse component)

---

## 🎯 Key Takeaways

1. **This experiment tests sparse attention on DeepSeek's production architecture** (MHLA + MoE)
2. **Experiment 4 was baseline validation** (sparse on standard attention)
3. **Results inform whether to use sparse in real DeepSeek-style models**
4. **Helps understand interaction between compression and sparsity**

---

## 💡 Tips

- **Run Experiment 4 first** to understand baseline sparse attention benefits
- **Compare parameter counts** - sparse adds ~5% more parameters
- **Watch memory usage** - MHLA already reduces memory, sparse may help further
- **Check training stability** - sparse can regularize or destabilize depending on config

---

*Experiment designed: October 2025*  
*Tests DeepSeek MHLA (from V3) + Sparse Attention (from V3.2-Exp)*

