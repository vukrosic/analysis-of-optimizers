# Qwen3-Next Architecture Reading Guide

## 🎯 How It Works - Quick Overview

The model dynamically chooses between **two types of attention** for each layer:
- **Full Attention** (O(n²)): Standard multi-head attention with RoPE
- **Linear Attention** (O(n)): Gated DeltaNet - efficient recurrent-style attention

Plus **Mixture of Experts (MoE)** for sparse computation.

---

## 📚 Reading Path (Start → End)

### 1️⃣ **Configuration** (Read First)
**File**: `models/qwen3_next/configuration_qwen3_next.py`

**Key lines**:
- **Lines 188-276**: `Qwen3NextConfig.__init__()`
  - `layer_types` (line 222): List like `["full_attention", "linear_attention", ...]`
  - `num_experts` (line 217): Number of MoE experts
  - `decoder_sparse_step` (line 213): How often to use MoE

**What to understand**: This config controls the entire architecture!

---

### 2️⃣ **Decoder Layer** (The Magic Happens Here) ⭐
**File**: `models/qwen3_next/modular_qwen3_next.py`

**Lines 621-641**: `Qwen3NextDecoderLayer.__init__()`

```python
# Line 627: Get layer type from config
self.layer_type = config.layer_types[layer_idx]

# Lines 628-631: Choose attention mechanism
if self.layer_type == "linear_attention":
    self.linear_attn = Qwen3NextGatedDeltaNet(config, layer_idx)  # O(n) complexity
elif self.layer_type == "full_attention":
    self.self_attn = Qwen3NextAttention(config, layer_idx)  # O(n²) complexity

# Lines 633-638: Choose MLP type (MoE or dense)
if config.num_experts > 0 and (layer_idx + 1) % config.decoder_sparse_step == 0:
    self.mlp = Qwen3NextSparseMoeBlock(config)  # Sparse MoE
else:
    self.mlp = Qwen3NextMLP(config)  # Dense MLP
```

**What to understand**: Each layer is customized based on its index!

---

### 3️⃣ **Full Attention** (Traditional Transformer)
**File**: `models/qwen3_next/modular_qwen3_next.py`

**Lines ~170-320**: `Qwen3NextAttention`

Key components:
- **Q, K, V projections** with Grouped-Query Attention (GQA)
- **RoPE** (Rotary Position Embeddings) for position info
- **Scaled dot-product attention**: O(n²) complexity

**Read if**: You want to understand standard transformer attention

---

### 4️⃣ **Linear Attention** (Efficient Alternative) 🚀
**File**: `models/qwen3_next/modular_qwen3_next.py`

**Lines ~400-580**: `Qwen3NextGatedDeltaNet`

Key innovation:
- **Linear complexity**: O(n) instead of O(n²)
- **Gated mechanism**: Controls information flow
- **Recurrent-style**: Can be computed sequentially or in parallel
- Uses **causal convolution** for local context

**Read if**: You want to understand how linear attention achieves efficiency

---

### 5️⃣ **Model Assembly**
**File**: `models/qwen3_next/modular_qwen3_next.py`

**Lines 717-728**: `Qwen3NextModel.__init__()`

```python
# Line 722: Create all layers using layer_types
self.layers = nn.ModuleList(
    [Qwen3NextDecoderLayer(config, layer_idx) 
     for layer_idx in range(config.num_hidden_layers)]
)
```

Each layer gets a unique `layer_idx`, which determines its attention type!

---

### 6️⃣ **Language Modeling Head**
**File**: `models/qwen3_next/modular_qwen3_next.py`

**Lines 809-820**: `Qwen3NextForCausalLM.__init__()`

```python
# Line 813: Use our custom model (not Mixtral's)
self.model = Qwen3NextModel(config)

# Line 815: Add prediction head
self.lm_head = nn.Linear(config.hidden_size, config.vocab_size)
```

**Important fix**: We override parent class to use `Qwen3NextModel` (with layer_types support)

---

## 🔍 How Data Flows Through the Model

```
Input tokens
    ↓
[Embedding Layer]
    ↓
[Layer 0] → Linear Attention (if layer_types[0] == "linear_attention")
           OR Full Attention (if layer_types[0] == "full_attention")
    ↓
[Layer 0 MLP] → MoE (if layer matches decoder_sparse_step)
               OR Dense (otherwise)
    ↓
[Layer 1] → (chosen attention type)
    ↓
[Layer 1 MLP]
    ↓
... (repeat for all layers)
    ↓
[Final LayerNorm]
    ↓
[LM Head] → Vocabulary logits
```

---

## 🧪 Experiment Files

**Your test script**: `experiments/exp6_qwen3_dsa_hybrid/test_attention_patterns.py`
- Tests 4 different `layer_types` patterns
- Measures: loss, accuracy, perplexity, speed

**Results showed**:
- **Linear-first** (L→L→F→F): Best performance! 🏆
- **Sandwich** (L→F→F→L): Close second
- **Full-first** (F→F→L→L): Worst

---

## 💡 Key Insights

1. **Hybrid is better than pure**: Mixing linear + full attention works better than all-linear or all-full

2. **Order matters**: Starting with linear attention (efficient) helps the model learn faster

3. **MoE adds capacity**: Experts allow specialization without massive compute cost

4. **The bug we found**: Originally `Qwen3NextForCausalLM` inherited from `MixtralForCausalLM` which ignored `layer_types`. We fixed it by creating our own `Qwen3NextModel` instance.

---

## 🎓 Further Reading

- **RoPE**: See `Qwen3NextRotaryEmbedding` (lines ~90-168)
- **MoE**: See `Qwen3NextSparseMoeBlock` in `qwen2_moe` imports
- **DeltaNet**: Original paper on linear attention mechanisms
- **Flash Attention**: Optional optimization (lines 60-63 check for it)

---

## 🐛 The Critical Bug Fix

**Before** (lines 809-811, old):
```python
class Qwen3NextForCausalLM(MixtralForCausalLM):
    def __init__(self, config):
        super().__init__(config)  # ❌ Creates MixtralModel, ignores layer_types!
```

**After** (lines 809-820, fixed):
```python
class Qwen3NextForCausalLM(MixtralForCausalLM):
    def __init__(self, config):
        PreTrainedModel.__init__(self, config)
        self.model = Qwen3NextModel(config)  # ✅ Uses our model with layer_types!
```

This was why all patterns had identical results initially!

