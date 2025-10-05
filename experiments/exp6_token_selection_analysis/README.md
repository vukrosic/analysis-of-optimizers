# Experiment 6: Token Selection Analysis with LLM Training

**Training an LLM while visualizing which tokens the Lightning Indexer selects and analyzing attention patterns.**

---

## 📋 Table of Contents

1. [Research Objective](#-research-objective)
2. [What's Being Analyzed](#-whats-being-analyzed)
3. [Quick Start](#-quick-start)
4. [Implementation Details](#-implementation-details)
5. [Visualization Tools](#-visualization-tools)
6. [Expected Insights](#-expected-insights)
7. [Customization](#-customization)

---

## 🎯 Research Objective

**Goal**: Train an LLM with sparse attention while visualizing and analyzing:
- Which tokens the Lightning Indexer selects
- How attention patterns evolve during training
- What patterns the indexer learns
- How token selection relates to model performance

**Key Questions**:
- Does the indexer select semantically meaningful tokens?
- How do selection patterns change during training?
- What types of tokens are consistently selected?
- How does selection relate to attention quality?

---

## 🔬 What's Being Analyzed

### **1. Token Selection Visualization**
- **Real-time selection**: Visualize which tokens are selected during training
- **Selection patterns**: Analyze patterns in token selection
- **Positional analysis**: Study selection bias towards certain positions
- **Content analysis**: Analyze what types of content get selected

### **2. Attention Pattern Analysis**
- **Sparse vs dense**: Compare attention patterns between sparse and dense models
- **Pattern evolution**: Track how patterns change during training
- **Head specialization**: Analyze what each attention head focuses on
- **Efficiency analysis**: Measure efficiency of different patterns

### **3. Indexer Behavior Analysis**
- **Relevance scores**: Study distribution of relevance scores
- **Head specialization**: Analyze what each indexer head learns
- **Selection criteria**: Understand ranking criteria for token selection
- **Adaptation patterns**: Study how indexer adapts during training

### **4. Performance Correlation**
- **Selection quality**: Correlate selection quality with model performance
- **Efficiency metrics**: Measure computational efficiency gains
- **Quality maintenance**: Assess quality preservation with sparsity

---

## 🚀 Quick Start

```bash
# Navigate to experiment directory
cd experiments/exp6_token_selection_analysis

# Install dependencies
pip install -r requirements.txt

# Run the experiment
python run_experiment.py

# View results
open results/token_selection_analysis.html
open results/attention_patterns.png
```

**What you'll get**:
- Interactive token selection visualization
- Attention pattern analysis
- Indexer behavior insights
- Performance correlation analysis

---

## 🏗️ Implementation Details

### **Model Architecture**
Based on Experiment 1's sparse attention with enhanced visualization:

```python
class SparseAttentionWithVisualization(nn.Module):
    def __init__(self, d_model, n_heads, max_seq_len, indexer_heads=4, indexer_dim=64):
        super().__init__()
        
        # Standard components
        self.qkv = nn.Linear(d_model, d_model * 3)
        self.w_o = nn.Linear(d_model, d_model)
        self.rotary = RotaryPositionalEmbeddings(d_model, max_seq_len)
        
        # DeepSeek sparse attention components
        self.indexer = LightningIndexer(d_model, indexer_heads, indexer_dim)
        self.selector = TopKTokenSelector()
        
        # Visualization components
        self.visualizer = AttentionVisualizer()
        self.pattern_analyzer = PatternAnalyzer()
        self.indexer_interpreter = IndexerInterpreter()
        
    def forward(self, x, return_analysis=False):
        batch_size, seq_len, d_model = x.shape
        
        # Standard QKV computation
        Q, K, V = self.qkv(x).split(d_model, dim=-1)
        Q, K = self.rotary(Q), self.rotary(K)
        
        # Lightning Indexer computation
        index_scores = self.indexer(x)  # [batch, heads, seq_len, seq_len]
        
        # Token selection
        top_k_mask, selected_indices = self.selector(index_scores, k=seq_len//2)
        
        # Sparse attention
        attn_mask = torch.where(top_k_mask, 0, -float('inf'))
        attn_output = F.scaled_dot_product_attention(Q, K, V, attn_mask=attn_mask)
        
        # Output projection
        output = self.w_o(attn_output)
        
        if return_analysis:
            # Perform analysis
            analysis = self._analyze_forward_pass(x, index_scores, top_k_mask, attn_output)
            return output, analysis
        
        return output
    
    def _analyze_forward_pass(self, x, index_scores, top_k_mask, attn_output):
        """Analyze the forward pass for visualization"""
        analysis = {}
        
        # Token selection analysis
        analysis['selected_tokens'] = top_k_mask
        analysis['index_scores'] = index_scores
        analysis['selection_patterns'] = self._analyze_selection_patterns(top_k_mask)
        
        # Attention pattern analysis
        analysis['attention_patterns'] = self._extract_attention_patterns(attn_output)
        
        # Indexer behavior analysis
        analysis['indexer_behavior'] = self._analyze_indexer_behavior(index_scores)
        
        return analysis
```

### **Training Loop with Visualization**
```python
class TrainingWithVisualization:
    def __init__(self, model, dataset, visualizer):
        self.model = model
        self.dataset = dataset
        self.visualizer = visualizer
        self.analysis_history = []
        
    def train_step(self, batch, step):
        # Forward pass with analysis
        output, analysis = self.model(batch, return_analysis=True)
        
        # Store analysis for visualization
        self.analysis_history.append({
            'step': step,
            'analysis': analysis,
            'batch': batch
        })
        
        # Visualize every N steps
        if step % 100 == 0:
            self._visualize_step(step, analysis, batch)
        
        # Compute loss and backward pass
        loss = self.compute_loss(output, batch)
        loss.backward()
        
        return loss.item()
    
    def _visualize_step(self, step, analysis, batch):
        """Visualize current step"""
        # Token selection visualization
        self.visualizer.visualize_token_selection(
            analysis['index_scores'],
            analysis['selected_tokens'],
            title=f"Token Selection - Step {step}"
        )
        
        # Attention pattern visualization
        self.visualizer.visualize_sparse_attention(
            analysis['attention_patterns'],
            analysis['selected_tokens'],
            title=f"Attention Patterns - Step {step}"
        )
        
        # Indexer behavior analysis
        self.visualizer.visualize_indexer_analysis(
            analysis['indexer_behavior'],
            title=f"Indexer Behavior - Step {step}"
        )
```

---

## 🎨 Visualization Tools

### **1. Interactive Token Selection Visualization**
```python
def create_interactive_visualization(analysis_history):
    """Create interactive HTML visualization"""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Token Selection Analysis</title>
        <script src="https://d3js.org/d3.v7.min.js"></script>
        <style>
            .token { fill: #1f77b4; }
            .selected { fill: #ff7f0e; }
            .attention-line { stroke: #2ca02c; stroke-width: 2; }
        </style>
    </head>
    <body>
        <h1>Token Selection Analysis</h1>
        <div id="visualization"></div>
        <script>
            // Interactive visualization code
            const data = """ + json.dumps(analysis_history) + """;
            // D3.js visualization implementation
        </script>
    </body>
    </html>
    """
    
    with open('results/token_selection_analysis.html', 'w') as f:
        f.write(html_content)
```

### **2. Real-time Attention Pattern Visualization**
```python
def visualize_attention_evolution(analysis_history):
    """Visualize how attention patterns evolve during training"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    steps = [h['step'] for h in analysis_history]
    
    # Plot 1: Selection frequency by position
    ax1 = axes[0, 0]
    for step_idx, step in enumerate(steps[::10]):  # Every 10th step
        analysis = analysis_history[step_idx * 10]
        selected_tokens = analysis['analysis']['selected_tokens']
        selection_freq = selected_tokens.float().mean(dim=0)
        ax1.plot(selection_freq, alpha=0.7, label=f'Step {step}')
    ax1.set_title('Selection Frequency by Position')
    ax1.set_xlabel('Token Position')
    ax1.set_ylabel('Selection Frequency')
    ax1.legend()
    
    # Plot 2: Attention entropy over time
    ax2 = axes[0, 1]
    entropies = []
    for analysis in analysis_history:
        entropy = compute_attention_entropy(analysis['analysis']['attention_patterns'])
        entropies.append(entropy)
    ax2.plot(steps, entropies)
    ax2.set_title('Attention Entropy Over Time')
    ax2.set_xlabel('Training Step')
    ax2.set_ylabel('Entropy')
    
    # Plot 3: Indexer score distribution
    ax3 = axes[0, 2]
    all_scores = []
    for analysis in analysis_history:
        scores = analysis['analysis']['index_scores']
        all_scores.extend(scores.flatten().tolist())
    ax3.hist(all_scores, bins=50, alpha=0.7)
    ax3.set_title('Indexer Score Distribution')
    ax3.set_xlabel('Score Value')
    ax3.set_ylabel('Frequency')
    
    # Plot 4: Selection efficiency
    ax4 = axes[1, 0]
    efficiencies = []
    for analysis in analysis_history:
        efficiency = compute_selection_efficiency(analysis['analysis'])
        efficiencies.append(efficiency)
    ax4.plot(steps, efficiencies)
    ax4.set_title('Selection Efficiency Over Time')
    ax4.set_xlabel('Training Step')
    ax4.set_ylabel('Efficiency')
    
    # Plot 5: Pattern clustering
    ax5 = axes[1, 1]
    pattern_clusters = cluster_attention_patterns(analysis_history)
    scatter = ax5.scatter(pattern_clusters[:, 0], pattern_clusters[:, 1], 
                         c=steps, cmap='viridis')
    ax5.set_title('Attention Pattern Clustering')
    ax5.set_xlabel('Cluster Dimension 1')
    ax5.set_ylabel('Cluster Dimension 2')
    plt.colorbar(scatter, ax=ax5, label='Training Step')
    
    # Plot 6: Performance correlation
    ax6 = axes[1, 2]
    performance_scores = [h['performance'] for h in analysis_history]
    ax6.scatter(efficiencies, performance_scores, alpha=0.7)
    ax6.set_title('Selection Efficiency vs Performance')
    ax6.set_xlabel('Selection Efficiency')
    ax6.set_ylabel('Performance Score')
    
    plt.tight_layout()
    plt.savefig('results/attention_patterns.png', dpi=300, bbox_inches='tight')
    plt.show()
```

### **3. Indexer Behavior Analysis**
```python
def analyze_indexer_behavior(analysis_history):
    """Analyze how indexer behavior changes during training"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Indexer head specialization
    ax1 = axes[0, 0]
    head_specializations = []
    for analysis in analysis_history:
        specialization = compute_head_specialization(analysis['analysis']['index_scores'])
        head_specializations.append(specialization)
    
    head_specializations = np.array(head_specializations)
    for head_idx in range(head_specializations.shape[1]):
        ax1.plot(steps, head_specializations[:, head_idx], 
                label=f'Head {head_idx}', marker='o')
    ax1.set_title('Indexer Head Specialization Over Time')
    ax1.set_xlabel('Training Step')
    ax1.set_ylabel('Specialization Score')
    ax1.legend()
    
    # Plot 2: Relevance score evolution
    ax2 = axes[0, 1]
    score_stats = []
    for analysis in analysis_history:
        scores = analysis['analysis']['index_scores']
        stats = {
            'mean': scores.mean().item(),
            'std': scores.std().item(),
            'min': scores.min().item(),
            'max': scores.max().item()
        }
        score_stats.append(stats)
    
    means = [s['mean'] for s in score_stats]
    stds = [s['std'] for s in score_stats]
    ax2.errorbar(steps, means, yerr=stds, capsize=5)
    ax2.set_title('Relevance Score Evolution')
    ax2.set_xlabel('Training Step')
    ax2.set_ylabel('Score Value')
    
    # Plot 3: Selection bias analysis
    ax3 = axes[1, 0]
    biases = []
    for analysis in analysis_history:
        bias = compute_selection_bias(analysis['analysis']['selected_tokens'])
        biases.append(bias)
    
    ax3.plot(steps, biases)
    ax3.set_title('Selection Bias Over Time')
    ax3.set_xlabel('Training Step')
    ax3.set_ylabel('Bias Score')
    
    # Plot 4: Token type analysis
    ax4 = axes[1, 1]
    token_types = analyze_selected_token_types(analysis_history)
    ax4.bar(token_types.keys(), token_types.values())
    ax4.set_title('Selected Token Types')
    ax4.set_xlabel('Token Type')
    ax4.set_ylabel('Selection Frequency')
    
    plt.tight_layout()
    plt.savefig('results/indexer_behavior.png', dpi=300, bbox_inches='tight')
    plt.show()
```

---

## 📊 Expected Insights

### **1. Token Selection Patterns**
- **Semantic relevance**: Whether selected tokens are semantically meaningful
- **Positional bias**: Bias towards certain positions (beginning, end, etc.)
- **Content type**: What types of content get selected (nouns, verbs, etc.)
- **Consistency**: How consistent selection is across different inputs

### **2. Attention Pattern Evolution**
- **Pattern specialization**: How attention heads specialize during training
- **Efficiency improvement**: Whether patterns become more efficient over time
- **Quality maintenance**: How attention quality is maintained with sparsity
- **Convergence behavior**: How patterns converge during training

### **3. Indexer Behavior Insights**
- **Head specialization**: What each indexer head learns to focus on
- **Relevance criteria**: What makes a token relevant for selection
- **Adaptation patterns**: How the indexer adapts during training
- **Efficiency characteristics**: Computational efficiency of the indexer

### **4. Performance Correlations**
- **Selection quality**: Correlation between selection quality and performance
- **Efficiency gains**: Computational efficiency improvements
- **Quality preservation**: How well quality is maintained with sparsity
- **Scaling properties**: How performance scales with sequence length

---

## 🎨 Customization

### **Change Visualization Frequency**
```python
# In run_experiment.py
VISUALIZATION_FREQUENCY = 50  # Visualize every 50 steps
ANALYSIS_FREQUENCY = 10       # Store analysis every 10 steps
```

### **Modify Analysis Components**
```python
# In model definition
class CustomSparseAttention(SparseAttentionWithVisualization):
    def _analyze_forward_pass(self, x, index_scores, top_k_mask, attn_output):
        analysis = super()._analyze_forward_pass(x, index_scores, top_k_mask, attn_output)
        
        # Add custom analysis
        analysis['custom_metric'] = self.compute_custom_metric(x, attn_output)
        
        return analysis
```

### **Adjust Model Configuration**
```python
# In run_experiment.py
MODEL_CONFIG = {
    'd_model': 512,
    'n_heads': 8,
    'n_layers': 6,
    'indexer_heads': 4,
    'indexer_dim': 64,
    'sparse_top_k': 256,  # 50% sparsity for 512 length
}
```

### **Change Dataset**
```python
# In run_experiment.py
DATASET_CONFIG = {
    'name': 'tiny_stories',  # or 'wikitext', 'c4', etc.
    'max_tokens': 100000,
    'num_documents': 1000,
    'sequence_lengths': [64, 128, 256, 512]
}
```

---

## 🔬 Research Applications

### **1. Understanding Sparse Attention**
- **Mechanistic insights**: How sparse attention actually works
- **Pattern analysis**: What patterns are learned
- **Efficiency characteristics**: Computational efficiency properties

### **2. Optimizing Lightning Indexer**
- **Behavior analysis**: How the indexer selects tokens
- **Optimization opportunities**: Where improvements can be made
- **Efficiency improvements**: How to make selection more efficient

### **3. Designing Better Architectures**
- **Pattern-based design**: Using learned patterns to design architectures
- **Content-aware mechanisms**: Adapting to content characteristics
- **Efficiency optimization**: Balancing quality and efficiency

### **4. Production Deployment**
- **Performance monitoring**: Monitoring selection quality in production
- **Adaptation strategies**: How to adapt to different content types
- **Efficiency optimization**: Continuous optimization based on usage patterns

---

## 📁 File Structure

```
exp6_token_selection_analysis/
├── run_experiment.py              # Main experiment script
├── sparse_attention_with_viz.py    # Enhanced sparse attention with visualization
├── training_with_viz.py           # Training loop with visualization
├── visualization_tools.py         # Visualization utilities
├── analysis_tools.py              # Analysis utilities
├── requirements.txt               # Dependencies
├── README.md                      # This file
└── results/                       # Experiment outputs
    ├── token_selection_analysis.html
    ├── attention_patterns.png
    ├── indexer_behavior.png
    ├── analysis_history.json
    └── model_checkpoints/
```

---

## 🎯 Success Criteria

### **Visualization Quality**
- ✅ **Clear token selection visualization**: Easy to see which tokens are selected
- ✅ **Attention pattern analysis**: Understand how attention patterns evolve
- ✅ **Indexer behavior insights**: Understand how the indexer works
- ✅ **Performance correlation**: See how selection relates to performance

### **Research Insights**
- ✅ **Mechanistic understanding**: Understand how sparse attention works
- ✅ **Pattern identification**: Identify key attention patterns
- ✅ **Optimization opportunities**: Find areas for improvement
- ✅ **Efficiency analysis**: Measure computational efficiency

### **Practical Applications**
- ✅ **Production insights**: Insights applicable to production deployment
- ✅ **Optimization guidance**: Guidance for optimizing sparse attention
- ✅ **Architecture design**: Principles for designing better architectures
- ✅ **Performance monitoring**: Tools for monitoring production performance

---

## 🚀 Getting Started

1. **Clone the repository** and switch to the interpretability branch
2. **Install dependencies** from requirements.txt
3. **Run the experiment** with `python run_experiment.py`
4. **View results** in the results/ directory
5. **Analyze insights** using the visualization tools
6. **Iterate and improve** based on findings

---

This experiment provides comprehensive visualization and analysis of token selection in sparse attention, enabling deep understanding of how the Lightning Indexer works and how attention patterns evolve during training.
