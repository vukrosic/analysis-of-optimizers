#!/usr/bin/env python3
"""
Experiment 6: Token Selection Analysis with LLM Training
Training an LLM while visualizing which tokens the Lightning Indexer selects
"""

import os
import sys
import json
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime
import argparse

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(project_root)

# Import project modules
from data.loader import DataLoader
from models.components import LightningIndexer, TopKTokenSelector, RotaryPositionalEmbeddings
from interpretability.attention_visualizer import AttentionVisualizer
from interpretability.pattern_analyzer import PatternAnalyzer
from interpretability.indexer_interpreter import IndexerInterpreter

class SparseAttentionWithVisualization(nn.Module):
    """Sparse attention with comprehensive visualization capabilities"""
    
    def __init__(self, d_model: int, n_heads: int, max_seq_len: int, 
                 indexer_heads: int = 4, indexer_dim: int = 64, sparse_top_k: int = 256):
        super().__init__()
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.max_seq_len = max_seq_len
        self.sparse_top_k = sparse_top_k
        
        # Standard attention components
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
        
    def forward(self, x: torch.Tensor, return_analysis: bool = False) -> Tuple[torch.Tensor, Optional[Dict]]:
        """Forward pass with optional analysis"""
        batch_size, seq_len, d_model = x.shape
        
        # Standard QKV computation
        Q, K, V = self.qkv(x).split(d_model, dim=-1)
        Q, K = self.rotary(Q), self.rotary(K)
        
        # Lightning Indexer computation
        index_scores = self.indexer(x)  # [batch, heads, seq_len, seq_len]
        
        # Token selection
        top_k = min(self.sparse_top_k, seq_len)
        top_k_mask, selected_indices = self.selector(index_scores, k=top_k)
        
        # Sparse attention
        attn_mask = torch.where(top_k_mask, 0, -float('inf'))
        attn_output = F.scaled_dot_product_attention(Q, K, V, attn_mask=attn_mask)
        
        # Output projection
        output = self.w_o(attn_output)
        
        if return_analysis:
            analysis = self._analyze_forward_pass(x, index_scores, top_k_mask, attn_output, selected_indices)
            return output, analysis
        
        return output, None
    
    def _analyze_forward_pass(self, x: torch.Tensor, index_scores: torch.Tensor, 
                            top_k_mask: torch.Tensor, attn_output: torch.Tensor,
                            selected_indices: torch.Tensor) -> Dict[str, Any]:
        """Analyze the forward pass for visualization"""
        analysis = {}
        
        # Token selection analysis
        analysis['selected_tokens'] = top_k_mask
        analysis['selected_indices'] = selected_indices
        analysis['index_scores'] = index_scores
        analysis['selection_patterns'] = self._analyze_selection_patterns(top_k_mask)
        
        # Attention pattern analysis
        analysis['attention_patterns'] = self._extract_attention_patterns(attn_output)
        
        # Indexer behavior analysis
        analysis['indexer_behavior'] = self._analyze_indexer_behavior(index_scores)
        
        # Efficiency metrics
        analysis['efficiency_metrics'] = self._compute_efficiency_metrics(x, top_k_mask, attn_output)
        
        return analysis
    
    def _analyze_selection_patterns(self, top_k_mask: torch.Tensor) -> Dict[str, Any]:
        """Analyze token selection patterns"""
        batch_size, seq_len = top_k_mask.shape
        
        # Selection frequency by position
        selection_freq = top_k_mask.float().mean(dim=0)
        
        # Selection consistency across batch
        selection_consistency = top_k_mask.float().std(dim=0)
        
        # Overall sparsity ratio
        sparsity_ratio = 1.0 - top_k_mask.float().mean()
        
        return {
            'selection_frequency': selection_freq,
            'selection_consistency': selection_consistency,
            'sparsity_ratio': sparsity_ratio
        }
    
    def _extract_attention_patterns(self, attn_output: torch.Tensor) -> torch.Tensor:
        """Extract attention patterns from output"""
        # For visualization, we'll use the output as a proxy for attention patterns
        # In a real implementation, you'd extract actual attention weights
        return attn_output
    
    def _analyze_indexer_behavior(self, index_scores: torch.Tensor) -> Dict[str, Any]:
        """Analyze Lightning Indexer behavior"""
        batch_size, n_heads, seq_len, _ = index_scores.shape
        
        # Score statistics
        score_stats = {
            'mean': index_scores.mean().item(),
            'std': index_scores.std().item(),
            'min': index_scores.min().item(),
            'max': index_scores.max().item()
        }
        
        # Head-wise analysis
        head_analysis = {}
        for head_idx in range(n_heads):
            head_scores = index_scores[:, head_idx, :, :]
            head_analysis[head_idx] = {
                'mean': head_scores.mean().item(),
                'std': head_scores.std().item(),
                'sparsity': (head_scores < 0.01).float().mean().item()
            }
        
        return {
            'score_stats': score_stats,
            'head_analysis': head_analysis
        }
    
    def _compute_efficiency_metrics(self, x: torch.Tensor, top_k_mask: torch.Tensor, 
                                   attn_output: torch.Tensor) -> Dict[str, float]:
        """Compute efficiency metrics"""
        batch_size, seq_len, d_model = x.shape
        
        # Computational cost
        dense_cost = seq_len * seq_len * d_model
        sparse_cost = seq_len * top_k_mask.sum().item() * d_model
        cost_reduction = (dense_cost - sparse_cost) / dense_cost
        
        # Memory efficiency
        dense_memory = seq_len * seq_len * 4  # 4 bytes per float
        sparse_memory = top_k_mask.sum().item() * 4
        memory_reduction = (dense_memory - sparse_memory) / dense_memory
        
        return {
            'cost_reduction': cost_reduction,
            'memory_reduction': memory_reduction,
            'sparsity_ratio': 1.0 - top_k_mask.float().mean().item()
        }

class TrainingWithVisualization:
    """Training loop with comprehensive visualization"""
    
    def __init__(self, model: nn.Module, dataloader: DataLoader, 
                 visualizer: AttentionVisualizer, config: Dict[str, Any]):
        self.model = model
        self.dataloader = dataloader
        self.visualizer = visualizer
        self.config = config
        
        # Analysis storage
        self.analysis_history = []
        self.performance_history = []
        
        # Visualization settings
        self.viz_frequency = config.get('viz_frequency', 100)
        self.analysis_frequency = config.get('analysis_frequency', 10)
        
    def train(self, num_steps: int) -> Dict[str, Any]:
        """Train the model with visualization"""
        print(f"🚀 Starting training with visualization for {num_steps} steps")
        
        # Training setup
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config['learning_rate'])
        criterion = nn.CrossEntropyLoss()
        
        # Training loop
        for step in range(num_steps):
            # Get batch
            batch = next(iter(self.dataloader))
            if isinstance(batch, dict):
                inputs = batch['input_ids']
                targets = batch['target_ids']
            else:
                inputs = batch
                targets = batch
            
            # Forward pass with analysis
            if step % self.analysis_frequency == 0:
                output, analysis = self.model(inputs, return_analysis=True)
                
                # Store analysis
                self.analysis_history.append({
                    'step': step,
                    'analysis': analysis,
                    'timestamp': datetime.now().isoformat()
                })
                
                # Visualize if needed
                if step % self.viz_frequency == 0:
                    self._visualize_step(step, analysis, inputs)
            else:
                output, _ = self.model(inputs, return_analysis=False)
            
            # Compute loss
            if output.shape[-1] == targets.shape[-1]:
                loss = criterion(output.view(-1, output.size(-1)), targets.view(-1))
            else:
                # Simple reconstruction loss
                loss = F.mse_loss(output, inputs)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Store performance
            self.performance_history.append({
                'step': step,
                'loss': loss.item(),
                'timestamp': datetime.now().isoformat()
            })
            
            # Log progress
            if step % 50 == 0:
                print(f"Step {step:4d}: Loss = {loss.item():.4f}")
        
        # Final analysis
        final_analysis = self._final_analysis()
        
        return {
            'analysis_history': self.analysis_history,
            'performance_history': self.performance_history,
            'final_analysis': final_analysis
        }
    
    def _visualize_step(self, step: int, analysis: Dict[str, Any], inputs: torch.Tensor):
        """Visualize current step"""
        print(f"📊 Visualizing step {step}")
        
        # Create visualization directory
        os.makedirs('results', exist_ok=True)
        
        # Token selection visualization
        if 'index_scores' in analysis and 'selected_tokens' in analysis:
            self.visualizer.visualize_token_selection(
                analysis['index_scores'],
                analysis['selected_tokens'],
                top_k=self.model.sparse_top_k,
                title=f"Token Selection - Step {step}",
                save_path=f'results/token_selection_step_{step}.png'
            )
        
        # Attention pattern visualization
        if 'attention_patterns' in analysis and 'selected_tokens' in analysis:
            self.visualizer.visualize_sparse_attention(
                analysis['attention_patterns'],
                analysis['selected_tokens'],
                title=f"Attention Patterns - Step {step}",
                save_path=f'results/attention_patterns_step_{step}.png'
            )
        
        # Indexer behavior analysis
        if 'indexer_behavior' in analysis:
            self.visualizer.visualize_indexer_analysis(
                analysis['indexer_behavior'],
                title=f"Indexer Behavior - Step {step}",
                save_path=f'results/indexer_behavior_step_{step}.png'
            )
    
    def _final_analysis(self) -> Dict[str, Any]:
        """Perform final comprehensive analysis"""
        print("🔍 Performing final analysis")
        
        # Create comprehensive visualizations
        self._create_comprehensive_visualizations()
        
        # Analyze patterns over time
        pattern_analysis = self._analyze_patterns_over_time()
        
        # Performance correlation analysis
        performance_analysis = self._analyze_performance_correlation()
        
        return {
            'pattern_analysis': pattern_analysis,
            'performance_analysis': performance_analysis
        }
    
    def _create_comprehensive_visualizations(self):
        """Create comprehensive visualizations"""
        # Evolution of attention patterns
        self._visualize_attention_evolution()
        
        # Token selection patterns over time
        self._visualize_selection_evolution()
        
        # Indexer behavior evolution
        self._visualize_indexer_evolution()
        
        # Performance trends
        self._visualize_performance_trends()
    
    def _visualize_attention_evolution(self):
        """Visualize how attention patterns evolve during training"""
        if not self.analysis_history:
            return
        
        # Extract attention patterns over time
        steps = [h['step'] for h in self.analysis_history]
        
        # Create evolution plot
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot 1: Selection frequency by position over time
        ax1 = axes[0, 0]
        for i, history in enumerate(self.analysis_history[::5]):  # Every 5th step
            if 'selection_patterns' in history['analysis']:
                selection_freq = history['analysis']['selection_patterns']['selection_frequency']
                ax1.plot(selection_freq, alpha=0.7, label=f'Step {history["step"]}')
        ax1.set_title('Selection Frequency by Position Over Time')
        ax1.set_xlabel('Token Position')
        ax1.set_ylabel('Selection Frequency')
        ax1.legend()
        
        # Plot 2: Sparsity ratio over time
        ax2 = axes[0, 1]
        sparsity_ratios = []
        for history in self.analysis_history:
            if 'selection_patterns' in history['analysis']:
                sparsity_ratios.append(history['analysis']['selection_patterns']['sparsity_ratio'])
        ax2.plot(steps[:len(sparsity_ratios)], sparsity_ratios)
        ax2.set_title('Sparsity Ratio Over Time')
        ax2.set_xlabel('Training Step')
        ax2.set_ylabel('Sparsity Ratio')
        
        # Plot 3: Efficiency metrics over time
        ax3 = axes[1, 0]
        cost_reductions = []
        memory_reductions = []
        for history in self.analysis_history:
            if 'efficiency_metrics' in history['analysis']:
                cost_reductions.append(history['analysis']['efficiency_metrics']['cost_reduction'])
                memory_reductions.append(history['analysis']['efficiency_metrics']['memory_reduction'])
        
        if cost_reductions:
            ax3.plot(steps[:len(cost_reductions)], cost_reductions, label='Cost Reduction')
            ax3.plot(steps[:len(memory_reductions)], memory_reductions, label='Memory Reduction')
            ax3.set_title('Efficiency Metrics Over Time')
            ax3.set_xlabel('Training Step')
            ax3.set_ylabel('Reduction Ratio')
            ax3.legend()
        
        # Plot 4: Performance over time
        ax4 = axes[1, 1]
        if self.performance_history:
            perf_steps = [p['step'] for p in self.performance_history]
            losses = [p['loss'] for p in self.performance_history]
            ax4.plot(perf_steps, losses)
            ax4.set_title('Training Loss Over Time')
            ax4.set_xlabel('Training Step')
            ax4.set_ylabel('Loss')
        
        plt.tight_layout()
        plt.savefig('results/attention_evolution.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _visualize_selection_evolution(self):
        """Visualize token selection patterns over time"""
        if not self.analysis_history:
            return
        
        # Create selection pattern heatmap
        steps = [h['step'] for h in self.analysis_history]
        
        # Extract selection patterns
        selection_matrix = []
        for history in self.analysis_history:
            if 'selection_patterns' in history['analysis']:
                selection_freq = history['analysis']['selection_patterns']['selection_frequency']
                selection_matrix.append(selection_freq.numpy())
        
        if selection_matrix:
            selection_matrix = np.array(selection_matrix)
            
            plt.figure(figsize=(12, 8))
            sns.heatmap(selection_matrix, cmap='viridis', cbar=True)
            plt.title('Token Selection Patterns Over Time')
            plt.xlabel('Token Position')
            plt.ylabel('Training Step')
            plt.savefig('results/selection_evolution.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    def _visualize_indexer_evolution(self):
        """Visualize indexer behavior evolution"""
        if not self.analysis_history:
            return
        
        # Extract indexer behavior over time
        steps = [h['step'] for h in self.analysis_history]
        
        # Score statistics over time
        score_means = []
        score_stds = []
        
        for history in self.analysis_history:
            if 'indexer_behavior' in history['analysis']:
                score_stats = history['analysis']['indexer_behavior']['score_stats']
                score_means.append(score_stats['mean'])
                score_stds.append(score_stats['std'])
        
        if score_means:
            plt.figure(figsize=(12, 6))
            plt.errorbar(steps[:len(score_means)], score_means, yerr=score_stds, capsize=5)
            plt.title('Indexer Score Evolution Over Time')
            plt.xlabel('Training Step')
            plt.ylabel('Score Value')
            plt.savefig('results/indexer_evolution.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    def _visualize_performance_trends(self):
        """Visualize performance trends"""
        if not self.performance_history:
            return
        
        steps = [p['step'] for p in self.performance_history]
        losses = [p['loss'] for p in self.performance_history]
        
        plt.figure(figsize=(12, 6))
        plt.plot(steps, losses)
        plt.title('Training Performance Over Time')
        plt.xlabel('Training Step')
        plt.ylabel('Loss')
        plt.grid(True, alpha=0.3)
        plt.savefig('results/performance_trends.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _analyze_patterns_over_time(self) -> Dict[str, Any]:
        """Analyze patterns over time"""
        if not self.analysis_history:
            return {}
        
        # Extract patterns
        patterns = []
        for history in self.analysis_history:
            if 'selection_patterns' in history['analysis']:
                patterns.append(history['analysis']['selection_patterns'])
        
        if not patterns:
            return {}
        
        # Analyze pattern evolution
        sparsity_ratios = [p['sparsity_ratio'] for p in patterns]
        
        return {
            'sparsity_evolution': {
                'mean': np.mean(sparsity_ratios),
                'std': np.std(sparsity_ratios),
                'trend': 'increasing' if sparsity_ratios[-1] > sparsity_ratios[0] else 'decreasing'
            }
        }
    
    def _analyze_performance_correlation(self) -> Dict[str, Any]:
        """Analyze correlation between selection patterns and performance"""
        if not self.analysis_history or not self.performance_history:
            return {}
        
        # Extract performance data
        perf_data = {p['step']: p['loss'] for p in self.performance_history}
        
        # Extract selection data
        selection_data = []
        performance_data = []
        
        for history in self.analysis_history:
            step = history['step']
            if step in perf_data and 'efficiency_metrics' in history['analysis']:
                efficiency = history['analysis']['efficiency_metrics']
                selection_data.append(efficiency['sparsity_ratio'])
                performance_data.append(perf_data[step])
        
        if len(selection_data) > 1:
            correlation = np.corrcoef(selection_data, performance_data)[0, 1]
            
            return {
                'sparsity_performance_correlation': correlation,
                'num_data_points': len(selection_data)
            }
        
        return {}

def main():
    """Main experiment function"""
    parser = argparse.ArgumentParser(description='Token Selection Analysis Experiment')
    parser.add_argument('--steps', type=int, default=1000, help='Number of training steps')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--seq_len', type=int, default=256, help='Sequence length')
    parser.add_argument('--d_model', type=int, default=512, help='Model dimension')
    parser.add_argument('--n_heads', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--viz_frequency', type=int, default=100, help='Visualization frequency')
    parser.add_argument('--analysis_frequency', type=int, default=10, help='Analysis frequency')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    
    args = parser.parse_args()
    
    # Configuration
    config = {
        'steps': args.steps,
        'batch_size': args.batch_size,
        'seq_len': args.seq_len,
        'd_model': args.d_model,
        'n_heads': args.n_heads,
        'learning_rate': args.learning_rate,
        'viz_frequency': args.viz_frequency,
        'analysis_frequency': args.analysis_frequency,
        'device': args.device
    }
    
    print("🔬 Experiment 6: Token Selection Analysis")
    print("=" * 50)
    print(f"Configuration: {config}")
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model
    model = SparseAttentionWithVisualization(
        d_model=args.d_model,
        n_heads=args.n_heads,
        max_seq_len=args.seq_len,
        sparse_top_k=args.seq_len // 2
    ).to(device)
    
    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Create dataloader
    dataloader = DataLoader(
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        max_tokens=50000,
        num_documents=1000
    )
    
    # Create visualizer
    visualizer = AttentionVisualizer()
    
    # Create trainer
    trainer = TrainingWithVisualization(model, dataloader, visualizer, config)
    
    # Train with visualization
    start_time = time.time()
    results = trainer.train(args.steps)
    training_time = time.time() - start_time
    
    print(f"✅ Training completed in {training_time:.2f} seconds")
    
    # Save results
    os.makedirs('results', exist_ok=True)
    
    # Save analysis history
    with open('results/analysis_history.json', 'w') as f:
        json.dump(results['analysis_history'], f, indent=2)
    
    # Save performance history
    with open('results/performance_history.json', 'w') as f:
        json.dump(results['performance_history'], f, indent=2)
    
    # Save final analysis
    with open('results/final_analysis.json', 'w') as f:
        json.dump(results['final_analysis'], f, indent=2)
    
    # Save model
    torch.save(model.state_dict(), 'results/model_checkpoint.pth')
    
    print("📊 Results saved to results/ directory")
    print("🎯 Experiment completed successfully!")
    
    return results

if __name__ == "__main__":
    results = main()
