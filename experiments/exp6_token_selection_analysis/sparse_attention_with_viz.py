#!/usr/bin/env python3
"""
Enhanced sparse attention with comprehensive visualization capabilities
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import json
from datetime import datetime

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
        
        # Analysis storage
        self.attention_weights_history = []
        self.selection_history = []
        self.indexer_scores_history = []
        
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
            analysis = self._analyze_forward_pass(x, index_scores, top_k_mask, attn_output, selected_indices, Q, K, V)
            return output, analysis
        
        return output, None
    
    def _analyze_forward_pass(self, x: torch.Tensor, index_scores: torch.Tensor, 
                            top_k_mask: torch.Tensor, attn_output: torch.Tensor,
                            selected_indices: torch.Tensor, Q: torch.Tensor, 
                            K: torch.Tensor, V: torch.Tensor) -> Dict[str, Any]:
        """Analyze the forward pass for visualization"""
        analysis = {}
        
        # Token selection analysis
        analysis['selected_tokens'] = top_k_mask
        analysis['selected_indices'] = selected_indices
        analysis['index_scores'] = index_scores
        analysis['selection_patterns'] = self._analyze_selection_patterns(top_k_mask)
        
        # Attention pattern analysis
        analysis['attention_patterns'] = self._extract_attention_patterns(Q, K, V, top_k_mask)
        
        # Indexer behavior analysis
        analysis['indexer_behavior'] = self._analyze_indexer_behavior(index_scores)
        
        # Efficiency metrics
        analysis['efficiency_metrics'] = self._compute_efficiency_metrics(x, top_k_mask, attn_output)
        
        # Content analysis
        analysis['content_analysis'] = self._analyze_content_selection(x, top_k_mask)
        
        # Pattern evolution
        analysis['pattern_evolution'] = self._analyze_pattern_evolution()
        
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
        
        # Positional bias analysis
        early_selection = top_k_mask[:, :seq_len//4].float().mean()
        middle_selection = top_k_mask[:, seq_len//4:3*seq_len//4].float().mean()
        late_selection = top_k_mask[:, 3*seq_len//4:].float().mean()
        
        # Selection diversity
        selection_diversity = self._compute_selection_diversity(top_k_mask)
        
        return {
            'selection_frequency': selection_freq,
            'selection_consistency': selection_consistency,
            'sparsity_ratio': sparsity_ratio,
            'positional_bias': {
                'early': early_selection.item(),
                'middle': middle_selection.item(),
                'late': late_selection.item()
            },
            'selection_diversity': selection_diversity
        }
    
    def _extract_attention_patterns(self, Q: torch.Tensor, K: torch.Tensor, 
                                  V: torch.Tensor, top_k_mask: torch.Tensor) -> torch.Tensor:
        """Extract actual attention patterns"""
        batch_size, seq_len, d_model = Q.shape
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(d_model)
        
        # Apply sparse mask
        attn_mask = torch.where(top_k_mask, 0, -float('inf'))
        scores = scores + attn_mask.unsqueeze(1)  # Add head dimension
        
        # Softmax to get attention weights
        attn_weights = F.softmax(scores, dim=-1)
        
        return attn_weights
    
    def _analyze_indexer_behavior(self, index_scores: torch.Tensor) -> Dict[str, Any]:
        """Analyze Lightning Indexer behavior"""
        batch_size, n_heads, seq_len, _ = index_scores.shape
        
        # Score statistics
        score_stats = {
            'mean': index_scores.mean().item(),
            'std': index_scores.std().item(),
            'min': index_scores.min().item(),
            'max': index_scores.max().item(),
            'median': index_scores.median().item()
        }
        
        # Head-wise analysis
        head_analysis = {}
        for head_idx in range(n_heads):
            head_scores = index_scores[:, head_idx, :, :]
            head_analysis[head_idx] = {
                'mean': head_scores.mean().item(),
                'std': head_scores.std().item(),
                'sparsity': (head_scores < 0.01).float().mean().item(),
                'concentration': self._compute_concentration(head_scores)
            }
        
        # Score distribution analysis
        score_distribution = self._analyze_score_distribution(index_scores)
        
        # Head specialization analysis
        head_specialization = self._analyze_head_specialization(index_scores)
        
        return {
            'score_stats': score_stats,
            'head_analysis': head_analysis,
            'score_distribution': score_distribution,
            'head_specialization': head_specialization
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
        
        # Attention efficiency
        attention_efficiency = self._compute_attention_efficiency(attn_output)
        
        return {
            'cost_reduction': cost_reduction,
            'memory_reduction': memory_reduction,
            'sparsity_ratio': 1.0 - top_k_mask.float().mean().item(),
            'attention_efficiency': attention_efficiency
        }
    
    def _analyze_content_selection(self, x: torch.Tensor, top_k_mask: torch.Tensor) -> Dict[str, Any]:
        """Analyze what types of content get selected"""
        batch_size, seq_len, d_model = x.shape
        
        # Analyze selection based on input features
        selected_inputs = x[top_k_mask]
        non_selected_inputs = x[~top_k_mask]
        
        # Feature statistics
        selected_stats = {
            'mean': selected_inputs.mean().item(),
            'std': selected_inputs.std().item(),
            'norm': selected_inputs.norm().item()
        }
        
        non_selected_stats = {
            'mean': non_selected_inputs.mean().item(),
            'std': non_selected_inputs.std().item(),
            'norm': non_selected_inputs.norm().item()
        }
        
        # Selection bias analysis
        selection_bias = {
            'mean_bias': selected_stats['mean'] - non_selected_stats['mean'],
            'std_bias': selected_stats['std'] - non_selected_stats['std'],
            'norm_bias': selected_stats['norm'] - non_selected_stats['norm']
        }
        
        return {
            'selected_stats': selected_stats,
            'non_selected_stats': non_selected_stats,
            'selection_bias': selection_bias
        }
    
    def _analyze_pattern_evolution(self) -> Dict[str, Any]:
        """Analyze how patterns evolve over time"""
        if len(self.selection_history) < 2:
            return {'evolution': 'insufficient_data'}
        
        # Analyze selection pattern evolution
        recent_patterns = self.selection_history[-10:]  # Last 10 patterns
        early_patterns = self.selection_history[:10]    # First 10 patterns
        
        if len(recent_patterns) >= 2 and len(early_patterns) >= 2:
            # Compute pattern similarity
            recent_similarity = self._compute_pattern_similarity(recent_patterns)
            early_similarity = self._compute_pattern_similarity(early_patterns)
            
            # Compute pattern stability
            pattern_stability = self._compute_pattern_stability()
            
            return {
                'recent_similarity': recent_similarity,
                'early_similarity': early_similarity,
                'pattern_stability': pattern_stability,
                'evolution_trend': 'stabilizing' if pattern_stability > 0.8 else 'evolving'
            }
        
        return {'evolution': 'insufficient_data'}
    
    def _compute_selection_diversity(self, top_k_mask: torch.Tensor) -> float:
        """Compute diversity of token selection"""
        batch_size, seq_len = top_k_mask.shape
        
        # Compute selection frequency
        selection_freq = top_k_mask.float().mean(dim=0)
        
        # Compute entropy of selection distribution
        selection_freq = selection_freq + 1e-8  # Avoid log(0)
        entropy = -torch.sum(selection_freq * torch.log(selection_freq))
        
        # Normalize by maximum possible entropy
        max_entropy = torch.log(torch.tensor(seq_len, dtype=torch.float))
        normalized_entropy = entropy / max_entropy
        
        return normalized_entropy.item()
    
    def _compute_concentration(self, scores: torch.Tensor) -> float:
        """Compute concentration of scores"""
        # Compute Gini coefficient as measure of concentration
        sorted_scores = torch.sort(scores.flatten())[0]
        n = len(sorted_scores)
        cumsum = torch.cumsum(sorted_scores, dim=0)
        
        gini = (n + 1 - 2 * torch.sum(cumsum) / cumsum[-1]) / n
        return gini.item()
    
    def _analyze_score_distribution(self, index_scores: torch.Tensor) -> Dict[str, Any]:
        """Analyze distribution of indexer scores"""
        all_scores = index_scores.flatten()
        
        # Compute percentiles
        percentiles = {
            '25th': torch.quantile(all_scores, 0.25).item(),
            '50th': torch.quantile(all_scores, 0.50).item(),
            '75th': torch.quantile(all_scores, 0.75).item(),
            '90th': torch.quantile(all_scores, 0.90).item(),
            '95th': torch.quantile(all_scores, 0.95).item(),
            '99th': torch.quantile(all_scores, 0.99).item()
        }
        
        # Compute skewness and kurtosis
        mean = all_scores.mean()
        std = all_scores.std()
        skewness = torch.mean(((all_scores - mean) / std) ** 3).item()
        kurtosis = torch.mean(((all_scores - mean) / std) ** 4).item()
        
        return {
            'percentiles': percentiles,
            'skewness': skewness,
            'kurtosis': kurtosis
        }
    
    def _analyze_head_specialization(self, index_scores: torch.Tensor) -> Dict[str, Any]:
        """Analyze specialization of indexer heads"""
        batch_size, n_heads, seq_len, _ = index_scores.shape
        
        # Compute head similarity matrix
        head_similarities = torch.zeros(n_heads, n_heads)
        for i in range(n_heads):
            for j in range(n_heads):
                head_i = index_scores[:, i, :, :].flatten()
                head_j = index_scores[:, j, :, :].flatten()
                similarity = F.cosine_similarity(head_i.unsqueeze(0), head_j.unsqueeze(0))
                head_similarities[i, j] = similarity
        
        # Compute specialization metrics
        avg_similarity = head_similarities.mean().item()
        max_similarity = head_similarities.max().item()
        min_similarity = head_similarities.min().item()
        
        return {
            'avg_similarity': avg_similarity,
            'max_similarity': max_similarity,
            'min_similarity': min_similarity,
            'specialization_score': 1.0 - avg_similarity  # Higher = more specialized
        }
    
    def _compute_attention_efficiency(self, attn_output: torch.Tensor) -> float:
        """Compute efficiency of attention computation"""
        # Simple efficiency metric based on output variance
        output_variance = attn_output.var().item()
        return output_variance
    
    def _compute_pattern_similarity(self, patterns: List[torch.Tensor]) -> float:
        """Compute similarity between patterns"""
        if len(patterns) < 2:
            return 0.0
        
        similarities = []
        for i in range(len(patterns)):
            for j in range(i + 1, len(patterns)):
                # Flatten patterns for comparison
                p1 = patterns[i].flatten()
                p2 = patterns[j].flatten()
                
                # Compute cosine similarity
                similarity = F.cosine_similarity(p1.unsqueeze(0), p2.unsqueeze(0))
                similarities.append(similarity.item())
        
        return np.mean(similarities) if similarities else 0.0
    
    def _compute_pattern_stability(self) -> float:
        """Compute stability of patterns over time"""
        if len(self.selection_history) < 3:
            return 0.0
        
        # Compute stability as inverse of change rate
        changes = []
        for i in range(1, len(self.selection_history)):
            prev_pattern = self.selection_history[i-1]
            curr_pattern = self.selection_history[i]
            
            # Compute change rate
            change = torch.abs(curr_pattern.float() - prev_pattern.float()).mean()
            changes.append(change.item())
        
        # Stability is inverse of average change
        avg_change = np.mean(changes)
        stability = 1.0 / (1.0 + avg_change)  # Normalize to [0, 1]
        
        return stability
    
    def store_analysis(self, analysis: Dict[str, Any]):
        """Store analysis for pattern evolution tracking"""
        # Store selection patterns
        if 'selected_tokens' in analysis:
            self.selection_history.append(analysis['selected_tokens'].clone())
        
        # Store attention weights
        if 'attention_patterns' in analysis:
            self.attention_weights_history.append(analysis['attention_patterns'].clone())
        
        # Store indexer scores
        if 'index_scores' in analysis:
            self.indexer_scores_history.append(analysis['index_scores'].clone())
        
        # Keep only recent history to avoid memory issues
        max_history = 100
        if len(self.selection_history) > max_history:
            self.selection_history = self.selection_history[-max_history:]
        if len(self.attention_weights_history) > max_history:
            self.attention_weights_history = self.attention_weights_history[-max_history:]
        if len(self.indexer_scores_history) > max_history:
            self.indexer_scores_history = self.indexer_scores_history[-max_history:]
    
    def get_analysis_summary(self) -> Dict[str, Any]:
        """Get summary of all analyses"""
        return {
            'total_analyses': len(self.selection_history),
            'pattern_evolution': self._analyze_pattern_evolution(),
            'selection_diversity_history': [self._compute_selection_diversity(p) for p in self.selection_history[-10:]],
            'attention_efficiency_history': [self._compute_attention_efficiency(w) for w in self.attention_weights_history[-10:]]
        }

# Import required modules (these would be imported from the main project)
try:
    from models.components import LightningIndexer, TopKTokenSelector, RotaryPositionalEmbeddings
    from interpretability.attention_visualizer import AttentionVisualizer
    from interpretability.pattern_analyzer import PatternAnalyzer
    from interpretability.indexer_interpreter import IndexerInterpreter
except ImportError:
    # Fallback implementations for testing
    class LightningIndexer(nn.Module):
        def __init__(self, d_model, n_heads, dim):
            super().__init__()
            self.n_heads = n_heads
            self.dim = dim
            self.proj = nn.Linear(d_model, n_heads * dim)
        
        def forward(self, x):
            batch_size, seq_len, d_model = x.shape
            scores = self.proj(x)
            scores = scores.view(batch_size, self.n_heads, seq_len, self.dim)
            return torch.softmax(scores, dim=-1)
    
    class TopKTokenSelector(nn.Module):
        def forward(self, scores, k):
            batch_size, n_heads, seq_len, _ = scores.shape
            # Select top-k tokens for each query
            top_k_values, top_k_indices = torch.topk(scores, k, dim=-1)
            mask = torch.zeros_like(scores, dtype=torch.bool)
            mask.scatter_(-1, top_k_indices, True)
            return mask, top_k_indices
    
    class RotaryPositionalEmbeddings(nn.Module):
        def __init__(self, d_model, max_seq_len):
            super().__init__()
            self.d_model = d_model
        
        def forward(self, x):
            return x
    
    class AttentionVisualizer:
        def visualize_token_selection(self, *args, **kwargs):
            print("Token selection visualization called")
        
        def visualize_sparse_attention(self, *args, **kwargs):
            print("Sparse attention visualization called")
        
        def visualize_indexer_analysis(self, *args, **kwargs):
            print("Indexer analysis visualization called")
    
    class PatternAnalyzer:
        pass
    
    class IndexerInterpreter:
        pass
