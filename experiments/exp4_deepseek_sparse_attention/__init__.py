"""
Experiment 4: DeepSeek Sparse Attention (DSA) Implementation & Comparison

This experiment implements the DeepSeek Sparse Attention mechanism from the 
DeepSeek-V3.2-Exp paper and compares it with classic dense attention using 
the GLM4 MoE architecture.

Main components:
- sparse_attention.py: Lightning indexer and sparse attention mechanism
- models.py: Sparse and classic attention model definitions
- run_experiment.py: Main experiment script
- visualize_attention.py: Attention pattern visualization
- config.py: Configuration presets

Quick start:
    python run_experiment.py
"""

from .sparse_attention import (
    LightningIndexer,
    TopKTokenSelector,
    DeepSeekSparseAttention,
    SparseAttentionMetrics
)

from .models import (
    SparseAttentionMoELLM,
    ClassicAttentionMoELLM,
    create_sparse_model,
    create_classic_model,
    count_parameters
)

from .config import (
    SparseAttentionConfig,
    ClassicAttentionConfig,
    get_sparse_config_small,
    get_sparse_config_medium,
    get_sparse_config_large,
    EXP4_OPTIMAL_PRESET
)

__all__ = [
    # Sparse Attention Components
    'LightningIndexer',
    'TopKTokenSelector',
    'DeepSeekSparseAttention',
    'SparseAttentionMetrics',
    
    # Models
    'SparseAttentionMoELLM',
    'ClassicAttentionMoELLM',
    'create_sparse_model',
    'create_classic_model',
    'count_parameters',
    
    # Configs
    'SparseAttentionConfig',
    'ClassicAttentionConfig',
    'get_sparse_config_small',
    'get_sparse_config_medium',
    'get_sparse_config_large',
    'EXP4_OPTIMAL_PRESET',
]

__version__ = '1.0.0'
__author__ = 'DeepSeek Sparse Attention Research'
