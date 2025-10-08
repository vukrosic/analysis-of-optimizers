"""
Experiment 6: Qwen3-Next with DeepSeek Sparse Attention

Three variants:
1. Baseline: Standard Qwen3-Next
2. DSA-Only: All attention replaced with DeepSeek Sparse Attention
3. Hybrid: DSA for full_attention, Gated DeltaNet for linear_attention
"""

from .models import BaselineQwen3, DSAQwen3, HybridQwen3, create_model
from .config import ExperimentConfig, SMALL_CONFIG, MEDIUM_CONFIG, LARGE_CONFIG

__all__ = [
    'BaselineQwen3',
    'DSAQwen3', 
    'HybridQwen3',
    'create_model',
    'ExperimentConfig',
    'SMALL_CONFIG',
    'MEDIUM_CONFIG',
    'LARGE_CONFIG',
]

