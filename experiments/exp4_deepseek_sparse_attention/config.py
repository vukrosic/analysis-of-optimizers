"""
Configuration for DeepSeek Sparse Attention Experiment
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class SparseAttentionConfig:
    """Configuration for Sparse Attention Model"""
    
    # Model Architecture
    vocab_size: int
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 6
    d_ff: int = 512
    max_seq_len: int = 128
    
    # MoE Configuration
    num_experts: int = 4
    expert_top_k: int = 2
    
    # Sparse Attention Configuration
    indexer_heads: int = 4       # Number of lightning indexer heads
    indexer_dim: int = 64         # Indexer query/key dimension
    sparse_top_k: int = 64        # Number of tokens to select (for seq_len=128)
    
    # Training Configuration
    batch_size: int = 16
    warmup_steps: int = 200       # Indexer warmup steps
    warmup_lr: float = 1e-3       # Warmup learning rate
    sparse_steps: int = 3000      # Sparse training steps
    learning_rate: float = 3e-3   # Main learning rate
    eval_every: int = 100
    
    # Data Configuration
    max_tokens: int = 50000
    num_documents: int = 1000
    
    # Other
    dropout: float = 0.1
    load_balancing_weight: float = 0.01
    device: str = 'cuda'
    results_dir: str = 'results'


@dataclass
class ClassicAttentionConfig:
    """Configuration for Classic Attention Model (Baseline)"""
    
    # Model Architecture
    vocab_size: int
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 6
    d_ff: int = 512
    max_seq_len: int = 128
    
    # MoE Configuration
    num_experts: int = 4
    expert_top_k: int = 2
    
    # Training Configuration
    batch_size: int = 16
    max_steps: int = 3000
    learning_rate: float = 3e-3
    eval_every: int = 100
    
    # Data Configuration
    max_tokens: int = 50000
    num_documents: int = 1000
    
    # Other
    dropout: float = 0.1
    load_balancing_weight: float = 0.01
    device: str = 'cuda'
    results_dir: str = 'results'


def get_sparse_config_small(vocab_size: int) -> dict:
    """Small sparse attention configuration"""
    return {
        'vocab_size': vocab_size,
        'd_model': 128,
        'n_heads': 4,
        'n_layers': 4,
        'd_ff': 256,
        'max_seq_len': 64,
        'num_experts': 2,
        'expert_top_k': 1,
        'indexer_heads': 2,
        'indexer_dim': 32,
        'sparse_top_k': 32,
        'batch_size': 32,
        'warmup_steps': 100,
        'sparse_steps': 1500,
        'learning_rate': 3e-3,
    }


def get_sparse_config_medium(vocab_size: int) -> dict:
    """Medium sparse attention configuration (default from Exp 3)"""
    return {
        'vocab_size': vocab_size,
        'd_model': 256,
        'n_heads': 8,
        'n_layers': 6,
        'd_ff': 512,
        'max_seq_len': 128,
        'num_experts': 4,
        'expert_top_k': 2,
        'indexer_heads': 4,
        'indexer_dim': 64,
        'sparse_top_k': 64,
        'batch_size': 16,
        'warmup_steps': 200,
        'sparse_steps': 3000,
        'learning_rate': 3e-3,
    }


def get_sparse_config_large(vocab_size: int) -> dict:
    """Large sparse attention configuration"""
    return {
        'vocab_size': vocab_size,
        'd_model': 512,
        'n_heads': 16,
        'n_layers': 12,
        'd_ff': 1024,
        'max_seq_len': 256,
        'num_experts': 8,
        'expert_top_k': 2,
        'indexer_heads': 8,
        'indexer_dim': 128,
        'sparse_top_k': 128,
        'batch_size': 8,
        'warmup_steps': 300,
        'sparse_steps': 5000,
        'learning_rate': 1e-3,
    }


def get_config_for_sequence_length(vocab_size: int, seq_len: int) -> dict:
    """
    Get appropriate configuration for a given sequence length
    
    The sparse_top_k is adjusted to maintain ~50% sparsity
    """
    base_config = get_sparse_config_medium(vocab_size)
    
    # Adjust sparse_top_k proportionally
    base_seq_len = 128
    base_top_k = 64
    
    new_top_k = int(base_top_k * (seq_len / base_seq_len))
    
    base_config['max_seq_len'] = seq_len
    base_config['sparse_top_k'] = new_top_k
    
    return base_config


# Presets based on DeepSeek-V3.2-Exp paper
DEEPSEEK_V32_PRESET = {
    'indexer_heads': 4,           # Small number of heads as per paper
    'sparse_top_k': 2048,         # From paper (for 128K context)
    'warmup_steps': 1000,         # From paper
    'warmup_lr': 1e-3,            # From paper
    'sparse_lr': 7.3e-6,          # From paper (for continued training)
    'activation': 'relu',         # ReLU for throughput as per paper
}

# Experiment 4 optimal preset (adapted for our scale)
EXP4_OPTIMAL_PRESET = {
    'indexer_heads': 4,
    'indexer_dim': 64,
    'sparse_top_k': 64,           # ~50% sparsity for seq_len=128
    'warmup_steps': 200,
    'warmup_lr': 1e-3,
    'sparse_steps': 3000,
    'learning_rate': 3e-3,
}
