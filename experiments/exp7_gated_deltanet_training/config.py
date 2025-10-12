"""
Configuration for Gated DeltaNet Training Experiment using FLA
"""

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class ExperimentConfig:
    """Configuration for FLA DeltaNet experiment"""
    
    # Model Architecture
    vocab_size: int = 50257
    hidden_size: int = 256
    num_hidden_layers: int = 8
    num_attention_heads: int = 8
    max_position_embeddings: int = 2048
    
    # DeltaNet specific
    expand_k: float = 1.0  # Key expansion ratio
    expand_v: float = 1.0  # Value expansion ratio
    
    # MLP configuration
    hidden_ratio: int = 4  # MLP expansion ratio (intermediate_size = hidden_size * hidden_ratio)
    intermediate_size: Optional[int] = None  # If None, will use hidden_size * hidden_ratio
    
    # Hybrid Model Configuration
    # Set to None for pure DeltaNet, or specify layers to use standard attention
    # Example: {'layers': [3, 7, 11], 'window_size': 2048} for attention on specific layers
    attn_config: Optional[dict] = None
    
    # Regularization
    rms_norm_eps: float = 1e-6
    
    # Training
    batch_size: int = 4
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    max_steps: int = 2000
    warmup_steps: int = 100
    gradient_clip: float = 1.0
    
    # Optimizer
    betas: tuple = (0.9, 0.95)
    eps: float = 1e-8
    
    # Data
    max_seq_len: int = 256
    num_documents: int = 50000
    max_tokens: int = 100_000_000
    
    # Evaluation
    eval_interval: int = 100
    eval_batches: int = 50
    
    # Logging
    log_interval: int = 50
    
    # Checkpointing
    save_interval: int = 500
    checkpoint_dir: str = "checkpoints"
    
    # Device
    device: str = "cuda"
    
    # Seed
    seed: int = 42
    
    def __post_init__(self):
        """Set intermediate size if not provided"""
        if self.intermediate_size is None:
            self.intermediate_size = self.hidden_size * self.hidden_ratio


# RTX 4090 Hybrid Configuration (DeltaNet + Attention)

def get_rtx4090_optimized_config():
    """Optimized for RTX 4090 (24GB VRAM) - 1000 steps with no data repetition"""
    return ExperimentConfig(
        # Larger model to use more GPU
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        hidden_ratio=4,
        
        # Longer sequences and large batch for GPU saturation
        max_seq_len=1024,
        batch_size=32,
        
        # Training params - 1000 steps
        max_steps=1000,
        warmup_steps=100,  # 10% warmup
        learning_rate=1e-3,  # Best from 200-step ablation (val_loss=6.161)
        gradient_clip=1.0,
        
        # Data - NO REPETITION for 1000 steps
        # Tokens needed: 32 batch × 1024 seq × 1000 steps = 32,768,000 (32.8M)
        # With 2x safety margin = 65,536,000 (65.5M)
        # At ~750 tokens per document (3000 chars), need ~93,000 documents for 70M tokens
        num_documents=100_000,  # Increased from 10k to ensure sufficient unique data
        max_tokens=70_000_000,  # 70M tokens (2x safety margin)
        
        # Evaluation settings
        eval_interval=50,
        eval_batches=20,
        log_interval=10,
    )


def get_hybrid_rtx4090_config():
    """
    Hybrid RTX 4090 config with strategic attention placement
    DeltaNet for efficiency, attention at key positions for quality
    
    Architecture:
    - 12 layers total
    - Attention on layers [3, 7, 11] (25%, 58%, 92% through network)
    - DeltaNet on layers [0, 1, 2, 4, 5, 6, 8, 9, 10]
    """
    config = get_rtx4090_optimized_config()
    # For 12 layers: attention on [3, 7, 11] (25%, 58%, 92%)
    config.attn_config = {
        'layers': [3, 7, 11],
        'window_size': 2048,
        'qkv_bias': False,
        'rope_theta': 10000.0,
    }
    return config


# H100 Configurations
# ====================
# Base configuration for all H100 experiments

def get_h100_base_config():
    """Base H100 config (80GB VRAM) - shared by all experiments"""
    return ExperimentConfig(
        # Larger model leveraging H100's 80GB VRAM (3.3x more than 4090)
        hidden_size=1536,
        num_hidden_layers=24,
        num_attention_heads=24,
        hidden_ratio=4,
        
        # Longer sequences and larger batch
        max_seq_len=2048,
        batch_size=48,
        
        # Training params - 1000 steps
        max_steps=1000,
        warmup_steps=100,
        learning_rate=1e-3,
        gradient_clip=1.0,
        
        # Data - NO REPETITION for 1000 steps
        # Tokens needed: 48 batch × 2048 seq × 1000 steps = 98,304,000 (98.3M)
        # With 2x safety margin = 196,608,000 (196.6M)
        num_documents=300_000,
        max_tokens=200_000_000,  # 200M tokens
        
        # Evaluation settings
        eval_interval=50,
        eval_batches=20,
        log_interval=10,
    )


# H100 Experiment Variants
# =========================

def get_h100_deltanet_only():
    """
    Experiment 1: Pure DeltaNet (baseline)
    - All 24 layers use DeltaNet
    - O(n) complexity throughout
    - No attention layers
    """
    return get_h100_base_config()


def get_h100_transformer_only():
    """
    Experiment 2: Pure Transformer (full attention)
    - All 24 layers use standard attention
    - O(n²) complexity throughout
    - Baseline comparison for attention quality
    """
    config = get_h100_base_config()
    config.attn_config = {
        'layers': list(range(24)),  # All layers [0-23]
        'window_size': 4096,
        'qkv_bias': False,
        'rope_theta': 10000.0,
    }
    return config


def get_h100_hybrid_sparse():
    """
    Experiment 3: Hybrid Sparse (~17% attention)
    - Attention on 4 layers: [5, 11, 17, 23] (21%, 46%, 71%, 96% through network)
    - DeltaNet on 20 layers
    - Strategic placement: distributed throughout network
    """
    config = get_h100_base_config()
    config.attn_config = {
        'layers': [5, 11, 17, 23],
        'window_size': 4096,
        'qkv_bias': False,
        'rope_theta': 10000.0,
    }
    return config


def get_h100_hybrid_alternating():
    """
    Experiment 4: Hybrid Alternating (50% attention)
    - Attention on 12 layers: [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23] (every other)
    - DeltaNet on 12 layers: [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22]
    - Balanced mix throughout the network
    """
    config = get_h100_base_config()
    config.attn_config = {
        'layers': [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23],
        'window_size': 4096,
        'qkv_bias': False,
        'rope_theta': 10000.0,
    }
    return config


def get_h100_hybrid_late():
    """
    Experiment 5: Hybrid Late (33% attention)
    - Attention on last 8 layers: [16, 17, 18, 19, 20, 21, 22, 23] (final 1/3)
    - DeltaNet on first 16 layers: [0-15]
    - Hypothesis: DeltaNet for early processing, attention for refinement
    """
    config = get_h100_base_config()
    config.attn_config = {
        'layers': [16, 17, 18, 19, 20, 21, 22, 23],
        'window_size': 4096,
        'qkv_bias': False,
        'rope_theta': 10000.0,
    }
    return config


# Legacy aliases for backward compatibility
get_h100_optimized_config = get_h100_base_config
get_hybrid_h100_config = get_h100_hybrid_sparse
