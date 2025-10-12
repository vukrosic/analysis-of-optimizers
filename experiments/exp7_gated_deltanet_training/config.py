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


# ============================================================================
# ARCHIVED: RTX 4090 Configurations (Shared Memory Issues with GatedDeltaNet)
# ============================================================================
# Note: GatedDeltaNet kernels exceed RTX 4090's shared memory limits
# Use H100 configurations instead

def get_rtx4090_optimized_config():
    """Optimized for RTX 4090 (24GB VRAM) - 1000 steps with no data repetition"""
    return ExperimentConfig(
        # Larger model to use more GPU
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        hidden_ratio=4,
        
        # Reduced batch size to fit GatedDeltaNet in memory (larger than DeltaNet)
        max_seq_len=1024,
        batch_size=16,  # Reduced from 32 to 16 for GatedDeltaNet
        
        # Training params - 1000 steps
        max_steps=1000,
        warmup_steps=100,  # 10% warmup
        learning_rate=1e-3,  # Best from 200-step ablation (val_loss=6.161)
        gradient_clip=1.0,
        
        # Data - NO REPETITION for 1000 steps
        # Tokens needed: 16 batch × 1024 seq × 1000 steps = 16,384,000 (16.4M)
        # With 2x safety margin = 32,768,000 (32.8M)
        # At ~750 tokens per document (3000 chars), need ~50,000 documents for 35M tokens
        num_documents=100_000,  # More than enough to ensure sufficient unique data
        max_tokens=70_000_000,  # 70M tokens (4x safety margin)
        
        # Evaluation settings
        eval_interval=50,
        eval_batches=20,
        log_interval=10,
    )


def get_hybrid_rtx4090_config():
    """
    Hybrid RTX 4090 config with strategic sparse attention placement
    DeltaNet for efficiency, attention at key positions for quality
    
    Architecture:
    - 12 layers total
    - Attention on layers [3, 7, 11] (25% - 3/12 layers)
    - DeltaNet on layers [0, 1, 2, 4, 5, 6, 8, 9, 10]
    - Strategic placement: early-mid, mid, near-end
    """
    config = get_rtx4090_optimized_config()
    config.attn_config = {
        'layers': [3, 7, 11],
        'window_size': 2048,
        'qkv_bias': False,
        'rope_theta': 10000.0,
    }
    return config


def get_hybrid_rtx4090_alternating():
    """
    Hybrid RTX 4090 config with alternating layers
    Balanced mix of DeltaNet and attention throughout
    
    Architecture:
    - 12 layers total
    - Attention on layers [1, 3, 5, 7, 9, 11] (50% - 6/12 layers)
    - DeltaNet on layers [0, 2, 4, 6, 8, 10]
    - Alternating pattern for balanced compute/quality trade-off
    """
    config = get_rtx4090_optimized_config()
    config.attn_config = {
        'layers': [1, 3, 5, 7, 9, 11],
        'window_size': 2048,
        'qkv_bias': False,
        'rope_theta': 10000.0,
    }
    return config


# H100 Configurations
# ====================
# Base configuration for all H100 experiments

def get_h100_base_config():
    """Base H100 config (80GB VRAM) - shared by all experiments
    
    Architecture matches exp6 for stability (proven to work):
    - Same model size: 768 hidden, 12 layers (~60M params)
    - Optimized batch size: 120 (for ~90% H100 memory utilization)
    - Sequence length: 1024 (proven stable)
    """
    return ExperimentConfig(
        # Model architecture - same as exp6 for stability
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        hidden_ratio=4,
        
        # Sequence and batch configuration - optimized for H100
        max_seq_len=1024,
        batch_size=120,
        
        # Training params - 1000 steps
        max_steps=1000,
        warmup_steps=100,
        learning_rate=1e-3,
        gradient_clip=1.0,
        
        # Data - NO REPETITION for 1000 steps
        # Tokens needed: 120 batch × 1024 seq × 1000 steps = 122,880,000 (122.9M)
        # With 2x safety margin = 245,760,000 (245.8M)
        num_documents=50_000,
        max_tokens=200_000_000,  # 200M tokens (1.6x safety margin)
        
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
    - All 12 layers use standard attention
    - O(n²) complexity throughout
    - Baseline comparison for attention quality
    """
    config = get_h100_base_config()
    config.attn_config = {
        'layers': list(range(12)),  # All layers [0-11]
        'window_size': 2048,
        'qkv_bias': False,
        'rope_theta': 10000.0,
    }
    return config


def get_h100_hybrid_sparse():
    """
    Experiment 3: Hybrid Sparse (~17% attention)
    - Attention on 2 layers: [5, 11] (42%, 92% through network)
    - DeltaNet on 10 layers
    - Strategic placement: mid and late layers
    """
    config = get_h100_base_config()
    config.attn_config = {
        'layers': [5, 11],
        'window_size': 2048,
        'qkv_bias': False,
        'rope_theta': 10000.0,
    }
    return config


def get_h100_hybrid_alternating():
    """
    Experiment 4: Hybrid Alternating (50% attention)
    - Attention on 6 layers: [1, 3, 5, 7, 9, 11] (every other)
    - DeltaNet on 6 layers: [0, 2, 4, 6, 8, 10]
    - Balanced mix throughout the network
    """
    config = get_h100_base_config()
    config.attn_config = {
        'layers': [1, 3, 5, 7, 9, 11],
        'window_size': 2048,
        'qkv_bias': False,
        'rope_theta': 10000.0,
    }
    return config


def get_h100_hybrid_late():
    """
    Experiment 5: Hybrid Late (33% attention)
    - Attention on last 4 layers: [8, 9, 10, 11] (final 1/3)
    - DeltaNet on first 8 layers: [0-7]
    - Hypothesis: DeltaNet for early processing, attention for refinement
    """
    config = get_h100_base_config()
    config.attn_config = {
        'layers': [8, 9, 10, 11],
        'window_size': 2048,
        'qkv_bias': False,
        'rope_theta': 10000.0,
    }
    return config


# Legacy aliases for backward compatibility
get_h100_optimized_config = get_h100_base_config
get_hybrid_h100_config = get_h100_hybrid_sparse
