"""
Test script to demonstrate hybrid DeltaNet + Attention models
Shows the architecture for different hybrid configurations
"""

import sys
import os

root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, root_dir)

from experiments.exp7_gated_deltanet_training.config import (
    get_medium_config,
    get_hybrid_config_alternating,
    get_hybrid_config_sparse_attention,
    get_hybrid_config_attention_last,
)
from experiments.exp7_gated_deltanet_training.models import GatedDeltaNetWrapper


def test_configuration(config_name, config):
    """Test a specific configuration and print its architecture"""
    print("\n" + "="*80)
    print(f"Testing: {config_name}")
    print("="*80)
    
    # Create model
    model_wrapper = GatedDeltaNetWrapper(config)
    
    # Print model info
    model_wrapper.print_info()
    
    # Print hybrid config details if it's a hybrid model
    if hasattr(config, 'attn_config') and config.attn_config is not None:
        print("\nHybrid Configuration Details:")
        print(f"  Attention Layers: {config.attn_config.get('layers', [])}")
        print(f"  Window Size: {config.attn_config.get('window_size', 2048)}")
        print(f"  QKV Bias: {config.attn_config.get('qkv_bias', False)}")
        print(f"  RoPE Theta: {config.attn_config.get('rope_theta', 10000.0)}")


def main():
    """Run tests for all configurations"""
    
    print("="*80)
    print("HYBRID MODEL DEMONSTRATION")
    print("Experiment 7: DeltaNet + Standard Attention")
    print("="*80)
    
    # Test configurations
    configurations = [
        ("Pure DeltaNet (Baseline)", get_medium_config()),
        ("Hybrid: Alternating (50% Attention)", get_hybrid_config_alternating()),
        ("Hybrid: Sparse (25% Attention)", get_hybrid_config_sparse_attention()),
        ("Hybrid: Attention Last (25% Attention)", get_hybrid_config_attention_last()),
    ]
    
    for config_name, config in configurations:
        test_configuration(config_name, config)
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print("\nHybrid architectures successfully combine:")
    print("  ✓ Gated DeltaNet layers (O(n) complexity)")
    print("  ✓ Standard attention layers (O(n²) complexity)")
    print("\nNext steps:")
    print("  1. Train pure DeltaNet baseline: python run_experiment.py --config medium")
    print("  2. Train hybrid models: python run_experiment.py --config hybrid_alternating")
    print("  3. Compare validation loss and training speed")
    print("  4. Analyze which hybrid pattern works best")
    print("="*80)


if __name__ == '__main__':
    main()

