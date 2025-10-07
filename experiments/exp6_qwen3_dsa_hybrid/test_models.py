"""
Quick test script to verify all three model variants can be instantiated
"""

import torch
import sys
import os

# Add root to path
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
sys.path.insert(0, root_dir)

from config import SMALL_CONFIG
from models import create_model


def test_model_creation():
    """Test that all three variants can be created"""
    variants = ['baseline', 'dsa', 'hybrid']
    
    print("Testing model creation...\n")
    
    for variant in variants:
        print(f"Testing {variant}...")
        try:
            model = create_model(variant, SMALL_CONFIG)
            num_params = sum(p.numel() for p in model.parameters())
            print(f"  ✓ {variant}: {num_params:,} parameters")
        except Exception as e:
            print(f"  ✗ {variant}: Failed with error: {e}")
            import traceback
            traceback.print_exc()
    
    print("\nDone!")


def test_forward_pass():
    """Test forward pass for all variants"""
    variants = ['baseline', 'dsa', 'hybrid']
    batch_size = 2
    seq_len = 64
    
    print("\nTesting forward pass...\n")
    
    for variant in variants:
        print(f"Testing {variant} forward pass...")
        try:
            model = create_model(variant, SMALL_CONFIG)
            
            # Create dummy input
            input_ids = torch.randint(0, SMALL_CONFIG.vocab_size, (batch_size, seq_len))
            labels = input_ids.clone()
            
            # Forward pass
            with torch.no_grad():
                outputs = model(input_ids=input_ids, labels=labels)
            
            loss = outputs.loss
            logits = outputs.logits
            
            print(f"  ✓ {variant}: Loss={loss.item():.4f}, Logits shape={logits.shape}")
        
        except Exception as e:
            print(f"  ✗ {variant}: Failed with error: {e}")
            import traceback
            traceback.print_exc()
    
    print("\nDone!")


if __name__ == "__main__":
    print("="*60)
    print("Model Variant Testing")
    print("="*60)
    
    test_model_creation()
    test_forward_pass()
    
    print("\n" + "="*60)
    print("All tests complete!")
    print("="*60)

