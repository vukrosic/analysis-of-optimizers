"""
Comprehensive Experiment: Test all attention variants
- 4 Original: full_attention + linear_attention
- 4 DSA: dsa_attention + linear_attention

Total: 8 architecture variants
"""

import torch
import sys
import os
import time
import json
from pathlib import Path

root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, root_dir)

from experiments.exp6_qwen3_dsa_hybrid.enhanced_models import EnhancedQwen3NextForCausalLM
from models.qwen3_next.configuration_qwen3_next import Qwen3NextConfig
from models.qwen3_next.modular_qwen3_next import Qwen3NextForCausalLM
from data.loader import load_and_cache_data
from data.dataset import TextTokenDataset
from utils.helpers import set_seed
from torch.utils.data import DataLoader

# Import training functions from baseline test
from experiments.exp6_qwen3_dsa_hybrid.test_qwen3_baseline import (
    count_parameters, train_epoch, evaluate
)


# 8 Attention Patterns to test
ALL_PATTERNS = {
    # Original patterns (F = full_attention, L = linear_attention)
    "1_sandwich": ["linear_attention", "full_attention", "full_attention", "linear_attention"],
    "2_alternating": ["full_attention", "linear_attention", "full_attention", "linear_attention"],
    "3_linear_first": ["linear_attention", "linear_attention", "full_attention", "full_attention"],
    "4_full_first": ["full_attention", "full_attention", "linear_attention", "linear_attention"],
    
    # DSA patterns (D = dsa_attention, L = linear_attention)
    "5_dsa_sandwich": ["linear_attention", "dsa_attention", "dsa_attention", "linear_attention"],
    "6_dsa_alternating": ["dsa_attention", "linear_attention", "dsa_attention", "linear_attention"],
    "7_dsa_linear_first": ["linear_attention", "linear_attention", "dsa_attention", "dsa_attention"],
    "8_dsa_full_first": ["dsa_attention", "dsa_attention", "linear_attention", "linear_attention"],
}


def test_pattern(pattern_name, layer_types, use_dsa=False):
    """Test a specific attention pattern"""
    print(f"\n{'='*70}")
    print(f"Pattern {pattern_name.upper()}")
    print(f"Layers: {' → '.join([t[0].upper() for t in layer_types])}")
    print(f"Details: {layer_types}")
    print(f"{'='*70}\n")
    
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Configuration
    config = Qwen3NextConfig(
        vocab_size=50257,
        hidden_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=512,
        max_position_embeddings=512,
        rope_theta=10000.0,
        attention_dropout=0.1,
        hidden_dropout_prob=0.1,
        rms_norm_eps=1e-6,
        head_dim=32,
        partial_rotary_factor=1.0,
        layer_types=layer_types,
        # Linear attention config
        linear_num_value_heads=2,
        linear_num_key_heads=2,
        linear_key_head_dim=64,
        linear_value_head_dim=64,
        linear_conv_kernel_dim=4,
        # MoE configuration
        num_experts=4,
        num_local_experts=4,
        num_experts_per_tok=2,
        router_jitter_noise=0.0,
        decoder_sparse_step=2,
        moe_intermediate_size=256,
        shared_expert_intermediate_size=0,
        mlp_only_layers=[],
        # DSA config
        indexer_heads=4,
        indexer_dim=32,
        sparse_top_k=64,  # Reduced for small models
    )
    
    # Load cached data
    from dataclasses import dataclass
    @dataclass
    class SimpleConfig:
        num_documents: int = 1000
        max_tokens: int = 2_000_000
        vocab_size: int = 50257
    
    data_config = SimpleConfig()
    texts, tokenizer, tokens = load_and_cache_data(data_config)
    config.vocab_size = len(tokenizer)
    
    max_seq_len = 128
    dataset = TextTokenDataset(tokens, max_seq_len)
    
    val_size = len(dataset) // 10
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, num_workers=0)
    
    # Create model - use Enhanced model if DSA is involved
    if use_dsa:
        model = EnhancedQwen3NextForCausalLM(config).to(device)
    else:
        model = Qwen3NextForCausalLM(config).to(device)
    
    num_params = count_parameters(model)
    print(f"Parameters: {num_params:,}")
    
    # Verify layer types
    print("Layer verification: ", end="")
    for i, layer in enumerate(model.model.layers):
        has_self_attn = hasattr(layer, 'self_attn')
        has_linear_attn = hasattr(layer, 'linear_attn')
        has_dsa_attn = hasattr(layer, 'dsa_attn')
        
        if has_self_attn:
            actual = "F"
        elif has_linear_attn:
            actual = "L"
        elif has_dsa_attn:
            actual = "D"
        else:
            actual = "?"
        print(actual, end=" ")
    print()
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), weight_decay=0.1)
    
    # Train for 300 steps
    max_steps = 300
    start_time = time.time()
    total_steps = 0
    
    train_loss, total_steps = train_epoch(model, train_loader, optimizer, device, max_steps, total_steps)
    val_metrics = evaluate(model, val_loader, device, max_batches=100)
    
    training_time = time.time() - start_time
    
    result = {
        'pattern': pattern_name,
        'layer_types': layer_types,
        'uses_dsa': use_dsa,
        'num_parameters': num_params,
        'training_time': training_time,
        'train_loss': train_loss,
        'val_loss': val_metrics['loss'],
        'val_accuracy': val_metrics['accuracy'],
        'val_perplexity': val_metrics['perplexity'],
    }
    
    print(f"\n{pattern_name.upper()} Results:")
    print(f"  Train Loss: {train_loss:.4f}")
    print(f"  Val Loss: {val_metrics['loss']:.4f}")
    print(f"  Val Acc: {val_metrics['accuracy']:.4f} ({val_metrics['accuracy']*100:.2f}%)")
    print(f"  Val PPL: {val_metrics['perplexity']:.2f}")
    print(f"  Time: {training_time:.1f}s")
    
    # Clear GPU
    del model, optimizer
    torch.cuda.empty_cache()
    
    return result


def main():
    print("="*70)
    print("COMPREHENSIVE ATTENTION VARIANTS EXPERIMENT")
    print("="*70)
    print("\nTesting 8 patterns:")
    print("  • 4 Original: Full Attention (F) + Linear Attention (L)")
    print("  • 4 DSA: DeepSeek Sparse Attention (D) + Linear Attention (L)")
    print("\nConfiguration:")
    print("  • 4 layers, 128 hidden dim, 14M parameters")
    print("  • 4 experts, top-2 routing, MoE every 2 layers")
    print("  • 300 training steps")
    print()
    
    results = {}
    for pattern_name, layer_types in ALL_PATTERNS.items():
        # Check if this pattern uses DSA
        use_dsa = "dsa" in pattern_name or any("dsa" in lt for lt in layer_types)
        
        try:
            result = test_pattern(pattern_name, layer_types, use_dsa)
            results[pattern_name] = result
        except Exception as e:
            print(f"\n❌ ERROR testing {pattern_name}: {e}")
            import traceback
            traceback.print_exc()
            torch.cuda.empty_cache()
            continue
    
    # Summary Tables
    print(f"\n{'='*70}")
    print("FINAL RESULTS - ALL 8 PATTERNS")
    print(f"{'='*70}\n")
    
    # Sort by validation loss (best first)
    sorted_results = sorted(results.items(), key=lambda x: x[1]['val_loss'])
    
    print(f"{'Rank':<6} {'Pattern':<20} {'Type':<8} {'Val Loss':<10} {'Val Acc':<10} {'Val PPL':<12} {'Time':<8}")
    print("-" * 90)
    
    for rank, (pattern_name, result) in enumerate(sorted_results, 1):
        pattern_type = "DSA" if result['uses_dsa'] else "Original"
        medal = "🏆" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else f"{rank}."
        
        print(f"{medal:<6} {pattern_name:<20} {pattern_type:<8} "
              f"{result['val_loss']:<10.4f} {result['val_accuracy']:<10.4f} "
              f"{result['val_perplexity']:<12.2f} {result['training_time']:<8.1f}s")
    
    # Category comparison
    print(f"\n{'='*70}")
    print("CATEGORY AVERAGES")
    print(f"{'='*70}\n")
    
    original_results = [r for r in results.values() if not r['uses_dsa']]
    dsa_results = [r for r in results.values() if r['uses_dsa']]
    
    if original_results:
        avg_orig_loss = sum(r['val_loss'] for r in original_results) / len(original_results)
        avg_orig_acc = sum(r['val_accuracy'] for r in original_results) / len(original_results)
        avg_orig_ppl = sum(r['val_perplexity'] for r in original_results) / len(original_results)
        avg_orig_time = sum(r['training_time'] for r in original_results) / len(original_results)
        
        print(f"Original (Full + Linear Attention):")
        print(f"  Avg Val Loss: {avg_orig_loss:.4f}")
        print(f"  Avg Val Acc:  {avg_orig_acc:.4f} ({avg_orig_acc*100:.2f}%)")
        print(f"  Avg Val PPL:  {avg_orig_ppl:.2f}")
        print(f"  Avg Time:     {avg_orig_time:.1f}s")
    
    if dsa_results:
        avg_dsa_loss = sum(r['val_loss'] for r in dsa_results) / len(dsa_results)
        avg_dsa_acc = sum(r['val_accuracy'] for r in dsa_results) / len(dsa_results)
        avg_dsa_ppl = sum(r['val_perplexity'] for r in dsa_results) / len(dsa_results)
        avg_dsa_time = sum(r['training_time'] for r in dsa_results) / len(dsa_results)
        
        print(f"\nDSA (DeepSeek Sparse + Linear Attention):")
        print(f"  Avg Val Loss: {avg_dsa_loss:.4f}")
        print(f"  Avg Val Acc:  {avg_dsa_acc:.4f} ({avg_dsa_acc*100:.2f}%)")
        print(f"  Avg Val PPL:  {avg_dsa_ppl:.2f}")
        print(f"  Avg Time:     {avg_dsa_time:.1f}s")
        
        if original_results:
            print(f"\nDSA vs Original:")
            print(f"  Loss diff:  {avg_dsa_loss - avg_orig_loss:+.4f}")
            print(f"  Acc diff:   {(avg_dsa_acc - avg_orig_acc)*100:+.2f}%")
            print(f"  PPL diff:   {avg_dsa_ppl - avg_orig_ppl:+.2f}")
            print(f"  Time diff:  {avg_dsa_time - avg_orig_time:+.1f}s")
    
    # Save results
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True, parents=True)
    
    with open(results_dir / 'comprehensive_comparison.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*70}")
    print(f"Results saved to: {results_dir / 'comprehensive_comparison.json'}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()

