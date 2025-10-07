"""
Test different attention layer patterns
Compare: 
1. Sandwich: linear, full, full, linear
2. Alternating: full, linear, full, linear  
3. Linear-first: linear, linear, full, full
4. Full-first: full, full, linear, linear
"""

import torch
import sys
import os
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, root_dir)

from experiments.exp6_qwen3_dsa_hybrid.test_qwen3_baseline import *
import json

PATTERNS = {
    "sandwich": ["linear_attention", "full_attention", "full_attention", "linear_attention"],
    "alternating": ["full_attention", "linear_attention", "full_attention", "linear_attention"],
    "linear_first": ["linear_attention", "linear_attention", "full_attention", "full_attention"],
    "full_first": ["full_attention", "full_attention", "linear_attention", "linear_attention"],
}

def test_pattern(pattern_name, layer_types):
    """Test a specific attention pattern"""
    print(f"\n{'='*60}")
    print(f"Testing Pattern: {pattern_name.upper()}")
    print(f"Layers: {layer_types}")
    print(f"{'='*60}\n")
    
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Configuration with the specific pattern
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
        layer_types=layer_types,  # Different pattern
        linear_num_value_heads=2,
        linear_num_key_heads=2,
        linear_key_head_dim=64,
        linear_value_head_dim=64,
        linear_conv_kernel_dim=4,
        num_experts=4,
        num_local_experts=4,
        num_experts_per_tok=2,
        router_jitter_noise=0.0,
        decoder_sparse_step=2,
        moe_intermediate_size=256,
        shared_expert_intermediate_size=0,
        mlp_only_layers=[],
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
    
    # Create model
    model = Qwen3NextForCausalLM(config).to(device)
    num_params = count_parameters(model)
    print(f"Parameters: {num_params:,}")
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), weight_decay=0.1)
    
    # Train for 300 steps (10x more)
    max_steps = 300
    start_time = time.time()
    total_steps = 0
    
    train_loss, total_steps = train_epoch(model, train_loader, optimizer, device, max_steps, total_steps)
    val_metrics = evaluate(model, val_loader, device, max_batches=100)
    
    training_time = time.time() - start_time
    
    result = {
        'pattern': pattern_name,
        'layer_types': layer_types,
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
    print(f"  Val Acc: {val_metrics['accuracy']:.4f}")
    print(f"  Val PPL: {val_metrics['perplexity']:.2f}")
    print(f"  Time: {training_time:.1f}s")
    
    # Clear GPU
    del model, optimizer
    torch.cuda.empty_cache()
    
    return result


def main():
    print("Testing Different Attention Patterns")
    print(f"Configuration: 4 experts, top-2 routing, 300 training steps\n")
    
    results = {}
    for pattern_name, layer_types in PATTERNS.items():
        try:
            result = test_pattern(pattern_name, layer_types)
            results[pattern_name] = result
        except Exception as e:
            print(f"ERROR testing {pattern_name}: {e}")
            import traceback
            traceback.print_exc()
            torch.cuda.empty_cache()
            continue
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY - All Patterns")
    print(f"{'='*60}\n")
    
    print(f"{'Pattern':<15} {'Val Loss':<10} {'Val Acc':<10} {'Val PPL':<12} {'Time (s)':<10}")
    print("-" * 60)
    for pattern_name, result in results.items():
        print(f"{pattern_name:<15} {result['val_loss']:<10.4f} {result['val_accuracy']:<10.4f} "
              f"{result['val_perplexity']:<12.2f} {result['training_time']:<10.1f}")
    
    # Save results
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True, parents=True)
    
    with open(results_dir / 'attention_patterns_comparison.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {results_dir / 'attention_patterns_comparison.json'}")


if __name__ == "__main__":
    main()

