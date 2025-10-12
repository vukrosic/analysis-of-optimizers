"""
Learning Rate Ablation Study for H100 Architectures
Tests 3 learning rates for each of 3 architectures:
- Full DeltaNet (h100_deltanet)
- Full Transformer (h100_transformer)  
- Hybrid Sparse (h100_hybrid_sparse - 17% attention)

Learning rates tested: 5e-4, 1e-3, 2e-3

Usage:
    python run_lr_ablation_h100.py
"""

import torch
import sys
import os
import json
import time
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# Fix tokenizer parallelism warning
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, root_dir)

from experiments.exp7_gated_deltanet_training.config import (
    get_h100_deltanet_only,
    get_h100_transformer_only,
    get_h100_hybrid_sparse,
)
from experiments.exp7_gated_deltanet_training.models import GatedDeltaNetWrapper
from experiments.exp7_gated_deltanet_training.run_experiment import Trainer
from data.loader import load_and_cache_data
from utils.helpers import set_seed


def run_lr_experiment(lr, config, train_loader, val_loader, device, run_name, arch_name):
    """Run a single LR experiment for a specific architecture"""
    print(f"\n{'='*70}")
    print(f"Architecture: {arch_name}")
    print(f"Testing Learning Rate: {lr:.2e}")
    print(f"{'='*70}")
    
    # Update config with this LR
    config.learning_rate = lr
    
    # Create model
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32
    model = GatedDeltaNetWrapper(config)
    model.print_info()
    model = model.to(device=device, dtype=dtype)
    
    # Create trainer with specific save directory
    save_dir = Path(__file__).parent / "lr_ablation_h100" / run_name
    trainer = Trainer(model, config, train_loader, val_loader, device, save_dir=save_dir)
    
    # Train
    start_time = time.time()
    results = trainer.train()
    elapsed = time.time() - start_time
    
    print(f"\n✓ {arch_name} @ LR {lr:.2e} completed in {elapsed:.1f}s")
    print(f"  Best Val Loss: {results['best_val_loss']:.4f}")
    if results['val_history']:
        final_val = results['val_history'][-1]
        print(f"  Final Val Loss: {final_val['loss']:.4f}")
        print(f"  Final Val Accuracy: {final_val['accuracy']*100:.2f}%")
        print(f"  Final Val Perplexity: {final_val['perplexity']:.2f}")
    
    return {
        'architecture': arch_name,
        'lr': lr,
        'best_val_loss': results['best_val_loss'],
        'train_history': results['train_history'],
        'val_history': results['val_history'],
        'elapsed_time': elapsed,
    }


def plot_lr_comparison(all_results, save_dir, is_partial=False):
    """Plot comparison of all learning rates across architectures"""
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    
    suffix = "_partial" if is_partial else ""
    
    # Group results by architecture
    arch_groups = {}
    for result in all_results:
        arch = result['architecture']
        if arch not in arch_groups:
            arch_groups[arch] = []
        arch_groups[arch].append(result)
    
    # Define colors for each architecture
    arch_colors = {
        'Full DeltaNet': '#1f77b4',
        'Full Transformer': '#ff7f0e',
        'Hybrid Sparse': '#2ca02c',
    }
    
    # Training loss curves
    for arch, results in arch_groups.items():
        color = arch_colors.get(arch, '#333333')
        for result in results:
            if result['train_history']:
                steps = [h['step'] for h in result['train_history']]
                losses = [h['loss'] for h in result['train_history']]
                label = f"{arch} LR={result['lr']:.2e}"
                axes[0, 0].plot(steps, losses, label=label, color=color, 
                              linewidth=2, alpha=0.7)
    
    axes[0, 0].set_xlabel('Step', fontweight='bold')
    axes[0, 0].set_ylabel('Train Loss', fontweight='bold')
    axes[0, 0].set_title('Training Loss by Architecture & Learning Rate', fontweight='bold')
    axes[0, 0].legend(fontsize=8)
    axes[0, 0].grid(True, alpha=0.3)
    
    # Validation loss curves
    for arch, results in arch_groups.items():
        color = arch_colors.get(arch, '#333333')
        for result in results:
            if result['val_history']:
                steps = [h['step'] for h in result['val_history']]
                losses = [h['loss'] for h in result['val_history']]
                label = f"{arch} LR={result['lr']:.2e}"
                axes[0, 1].plot(steps, losses, label=label, color=color,
                              linewidth=2, marker='o', alpha=0.7)
    
    axes[0, 1].set_xlabel('Step', fontweight='bold')
    axes[0, 1].set_ylabel('Val Loss', fontweight='bold')
    axes[0, 1].set_title('Validation Loss by Architecture & Learning Rate', fontweight='bold')
    axes[0, 1].legend(fontsize=8)
    axes[0, 1].grid(True, alpha=0.3)
    
    # Validation accuracy curves
    for arch, results in arch_groups.items():
        color = arch_colors.get(arch, '#333333')
        for result in results:
            if result['val_history']:
                steps = [h['step'] for h in result['val_history']]
                accs = [h['accuracy'] * 100 for h in result['val_history']]
                label = f"{arch} LR={result['lr']:.2e}"
                axes[1, 0].plot(steps, accs, label=label, color=color,
                              linewidth=2, marker='s', alpha=0.7)
    
    axes[1, 0].set_xlabel('Step', fontweight='bold')
    axes[1, 0].set_ylabel('Val Accuracy (%)', fontweight='bold')
    axes[1, 0].set_title('Validation Accuracy by Architecture & Learning Rate', fontweight='bold')
    axes[1, 0].legend(fontsize=8)
    axes[1, 0].grid(True, alpha=0.3)
    
    # Best validation loss grouped bar chart
    architectures = list(arch_groups.keys())
    x = range(len(architectures))
    width = 0.25
    
    # Get learning rates (assuming same for all architectures)
    lrs = sorted(list(set([r['lr'] for r in all_results])))
    
    for i, lr in enumerate(lrs):
        losses = []
        for arch in architectures:
            arch_results = [r for r in arch_groups[arch] if r['lr'] == lr]
            if arch_results:
                losses.append(arch_results[0]['best_val_loss'])
            else:
                losses.append(None)
        
        positions = [xi + (i - 1) * width for xi in x]
        bars = axes[1, 1].bar(positions, losses, width, label=f"LR {lr:.2e}")
    
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(architectures, rotation=15, ha='right')
    axes[1, 1].set_xlabel('Architecture', fontweight='bold')
    axes[1, 1].set_ylabel('Best Val Loss', fontweight='bold')
    axes[1, 1].set_title('Best Validation Loss by Architecture & LR', fontweight='bold')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    filename = f'lr_ablation_h100_comparison{suffix}.png'
    plt.savefig(save_dir / filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    if not is_partial:
        print(f"\n📊 Final comparison plot saved to: {save_dir / filename}")
    else:
        print(f"📊 Progress plot saved to: {save_dir / filename} ({len(all_results)} experiments completed)")


def main():
    print("="*70)
    print("LEARNING RATE ABLATION STUDY - H100 Architectures")
    print("="*70)
    
    # Define architectures and learning rates to test
    architectures = [
        ('Full DeltaNet', 'h100_deltanet', get_h100_deltanet_only),
        ('Full Transformer', 'h100_transformer', get_h100_transformer_only),
        ('Hybrid Sparse', 'h100_hybrid_sparse', get_h100_hybrid_sparse),
    ]
    
    learning_rates = [
        5e-4,   # Lower
        1e-3,   # Middle (current default)
        2e-3,   # Higher
    ]
    
    print(f"\nTesting {len(architectures)} architectures × {len(learning_rates)} learning rates = {len(architectures) * len(learning_rates)} total experiments")
    print(f"\nArchitectures:")
    for name, _, _ in architectures:
        print(f"  • {name}")
    print(f"\nLearning Rates:")
    for lr in learning_rates:
        print(f"  • {lr:.2e}")
    
    # Ablation configuration - short runs to find optimal LR
    # Using 200 steps like the RTX 4090 ablation
    ablation_steps = 200
    warmup_steps = 20  # 10% warmup
    eval_interval = 40
    log_interval = 10
    save_interval = 200
    
    print(f"\nAblation Configuration:")
    print(f"  Steps: {ablation_steps}")
    print(f"  Warmup: {warmup_steps}")
    print(f"  Eval Interval: {eval_interval}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    # Load data once (reuse for all experiments)
    print("\n" + "="*70)
    print("Loading Data")
    print("="*70)
    
    # Get base config to determine data requirements
    base_config = get_h100_deltanet_only()
    
    # Override training steps for ablation
    base_config.max_steps = ablation_steps
    base_config.warmup_steps = warmup_steps
    base_config.eval_interval = eval_interval
    base_config.log_interval = log_interval
    base_config.save_interval = save_interval
    
    # Calculate tokens needed for no repetition:
    # H100: batch_size=48, max_seq_len=2048
    # Tokens per step = 48 × 2048 = 98,304
    # For 200 steps: 98,304 × 200 = 19,660,800 tokens (~19.7M)
    # With 2x safety margin = 39,321,600 (~39.3M)
    base_config.num_documents = 60_000
    base_config.max_tokens = 40_000_000  # 40M tokens for 200 steps (2x margin)
    
    print(f"Batch size: {base_config.batch_size}")
    print(f"Seq length: {base_config.max_seq_len}")
    print(f"Steps: {base_config.max_steps}")
    print(f"Tokens needed: {base_config.batch_size * base_config.max_seq_len * base_config.max_steps:,}")
    print(f"Data budget: {base_config.max_tokens:,} tokens (no repetition)")
    
    from dataclasses import dataclass
    @dataclass
    class DataConfig:
        num_documents: int = base_config.num_documents
        max_tokens: int = base_config.max_tokens
        vocab_size: int = base_config.vocab_size
    
    data_config = DataConfig()
    texts, tokenizer, tokens = load_and_cache_data(data_config)
    vocab_size = len(tokenizer)
    
    print(f"Vocabulary size: {vocab_size}")
    print(f"Total tokens loaded: {len(tokens):,}")
    
    # Split data
    val_split_ratio = 0.1
    val_token_start = int(len(tokens) * (1 - val_split_ratio))
    train_tokens = tokens[:val_token_start]
    val_tokens = tokens[val_token_start:]
    
    print(f"Train tokens: {len(train_tokens):,}")
    print(f"Val tokens: {len(val_tokens):,}")
    
    # Create progressive data loaders (never repeat data)
    from data.streaming_dataset import create_progressive_loaders
    
    # Run experiments
    all_results = []
    results_dir = Path(__file__).parent / "lr_ablation_h100"
    results_dir.mkdir(exist_ok=True, parents=True)
    
    total_experiments = len(architectures) * len(learning_rates)
    experiment_num = 0
    
    for arch_name, arch_id, get_config_fn in architectures:
        print(f"\n\n{'#'*70}")
        print(f"ARCHITECTURE: {arch_name}")
        print(f"{'#'*70}")
        
        for lr in learning_rates:
            experiment_num += 1
            
            print(f"\n{'='*70}")
            print(f"Experiment {experiment_num}/{total_experiments}")
            print(f"Architecture: {arch_name} | Learning Rate: {lr:.2e}")
            print(f"{'='*70}")
            
            # Get fresh config for this experiment
            config = get_config_fn()
            config.vocab_size = vocab_size
            
            # Apply ablation settings
            config.max_steps = ablation_steps
            config.warmup_steps = warmup_steps
            config.eval_interval = eval_interval
            config.log_interval = log_interval
            config.save_interval = save_interval
            
            # Set seed for reproducibility
            set_seed(config.seed + experiment_num)  # Different seed per experiment
            
            # Create fresh data loaders for this experiment
            train_loader, val_loader = create_progressive_loaders(
                train_tokens, val_tokens,
                config.max_seq_len, config.batch_size,
                train_data_state=None, val_data_state=None
            )
            
            # Run experiment
            run_name = f"{arch_id}_lr_{lr:.2e}".replace('.', '_').replace('-', '_')
            result = run_lr_experiment(lr, config, train_loader, val_loader, device, run_name, arch_name)
            all_results.append(result)
            
            # Save intermediate results
            intermediate_summary = {
                'completed_experiments': len(all_results),
                'total_experiments': total_experiments,
                'results_so_far': [
                    {
                        'architecture': r['architecture'],
                        'lr': r['lr'],
                        'best_val_loss': r['best_val_loss'],
                        'final_val_loss': r['val_history'][-1]['loss'] if r['val_history'] else None,
                    }
                    for r in all_results
                ]
            }
            with open(results_dir / 'lr_ablation_h100_progress.json', 'w') as f:
                json.dump(intermediate_summary, f, indent=2)
            
            # Generate progress plot after each experiment (if we have at least 2)
            if len(all_results) > 1:
                plot_lr_comparison(all_results, results_dir, is_partial=True)
            
            print(f"\n✅ Completed {len(all_results)}/{total_experiments} experiments")
    
    # Analyze results
    print(f"\n\n{'='*70}")
    print("ABLATION RESULTS SUMMARY")
    print(f"{'='*70}\n")
    
    # Group by architecture
    arch_groups = {}
    for result in all_results:
        arch = result['architecture']
        if arch not in arch_groups:
            arch_groups[arch] = []
        arch_groups[arch].append(result)
    
    # Print results for each architecture
    for arch_name in ['Full DeltaNet', 'Full Transformer', 'Hybrid Sparse']:
        if arch_name not in arch_groups:
            continue
            
        results = arch_groups[arch_name]
        sorted_results = sorted(results, key=lambda x: x['best_val_loss'])
        
        print(f"\n{arch_name}:")
        print(f"{'Rank':<6} {'LR':<12} {'Best Val Loss':<15} {'Final Val Loss':<15} {'Time (s)':<10}")
        print("-" * 70)
        
        for rank, result in enumerate(sorted_results, 1):
            final_val_loss = result['val_history'][-1]['loss'] if result['val_history'] else float('nan')
            medal = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else "  "
            print(f"{medal} {rank:<4} {result['lr']:.2e}    {result['best_val_loss']:<15.4f} "
                  f"{final_val_loss:<15.4f} {result['elapsed_time']:<10.1f}")
        
        best_lr = sorted_results[0]['lr']
        print(f"  → Best LR for {arch_name}: {best_lr:.2e} (Val Loss: {sorted_results[0]['best_val_loss']:.4f})")
    
    # Overall best
    overall_best = min(all_results, key=lambda x: x['best_val_loss'])
    print(f"\n{'='*70}")
    print(f"🏆 OVERALL BEST:")
    print(f"   Architecture: {overall_best['architecture']}")
    print(f"   Learning Rate: {overall_best['lr']:.2e}")
    print(f"   Best Val Loss: {overall_best['best_val_loss']:.4f}")
    print(f"{'='*70}")
    
    # Save summary
    summary = {
        'config': {
            'ablation_steps': ablation_steps,
            'hidden_size': base_config.hidden_size,
            'num_layers': base_config.num_hidden_layers,
            'batch_size': base_config.batch_size,
            'max_seq_len': base_config.max_seq_len,
        },
        'architectures_tested': [name for name, _, _ in architectures],
        'learning_rates_tested': learning_rates,
        'total_experiments': total_experiments,
        'results_by_architecture': {
            arch_name: [
                {
                    'lr': r['lr'],
                    'best_val_loss': r['best_val_loss'],
                    'final_val_loss': r['val_history'][-1]['loss'] if r['val_history'] else None,
                    'final_val_accuracy': r['val_history'][-1]['accuracy'] if r['val_history'] else None,
                    'elapsed_time': r['elapsed_time'],
                }
                for r in arch_groups[arch_name]
            ]
            for arch_name in arch_groups.keys()
        },
        'best_per_architecture': {
            arch_name: {
                'lr': min(arch_groups[arch_name], key=lambda x: x['best_val_loss'])['lr'],
                'best_val_loss': min(arch_groups[arch_name], key=lambda x: x['best_val_loss'])['best_val_loss'],
            }
            for arch_name in arch_groups.keys()
        },
        'overall_best': {
            'architecture': overall_best['architecture'],
            'lr': overall_best['lr'],
            'best_val_loss': overall_best['best_val_loss'],
        },
        'recommendation': f"200-step ablation complete. Best: {overall_best['architecture']} @ LR {overall_best['lr']:.2e}. Use for full 1000-step training."
    }
    
    with open(results_dir / 'lr_ablation_h100_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n💾 Results saved to: {results_dir / 'lr_ablation_h100_summary.json'}")
    
    # Plot final comparison
    plot_lr_comparison(all_results, results_dir, is_partial=False)
    
    print(f"\n{'='*70}")
    print("RECOMMENDATIONS FOR FULL TRAINING")
    print(f"{'='*70}")
    
    for arch_name in ['Full DeltaNet', 'Full Transformer', 'Hybrid Sparse']:
        if arch_name not in arch_groups:
            continue
        best = min(arch_groups[arch_name], key=lambda x: x['best_val_loss'])
        print(f"\n{arch_name}:")
        print(f"  Best LR: {best['lr']:.2e}")
        print(f"  Best Val Loss: {best['best_val_loss']:.4f}")
    
    print(f"\n{'='*70}")
    print("H100 LR Ablation Study Completed! ✨")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()

