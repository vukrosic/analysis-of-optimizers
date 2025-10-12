"""
Full Architecture Comparison - Comprehensive Mixture Ablation
Tests 13 architectures covering the full spectrum from 0% to 100% attention

Architectures tested (DeltaNet/Attention mix):
1.  Full DeltaNet (0% attention) - LR 0.001 (ablation proven)
2.  Hybrid 8% (1/12 layers) - LR 0.002 (last layer only)
3.  Hybrid Sparse 17% (2/12 layers) - LR 0.002 (ablation proven - WINNER)
4.  Hybrid 25% (3/12 layers) - LR 0.002 (evenly spread)
5.  Hybrid Late 33% (4/12 layers) - LR 0.002 (final 1/3)
6.  Hybrid 42% (5/12 layers) - LR 0.002 
7.  Hybrid Alternating 50% (6/12 layers) - LR 0.002 (every other)
8.  Hybrid 58% (7/12 layers) - LR 0.002
9.  Hybrid 67% (8/12 layers) - LR 0.002
10. Hybrid 75% (9/12 layers) - LR 0.002
11. Hybrid 83% (10/12 layers) - LR 0.002
12. Hybrid 92% (11/12 layers) - LR 0.002 (all but first)
13. Full Transformer (100% attention) - LR 0.002 (ablation proven)

Training Configuration:
- Steps: Configurable via TRAINING_STEPS variable (default: 10)
- Warmup: 10% of total steps
- Eval interval: Dynamic (adjusted for short runs)
- Model: 768 hidden, 12 layers, ~188M-302M params (varies by architecture)
- Batch: 48, Seq len: 1024

Goal: Find the optimal DeltaNet/Attention mixture ratio for language modeling

Usage:
    python run_full_architecture_comparison.py
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
import numpy as np

# Fix tokenizer parallelism warning
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, root_dir)

from experiments.exp7_hybrid_deltanet_ablation.config import (
    get_h100_deltanet_only,
    get_h100_transformer_only,
    get_h100_hybrid_sparse,
    get_h100_hybrid_alternating,
    get_h100_hybrid_late,
    get_h100_hybrid_8,
    get_h100_hybrid_25,
    get_h100_hybrid_42,
    get_h100_hybrid_58,
    get_h100_hybrid_67,
    get_h100_hybrid_75,
    get_h100_hybrid_83,
    get_h100_hybrid_92,
)
from experiments.exp7_hybrid_deltanet_ablation.models import GatedDeltaNetWrapper
from experiments.exp7_hybrid_deltanet_ablation.run_experiment import Trainer
from data.loader import load_and_cache_data
from utils.helpers import set_seed


def run_architecture_experiment(arch_name, arch_id, get_config_fn, lr, 
                                train_loader, val_loader, device, save_dir,
                                max_steps, warmup_steps, eval_interval, 
                                log_interval, save_interval, vocab_size):
    """Run a single architecture experiment"""
    print(f"\n{'='*80}")
    print(f"Architecture: {arch_name}")
    print(f"Learning Rate: {lr:.2e} (from ablation study)")
    print(f"{'='*80}")
    
    # Get config and set all training parameters
    config = get_config_fn()
    config.learning_rate = lr
    config.vocab_size = vocab_size
    config.max_steps = max_steps
    config.warmup_steps = warmup_steps
    config.eval_interval = eval_interval
    config.log_interval = log_interval
    config.save_interval = save_interval
    
    # Create model
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32
    model = GatedDeltaNetWrapper(config)
    model.print_info()
    model = model.to(device=device, dtype=dtype)
    
    # Create trainer
    experiment_dir = save_dir / arch_id
    trainer = Trainer(model, config, train_loader, val_loader, device, save_dir=experiment_dir)
    
    # Train
    start_time = time.time()
    results = trainer.train()
    elapsed = time.time() - start_time
    
    # Calculate tokens processed
    tokens_per_step = config.batch_size * config.max_seq_len
    total_tokens = tokens_per_step * config.max_steps
    tokens_per_second = total_tokens / elapsed
    
    print(f"\n✓ {arch_name} completed!")
    print(f"  Training Time: {elapsed:.1f}s ({elapsed/60:.2f} min)")
    print(f"  Tokens Processed: {total_tokens:,}")
    print(f"  Throughput: {tokens_per_second:,.0f} tokens/sec")
    print(f"  Best Val Loss: {results['best_val_loss']:.4f}")
    
    if results['val_history']:
        final_val = results['val_history'][-1]
        print(f"  Final Val Loss: {final_val['loss']:.4f}")
        print(f"  Final Val Accuracy: {final_val['accuracy']*100:.2f}%")
        print(f"  Final Val Perplexity: {final_val['perplexity']:.2f}")
    
    return {
        'architecture': arch_name,
        'arch_id': arch_id,
        'lr': lr,
        'best_val_loss': results['best_val_loss'],
        'train_history': results['train_history'],
        'val_history': results['val_history'],
        'elapsed_time': elapsed,
        'total_tokens': total_tokens,
        'tokens_per_second': tokens_per_second,
        'final_train_loss': results['train_history'][-1]['loss'] if results['train_history'] else None,
        'final_val_loss': final_val['loss'] if results['val_history'] else None,
        'final_val_accuracy': final_val['accuracy'] if results['val_history'] else None,
        'final_val_perplexity': final_val['perplexity'] if results['val_history'] else None,
        'attention_percentage': get_attention_percentage(arch_id),
    }


def get_attention_percentage(arch_id):
    """Calculate percentage of attention layers"""
    if 'deltanet' in arch_id and 'hybrid' not in arch_id:
        return 0.0
    elif 'transformer' in arch_id:
        return 100.0
    elif 'hybrid_8' in arch_id:
        return 8.3  # 1/12 layers
    elif 'sparse' in arch_id:
        return 16.7  # 2/12 layers
    elif 'hybrid_25' in arch_id:
        return 25.0  # 3/12 layers
    elif 'late' in arch_id:
        return 33.3  # 4/12 layers
    elif 'hybrid_42' in arch_id:
        return 41.7  # 5/12 layers
    elif 'alternating' in arch_id:
        return 50.0  # 6/12 layers
    elif 'hybrid_58' in arch_id:
        return 58.3  # 7/12 layers
    elif 'hybrid_67' in arch_id:
        return 66.7  # 8/12 layers
    elif 'hybrid_75' in arch_id:
        return 75.0  # 9/12 layers
    elif 'hybrid_83' in arch_id:
        return 83.3  # 10/12 layers
    elif 'hybrid_92' in arch_id:
        return 91.7  # 11/12 layers
    return 0.0


def plot_comprehensive_comparison(all_results, save_dir):
    """Create comprehensive comparison plots"""
    # Larger figure for more architectures
    fig = plt.figure(figsize=(24, 18))
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)
    
    # Sort results by attention percentage for consistent coloring
    results_sorted = sorted(all_results, key=lambda x: x['attention_percentage'])
    
    # Define color gradient from blue (DeltaNet) to red (Transformer)
    # Use a smooth gradient across the attention spectrum
    import matplotlib.cm as cm
    n_archs = len(results_sorted)
    cmap = cm.get_cmap('coolwarm')  # Blue to red gradient
    colors = [cmap(i / (n_archs - 1)) for i in range(n_archs)]
    color_map = {r['arch_id']: colors[i] for i, r in enumerate(results_sorted)}
    
    # 1. Training Loss Curves (top left)
    ax1 = fig.add_subplot(gs[0, 0])
    for result in results_sorted:
        if result['train_history']:
            steps = [h['step'] for h in result['train_history']]
            losses = [h['loss'] for h in result['train_history']]
            ax1.plot(steps, losses, label=result['architecture'], 
                    color=color_map[result['arch_id']], linewidth=2, alpha=0.8)
    ax1.set_xlabel('Step', fontweight='bold')
    ax1.set_ylabel('Train Loss', fontweight='bold')
    ax1.set_title('Training Loss Curves', fontweight='bold', fontsize=14)
    ax1.legend(fontsize=7, ncol=2 if len(results_sorted) > 7 else 1)
    ax1.grid(True, alpha=0.3)
    
    # 2. Validation Loss Curves (top middle)
    ax2 = fig.add_subplot(gs[0, 1])
    for result in results_sorted:
        if result['val_history']:
            steps = [h['step'] for h in result['val_history']]
            losses = [h['loss'] for h in result['val_history']]
            ax2.plot(steps, losses, label=result['architecture'], 
                    color=color_map[result['arch_id']], linewidth=2, marker='o', 
                    markersize=5, alpha=0.8)
    ax2.set_xlabel('Step', fontweight='bold')
    ax2.set_ylabel('Val Loss', fontweight='bold')
    ax2.set_title('Validation Loss Curves', fontweight='bold', fontsize=14)
    ax2.legend(fontsize=7, ncol=2 if len(results_sorted) > 7 else 1)
    ax2.grid(True, alpha=0.3)
    
    # 3. Validation Accuracy Curves (top right)
    ax3 = fig.add_subplot(gs[0, 2])
    for result in results_sorted:
        if result['val_history']:
            steps = [h['step'] for h in result['val_history']]
            accs = [h['accuracy'] * 100 for h in result['val_history']]
            ax3.plot(steps, accs, label=result['architecture'], 
                    color=color_map[result['arch_id']], linewidth=2, marker='s', 
                    markersize=5, alpha=0.8)
    ax3.set_xlabel('Step', fontweight='bold')
    ax3.set_ylabel('Val Accuracy (%)', fontweight='bold')
    ax3.set_title('Validation Accuracy Curves', fontweight='bold', fontsize=14)
    ax3.legend(fontsize=7, ncol=2 if len(results_sorted) > 7 else 1)
    ax3.grid(True, alpha=0.3)
    
    # 4. Best Val Loss Comparison (middle left)
    ax4 = fig.add_subplot(gs[1, 0])
    arch_names = [r['architecture'] for r in results_sorted]
    best_losses = [r['best_val_loss'] for r in results_sorted]
    bars = ax4.barh(arch_names, best_losses, color=[color_map[r['arch_id']] for r in results_sorted])
    ax4.set_xlabel('Best Val Loss', fontweight='bold')
    ax4.set_title('Best Validation Loss (Lower is Better)', fontweight='bold', fontsize=14)
    ax4.grid(True, alpha=0.3, axis='x')
    ax4.tick_params(axis='y', labelsize=8)  # Smaller y-axis labels for more architectures
    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, best_losses)):
        ax4.text(val, i, f' {val:.4f}', va='center', fontweight='bold', fontsize=7)
    
    # 5. Training Time Comparison (middle middle)
    ax5 = fig.add_subplot(gs[1, 1])
    times = [r['elapsed_time'] / 60 for r in results_sorted]  # Convert to minutes
    bars = ax5.barh(arch_names, times, color=[color_map[r['arch_id']] for r in results_sorted])
    ax5.set_xlabel('Training Time (minutes)', fontweight='bold')
    ax5.set_title('Training Time (Lower is Better)', fontweight='bold', fontsize=14)
    ax5.grid(True, alpha=0.3, axis='x')
    ax5.tick_params(axis='y', labelsize=8)
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, times)):
        ax5.text(val, i, f' {val:.1f}m', va='center', fontweight='bold', fontsize=7)
    
    # 6. Throughput Comparison (middle right)
    ax6 = fig.add_subplot(gs[1, 2])
    throughputs = [r['tokens_per_second'] / 1000 for r in results_sorted]  # K tokens/sec
    bars = ax6.barh(arch_names, throughputs, color=[color_map[r['arch_id']] for r in results_sorted])
    ax6.set_xlabel('Throughput (K tokens/sec)', fontweight='bold')
    ax6.set_title('Training Throughput (Higher is Better)', fontweight='bold', fontsize=14)
    ax6.grid(True, alpha=0.3, axis='x')
    ax6.tick_params(axis='y', labelsize=8)
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, throughputs)):
        ax6.text(val, i, f' {val:.1f}K', va='center', fontweight='bold', fontsize=7)
    
    # 7. Efficiency Scatter: Time vs Val Loss (bottom left)
    ax7 = fig.add_subplot(gs[2, 0])
    for result in results_sorted:
        ax7.scatter(result['elapsed_time'] / 60, result['best_val_loss'], 
                   s=200, color=color_map[result['arch_id']], 
                   label=result['architecture'], alpha=0.7, edgecolors='black', linewidth=2)
    ax7.set_xlabel('Training Time (minutes)', fontweight='bold')
    ax7.set_ylabel('Best Val Loss', fontweight='bold')
    ax7.set_title('Efficiency Frontier: Time vs Quality', fontweight='bold', fontsize=14)
    ax7.legend(fontsize=6, ncol=2 if len(results_sorted) > 7 else 1, loc='best')
    ax7.grid(True, alpha=0.3)
    # Annotate points
    for result in results_sorted:
        ax7.annotate(f"{result['attention_percentage']:.0f}% attn", 
                    (result['elapsed_time'] / 60, result['best_val_loss']),
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    # 8. Attention Ratio vs Performance (bottom middle)
    ax8 = fig.add_subplot(gs[2, 1])
    attn_pcts = [r['attention_percentage'] for r in results_sorted]
    ax8.plot(attn_pcts, best_losses, marker='o', markersize=10, linewidth=2, 
            color='#2ca02c', markerfacecolor='white', markeredgewidth=2)
    for result in results_sorted:
        ax8.scatter(result['attention_percentage'], result['best_val_loss'],
                   s=200, color=color_map[result['arch_id']], alpha=0.7, 
                   edgecolors='black', linewidth=2)
    ax8.set_xlabel('Attention Percentage (%)', fontweight='bold')
    ax8.set_ylabel('Best Val Loss', fontweight='bold')
    ax8.set_title('Attention Ratio vs Quality', fontweight='bold', fontsize=14)
    ax8.grid(True, alpha=0.3)
    # Dynamic x-ticks based on actual attention percentages tested
    attn_ticks = sorted(set([0, 25, 50, 75, 100] + [int(r['attention_percentage']) for r in results_sorted]))
    ax8.set_xticks(attn_ticks)
    
    # 9. Final Accuracy Comparison (bottom right)
    ax9 = fig.add_subplot(gs[2, 2])
    final_accs = [r['final_val_accuracy'] * 100 if r['final_val_accuracy'] else 0 for r in results_sorted]
    bars = ax9.barh(arch_names, final_accs, color=[color_map[r['arch_id']] for r in results_sorted])
    ax9.set_xlabel('Final Val Accuracy (%)', fontweight='bold')
    ax9.set_title('Final Validation Accuracy (Higher is Better)', fontweight='bold', fontsize=14)
    ax9.grid(True, alpha=0.3, axis='x')
    ax9.tick_params(axis='y', labelsize=8)
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, final_accs)):
        ax9.text(val, i, f' {val:.2f}%', va='center', fontweight='bold', fontsize=7)
    
    # Get TRAINING_STEPS from the first result's config (or pass as parameter)
    steps = all_results[0]['train_history'][-1]['step'] if all_results[0]['train_history'] else 'N'
    
    plt.suptitle(f'{steps}-Step Architecture Comparison - All Metrics', 
                fontsize=16, fontweight='bold', y=0.995)
    
    plt.savefig(save_dir / f'architecture_comparison_{steps}steps.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n📊 Comprehensive comparison plot saved!")


def main():
    # ============================================================================
    # SINGLE SOURCE OF TRUTH - TRAINING CONFIGURATION
    # ============================================================================
    TRAINING_STEPS = 10  # <-- Change this value to adjust training length
    
    # Calculate derived values
    warmup_steps = int(TRAINING_STEPS * 0.1)  # 10% warmup
    eval_interval = min(50, max(TRAINING_STEPS // 2, 1))  # Adjust for short runs
    log_interval = max(TRAINING_STEPS // 10, 1)
    save_interval = 999999  # Disable intermediate checkpoints (only save best model)
    
    print("="*80)
    print(f"FULL ARCHITECTURE COMPARISON - {TRAINING_STEPS} STEP TRAINING")
    print("="*80)
    
    # Define all architectures with their optimal learning rates
    # Full spectrum from 0% to 100% attention
    architectures = [
        # Pure architectures (from ablation study - proven LRs)
        ('Full DeltaNet (0%)', f'h100_deltanet_{TRAINING_STEPS}', get_h100_deltanet_only, 0.001),
        ('Full Transformer (100%)', f'h100_transformer_{TRAINING_STEPS}', get_h100_transformer_only, 0.002),
        
        # Hybrids (LR 0.002 based on ablation pattern - any attention prefers 0.002)
        ('Hybrid 8%', f'h100_hybrid_8_{TRAINING_STEPS}', get_h100_hybrid_8, 0.002),
        ('Hybrid Sparse 17%', f'h100_hybrid_sparse_{TRAINING_STEPS}', get_h100_hybrid_sparse, 0.002),
        ('Hybrid 25%', f'h100_hybrid_25_{TRAINING_STEPS}', get_h100_hybrid_25, 0.002),
        ('Hybrid Late 33%', f'h100_hybrid_late_{TRAINING_STEPS}', get_h100_hybrid_late, 0.002),
        ('Hybrid 42%', f'h100_hybrid_42_{TRAINING_STEPS}', get_h100_hybrid_42, 0.002),
        ('Hybrid Alternating 50%', f'h100_hybrid_alternating_{TRAINING_STEPS}', get_h100_hybrid_alternating, 0.002),
        ('Hybrid 58%', f'h100_hybrid_58_{TRAINING_STEPS}', get_h100_hybrid_58, 0.002),
        ('Hybrid 67%', f'h100_hybrid_67_{TRAINING_STEPS}', get_h100_hybrid_67, 0.002),
        ('Hybrid 75%', f'h100_hybrid_75_{TRAINING_STEPS}', get_h100_hybrid_75, 0.002),
        ('Hybrid 83%', f'h100_hybrid_83_{TRAINING_STEPS}', get_h100_hybrid_83, 0.002),
        ('Hybrid 92%', f'h100_hybrid_92_{TRAINING_STEPS}', get_h100_hybrid_92, 0.002),
    ]
    
    print(f"\nTesting {len(architectures)} architectures:")
    for name, _, _, lr in architectures:
        print(f"  • {name:<25} @ LR {lr:.2e}")
    
    print(f"\nTraining Configuration:")
    print(f"  Steps: {TRAINING_STEPS}")
    print(f"  Warmup: {warmup_steps} ({warmup_steps/TRAINING_STEPS*100:.0f}%)")
    print(f"  Eval Interval: {eval_interval} (will eval at steps: {', '.join(str(i) for i in range(eval_interval, TRAINING_STEPS+1, eval_interval))})")
    print(f"  Model: 768 hidden, 12 layers (~302M params)")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Device: {device}")
    
    # Load data once
    print("\n" + "="*80)
    print("Loading Data")
    print("="*80)
    
    base_config = get_h100_deltanet_only()
    base_config.max_steps = TRAINING_STEPS
    base_config.warmup_steps = warmup_steps
    base_config.eval_interval = eval_interval
    base_config.log_interval = log_interval
    base_config.save_interval = save_interval
    
    # Calculate data needs dynamically based on TRAINING_STEPS
    # batch_size=48, seq_len=1024
    # Tokens per step = 48 × 1024 = 49,152
    tokens_needed = base_config.batch_size * base_config.max_seq_len * TRAINING_STEPS
    base_config.num_documents = 80_000
    base_config.max_tokens = max(80_000_000, int(tokens_needed * 2.3))  # 2.3x safety margin
    
    print(f"Batch size: {base_config.batch_size}")
    print(f"Seq length: {base_config.max_seq_len}")
    print(f"Steps: {TRAINING_STEPS}")
    print(f"Tokens per step: {base_config.batch_size * base_config.max_seq_len:,}")
    print(f"Total tokens needed: {tokens_needed:,}")
    print(f"Data budget: {base_config.max_tokens:,} tokens ({base_config.max_tokens/tokens_needed:.1f}x margin)")
    
    from dataclasses import dataclass
    @dataclass
    class DataConfig:
        num_documents: int = base_config.num_documents
        max_tokens: int = base_config.max_tokens
        vocab_size: int = base_config.vocab_size
    
    data_config = DataConfig()
    texts, tokenizer, tokens = load_and_cache_data(data_config)
    vocab_size = len(tokenizer)
    
    print(f"\nVocabulary size: {vocab_size}")
    print(f"Total tokens loaded: {len(tokens):,}")
    
    # Split data
    val_split_ratio = 0.1
    val_token_start = int(len(tokens) * (1 - val_split_ratio))
    train_tokens = tokens[:val_token_start]
    val_tokens = tokens[val_token_start:]
    
    print(f"Train tokens: {len(train_tokens):,}")
    print(f"Val tokens: {len(val_tokens):,}")
    
    # Results directory - dynamically named based on TRAINING_STEPS
    results_dir = Path(__file__).parent / f"architecture_comparison_{TRAINING_STEPS}steps"
    results_dir.mkdir(exist_ok=True, parents=True)
    
    # Run all experiments
    all_results = []
    total_start_time = time.time()
    
    for idx, (arch_name, arch_id, get_config_fn, lr) in enumerate(architectures, 1):
        print(f"\n\n{'#'*80}")
        print(f"EXPERIMENT {idx}/{len(architectures)}: {arch_name}")
        print(f"{'#'*80}")
        
        # Get base config for data loader creation
        config = get_config_fn()
        
        # Set seed for reproducibility
        set_seed(config.seed + idx)
        
        # Create fresh data loaders
        from data.streaming_dataset import create_progressive_loaders
        train_loader, val_loader = create_progressive_loaders(
            train_tokens, val_tokens,
            config.max_seq_len, config.batch_size,
            train_state=None, val_state=None
        )
        
        # Run experiment
        result = run_architecture_experiment(
            arch_name, arch_id, get_config_fn, lr,
            train_loader, val_loader, device, results_dir,
            TRAINING_STEPS, warmup_steps, eval_interval,
            log_interval, save_interval, vocab_size
        )
        all_results.append(result)
        
        print(f"\n✅ Completed {idx}/{len(architectures)} experiments")
        print(f"   Time elapsed: {(time.time() - total_start_time) / 60:.1f} minutes")
    
    total_time = time.time() - total_start_time
    
    # Analyze and print results
    print(f"\n\n{'='*80}")
    print("FINAL RESULTS SUMMARY")
    print(f"{'='*80}\n")
    
    # Sort by best val loss
    ranked_results = sorted(all_results, key=lambda x: x['best_val_loss'])
    
    print(f"{'Rank':<6} {'Architecture':<25} {'Attn%':<8} {'Best Val':<12} {'Accuracy':<10} {'Time (m)':<10} {'Throughput':<15}")
    print("-" * 100)
    
    for rank, result in enumerate(ranked_results, 1):
        medal = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else "  "
        acc = result['final_val_accuracy'] * 100 if result['final_val_accuracy'] else 0
        throughput = result['tokens_per_second'] / 1000  # K tokens/sec
        print(f"{medal} {rank:<4} {result['architecture']:<25} {result['attention_percentage']:>5.1f}%  "
              f"{result['best_val_loss']:<12.4f} {acc:<9.2f}% {result['elapsed_time']/60:<9.1f}  "
              f"{throughput:>6.1f}K tok/s")
    
    print(f"\n{'='*80}")
    print(f"🏆 WINNER: {ranked_results[0]['architecture']}")
    print(f"   Best Val Loss: {ranked_results[0]['best_val_loss']:.4f}")
    winner_acc = ranked_results[0]['final_val_accuracy']
    if winner_acc is not None:
        print(f"   Final Accuracy: {winner_acc*100:.2f}%")
    else:
        print(f"   Final Accuracy: N/A (no validation run)")
    print(f"   Training Time: {ranked_results[0]['elapsed_time']/60:.1f} minutes")
    print(f"   Attention Ratio: {ranked_results[0]['attention_percentage']:.0f}%")
    print(f"{'='*80}")
    
    # Calculate efficiency metrics
    print(f"\n{'='*80}")
    print("EFFICIENCY ANALYSIS")
    print(f"{'='*80}\n")
    
    fastest = min(all_results, key=lambda x: x['elapsed_time'])
    slowest = max(all_results, key=lambda x: x['elapsed_time'])
    
    print(f"Fastest: {fastest['architecture']} ({fastest['elapsed_time']/60:.1f} min)")
    print(f"Slowest: {slowest['architecture']} ({slowest['elapsed_time']/60:.1f} min)")
    print(f"Speed range: {slowest['elapsed_time']/fastest['elapsed_time']:.2f}x difference")
    print(f"\nTotal experiment time: {total_time/60:.1f} minutes ({total_time/3600:.2f} hours)")
    
    # Save comprehensive summary
    summary = {
        'config': {
            'training_steps': TRAINING_STEPS,
            'hidden_size': base_config.hidden_size,
            'num_layers': base_config.num_hidden_layers,
            'batch_size': base_config.batch_size,
            'max_seq_len': base_config.max_seq_len,
            'total_tokens_processed': base_config.batch_size * base_config.max_seq_len * TRAINING_STEPS,
        },
        'architectures_tested': len(architectures),
        'total_training_time_minutes': total_time / 60,
        'results': [
            {
                'rank': rank,
                'architecture': r['architecture'],
                'attention_percentage': r['attention_percentage'],
                'learning_rate': r['lr'],
                'best_val_loss': r['best_val_loss'],
                'final_val_loss': r['final_val_loss'],
                'final_val_accuracy': r['final_val_accuracy'],
                'final_val_perplexity': r['final_val_perplexity'],
                'training_time_minutes': r['elapsed_time'] / 60,
                'tokens_processed': r['total_tokens'],
                'throughput_tokens_per_sec': r['tokens_per_second'],
            }
            for rank, r in enumerate(ranked_results, 1)
        ],
        'winner': {
            'architecture': ranked_results[0]['architecture'],
            'attention_percentage': ranked_results[0]['attention_percentage'],
            'best_val_loss': ranked_results[0]['best_val_loss'],
            'improvement_over_second': (
                (ranked_results[1]['best_val_loss'] - ranked_results[0]['best_val_loss']) / ranked_results[1]['best_val_loss'] * 100
                if ranked_results[0]['best_val_loss'] != float('inf') and ranked_results[1]['best_val_loss'] != float('inf')
                else None
            ),
        },
        'insights': {
            'fastest_architecture': fastest['architecture'],
            'slowest_architecture': slowest['architecture'],
            'speed_range_multiplier': slowest['elapsed_time'] / fastest['elapsed_time'],
            'best_quality_architecture': ranked_results[0]['architecture'],
            'best_efficiency': min(all_results, key=lambda x: x['best_val_loss'] * x['elapsed_time'])['architecture'],
        }
    }
    
    with open(results_dir / 'architecture_comparison_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n💾 Summary saved to: {results_dir / 'architecture_comparison_summary.json'}")
    
    # Generate comprehensive plots
    plot_comprehensive_comparison(all_results, results_dir)
    
    print(f"\n{'='*80}")
    print(f"{TRAINING_STEPS}-STEP ARCHITECTURE COMPARISON COMPLETE! ✨")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()

