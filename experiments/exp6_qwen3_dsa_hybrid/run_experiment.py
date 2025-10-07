"""
Run Experiment 6: Qwen3-Next with DeepSeek Sparse Attention

Trains and evaluates three model variants:
1. Baseline: Standard Qwen3-Next
2. DSA-Only: All attention replaced with DeepSeek Sparse Attention
3. Hybrid: DSA for full_attention, Gated DeltaNet for linear_attention
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import time
import json
import os
from pathlib import Path
import sys

# Add paths - must be done before any local imports
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, root_dir)

# Now we can import from the experiment directory
from experiments.exp6_qwen3_dsa_hybrid.config import MEDIUM_CONFIG
from experiments.exp6_qwen3_dsa_hybrid.models import create_model
from data.loader import load_and_cache_data
from data.dataset import TextTokenDataset
from utils.helpers import set_seed


def count_parameters(model):
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_epoch(model, dataloader, optimizer, device, config, epoch, total_steps):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    total_tokens = 0
    steps_this_epoch = 0
    
    for i, batch in enumerate(dataloader):
        # Stop if we've reached max_steps across all epochs
        if total_steps >= config.max_steps:
            break
        
        # Handle tuple output from dataset (x, y)
        if isinstance(batch, (list, tuple)):
            input_ids = batch[0].to(device)
        else:
            input_ids = batch.to(device)
        labels = input_ids.clone()
        
        outputs = model(input_ids=input_ids, labels=labels)
        loss = outputs.loss
        
        loss.backward()
        
        if (i + 1) % config.gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()
        
        total_loss += loss.item() * input_ids.numel()
        total_tokens += input_ids.numel()
        steps_this_epoch += 1
        total_steps += 1
        
        if total_steps % config.log_interval == 0:
            avg_loss = total_loss / total_tokens
            print(f"  Step {total_steps}/{config.max_steps}, Loss: {avg_loss:.4f}")
    
    avg_loss = total_loss / total_tokens if total_tokens > 0 else 0
    return avg_loss, total_steps


@torch.no_grad()
def evaluate(model, dataloader, device):
    """Evaluate model"""
    model.eval()
    total_loss = 0
    total_tokens = 0
    correct = 0
    
    for batch in dataloader:
        # Handle tuple output from dataset (x, y)
        if isinstance(batch, (list, tuple)):
            input_ids = batch[0].to(device)
        else:
            input_ids = batch.to(device)
        labels = input_ids.clone()
        
        outputs = model(input_ids=input_ids, labels=labels)
        loss = outputs.loss
        logits = outputs.logits
        
        total_loss += loss.item() * input_ids.numel()
        total_tokens += input_ids.numel()
        
        # Compute accuracy
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        predictions = shift_logits.argmax(dim=-1)
        correct += (predictions == shift_labels).sum().item()
    
    avg_loss = total_loss / total_tokens
    accuracy = correct / total_tokens
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    
    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'perplexity': perplexity,
    }


def train_variant(variant_name, config, train_loader, val_loader, device):
    """Train a single model variant"""
    print(f"\n{'='*60}")
    print(f"Training Variant: {variant_name.upper()}")
    print(f"{'='*60}")
    
    # Create model
    model = create_model(variant_name, config).to(device)
    num_params = count_parameters(model)
    print(f"Parameters: {num_params:,}")
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=0.1,
    )
    
    # Training
    start_time = time.time()
    best_val_loss = float('inf')
    results = {
        'variant': variant_name,
        'num_parameters': num_params,
        'config': {
            'hidden_size': config.hidden_size,
            'num_layers': config.num_hidden_layers,
            'num_heads': config.num_attention_heads,
            'max_seq_len': config.max_seq_len,
            'sparse_top_k': config.sparse_top_k if variant_name != 'baseline' else None,
        },
        'training_history': [],
    }
    
    # Calculate maximum epochs needed (in case dataset is smaller than expected)
    num_epochs = (config.max_steps + len(train_loader) - 1) // len(train_loader)
    print(f"Dataset: {len(train_loader)} batches per epoch")
    print(f"Will train for up to {num_epochs} epoch(s) to reach {config.max_steps} steps")
    
    total_steps = 0
    for epoch in range(num_epochs):
        if total_steps >= config.max_steps:
            break
            
        print(f"\nEpoch {epoch + 1}/{num_epochs} (Steps: {total_steps}/{config.max_steps})")
        
        train_loss, total_steps = train_epoch(model, train_loader, optimizer, device, config, epoch, total_steps)
        val_metrics = evaluate(model, val_loader, device)
        
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_metrics['loss']:.4f}, "
              f"Val Acc: {val_metrics['accuracy']:.4f}, "
              f"Val PPL: {val_metrics['perplexity']:.2f}")
        
        results['training_history'].append({
            'epoch': epoch + 1,
            'total_steps': total_steps,
            'train_loss': train_loss,
            'val_loss': val_metrics['loss'],
            'val_accuracy': val_metrics['accuracy'],
            'val_perplexity': val_metrics['perplexity'],
        })
        
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
    
    results['total_steps_trained'] = total_steps
    
    training_time = time.time() - start_time
    
    # Final evaluation
    final_metrics = evaluate(model, val_loader, device)
    
    results['final_metrics'] = final_metrics
    results['training_time_seconds'] = training_time
    results['training_time_minutes'] = training_time / 60
    
    print(f"\n✓ Completed {variant_name}")
    print(f"Training time: {training_time/60:.1f} minutes")
    print(f"Final metrics: Loss={final_metrics['loss']:.4f}, "
          f"Acc={final_metrics['accuracy']:.4f}, "
          f"PPL={final_metrics['perplexity']:.2f}")
    
    return results


def main():
    """Run all three experiments"""
    # Setup
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Load data
    config = MEDIUM_CONFIG
    print("\nLoading data...")
    texts, tokenizer, tokens = load_and_cache_data(config)
    config.vocab_size = len(tokenizer)
    
    dataset = TextTokenDataset(tokens, config.max_seq_len)
    
    # Train/val split
    val_size = len(dataset) // 10
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.batch_size, 
        shuffle=True, 
        num_workers=2
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config.batch_size, 
        shuffle=False, 
        num_workers=2
    )
    
    print(f"Dataset: {len(train_dataset)} train, {len(val_dataset)} val samples")
    
    # Create results directory
    results_dir = Path(__file__).parent / config.save_dir
    results_dir.mkdir(exist_ok=True, parents=True)
    
    # Run experiments for all three variants
    variants = ['baseline', 'dsa', 'hybrid']
    all_results = {}
    
    for variant in variants:
        try:
            results = train_variant(variant, config, train_loader, val_loader, device)
            all_results[variant] = results
            
            # Save individual results
            variant_file = results_dir / f"{variant}_results.json"
            with open(variant_file, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"Saved results to {variant_file}")
            
            # Clear GPU memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        except Exception as e:
            print(f"\n❌ Error training {variant}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Create summary
    summary = {
        'experiment': 'Qwen3-Next DSA Hybrid Comparison',
        'variants': list(all_results.keys()),
        'config': {
            'hidden_size': config.hidden_size,
            'num_layers': config.num_hidden_layers,
            'num_heads': config.num_attention_heads,
            'max_seq_len': config.max_seq_len,
            'max_steps': config.max_steps,
            'batch_size': config.batch_size,
        },
        'comparison': {},
    }
    
    for variant, results in all_results.items():
        summary['comparison'][variant] = {
            'num_parameters': results['num_parameters'],
            'final_loss': results['final_metrics']['loss'],
            'final_accuracy': results['final_metrics']['accuracy'],
            'final_perplexity': results['final_metrics']['perplexity'],
            'training_time_minutes': results['training_time_minutes'],
        }
    
    # Save summary
    summary_file = results_dir / 'summary.json'
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n{'='*60}")
    print("EXPERIMENT COMPLETE")
    print(f"{'='*60}")
    print(f"\nResults saved to: {results_dir}")
    print("\nSummary:")
    for variant, metrics in summary['comparison'].items():
        print(f"\n{variant.upper()}:")
        print(f"  Parameters: {metrics['num_parameters']:,}")
        print(f"  Final Loss: {metrics['final_loss']:.4f}")
        print(f"  Final Accuracy: {metrics['final_accuracy']:.4f}")
        print(f"  Final Perplexity: {metrics['final_perplexity']:.2f}")
        print(f"  Training Time: {metrics['training_time_minutes']:.1f} min")


if __name__ == "__main__":
    main()

