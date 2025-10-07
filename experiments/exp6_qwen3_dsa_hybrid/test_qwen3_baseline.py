"""
Test script: Train just the baseline Qwen3-Next model
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import time
import json
import os
from pathlib import Path
import sys

# Add paths
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, root_dir)

from models.qwen3_next.configuration_qwen3_next import Qwen3NextConfig
from models.qwen3_next.modular_qwen3_next import Qwen3NextForCausalLM
from data.loader import load_and_cache_data
from data.dataset import TextTokenDataset
from utils.helpers import set_seed


def count_parameters(model):
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_epoch(model, dataloader, optimizer, device, max_steps, total_steps):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    total_tokens = 0
    
    for i, batch in enumerate(dataloader):
        # Stop if we've reached max_steps across all epochs
        if total_steps >= max_steps:
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
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()
        
        total_loss += loss.item() * input_ids.numel()
        total_tokens += input_ids.numel()
        total_steps += 1
        
        if total_steps % 50 == 0:  # Log every 50 steps for longer training
            avg_loss = total_loss / total_tokens
            print(f"  Step {total_steps}/{max_steps}, Loss: {avg_loss:.4f}")
    
    avg_loss = total_loss / total_tokens if total_tokens > 0 else 0
    return avg_loss, total_steps


@torch.no_grad()
def evaluate(model, dataloader, device, max_batches=100):
    """Evaluate model"""
    model.eval()
    total_loss = 0
    total_tokens = 0
    correct = 0
    
    for i, batch in enumerate(dataloader):
        if i >= max_batches:
            break
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


def main():
    """Run baseline Qwen3-Next training"""
    # Setup
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Simple configuration
    print("\nCreating Qwen3-Next config...")
    config = Qwen3NextConfig(
        vocab_size=50257,  # Will be updated from tokenizer
        hidden_size=128,  # Reduced from 256
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=512,  # Reduced from 1024
        max_position_embeddings=512,
        rope_theta=10000.0,
        attention_dropout=0.1,
        hidden_dropout_prob=0.1,
        rms_norm_eps=1e-6,
        head_dim=32,  # hidden_size // num_attention_heads = 128 // 4 = 32
        partial_rotary_factor=1.0,  # Use full rotary embeddings (not partial)
        # Layer types: Try different patterns
        # Pattern: linear, full, full, linear (sandwich pattern)
        layer_types=["linear_attention", "full_attention", "full_attention", "linear_attention"],
        # Linear attention config
        linear_num_value_heads=2,
        linear_num_key_heads=2,
        linear_key_head_dim=64,
        linear_value_head_dim=64,
        linear_conv_kernel_dim=4,
        # MoE configuration
        num_experts=4,  # 4 experts
        num_local_experts=4,  # Same as num_experts
        num_experts_per_tok=2,  # Route to top 2 experts
        router_jitter_noise=0.0,
        decoder_sparse_step=2,  # Use MoE every 2 layers
        moe_intermediate_size=256,  # Reduced from 512
        shared_expert_intermediate_size=0,  # No shared experts for simplicity
        mlp_only_layers=[],
    )
    
    # Load data
    print("\nLoading data...")
    from dataclasses import dataclass
    
    @dataclass
    class SimpleConfig:
        num_documents: int = 1000
        max_tokens: int = 2_000_000
        vocab_size: int = 50257
    
    data_config = SimpleConfig()
    texts, tokenizer, tokens = load_and_cache_data(data_config)
    config.vocab_size = len(tokenizer)
    
    max_seq_len = 128  # Reduced from 256
    dataset = TextTokenDataset(tokens, max_seq_len)
    
    # Train/val split
    val_size = len(dataset) // 10
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=2,  # Reduced from 4
        shuffle=True, 
        num_workers=0  # Avoid fork issues
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=2,  # Reduced from 4
        shuffle=False, 
        num_workers=0
    )
    
    print(f"Dataset: {len(train_dataset)} train, {len(val_dataset)} val samples")
    
    # Create model
    print("\nCreating Qwen3-Next model...")
    model = Qwen3NextForCausalLM(config).to(device)
    num_params = count_parameters(model)
    print(f"Parameters: {num_params:,}")
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=3e-4,
        betas=(0.9, 0.95),
        weight_decay=0.1,
    )
    
    # Training
    max_steps = 30
    print(f"\nTraining for {max_steps} steps...")
    start_time = time.time()
    
    total_steps = 0
    num_epochs = (max_steps + len(train_loader) - 1) // len(train_loader)
    
    for epoch in range(num_epochs):
        if total_steps >= max_steps:
            break
            
        print(f"\nEpoch {epoch + 1}/{num_epochs} (Steps: {total_steps}/{max_steps})")
        
        train_loss, total_steps = train_epoch(model, train_loader, optimizer, device, max_steps, total_steps)
        print("Evaluating...")
        val_metrics = evaluate(model, val_loader, device)
        
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_metrics['loss']:.4f}, "
              f"Val Acc: {val_metrics['accuracy']:.4f}, "
              f"Val PPL: {val_metrics['perplexity']:.2f}")
    
    training_time = time.time() - start_time
    
    # Final evaluation (quick eval with subset)
    print("\nFinal evaluation...")
    final_metrics = evaluate(model, val_loader, device, max_batches=50)
    
    print(f"\n{'='*60}")
    print("TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"Training time: {training_time:.1f} seconds")
    print(f"Final metrics: Loss={final_metrics['loss']:.4f}, "
          f"Acc={final_metrics['accuracy']:.4f}, "
          f"PPL={final_metrics['perplexity']:.2f}")
    
    # Save results
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True, parents=True)
    
    results = {
        'model': 'Qwen3-Next Baseline',
        'num_parameters': num_params,
        'total_steps': total_steps,
        'training_time_seconds': training_time,
        'final_metrics': final_metrics,
    }
    
    results_file = results_dir / 'baseline_test_results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_file}")


if __name__ == "__main__":
    main()

