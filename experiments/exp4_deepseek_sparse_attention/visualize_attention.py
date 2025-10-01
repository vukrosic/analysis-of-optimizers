"""
Attention Pattern Visualization for DeepSeek Sparse Attention

This script visualizes and analyzes the attention patterns of sparse attention models.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
import os

# Add parent directories to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from models import create_sparse_model
from sparse_attention import TopKTokenSelector
from data.dataset import WikiTextDataset
from data.loader import get_vocab_size
from torch.utils.data import DataLoader


def visualize_sparse_attention_pattern(model, input_ids, layer_idx=0, save_path=None):
    """
    Visualize the sparse attention pattern for a given input
    
    Args:
        model: Sparse attention model
        input_ids: Input token IDs [1, seq_len]
        layer_idx: Which layer to visualize
        save_path: Where to save the visualization
    """
    model.eval()
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)
    
    with torch.no_grad():
        # Get index scores
        logits, _, index_scores_list = model(input_ids, return_index_scores=True)
        
        if not index_scores_list or layer_idx >= len(index_scores_list):
            print(f"Warning: Layer {layer_idx} not available")
            return
        
        index_scores = index_scores_list[layer_idx].cpu().numpy()[0]  # [seq_len, seq_len]
        
        # Get top-k selection
        selector = TopKTokenSelector(top_k=model.blocks[0].attention.sparse_top_k)
        top_k_mask, _ = selector(
            torch.from_numpy(index_scores).unsqueeze(0),
            apply_causal_mask=True
        )
        top_k_mask = top_k_mask.cpu().numpy()[0]
    
    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    # 1. Index Scores Heatmap
    sns.heatmap(index_scores, cmap='viridis', ax=axes[0], cbar=True)
    axes[0].set_title(f'Lightning Indexer Scores (Layer {layer_idx})', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Key Position')
    axes[0].set_ylabel('Query Position')
    
    # 2. Top-k Selection Mask
    sns.heatmap(top_k_mask.astype(float), cmap='RdYlGn', ax=axes[1], cbar=True, 
                vmin=0, vmax=1, cbar_kws={'label': 'Selected'})
    axes[1].set_title(f'Top-k Selection Pattern (Layer {layer_idx})', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Key Position')
    axes[1].set_ylabel('Query Position')
    
    # 3. Sparsity visualization (selected vs non-selected)
    sparsity = 1.0 - top_k_mask.mean()
    sparse_pattern = np.where(top_k_mask, index_scores, np.nan)
    sns.heatmap(sparse_pattern, cmap='plasma', ax=axes[2], cbar=True)
    axes[2].set_title(f'Sparse Attention Weights (Sparsity: {sparsity:.2%})', 
                     fontsize=14, fontweight='bold')
    axes[2].set_xlabel('Key Position')
    axes[2].set_ylabel('Query Position')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ Saved attention pattern to {save_path}")
    else:
        plt.show()
    
    plt.close()
    
    return index_scores, top_k_mask


def analyze_attention_statistics(model, dataloader, num_samples=10):
    """
    Analyze attention pattern statistics across multiple samples
    
    Args:
        model: Sparse attention model
        dataloader: Data loader
        num_samples: Number of samples to analyze
    """
    model.eval()
    device = next(model.parameters()).device
    
    all_sparsities = []
    all_distances = []
    all_coverage = []
    
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= num_samples:
                break
            
            input_ids = batch['input_ids'][:1].to(device)  # Take first sample
            
            # Get index scores
            _, _, index_scores_list = model(input_ids, return_index_scores=True)
            
            for layer_idx, index_scores in enumerate(index_scores_list):
                # Get top-k selection
                selector = TopKTokenSelector(top_k=model.blocks[0].attention.sparse_top_k)
                top_k_mask, top_k_indices = selector(index_scores, apply_causal_mask=True)
                
                # Compute sparsity
                sparsity = 1.0 - top_k_mask.float().mean().item()
                all_sparsities.append(sparsity)
                
                # Compute average distance to selected tokens
                seq_len = index_scores.shape[1]
                positions = torch.arange(seq_len, device=device).unsqueeze(0).unsqueeze(2)
                distances = (positions - top_k_indices).abs().float().mean().item()
                all_distances.append(distances)
                
                # Compute coverage (how many unique tokens are selected)
                coverage = top_k_indices.unique().numel() / seq_len
                all_coverage.append(coverage)
    
    # Print statistics
    print("\n" + "="*60)
    print("ATTENTION PATTERN STATISTICS")
    print("="*60)
    print(f"\nSparsity:")
    print(f"  Mean: {np.mean(all_sparsities):.3f} ± {np.std(all_sparsities):.3f}")
    print(f"  Range: [{np.min(all_sparsities):.3f}, {np.max(all_sparsities):.3f}]")
    
    print(f"\nAverage Distance to Selected Tokens:")
    print(f"  Mean: {np.mean(all_distances):.2f} ± {np.std(all_distances):.2f}")
    print(f"  Range: [{np.min(all_distances):.2f}, {np.max(all_distances):.2f}]")
    
    print(f"\nToken Coverage:")
    print(f"  Mean: {np.mean(all_coverage):.3f} ± {np.std(all_coverage):.3f}")
    print(f"  Range: [{np.min(all_coverage):.3f}, {np.max(all_coverage):.3f}]")
    
    # Create distribution plots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    axes[0].hist(all_sparsities, bins=30, alpha=0.7, edgecolor='black')
    axes[0].set_xlabel('Sparsity')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Sparsity Distribution')
    axes[0].axvline(np.mean(all_sparsities), color='red', linestyle='--', 
                    label=f'Mean: {np.mean(all_sparsities):.3f}')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].hist(all_distances, bins=30, alpha=0.7, edgecolor='black')
    axes[1].set_xlabel('Average Distance')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Distance to Selected Tokens Distribution')
    axes[1].axvline(np.mean(all_distances), color='red', linestyle='--',
                    label=f'Mean: {np.mean(all_distances):.2f}')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    axes[2].hist(all_coverage, bins=30, alpha=0.7, edgecolor='black')
    axes[2].set_xlabel('Coverage')
    axes[2].set_ylabel('Frequency')
    axes[2].set_title('Token Coverage Distribution')
    axes[2].axvline(np.mean(all_coverage), color='red', linestyle='--',
                    label=f'Mean: {np.mean(all_coverage):.3f}')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = Path('results/comparison/attention_statistics.png')
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ Saved statistics plot to {save_path}")
    plt.close()
    
    return {
        'sparsity': {'mean': np.mean(all_sparsities), 'std': np.std(all_sparsities)},
        'distance': {'mean': np.mean(all_distances), 'std': np.std(all_distances)},
        'coverage': {'mean': np.mean(all_coverage), 'std': np.std(all_coverage)}
    }


def visualize_indexer_head_contributions(model, input_ids, layer_idx=0, save_path=None):
    """
    Visualize the contribution of each indexer head
    
    Args:
        model: Sparse attention model
        input_ids: Input token IDs
        layer_idx: Which layer to visualize
        save_path: Where to save the visualization
    """
    model.eval()
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)
    
    with torch.no_grad():
        # Get embeddings
        x = model.embed(input_ids)
        
        # Pass through blocks up to target layer
        for i, block in enumerate(model.blocks):
            if i < layer_idx:
                x, _, _ = block(x, return_index_scores=False)
            elif i == layer_idx:
                # Get indexer components
                indexer = block.attention.indexer
                
                # Compute queries, keys, and weights
                batch_size, seq_len, _ = x.shape
                queries = indexer.q_proj(x).reshape(
                    batch_size, seq_len, indexer.indexer_heads, indexer.indexer_dim
                )
                keys = indexer.k_proj(x)
                weights = indexer.w_proj(x)
                
                # Compute per-head contributions
                dots = torch.einsum('bthd,bsd->bths', queries, keys)
                activated = torch.nn.functional.relu(dots)
                
                # Weight each head
                weighted = activated * weights.unsqueeze(2)
                
                # Get contribution of each head
                head_contributions = weighted[0].cpu().numpy()  # [seq_len, seq_len, heads]
                
                break
    
    # Visualize each head
    num_heads = head_contributions.shape[-1]
    fig, axes = plt.subplots(2, (num_heads + 1) // 2, figsize=(20, 10))
    axes = axes.flatten()
    
    for head_idx in range(num_heads):
        sns.heatmap(head_contributions[:, :, head_idx], cmap='viridis', ax=axes[head_idx], cbar=True)
        axes[head_idx].set_title(f'Indexer Head {head_idx}', fontsize=12, fontweight='bold')
        axes[head_idx].set_xlabel('Key Position')
        axes[head_idx].set_ylabel('Query Position')
    
    # Hide unused subplots
    for idx in range(num_heads, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle(f'Indexer Head Contributions (Layer {layer_idx})', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ Saved indexer head visualization to {save_path}")
    else:
        plt.show()
    
    plt.close()


def main():
    """Main visualization script"""
    print("\n" + "="*80)
    print("ATTENTION PATTERN VISUALIZATION")
    print("="*80 + "\n")
    
    # Load config
    config = {
        'vocab_size': get_vocab_size(),
        'd_model': 256,
        'n_heads': 8,
        'n_layers': 6,
        'd_ff': 512,
        'max_seq_len': 128,
        'num_experts': 4,
        'expert_top_k': 2,
        'indexer_heads': 4,
        'indexer_dim': 64,
        'sparse_top_k': 64,
        'dropout': 0.1,
    }
    
    # Create model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"📊 Creating model on {device}...")
    model = create_sparse_model(config).to(device)
    
    # Load checkpoint if available
    checkpoint_path = Path('results/sparse/final_model.pt')
    if checkpoint_path.exists():
        print(f"📂 Loading checkpoint from {checkpoint_path}...")
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    else:
        print("⚠️  No checkpoint found, using randomly initialized model")
    
    # Load dataset
    print("📚 Loading dataset...")
    dataset = WikiTextDataset(
        split='validation',
        max_seq_len=config['max_seq_len'],
        max_tokens=5000
    )
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    # Get a sample
    batch = next(iter(dataloader))
    input_ids = batch['input_ids'][:1]
    
    # Create visualization directory
    vis_dir = Path('results/comparison')
    vis_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Visualize attention pattern
    print("\n1️⃣  Visualizing attention pattern...")
    visualize_sparse_attention_pattern(
        model, input_ids, layer_idx=0,
        save_path=vis_dir / 'attention_pattern_layer0.png'
    )
    
    # 2. Visualize indexer head contributions
    print("\n2️⃣  Visualizing indexer head contributions...")
    visualize_indexer_head_contributions(
        model, input_ids, layer_idx=0,
        save_path=vis_dir / 'indexer_heads_layer0.png'
    )
    
    # 3. Analyze attention statistics
    print("\n3️⃣  Analyzing attention statistics...")
    stats = analyze_attention_statistics(model, dataloader, num_samples=10)
    
    print("\n" + "="*80)
    print("✅ VISUALIZATION COMPLETED")
    print("="*80)
    print(f"\nVisualizations saved to: {vis_dir}/")
    print(f"  - attention_pattern_layer0.png")
    print(f"  - indexer_heads_layer0.png")
    print(f"  - attention_statistics.png")


if __name__ == '__main__':
    main()
