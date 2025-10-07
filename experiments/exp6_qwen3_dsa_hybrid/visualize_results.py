"""
Visualize experiment results comparing the three model variants
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def load_results():
    """Load results from JSON files"""
    results_dir = Path(__file__).parent / "results"
    
    results = {}
    for variant in ['baseline', 'dsa', 'hybrid']:
        result_file = results_dir / f"{variant}_results.json"
        if result_file.exists():
            with open(result_file, 'r') as f:
                results[variant] = json.load(f)
        else:
            print(f"Warning: {result_file} not found")
    
    return results


def plot_training_curves(results):
    """Plot training curves for all variants"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    metrics = ['train_loss', 'val_loss', 'val_perplexity']
    titles = ['Training Loss', 'Validation Loss', 'Validation Perplexity']
    
    for idx, (metric, title) in enumerate(zip(metrics, titles)):
        ax = axes[idx]
        
        for variant, data in results.items():
            if 'training_history' in data:
                epochs = [h['epoch'] for h in data['training_history']]
                values = [h[metric] for h in data['training_history']]
                ax.plot(epochs, values, label=variant.upper(), marker='o', linewidth=2)
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_final_comparison(results):
    """Create bar charts comparing final metrics"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    variants = list(results.keys())
    metrics = [
        ('final_loss', 'Final Loss (lower is better)', axes[0, 0]),
        ('final_accuracy', 'Final Accuracy (higher is better)', axes[0, 1]),
        ('final_perplexity', 'Final Perplexity (lower is better)', axes[1, 0]),
        ('training_time_minutes', 'Training Time (minutes)', axes[1, 1]),
    ]
    
    for metric_key, title, ax in metrics:
        values = []
        colors = []
        
        for variant in variants:
            if 'final_metrics' in results[variant]:
                if metric_key == 'training_time_minutes':
                    values.append(results[variant][metric_key])
                else:
                    values.append(results[variant]['final_metrics'][metric_key.replace('final_', '')])
                
                # Color coding
                if variant == 'baseline':
                    colors.append('#3498db')  # Blue
                elif variant == 'dsa':
                    colors.append('#e74c3c')  # Red
                else:
                    colors.append('#2ecc71')  # Green
        
        bars = ax.bar(range(len(variants)), values, color=colors)
        ax.set_xticks(range(len(variants)))
        ax.set_xticklabels([v.upper() for v in variants])
        ax.set_title(title)
        ax.set_ylabel(title.split('(')[0].strip())
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}' if height < 10 else f'{height:.1f}',
                   ha='center', va='bottom')
    
    plt.tight_layout()
    return fig


def plot_parameter_comparison(results):
    """Compare number of parameters"""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    variants = list(results.keys())
    params = [results[v]['num_parameters'] / 1e6 for v in variants]  # Convert to millions
    colors = ['#3498db', '#e74c3c', '#2ecc71']
    
    bars = ax.bar(range(len(variants)), params, color=colors)
    ax.set_xticks(range(len(variants)))
    ax.set_xticklabels([v.upper() for v in variants])
    ax.set_ylabel('Parameters (millions)')
    ax.set_title('Model Size Comparison')
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.2f}M',
               ha='center', va='bottom')
    
    plt.tight_layout()
    return fig


def create_summary_table(results):
    """Create a text summary table"""
    print("\n" + "="*80)
    print("EXPERIMENT RESULTS SUMMARY")
    print("="*80)
    
    print(f"\n{'Variant':<15} {'Params':<12} {'Loss':<10} {'Acc':<10} {'PPL':<10} {'Time(min)':<10}")
    print("-"*80)
    
    for variant, data in results.items():
        params = f"{data['num_parameters']/1e6:.2f}M"
        loss = f"{data['final_metrics']['loss']:.4f}"
        acc = f"{data['final_metrics']['accuracy']:.4f}"
        ppl = f"{data['final_metrics']['perplexity']:.2f}"
        time = f"{data['training_time_minutes']:.1f}"
        
        print(f"{variant.upper():<15} {params:<12} {loss:<10} {acc:<10} {ppl:<10} {time:<10}")
    
    print("="*80)
    
    # Find best performer
    best_loss = min(results.items(), key=lambda x: x[1]['final_metrics']['loss'])
    best_acc = max(results.items(), key=lambda x: x[1]['final_metrics']['accuracy'])
    best_ppl = min(results.items(), key=lambda x: x[1]['final_metrics']['perplexity'])
    fastest = min(results.items(), key=lambda x: x[1]['training_time_minutes'])
    
    print("\n🏆 Best Performers:")
    print(f"  Lowest Loss: {best_loss[0].upper()} ({best_loss[1]['final_metrics']['loss']:.4f})")
    print(f"  Highest Accuracy: {best_acc[0].upper()} ({best_acc[1]['final_metrics']['accuracy']:.4f})")
    print(f"  Lowest Perplexity: {best_ppl[0].upper()} ({best_ppl[1]['final_metrics']['perplexity']:.2f})")
    print(f"  Fastest Training: {fastest[0].upper()} ({fastest[1]['training_time_minutes']:.1f} min)")
    print("="*80 + "\n")


def main():
    """Main visualization function"""
    results = load_results()
    
    if not results:
        print("No results found. Run the experiment first with: python run_experiment.py")
        return
    
    print(f"Loaded results for: {', '.join(results.keys())}")
    
    # Create text summary
    create_summary_table(results)
    
    # Create visualizations
    print("Creating visualizations...")
    
    # Training curves
    fig1 = plot_training_curves(results)
    fig1.savefig(Path(__file__).parent / "results" / "training_curves.png", dpi=150)
    print("  ✓ Saved training_curves.png")
    
    # Final comparison
    fig2 = plot_final_comparison(results)
    fig2.savefig(Path(__file__).parent / "results" / "final_comparison.png", dpi=150)
    print("  ✓ Saved final_comparison.png")
    
    # Parameter comparison
    fig3 = plot_parameter_comparison(results)
    fig3.savefig(Path(__file__).parent / "results" / "parameter_comparison.png", dpi=150)
    print("  ✓ Saved parameter_comparison.png")
    
    print("\nDone! Check the results/ directory for visualizations.")
    
    # Optionally show plots
    try:
        plt.show()
    except:
        pass


if __name__ == "__main__":
    main()

