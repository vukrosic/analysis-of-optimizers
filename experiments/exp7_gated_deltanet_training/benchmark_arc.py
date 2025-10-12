"""
Benchmark trained Gated DeltaNet model on ARC-Challenge dataset
ARC tests grade-school level science question answering
"""

import torch
import sys
import os
import json
from pathlib import Path
from tqdm import tqdm
import numpy as np

# Fix tokenizer warning
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, root_dir)

from experiments.exp7_gated_deltanet_training.models import GatedDeltaNetWrapper
from experiments.exp7_gated_deltanet_training.config import ExperimentConfig
from transformers import AutoTokenizer
from datasets import load_dataset


def load_model(checkpoint_path, device='cuda', dtype=torch.bfloat16):
    """Load trained model from checkpoint"""
    print(f"Loading checkpoint from: {checkpoint_path}")
    
    # Load checkpoint (allowlist ExperimentConfig for PyTorch 2.6+ safety)
    torch.serialization.add_safe_globals([ExperimentConfig])
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    # Get config
    config = checkpoint['config']
    
    # Print hybrid config info
    if hasattr(config, 'attn_config') and config.attn_config is not None:
        print(f"✓ Hybrid model detected")
        print(f"  Attention layers: {config.attn_config.get('layers', [])}")
    else:
        print(f"✓ Pure DeltaNet model")
    
    # Create model
    model = GatedDeltaNetWrapper(config)
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Move to device and convert to dtype (Flash Attention requires fp16/bf16)
    model = model.to(device=device, dtype=dtype)
    model.eval()
    
    print(f"✅ Model loaded successfully!")
    print(f"   Steps trained: {checkpoint['global_step']}")
    print(f"   Best val loss: {checkpoint.get('best_val_loss', 'N/A')}")
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"   Precision: {dtype}")
    
    return model, config


def compute_choice_loglikelihood(model, tokenizer, question, choice, device='cuda'):
    """
    Compute log-likelihood of a choice given the question
    Higher log-likelihood = more likely answer
    """
    # Format as: "Question: ... Answer: ..."
    full_text = f"Question: {question}\nAnswer: {choice}"
    
    # Tokenize
    inputs = tokenizer(full_text, return_tensors='pt', truncation=True, max_length=512)
    input_ids = inputs['input_ids'].to(device)
    
    # Skip if too short
    if input_ids.shape[1] < 2:
        return float('-inf')
    
    with torch.no_grad():
        with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
            # Forward pass
            outputs = model(input_ids, labels=input_ids)
            loss = outputs.loss
            
            # Log-likelihood = -loss
            log_likelihood = -loss.item()
    
    return log_likelihood


def evaluate_arc_sample(model, tokenizer, sample, device='cuda'):
    """
    Evaluate a single ARC sample
    Returns: predicted_label, correct_label, is_correct, log_likelihoods
    """
    question = sample['question']
    choices_text = sample['choices']['text']
    choices_label = sample['choices']['label']
    correct_label = sample['answerKey']
    
    # Compute log-likelihood for each choice
    log_likelihoods = []
    for choice_text in choices_text:
        log_lik = compute_choice_loglikelihood(model, tokenizer, question, choice_text, device)
        log_likelihoods.append(log_lik)
    
    # Choose answer with highest log-likelihood
    predicted_idx = np.argmax(log_likelihoods)
    predicted_label = choices_label[predicted_idx]
    
    is_correct = (predicted_label == correct_label)
    
    return predicted_label, correct_label, is_correct, log_likelihoods


def evaluate_arc(model, tokenizer, split='validation', max_samples=None, device='cuda'):
    """
    Evaluate model on ARC-Challenge dataset
    
    Args:
        model: Trained model
        tokenizer: Tokenizer
        split: Dataset split ('train', 'validation', 'test')
        max_samples: Maximum number of samples to evaluate (None = all)
        device: Device to run on
    
    Returns:
        results: Dict with accuracy and per-sample results
    """
    print(f"\nLoading ARC-Challenge dataset ({split} split)...")
    dataset = load_dataset("allenai/ai2_arc", "ARC-Challenge", split=split)
    
    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    
    print(f"Evaluating on {len(dataset)} samples...")
    
    correct = 0
    total = 0
    results_list = []
    
    for sample in tqdm(dataset, desc="Evaluating ARC-Challenge"):
        predicted_label, correct_label, is_correct, log_likelihoods = evaluate_arc_sample(
            model, tokenizer, sample, device
        )
        
        if is_correct:
            correct += 1
        total += 1
        
        # Store result
        results_list.append({
            'id': sample['id'],
            'question': sample['question'],
            'predicted': predicted_label,
            'correct': correct_label,
            'is_correct': is_correct,
            'log_likelihoods': [float(ll) for ll in log_likelihoods],
            'choices': sample['choices']['text'],
        })
    
    accuracy = correct / total if total > 0 else 0.0
    
    results = {
        'dataset': 'ARC-Challenge',
        'split': split,
        'total_samples': total,
        'correct': correct,
        'accuracy': accuracy,
        'accuracy_percent': accuracy * 100,
        'samples': results_list,
    }
    
    return results


def main():
    """Main benchmark function"""
    print("="*70)
    print("ARC-Challenge Benchmark for Exp7 Models")
    print("="*70)
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Choose dtype based on GPU
    if torch.cuda.is_available():
        if torch.cuda.is_bf16_supported():
            dtype = torch.bfloat16
            print(f"Using bfloat16")
        else:
            dtype = torch.float16
            print(f"Using float16")
    else:
        dtype = torch.float32
        print(f"Using float32")
    
    print(f"Device: {device}")
    
    # Load model
    checkpoint_path = Path(__file__).parent / "checkpoints" / "best_model.pt"
    
    if not checkpoint_path.exists():
        print(f"\n❌ Checkpoint not found: {checkpoint_path}")
        print("Please train the model first by running: python run_experiment.py")
        return
    
    model, config = load_model(checkpoint_path, device, dtype=dtype)
    
    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-135M")
    print(f"✅ Tokenizer loaded (vocab size: {len(tokenizer)})")
    
    # Evaluate on validation split
    print("\n" + "="*70)
    print("Running ARC-Challenge Evaluation")
    print("="*70)
    
    results = evaluate_arc(
        model,
        tokenizer,
        split='validation',  # Use validation split
        max_samples=None,  # Evaluate on all samples
        device=device
    )
    
    # Print results
    print("\n" + "="*70)
    print("ARC-Challenge Results")
    print("="*70)
    print(f"Total samples: {results['total_samples']}")
    print(f"Correct: {results['correct']}")
    print(f"Accuracy: {results['accuracy_percent']:.2f}%")
    print("="*70)
    
    # Save results
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)
    
    results_path = results_dir / "arc_challenge_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Results saved to: {results_path}")
    
    # Print some example predictions
    print("\n" + "="*70)
    print("Example Predictions (first 3)")
    print("="*70)
    
    for i, sample in enumerate(results['samples'][:3]):
        print(f"\n[{i+1}] Question: {sample['question']}")
        print(f"    Choices: {sample['choices']}")
        print(f"    Predicted: {sample['predicted']}")
        print(f"    Correct: {sample['correct']}")
        print(f"    ✓ Correct!" if sample['is_correct'] else "    ✗ Incorrect")
    
    print("\n" + "="*70)
    print("✅ Benchmark completed!")
    print("="*70)


if __name__ == "__main__":
    main()

