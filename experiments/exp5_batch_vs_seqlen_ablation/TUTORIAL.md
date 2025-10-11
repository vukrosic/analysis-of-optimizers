# Understanding Experiment 5: A Step-by-Step Coding Tutorial

This tutorial walks you through the code for the ablation study in `experiments/exp5_batch_vs_seqlen_ablation/ablation_batch_vs_seqlen.py`. We'll break down the script piece by piece to understand how to compare different training strategies in a Mixture-of-Experts (MoE) LLM.

The core research question of this experiment is: to maximize GPU memory usage, is it better to use a large batch size with short sequences, or a small batch size with long sequences?

---

## 1. Setup and Imports

First, we import the necessary libraries. This includes standard libraries like `os` and `torch`, and also custom components from this project.

```python
import sys
import os
import torch
# ... other imports
```
This sets up our basic tools.

```python
# Add project root and experiment directory to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)
```
This is a crucial step to ensure Python can find our custom modules, like the model and data loader, by adding the project's root directory to the system path.

---

## 2. Configuring the Experiment

To run an ablation study, we need a systematic way to define and manage different configurations.

### The `AblationConfig` Dataclass

```python
from dataclasses import dataclass

@dataclass
class AblationConfig:
    name: str
    batch_size: int
    seq_len: int
    lr: float
    grad_accum: int = 1
```
A `dataclass` is a great way to create a simple class for storing data. Here, `AblationConfig` holds the parameters for a single experiment run, such as `batch_size`, `seq_len` (sequence length), and `lr` (learning rate).

### Generating a Matrix of Experiments

The `create_ablation_configs` function defines the different strategies we want to compare.

```python
def create_ablation_configs():
    configs = []
```
We initialize an empty list to hold all our experiment configurations.

```python
    # STRATEGY A: Large Batch + Short Sequences
    for lr in [0.005, 0.01, 0.02]:
        configs.append(AblationConfig(
            name=f"large_batch_lr{lr}",
            batch_size=64,
            seq_len=256,
            lr=lr,
            grad_accum=2
        ))
```
Here, we define our first strategy. We test three different learning rates for a setup with a large batch size (64) and short sequences (256). `grad_accum` stands for gradient accumulation, a technique to simulate a larger batch size.

We do the same for a "Small Batch + Long Sequences" strategy and a "Balanced" baseline strategy.

---

## 3. Data Loading and Preparation

With our configurations defined, we need to prepare the data.

### Loading the Raw Data

```python
from data.loader import load_and_cache_data

texts, tokenizer, tokens = load_and_cache_data(temp_moe_config, cache_dir="data_cache")
```
This helper function from `data/loader.py` handles the heavy lifting of loading a text dataset, preparing a tokenizer, and converting the text into numerical tokens that the model can understand. It also caches the result to speed up future runs.

### Creating a Custom Dataset

```python
from data.dataset import TextTokenDataset

dataset = TextTokenDataset(tokens, abl_config.seq_len)
```
The `TextTokenDataset` is a custom PyTorch `Dataset` class. It takes the long sequence of all tokens and chops it up into smaller chunks of the desired `seq_len`. This is how we prepare the data for each specific ablation configuration.

### The `DataLoader`

```python
from torch.utils.data import DataLoader

train_loader = DataLoader(
    train_dataset,
    batch_size=abl_config.batch_size,
    shuffle=True
)
```
The `DataLoader` is a standard PyTorch utility that takes a `Dataset` and prepares batches of data for training. It handles shuffling the data, setting the `batch_size`, and can even use multiple workers to load data in parallel.

---

## 4. Model and Optimizer Setup

Now we set up the model and the optimizers that will train it.

### Initializing the Model

```python
from models.moe_llm import MoEMinimalLLM

model = MoEMinimalLLM(moe_config).to(device)
```
We create an instance of our Mixture-of-Experts model, `MoEMinimalLLM`, passing it a configuration object. `.to(device)` moves the model's parameters to the appropriate device (CPU or GPU).

### Setting Up a Hybrid Optimizer Strategy

This experiment uses two different optimizers for different parts of the model.

```python
muon_params = []
adamw_params = []

for name, param in model.named_parameters():
    if param.ndim == 2 and 'token_embedding' not in name and 'norm' not in name:
        muon_params.append(param)
    else:
        adamw_params.append(param)
```
We iterate through all the model's parameters. We separate the 2D weight matrices (which benefit from the custom `Muon` optimizer) from other parameters like biases and normalization layers (which use the standard `AdamW` optimizer).

```python
from optimizers.muon import Muon

optimizers = [
    Muon(muon_params, lr=config.muon_lr),
    torch.optim.AdamW(adamw_params, lr=config.adamw_lr)
]
```
We create a list containing our two configured optimizers. During training, we will loop through this list to update all model parameters.

---

## 5. The Core Training Loop

The `train_single_config` function contains the main logic for training the model.

### Iterating and Fetching Data

```python
train_iter = iter(train_loader)

while step < max_steps:
    try:
        x, y = next(train_iter)
    except StopIteration:
        train_iter = iter(train_loader)
        x, y = next(train_iter)
```
We create an iterator from our `DataLoader`. This allows us to continuously draw batches of data. When the iterator is exhausted (i.e., we've gone through the whole dataset once), we simply create a new one to continue training for the desired number of steps. `x` is the input sequence and `y` is the target sequence (usually `x` shifted by one position).

### Mixed-Precision Forward Pass

```python
from torch.amp import autocast

with autocast('cuda', dtype=torch.float16):
    logits, aux_loss = model(x, return_aux_loss=True)
```
`autocast` enables automatic mixed-precision training. It runs operations in `float16` for performance while maintaining `float32` for numerical stability where needed. This significantly speeds up training on modern GPUs. The model returns `logits` (the raw predictions) and an `aux_loss` (an auxiliary loss specific to MoE models to encourage expert diversity).

### Calculating Loss

```python
import torch.nn.functional as F

ce_loss = F.cross_entropy(logits.view(-1, config.vocab_size), y.view(-1))
total_loss = ce_loss + (aux_loss if aux_loss is not None else 0)
```
We calculate the primary loss using `cross_entropy`, which is standard for language modeling. We then add the auxiliary loss to get the final `total_loss` that we will use for backpropagation.

### Gradient Accumulation and Backward Pass

```python
loss = total_loss / config.gradient_accumulation_steps
loss.backward()
```
Instead of performing an optimizer step after every batch, we first divide the loss by the number of accumulation steps. We then call `loss.backward()` to compute gradients.

```python
if (step + 1) % config.gradient_accumulation_steps == 0:
    # ... optimizer step logic ...
```
The actual update of the model's weights only happens every `gradient_accumulation_steps`. This allows us to simulate a much larger batch size than what can fit in GPU memory at one time. For example, a batch size of 8 with 8 accumulation steps is computationally similar to a batch size of 64.

### Updating the Weights

```python
from torch.amp import GradScaler

scaler = GradScaler()

# Inside the accumulation block
scaler.unscale_(optimizer)
torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
scaler.step(optimizer)
scaler.update()
```
The `GradScaler` is used with mixed-precision training to prevent gradients from becoming zero ("underflowing") when they are small.
1.  `scaler.scale(loss).backward()`: Scales the loss before the backward pass.
2.  `scaler.unscale_`: Unscales the gradients before clipping.
3.  `clip_grad_norm_`: Clips the gradients to prevent them from exploding, which helps stabilize training.
4.  `scaler.step(optimizer)`: Performs the optimizer step, automatically unscaling gradients.
5.  `scaler.update()`: Updates the scale for the next iteration.

For each optimizer, we call `optimizer.zero_grad()` to reset the gradients for the next accumulation cycle.

---

## 6. Running the Main Script

The `main` function ties everything together.

### Command-Line Arguments

```python
import argparse

parser = argparse.ArgumentParser(description='MoE Ablation: Batch vs SeqLen')
parser.add_argument('--steps', type=int, default=20, help='Training steps')
# ... other arguments
args = parser.parse_args()
```
`argparse` allows you to run the script with custom parameters from the command line, for example, to test a single configuration quickly or to change the number of training steps.

### The Main Loop

```python
for abl_config in ablation_configs:
    # ... create dataset and dataloader for this config ...

    result = train_single_config(
        abl_config, base_config, train_loader, val_loader, device, max_steps=args.steps
    )
    all_results.append(result)
```
The `main` function iterates through each `AblationConfig` we created. For each one, it sets up the specific `DataLoader` and then calls `train_single_config` to run the training and validation. The results are collected in a list.

### Saving the Results

```python
import json

with open('results/ablation_batch_seqlen/results.json', 'w') as f:
    json.dump(save_data, f, indent=2)
```
Finally, all the collected results, including training losses, validation metrics, and configuration details, are saved to a JSON file for later analysis and plotting.
