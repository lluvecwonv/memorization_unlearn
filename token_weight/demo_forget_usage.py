#!/usr/bin/env python3
"""
Demo script showing how to use the forget token mask in training.
This demonstrates how to apply penalties to selected forget tokens.
"""

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

def compute_forget_loss(
    logits: torch.Tensor,        # Model logits (batch_size, seq_len, vocab_size)
    labels: torch.Tensor,        # Target labels (batch_size, seq_len)
    forget_mask: torch.Tensor,   # Forget mask (batch_size, seq_len)
    forget_penalty: float = 10.0, # Penalty weight for forget tokens
    base_loss_weight: float = 1.0 # Weight for regular loss
) -> torch.Tensor:
    """
    Compute loss with penalties for forget tokens.
    
    Args:
        logits: Model output logits
        labels: Target token IDs  
        forget_mask: Boolean mask (True = forget token, False = keep)
        forget_penalty: Penalty multiplier for forget tokens
        base_loss_weight: Weight for base language modeling loss
    
    Returns:
        Combined loss with forget penalties
    """
    # Standard cross-entropy loss
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    shift_forget_mask = forget_mask[..., 1:].contiguous()
    
    # Compute per-token losses
    per_token_loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        reduction='none'
    )
    per_token_loss = per_token_loss.view(shift_labels.shape)
    
    # Separate forget and keep tokens
    forget_tokens = shift_forget_mask.bool()
    keep_tokens = ~forget_tokens & (shift_labels != -100)  # Exclude padding
    
    # Compute weighted losses
    base_loss = per_token_loss[keep_tokens].mean() if keep_tokens.any() else torch.tensor(0.0, device=logits.device)
    forget_loss = per_token_loss[forget_tokens].mean() if forget_tokens.any() else torch.tensor(0.0, device=logits.device)
    
    # Combined loss
    total_loss = base_loss_weight * base_loss + forget_penalty * forget_loss
    
    return total_loss, base_loss, forget_loss


def demo_training_step():
    """Demo of how to use forget mask in training"""
    print("🚀 Demo: Using forget token mask in training")
    
    # Load mask (replace with your actual path)
    mask_path = "./forget_masks/forget_token_mask_example.pt"
    try:
        forget_mask = torch.load(mask_path, map_location='cpu')
        print(f"✅ Loaded forget mask: shape={forget_mask.shape}")
    except FileNotFoundError:
        print(f"⚠️  Demo mask not found at {mask_path}, creating dummy mask")
        forget_mask = torch.randint(0, 2, (4, 128), dtype=torch.bool)  # Dummy mask
    
    # Dummy model inputs (replace with your actual data)
    batch_size, seq_len = forget_mask.shape
    vocab_size = 32000
    
    # Simulate model outputs
    logits = torch.randn(batch_size, seq_len, vocab_size)  # Model predictions
    labels = torch.randint(0, vocab_size, (batch_size, seq_len))  # Target tokens
    
    # Apply mask padding (-100 for ignored tokens)
    labels = labels.clone()
    labels[torch.randint(0, 2, labels.shape, dtype=torch.bool)] = -100  # Some padding
    
    print(f"📊 Batch shape: {batch_size} samples x {seq_len} tokens")
    print(f"🎯 Forget tokens: {forget_mask.sum().item()} / {forget_mask.numel()} ({forget_mask.float().mean():.1%})")
    
    # Compute loss with forget penalties
    total_loss, base_loss, forget_loss = compute_forget_loss(
        logits=logits,
        labels=labels,
        forget_mask=forget_mask,
        forget_penalty=5.0,  # 5x penalty for forget tokens
        base_loss_weight=1.0
    )
    
    print(f"💫 Loss breakdown:")
    print(f"  - Base loss: {base_loss:.4f}")
    print(f"  - Forget loss: {forget_loss:.4f}")
    print(f"  - Total loss: {total_loss:.4f}")
    
    # Backward pass (in real training)
    total_loss.backward()
    print("✅ Backward pass completed")
    
    return total_loss


def analyze_forget_mask(mask_path: str):
    """Analyze properties of a forget mask"""
    print(f"🔍 Analyzing forget mask: {mask_path}")
    
    try:
        mask = torch.load(mask_path, map_location='cpu')
    except FileNotFoundError:
        print(f"❌ Mask file not found: {mask_path}")
        return
    
    print(f"📊 Mask statistics:")
    print(f"  - Shape: {mask.shape}")
    print(f"  - Total tokens: {mask.numel():,}")
    print(f"  - Forget tokens: {mask.sum().item():,}")
    print(f"  - Forget ratio: {mask.float().mean():.1%}")
    print(f"  - Affected samples: {(mask.sum(dim=1) > 0).sum().item():,} / {mask.shape[0]:,}")
    
    # Per-sample statistics
    tokens_per_sample = mask.sum(dim=1)
    print(f"  - Forget tokens per sample: min={tokens_per_sample.min().item()}, "
          f"max={tokens_per_sample.max().item()}, "
          f"mean={tokens_per_sample.float().mean():.1f}")


if __name__ == "__main__":
    print("🎭 Forget Token Usage Demo\n")
    
    # Demo 1: Training with forget mask
    demo_training_step()
    print()
    
    # Demo 2: Analyze mask (replace with actual path)
    example_mask_path = "./forget_masks/forget_token_mask_example.pt"
    analyze_forget_mask(example_mask_path)
    print()
    
    print("✅ Demo completed! Integrate similar logic into your training loop.")