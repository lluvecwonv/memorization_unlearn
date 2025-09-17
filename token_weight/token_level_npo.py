#!/usr/bin/env python3
"""
Token-level Negative Preference Optimization (NPO) for Machine Unlearning

This code implements token-level NPO that selectively applies unlearning penalties
to specific tokens identified by influence analysis, rather than entire sequences.

Usage:
    python token_level_npo.py --config config/npo_config.yaml
"""

import torch
torch.backends.cuda.matmul.allow_tf32 = True
import torch.nn.functional as F
import torch.nn as nn
import transformers
from omegaconf import DictConfig, OmegaConf

import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    StateDictType,
    BackwardPrefetch,
    ShardingStrategy,
    CPUOffload,
)
from torch.distributed.fsdp.api import FullStateDictConfig, FullOptimStateDictConfig
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
import contextlib

import numpy as np
import wandb
import tqdm
import random
import os
from collections import defaultdict
import time
import json
import functools
from typing import Optional, Dict, List, Union, Tuple
import argparse
import logging
from dataclasses import dataclass

# Local imports (you'll need to adapt these to your setup)
from preference_datasets import get_batch_iterator
from utils import (
    slice_and_move_batch_for_device,
    formatted_dict,
    all_gather_if_needed,
    pad_to_length,
    get_block_class_from_model,
    rank0_print,
    get_local_dir,
)


def token_level_npo_loss(
    forget_logits: torch.FloatTensor,
    forget_labels: torch.LongTensor,
    forget_mask: torch.BoolTensor,
    reference_logits: torch.FloatTensor,
    beta: float = 0.1,
    temperature: float = 1.0,
    forget_weight: float = 5.0,
    retain_weight: float = 0.1
) -> Tuple[torch.FloatTensor, torch.FloatTensor, Dict]:
    """
    Compute token-level NPO loss using weighted approach.
    
    Args:
        forget_logits: Current model logits for forget samples (batch_size, seq_len, vocab_size)
        forget_labels: True labels for forget samples (batch_size, seq_len)  
        forget_mask: Boolean mask indicating which tokens to forget (batch_size, seq_len)
        reference_logits: Reference model logits (batch_size, seq_len, vocab_size)
        beta: NPO loss temperature parameter
        temperature: Softmax temperature for logits
        forget_weight: Weight multiplier for forget tokens
        retain_weight: Weight multiplier for retain tokens
        
    Returns:
        loss: NPO loss with token weighting
        forget_loss_component: Forget component of loss
        metrics: Dictionary of metrics for logging
    """
    assert forget_logits.shape[:-1] == forget_labels.shape
    assert forget_mask.shape == forget_labels.shape
    
    batch_size, seq_len = forget_labels.shape
    vocab_size = forget_logits.shape[-1]
    
    # Shift for causal LM (predict next token)
    shift_logits = forget_logits[..., :-1, :].contiguous() / temperature
    shift_labels = forget_labels[..., 1:].contiguous() 
    shift_forget_mask = forget_mask[..., 1:].contiguous()
    shift_ref_logits = reference_logits[..., :-1, :].contiguous() / temperature
    
    # Valid token mask (ignore -100 labels)
    valid_mask = (shift_labels != -100)
    
    # Compute per-token cross-entropy losses for both models
    def compute_token_losses(logits, labels):
        flat_logits = logits.view(-1, vocab_size)
        flat_labels = labels.view(-1)
        token_losses = F.cross_entropy(flat_logits, flat_labels, 
                                     ignore_index=-100, reduction='none')
        return token_losses.view(labels.shape)
    
    # Current model losses per token
    current_losses = compute_token_losses(shift_logits, shift_labels)
    
    # Reference model losses per token
    with torch.no_grad():
        ref_losses = compute_token_losses(shift_ref_logits, shift_labels)
    
    # Compute log ratios (current_loss - ref_loss) per token
    # Higher current loss = model is worse at predicting = good for forgetting
    log_ratios = current_losses - ref_losses
    
    # Apply token-level weights
    forget_tokens_mask = shift_forget_mask & valid_mask
    retain_tokens_mask = (~shift_forget_mask) & valid_mask
    
    # Create weight tensor
    weights = torch.zeros_like(log_ratios)
    weights[forget_tokens_mask] = forget_weight
    weights[retain_tokens_mask] = retain_weight
    
    # NPO loss: -log_sigmoid(beta * log_ratios) 
    # We want higher loss (worse performance) on forget tokens
    npo_per_token = -F.logsigmoid(beta * log_ratios)
    
    # Apply weights and compute weighted average
    weighted_npo_losses = npo_per_token * weights
    valid_weights = weights * valid_mask.float()
    
    total_weighted_loss = weighted_npo_losses.sum()
    total_weight = valid_weights.sum()
    
    if total_weight > 0:
        npo_loss = total_weighted_loss / total_weight * 2.0 / beta  # Scale like original NPO
    else:
        npo_loss = torch.tensor(0.0, device=forget_logits.device)
    
    # Compute forget loss component for monitoring
    if forget_tokens_mask.any():
        forget_component = (npo_per_token * forget_tokens_mask.float()).sum() / (forget_tokens_mask.sum() + 1e-8)
    else:
        forget_component = torch.tensor(0.0, device=forget_logits.device)
    
    # Compute metrics
    metrics = {
        'forget_tokens': forget_tokens_mask.sum().item(),
        'retain_tokens': retain_tokens_mask.sum().item(), 
        'total_tokens': valid_mask.sum().item(),
        'forget_ratio': forget_tokens_mask.sum().item() / max(valid_mask.sum().item(), 1),
        'avg_forget_log_ratio': log_ratios[forget_tokens_mask].mean().item() if forget_tokens_mask.any() else 0.0,
        'avg_retain_log_ratio': log_ratios[retain_tokens_mask].mean().item() if retain_tokens_mask.any() else 0.0,
        'forget_weight': forget_weight,
        'retain_weight': retain_weight,
    }
    
    return npo_loss, forget_component, metrics


def get_batch_loss(logits: torch.FloatTensor, labels: torch.LongTensor) -> torch.FloatTensor:
    """Compute standard cross-entropy loss for a batch"""
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    
    loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100
    )
    return loss


@dataclass 
class NPOConfig:
    """Configuration for Token-level NPO training"""
    # Model settings
    model_name_or_path: str = "EleutherAI/pythia-1.4b-deduped"
    tokenizer_name_or_path: Optional[str] = None
    reference_model_path: Optional[str] = None  # If None, use original model as reference
    
    # Training settings
    learning_rate: float = 1e-6
    beta: float = 0.1  # NPO temperature 
    temperature: float = 1.0  # Logits temperature
    batch_size: int = 4
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    warmup_steps: int = 100
    num_train_steps: int = 1000
    eval_every: int = 100
    save_every: int = 500
    
    # Data settings
    max_length: int = 512
    forget_dataset_path: str = "locuslab/TOFU"
    forget_split: str = "forget10"
    retain_dataset_path: Optional[str] = None  # If None, use retain split from same dataset
    retain_split: str = "retain90"
    
    # Forget token settings
    forget_mask_path: str = "./forget_masks/forget_token_mask.pt"
    apply_token_masking: bool = True
    forget_weight: float = 5.0  # Weight multiplier for forget tokens
    retain_weight: float = 0.1  # Weight multiplier for retain tokens
    
    # Output settings
    output_dir: str = "./token_npo_output"
    run_name: str = "token_npo_experiment"
    log_level: str = "INFO"
    
    # Distributed training
    local_rank: int = 0
    world_size: int = 1
    
    # Wandb logging
    use_wandb: bool = False
    wandb_project: str = "token-npo"


class TokenLevelNPOTrainer:
    """Trainer for token-level NPO unlearning"""
    
    def __init__(self, config: NPOConfig):
        self.config = config
        self.setup_logging()
        
        # Initialize distributed training if needed
        if config.world_size > 1:
            dist.init_process_group(backend='nccl')
            torch.cuda.set_device(config.local_rank)
        
        self.device = torch.device(f'cuda:{config.local_rank}' if torch.cuda.is_available() else 'cpu')
        self.is_main_process = config.local_rank == 0
        
        # Load models
        self.setup_models()
        
        # Load data
        self.setup_data()
        
        # Setup optimizer
        self.setup_optimizer()
        
        # Load forget mask
        if config.apply_token_masking:
            self.load_forget_mask()
        else:
            self.forget_mask = None
            
        # Initialize wandb
        if config.use_wandb and self.is_main_process:
            wandb.init(project=config.wandb_project, name=config.run_name, config=config.__dict__)
            
    def setup_logging(self):
        logging.basicConfig(
            level=getattr(logging, self.config.log_level),
            format='[%(asctime)s] [%(levelname)s] %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
    def setup_models(self):
        """Load policy and reference models"""
        self.logger.info(f"Loading model: {self.config.model_name_or_path}")
        
        # Load tokenizer
        tokenizer_path = self.config.tokenizer_name_or_path or self.config.model_name_or_path
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_path)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            
        # Load policy model (the one we're training)
        self.policy_model = transformers.AutoModelForCausalLM.from_pretrained(
            self.config.model_name_or_path,
            torch_dtype=torch.bfloat16,
            device_map='auto' if self.config.world_size == 1 else None
        )
        
        if self.config.world_size > 1:
            self.policy_model = self.policy_model.to(self.device)
        
        # Load reference model (frozen)
        ref_path = self.config.reference_model_path or self.config.model_name_or_path
        self.logger.info(f"Loading reference model: {ref_path}")
        
        self.reference_model = transformers.AutoModelForCausalLM.from_pretrained(
            ref_path,
            torch_dtype=torch.bfloat16,
            device_map='auto' if self.config.world_size == 1 else None
        )
        
        if self.config.world_size > 1:
            self.reference_model = self.reference_model.to(self.device)
            
        self.reference_model.eval()
        
        # Freeze reference model
        for param in self.reference_model.parameters():
            param.requires_grad = False
            
    def setup_data(self):
        """Setup data loaders for forget and retain datasets"""
        from datasets import load_dataset
        
        # Load forget dataset
        self.logger.info(f"Loading forget dataset: {self.config.forget_dataset_path}")
        forget_ds = load_dataset(self.config.forget_dataset_path, split=self.config.forget_split)
        
        # Load retain dataset (optional, for regularization)
        if self.config.retain_dataset_path:
            retain_ds = load_dataset(self.config.retain_dataset_path, split=self.config.retain_split)
        else:
            retain_ds = load_dataset(self.config.forget_dataset_path, split=self.config.retain_split)
            
        self.forget_dataset = self.tokenize_dataset(forget_ds, "forget")
        self.retain_dataset = self.tokenize_dataset(retain_ds, "retain") if retain_ds else None
        
        self.logger.info(f"Forget dataset size: {len(self.forget_dataset)}")
        if self.retain_dataset:
            self.logger.info(f"Retain dataset size: {len(self.retain_dataset)}")
        
    def tokenize_dataset(self, dataset, split_name):
        """Tokenize a dataset"""
        def tokenize_function(examples):
            # Adapt this based on your dataset structure
            if 'question' in examples and 'answer' in examples:
                texts = [f"Question: {q}\nAnswer: {a}" for q, a in zip(examples['question'], examples['answer'])]
            elif 'text' in examples:
                texts = examples['text']
            else:
                raise ValueError(f"Unknown dataset format for {split_name}")
                
            tokenized = self.tokenizer(
                texts,
                truncation=True,
                padding='max_length',
                max_length=self.config.max_length,
                return_tensors='pt'
            )
            
            # Create labels (same as input_ids for causal LM)
            tokenized['labels'] = tokenized['input_ids'].clone()
            
            return tokenized
            
        return dataset.map(tokenize_function, batched=True, remove_columns=dataset.column_names)
    
    def setup_optimizer(self):
        """Setup optimizer and scheduler"""
        self.optimizer = torch.optim.AdamW(
            self.policy_model.parameters(),
            lr=self.config.learning_rate
        )
        
        self.scheduler = torch.optim.lr_scheduler.LinearLR(
            self.optimizer,
            start_factor=0.1,
            total_iters=self.config.warmup_steps
        )
        
    def load_forget_mask(self):
        """Load the forget token mask"""
        self.logger.info(f"Loading forget mask: {self.config.forget_mask_path}")
        self.forget_mask = torch.load(self.config.forget_mask_path, map_location='cpu')
        self.logger.info(f"Forget mask shape: {self.forget_mask.shape}")
        self.logger.info(f"Forget tokens: {self.forget_mask.sum().item()} / {self.forget_mask.numel()}")
        
    def get_batch(self, split: str = "forget"):
        """Get a batch of data"""
        dataset = self.forget_dataset if split == "forget" else self.retain_dataset
        
        indices = torch.randint(0, len(dataset), (self.config.batch_size,))
        batch = {k: torch.stack([dataset[i][k] for i in indices]) for k in dataset[0].keys()}
        
        # Move to device
        batch = {k: v.to(self.device) for k, v in batch.items()}
        
        # Apply forget mask if available and requested
        if split == "forget" and self.forget_mask is not None and self.config.apply_token_masking:
            # Ensure forget mask matches batch size
            if self.forget_mask.shape[0] >= self.config.batch_size:
                batch_forget_mask = self.forget_mask[:self.config.batch_size].to(self.device)
            else:
                # Repeat mask to match batch size
                repeat_factor = (self.config.batch_size + self.forget_mask.shape[0] - 1) // self.forget_mask.shape[0]
                extended_mask = self.forget_mask.repeat(repeat_factor, 1)
                batch_forget_mask = extended_mask[:self.config.batch_size].to(self.device)
            
            batch['forget_mask'] = batch_forget_mask
        else:
            # No masking - forget all tokens
            batch['forget_mask'] = torch.ones_like(batch['labels'], dtype=torch.bool, device=self.device)
            
        return batch
        
    def train_step(self) -> Dict:
        """Single training step"""
        self.policy_model.train()
        
        # Get batch
        batch = self.get_batch("forget")
        
        # Forward pass through policy model
        policy_outputs = self.policy_model(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            labels=batch['labels']
        )
        
        # Forward pass through reference model (no gradients)
        with torch.no_grad():
            ref_outputs = self.reference_model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask']
            )
            
        # Compute token-level NPO loss
        loss, forget_loss, metrics = token_level_npo_loss(
            forget_logits=policy_outputs.logits,
            forget_labels=batch['labels'],
            forget_mask=batch['forget_mask'],
            reference_logits=ref_outputs.logits,
            beta=self.config.beta,
            temperature=self.config.temperature,
            forget_weight=self.config.forget_weight,
            retain_weight=self.config.retain_weight
        )
        
        return loss, metrics
        
    def eval_step(self) -> Dict:
        """Evaluation step"""
        self.policy_model.eval()
        
        all_metrics = defaultdict(list)
        
        with torch.no_grad():
            for _ in range(10):  # Evaluate on 10 batches
                batch = self.get_batch("forget")
                
                # Forward passes
                policy_outputs = self.policy_model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    labels=batch['labels']
                )
                
                ref_outputs = self.reference_model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask']
                )
                
                # Compute metrics
                loss, forget_loss, metrics = token_level_npo_loss(
                    forget_logits=policy_outputs.logits,
                    forget_labels=batch['labels'],
                    forget_mask=batch['forget_mask'],
                    reference_logits=ref_outputs.logits,
                    beta=self.config.beta,
                    temperature=self.config.temperature,
                    forget_weight=self.config.forget_weight,
                    retain_weight=self.config.retain_weight
                )
                
                metrics['loss'] = loss.item()
                for k, v in metrics.items():
                    all_metrics[k].append(v)
                    
        # Average metrics
        avg_metrics = {k: np.mean(v) for k, v in all_metrics.items()}
        return avg_metrics
        
    def train(self):
        """Main training loop"""
        self.logger.info("Starting token-level NPO training...")
        
        step = 0
        accumulated_loss = 0.0
        
        for step in range(self.config.num_train_steps):
            # Training step
            loss, train_metrics = self.train_step()
            
            # Backward pass
            (loss / self.config.gradient_accumulation_steps).backward()
            accumulated_loss += loss.item()
            
            if (step + 1) % self.config.gradient_accumulation_steps == 0:
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.policy_model.parameters(), self.config.max_grad_norm)
                
                # Optimizer step
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
                
                # Log metrics
                if self.is_main_process:
                    avg_loss = accumulated_loss / self.config.gradient_accumulation_steps
                    
                    log_dict = {
                        'step': step,
                        'train_loss': avg_loss,
                        'learning_rate': self.scheduler.get_last_lr()[0],
                        **{f'train_{k}': v for k, v in train_metrics.items()}
                    }
                    
                    if step % 10 == 0:
                        self.logger.info(f"Step {step}: loss={avg_loss:.4f}, forget_tokens={train_metrics.get('forget_tokens', 0)}")
                        
                    if self.config.use_wandb:
                        wandb.log(log_dict, step=step)
                        
                accumulated_loss = 0.0
            
            # Evaluation
            if (step + 1) % self.config.eval_every == 0:
                self.logger.info(f"Running evaluation at step {step + 1}")
                eval_metrics = self.eval_step()
                
                if self.is_main_process:
                    self.logger.info(f"Eval metrics: {eval_metrics}")
                    
                    if self.config.use_wandb:
                        wandb.log({f'eval_{k}': v for k, v in eval_metrics.items()}, step=step)
                        
            # Save checkpoint  
            if (step + 1) % self.config.save_every == 0 and self.is_main_process:
                self.save_checkpoint(step + 1)
                
        # Final save
        if self.is_main_process:
            self.save_checkpoint(step + 1, final=True)
            
        self.logger.info("Training completed!")
        
    def save_checkpoint(self, step: int, final: bool = False):
        """Save model checkpoint"""
        output_dir = os.path.join(self.config.output_dir, f"checkpoint-{step}" if not final else "final")
        os.makedirs(output_dir, exist_ok=True)
        
        self.logger.info(f"Saving checkpoint to {output_dir}")
        
        # Save model and tokenizer
        self.policy_model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
        # Save config
        with open(os.path.join(output_dir, "config.json"), "w") as f:
            json.dump(self.config.__dict__, f, indent=2)
            
        self.logger.info("Checkpoint saved successfully")


def main():
    parser = argparse.ArgumentParser(description="Token-level NPO Training")
    parser.add_argument("--config", type=str, default="config/npo_config.yaml", help="Config file path")
    parser.add_argument("--model_name_or_path", type=str, help="Model path")
    parser.add_argument("--forget_mask_path", type=str, help="Path to forget token mask")
    parser.add_argument("--output_dir", type=str, default="./token_npo_output", help="Output directory")
    parser.add_argument("--learning_rate", type=float, help="Learning rate")
    parser.add_argument("--beta", type=float, help="NPO beta parameter")
    parser.add_argument("--num_train_steps", type=int, help="Number of training steps")
    parser.add_argument("--local_rank", type=int, default=0, help="Local rank for distributed training")
    
    args = parser.parse_args()
    
    # Load config from file if it exists
    if os.path.exists(args.config):
        config_dict = OmegaConf.load(args.config)
        config = NPOConfig(**config_dict)
    else:
        config = NPOConfig()
    
    # Override config with command line arguments
    for key, value in vars(args).items():
        if value is not None:
            setattr(config, key, value)
            
    # Set up distributed training
    if 'WORLD_SIZE' in os.environ:
        config.world_size = int(os.environ['WORLD_SIZE'])
        config.local_rank = int(os.environ['LOCAL_RANK'])
    
    # Create trainer and start training
    trainer = TokenLevelNPOTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()