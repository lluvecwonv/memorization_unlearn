#!/usr/bin/env python3
"""
Evaluation script for token-level NPO unlearning.
Compares model performance before and after unlearning.
"""

import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import numpy as np
import argparse
import logging
import json
import os
from typing import List, Dict, Tuple
from tqdm import tqdm

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] [%(levelname)s] %(message)s'
    )
    return logging.getLogger(__name__)

def load_model_and_tokenizer(model_path: str, device: str = "auto"):
    """Load model and tokenizer"""
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map=device
    )
    model.eval()
    
    return model, tokenizer

def compute_perplexity(model, tokenizer, texts: List[str], device: str = "cuda") -> float:
    """Compute average perplexity on a list of texts"""
    total_loss = 0.0
    total_tokens = 0
    
    with torch.no_grad():
        for text in tqdm(texts, desc="Computing perplexity"):
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            outputs = model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss
            num_tokens = (inputs["input_ids"] != tokenizer.pad_token_id).sum().item()
            
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens
    
    avg_loss = total_loss / total_tokens
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    
    return perplexity

def evaluate_generation_quality(model, tokenizer, prompts: List[str], max_length: int = 200) -> List[str]:
    """Generate responses for a list of prompts"""
    generations = []
    
    with torch.no_grad():
        for prompt in tqdm(prompts, desc="Generating responses"):
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
            outputs = model.generate(
                **inputs,
                max_length=max_length,
                num_return_sequences=1,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.pad_token_id
            )
            
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Remove the prompt from the generated text
            generated_response = generated_text[len(prompt):].strip()
            generations.append(generated_response)
            
    return generations

def load_tofu_data(split: str = "forget10", max_samples: int = 100):
    """Load TOFU dataset"""
    dataset = load_dataset("locuslab/TOFU", split=split)
    
    # Convert to text format
    texts = []
    prompts = []
    
    for i, example in enumerate(dataset):
        if i >= max_samples:
            break
            
        question = example["question"]
        answer = example["answer"]
        
        full_text = f"Question: {question}\nAnswer: {answer}"
        prompt = f"Question: {question}\nAnswer:"
        
        texts.append(full_text)
        prompts.append(prompt)
    
    return texts, prompts

def evaluate_model(model_path: str, forget_mask_path: str = None, max_samples: int = 50):
    """Evaluate a model on forget and retain sets"""
    logger = setup_logging()
    
    logger.info(f"Loading model: {model_path}")
    model, tokenizer = load_model_and_tokenizer(model_path)
    
    # Load forget mask if provided
    forget_mask = None
    if forget_mask_path and os.path.exists(forget_mask_path):
        logger.info(f"Loading forget mask: {forget_mask_path}")
        forget_mask = torch.load(forget_mask_path, map_location='cpu')
        logger.info(f"Forget mask shape: {forget_mask.shape}")
    
    results = {}
    
    # Evaluate on forget set
    logger.info("Evaluating on forget set...")
    forget_texts, forget_prompts = load_tofu_data("forget10", max_samples)
    
    forget_perplexity = compute_perplexity(model, tokenizer, forget_texts)
    forget_generations = evaluate_generation_quality(model, tokenizer, forget_prompts[:10])  # Just 10 for demo
    
    results["forget"] = {
        "perplexity": forget_perplexity,
        "num_samples": len(forget_texts),
        "sample_generations": list(zip(forget_prompts[:10], forget_generations))
    }
    
    logger.info(f"Forget set perplexity: {forget_perplexity:.2f}")
    
    # Evaluate on retain set  
    logger.info("Evaluating on retain set...")
    retain_texts, retain_prompts = load_tofu_data("retain90", max_samples)
    
    retain_perplexity = compute_perplexity(model, tokenizer, retain_texts)
    retain_generations = evaluate_generation_quality(model, tokenizer, retain_prompts[:10])
    
    results["retain"] = {
        "perplexity": retain_perplexity,
        "num_samples": len(retain_texts),
        "sample_generations": list(zip(retain_prompts[:10], retain_generations))
    }
    
    logger.info(f"Retain set perplexity: {retain_perplexity:.2f}")
    
    # Compute metrics
    results["metrics"] = {
        "forget_perplexity": forget_perplexity,
        "retain_perplexity": retain_perplexity,
        "perplexity_ratio": forget_perplexity / retain_perplexity,
        "unlearning_effectiveness": forget_perplexity / retain_perplexity > 1.0  # Higher is better for forgetting
    }
    
    return results

def compare_models(original_path: str, unlearned_path: str, forget_mask_path: str = None):
    """Compare original and unlearned models"""
    logger = setup_logging()
    
    logger.info("Evaluating original model...")
    original_results = evaluate_model(original_path, forget_mask_path)
    
    logger.info("Evaluating unlearned model...")
    unlearned_results = evaluate_model(unlearned_path, forget_mask_path)
    
    # Compute comparison metrics
    comparison = {
        "original": original_results,
        "unlearned": unlearned_results,
        "comparison": {
            "forget_perplexity_increase": unlearned_results["forget"]["perplexity"] / original_results["forget"]["perplexity"],
            "retain_perplexity_change": unlearned_results["retain"]["perplexity"] / original_results["retain"]["perplexity"],
            "unlearning_success": unlearned_results["forget"]["perplexity"] > original_results["forget"]["perplexity"],
            "retention_maintained": abs(unlearned_results["retain"]["perplexity"] - original_results["retain"]["perplexity"]) < 5.0
        }
    }
    
    # Print summary
    logger.info("\n" + "="*50)
    logger.info("EVALUATION SUMMARY")
    logger.info("="*50)
    logger.info(f"Original model forget perplexity: {original_results['forget']['perplexity']:.2f}")
    logger.info(f"Unlearned model forget perplexity: {unlearned_results['forget']['perplexity']:.2f}")
    logger.info(f"Forget perplexity increase: {comparison['comparison']['forget_perplexity_increase']:.2f}x")
    logger.info("")
    logger.info(f"Original model retain perplexity: {original_results['retain']['perplexity']:.2f}")
    logger.info(f"Unlearned model retain perplexity: {unlearned_results['retain']['perplexity']:.2f}")
    logger.info(f"Retain perplexity change: {comparison['comparison']['retain_perplexity_change']:.2f}x")
    logger.info("")
    logger.info(f"Unlearning successful: {comparison['comparison']['unlearning_success']}")
    logger.info(f"Retention maintained: {comparison['comparison']['retention_maintained']}")
    
    return comparison

def main():
    parser = argparse.ArgumentParser(description="Evaluate unlearning performance")
    parser.add_argument("--original_model", type=str, required=True, help="Path to original model")
    parser.add_argument("--unlearned_model", type=str, help="Path to unlearned model")
    parser.add_argument("--forget_mask_path", type=str, help="Path to forget mask")
    parser.add_argument("--output_path", type=str, default="evaluation_results.json", help="Output file path")
    parser.add_argument("--max_samples", type=int, default=50, help="Maximum samples to evaluate")
    parser.add_argument("--single_model", action="store_true", help="Evaluate single model only")
    
    args = parser.parse_args()
    
    if args.single_model or args.unlearned_model is None:
        # Evaluate single model
        results = evaluate_model(args.original_model, args.forget_mask_path, args.max_samples)
    else:
        # Compare two models
        results = compare_models(args.original_model, args.unlearned_model, args.forget_mask_path)
    
    # Save results
    with open(args.output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {args.output_path}")

if __name__ == "__main__":
    main()