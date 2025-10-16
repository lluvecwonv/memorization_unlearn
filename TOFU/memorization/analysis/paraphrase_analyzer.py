import sys
import os
import logging
import numpy as np
from typing import Dict, List, Any
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt

# Add parent directory to path for imports
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(script_dir))
sys.path.insert(0, parent_dir)

from evaluate_util import run_generation
from memorization.utils import compute_per_token_accuracy, combine_dual_model_results

logger = logging.getLogger(__name__)


class DualModelAnalyzer:

    def __init__(self, batch_size: int, output_dir: str, model_config: Any):
      
        self.batch_size = batch_size
        self.output_dir = output_dir
        self.model_config = model_config

        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"🚀 DualModelAnalyzer initialized with output_dir={output_dir}")

    def run_dual_model_analysis(self, full_model, retain_model, tokenizer, dataset, tag: str = "analysis") -> List[Dict[str, Any]]:

        logger.info(f"Running dual model analysis with tag '{tag}'...")

        # Set models to eval mode
        full_model.eval()
        retain_model.eval()

        # Custom collate function for TextDatasetQA
        def collate_fn(samples):
            # TextDatasetQA returns (input_ids, labels, attention_mask) directly
            input_ids = torch.stack([sample[0] for sample in samples])
            labels = torch.stack([sample[1] for sample in samples])
            attention_mask = torch.stack([sample[2] for sample in samples])

            return input_ids, labels, attention_mask

        # Create dataloader
        from torch.utils.data import DataLoader
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False, collate_fn=collate_fn)

        all_results = []

        with torch.inference_mode():
            for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Analyzing ({tag})")):
                # Unpack batch from collate_fn
                input_ids, labels, attention_mask = batch

                # Move to device
                input_ids = input_ids.to(full_model.device)
                labels = labels.to(full_model.device)
                attention_mask = attention_mask.to(full_model.device)

                batch_dict = {
                    'input_ids': input_ids,
                    'attention_mask': attention_mask,
                    'labels': labels
                }

                # Calculate counterfactual scores
                acc_in_scores, acc_out_scores = self._calculate_counterfactual_scores(
                    batch_dict, full_model, retain_model, tokenizer
                )

                # Calculate memorization and simplicity
                memorization_scores = [acc_in - acc_out for acc_in, acc_out in zip(acc_in_scores, acc_out_scores)]
                simplicity_scores = [acc_in + acc_out for acc_in, acc_out in zip(acc_in_scores, acc_out_scores)]

                # Generate text for inspection
                input_texts, output_texts, ground_truths = run_generation(
                    self.model_config, batch_dict, full_model, tokenizer
                )

                # Store results
                for i in range(len(output_texts)):
                    result = {
                        'question': input_texts[i],
                        'generated_answer': output_texts[i],
                        'ground_truth': ground_truths[i],
                        'acc_in_score': acc_in_scores[i],
                        'acc_out_score': acc_out_scores[i],
                        'memorization_score': memorization_scores[i],
                        'simplicity_score': simplicity_scores[i],
                        'batch_idx': batch_idx,
                        'example_idx': i,
                        'tag': tag
                    }
                    all_results.append(result)

        logger.info(f"✅ Dual model analysis completed: {len(all_results)} examples processed")

        # Calculate and log summary statistics
        if all_results:
            avg_acc_in = np.mean([r['acc_in_score'] for r in all_results])
            avg_acc_out = np.mean([r['acc_out_score'] for r in all_results])
            avg_memorization = np.mean([r['memorization_score'] for r in all_results])
            avg_simplicity = np.mean([r['simplicity_score'] for r in all_results])

            logger.info(f"📊 Summary for '{tag}':")
            logger.info(f"  • Avg Acc_IN (full model): {avg_acc_in:.4f}")
            logger.info(f"  • Avg Acc_OUT (retain model): {avg_acc_out:.4f}")
            logger.info(f"  • Avg Memorization: {avg_memorization:.4f}")
            logger.info(f"  • Avg Simplicity: {avg_simplicity:.4f}")

        return all_results

    def combine_and_save_results(self, original_results: List[Dict[str, Any]],
                                 paraphrase_results: List[Dict[str, Any]],
                                 tag: str = "combined") -> Dict[str, Any]:
        """Combine original and paraphrase results and save to disk

        Args:
            original_results: Results from analyzing original questions
            paraphrase_results: Results from analyzing paraphrased questions
            tag: Tag for identifying this combined analysis

        Returns:
            Combined results dictionary
        """
        return combine_dual_model_results(original_results, paraphrase_results, self.output_dir, tag)

    def _calculate_counterfactual_scores(self, batch_dict, full_model, retain_model, tokenizer):
        """Calculate counterfactual memorization scores using per-token accuracy"""
        with torch.no_grad():
            # Calculate per-token accuracy for full model (IN condition)
            full_model.eval()
            # Move batch to full_model's device
            batch_full = {k: v.to(full_model.device) if isinstance(v, torch.Tensor) else v
                         for k, v in batch_dict.items()}
            acc_in_scores = compute_per_token_accuracy(batch_full, full_model, tokenizer)

            # Calculate per-token accuracy for retain model (OUT condition)
            retain_model.eval()
            # Move batch to retain_model's device
            batch_retain = {k: v.to(retain_model.device) if isinstance(v, torch.Tensor) else v
                           for k, v in batch_dict.items()}
            acc_out_scores = compute_per_token_accuracy(batch_retain, retain_model, tokenizer)

        return acc_in_scores, acc_out_scores

    def _create_memorization_simplicity_plot(self, results: List[Dict[str, Any]], tag: str):
        """Create memorization vs simplicity scatter plot"""
        try:
            if not results:
                logger.warning("Empty data for memorization-simplicity plot")
                return
                
            memorization = [r.get('memorization_score', 0) for r in results]
            simplicity = [r.get('simplicity_score', 0) for r in results]
            
            plt.figure(figsize=(10, 8))
            plt.scatter(memorization, simplicity, alpha=0.6, s=50)
            plt.xlabel('Memorization Score')
            plt.ylabel('Simplicity Score')
            plt.title(f'Memorization vs Simplicity Analysis ({tag})')
            plt.grid(True, alpha=0.3)
            
            # Save plot
            plot_path = os.path.join(self.output_dir, f'{tag}_memorization_simplicity.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Saved memorization-simplicity plot to {plot_path}")
            
        except Exception as e:
            logger.error(f"Failed to create memorization-simplicity plot: {e}")

    def _create_original_vs_paraphrase_plot(self, original_results: List[Dict[str, Any]], 
                                          paraphrase_results: List[Dict[str, Any]], tag: str):
        """Create original vs paraphrase comparison plot"""
        try:
            if not original_results or not paraphrase_results:
                logger.warning("Empty data for original-paraphrase plot")
                return
                
            # Extract memorization scores
            original_scores = [r.get('memorization_score', 0) for r in original_results]
            paraphrase_scores = [r.get('memorization_score', 0) for r in paraphrase_results]
            
            # Ensure same length
            min_len = min(len(original_scores), len(paraphrase_scores))
            original_scores = original_scores[:min_len]
            paraphrase_scores = paraphrase_scores[:min_len]
            
            plt.figure(figsize=(10, 8))
            plt.scatter(original_scores, paraphrase_scores, alpha=0.6, s=50)
            
            # Add diagonal line for reference
            min_val = min(min(original_scores), min(paraphrase_scores))
            max_val = max(max(original_scores), max(paraphrase_scores))
            plt.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.7, label='y=x')
            
            plt.xlabel('Original Memorization Scores')
            plt.ylabel('Paraphrase Memorization Scores')
            plt.title(f'Original vs Paraphrase Analysis ({tag})')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Save plot
            plot_path = os.path.join(self.output_dir, f'{tag}_original_vs_paraphrase.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Saved original vs paraphrase plot to {plot_path}")
            
        except Exception as e:
            logger.error(f"Failed to create original vs paraphrase plot: {e}")

    def _create_score_distribution_plot(self, results: List[Dict[str, Any]], tag: str):
        """Create score distribution histograms"""
        try:
            if not results:
                logger.warning("Empty data for score distribution plot")
                return
                
            acc_in = [r.get('acc_in_score', 0) for r in results]
            acc_out = [r.get('acc_out_score', 0) for r in results]
            memorization = [r.get('memorization_score', 0) for r in results]
            simplicity = [r.get('simplicity_score', 0) for r in results]
            
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            
            # Acc IN scores
            axes[0, 0].hist(acc_in, bins=30, alpha=0.7, color='blue')
            axes[0, 0].set_title('Acc IN Score Distribution')
            axes[0, 0].set_xlabel('Score')
            axes[0, 0].set_ylabel('Frequency')
            axes[0, 0].grid(True, alpha=0.3)
            
            # Acc OUT scores
            axes[0, 1].hist(acc_out, bins=30, alpha=0.7, color='green')
            axes[0, 1].set_title('Acc OUT Score Distribution')
            axes[0, 1].set_xlabel('Score')
            axes[0, 1].set_ylabel('Frequency')
            axes[0, 1].grid(True, alpha=0.3)
            
            # Memorization scores
            axes[1, 0].hist(memorization, bins=30, alpha=0.7, color='red')
            axes[1, 0].set_title('Memorization Score Distribution')
            axes[1, 0].set_xlabel('Score')
            axes[1, 0].set_ylabel('Frequency')
            axes[1, 0].grid(True, alpha=0.3)
            
            # Simplicity scores
            axes[1, 1].hist(simplicity, bins=30, alpha=0.7, color='orange')
            axes[1, 1].set_title('Simplicity Score Distribution')
            axes[1, 1].set_xlabel('Score')
            axes[1, 1].set_ylabel('Frequency')
            axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save plot
            plot_path = os.path.join(self.output_dir, f'{tag}_score_distributions.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Saved score distribution plot to {plot_path}")
            
        except Exception as e:
            logger.error(f"Failed to create score distribution plot: {e}")
