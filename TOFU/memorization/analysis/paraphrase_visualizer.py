"""
Paraphrase Analysis Visualization Module

Handles all visualization tasks for paraphrase memorization analysis.
"""

import os
import logging
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


class ParaphraseVisualizer:
    """Handles visualization for paraphrase analysis results"""

    def __init__(self, output_dir: str):
        """
        Initialize visualizer

        Args:
            output_dir: Directory to save visualization plots
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Set matplotlib style
        plt.style.use('default')
        logger.info(f"📊 ParaphraseVisualizer initialized with output_dir={output_dir}")

    def create_memorization_simplicity_plot(self, results: List[Dict[str, Any]], tag: str):
        """Create scatter plot of memorization vs simplicity scores

        Args:
            results: List of analysis results with memorization and simplicity scores
            tag: Tag for naming the output file
        """
        if not results:
            logger.warning(f"No results provided for memorization-simplicity plot ({tag})")
            return

        try:
            memorization_scores = [r['memorization_score'] for r in results]
            simplicity_scores = [r['simplicity_score'] for r in results]

            plt.figure(figsize=(10, 6))
            plt.scatter(simplicity_scores, memorization_scores, alpha=0.6, s=50)
            plt.xlabel('Simplicity Score', fontsize=12)
            plt.ylabel('Memorization Score', fontsize=12)
            plt.title(f'Memorization vs Simplicity ({tag})', fontsize=14)
            plt.grid(True, alpha=0.3)

            # Add diagonal reference line
            plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
            plt.axvline(x=0, color='gray', linestyle='--', alpha=0.5)

            plot_path = os.path.join(self.output_dir, f'memorization_simplicity_{tag}.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()

            logger.info(f"✅ Saved memorization-simplicity plot to {plot_path}")

        except Exception as e:
            logger.error(f"Failed to create memorization-simplicity plot for {tag}: {e}")

    def create_original_vs_paraphrase_plot(self, original_results: List[Dict[str, Any]],
                                          paraphrase_results: List[Dict[str, Any]],
                                          tag: str = "comparison"):
        """Create comparison plot between original and paraphrase memorization

        Args:
            original_results: Results from original questions
            paraphrase_results: Results from paraphrased questions
            tag: Tag for naming the output file
        """
        if not original_results or not paraphrase_results:
            logger.warning(f"Insufficient results for comparison plot ({tag})")
            return

        try:
            orig_mem = np.mean([r['memorization_score'] for r in original_results])
            para_mem = np.mean([r['memorization_score'] for r in paraphrase_results])
            orig_simp = np.mean([r['simplicity_score'] for r in original_results])
            para_simp = np.mean([r['simplicity_score'] for r in paraphrase_results])

            fig, axes = plt.subplots(1, 2, figsize=(14, 6))

            # Memorization comparison
            axes[0].bar(['Original', 'Paraphrase'], [orig_mem, para_mem],
                       color=['#2E86AB', '#A23B72'], alpha=0.7)
            axes[0].set_ylabel('Memorization Score', fontsize=12)
            axes[0].set_title('Average Memorization Score', fontsize=14)
            axes[0].grid(True, alpha=0.3, axis='y')
            axes[0].axhline(y=0, color='gray', linestyle='--', alpha=0.5)

            # Simplicity comparison
            axes[1].bar(['Original', 'Paraphrase'], [orig_simp, para_simp],
                       color=['#2E86AB', '#A23B72'], alpha=0.7)
            axes[1].set_ylabel('Simplicity Score', fontsize=12)
            axes[1].set_title('Average Simplicity Score', fontsize=14)
            axes[1].grid(True, alpha=0.3, axis='y')
            axes[1].axhline(y=0, color='gray', linestyle='--', alpha=0.5)

            plt.tight_layout()
            plot_path = os.path.join(self.output_dir, f'original_vs_paraphrase_{tag}.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()

            logger.info(f"✅ Saved original vs paraphrase plot to {plot_path}")

        except Exception as e:
            logger.error(f"Failed to create original vs paraphrase plot for {tag}: {e}")

    def create_score_distribution_plot(self, results: List[Dict[str, Any]], tag: str):
        """Create distribution histograms for memorization and simplicity scores

        Args:
            results: List of analysis results
            tag: Tag for naming the output file
        """
        if not results:
            logger.warning(f"No results provided for distribution plot ({tag})")
            return

        try:
            memorization_scores = [r['memorization_score'] for r in results]
            simplicity_scores = [r['simplicity_score'] for r in results]

            fig, axes = plt.subplots(1, 2, figsize=(14, 6))

            # Memorization distribution
            axes[0].hist(memorization_scores, bins=30, alpha=0.7, color='#2E86AB', edgecolor='black')
            axes[0].set_xlabel('Memorization Score', fontsize=12)
            axes[0].set_ylabel('Frequency', fontsize=12)
            axes[0].set_title(f'Memorization Score Distribution ({tag})', fontsize=14)
            axes[0].axvline(x=np.mean(memorization_scores), color='red',
                           linestyle='--', label=f'Mean: {np.mean(memorization_scores):.3f}')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3, axis='y')

            # Simplicity distribution
            axes[1].hist(simplicity_scores, bins=30, alpha=0.7, color='#A23B72', edgecolor='black')
            axes[1].set_xlabel('Simplicity Score', fontsize=12)
            axes[1].set_ylabel('Frequency', fontsize=12)
            axes[1].set_title(f'Simplicity Score Distribution ({tag})', fontsize=14)
            axes[1].axvline(x=np.mean(simplicity_scores), color='red',
                           linestyle='--', label=f'Mean: {np.mean(simplicity_scores):.3f}')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3, axis='y')

            plt.tight_layout()
            plot_path = os.path.join(self.output_dir, f'score_distribution_{tag}.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()

            logger.info(f"✅ Saved score distribution plot to {plot_path}")

        except Exception as e:
            logger.error(f"Failed to create distribution plot for {tag}: {e}")

    def create_all_visualizations(self, original_results: List[Dict[str, Any]],
                                 paraphrase_results: List[Dict[str, Any]]):
        """Create all standard visualizations for paraphrase analysis

        Args:
            original_results: Results from original questions
            paraphrase_results: Results from paraphrased questions
        """
        logger.info("📊 Creating all visualizations...")

        # Individual plots
        self.create_memorization_simplicity_plot(original_results, "original")
        self.create_memorization_simplicity_plot(paraphrase_results, "paraphrase")

        # Comparison plot
        self.create_original_vs_paraphrase_plot(original_results, paraphrase_results, "comparison")

        # Distribution plots
        self.create_score_distribution_plot(original_results, "original")
        self.create_score_distribution_plot(paraphrase_results, "paraphrase")

        logger.info("✅ All visualizations created successfully")
