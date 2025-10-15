import sys
import os
import torch
import json
import numpy as np
from typing import List, Dict, Any
import logging

# Add parent directory to path
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.insert(0, parent_dir)

from transformers import AutoTokenizer, AutoModelForCausalLM
from utils import get_model_identifiers_from_yaml

# Setup logger
logger = logging.getLogger(__name__)


def create_paraphrased_dataset(original_dataset, paraphrases_list, tokenizer, model_family):
    """Create dataset from paraphrased questions"""
    from torch.utils.data import Dataset

    class ParaphraseDataset(Dataset):
        def __init__(self, original_dataset, paraphrases_list, tokenizer, model_family):
            self.data = []

            # Flatten paraphrases
            for i, paraphrases in enumerate(paraphrases_list):
                original_item = original_dataset[i]
                for para in paraphrases:
                    self.data.append({
                        'question': para,
                        'answer': original_item['answer'],
                        'original_idx': i
                    })

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            return self.data[idx]

    return ParaphraseDataset(original_dataset, paraphrases_list, tokenizer, model_family)


def compute_per_token_accuracy(batch: dict, model: AutoModelForCausalLM, tokenizer: AutoTokenizer) -> List[float]:
    """토큰별 정확도 계산 - 간단 버전"""

    with torch.no_grad():
        model.eval()
        outputs = model(**batch, return_dict=True)
        logits = outputs.logits

        # 예측값과 정답값 추출
        predictions = torch.argmax(logits, dim=-1)
        labels = batch["labels"]

        # 배치별 정확도 계산
        accuracies = []
        for i in range(predictions.size(0)):
            # 유효한 토큰만 선택 (labels != -100)
            valid_mask = labels[i] != -100
            if valid_mask.sum() == 0:
                accuracies.append(0.0)
                continue

            # 정확도 계산
            pred_tokens = predictions[i][valid_mask]
            true_tokens = labels[i][valid_mask]
            accuracy = (pred_tokens == true_tokens).float().mean().item()
            accuracies.append(accuracy)

    return accuracies


def calculate_counterfactual_scores(batch_dict: dict, full_model: AutoModelForCausalLM,
                                    retain_model: AutoModelForCausalLM, tokenizer: AutoTokenizer):
    """
    Calculate counterfactual memorization scores using per-token accuracy

    Returns:
        acc_in_scores: Accuracy when example is in training set (full model)
        acc_out_scores: Accuracy when example is excluded (retain model)
    """
    with torch.no_grad():
        full_model.eval()
        acc_in_scores = compute_per_token_accuracy(batch_dict, full_model, tokenizer)

        retain_model.eval()
        acc_out_scores = compute_per_token_accuracy(batch_dict, retain_model, tokenizer)

    return acc_in_scores, acc_out_scores


def load_models_and_tokenizer(model_family, full_model_path=None, retain_model_path=None):
    model_cfg = get_model_identifiers_from_yaml(model_family)
    model_id = model_cfg["hf_key"]

    # Use default paths from model_config.yaml if not specified
    if full_model_path is None:
        full_model_path = model_id
    if retain_model_path is None:
        retain_model_path = model_cfg.get("ft_model_path", model_id)

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token

    # Full model (trained on all data)
    print(f"Loading full model from: {full_model_path}")
    full_model = AutoModelForCausalLM.from_pretrained(full_model_path)

    # Retain model (trained without forget set)
    print(f"Loading retain model from: {retain_model_path}")
    retain_model = AutoModelForCausalLM.from_pretrained(retain_model_path)

    return full_model, retain_model, tokenizer


def calculate_counterfactual_memorization(acc_in_scores, acc_out_scores):
    """
    Calculate counterfactual memorization and simplicity metrics

    Based on Feldman (2020):
    - mem(x) = M(A(S ∪ {x}), x) - M(A(S), x)
    - simplicity(x) = min(M(A(S ∪ {x}), x), M(A(S), x))

    Returns:
        memorization_scores: Counterfactual memorization scores
        simplicity_scores: Generalization/simplicity scores
    """
    acc_in_array = np.array(acc_in_scores)
    acc_out_array = np.array(acc_out_scores)

    # Counterfactual memorization: mem(x) = Acc_IN - Acc_OUT
    memorization_scores = acc_in_array - acc_out_array

    # Simplicity score: min(Acc_IN, Acc_OUT)
    simplicity_scores = np.minimum(acc_in_array, acc_out_array)

    return memorization_scores.tolist(), simplicity_scores.tolist()


def get_output_dir(config) -> str:
    """Get output directory for saving results"""
    out_dir = getattr(config.analysis, "output_dir", None)
    if out_dir:
        base = out_dir
    else:
        output_file = getattr(config.analysis, "output_file", "")
        if output_file and os.path.dirname(output_file):
            base = os.path.dirname(output_file)
        else:
            base = "results"
    os.makedirs(base, exist_ok=True)
    return base


def save_results(results: Any, tag: str, output_dir: str) -> str:
    """Save results to JSON file"""
    output_file = os.path.join(output_dir, f"{tag}_results.json")

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    logger.info(f"✅ Saved results to {output_file}")
    return output_file


def compute_comparison_summary(comparisons: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute summary statistics for comparisons"""
    if not comparisons:
        return {}

    mem_diffs = [c['memorization_diff'] for c in comparisons]
    simp_diffs = [c['simplicity_diff'] for c in comparisons]

    summary = {
        'num_comparisons': len(comparisons),
        'memorization_diff': {
            'mean': float(np.mean(mem_diffs)),
            'std': float(np.std(mem_diffs)),
            'min': float(np.min(mem_diffs)),
            'max': float(np.max(mem_diffs))
        },
        'simplicity_diff': {
            'mean': float(np.mean(simp_diffs)),
            'std': float(np.std(simp_diffs)),
            'min': float(np.min(simp_diffs)),
            'max': float(np.max(simp_diffs))
        }
    }

    return summary


def combine_and_save_results(
    original_results: List[Dict[str, Any]],
    paraphrase_results: List[Dict[str, Any]],
    output_dir: str,
    tag: str = "combined"
) -> Dict[str, Any]:
    """
    Combine original and paraphrase results, compute comparisons, save

    Args:
        original_results: Results from step 1
        paraphrase_results: Results from step 3
        output_dir: Directory to save results
        tag: Tag for combined results

    Returns:
        Combined results dictionary
    """
    logger.info(f"🚀 Combining and saving results: {tag}")

    # Create comparison data
    comparisons = []
    for i in range(min(len(original_results), len(paraphrase_results))):
        orig = original_results[i]
        para = paraphrase_results[i]

        comparison = {
            'original_question': orig['question'],
            'paraphrase_question': para['question'],
            'original_answer': orig['answer'],
            'paraphrase_answer': para['answer'],
            'original_memorization': orig['memorization'],
            'paraphrase_memorization': para['memorization'],
            'original_simplicity': orig['simplicity'],
            'paraphrase_simplicity': para['simplicity'],
            'memorization_diff': para['memorization'] - orig['memorization'],
            'simplicity_diff': para['simplicity'] - orig['simplicity']
        }
        comparisons.append(comparison)

    # Compute summary statistics
    summary = compute_comparison_summary(comparisons)

    # Combined results
    combined_results = {
        'original_results': original_results,
        'paraphrase_results': paraphrase_results,
        'comparisons': comparisons,
        'summary': summary,
        'tag': tag
    }

    # Save combined results
    save_results(combined_results, tag, output_dir)

    # Log summary
    logger.info(f"✅ Combined {len(comparisons)} result pairs")
    logger.info(f"📊 Memorization diff - Mean: {summary['memorization_diff']['mean']:.4f}, Std: {summary['memorization_diff']['std']:.4f}")
    logger.info(f"📊 Simplicity diff - Mean: {summary['simplicity_diff']['mean']:.4f}, Std: {summary['simplicity_diff']['std']:.4f}")

    return combined_results


def save_analysis_results(results: Dict[str, Any], output_dir: str, tag: str) -> str:
    """Save analysis results to JSON file

    Args:
        results: Results dictionary to save
        output_dir: Directory to save results
        tag: Tag for the results file

    Returns:
        Path to saved JSON file
    """
    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, f"{tag}_results.json")

    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    logger.info(f"Results saved to {json_path}")
    return json_path


def combine_dual_model_results(
    original_results: List[Dict[str, Any]],
    paraphrase_results: List[Dict[str, Any]],
    output_dir: str,
    tag: str = "combined"
) -> Dict[str, Any]:
    """Combine original and paraphrase results from dual model analysis

    Args:
        original_results: Results from analyzing original questions
        paraphrase_results: Results from analyzing paraphrased questions
        output_dir: Directory to save results
        tag: Tag for identifying this combined analysis

    Returns:
        Combined results dictionary
    """
    logger.info(f"Combining dual model results with tag '{tag}'...")

    combined_results = {
        'original_results': original_results,
        'paraphrase_results': paraphrase_results,
        'summary': {
            'num_original': len(original_results),
            'num_paraphrase': len(paraphrase_results),
            'original_avg_memorization': np.mean([r['memorization_score'] for r in original_results]) if original_results else 0.0,
            'paraphrase_avg_memorization': np.mean([r['memorization_score'] for r in paraphrase_results]) if paraphrase_results else 0.0,
            'memorization_difference': 0.0
        },
        'tag': tag
    }

    # Calculate memorization difference
    if original_results and paraphrase_results:
        combined_results['summary']['memorization_difference'] = (
            combined_results['summary']['paraphrase_avg_memorization'] -
            combined_results['summary']['original_avg_memorization']
        )

    # Save results
    save_analysis_results(combined_results, output_dir, tag)

    logger.info(f"✅ Combined results saved successfully")
    logger.info(f"📊 Memorization difference (paraphrase - original): {combined_results['summary']['memorization_difference']:.4f}")

    return combined_results
