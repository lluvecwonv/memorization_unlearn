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
    """Create dataset from paraphrased questions

    Returns dataset compatible with TextForgetDatasetQA format:
    Each item returns [forget_data, retain_data] where each is (input_ids, labels, attention_mask)
    """
    from torch.utils.data import Dataset
    from data_module import convert_raw_data_to_model_format

    class ParaphraseDataset(Dataset):
        def __init__(self, original_dataset, paraphrases_list, tokenizer, model_family):
            self.original_dataset = original_dataset
            self.paraphrases_list = paraphrases_list
            self.tokenizer = tokenizer
            self.model_family = model_family
            self.model_configs = get_model_identifiers_from_yaml(model_family)

            # Get max_length from original dataset
            self.max_length = getattr(original_dataset, 'max_length', 256)

        def __len__(self):
            # Total number of paraphrases across all questions
            return sum(len(paraphrases) for paraphrases in self.paraphrases_list)

        def __getitem__(self, idx):
            # Find which question this paraphrase belongs to
            cumulative = 0
            question_idx = 0
            for i, paraphrases in enumerate(self.paraphrases_list):
                if idx < cumulative + len(paraphrases):
                    question_idx = i
                    paraphrase_idx = idx - cumulative
                    break
                cumulative += len(paraphrases)

            # Get original answer from forget_data
            forget_data_orig = self.original_dataset.forget_data[question_idx]
            retain_data_orig = self.original_dataset.retain_data[question_idx]
            answer = forget_data_orig['answer']

            # Get paraphrased question
            paraphrased_question = self.paraphrases_list[question_idx][paraphrase_idx]

            # Convert paraphrased question with original answer (forget)
            forget_converted = convert_raw_data_to_model_format(
                self.tokenizer,
                self.max_length,
                paraphrased_question,
                answer,
                self.model_configs
            )

            # Use original retain data
            retain_converted = convert_raw_data_to_model_format(
                self.tokenizer,
                self.max_length,
                retain_data_orig['question'],
                retain_data_orig['answer'],
                self.model_configs
            )

            # Return same format as TextForgetDatasetQA
            return [forget_converted, retain_converted]

    return ParaphraseDataset(original_dataset, paraphrases_list, tokenizer, model_family)

def compute_per_token_accuracy(batch: dict, model: AutoModelForCausalLM, tokenizer: AutoTokenizer) -> List[float]:
    """토큰별 정확도 계산 (next-token shift 반영)

    - logits[:, t]는 토큰 t+1을 예측하므로, labels는 1칸 오른쪽으로 시프트하여 비교한다.
    - labels == -100인 위치는 무시한다.
    """

    with torch.no_grad():
        model.eval()
        outputs = model(**batch, return_dict=True)
        logits = outputs.logits  # [B, T, V]

        labels = batch["labels"]  # [B, T]

        # next-token prediction 기준으로 시프트
        shifted_logits = logits[:, :-1, :].contiguous()  # 예측 t -> 실제 t+1
        shifted_labels = labels[:, 1:].contiguous()

        # 예측값 (argmax)
        predictions = torch.argmax(shifted_logits, dim=-1)  # [B, T-1]

        # 배치별 정확도 계산 (labels != -100 위치만)
        accuracies = []
        for i in range(predictions.size(0)):
            valid_mask = shifted_labels[i] != -100
            if not torch.any(valid_mask):
                # 평가 가능한 토큰이 없으면 0.0으로 반환하되 경고 로그 남김
                logger.warning("No valid tokens to evaluate for a sample (all labels == -100). Returning 0.0 accuracy.")
                accuracies.append(0.0)
                continue

            pred_tokens = predictions[i][valid_mask]
            true_tokens = shifted_labels[i][valid_mask]
            accuracy = (pred_tokens == true_tokens).float().mean().item()
            accuracies.append(accuracy)

    return accuracies


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


def save_results_for_notebook(original_results: List[Dict[str, Any]],
                               paraphrase_results: List[Dict[str, Any]],
                               output_dir: str,
                               data_path: str,
                               split: str) -> str:
    """Save results in notebook-compatible format

    Args:
        original_results: Results from original questions
        paraphrase_results: Results from paraphrased questions
        output_dir: Directory to save results
        data_path: Dataset path
        split: Dataset split name

    Returns:
        Path to saved JSON file
    """
    formatted_results = []

    for i, orig in enumerate(original_results):
        formatted_results.append({
            "question": orig.get('question', ''),
            "generated_answer": orig.get('generated_answer', ''),
            "ground_truth": orig.get('ground_truth', ''),
            "acc_in_score": orig.get('acc_in_score', 0.0),
            "acc_out_score": orig.get('acc_out_score', 0.0),
            "memorization_score": orig.get('memorization_score', 0.0),
            "simplicity_score": orig.get('simplicity_score', 0.0),
            "batch_idx": orig.get('batch_idx', i),
            "example_idx": orig.get('example_idx', 0)
        })

    output_data = {
        "dataset": data_path,
        "split": split,
        "num_questions": len(formatted_results),
        "results": formatted_results
    }

    # Save with dataset name in filename
    dataset_name = data_path.replace('/', '_')
    filename = f"paraphrase_{dataset_name}_{split}_results.json"

    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, filename)

    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    logger.info(f"Notebook-compatible results saved to {json_path}")
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
