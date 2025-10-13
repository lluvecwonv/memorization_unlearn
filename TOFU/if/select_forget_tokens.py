#!/usr/bin/env python3
"""
Script to select tokens for forgetting based on differential influence scores
from compute_influence.py output. Uses harmonic mean ranking similar to 
the toxic token mask approach.

Usage:
    python select_forget_tokens.py --scores_path /path/to/scores --output_path /path/to/save
"""

import argparse
import logging
import math
import os
import numpy as np
import torch
from transformers import AutoTokenizer
from colorama import Fore, init
from tqdm import tqdm
import yaml
from dataclasses import dataclass
import re
from collections import Counter, defaultdict
from typing import Optional, Tuple, List, Dict

# Local imports - using local data loading functionality
import sys
import os

# Add TOFU directory to Python path (relative to this script's location)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TOFU_DIR = os.path.dirname(SCRIPT_DIR)  # Go up one level from 'if' to 'TOFU'

# IMPORTANT: Add TOFU_DIR first to use TOFU/data_module.py, not if/analysis/data_module.py
if TOFU_DIR not in sys.path:
    sys.path.insert(0, TOFU_DIR)

import datasets
from data_module import TextDatasetQA

init()

def _generate_json_files(tokenizer, forget_dataset, forget_mask, influence_scores, args):
    """Generate JSON files with SI scores and context"""
    # 추가: 선택된 토큰들의 SI 점수만 저장하는 기능
    _save_selected_tokens_si_scores(
        tokenizer=tokenizer,
        dataset=forget_dataset,
        forget_mask=forget_mask,
        influence_scores=influence_scores,
        output_path=args.output_path
    )
    
    # 추가: 모든 토큰의 SI 점수를 저장하는 기능
    _save_all_tokens_si_scores(
        tokenizer=tokenizer,
        dataset=forget_dataset,
        influence_scores=influence_scores,
        output_path=args.output_path
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Select tokens for forgetting using differential influence scores.")
    
    # Input paths
    parser.add_argument("--scores_path", type=str, required=True, 
                       help="Path to the influence scores file (.pkl or directory)")
    parser.add_argument("--checkpoint_dir", type=str, default=None,
                       help="Path to model checkpoint for tokenizer")
    parser.add_argument("--model_config_path", type=str, default=None,
                       help="Path to model_config.yaml (default resolves relative to this script)")
    
    # Data config (to match the original data used for scoring)
    parser.add_argument("--tofu_data_path", type=str, default="locuslab/TOFU",
                       help="TOFU dataset path")
    parser.add_argument("--forget_split", type=str, default="forget10", 
                       help="Forget split name")
    parser.add_argument("--model_family", type=str, default="pythia-1.4",
                       help="Model family for tokenizer loading")
    parser.add_argument("--max_length", type=int, default=256,
                       help="Maximum sequence length")
    
    # Selection parameters
    parser.add_argument("--top_n_per_sample", type=int, default=1,
                       help="Number of top tokens to select per sample")
    parser.add_argument("--context_window", type=int, default=3,
                       help="Context window around high-influence tokens")
    parser.add_argument("--min_score", type=float, default=0.0,
                       help="Minimum influence score threshold for token selection")
    parser.add_argument("--word_masking", action="store_true", default=False,
                       help="Expand selected token to cover the full reconstructed word (BPE pieces)")

    # Deprecated parameters (kept for backwards compatibility)
    parser.add_argument("--max_forget_tokens", type=int, default=1000000,
                       help="(Deprecated) Not used anymore")
    parser.add_argument("--top_k_samples", type=int, default=None,
                       help="(Deprecated) Not used anymore")
    parser.add_argument("--select_mode", type=str, default="topk_per_sample",
                       choices=["topk_per_sample", "topk_per_sentence"],
                       help="(Deprecated) Not used anymore")
    parser.add_argument("--sentence_seps", type=str, default=".?!\n",
                       help="(Deprecated) Not used anymore")
    
    # Output
    parser.add_argument("--output_path", type=str, default="./forget_token_mask.pt",
                       help="Path to save the forget token mask")
    parser.add_argument("--si_output_path", type=str, default=None,
                       help="Path to save per-token SI scores tensor (.pt). Default: same dir as output_path with name si_scores.pt")
    parser.add_argument("--inspection_idx", type=int, default=0,
                       help="Index of sample to inspect and visualize")
    parser.add_argument("--verbose", action="store_true", default=False,
                       help="Enable verbose logging")

    # Reporting (optional): build token/word/name top lists with template exclusion
    parser.add_argument("--emit_token_report", action="store_true", default=False,
                       help="Emit top tokens/words/names report from selected (forget) tokens")
    parser.add_argument("--template_exclude", type=str, default="Question,Answer,:,Ċ",
                       help="Comma-separated token strings to exclude from token-level report")
    parser.add_argument("--report_top_k", type=int, default=20, help="Top-K items to list in reports")
    parser.add_argument("--report_output", type=str, default=None,
                       help="Optional path to save JSON report (stdout if None)")
    parser.add_argument("--report_count_context", action="store_true", default=False,
                       help="Count tokens/words within the context window around selected tokens in the report")
    
    return parser.parse_args()


def build_forget_token_mask(
    influence_scores: torch.Tensor,  # shape (num_samples, seq_len)
    top_n_per_sample: int = 1,
    context_window: int = 3,
    min_score: float = 0.0,
    *,
    tokenizer=None,
    dataset=None,
    word_masking: bool = False,
    model_family: str = "pythia-1.4",
) -> torch.Tensor:
    """
    Build a forget token mask using per-sample top-k selection from answer portions.

    Args:
        influence_scores: Tensor of shape (num_samples, seq_len)
        top_n_per_sample: Number of top tokens to select per sample
        context_window: Window around high-influence tokens to include
        min_score: Minimum score threshold for token selection
        tokenizer: Tokenizer for processing
        dataset: Dataset to access labels
        word_masking: Whether to expand selection to full words
        model_family: Model family to determine split symbol

    Returns:
        mask: Boolean tensor same shape as influence_scores
    """
    scores = influence_scores.detach().float()
    B, T = scores.shape

    # Determine split symbol based on model family
    split_symbol = " [/INST]" if model_family == 'llama2-7b' else 'Answer: '

    # Step 1: Create answer mask using labels (where labels != -100)
    answer_mask = torch.zeros_like(scores, dtype=torch.bool)

    logging.info(f"Identifying answer portions using labels from dataset")
    for b in range(B):
        # Get labels from dataset
        item = dataset[b]
        if isinstance(item, (tuple, list)) and len(item) >= 2:
            # Format: (input_ids, labels, attention_mask)
            labels = item[1]
        elif isinstance(item, dict) and "labels" in item:
            labels = item["labels"]
        else:
            # Fallback: decode and find split pattern
            if isinstance(item, (tuple, list)):
                input_ids = item[0]
            else:
                input_ids = item.get("input_ids", torch.zeros(T))

            if torch.is_tensor(input_ids):
                input_ids_list = input_ids.detach().cpu().tolist()
            else:
                input_ids_list = list(input_ids)

            # Decode and find split
            text = tokenizer.decode(input_ids_list, skip_special_tokens=True)
            if split_symbol in text:
                answer_start_char = text.find(split_symbol) + len(split_symbol)
                # Rough estimate: answer starts after split_symbol
                answer_start = int(answer_start_char * len(input_ids_list) / len(text))
            else:
                answer_start = 0

            # Mark from answer_start to end
            answer_length = min(T, len(input_ids_list)) - answer_start
            if answer_length > 0:
                answer_mask[b, answer_start:answer_start+answer_length] = True
            continue

        if torch.is_tensor(labels):
            labels_tensor = labels.detach().cpu()
        else:
            labels_tensor = torch.tensor(labels)

        # Answer tokens are where labels != -100
        L = min(T, labels_tensor.shape[0])
        answer_mask[b, :L] = (labels_tensor[:L] != -100)

    # Step 2: Apply min_score filter and extract answer scores
    answer_scores_masked = torch.where(
        answer_mask,
        scores,
        torch.full_like(scores, float('-inf'))
    )

    if min_score > 0:
        answer_scores_masked = torch.where(
            answer_scores_masked >= min_score,
            answer_scores_masked,
            torch.full_like(scores, float('-inf'))
        )

    # Step 3: Per-sample top-k selection
    forget_mask = torch.zeros_like(scores, dtype=torch.bool)
    total_selected = 0
    samples_with_selection = 0

    for b in range(B):
        # Get scores for this sample's answer portion
        sample_answer_scores = answer_scores_masked[b]
        valid_mask = sample_answer_scores > float('-inf')
        valid_count = valid_mask.sum().item()

        if valid_count > 0:
            # Select top-k tokens for this sample
            k = min(top_n_per_sample, valid_count)
            topk_values, topk_indices = torch.topk(sample_answer_scores, k=k, largest=True)

            # Mark selected tokens
            for idx in topk_indices:
                if sample_answer_scores[idx] > float('-inf'):
                    forget_mask[b, idx] = True
                    total_selected += 1

            samples_with_selection += 1

    logging.info(f"Per-sample top-{top_n_per_sample} selection: selected {total_selected} tokens from {samples_with_selection}/{B} samples")

    # Step 5: Apply context window and word masking expansion
    if context_window > 0 or word_masking:
        logging.info(f"Applying expansions (context_window={context_window}, word_masking={word_masking})")
        expanded_mask = forget_mask.clone()

        for b in range(B):
            # Get token-to-word mapping if needed
            token_to_word = None
            if word_masking:
                item = dataset[b]
                if isinstance(item, dict) and "input_ids" in item:
                    input_ids = item["input_ids"]
                elif isinstance(item, (tuple, list)) and len(item) >= 1:
                    input_ids = item[0]
                else:
                    input_ids = None

                if input_ids is not None:
                    input_ids_tensor = input_ids.detach().cpu() if torch.is_tensor(input_ids) else torch.tensor(input_ids)
                    L = min(T, int(input_ids_tensor.shape[0]))
                    tokens = tokenizer.convert_ids_to_tokens(input_ids_tensor.tolist()[:L])
                    _, token_to_word = _map_token_index_to_word(tokens)

            # Find selected positions for this sample
            selected_positions = forget_mask[b].nonzero(as_tuple=True)[0].tolist()

            for center in selected_positions:
                expansion_positions = set()

                # Word expansion
                if word_masking and token_to_word is not None:
                    wi = token_to_word.get(center)
                    if wi is not None:
                        for p in range(T):
                            if token_to_word.get(p) == wi:
                                expansion_positions.add(p)

                # Context window expansion
                if context_window > 0:
                    start = max(0, center - context_window)
                    end = min(T, center + context_window + 1)
                    for p in range(start, end):
                        expansion_positions.add(p)

                # Apply expansions
                for p in expansion_positions:
                    expanded_mask[b, p] = True

        forget_mask = expanded_mask

    total_forget_tokens = forget_mask.sum().item()
    affected_samples = (forget_mask.sum(dim=1) > 0).sum().item()

    logging.info(f"Selected {total_forget_tokens} tokens across {affected_samples} samples")
    return forget_mask


def _map_token_index_to_word(tokens: List[str]) -> Tuple[List[str], Dict[int, int]]:
    """
    Robust token->word mapping for GPT-2 BPE (Ġ), SentencePiece (▁), and plain BPE pieces.
    Returns (words, token_index_to_word_index).
    - 공백 접두 자동 감지: GPT-2(Ġ), SentencePiece(▁)
    - 줄바꿈/스페셜 토큰은 단어 경계로 처리
    - 구두점은 단독 토큰일 때 단어 경계로 처리
    - 하이픈('-')은 단어 내부 연결자로 유지 (기존 동작 보존)
    """
    if not tokens:
        return [], {}

    # 자동 감지: 공백 접두
    has_gpt2_space = any(t.startswith('Ġ') for t in tokens)
    has_sp_space = any(t.startswith('▁') for t in tokens)

    # 구두점/스페셜
    # 하이픈('-')은 단어 내부 연결자로 사용하므로 제외
    punct = set(list(".,!?;:()[]{}\"'`…—–"))
    # Include LLaMA instruction tokens as special tokens
    special_tokens = {'<|endoftext|>', '<s>', '</s>', '<unk>', '<pad>', '[INST]', '[/INST]', '<<SYS>>', '<</SYS>>'}

    # LLaMA 특수 패턴을 위한 버퍼
    llama_buffer = []

    words: List[str] = []
    t2w: Dict[int, int] = {}

    current: List[str] = []

    def flush():
        nonlocal current
        if current:
            words.append(''.join(current))
            current = []

    for i, tok in enumerate(tokens):
        # 줄바꿈/스페셜 토큰은 단어 경계로 취급
        if tok in special_tokens or tok == 'Ċ':
            flush()
            continue

        # 공백 접두 처리 및 정리
        is_space = False
        if has_gpt2_space and tok.startswith('Ġ'):
            tok_clean = tok[1:]
            is_space = True
        elif has_sp_space and tok.startswith('▁'):
            tok_clean = tok[1:]
            is_space = True
        else:
            tok_clean = tok

        # LLaMA 특수 토큰 부분들은 연결하여 처리
        if tok_clean in ['[', ']', 'INST', '/', 'SYS']:
            if not current:
                t2w[i] = len(words)
            else:
                t2w[i] = len(words)
            current.append(tok_clean)
            continue
            
        # 단독 구두점 토큰이면 단어 경계로 처리
        if tok_clean in punct and len(tok_clean) == 1:
            flush()
            # 구두점 자체에 단어 인덱스를 부여하지 않음 (확장 목적 아님)
            continue

        # 공백 마커가 있으면 이전 단어 플러시 후 새 단어 시작
        if is_space:
            flush()

        # 하이픈은 단어 내부 연결자로 유지
        if tok_clean == '-':
            if not current:
                # 단어 시작이 하이픈이면 하이픈으로 시작
                t2w[i] = len(words)
                current.append('-')
            else:
                t2w[i] = len(words)
                current.append('-')
            continue

        # 조각을 현재 단어에 추가 (없으면 새로 시작)
        if not current:
            t2w[i] = len(words)
            current.append(tok_clean)
        else:
            t2w[i] = len(words)
            current.append(tok_clean)

    flush()
    # 공백만 있는 조합 제거
    words = [w for w in words if w.strip() != ""]

    return words, t2w


def _save_selected_tokens_si_scores(
    tokenizer,
    dataset,
    forget_mask: torch.Tensor,
    influence_scores: torch.Tensor,
    output_path: str,
):
    """선택된 토큰들의 SI 점수만을 저장하는 함수"""
    num_samples, seq_len = forget_mask.shape
    selected_tokens_si = []

    # 특수 토큰 패턴 정의
    special_token_patterns = ['[INST]', '[/INST]', '<<SYS>>', '<</SYS>>']

    for sample_index in range(num_samples):
        item = dataset[sample_index]
        if isinstance(item, dict) and "input_ids" in item:
            input_ids = item["input_ids"]
        elif isinstance(item, (tuple, list)) and len(item) >= 1:
            input_ids = item[0]
        else:
            continue

        if torch.is_tensor(input_ids):
            input_ids_tensor = input_ids.detach().cpu()
        else:
            input_ids_tensor = torch.tensor(input_ids)

        scores_row = influence_scores[sample_index].detach().cpu()
        mask_row = forget_mask[sample_index].detach().cpu()
        L = min(int(scores_row.shape[0]), int(input_ids_tensor.shape[0]), int(mask_row.shape[0]))
        if L <= 0:
            continue

        input_ids_tensor = input_ids_tensor[:L]
        scores_row = scores_row[:L]
        mask_row = mask_row[:L]

        tokens = tokenizer.convert_ids_to_tokens(input_ids_tensor.tolist())
        words, token_to_word = _map_token_index_to_word(tokens)

        # 전체 컨텍스트 생성 (토크나이저의 decode 사용)
        full_context = tokenizer.decode(input_ids_tensor[:L].tolist(), skip_special_tokens=True)

        # 선택된 토큰들의 정보 수집
        selected_positions = [i for i, flag in enumerate(mask_row.tolist()) if flag]

        for pos in selected_positions:
            tok = tokens[pos]
            score_val = float(scores_row[pos]) if pos < len(scores_row) else 0.0

            # Only include if SI score is positive
            if score_val <= 0:
                continue

            # 단어 정보
            wi = token_to_word.get(pos)
            word = words[wi] if (wi is not None and 0 <= wi < len(words)) else tok

            # 특수 토큰 패턴이 포함된 단어는 건너뛰기
            if any(pattern in word for pattern in special_token_patterns):
                continue

            token_si_info = {
                "sample_index": sample_index,
                "token_index": pos,
                "word_index": wi if wi is not None else -1,
                "token": tok,
                "word": word,
                "si_score": score_val,
                "full_context": full_context
            }
            selected_tokens_si.append(token_si_info)
    
    # SI 점수별로 정렬
    selected_tokens_si.sort(key=lambda x: x["si_score"], reverse=True)
    
    # 저장 경로 설정
    si_scores_path = output_path.replace('.pt', '_selected_tokens_si.json')
    
    # JSON 파일로 저장
    import json
    out_dir = os.path.dirname(os.path.abspath(si_scores_path))
    os.makedirs(out_dir, exist_ok=True)
    
    with open(si_scores_path, 'w', encoding='utf-8') as f:
        json.dump(selected_tokens_si, f, ensure_ascii=False, indent=2)
    
    logging.info(f"💾 Saved selected tokens SI scores to: {si_scores_path}")
    logging.info(f"📊 Total selected tokens: {len(selected_tokens_si)}")


def _save_all_tokens_si_scores(
    tokenizer,
    dataset,
    influence_scores: torch.Tensor,
    output_path: str,
):
    """모든 토큰의 SI 점수를 저장하는 함수"""
    num_samples, seq_len = influence_scores.shape
    all_tokens_si = []

    # 특수 토큰 패턴 정의
    special_token_patterns = ['[INST]', '[/INST]', '<<SYS>>', '<</SYS>>']

    logging.info(f"📊 Processing all tokens SI scores for {num_samples} samples...")
    pbar = tqdm(total=num_samples, desc="Processing all tokens", unit="samples")

    for sample_index in range(num_samples):
        item = dataset[sample_index]
        if isinstance(item, dict) and "input_ids" in item:
            input_ids = item["input_ids"]
        elif isinstance(item, (tuple, list)) and len(item) >= 1:
            input_ids = item[0]
        else:
            pbar.update(1)
            continue

        if torch.is_tensor(input_ids):
            input_ids_tensor = input_ids.detach().cpu()
        else:
            input_ids_tensor = torch.tensor(input_ids)

        scores_row = influence_scores[sample_index].detach().cpu()
        L = min(int(scores_row.shape[0]), int(input_ids_tensor.shape[0]))
        if L <= 0:
            pbar.update(1)
            continue

        input_ids_tensor = input_ids_tensor[:L]
        scores_row = scores_row[:L]

        tokens = tokenizer.convert_ids_to_tokens(input_ids_tensor.tolist())
        words, token_to_word = _map_token_index_to_word(tokens)

        # 전체 컨텍스트 생성 (토크나이저의 decode 사용)
        full_context = tokenizer.decode(input_ids_tensor[:L].tolist(), skip_special_tokens=True)

        # 모든 토큰의 정보 수집
        sample_tokens_info = {
            "sample_index": sample_index,
            "full_context": full_context,
            "tokens": []
        }

        for pos in range(L):
            tok = tokens[pos]
            score_val = float(scores_row[pos]) if pos < len(scores_row) else 0.0

            # 단어 정보
            wi = token_to_word.get(pos)
            word = words[wi] if (wi is not None and 0 <= wi < len(words)) else tok

            # 특수 토큰 패턴이 포함된 단어는 건너뛰기
            if any(pattern in word for pattern in special_token_patterns):
                continue

            token_info = {
                "token_index": pos,
                "word_index": wi if wi is not None else -1,
                "token": tok,
                "word": word,
                "si_score": score_val
            }
            sample_tokens_info["tokens"].append(token_info)

        all_tokens_si.append(sample_tokens_info)
        pbar.update(1)
    
    pbar.close()
    
    # 저장 경로 설정
    all_tokens_si_path = output_path.replace('.pt', '_all_tokens_si.json')
    
    # JSON 파일로 저장
    import json
    out_dir = os.path.dirname(os.path.abspath(all_tokens_si_path))
    os.makedirs(out_dir, exist_ok=True)
    
    with open(all_tokens_si_path, 'w', encoding='utf-8') as f:
        json.dump(all_tokens_si, f, ensure_ascii=False, indent=2)
    
    logging.info(f"💾 Saved all tokens SI scores to: {all_tokens_si_path}")
    logging.info(f"📊 Total samples processed: {len(all_tokens_si)}")
    
    # 통계 정보 출력
    total_tokens = sum(len(sample["tokens"]) for sample in all_tokens_si)
    avg_tokens_per_sample = total_tokens / len(all_tokens_si) if all_tokens_si else 0
    logging.info(f"📊 Total tokens: {total_tokens:,}")
    logging.info(f"📊 Average tokens per sample: {avg_tokens_per_sample:.1f}")


def load_influence_scores(scores_path: str) -> torch.Tensor:
    """Load influence scores from safetensors file"""
    from kronfluence.utils.save import load_file
    from pathlib import Path

    # If directory path, append pairwise_scores.safetensors
    if os.path.isdir(scores_path):
        scores_path = os.path.join(scores_path, "pairwise_scores.safetensors")

    if not os.path.exists(scores_path):
        raise FileNotFoundError(f"Score file not found: {scores_path}")

    # Load safetensors file
    scores_data = load_file(Path(scores_path))

    # Extract 'all_modules' tensor
    if 'all_modules' in scores_data:
        scores = scores_data['all_modules']
    else:
        raise KeyError(f"'all_modules' key not found in {scores_path}")

    logging.info(f"Loaded influence scores: shape={scores.shape}, dtype={scores.dtype}")
    return scores


def main():
    args = parse_args()
    
    # Setup logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='[%(asctime)s] [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    logging.info("🔍 Loading differential influence scores for forget token selection...")
    
    # Load model config for tokenizer
    # Resolve path: prefer CLI arg; otherwise use script-relative ../config/model_config.yaml
    config_path = args.model_config_path
    if config_path is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.abspath(os.path.join(script_dir, "..", "config", "model_config.yaml"))
    logging.info(f"Loading model config from: {config_path}")
    with open(config_path, 'r') as f:
        model_configs = yaml.safe_load(f)
    
    if args.model_family in model_configs:
        hf_key = model_configs[args.model_family]["hf_key"]
        logging.info(f"Loading tokenizer from: {hf_key}")
    else:
        raise ValueError(f"Model family '{args.model_family}' not found in config")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(hf_key)
    tokenizer.pad_token = tokenizer.eos_token

    # Load forget dataset using TextDatasetQA (simple and clean!)
    logging.info(f"Loading forget dataset from: {args.tofu_data_path}")
    forget_dataset = TextDatasetQA(
        data_path=args.tofu_data_path,
        tokenizer=tokenizer,
        model_family=args.model_family,
        max_length=args.max_length,
        split=args.forget_split,  # forget10, forget90, etc.
        question_key='question',
        answer_key='answer'
    )
    
    # Load influence scores
    logging.info(f"Loading scores from: {args.scores_path}")
    influence_scores = load_influence_scores(args.scores_path)

    num_samples, seq_len = influence_scores.shape
    logging.info(f"Processing {num_samples} samples with sequence length {seq_len}")

    # Build forget token mask using per-sample top-k selection from answer portions
    forget_mask = build_forget_token_mask(
        influence_scores=influence_scores,
        top_n_per_sample=args.top_n_per_sample,
        context_window=args.context_window,
        min_score=args.min_score,
        tokenizer=tokenizer,
        dataset=forget_dataset,
        word_masking=args.word_masking,
        model_family=args.model_family,
    )
    
    # Save the mask
    out_dir = os.path.dirname(os.path.abspath(args.output_path))
    os.makedirs(out_dir, exist_ok=True)
    torch.save(forget_mask, args.output_path)
    logging.info(f"💾 Saved forget token mask to: {args.output_path}")

    # Save per-token SI scores (float tensor)
    si_out_path = args.si_output_path
    if si_out_path is None:
        si_out_path = os.path.join(out_dir, "si_scores.pt")
    torch.save(influence_scores.detach().float(), si_out_path)
    logging.info(f"💾 Saved per-token SI scores to: {si_out_path}")
    
    # Simple statistics
    total_forget_tokens = forget_mask.sum().item()
    affected_samples = (forget_mask.sum(dim=1) > 0).sum().item()
    logging.info(f"✅ Selected {total_forget_tokens:,} tokens across {affected_samples:,} samples")

    # 추가: 선택된 토큰들의 SI 점수만 저장하는 기능
    _save_selected_tokens_si_scores(
        tokenizer=tokenizer,
        dataset=forget_dataset,
        forget_mask=forget_mask,
        influence_scores=influence_scores,
        output_path=args.output_path
    )
    
    # Generate JSON files with SI scores and context  
    _generate_json_files(tokenizer, forget_dataset, forget_mask, influence_scores, args)


if __name__ == "__main__":
    main()