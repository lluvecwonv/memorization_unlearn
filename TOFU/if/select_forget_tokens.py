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

# Local imports - using local data loading functionality
import sys
import os
sys.path.append('/root/Unlearn-Simple/TOFU')
sys.path.append('/root/Unlearn-Simple/TOFU/if/analysis')
import datasets
from data_module import TextForgetDatasetQA

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

def load_datasets(config, tokenizer, model_family, max_length):
    """Load datasets function to replace the missing local_utils import"""
    try:
        data_path = getattr(config, 'tofu_data_path', 'locuslab/TOFU')
        forget_split = getattr(config, 'forget_split', 'forget10')
        
        # Load forget dataset
        if './TOFU_data' not in data_path:
            forget_data = datasets.load_dataset(data_path, forget_split)["train"]
        else:
            forget_data = datasets.load_dataset('json', data_files=os.path.join(data_path, forget_split+'.json'))['train']
        
        # Load retain dataset
        retain_split = "retain" + str(100 - int(forget_split.replace("forget", ""))).zfill(2)
        if './TOFU_data' not in data_path:
            retain_data = datasets.load_dataset(data_path, retain_split)["train"]
        else:
            retain_data = datasets.load_dataset('json', data_files=os.path.join(data_path, retain_split+'.json'))['train']
        
        return forget_data, retain_data
    except Exception as e:
        print(f"Error loading datasets: {e}")
        return None, None

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
    parser.add_argument("--forget_threshold", type=float, default=0.95,
                       help="Threshold percentile for selecting high-influence tokens (0-1)")
    parser.add_argument("--context_window", type=int, default=3,
                       help="Context window around high-influence tokens")
    parser.add_argument("--max_forget_tokens", type=int, default=1000000,
                       help="Maximum number of tokens to mark for forgetting")
    parser.add_argument("--top_k_samples", type=int, default=None,
                       help="Only consider top K samples by influence (None = all)")
    parser.add_argument("--threshold_mode", type=str, default="global_percentile",
                       choices=["global_percentile", "positive_percentile", "per_sample_percentile", "topk_per_sample"],
                       help="How to determine high-influence tokens")
    parser.add_argument("--top_n_per_sample", type=int, default=1,
                       help="Number of top tokens per sample when using topk_per_sample mode")
    parser.add_argument("--min_score", type=float, default=0.0,
                       help="Minimum score cutoff; tokens must exceed this to be considered")
    parser.add_argument("--select_mode", type=str, default="topk_per_sample",
                       choices=["topk_per_sample", "topk_per_sentence"],
                       help="Selection granularity: per sample or per sentence")
    parser.add_argument("--word_masking", action="store_true", default=False,
                       help="Expand selected token to cover the full reconstructed word (BPE pieces)")
    parser.add_argument("--sentence_seps", type=str, default=".?!\n",
                       help="Characters treated as sentence terminators when select_mode=topk_per_sentence")
    
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


def color_strength(word: str, strength: float) -> None:
    """Colorize word based on forgetting strength (red = high influence)"""
    strength = max(0.0, min(1.0, strength))
    intensity = int(strength * 255)
    color = f"\033[38;2;{intensity};0;0m"
    print(f"{color}{word}{Fore.RESET}", end="")


def min_max_normalize(tensor: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Normalize tensor to [0, 1] range"""
    min_val = tensor.min()
    max_val = tensor.max()
    return (tensor - min_val) / (max_val - min_val + eps)


def harmonic_mean(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Compute harmonic mean of two tensors"""
    a = a.float() + eps
    b = b.float() + eps
    return 2 * a * b / (a + b)


def build_forget_token_mask(
    influence_scores: torch.Tensor,  # shape (num_samples, seq_len)
    forget_threshold: float = 0.95,
    context_window: int = 3,
    max_forget_tokens: int = 1000000,
    top_k_samples: int = None,
    threshold_mode: str = "global_percentile",
    top_n_per_sample: int = 1,
    min_score: float = 0.0,
    *,
    select_mode: str = "topk_per_sample",
    tokenizer=None,
    dataset=None,
    word_masking: bool = False,
    sentence_seps: str = ".?!\n",
) -> tuple[torch.Tensor, list]:
    """
    Build a forget token mask where True = token to forget, False = keep token.
    
    Args:
        influence_scores: Tensor of shape (num_samples, seq_len)
        forget_threshold: Percentile threshold for high-influence tokens
        context_window: Window around high-influence tokens to include
        max_forget_tokens: Maximum number of tokens to mark for forgetting
        top_k_samples: Only consider top K samples (None = all samples)
    
    Returns:
        mask: Boolean tensor same shape as influence_scores
        sample_ranking: List of sample indices sorted by forget priority
    """
    scores = influence_scores.detach().float()
    B, T = scores.shape
    
    # Select top-k tokens per sample
    above_thresh_mask = torch.zeros_like(scores)
    k = max(1, int(top_n_per_sample))
    
    for b in range(B):
        # Apply minimum score filter
        masked = scores[b].clone()
        if min_score > 0:
            masked = torch.where(masked >= min_score, masked, torch.full_like(masked, float('-inf')))
        
        # Count how many tokens are above min_score
        valid_tokens = (masked > float('-inf')).sum().item()
        actual_k = min(k, valid_tokens)  # Don't select more than available valid tokens
        
        if actual_k > 0:
            # Get top-k tokens (only from valid ones)
            topk = torch.topk(masked, k=actual_k, dim=0)
            above = torch.zeros_like(scores[b])
            above[topk.indices] = 1.0
            above_thresh_mask[b] = above
    
    logging.info(f"Top-k per-sample mode: k={k}, min_score={min_score}")
    count_above = above_thresh_mask.sum(dim=1)  # Number of high-influence tokens per sample
    sum_above = (scores * above_thresh_mask).sum(dim=1)  # Sum of high-influence values per sample
    
    # Normalize metrics
    count_above_norm = min_max_normalize(count_above)
    sum_above_norm = min_max_normalize(sum_above)
    
    # Compute harmonic mean for balanced ranking
    harmonic_scores = harmonic_mean(count_above_norm, sum_above_norm)
    
    # Sort samples by harmonic score (highest first)
    sample_ranking = torch.argsort(harmonic_scores, descending=True).tolist()
    
    # Limit to top-k samples if specified
    if top_k_samples is not None:
        sample_ranking = sample_ranking[:top_k_samples]
        logging.info(f"Limiting to top {top_k_samples} samples")
    
    # Build forget mask
    forget_mask = torch.zeros_like(scores, dtype=torch.bool)
    already_selected = torch.zeros_like(scores, dtype=torch.bool)
    total_forget_tokens = 0
    
    logging.info(f"Building forget mask for {len(sample_ranking)} samples (context_window={context_window}, select_mode={select_mode}, word_masking={word_masking})...")
    pbar = tqdm(total=len(sample_ranking), desc="Processing samples", unit="samples")
    
    for sample_idx in sample_ranking:
        # Default: use preselected positions from above_thresh_mask (per-sample top-k)
        selected_positions: list[int] = []

        if select_mode == "topk_per_sentence" and tokenizer is not None and dataset is not None:
            # Acquire tokens for this sample
            item = dataset[sample_idx]
            if isinstance(item, dict) and "input_ids" in item:
                input_ids = item["input_ids"]
            elif isinstance(item, (tuple, list)) and len(item) >= 1:
                input_ids = item[0]
            else:
                input_ids = None
            if input_ids is not None:
                input_ids_tensor = input_ids.detach().cpu() if torch.is_tensor(input_ids) else torch.tensor(input_ids)
                L = min(int(scores.shape[1]), int(input_ids_tensor.shape[0]))
                # Align to available length
                positions = list(range(L))
                tokens = tokenizer.convert_ids_to_tokens(input_ids_tensor.tolist()[:L])
                # Map tokens to sentence ids
                term_set = set(list(sentence_seps))
                sent_id = 0
                token_sent_ids = []
                for t in tokens:
                    token_sent_ids.append(sent_id)
                    # Sentence boundary if token is newline or a single terminator token
                    if t == 'Ċ' or (len(t) == 1 and t in term_set):
                        sent_id += 1
                # Group positions by sentence
                from collections import defaultdict
                sent_to_pos = defaultdict(list)
                for i in positions:
                    sent_to_pos[token_sent_ids[i]].append(i)
                # Select top-k within each sentence
                k = max(1, int(top_n_per_sample))
                row_scores = scores[sample_idx].detach().cpu()
                for _, pos_list in sent_to_pos.items():
                    if not pos_list:
                        continue
                    # Apply min_score within sentence
                    candidate_scores = torch.tensor([row_scores[i].item() for i in pos_list])
                    if min_score > 0:
                        mask_valid = candidate_scores >= min_score
                    else:
                        mask_valid = torch.ones_like(candidate_scores, dtype=torch.bool)
                    valid_indices = [pos_list[i] for i, v in enumerate(mask_valid.tolist()) if v]
                    if not valid_indices:
                        continue
                    # Determine actual k
                    actual_k = min(k, len(valid_indices))
                    # Top-k over valid indices
                    cand_vals = torch.tensor([row_scores[i].item() for i in valid_indices])
                    topk = torch.topk(cand_vals, k=actual_k, dim=0)
                    for j in topk.indices.tolist():
                        selected_positions.append(valid_indices[j])
            else:
                # Fallback to per-sample
                high_influence_positions = above_thresh_mask[sample_idx].nonzero(as_tuple=True)[0]
                selected_positions = [p.item() for p in high_influence_positions]
        else:
            # Per-sample default selection
            high_influence_positions = above_thresh_mask[sample_idx].nonzero(as_tuple=True)[0]
            selected_positions = [p.item() for p in high_influence_positions]

        # Optional word masking uses reconstructed words
        words = None
        token_to_word = None
        if word_masking and tokenizer is not None and dataset is not None:
            item = dataset[sample_idx]
            if isinstance(item, dict) and "input_ids" in item:
                input_ids = item["input_ids"]
            elif isinstance(item, (tuple, list)) and len(item) >= 1:
                input_ids = item[0]
            else:
                input_ids = None
            if input_ids is not None:
                input_ids_tensor = input_ids.detach().cpu() if torch.is_tensor(input_ids) else torch.tensor(input_ids)
                L = min(int(scores.shape[1]), int(input_ids_tensor.shape[0]))
                tokens = tokenizer.convert_ids_to_tokens(input_ids_tensor.tolist()[:L])
                words, token_to_word = _map_token_index_to_word(tokens)

        # Apply context window (conditional) and word expansion
        for center in selected_positions:
            # Build expansion set for this center
            expansion_positions = set()
            # Word expansion first (always, if enabled)
            if word_masking and token_to_word is not None:
                wi = token_to_word.get(center)
                if wi is not None:
                    for p in range(T):
                        if token_to_word.get(p) == wi:
                            expansion_positions.add(p)
            # Context expansion only if SI(center) > 0
            si_center = float(scores[sample_idx, center].item()) if 0 <= center < scores.shape[1] else 0.0
            if si_center > 0 and context_window and context_window > 0:
                start = max(0, center - int(max(0, context_window)))
                end = min(T, center + int(max(0, context_window)) + 1)
                for p in range(start, end):
                    expansion_positions.add(p)
            # If neither added, at least include the center
            if not expansion_positions:
                expansion_positions.add(center)

            # Mark
            for p in sorted(expansion_positions):
                if total_forget_tokens >= max_forget_tokens:
                    break
                if already_selected[sample_idx, p]:
                    continue
                forget_mask[sample_idx, p] = True
                already_selected[sample_idx, p] = True
                total_forget_tokens += 1
            if total_forget_tokens >= max_forget_tokens:
                break
        
        # Update progress bar per sample
        pbar.update(1)
        if total_forget_tokens >= max_forget_tokens:
            break
    
    pbar.close()
    logging.info(f"Selected {total_forget_tokens} tokens for forgetting across {len(sample_ranking)} samples")
    return forget_mask, sample_ranking


def _detokenize_window(tokens: list[str]) -> str:
    output_fragments: list[str] = []
    for i, token in enumerate(tokens):
        # 특수 토큰들 처리
        if token in ['<|endoftext|>', '<s>', '</s>', '<unk>', '<pad>']:
            # <|endoftext|>는 문장 끝을 의미하므로 여기서 중단
            if token == '<|endoftext|>':
                break
            continue
        elif token == 'Ċ':
            output_fragments.append('\n')
        elif token.startswith('Ġ'):
            # GPT-2 스타일 공백
            output_fragments.append(' ' + token[1:])
        elif token.startswith('▁'):
            # SentencePiece 스타일 공백 (LLaMA, T5 등)
            if i == 0:
                # 문장 시작의 ▁는 제거
                output_fragments.append(token[1:])
            else:
                # 이후의 ▁는 공백으로 변환
                output_fragments.append(' ' + token[1:])
        elif token == '-':
            output_fragments.append('-')
        else:
            output_fragments.append(token)
    return ''.join(output_fragments).strip()


def _map_token_index_to_word(tokens: list[str]) -> tuple[list[str], dict[int, int]]:
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
    special_tokens = {'<|endoftext|>', '<s>', '</s>', '<unk>', '<pad>'}
    
    # LLaMA 특수 패턴을 위한 버퍼
    llama_buffer = []

    words: list[str] = []
    t2w: dict[int, int] = {}

    current: list[str] = []

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


def _emit_token_report(
    tokenizer,
    dataset,
    forget_mask: torch.Tensor,
    influence_scores: torch.Tensor,
    top_k: int,
    context_window: int,
    template_exclude: set[str],
    report_output: str | None,
    count_context: bool = False,
):
    """Aggregate token/word/name stats from forget_mask and print/save report.

    - Token Top-K: frequency among forget tokens, excluding template_exclude
    - Word Top-K: recomposed words at forget positions
    - Name Top-K: capitalized/hyphenated words, with short context examples
    - Selected tokens: detailed info about each selected token with full context
    """
    num_samples, seq_len = forget_mask.shape
    token_counter: Counter[str] = Counter()
    token_weight_sum: Counter[str] = Counter()
    word_counter: Counter[str] = Counter()
    word_weight_sum: Counter[str] = Counter()
    name_counter: Counter[str] = Counter()
    name_weight_sum: Counter[str] = Counter()
    name_contexts: dict[str, list[str]] = defaultdict(list)
    
    # 새로운 데이터 구조: 선택된 토큰들의 상세 정보
    selected_tokens_details = []

    name_pattern = re.compile(r'^[A-Z][a-z]+(?:-[A-Z][a-z]+)*$')

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

        # positions to count: center only or context window
        selected_positions = [i for i, flag in enumerate(mask_row.tolist()) if flag]
        if count_context and context_window > 0:
            expanded = set()
            for p0 in selected_positions:
                s = max(0, p0 - context_window)
                e = min(len(tokens), p0 + context_window + 1)
                expanded.update(range(s, e))
            positions_to_count = sorted(expanded)
        else:
            positions_to_count = selected_positions

        for pos in positions_to_count:
            tok = tokens[pos]
            score_val = float(scores_row[pos]) if pos < len(scores_row) else 0.0

            # 선택된 토큰의 상세 정보 수집
            if pos in selected_positions:  # 실제로 선택된 토큰만
                # 단어 정보
                wi = token_to_word.get(pos)
                word = words[wi] if (wi is not None and 0 <= wi < len(words)) else tok
                
                # 컨텍스트 윈도우 생성 (토크나이저의 decode 사용)
                s = max(0, pos - context_window)
                e = min(len(tokens), pos + context_window + 1)
                context_ids = input_ids_tensor[s:e].tolist()
                context_text = tokenizer.decode(context_ids, skip_special_tokens=True)
                
                # 선택된 토큰의 상세 정보
                token_detail = {
                    "sample_index": sample_index,
                    "token_index": pos,
                    "word_index": wi if wi is not None else -1,
                    "token": tok,
                    "word": word,
                    "si_score": score_val,
                    "context": context_text,
                    "full_context": full_context,
                    "expanded": pos in selected_positions
                }
                selected_tokens_details.append(token_detail)

            # Token-level, with template exclusion
            if tok not in template_exclude:
                token_counter[tok] += 1
                token_weight_sum[tok] += score_val

            # Word-level
            wi = token_to_word.get(pos)
            if wi is None or wi < 0 or wi >= len(words):
                continue
            w = words[wi]
            if not w:
                continue
            word_counter[w] += 1
            word_weight_sum[w] += score_val

            # Name-level (capitalized and optional hyphen join)
            if name_pattern.match(w):
                name_counter[w] += 1
                name_weight_sum[w] += score_val
                # context examples use center positions only to avoid duplication
                if pos in selected_positions and len(name_contexts[w]) < 3:
                    s = max(0, pos - context_window)
                    e = min(len(tokens), pos + context_window + 1)
                    ctx_ids = input_ids_tensor[s:e].tolist()
                    ctx = tokenizer.decode(ctx_ids, skip_special_tokens=True)
                    name_contexts[w].append(ctx)

    def _top_list(counter: Counter[str], wsum: Counter[str], k: int) -> list[tuple[str, int, float]]:
        items: list[tuple[str, int, float]] = []
        for token, count in counter.most_common(k):
            avg = float(wsum[token]) / count if count else 0.0
            items.append((token, count, round(avg, 6)))
        return items

    top_tokens = _top_list(token_counter, token_weight_sum, top_k)
    top_words = _top_list(word_counter, word_weight_sum, top_k)
    top_names = _top_list(name_counter, name_weight_sum, top_k)

    # SI 점수별로 정렬된 선택된 토큰들
    selected_tokens_details.sort(key=lambda x: x["si_score"], reverse=True)

    report = {
        "top_tokens_filtered": top_tokens,
        "top_words": top_words,
        "top_names": top_names,
        "name_context_examples": {k: v for k, v in name_contexts.items() if k in dict(top_names)},
        "selected_tokens_details": selected_tokens_details[:top_k * 10],  # 상위 토큰들만 저장
        "summary": {
            "total_selected_tokens": len(selected_tokens_details),
            "total_samples": num_samples,
            "context_window": context_window
        }
    }

    if report_output:
        import json
        out_dir = os.path.dirname(os.path.abspath(report_output))
        os.makedirs(out_dir, exist_ok=True)
        with open(report_output, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        logging.info(f"💾 Saved token report to: {report_output}")
    else:
        logging.info(f"Top tokens (filtered): {top_tokens}")
        logging.info(f"Top words: {top_words}")
        logging.info(f"Top names: {top_names}")
        logging.info(f"Selected tokens details: {len(selected_tokens_details)} tokens")
        # Print up to 5 name contexts
        shown = 0
        for name, _c, _a in top_names:
            if shown >= 5:
                break
            ctxs = name_contexts.get(name, [])
            if not ctxs:
                continue
            logging.info(f"name {name} contexts: {ctxs}")
            shown += 1


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
            
            # 단어 정보
            wi = token_to_word.get(pos)
            word = words[wi] if (wi is not None and 0 <= wi < len(words)) else tok
            
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
    """Load influence scores from various formats"""
    if os.path.isfile(scores_path):
        # Single file
        if scores_path.endswith('.pt') or scores_path.endswith('.pth'):
            scores = torch.load(scores_path, map_location='cpu')
            # Support dict format saved by compute_influence (e.g., {'all_modules': tensor})
            if isinstance(scores, dict):
                if 'all_modules' in scores:
                    scores = scores['all_modules']
                else:
                    # Take the first tensor-like entry
                    try:
                        first_val = next(iter(scores.values()))
                        scores = first_val
                    except Exception:
                        raise RuntimeError('Loaded .pt file is a dict without tensor values that can be used as scores')
        elif scores_path.endswith('.pkl'):
            import pickle
            with open(scores_path, 'rb') as f:
                scores = pickle.load(f)
        elif scores_path.endswith('.safetensors'):
            from kronfluence.utils.save import load_file
            from pathlib import Path
            scores_data = load_file(Path(scores_path))
            if 'all_modules' in scores_data:
                scores = scores_data['all_modules']
            else:
                scores = list(scores_data.values())[0]  # Take first tensor
        else:
            raise ValueError(f"Unsupported score file format: {scores_path}")
    else:
        # Directory - load kronfluence pairwise scores
        try:
            from kronfluence.score.pairwise import load_pairwise_scores
            from pathlib import Path
            scores_data = load_pairwise_scores(output_dir=Path(scores_path))
            if 'all_modules' in scores_data:
                scores = scores_data['all_modules']
            else:
                # Get the first module's scores if 'all_modules' doesn't exist
                module_names = list(scores_data.keys())
                logging.info(f"Available modules: {module_names}")
                scores = scores_data[module_names[0]]
        except ImportError:
            # Fallback: try to find and load safetensors file directly
            safetensors_path = os.path.join(scores_path, "pairwise_scores.safetensors")
            if os.path.exists(safetensors_path):
                return load_influence_scores(safetensors_path)  # Recursive call
            else:
                raise FileNotFoundError(f"No pairwise_scores.safetensors found in {scores_path}")
        except Exception as e:
            logging.error(f"Failed to load scores from directory {scores_path}: {e}")
            # Try alternative: load as Analyzer result directory
            try:
                from kronfluence.analyzer import Analyzer
                analyzer = Analyzer("temp", None, None, output_dir=scores_path)
                scores_data = analyzer.load_pairwise_scores(scores_name=os.path.basename(scores_path))
                if 'all_modules' in scores_data:
                    scores = scores_data['all_modules']
                else:
                    scores = list(scores_data.values())[0]
            except Exception as e2:
                logging.error(f"Fallback loading also failed: {e2}")
                raise e
    
    # Ensure we have a tensor
    if not isinstance(scores, torch.Tensor):
        scores = torch.tensor(scores)
    
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
    
    # Load dataset for inspection
    @dataclass
    class DataConfig:
        data_path: str = args.tofu_data_path
        forget_split: str = args.forget_split
        question_key: str = "question"
    
    config = DataConfig()
    forget_dataset, _ = load_datasets(
        config=config, 
        tokenizer=tokenizer, 
        model_family=args.model_family, 
        max_length=args.max_length
    )
    
    # Load influence scores
    logging.info(f"Loading scores from: {args.scores_path}")
    influence_scores = load_influence_scores(args.scores_path)
    
    # Validate dimensions
    if len(influence_scores.shape) != 2:
        raise ValueError(f"Expected 2D scores (samples, tokens), got shape: {influence_scores.shape}")
    
    num_samples, seq_len = influence_scores.shape
    logging.info(f"Processing {num_samples} samples with sequence length {seq_len}")
    
    # Build forget token mask
    forget_mask, sample_ranking = build_forget_token_mask(
        influence_scores=influence_scores,
        forget_threshold=args.forget_threshold,
        context_window=args.context_window, 
        max_forget_tokens=args.max_forget_tokens,
        top_k_samples=args.top_k_samples,
        threshold_mode=args.threshold_mode,
        top_n_per_sample=args.top_n_per_sample,
        min_score=args.min_score,
        select_mode=args.select_mode,
        tokenizer=tokenizer,
        dataset=forget_dataset,
        word_masking=args.word_masking,
        sentence_seps=args.sentence_seps,
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
    
    # Statistics
    total_forget_tokens = forget_mask.sum().item()
    affected_samples = (forget_mask.sum(dim=1) > 0).sum().item()
    avg_tokens_per_sample = total_forget_tokens / max(affected_samples, 1)
    
    logging.info(f"📊 Statistics:")
    logging.info(f"  - Total forget tokens: {total_forget_tokens:,}")
    logging.info(f"  - Affected samples: {affected_samples:,} / {num_samples:,}")
    logging.info(f"  - Avg tokens per affected sample: {avg_tokens_per_sample:.1f}")
    
    # Inspect a sample
    if args.inspection_idx < len(sample_ranking):
        inspect_idx = sample_ranking[args.inspection_idx]
        logging.info(f"🔍 Inspecting sample {inspect_idx} (rank {args.inspection_idx}):")
        
        # Get tokens and mask for this sample
        sample_data = forget_dataset[inspect_idx]
        if isinstance(sample_data, dict) and "input_ids" in sample_data:
            sample_input_ids = sample_data["input_ids"]
        elif isinstance(sample_data, (tuple, list)) and len(sample_data) >= 1:
            sample_input_ids = sample_data[0]  # Assume first element is input_ids
        else:
            logging.error(f"Cannot extract input_ids from sample data: {type(sample_data)}")
            logging.info("Skipping inspection due to data format, continuing with JSON generation...")
            # Jump to the JSON generation section
            return _generate_json_files(tokenizer, forget_dataset, forget_mask, influence_scores, args)
            
        # Align lengths and convert to tokens  
        if torch.is_tensor(sample_input_ids):
            input_ids_tensor = sample_input_ids.detach().cpu()
        else:
            input_ids_tensor = torch.tensor(sample_input_ids)
        L = min(int(influence_scores.shape[1]), int(input_ids_tensor.shape[0]), int(forget_mask.shape[1]))
        sample_tokens = tokenizer.convert_ids_to_tokens(input_ids_tensor.tolist())
        sample_mask = forget_mask[inspect_idx][:L]
        sample_scores = influence_scores[inspect_idx][:L]
        
        # Build word-level view
        words, token_to_word = _map_token_index_to_word(sample_tokens)
        if not words:
            # Fallback to token rendering if word mapping failed
            special_tokens = {'<|endoftext|>', '<s>', '</s>', '<unk>', '<pad>'}
            def render_token(tok: str) -> str:
                if tok in special_tokens:
                    return ""
                if tok == 'Ċ':
                    return "\n"
                if tok.startswith('Ġ'):
                    return ' ' + tok[1:]
                return tok
            logging.info("Sample text with forget tokens marked:")
            print("\n" + "="*80)
            output_line = ""
            for token, should_forget in zip(sample_tokens, sample_mask):
                seg = render_token(token)
                if not seg:
                    continue
                if should_forget:
                    output_line += f"[FORGET:{seg}]"
                else:
                    output_line += seg
            print(output_line)
            print("="*80)
            print("\nForget tokens highlighted:")
            clean_text = ""
            for token, should_forget in zip(sample_tokens, sample_mask):
                seg = render_token(token)
                if not seg:
                    continue
                if should_forget:
                    clean_text += f"**{seg}**"
                else:
                    clean_text += seg
            print(clean_text)
            print("="*80)
        else:
            # Compute word-level forget flags
            word_flags = [False] * len(words)
            for pos, flag in enumerate(sample_mask.tolist()):
                wi = token_to_word.get(pos)
                if wi is not None and 0 <= wi < len(word_flags):
                    word_flags[wi] = word_flags[wi] or bool(flag)
            
            logging.info("Sample text with forget tokens marked (word-level):")
            print("\n" + "="*80)
            output_words = []
            for w, f in zip(words, word_flags):
                if not w:
                    continue
                output_words.append(f"[FORGET:{w}]" if f else w)
            print(' '.join(output_words))
            print("="*80)
            
            print("\nForget tokens highlighted (word-level):")
            clean_words = []
            for w, f in zip(words, word_flags):
                if not w:
                    continue
                clean_words.append(f"**{w}**" if f else w)
            print(' '.join(clean_words))
        print("="*80)
        
        # Show detailed statistics for this sample
        forget_count = sample_mask.sum().item()
        max_score = sample_scores.max().item()
        min_score = sample_scores.min().item()
        avg_score = sample_scores.mean().item()
        
        logging.info(f"\n📈 Sample {inspect_idx} statistics:")
        logging.info(f"  - Forget tokens: {forget_count}/{len(sample_mask)} ({forget_count/len(sample_mask):.1%})")
        logging.info(f"  - Score range: {min_score} to {max_score}")
        logging.info(f"  - Average score: {avg_score}")
        
        # Show forget token positions
        forget_positions = [i for i, mask in enumerate(sample_mask) if mask]
        if forget_positions:
            logging.info(f"  - Forget positions: {forget_positions[:10]}{'...' if len(forget_positions) > 10 else ''}")
            
        # Show top forget tokens with their scores
        forget_tokens_with_scores = [(i, token, score.item()) for i, (token, mask, score) in 
                                    enumerate(zip(sample_tokens, sample_mask, sample_scores)) if mask]
        forget_tokens_with_scores.sort(key=lambda x: x[2], reverse=True)  # Sort by score
        
        logging.info(f"  - Top forget tokens:")
        for i, (pos, token, score) in enumerate(forget_tokens_with_scores[:5]):
            logging.info(f"    {i+1}. pos={pos:3d} score={score} token='{token}'")
    
    logging.info("✅ Forget token selection completed!")

    # Optional: emit token/word/name report (template excluded)
    if args.emit_token_report:
        template_exclude = set([s for s in args.template_exclude.split(',') if s != ''])
        logging.info(f"📊 Emitting token report with template exclusion: {template_exclude}")
        _emit_token_report(
            tokenizer=tokenizer,
            dataset=forget_dataset,
            forget_mask=forget_mask,
            influence_scores=influence_scores,
            top_k=args.report_top_k,
            context_window=args.context_window,
            template_exclude=template_exclude,
            report_output=args.report_output,
            count_context=args.report_count_context,
        )
    
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