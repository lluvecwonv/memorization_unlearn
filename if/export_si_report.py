#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Export SI(differential influence) report to JSONL and HTML/PNG visualizations.

Inputs:
- Influence scores (2D tensor: [num_samples, seq_len]) from compute_influence.py
- Optional: forget token mask (same shape, bool) from select_forget_tokens.py
- Tokenizer (via model_config.yaml -> hf_key) and dataset (for input_ids)

Outputs:
- JSONL per-sample records: {text, tokens, token_ids, si_scores, selected_mask, ...}
- Sample HTML with SI heatmap and selected-token highlighting
- Aggregated figures: SI histogram, selected-count histogram, top tokens by avg SI
- Manifest metadata for reproducibility
"""

import argparse
import json
import logging
import math
import os
import random
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

# Matplotlib headless backend
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export SI report (JSONL + visualizations)")

    # Inputs
    parser.add_argument("--scores_path", type=str, required=True, help="Path to SI scores (.pt/.pth/.pkl/.safetensors or kronfluence dir)")
    parser.add_argument("--checkpoint_dir", type=str, required=True, help="Model checkpoint directory (for tokenizer compatibility)")
    parser.add_argument("--model_config_path", type=str, default=None, help="Path to model_config.yaml (default: ../config/model_config.yaml relative to this file)")

    # Data
    parser.add_argument("--tofu_data_path", type=str, default="locuslab/TOFU", help="TOFU dataset path")
    parser.add_argument("--forget_split", type=str, default="forget10", help="Forget split (e.g., forget10)")
    parser.add_argument("--model_family", type=str, default="gpt", help="Model family key in model_config.yaml")
    parser.add_argument("--max_length", type=int, default=512, help="Maximum sequence length for tokenization")

    # Mask options
    parser.add_argument("--mask_path", type=str, default=None, help="Optional path to forget token mask (.pt)")
    parser.add_argument("--top_n_per_sample", type=int, default=1, help="If mask not provided, build mask with this top-N per sample")
    parser.add_argument("--min_score", type=float, default=0.0, help="Minimum SI cutoff when building mask on the fly")
    parser.add_argument("--mask_context_window", type=int, default=0, help="Context window size to include around selected tokens when building mask on the fly")

    # Output
    parser.add_argument("--output_dir", type=str, default="./si_report", help="Output directory root")
    parser.add_argument("--limit_samples", type=int, default=None, help="Limit number of samples to export (debug)")
    parser.add_argument("--viz_top_k_samples", type=int, default=50, help="Number of top samples to render as HTML")
    parser.add_argument("--float_precision", type=int, default=6, help="Float rounding for SI scores in JSONL")
    parser.add_argument("--top_tokens_selected_only", action="store_true", default=False, help="Compute top-tokens chart using only selected (masked) tokens")
    parser.add_argument("--top_k_tokens", type=int, default=30, help="How many top tokens to plot (set -1 for ALL; may be unreadable)")
    # Context reporting
    parser.add_argument("--context_window_words", type=int, default=1, help="Word-level context window size on each side for examples")
    parser.add_argument("--examples_per_item", type=int, default=2, help="Number of context examples to store per top token/word item")
    parser.add_argument("--top_k_words", type=int, default=30, help="How many top words to include in context JSON (pos/neg each)")

    # Misc
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose", action="store_true", default=False)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_model_config_path(cli_path: Optional[str]) -> str:
    if cli_path is not None:
        return cli_path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_path = os.path.abspath(os.path.join(script_dir, "..", "config", "model_config.yaml"))
    return default_path


def load_tokenizer_and_dataset(args) -> Tuple[object, object]:
    import yaml
    from transformers import AutoTokenizer

    # model_config.yaml -> hf_key
    config_path = resolve_model_config_path(args.model_config_path)
    logging.info(f"Loading model config from: {config_path}")
    with open(config_path, "r") as f:
        model_configs = yaml.safe_load(f)

    if args.model_family not in model_configs:
        raise ValueError(f"Model family '{args.model_family}' not found in config")
    hf_key = model_configs[args.model_family]["hf_key"]
    logging.info(f"Loading tokenizer from: {hf_key}")
    tokenizer = AutoTokenizer.from_pretrained(hf_key)

    # dataset
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if script_dir not in sys.path:
        sys.path.append(script_dir)
    # local_utils.data_utils is expected to be importable via project root
    project_root = os.path.abspath(os.path.join(script_dir, ".."))
    if project_root not in sys.path:
        sys.path.append(project_root)

    @dataclass
    class DataConfig:
        data_path: str = args.tofu_data_path
        forget_split: str = args.forget_split
        question_key: str = "question"

    from local_utils.data_utils import load_datasets

    data_cfg = DataConfig()
    forget_dataset, _ = load_datasets(
        config=data_cfg,
        tokenizer=tokenizer,
        model_family=args.model_family,
        max_length=args.max_length,
    )
    return tokenizer, forget_dataset


def try_import_selection_utils():
    """Import utilities from select_forget_tokens.py with fallback implementations."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if script_dir not in sys.path:
        sys.path.append(script_dir)
    try:
        from select_forget_tokens import load_influence_scores as _load_scores
        from select_forget_tokens import build_forget_token_mask as _build_mask
        from select_forget_tokens import _map_token_index_to_word as _map_tokens
        return _load_scores, _build_mask, _map_tokens
    except Exception as e:
        logging.warning(f"Falling back: could not import from select_forget_tokens.py ({e}). Using internal minimal loaders.")

        def _load_scores(scores_path: str) -> torch.Tensor:
            # Minimal loader (pt/pth/pkl); extend as needed
            if os.path.isfile(scores_path):
                if scores_path.endswith((".pt", ".pth")):
                    return torch.load(scores_path, map_location="cpu")
                if scores_path.endswith(".pkl"):
                    import pickle
                    with open(scores_path, "rb") as f:
                        return torch.tensor(pickle.load(f))
            raise ValueError(f"Unsupported or missing scores at: {scores_path}")

        def _build_mask(
            influence_scores: torch.Tensor,
            forget_threshold: float = 0.95,
            context_window: int = 0,
            max_forget_tokens: int = 10**9,
            top_k_samples: Optional[int] = None,
            threshold_mode: str = "topk_per_sample",
            top_n_per_sample: int = 1,
            min_score: float = 0.0,
        ) -> Tuple[torch.Tensor, List[int]]:
            scores = influence_scores.detach().float()
            B, T = scores.shape
            mask = torch.zeros(B, T, dtype=torch.bool)
            for b in range(B):
                row = scores[b]
                if min_score > 0:
                    row = torch.where(row >= min_score, row, torch.full_like(row, float("-inf")))
                valid = (row > float("-inf")).sum().item()
                k = max(1, min(int(top_n_per_sample), valid))
                if k > 0:
                    topk = torch.topk(row, k=k, dim=0)
                    mask[b, topk.indices] = True
            ranking = list(range(B))
            return mask, ranking

        def _map_tokens(tokens: list[str]) -> tuple[list[str], dict[int, int]]:
            if not tokens:
                return [], {}
            has_gpt2_space = any(t.startswith('Ġ') for t in tokens)
            has_sp_space = any(t.startswith('▁') for t in tokens)
            punct = set(list(".,!?;:()[]{}\"'`…—–"))
            special_tokens = {'<|endoftext|>', '<s>', '</s>', '<unk>', '<pad>'}
            words: list[str] = []
            t2w: dict[int, int] = {}
            current: list[str] = []
            def flush():
                nonlocal current
                if current:
                    words.append(''.join(current))
                    current = []
            for i, tok in enumerate(tokens):
                if tok in special_tokens or tok == 'Ċ':
                    flush()
                    continue
                is_space = False
                if has_gpt2_space and tok.startswith('Ġ'):
                    tok_clean = tok[1:]
                    is_space = True
                elif has_sp_space and tok.startswith('▁'):
                    tok_clean = tok[1:]
                    is_space = True
                else:
                    tok_clean = tok
                if tok_clean in punct and len(tok_clean) == 1:
                    flush()
                    continue
                if is_space:
                    flush()
                if tok_clean == '-':
                    if not current:
                        t2w[i] = len(words)
                        current.append('-')
                    else:
                        t2w[i] = len(words)
                        current.append('-')
                    continue
                if not current:
                    t2w[i] = len(words)
                    current.append(tok_clean)
                else:
                    t2w[i] = len(words)
                    current.append(tok_clean)
            flush()
            words = [w for w in words if w.strip() != ""]
            return words, t2w

        return _load_scores, _build_mask, _map_tokens


def min_max_normalize(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    x_min = float(np.min(x))
    x_max = float(np.max(x))
    return (x - x_min) / (max(eps, x_max - x_min + eps))


def si_to_color(value: float, vmin: float, vmax: float) -> str:
    """Diverging color map for SI: negative=blue, positive=red, zero=transparent.

    - Uses symmetric range around zero with vabs=max(|vmin|, |vmax|) for consistent scale
    - Alpha ~ |value| / vabs
    - Positive -> red; Negative -> blue
    """
    vabs = max(1e-12, float(max(abs(vmin), abs(vmax))))
    magnitude = min(1.0, max(0.0, abs(float(value)) / vabs))
    if value >= 0:
        r, g, b = 255, 0, 0   # red for positive
    else:
        r, g, b = 0, 80, 255  # blue for negative
    background = f"rgba({r}, {g}, {b}, {magnitude:.3f})"
    return background


def render_sample_html(
    out_path: str,
    tokens: List[str],
    si_scores: List[float],
    selected_mask: List[bool],
    header_meta: Dict[str, object],
) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    vmin = float(np.percentile(si_scores, 1)) if len(si_scores) > 0 else 0.0
    vmax = float(np.percentile(si_scores, 99)) if len(si_scores) > 0 else 1.0

    lines = []
    lines.append("<html><head><meta charset='utf-8'><style>")
    lines.append("body{font-family:Arial,Helvetica,sans-serif;line-height:1.65;padding:16px;}")
    lines.append(".tok{display:inline-block;padding:2px 3px;margin:1px;border-radius:4px;border:1px solid rgba(0,0,0,0.06);}")
    lines.append(".sel{font-weight:bold;text-decoration:underline;box-shadow:inset 0 0 0 1px rgba(0,0,0,0.25);}")
    lines.append(".header{margin-bottom:10px;padding:8px 10px;background:#f7f7f7;border-radius:6px;border:1px solid #eee;}")
    lines.append(".meta{color:#444;font-size:13px;}")
    lines.append("</style></head><body>")

    # Header
    lines.append("<div class='header'>")
    lines.append("<div class='meta'><b>Sample metadata</b></div>")
    for k, v in header_meta.items():
        lines.append(f"<div class='meta'>{k}: {v}</div>")
    lines.append("</div>")

    # Tokens
    lines.append("<div class='content'>")
    for t, s, sel in zip(tokens, si_scores, selected_mask):
        bg = si_to_color(float(s), vmin, vmax)
        cls = "tok sel" if sel else "tok"
        safe_t = (t or "").replace("<", "&lt;").replace(">", "&gt;")
        lines.append(f"<span class='{cls}' style='background:{bg}' title='SI={s:.6f}'>{safe_t}</span>")
    lines.append("</div>")

    lines.append("</body></html>")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def save_manifest(output_dir: str, manifest: Dict[str, object]) -> None:
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "manifest.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)


def export_report(args: argparse.Namespace) -> None:
    # Setup
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=level, format='[%(asctime)s] [%(levelname)s] %(message)s')
    set_seed(args.seed)

    # I/O layout
    output_dir = os.path.abspath(args.output_dir)
    samples_dir = os.path.join(output_dir, "samples")
    figs_dir = os.path.join(output_dir, "figs")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(samples_dir, exist_ok=True)
    os.makedirs(figs_dir, exist_ok=True)

    # Load tokenizer and dataset
    tokenizer, forget_dataset = load_tokenizer_and_dataset(args)

    # Load scores & mask utils
    load_influence_scores, build_forget_token_mask, map_token_index_to_word = try_import_selection_utils()

    logging.info(f"Loading SI scores from: {args.scores_path}")
    influence_scores = load_influence_scores(args.scores_path)
    if not isinstance(influence_scores, torch.Tensor):
        influence_scores = torch.tensor(influence_scores)
    if influence_scores.dim() != 2:
        raise ValueError(f"Expected 2D SI scores (B, T), got shape {tuple(influence_scores.shape)}")

    num_samples, seq_len = influence_scores.shape
    logging.info(f"Scores loaded: shape={tuple(influence_scores.shape)}, dtype={influence_scores.dtype}")

    # Load or build mask
    forget_mask: Optional[torch.Tensor] = None
    if args.mask_path is not None and os.path.exists(args.mask_path):
        logging.info(f"Loading forget mask from: {args.mask_path}")
        forget_mask = torch.load(args.mask_path, map_location="cpu")
    else:
        logging.info("No mask provided; building forget mask on the fly (top-k per sample)")
        forget_mask, _ = build_forget_token_mask(
            influence_scores=influence_scores,
            forget_threshold=0.95,
            context_window=int(max(0, args.mask_context_window)),
            max_forget_tokens=10**9,
            top_k_samples=None,
            threshold_mode="topk_per_sample",
            top_n_per_sample=args.top_n_per_sample,
            min_score=args.min_score,
        )
    if not isinstance(forget_mask, torch.Tensor):
        forget_mask = torch.tensor(forget_mask, dtype=torch.bool)
    if forget_mask.shape != influence_scores.shape:
        logging.warning(f"Mask shape {tuple(forget_mask.shape)} != scores shape {tuple(influence_scores.shape)}; aligning by min dims")
        B = min(forget_mask.shape[0], influence_scores.shape[0])
        T = min(forget_mask.shape[1], influence_scores.shape[1])
        forget_mask = forget_mask[:B, :T]
        influence_scores = influence_scores[:B, :T]
        num_samples, seq_len = influence_scores.shape

    # JSONL writer
    jsonl_path = os.path.join(output_dir, "report.jsonl")
    jsonl_file = open(jsonl_path, "w", encoding="utf-8")

    # Aggregates
    all_si_values: List[float] = []
    selected_counts: List[int] = []
    token_to_values: Dict[str, List[float]] = {}
    word_to_values: Dict[str, List[float]] = {}
    word_examples: Dict[str, List[Dict[str, object]]] = {}

    # Sample ranking for viz: sum of SI per sample
    per_sample_sum = torch.sum(influence_scores.float(), dim=1)
    ranked_indices = torch.argsort(per_sample_sum, descending=True).tolist()
    if args.limit_samples is not None:
        ranked_indices = ranked_indices[: args.limit_samples]

    float_precision = int(max(0, args.float_precision))

    viz_limit = len(ranked_indices) if int(args.viz_top_k_samples) < 0 else int(args.viz_top_k_samples)
    for rank_idx, sample_idx in enumerate(ranked_indices):
        sample_idx_int = int(sample_idx)
        # Get dataset item
        sample = forget_dataset[sample_idx_int]
        if isinstance(sample, dict) and "input_ids" in sample:
            input_ids = sample["input_ids"]
        elif isinstance(sample, (tuple, list)) and len(sample) >= 1:
            input_ids = sample[0]
        else:
            logging.error(f"Cannot extract input_ids for sample {sample_idx_int}; skipping.")
            continue

        if torch.is_tensor(input_ids):
            input_ids_tensor = input_ids.detach().cpu()
        else:
            input_ids_tensor = torch.tensor(input_ids)

        # Align lengths
        scores_row = influence_scores[sample_idx_int].detach().cpu()
        mask_row = forget_mask[sample_idx_int].detach().cpu()
        L = min(int(scores_row.shape[0]), int(input_ids_tensor.shape[0]), int(mask_row.shape[0]))
        if L <= 0:
            logging.warning(f"Empty row for sample {sample_idx_int}; skipping")
            continue
        input_ids_tensor = input_ids_tensor[:L]
        scores_row = scores_row[:L]
        mask_row = mask_row[:L]

        # Convert to tokens and text
        tokens: List[str] = tokenizer.convert_ids_to_tokens(input_ids_tensor.tolist())
        try:
            text = tokenizer.decode(input_ids_tensor.tolist(), skip_special_tokens=False, clean_up_tokenization_spaces=True)
        except Exception:
            text = ""

        # Update aggregates (token-level)
        all_si_values.extend([float(x) for x in scores_row.tolist()])
        selected_count = int(mask_row.sum().item())
        selected_counts.append(selected_count)
        for t_str, si_val in zip(tokens, scores_row.tolist()):
            token_to_values.setdefault(t_str, []).append(float(si_val))

        # Update aggregates (word-level)
        words, t2w = map_token_index_to_word(tokens)
        if words:
            word_scores: Dict[int, List[float]] = {}
            for ti, si_val in enumerate(scores_row.tolist()):
                wi = t2w.get(ti)
                if wi is None or wi < 0 or wi >= len(words):
                    continue
                word_scores.setdefault(wi, []).append(float(si_val))
            for wi, vals in word_scores.items():
                if len(vals) == 0:
                    continue
                word_to_values.setdefault(words[wi], []).append(float(np.mean(vals)))
                # Store a few context examples per word
                if len(word_examples.setdefault(words[wi], [])) < int(max(0, args.examples_per_item)):
                    s = max(0, wi - int(max(0, args.context_window_words)))
                    e = min(len(words), wi + int(max(0, args.context_window_words)) + 1)
                    ctx_words = words[s:wi] + [f"[{words[wi]}]"] + words[wi+1:e]
                    word_examples[words[wi]].append({
                        "sample_index": sample_idx_int,
                        "context": " ".join(ctx_words),
                        "avg_si_local": float(np.mean(vals)),
                    })

        # JSON record
        if float_precision >= 0:
            si_list = [round(float(x), float_precision) for x in scores_row.tolist()]
        else:
            si_list = [float(x) for x in scores_row.tolist()]

        record = {
            "sample_index": sample_idx_int,
            "dataset": {"name": "TOFU", "split": args.forget_split},
            "model": {"family": args.model_family, "checkpoint_dir": args.checkpoint_dir},
            "seq_len": L,
            "text": text,
            "tokens": tokens,
            "token_ids": input_ids_tensor.tolist(),
            "si_scores": si_list,
            "selected_mask": mask_row.tolist(),
            "selected_indices": torch.nonzero(mask_row, as_tuple=True)[0].tolist(),
            "meta": {"min_score": args.min_score, "top_n_per_sample": args.top_n_per_sample},
        }
        jsonl_file.write(json.dumps(record, ensure_ascii=False) + "\n")

        # Per-sample HTML (only for top K by SI sum)
        if rank_idx < viz_limit:
            header = {
                "sample_index": sample_idx_int,
                "seq_len": L,
                "selected_count": selected_count,
            }
            out_html = os.path.join(samples_dir, f"sample_{sample_idx_int}.html")
            render_sample_html(out_html, tokens, [float(x) for x in scores_row.tolist()], mask_row.tolist(), header)

    jsonl_file.close()
    logging.info(f"Saved JSONL to: {jsonl_path}")

    # Aggregated visualizations
    if len(all_si_values) > 0:
        # SI histogram
        plt.figure(figsize=(6, 4))
        plt.hist(all_si_values, bins=80, color="#d33", alpha=0.85)
        plt.xlabel("SI score")
        plt.ylabel("Count")
        plt.title("Distribution of SI scores")
        si_hist_path = os.path.join(figs_dir, "si_hist.png")
        plt.tight_layout()
        plt.savefig(si_hist_path)
        plt.close()
        logging.info(f"Saved SI histogram: {si_hist_path}")

    if len(selected_counts) > 0:
        plt.figure(figsize=(6, 4))
        plt.hist(selected_counts, bins=min(60, max(5, int(np.sqrt(len(selected_counts))))), color="#369", alpha=0.85)
        plt.xlabel("# selected tokens per sample")
        plt.ylabel("Count")
        plt.title("Selected tokens per sample")
        sel_hist_path = os.path.join(figs_dir, "selected_count_hist.png")
        plt.tight_layout()
        plt.savefig(sel_hist_path)
        plt.close()
        logging.info(f"Saved selected-count histogram: {sel_hist_path}")

    # Top tokens/words by average SI
    if len(token_to_values) > 0:
        if args.top_tokens_selected_only:
            # Recompute token_to_values using only selected tokens
            token_to_values_sel: Dict[str, List[float]] = {}
            jsonl_for_sel = os.path.join(output_dir, "report.jsonl")
            with open(jsonl_for_sel, "r", encoding="utf-8") as jf:
                for line in jf:
                    rec = json.loads(line)
                    toks = rec.get("tokens", [])
                    scores = rec.get("si_scores", [])
                    sel_idx = set(rec.get("selected_indices", []))
                    for i, t in enumerate(toks):
                        if i in sel_idx:
                            token_to_values_sel.setdefault(t, []).append(float(scores[i]) if i < len(scores) else 0.0)
            token_avg = [(tok, float(np.mean(vals))) for tok, vals in token_to_values_sel.items() if len(vals) > 0]
        else:
            token_avg = [(tok, float(np.mean(vals))) for tok, vals in token_to_values.items() if len(vals) > 0]
        token_avg.sort(key=lambda x: x[1], reverse=True)
        k_tok = int(args.top_k_tokens)
        top_k = token_avg if k_tok < 0 else token_avg[:k_tok]
        if len(top_k) > 0:
            labels = [t for t, _ in top_k]
            values = [v for _, v in top_k]
            plt.figure(figsize=(max(8, len(top_k) * 0.4), 4 if k_tok <= 50 or k_tok < 0 else 6))
            plt.bar(range(len(top_k)), values, color="#7a1")
            plt.xticks(range(len(top_k)), labels, rotation=60, ha="right")
            plt.ylabel("Avg SI")
            plt.title("Top tokens by average SI (positive)")
            top_tokens_path = os.path.join(figs_dir, "top_tokens.png")
            plt.tight_layout()
            plt.savefig(top_tokens_path)
            plt.close()
            logging.info(f"Saved top-tokens chart: {top_tokens_path}")

        # Also save chart for most negative average SI tokens
        token_avg_neg = [(tok, val) for tok, val in token_avg if val < 0]
        token_avg_neg.sort(key=lambda x: x[1])  # ascending (most negative first)
        top_k_neg = token_avg_neg if k_tok < 0 else token_avg_neg[:k_tok]
        if len(top_k_neg) > 0:
            labels = [t for t, _ in top_k_neg]
            values = [v for _, v in top_k_neg]
            plt.figure(figsize=(max(8, len(top_k_neg) * 0.4), 4 if k_tok <= 50 or k_tok < 0 else 6))
            plt.bar(range(len(top_k_neg)), values, color="#17a")
            plt.xticks(range(len(top_k_neg)), labels, rotation=60, ha="right")
            plt.ylabel("Avg SI")
            plt.title("Top tokens by average SI (negative)")
            top_tokens_neg_path = os.path.join(figs_dir, "top_tokens_negative.png")
            plt.tight_layout()
            plt.savefig(top_tokens_neg_path)
            plt.close()
            logging.info(f"Saved top-tokens negative chart: {top_tokens_neg_path}")

    if len(word_to_values) > 0:
        word_avg = [(w, float(np.mean(vals))) for w, vals in word_to_values.items() if len(vals) > 0]
        word_avg.sort(key=lambda x: x[1], reverse=True)
        k_w = int(args.top_k_words)
        top_k_w = word_avg if k_w < 0 else word_avg[:k_w]
        if len(top_k_w) > 0:
            labels = [w for w, _ in top_k_w]
            values = [v for _, v in top_k_w]
            plt.figure(figsize=(max(8, len(top_k_w) * 0.5), 4 if k_w <= 50 or k_w < 0 else 6))
            plt.bar(range(len(top_k_w)), values, color="#1a7")
            plt.xticks(range(len(top_k_w)), labels, rotation=60, ha="right")
            plt.ylabel("Avg SI (word)")
            plt.title("Top words by average SI (positive)")
            top_words_path = os.path.join(figs_dir, "top_words.png")
            plt.tight_layout()
            plt.savefig(top_words_path)
            plt.close()
            logging.info(f"Saved top-words chart: {top_words_path}")

        # Negative words
        word_avg_neg = [(w, val) for w, val in word_avg if val < 0]
        word_avg_neg.sort(key=lambda x: x[1])
        top_k_w_neg = word_avg_neg if k_w < 0 else word_avg_neg[:k_w]
        if len(top_k_w_neg) > 0:
            labels = [w for w, _ in top_k_w_neg]
            values = [v for _, v in top_k_w_neg]
            plt.figure(figsize=(max(8, len(top_k_w_neg) * 0.5), 4 if k_w <= 50 or k_w < 0 else 6))
            plt.bar(range(len(top_k_w_neg)), values, color="#17a")
            plt.xticks(range(len(top_k_w_neg)), labels, rotation=60, ha="right")
            plt.ylabel("Avg SI (word)")
            plt.title("Top words by average SI (negative)")
            top_words_neg_path = os.path.join(figs_dir, "top_words_negative.png")
            plt.tight_layout()
            plt.savefig(top_words_neg_path)
            plt.close()
            logging.info(f"Saved top-words negative chart: {top_words_neg_path}")

    # Manifest
    manifest = {
        "scores_path": args.scores_path,
        "checkpoint_dir": args.checkpoint_dir,
        "model_family": args.model_family,
        "forget_split": args.forget_split,
        "max_length": args.max_length,
        "output_dir": output_dir,
        "limit_samples": args.limit_samples,
        "viz_top_k_samples": args.viz_top_k_samples,
        "float_precision": args.float_precision,
        "seed": args.seed,
        "num_samples_exported": len(ranked_indices),
    }
    save_manifest(output_dir, manifest)
    logging.info("✅ Export completed.")

    # Save top word context examples to JSON
    try:
        import json as _json
        # Build sorted top lists for context
        word_avg_all = [(w, float(np.mean(vs))) for w, vs in word_to_values.items() if len(vs) > 0]
        word_avg_all.sort(key=lambda x: x[1], reverse=True)
        pos_words = [w for w, _ in word_avg_all[: int(max(1, args.top_k_words))]]
        word_avg_all.sort(key=lambda x: x[1])
        neg_words = [w for w, _ in word_avg_all[: int(max(1, args.top_k_words))]]
        ctx = {
            "top_words_positive": {w: word_examples.get(w, [])[: int(max(0, args.examples_per_item))] for w in pos_words},
            "top_words_negative": {w: word_examples.get(w, [])[: int(max(0, args.examples_per_item))] for w in neg_words},
            "context_window_words": int(max(0, args.context_window_words)),
        }

        # Build selection-based contexts for ALL selected tokens across samples
        # Expansion rule: expand ±context_window_words only if SI(center) > 0; otherwise keep center only
        selected_context = []
        with open(jsonl_path, "r", encoding="utf-8") as jf:
            for line in jf:
                rec = _json.loads(line)
                toks = rec.get("tokens", [])
                scores = rec.get("si_scores", [])
                sel_idx = rec.get("selected_indices", []) or []
                sample_index = rec.get("sample_index", None)
                words, t2w = map_token_index_to_word(toks)
                for ti in sel_idx:
                    if ti < 0 or ti >= len(scores):
                        continue
                    si_val = float(scores[ti])
                    expand = (si_val > 0) and (int(max(0, args.context_window_words)) > 0)
                    wi = t2w.get(ti) if isinstance(t2w, dict) else None
                    if wi is not None and 0 <= wi < len(words):
                        cwin = int(max(0, args.context_window_words)) if expand else 0
                        left_w = max(0, wi - cwin)
                        right_w = min(len(words), wi + 1 + cwin)
                        ctx_words = words[left_w:wi] + [f"[{words[wi]}]"] + words[wi+1:right_w]
                        selected_context.append({
                            "sample_index": sample_index,
                            "word_index": int(wi),
                            "center": words[wi],
                            "si": si_val,
                            "expanded": bool(expand),
                            "context": " ".join(ctx_words),
                        })
                    else:
                        # Fallback to token-level window if word mapping is unavailable
                        cwin = 1 if expand else 0
                        left_t = max(0, ti - cwin)
                        right_t = min(len(toks), ti + 1 + cwin)
                        ctx_toks = toks[left_t:ti] + [f"[{toks[ti]}]"] + toks[ti+1:right_t]
                        selected_context.append({
                            "sample_index": sample_index,
                            "token_index": int(ti),
                            "center": toks[ti],
                            "si": si_val,
                            "expanded": bool(expand),
                            "context": " ".join(ctx_toks),
                        })

        ctx["selected_context"] = selected_context
        ctx["selected_context_count"] = len(selected_context)
        ctx_path = os.path.join(output_dir, "top_words_context.json")
        with open(ctx_path, "w", encoding="utf-8") as jf:
            _json.dump(ctx, jf, ensure_ascii=False, indent=2)
        logging.info(f"Saved top word context JSON: {ctx_path}")
    except Exception as e:
        logging.warning(f"Failed to save top word context JSON: {e}")


def main() -> None:
    args = parse_args()
    export_report(args)


if __name__ == "__main__":
    main()


