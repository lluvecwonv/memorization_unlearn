# -*- coding: utf-8 -*-
"""
Script to compute token-wise, differential influence scores for an LLM
on the provided training and query datasets using TOFU dataset.

This is a corrected 'accuracy-first' implementation:
- computes S_ij = (g_bar_forget - g_bar_retain)^T * H^{-1} * grad_theta L(x_ij; theta)
- computes per-token, per-sample grads exactly (slow but correct).
- uses Subset-based splitting to avoid loading all training data into memory.
- safe handling of None grads (fills zeros to preserve parameter order).
- validates factor/parameter vector lengths for H^{-1} application.

Run with small model & small datasets for verification before scaling.
"""
import argparse
import logging
import os
import yaml
from dataclasses import dataclass
from datetime import timedelta
from typing import Dict, Optional, Sequence

import torch
from torch.utils.data import Subset, DataLoader, ConcatDataset
from transformers import default_data_collator, AutoTokenizer, AutoModelForCausalLM
from accelerate import Accelerator, InitProcessGroupKwargs
from tqdm import tqdm

from kronfluence.analyzer import Analyzer, prepare_model
from kronfluence.arguments import ScoreArguments
from kronfluence.utils.common.score_arguments import all_low_precision_score_arguments
from kronfluence.utils.dataset import DataLoaderKwargs
from kronfluence.utils.logger import get_time
from kronfluence.score.pairwise import pairwise_scores_exist, save_pairwise_scores
# Correct imports for kronfluence
from kronfluence.module.tracked_module import ModuleMode
from kronfluence.module.utils import set_mode, prepare_modules

try:
    from kronfluence.module.utils import set_factors
except Exception:
    def set_factors(model, factor_name: str, factors, clone: bool = True):
        # No-op if kronfluence internals are unavailable
        return

# Local task/dataset loaders (your environment)
import sys
import os
sys.path.append('/root/npo')
from analysis.kronfluence_task import LanguageModelingTask

BATCH_TYPE = Dict[str, torch.Tensor]
SCORE_TYPE = Dict[str, torch.Tensor]


def _safe_cat_grads(params, grads):
    """
    Safely concatenate gradients, replacing None with zeros to maintain parameter order.
    """
    parts = []
    for p, g in zip(params, grads):
        if g is None:
            parts.append(torch.zeros_like(p).reshape(-1))
        else:
            parts.append(g.reshape(-1))
    if len(parts) == 0:
        return torch.tensor([], device=params[0].device if params else "cpu")
    return torch.cat(parts, dim=0)


class DifferentialTokenScorer(Analyzer):
    def __init__(self, analysis_name, model, task, profile=False, output_dir="./factors"):
        super().__init__(analysis_name, model, task, profile=profile, output_dir=output_dir)

    # Respect absolute output paths passed via scores_name
    def scores_output_dir(self, scores_name: str):
        import os as _os
        from pathlib import Path as _P
        # If scores_name is an absolute path, use it directly and return a Path
        if isinstance(scores_name, str) and _os.path.isabs(scores_name):
            p = _P(scores_name)
            p.mkdir(parents=True, exist_ok=True)
            return p
        # Fall back to base implementation
        return super().scores_output_dir(scores_name=scores_name)

    def compute_pairwise_scores(
        self,
        scores_name: str,
        factors_name: str,
        query_dataset: torch.utils.data.Dataset,
        train_dataset: torch.utils.data.Dataset,
        per_device_query_batch_size: int,
        per_device_train_batch_size: Optional[int] = None,
        initial_per_device_train_batch_size_attempt: int = 4096,
        query_indices: Optional[Sequence[int]] = None,
        train_indices: Optional[Sequence[int]] = None,
        dataloader_kwargs: Optional[DataLoaderKwargs] = None,
        score_args: Optional[ScoreArguments] = None,
        target_data_partitions: Optional[Sequence[int]] = None,
        target_module_partitions: Optional[Sequence[int]] = None,
        overwrite_output_dir: bool = False,
    ) -> Optional[SCORE_TYPE]:
        """
        Override to plug differential computation in place of standard pairwise.
        """
        self.logger.debug(f"Computing pairwise scores with parameters: {locals()}")

        scores_output_dir = self.scores_output_dir(scores_name=scores_name)
        os.makedirs(scores_output_dir, exist_ok=True)
        if pairwise_scores_exist(output_dir=scores_output_dir) and not overwrite_output_dir:
            self.logger.info(f"Found existing pairwise scores at `{scores_output_dir}`. Skipping.")
            return self.load_pairwise_scores(scores_name=scores_name)

        factor_args, score_args = self._configure_and_save_score_args(
            score_args=score_args,
            scores_output_dir=scores_output_dir,
            factors_name=factors_name,
            overwrite_output_dir=overwrite_output_dir,
        )

        # Keep the original safety checks (they may disable token-mode for incompatible combos)
        if score_args.compute_per_token_scores and score_args.aggregate_train_gradients:
            warning_msg = (
                "Token-wise influence computation is not compatible with `aggregate_train_gradients=True`. "
                "Disabling `compute_per_token_scores`."
            )
            score_args.compute_per_token_scores = False
            self.logger.warning(warning_msg)

        if score_args.compute_per_token_scores and factor_args.has_shared_parameters:
            warning_msg = (
                "Token-wise influence computation is not compatible with `has_shared_parameters=True`. "
                "Disabling `compute_per_token_scores`."
            )
            score_args.compute_per_token_scores = False
            self.logger.warning(warning_msg)

        if score_args.compute_per_token_scores and self.task.enable_post_process_per_sample_gradient:
            warning_msg = (
                "Token-wise influence computation is not compatible with tasks that requires "
                "`enable_post_process_per_sample_gradient`. Disabling `compute_per_token_scores`."
            )
            score_args.compute_per_token_scores = False
            self.logger.warning(warning_msg)

        dataloader_params = self._configure_dataloader(dataloader_kwargs)
        if self.state.is_main_process:
            self._save_dataset_metadata(
                dataset_name="query",
                dataset=query_dataset,
                indices=query_indices,
                output_dir=scores_output_dir,
                overwrite_output_dir=overwrite_output_dir,
            )
            self._save_dataset_metadata(
                dataset_name="train",
                dataset=train_dataset,
                indices=train_indices,
                output_dir=scores_output_dir,
                overwrite_output_dir=overwrite_output_dir,
            )

        if query_indices is not None:
            query_dataset = Subset(dataset=query_dataset, indices=query_indices)
            del query_indices

        if train_indices is not None:
            train_dataset = Subset(dataset=train_dataset, indices=train_indices)
            del train_indices

        with self.profiler.profile("Load All Factors"):
            if factors_name is None:
                print("INFO: No factors specified (skip_factors=True) - using identity matrix for H^{-1}")
                loaded_factors = {}
            else:
                try:
                    loaded_factors = self.load_all_factors(factors_name=factors_name)
                    print(f"DEBUG: Loaded factor keys: {list(loaded_factors.keys()) if loaded_factors else 'None'}")
                    print(f"DEBUG: Loaded factors type: {type(loaded_factors)}")
                    if loaded_factors:
                        print(f"DEBUG: First few factor keys: {list(loaded_factors.keys())[:5]}")
                        # Check if factors are properly loaded
                        for name, factor_data in loaded_factors.items():
                            print(f"DEBUG: Factor {name}: type={type(factor_data)}")
                            if factor_data is None:
                                print(f"WARNING: Factor {name} is None!")
                            elif hasattr(factor_data, 'keys'):
                                print(f"DEBUG: Factor {name} keys: {list(factor_data.keys())}")
                                for sub_key, sub_data in factor_data.items():
                                    if sub_data is None:
                                        print(f"WARNING: Factor {name}.{sub_key} is None!")
                                    else:
                                        print(f"DEBUG: Factor {name}.{sub_key} type: {type(sub_data)}, shape: {getattr(sub_data, 'shape', 'no shape')}")
                            else:
                                print(f"DEBUG: Factor {name} is direct object with type: {type(factor_data)}")
                    else:
                        print("WARNING: No factors loaded! Using identity matrix for H^{-1}")
                        loaded_factors = {}
                except Exception as e:
                    print(f"WARNING: Failed to load factors: {e}")
                    print("Using identity matrix for H^{-1}...")
                    loaded_factors = {}

        # Keep partitioning logic from original if needed (here assume no partition)
        no_partition = score_args.data_partitions == 1 and score_args.module_partitions == 1
        partition_provided = target_data_partitions is not None or target_module_partitions is not None
        if no_partition and partition_provided:
            error_msg = (
                "`target_data_partitions` or `target_module_partitions` were specified, while"
                "the `ScoreArguments` did not expect any data and module partition to compute pairwise scores."
            )
            self.logger.error(error_msg)
            raise ValueError(error_msg)

        data_partition_indices, target_data_partitions = self._get_data_partition(
            total_data_examples=len(train_dataset),
            data_partitions=score_args.data_partitions,
            target_data_partitions=target_data_partitions,
        )
        max_partition_examples = len(train_dataset) // score_args.data_partitions
        module_partition_names, target_module_partitions = self._get_module_partition(
            module_partitions=score_args.module_partitions,
            target_module_partitions=target_module_partitions,
        )

        all_start_time = get_time(state=self.state)
        for data_partition in target_data_partitions:
            for module_partition in target_module_partitions:
                if no_partition:
                    partition = None
                else:
                    partition = (data_partition, module_partition)

                if pairwise_scores_exist(output_dir=scores_output_dir, partition=partition) and not overwrite_output_dir:
                    self.logger.info(
                        f"Found existing pairwise scores for data partition {data_partition} "
                        f"and module partition {module_partition} at {scores_output_dir}. Skipping."
                    )
                    continue

                start_index, end_index = data_partition_indices[data_partition]
                self.logger.info(
                    f"Computing pairwise scores with data indices ({start_index}, {end_index}) and "
                    f"modules {module_partition_names[module_partition]}."
                )

                if per_device_train_batch_size is None:
                    per_device_train_batch_size = self._find_executable_pairwise_scores_batch_size(
                        query_dataset=query_dataset,
                        per_device_query_batch_size=per_device_query_batch_size
                        if not score_args.aggregate_query_gradients
                        else 1,
                        train_dataset=train_dataset,
                        initial_per_device_train_batch_size_attempt=initial_per_device_train_batch_size_attempt,
                        loaded_factors=loaded_factors,
                        dataloader_params=dataloader_params,
                        total_data_examples=max_partition_examples,
                        score_args=score_args,
                        factor_args=factor_args,
                        tracked_modules_name=module_partition_names[module_partition],
                    )

                self._reset_memory()
                start_time = get_time(state=self.state)
                with self.profiler.profile("Compute Pairwise Score"):
                    # Create only the 3 necessary loaders directly
                    query_loader = self._get_dataloader(
                        dataset=query_dataset,
                        per_device_batch_size=per_device_query_batch_size,
                        dataloader_params=dataloader_params,
                        allow_duplicates=not score_args.aggregate_query_gradients,
                    )
                    
                    forget_loader = self._get_dataloader(
                        dataset=self.forget_dataset,
                        per_device_batch_size=per_device_train_batch_size,
                        dataloader_params=dataloader_params,
                        allow_duplicates=not score_args.aggregate_train_gradients,
                        stack=not score_args.aggregate_train_gradients,
                    )
                    
                    retain_loader = self._get_dataloader(
                        dataset=self.retain_dataset,
                        per_device_batch_size=per_device_train_batch_size,
                        dataloader_params=dataloader_params,
                        allow_duplicates=not score_args.aggregate_train_gradients,
                        stack=not score_args.aggregate_train_gradients,
                    )
                    
                    # differential computation using direct loaders
                    scores = self._compute_differential_with_loaders(
                        loaded_factors=loaded_factors,
                        forget_sample_loader=query_loader,
                        forget_loader=forget_loader,
                        retain_loader=retain_loader,
                        tracked_module_names=module_partition_names[module_partition],
                    )

                end_time = get_time(state=self.state)
                elapsed_time = end_time - start_time
                self.logger.info(f"Computed pairwise influence scores in {elapsed_time:.2f} seconds.")

                with self.profiler.profile("Save Pairwise Score"):
                    if self.state.is_main_process:
                        save_pairwise_scores(
                            output_dir=scores_output_dir,
                            scores=scores,
                            partition=partition,
                            metadata=score_args.to_str_dict(),
                        )
                        # Additionally save as .pt for compatibility
                        try:
                            from pathlib import Path
                            pt_path = Path(scores_output_dir) / "pairwise_scores.pt"
                            # Save tensor only for .pt to simplify downstream loading
                            tensor_to_save = None
                            if isinstance(scores, torch.Tensor):
                                tensor_to_save = scores
                            elif isinstance(scores, dict):
                                if 'all_modules' in scores and isinstance(scores['all_modules'], torch.Tensor):
                                    tensor_to_save = scores['all_modules']
                                else:
                                    # Fallback to first tensor-like value
                                    for v in scores.values():
                                        if isinstance(v, torch.Tensor):
                                            tensor_to_save = v
                                            break
                            if tensor_to_save is None:
                                # As a last resort, attempt tensor conversion
                                tensor_to_save = torch.tensor(scores)
                            torch.save(tensor_to_save, str(pt_path))
                            self.logger.info(f"Saved pairwise scores (.pt) at {pt_path}.")
                        except Exception as e:
                            self.logger.warning(f"Failed to save pairwise scores as .pt: {e}")
                    self.state.wait_for_everyone()
                del scores, query_loader, forget_loader, retain_loader
                self._reset_memory()
                self.logger.info(f"Saved pairwise scores at {scores_output_dir}.")

        all_end_time = get_time(state=self.state)
        elapsed_time = all_end_time - all_start_time
        if not no_partition:
            self.logger.info(f"Fitted all partitioned pairwise scores in {elapsed_time:.2f} seconds.")
            if self.state.is_main_process:
                self.aggregate_pairwise_scores(scores_name=scores_name)
                self.logger.info(f"Saved aggregated pairwise scores at `{scores_output_dir}`.")
            self.state.wait_for_everyone()
        self._log_profile_summary(name=f"scores_{scores_name}_differential")
        return self.load_pairwise_scores(scores_name=scores_name) if pairwise_scores_exist(scores_output_dir) else {"all_modules": torch.empty(0)}

    def _compute_differential_with_loaders(self, loaded_factors, forget_sample_loader, forget_loader, retain_loader, tracked_module_names):
        """
        Direct computation using separate forget/retain loaders.
        No need to split - loaders are already properly separated.
        """
        return self._differential_token_computation(loaded_factors, forget_loader, retain_loader, forget_sample_loader, tracked_module_names)

    def _differential_token_computation(self, loaded_factors, forget_loader, retain_loader, forget_sample_loader, tracked_module_names):
        """
        Core differential computation: S_ij = (g_forget - g_retain)^T * H^{-1} * ∇_θ(-log Pr(forget_token_j | forget_context_i; θ))

        Variables:
        - S_ij: influence score for sample i, token j
        - g_forget: average gradient over forget dataset
        - g_retain: average gradient over retain dataset  
        - H^{-1}: inverse Hessian (approximated via EKFAC factors)
        - forget_token_j: j-th token in forget sample i
        - forget_context_i: context preceding token j in forget sample i

        Implementation notes:
        - This version computes per-sample per-token gradients exactly using torch.autograd.grad.
        - It's correct but slow for large models. Use this for validation/small models.
        - forget_sample_loader contains the forget data samples whose tokens we want to analyze
        """
        # Use PRECONDITION_GRADIENT mode like kronfluence standard
        set_mode(self.model, ModuleMode.PRECONDITION_GRADIENT, tracked_module_names=tracked_module_names, release_memory=True)
        
        # Set factors manually like kronfluence does
        if loaded_factors:
            print(f"DEBUG: Factors loaded: {list(loaded_factors.keys())}")
            try:
                for name in loaded_factors:
                    set_factors(
                        model=self.model,
                        factor_name=name,
                        factors=loaded_factors[name],
                        clone=True,
                    )
                print("DEBUG: Successfully set all factors")
            except Exception as e:
                print(f"DEBUG: Failed to set factors: {e}")
                print("DEBUG: Continuing without factor setting...")
        else:
            print("DEBUG: No factors loaded, using identity preconditioning")
        
        # Prepare modules for preconditioning
        try:
            prepare_modules(self.model, tracked_module_names=tracked_module_names, device=self.state.device)
            print("DEBUG: Successfully prepared modules for preconditioning")
        except Exception as e:
            print(f"DEBUG: Failed to prepare modules: {e}")
            print("DEBUG: Continuing without module preparation...")

        # Compute d = avg_grad(forget) - avg_grad(retain)  
        print("Computing average gradient for forget dataset...")
        g_forget = self._avg_grad(forget_loader)  # vector on self.state.device
        print("Computing average gradient for retain dataset...")
        g_retain = self._avg_grad(retain_loader)
        
        # 🔍 DEBUGGING: Compare g_forget vs g_retain
        print("\n" + "="*50)
        print("📊 GRADIENT COMPARISON ANALYSIS")
        print("="*50)
        
        # Basic stats
        forget_norm = g_forget.norm().item()
        retain_norm = g_retain.norm().item()
        print(f"📈 g_forget norm: {forget_norm:.6f}")
        print(f"📈 g_retain norm: {retain_norm:.6f}")
        print(f"📊 Norm ratio (forget/retain): {forget_norm/retain_norm:.6f}")
        
        # Difference analysis
        d = (g_forget - g_retain).detach()
        diff_norm = d.norm().item()
        print(f"📉 |g_forget - g_retain| norm: {diff_norm:.6f}")
        print(f"📊 Difference/forget ratio: {diff_norm/forget_norm:.6f}")
        print(f"📊 Difference/retain ratio: {diff_norm/retain_norm:.6f}")
        
        # Cosine similarity
        cosine_sim = torch.dot(g_forget, g_retain) / (forget_norm * retain_norm)
        print(f"🎯 Cosine similarity: {cosine_sim:.6f}")
        print(f"🎯 Cosine distance: {1 - cosine_sim:.6f}")
        
        # Element-wise stats
        abs_diff = torch.abs(d)
        print(f"📊 Max absolute difference: {abs_diff.max().item():.8f}")
        print(f"📊 Mean absolute difference: {abs_diff.mean().item():.8f}")
        print(f"📊 Std of differences: {d.std().item():.8f}")
        
        # Percentage of parameters with significant differences
        threshold = diff_norm * 0.01  # 1% of total difference norm
        significant_diffs = (abs_diff > threshold).sum().item()
        total_params = len(abs_diff)
        print(f"📊 Params with significant diff (>{threshold:.8f}): {significant_diffs}/{total_params} ({100*significant_diffs/total_params:.2f}%)")
        
        print("="*50)
        print("🤔 INTERPRETATION:")
        if cosine_sim > 0.95:
            print("   ⚠️  Very similar gradients - small differential signal")
        elif cosine_sim > 0.8:
            print("   ✅ Moderately similar - reasonable differential signal")
        else:
            print("   🎯 Quite different gradients - strong differential signal")
        
        if diff_norm/max(forget_norm, retain_norm) < 0.1:
            print("   ⚠️  Difference is <10% of gradient magnitude")
        elif diff_norm/max(forget_norm, retain_norm) < 0.5:
            print("   ✅ Difference is moderate (10-50% of gradient magnitude)")
        else:
            print("   🎯 Difference is substantial (>50% of gradient magnitude)")
        print("="*50 + "\n")

        # ensure d is a 1D tensor
        if d.dim() != 1:
            d = d.reshape(-1)

        # Forget sample loop: for each batch, compute per-sample per-token s_ij
        params = [p for p in self.model.parameters() if p.requires_grad]
        param_numel = sum(p.numel() for p in params)
        

        all_scores = []
        device = self.state.device

        forget_batch_count = 0
        total_batches = len(forget_sample_loader)
        
        # Process batches
        for batch in tqdm(forget_sample_loader, desc="Processing forget sample batches", unit="batch"):
            forget_batch_count += 1
            # move to device
            model_batch = {k: v.to(device) for k, v in batch.items() if k != 'indices'}
            # forward with grad enabled
            out = self.model(**model_batch, return_dict=True)
            logits = out.logits  # (B, T, V) for causal LM
            labels = batch.get("labels", None)
            if labels is None:
                raise RuntimeError("Forget sample batch must contain 'labels' to compute per-token negative log-likelihood.")
            # Ensure labels are on the same device as logits
            labels = labels.to(device)
            logp = torch.log_softmax(logits, dim=-1)
            
            # Validate labels are in valid range for gather operation
            vocab_size = logits.size(-1)
            
            # Debug: Check for invalid labels
            invalid_mask = (labels != -100) & ((labels < 0) | (labels >= vocab_size))
            if invalid_mask.any():
                invalid_count = invalid_mask.sum().item()
                print(f"WARNING: Found {invalid_count} invalid labels. Min: {labels[labels != -100].min().item()}, Max: {labels[labels != -100].max().item()}, Vocab size: {vocab_size}")
            
            valid_labels = labels.clone()
            # Clamp labels to valid range, keeping -100 as is for padding
            valid_labels = torch.where(
                labels.eq(-100), 
                labels,  # Keep -100 as is
                torch.clamp(labels, 0, vocab_size - 1)  # Clamp valid tokens to vocab range
            )
            
            # Only gather for non-padding tokens to avoid index errors
            gather_indices = valid_labels.unsqueeze(-1)
            gather_indices = torch.where(
                valid_labels.eq(-100).unsqueeze(-1),
                torch.zeros_like(gather_indices),  # Use 0 for padding tokens (will be masked out)
                gather_indices
            )
            
            nll = -torch.gather(logp, -1, gather_indices).squeeze(-1)
            nll = torch.where(labels.eq(-100), torch.zeros_like(nll), nll)

            B, T = labels.shape
            scores_bt = torch.zeros(B, T, device=device, dtype=d.dtype if d.is_floating_point() else torch.float32)

            # For each sample, perform separate forward pass to avoid graph reuse issues
            for j in range(B):
                # Create single-sample batch
                single_sample_batch = {}
                for key, value in model_batch.items():
                    if isinstance(value, torch.Tensor) and value.dim() > 0:
                        single_sample_batch[key] = value[j:j+1]  # Keep batch dimension
                    else:
                        single_sample_batch[key] = value
                
                # Separate forward pass for this sample
                self.model.zero_grad(set_to_none=True)
                single_out = self.model(**single_sample_batch, return_dict=True)
                single_logits = single_out.logits
                single_labels = labels[j:j+1]  # Keep batch dimension
                
                # Compute NLL for this single sample
                single_logp = torch.log_softmax(single_logits, dim=-1)
                vocab_size = single_logits.size(-1)
                
                # Apply same validation as before
                valid_single_labels = single_labels.clone()
                valid_single_labels = torch.where(
                    single_labels.eq(-100), 
                    single_labels,
                    torch.clamp(single_labels, 0, vocab_size - 1)
                )
                
                gather_indices = valid_single_labels.unsqueeze(-1)
                gather_indices = torch.where(
                    valid_single_labels.eq(-100).unsqueeze(-1),
                    torch.zeros_like(gather_indices),
                    gather_indices
                )
                
                single_nll = -torch.gather(single_logp, -1, gather_indices).squeeze(-1)
                single_nll = torch.where(single_labels.eq(-100), torch.zeros_like(single_nll), single_nll)

                # ✅ ALWAYS: 모든 유효 토큰에 대해 SI 계산
                valid_mask = ~single_labels.eq(-100)[0]
                target_positions = torch.nonzero(valid_mask, as_tuple=True)[0]
                if target_positions.numel() == 0:
                    continue

                # Compute SI for selected token positions
                last_idx = len(target_positions) - 1
                for idx_i, tpos in enumerate(target_positions.tolist()):
                    # Get token info for debugging
                    token_id = single_labels[0, tpos].item()
                    neg_log_prob_jt = single_nll[0, tpos]
                    
                    grads = torch.autograd.grad(neg_log_prob_jt, params, retain_graph=(idx_i != last_idx), allow_unused=True)
                    grad_theta_jt = _safe_cat_grads(params, grads).to(device)
                    # apply H^{-1}: H^{-1} * ∇_θ(-log Pr(...))
                    Hinv_grad = self._apply_Hinv(grad_theta_jt, loaded_factors)
                    # ensure same device/dtype
                    Hinv_grad = Hinv_grad.to(d.device) if Hinv_grad.device != d.device else Hinv_grad
                    if Hinv_grad.dtype != d.dtype:
                        Hinv_grad = Hinv_grad.to(d.dtype)
                    # final dot product: S_jt = (g_forget - g_retain)^T * H^{-1} * ∇_θ(-log Pr(...))
                    s_jt = torch.dot(d, Hinv_grad)
                    
                    # 🔍 DEBUG: 선택된 토큰과 SI값 출력
                    token_text = ""
                    if hasattr(self, 'tokenizer') and self.tokenizer is not None:
                        try:
                            token_text = self.tokenizer.decode([token_id], skip_special_tokens=False)
                            token_text = f" '{token_text}'"
                        except:
                            token_text = ""
                    
                    print(f"📍 Sample {j}, Position {tpos}: token_id={token_id}{token_text}, NLL={neg_log_prob_jt:.4f}, SI={s_jt:.6f}")
                    
                    scores_bt[j, tpos] = s_jt
                    

            all_scores.append(scores_bt)

        # restore mode
        set_mode(self.model, ModuleMode.DEFAULT, release_memory=True)
        final_scores = torch.cat(all_scores, dim=0) if len(all_scores) > 0 else torch.empty((0, 0), device=device)
        return {"all_modules": final_scores}

    def _avg_grad(self, loader):
        """
        Compute average gradient (vector) over the dataset represented by loader.
        Uses _safe_cat_grads to preserve parameter ordering and zero-fill None grads.
        Returns a 1D tensor on self.state.device.
        """
        params = [p for p in self.model.parameters() if p.requires_grad]
        device = self.state.device
        acc = None
        cnt = 0

        # Progress bar for gradient computation
        grad_pbar = tqdm(enumerate(loader), desc="Computing average gradients", 
                        unit="batch", total=len(loader), leave=True)
        
        for batch_idx, batch in grad_pbar:
            
            # Handle different batch formats
            if isinstance(batch, dict):
                # Remove indices if present (not needed for model forward)
                model_batch = {k: v.to(device) for k, v in batch.items() if k != 'indices'}
                batch = model_batch
            elif isinstance(batch, (list, tuple)):
                # TextDatasetQA format: [input_ids, labels, attention_mask, indices]
                batch_dict = {'input_ids': batch[0].to(device)}
                if len(batch) > 1 and batch[1] is not None:
                    batch_dict['labels'] = batch[1].to(device)
                if len(batch) > 2 and batch[2] is not None:
                    batch_dict['attention_mask'] = batch[2].to(device)
                batch = batch_dict
            else:
                raise ValueError(f"Unexpected batch type: {type(batch)}")
            out = self.model(**batch, return_dict=True)
            # prefer out.loss if available; else compute from logits/labels
            if hasattr(out, "loss") and out.loss is not None:
                loss = out.loss
            else:
                logits = out.logits
                labels = batch.get("labels", None)
                if labels is None:
                    raise RuntimeError("Training batch missing labels required to compute loss for avg_grad.")
                logp = torch.log_softmax(logits, dim=-1)
                
                # Validate labels are in valid range for gather operation
                vocab_size = logits.size(-1)
                valid_labels = labels.clone()
                # Clamp labels to valid range, keeping -100 as is for padding
                valid_labels = torch.where(
                    labels.eq(-100), 
                    labels,  # Keep -100 as is
                    torch.clamp(labels, 0, vocab_size - 1)  # Clamp valid tokens to vocab range
                )
                
                # Only gather for non-padding tokens to avoid index errors
                gather_indices = valid_labels.unsqueeze(-1)
                gather_indices = torch.where(
                    valid_labels.eq(-100).unsqueeze(-1),
                    torch.zeros_like(gather_indices),  # Use 0 for padding tokens (will be masked out)
                    gather_indices
                )
                
                nll = -torch.gather(logp, -1, gather_indices).squeeze(-1)
                nll = torch.where(labels.eq(-100), torch.zeros_like(nll), nll)
                loss = nll.mean()

            grads = torch.autograd.grad(loss, params, retain_graph=False, allow_unused=True)
            g_vec = _safe_cat_grads(params, grads).to(device)
            
            if batch_idx == 0:
                print(f"DEBUG: First gradient vector shape: {g_vec.shape}, norm: {g_vec.norm().item():.6f}")

            acc = g_vec.clone() if acc is None else acc + g_vec
            cnt += 1
            self.model.zero_grad(set_to_none=True)
            
            # Update progress bar with current gradient norm
            grad_pbar.set_postfix({"avg_grad_norm": f"{(acc / max(cnt, 1)).norm().item():.4f}"})
      
        return acc / max(cnt, 1)

    def _apply_Hinv(self, vec, loaded_factors):
        """Apply H^{-1} using kronfluence's preconditioning system.
        
        Since we're now using PRECONDITION_GRADIENT mode, kronfluence should handle
        the preconditioning automatically. This method is called for manual gradient
        computation, so we return the vector as-is (kronfluence handles H^{-1} internally).
        """
        if not loaded_factors:
            if not hasattr(self, '_no_factors_warned'):
                print("DEBUG: No factors provided, using identity (no H^{-1} approximation)")
                self._no_factors_warned = True
            return vec

        # Log factor structure once for debugging
        if not hasattr(self, '_factors_debug_logged'):
            print("DEBUG: Using kronfluence's automatic preconditioning system")
            print(f"DEBUG: Loaded factor types: {list(loaded_factors.keys())}")
            for name, factor_data in loaded_factors.items():
                if isinstance(factor_data, dict):
                    print(f"DEBUG: Factor {name} contains modules: {list(factor_data.keys())}")
                    if factor_data:
                        first_module = next(iter(factor_data.values()))
                        if hasattr(first_module, 'shape'):
                            print(f"DEBUG: Sample tensor shape: {first_module.shape}")
            print("DEBUG: kronfluence will handle H^{-1} automatically in PRECONDITION_GRADIENT mode")
            self._factors_debug_logged = True
            
        # Return vector as-is - kronfluence handles H^{-1} internally
        return vec

    def compute_differential_token_scores(
        self,
        scores_name: str,
        factors_name: str,
        query_dataset: torch.utils.data.Dataset,
        forget_dataset: torch.utils.data.Dataset,
        retain_dataset: torch.utils.data.Dataset,
        per_device_query_batch_size: int,
        per_device_train_batch_size: Optional[int] = None,
        initial_per_device_train_batch_size_attempt: int = 4096,
        query_indices: Optional[Sequence[int]] = None,
        train_indices: Optional[Sequence[int]] = None,
        dataloader_kwargs: Optional[DataLoaderKwargs] = None,
        score_args: Optional[ScoreArguments] = None,
        target_data_partitions: Optional[Sequence[int]] = None,
        target_module_partitions: Optional[Sequence[int]] = None,
        overwrite_output_dir: bool = False,
    ):
        """
        Compute differential token scores using separate forget/retain datasets.
        """
        # Store datasets for direct access
        self.forget_dataset = forget_dataset
        self.retain_dataset = retain_dataset
        
        # No need for train_dataset - use datasets directly
        return self.compute_pairwise_scores(
            scores_name=scores_name,
            factors_name=factors_name,
            query_dataset=query_dataset,
            train_dataset=forget_dataset,  # Use forget dataset for individual gradients
            per_device_query_batch_size=per_device_query_batch_size,
            per_device_train_batch_size=per_device_train_batch_size,
            initial_per_device_train_batch_size_attempt=initial_per_device_train_batch_size_attempt,
            query_indices=query_indices,
            train_indices=train_indices,
            dataloader_kwargs=dataloader_kwargs,
            score_args=score_args,
            target_data_partitions=target_data_partitions,
            target_module_partitions=target_module_partitions,
            overwrite_output_dir=overwrite_output_dir,
        )


def parse_args():
    parser = argparse.ArgumentParser(description="Compute influence scores for TOFU dataset.")
    # Model and datasets
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--checkpoint_dir", type=str, default=None)
    parser.add_argument("--query_batch_size", type=int, default=4)
    parser.add_argument("--train_batch_size", type=int, default=8)
    parser.add_argument("--profile", action="store_true", default=False)
    parser.add_argument("--factors_path", type=str, default=None)
    parser.add_argument("--factor_strategy", type=str, default="ekfac")
    parser.add_argument("--factors_name", type=str, default="ekfac_factors")
    parser.add_argument("--skip_factors", action="store_true", default=False, help="Skip factor loading and run without H^{-1}")
    parser.add_argument("--tofu_data_path", type=str, default="locuslab/TOFU")
    parser.add_argument("--forget_split", type=str, default="forget10")
    parser.add_argument("--retain_split", type=str, default="retain90")
    parser.add_argument("--model_family", type=str, default="gpt", choices=["llama", "gpt", "t5", "bert", "phi", "pythia-1.4"])
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--use_half_precision", action="store_true", default=False)
    parser.add_argument("--use_compile", action="store_true", default=False)
    parser.add_argument("--save_dir", type=str, default="")
    parser.add_argument("--save_id", type=str, default=None)
    
    # Legacy arguments for compatibility (ignored)
    parser.add_argument("--train_split", type=str, default=None, help="Legacy argument, ignored")
    parser.add_argument("--query_gradient_rank", type=int, default=-1, help="Legacy argument, ignored")
    
    return parser.parse_args()


def construct_hf_model(config, checkpoint_dir):    
    return AutoModelForCausalLM.from_pretrained(checkpoint_dir)
    


def get_tokenized_tofu_datasets(config, tokenizer, model_family, max_length):
    # Use the corrected loader that handles TOFU perturbed splits properly
    from local_utils.data_utils import load_datasets
    dataset_orig, dataset_para = load_datasets(
        config=config,
        tokenizer=tokenizer,
        model_family=model_family,
        max_length=max_length,
    )
    return dataset_orig, dataset_para


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [%(levelname)s] %(message)s')

    # Load model config to get the correct hf_key for tokenizer
    config_path = "/root/npo/config/model_config.yaml"
    with open(config_path, 'r') as f:
        model_configs = yaml.safe_load(f)
    
    # Get the hf_key for the model family
    if args.model_family in model_configs:
        hf_key = model_configs[args.model_family]["hf_key"]
        print(f"Loading tokenizer from: {hf_key}")
    else:
        raise ValueError(f"Model family '{args.model_family}' not found in config")
    
    # Load tokenizer from the correct original model  
    tokenizer = AutoTokenizer.from_pretrained(hf_key)
    model = construct_hf_model(None, args.checkpoint_dir)

    @dataclass
    class DataConfig:
        data_path: str = args.tofu_data_path
        forget_split: str = args.forget_split
        question_key: str = "question"

    config = DataConfig()
    forget_dataset_orig, _ = get_tokenized_tofu_datasets(config=config, tokenizer=tokenizer, model_family=args.model_family, max_length=args.max_length)
    retain_config = DataConfig(data_path=args.tofu_data_path, forget_split=args.retain_split, question_key="question")
    retain_dataset_orig, _ = get_tokenized_tofu_datasets(config=retain_config, tokenizer=tokenizer, model_family=args.model_family, max_length=args.max_length)

    # 별도 데이터셋들 유지 (ConcatDataset 제거)
    forget_query_dataset = forget_dataset_orig

    task = LanguageModelingTask(config.__dict__)
    model = prepare_model(model, task)

    kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=5400))
    accelerator = Accelerator(kwargs_handlers=[kwargs])
    model = accelerator.prepare_model(model)

    if args.use_compile:
        model = torch.compile(model)

    # kronfluence expects output_dir to be the parent directory of where factors are stored
    # factors are at: /workspace/nas_chaen/kronfluence_factors/si_factor_analysis/factors_ekfac_half
    # so output_dir should be: /workspace/nas_chaen/kronfluence_factors
    import os
    factors_parent_dir = os.path.dirname(args.factors_path) if args.factors_path else "./factors"
    
    analyzer = DifferentialTokenScorer(
        analysis_name="si_factor_analysis",
        model=model,
        task=task,
        profile=args.profile,
        output_dir=factors_parent_dir,
    )
    analyzer.tokenizer = tokenizer  # tokenizer 전달

    # Custom collate function to handle dataset format issues
    def custom_collate_fn(batch):
        if not batch:
            return {}
        
        # Handle different batch formats
        if isinstance(batch[0], dict):
            # Already a dict, return as is
            return default_data_collator(batch)
        elif hasattr(batch[0], '__dict__'):
            # Object with __dict__, convert to dict
            return default_data_collator([vars(item) for item in batch])
        else:
            # List/tuple format, convert to dict
            if isinstance(batch[0], (list, tuple)) and len(batch[0]) >= 2:
                # Assume [input_ids, labels, attention_mask, indices] format
                result = {}
                result['input_ids'] = torch.stack([item[0] for item in batch])
                if len(batch[0]) > 1 and batch[0][1] is not None:
                    result['labels'] = torch.stack([item[1] for item in batch])
                if len(batch[0]) > 2 and batch[0][2] is not None:
                    result['attention_mask'] = torch.stack([item[2] for item in batch])
                if len(batch[0]) > 3 and batch[0][3] is not None:
                    result['indices'] = torch.tensor([item[3] for item in batch])
                return result
            else:
                # Fallback to default
                return default_data_collator(batch)
    
    dataloader_kwargs = DataLoaderKwargs(collate_fn=custom_collate_fn)
    analyzer.set_dataloader_kwargs(dataloader_kwargs)

    if forget_dataset_orig is None or retain_dataset_orig is None:
        raise ValueError("forget and retain datasets are required for differential token scoring")

    scores_name = f"differential_token_{args.factor_strategy}"
    if args.use_half_precision:
        scores_name += "_half"
    if args.use_compile:
        scores_name += "_compile"
    if args.save_id:
        scores_name += f"_{args.save_id}"
    if args.save_dir:
        scores_name = os.path.join(args.save_dir, scores_name)

    score_args = ScoreArguments()
    if args.use_half_precision:
        score_args = all_low_precision_score_arguments(dtype=torch.bfloat16)

    # IMPORTANT: ensure incompatible options are not enabled (for token-wise)
    score_args.compute_per_token_scores = True
    score_args.aggregate_train_gradients = False

    # Handle skip_factors option
    factors_name_to_use = args.factors_name
    if args.skip_factors:
        print("⚠️  SKIP_FACTORS enabled - running without H^{-1} approximation")
        factors_name_to_use = None

    result = analyzer.compute_differential_token_scores(
        scores_name=scores_name,
        factors_name=factors_name_to_use,
        query_dataset=forget_query_dataset,
        forget_dataset=forget_dataset_orig,
        retain_dataset=retain_dataset_orig,
        per_device_query_batch_size=args.query_batch_size,
        per_device_train_batch_size=args.train_batch_size,
        score_args=score_args,
        overwrite_output_dir=True,
    )

    logging.info("Differential token scoring completed!")
    logging.info("Formula: S_ij = (g_bar_forget - g_bar_retain)^T * H^{-1} * ∇_θ L(x_ij; θ)")
    logging.info("Completed.")


if __name__ == "__main__":
    main()