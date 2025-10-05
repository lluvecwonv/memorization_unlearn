"""
Kronfluence factor fitting script with embed_out diagonal approximation
"""

import argparse
import logging
import os
import sys
import torch
import time
from pathlib import Path
from transformers import default_data_collator
from datetime import timedelta
from accelerate import Accelerator, InitProcessGroupKwargs, DistributedDataParallelKwargs
from peft import PeftModel

# Kronfluence imports
from kronfluence.analyzer import Analyzer, prepare_model
from kronfluence.arguments import FactorArguments
from kronfluence.utils.common.factor_arguments import all_low_precision_factor_arguments
from kronfluence.utils.dataset import DataLoaderKwargs

from typing import Optional
from tqdm import tqdm
from torch import nn

# Configure CUDA memory
os.environ.setdefault(
    "PYTORCH_CUDA_ALLOC_CONF",
    "max_split_size_mb:128,garbage_collection_threshold:0.8,expandable_segments:True",
)

# Add current directory and parent directory to path for local imports
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.insert(0, script_dir)
sys.path.insert(0, parent_dir)

# Import your existing utilities
from local_utils.logger import setup_logging
from local_utils.model_utils import load_model_and_tokenizer
from local_utils.data_utils import load_datasets

# Import local analysis module
from local_utils.kronfluence_task import LanguageModelingTask


def perform_eigendecomposition(
    self,
    factors_name: str,
    factor_args: Optional[FactorArguments] = None,
    overwrite_output_dir: bool = False,
    load_from_factors_name: Optional[str] = None,
) -> None:
    """Performs eigendecomposition with embed_out diagonal approximation."""
    from kronfluence.utils.constants import FACTOR_ARGUMENTS_NAME
    from kronfluence.factor.covariance import load_covariance_matrices
    from kronfluence.factor.eigen import save_eigendecomposition, eigendecomposition_exist
    from kronfluence.utils.exceptions import FactorsNotFoundError
    from kronfluence.factor.config import FactorConfig
    
    factors_output_dir = self.factors_output_dir(factors_name=factors_name)
    os.makedirs(factors_output_dir, exist_ok=True)
    
    if eigendecomposition_exist(output_dir=factors_output_dir) and not overwrite_output_dir:
        self.logger.info(f"Found existing eigendecomposition results at `{factors_output_dir}`. Skipping.")
        return

    factor_args = self._configure_and_save_factor_args(
        factor_args=factor_args, 
        factors_output_dir=factors_output_dir, 
        overwrite_output_dir=overwrite_output_dir
    )

    if not FactorConfig.CONFIGS[factor_args.strategy].requires_eigendecomposition:
        self.logger.info(f"Strategy `{factor_args.strategy}` does not require eigendecomposition. Skipping.")
        return

    load_factors_output_dir = factors_output_dir
    if load_from_factors_name is not None:
        load_factors_output_dir = self.factors_output_dir(factors_name=load_from_factors_name)

    # Load covariance matrices
    with self.profiler.profile("Load Covariance"):
        covariance_factors = load_covariance_matrices(output_dir=load_factors_output_dir)

    self._reset_memory()
    if self.state.is_main_process:
        start_time = time.time()
        with self.profiler.profile("Perform Eigendecomposition"):
            eigen_factors = simple_eigendecomposition(
                covariance_factors=covariance_factors,
                model=self.model,
                state=self.state,
                factor_args=factor_args,
                disable_tqdm=self.disable_tqdm,
            )
        
        elapsed_time = time.time() - start_time
        self.logger.info(f"Performed eigendecomposition in {elapsed_time:.2f} seconds.")

        with self.profiler.profile("Save Eigendecomposition"):
            save_eigendecomposition(
                output_dir=factors_output_dir, 
                factors=eigen_factors, 
                metadata=factor_args.to_str_dict()
            )
        self.logger.info(f"Saved eigendecomposition results at `{factors_output_dir}`.")
        del eigen_factors
        self._reset_memory()
    self.state.wait_for_everyone()


@torch.no_grad()
def simple_eigendecomposition(
    covariance_factors,
    model: nn.Module,
    state,
    factor_args: FactorArguments,
    disable_tqdm: bool = False,
):
    """Simple eigendecomposition with embed_out diagonal approximation."""
    from kronfluence.utils.constants import (
        EIGENDECOMPOSITION_FACTOR_NAMES, ACTIVATION_COVARIANCE_MATRIX_NAME,
        GRADIENT_COVARIANCE_MATRIX_NAME, NUM_ACTIVATION_COVARIANCE_PROCESSED,
        NUM_GRADIENT_COVARIANCE_PROCESSED, ACTIVATION_EIGENVECTORS_NAME,
        ACTIVATION_EIGENVALUES_NAME, GRADIENT_EIGENVECTORS_NAME,
        GRADIENT_EIGENVALUES_NAME
    )
    try:
        from kronfluence.utils.constants import TQDM_BAR_FORMAT
    except ImportError:
        TQDM_BAR_FORMAT = "{l_bar}{bar:10}{r_bar}"
    from kronfluence.module.utils import get_tracked_module_names
    from accelerate.utils.memory import should_reduce_batch_size
    from kronfluence.utils.state import release_memory

    eigen_factors = {name: {} for name in EIGENDECOMPOSITION_FACTOR_NAMES}
    tracked_module_names = get_tracked_module_names(model=model)

    with tqdm(
        total=len(tracked_module_names),
        desc="Performing Eigendecomposition",
        bar_format=TQDM_BAR_FORMAT,
        disable=not state.is_main_process or disable_tqdm,
    ) as pbar:
        for module_name in tracked_module_names:
            for covariance_name, num_processed_name, eigenvectors_name, eigenvalues_name in [
                (ACTIVATION_COVARIANCE_MATRIX_NAME, NUM_ACTIVATION_COVARIANCE_PROCESSED,
                 ACTIVATION_EIGENVECTORS_NAME, ACTIVATION_EIGENVALUES_NAME),
                (GRADIENT_COVARIANCE_MATRIX_NAME, NUM_GRADIENT_COVARIANCE_PROCESSED,
                 GRADIENT_EIGENVECTORS_NAME, GRADIENT_EIGENVALUES_NAME),
            ]:
                if (covariance_name not in covariance_factors or 
                    module_name not in covariance_factors[covariance_name]):
                    continue

                matrix = covariance_factors[covariance_name][module_name]
                n = matrix.size(-1)

                # 🎯 Diagonal approximation for embed_out (핵심 코드!)
                if module_name == 'embed_out' and n >= 16384:
                    print(f"  🎯 Using DIAGONAL approximation for {module_name} (n={n})")
                    
                    # 정규화
                    matrix = matrix.to('cpu', dtype=torch.float32)
                    matrix.div_(covariance_factors[num_processed_name][module_name].to('cpu'))
                    matrix = matrix + matrix.t()
                    matrix.mul_(0.5)
                    
                    # 대각선만 추출
                    diag = torch.diag(matrix)
                    eigenvalues = diag.contiguous()
                    eigenvectors = torch.eye(n, dtype=torch.float32)
                    
                    # 저장
                    eigen_factors[eigenvalues_name][module_name] = eigenvalues
                    eigen_factors[eigenvectors_name][module_name] = eigenvectors
                    
                    del matrix, diag, eigenvalues, eigenvectors
                    print(f"  ✅ Diagonal approximation completed for {module_name}")
                    continue

                # 일반 모듈 처리
                original_dtype = matrix.dtype
                covariance_matrix = matrix.to(
                    device=state.device,
                    dtype=factor_args.eigendecomposition_dtype,
                )
                covariance_matrix.div_(
                    covariance_factors[num_processed_name][module_name].to(device=state.device)
                )
                covariance_matrix = covariance_matrix + covariance_matrix.t()
                covariance_matrix.mul_(0.5)
                
                try:
                    eigenvalues, eigenvectors = torch.linalg.eigh(covariance_matrix)
                except Exception as e:
                    if should_reduce_batch_size(exception=e):
                        release_memory()
                        eigenvalues, eigenvectors = torch.linalg.eigh(covariance_matrix)
                    else:
                        raise
                
                del covariance_matrix
                eigen_factors[eigenvalues_name][module_name] = eigenvalues.contiguous().to(
                    dtype=original_dtype, device='cpu'
                )
                eigen_factors[eigenvectors_name][module_name] = eigenvectors.contiguous().to(
                    dtype=original_dtype, device='cpu'
                )
                del eigenvalues, eigenvectors
            pbar.update(1)
    
    return eigen_factors


def kronfluence_data_collator(batch):
    """Custom data collator for Kronfluence."""
    input_ids = []
    labels = []
    attention_mask = []
    
    for item in batch:
        if isinstance(item, (tuple, list)) and len(item) >= 3:
            input_ids.append(item[0])
            labels.append(item[1])
            attention_mask.append(item[2])
        else:
            raise ValueError(f"Expected tuple/list with at least 3 items, got {type(item)}")
    
    return {
        'input_ids': torch.stack(input_ids),
        'labels': torch.stack(labels),
        'attention_mask': torch.stack(attention_mask)
    }


def parse_factor_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Fit Kronfluence factors with embed_out diagonal approximation")
    
    # Model arguments  
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--model_family", type=str, required=True)
    parser.add_argument("--max_length", type=int, default=256)
    
    # Data arguments
    parser.add_argument("--data_path", type=str, default="locuslab/TOFU")
    parser.add_argument("--split", type=str, default="full")
    parser.add_argument("--question_key", type=str, default="question")
    parser.add_argument("--forget_split", type=str, default="forget10")
    parser.add_argument("--retain_split", type=str, default="retain90")
    
    # Factor computation
    parser.add_argument("--factor_strategy", type=str, default="ekfac", choices=["ekfac", "kfac", "diagonal"])
    parser.add_argument("--use_half_precision", action="store_true", default=False)
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--output_dir", type=str, default="./kronfluence_factors")
    
    # Partition arguments (for backward compatibility)
    parser.add_argument("--covariance_module_partitions", type=int, default=4)
    parser.add_argument("--lambda_module_partitions", type=int, default=4)
    parser.add_argument("--covariance_data_partitions", type=int, default=4)
    parser.add_argument("--lambda_data_partitions", type=int, default=4)
    
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_factor_args()
    
    # Create config objects
    from config.config import ModelConfig, DataConfig
    
    model_config = ModelConfig(
        model_name=args.model_name,
        model_family=args.model_family, 
        max_length=args.max_length,
        max_new_tokens=256
    )
    
    data_config = DataConfig(
        data_path=args.data_path,
        split=args.split,
        question_key=args.question_key,
        question_key_paraphrase=args.question_key,
        forget_split=args.forget_split,
        retain_split=args.retain_split
    )
    
    # Setup
    logger = setup_logging()
    logger.info("🚀 Starting Kronfluence factor fitting with embed_out diagonal approximation")
    
    # Load model and data
    model, tokenizer = load_model_and_tokenizer(model_config)
    dataset_orig, _ = load_datasets(data_config, tokenizer, model_config.model_family, model_config.max_length)
    
    task = LanguageModelingTask(config={"model_family": model_config.model_family})
    model = prepare_model(model, task)
    
    # Setup accelerator
    init_kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=5400))
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(kwargs_handlers=[init_kwargs, ddp_kwargs])
    model = accelerator.prepare_model(model)
    
    os.makedirs(args.output_dir, exist_ok=True)

    # Create analyzer
    analyzer = Analyzer(
        analysis_name="si_factor_analysis",
        model=model,
        task=task,
        output_dir=args.output_dir,
    )
    
    dataloader_kwargs = DataLoaderKwargs(
        num_workers=0,
        collate_fn=kronfluence_data_collator,
        pin_memory=False
    )
    analyzer.set_dataloader_kwargs(dataloader_kwargs)
    
    # Configure factors
    factor_args = FactorArguments(strategy=args.factor_strategy)
    factors_name = args.factor_strategy
    
    if args.use_half_precision:
        factor_args = all_low_precision_factor_arguments(strategy=args.factor_strategy, dtype=torch.bfloat16)
        factors_name += "_half"
    
    # Set partition parameters
    factor_args.covariance_module_partitions = args.covariance_module_partitions
    factor_args.lambda_module_partitions = args.lambda_module_partitions
    factor_args.covariance_data_partitions = args.covariance_data_partitions
    factor_args.lambda_data_partitions = args.lambda_data_partitions
    
    # Fit factors
    logger.info("🚀 Fitting covariance matrices...")
    analyzer.fit_covariance_matrices(
        factors_name=factors_name,
        dataset=dataset_orig,
        per_device_batch_size=args.train_batch_size,
        factor_args=factor_args,
        overwrite_output_dir=False,
    )
    
    logger.info("🚀 Performing eigendecomposition with embed_out diagonal approximation...")
    # Use our custom eigendecomposition
    analyzer.perform_eigendecomposition = perform_eigendecomposition.__get__(analyzer, Analyzer)
    analyzer.perform_eigendecomposition(
        factors_name=factors_name,
        factor_args=factor_args,
        overwrite_output_dir=True,
    )

    logger.info("🚀 Fitting lambda matrices...")
    analyzer.fit_lambda_matrices(
        factors_name=factors_name,
        dataset=dataset_orig,
        per_device_batch_size=args.train_batch_size,
        factor_args=factor_args,
        overwrite_output_dir=True
    )
    
    logger.info("✅ Factor fitting completed!")


if __name__ == "__main__":
    main()