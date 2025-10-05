"""
개선된 데이터 처리 유틸리티 - TOFU 데이터셋 구조에 맞게 수정
"""

import csv
import os
from typing import Dict, List, Any, Tuple, Optional
from torch.utils.data import DataLoader

from tofu.data_module import TextDatasetQA, custom_data_collator_with_indices
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from config.config import DataConfig, AnalysisConfig


def is_perturbed_split(split_name: str) -> bool:
    """split이 perturbed 데이터셋인지 확인"""
    return 'perturbed' in split_name.lower()


def validate_dataset_columns(config: DataConfig) -> Tuple[str, Optional[str]]:
    """
    데이터셋 split에 따라 사용 가능한 컬럼 검증 및 반환
    
    Returns:
        tuple: (question_key, paraphrased_question_key or None)
    """
    if is_perturbed_split(config.split):
        # Perturbed 데이터셋: 원본과 패러프레이즈 둘 다 가능
        return config.question_key, config.question_key_paraphrase
    else:
        # 표준 데이터셋: 원본 질문만 가능
        print(f"⚠️  Standard dataset '{config.split}' only has '{config.question_key}' column")
        return config.question_key, None


def load_datasets_smart(config: DataConfig, tokenizer, model_family: str, max_length: int):
    """
    데이터셋 유형에 따라 스마트하게 로드
    
    Returns:
        tuple: (dataset_orig, dataset_para) 
               표준 데이터셋의 경우 dataset_para는 None
    """
    question_key, paraphrase_key = validate_dataset_columns(config)
    
    # 원본 데이터셋은 항상 로드
    dataset_orig = TextDatasetQA(
        config.data_path,
        tokenizer,
        model_family,
        max_length,
        config.split,
        question_key,
    )
    print(f"✅ Loaded original dataset: {config.split} ({len(dataset_orig)} samples)")
    
    # 패러프레이즈 데이터셋은 가능한 경우에만 로드
    if paraphrase_key:
        try:
            dataset_para = TextDatasetQA(
                config.data_path,
                tokenizer,
                model_family,
                max_length,
                config.split,
                paraphrase_key,
            )
            print(f"✅ Loaded paraphrased dataset: {config.split} ({len(dataset_para)} samples)")
            return dataset_orig, dataset_para
        except KeyError as e:
            print(f"⚠️  Paraphrased column '{paraphrase_key}' not found: {e}")
            print("📝 Using original dataset only")
            return dataset_orig, None
    else:
        print("📝 No paraphrased version available for standard datasets")
        return dataset_orig, None


def load_datasets_with_fallback(config: DataConfig, tokenizer, model_family: str, max_length: int):
    """
    기존 API 호환성을 위한 fallback 버전
    표준 데이터셋에서는 동일한 데이터셋을 반환 (의미없는 비교 방지를 위해 경고)
    """
    question_key, paraphrase_key = validate_dataset_columns(config)
    
    dataset_orig = TextDatasetQA(
        config.data_path,
        tokenizer,
        model_family,
        max_length,
        config.split,
        question_key,
    )
    
    if paraphrase_key:
        # Perturbed 데이터셋: 실제 패러프레이즈 로드
        dataset_para = TextDatasetQA(
            config.data_path,
            tokenizer,
            model_family,
            max_length,
            config.split,
            paraphrase_key,
        )
        print(f"✅ Loaded both original and paraphrased datasets")
    else:
        # 표준 데이터셋: 같은 데이터셋 반환 (하지만 경고)
        dataset_para = dataset_orig
        print("⚠️  WARNING: Using same dataset for both orig and para - comparisons will be meaningless!")
        print("💡 Consider using load_datasets_smart() for better handling")
    
    return dataset_orig, dataset_para


def create_dataloaders_flexible(dataset_orig, dataset_para, batch_size: int):
    """
    dataset_para가 None일 수 있는 경우를 처리하는 데이터로더 생성
    """
    dataloader_orig = DataLoader(
        dataset_orig,
        batch_size=batch_size,
        collate_fn=custom_data_collator_with_indices,
    )
    
    if dataset_para is not None:
        dataloader_para = DataLoader(
            dataset_para,
            batch_size=batch_size,
            collate_fn=custom_data_collator_with_indices,
        )
        return dataloader_orig, dataloader_para
    else:
        print("📝 No paraphrased dataloader created")
        return dataloader_orig, None


def load_forget_retain_datasets(config: DataConfig, tokenizer, model_family: str, max_length: int):
    """TOFU 데이터셋의 실제 구조에 맞는 forget/retain 데이터셋 로드"""
    
    # TOFU 데이터셋의 실제 split 이름들
    tofu_splits = {
        'forget': ['forget01', 'forget05', 'forget10'], 
        'retain': ['retain99', 'retain95', 'retain90'],
        'perturbed': ['forget01_perturbed', 'forget05_perturbed', 'forget10_perturbed']
    }
    
    # config에서 지정된 split 확인 및 기본값 설정
    forget_split = getattr(config, 'forget_split', 'forget10')
    retain_split = getattr(config, 'retain_split', 'retain90')
    
    # forget/retain 매칭 확인 (forget10 -> retain90)
    forget_to_retain_map = {
        'forget01': 'retain99',
        'forget05': 'retain95', 
        'forget10': 'retain90'
    }
    
    if forget_split in forget_to_retain_map:
        retain_split = forget_to_retain_map[forget_split]
        print(f"✅ Using matched pair: {forget_split} <-> {retain_split}")
    
    try:
        # TOFU 데이터셋에서 forget 데이터 로드
        forget_dataset = TextDatasetQA(
            config.data_path,  # "locuslab/TOFU"
            tokenizer,
            model_family,
            max_length,
            forget_split,
            config.question_key,
        )
        
        # TOFU 데이터셋에서 retain 데이터 로드  
        retain_dataset = TextDatasetQA(
            config.data_path,  # "locuslab/TOFU"
            tokenizer,
            model_family,
            max_length,
            retain_split,
            config.question_key,
        )
        
        print(f"✅ Loaded TOFU forget dataset: {forget_split} ({len(forget_dataset)} samples)")
        print(f"✅ Loaded TOFU retain dataset: {retain_split} ({len(retain_dataset)} samples)")
        
        # 데이터셋 샘플 정보 출력
        if len(forget_dataset) > 0:
            sample = forget_dataset[0]
            print(f"📋 Forget sample keys: {list(sample.keys()) if hasattr(sample, 'keys') else 'tensor data'}")
            
    except Exception as e:
        print(f"❌ Error loading TOFU datasets: {e}")
        print("💡 Make sure you're using 'locuslab/TOFU' as data_path")
        print("💡 Available splits:", [s for split_list in tofu_splits.values() for s in split_list])
        raise
    
    return forget_dataset, retain_dataset


def save_results_to_csv(results: List[Dict[str, Any]], output_path: str, tag: str = ""):
    """결과를 CSV 파일로 저장"""
    
    # 파일 경로 준비
    if tag:
        stage_file = output_path.replace(".csv", f"_{tag}.csv")
    else:
        stage_file = output_path
    
    # 디렉토리 생성
    os.makedirs(os.path.dirname(stage_file), exist_ok=True)
    
    # CSV 저장
    with open(stage_file, "w", newline="", encoding="utf-8") as f:
        if results:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            for row in results:
                # 리스트 형태의 값들을 문자열로 변환
                processed_row = {}
                for key, value in row.items():
                    if isinstance(value, list):
                        processed_row[key] = ";".join([f"{v:.4f}" if isinstance(v, float) else str(v) for v in value])
                    elif isinstance(value, float):
                        processed_row[key] = f"{value:.4f}"
                    else:
                        processed_row[key] = str(value)
                writer.writerow(processed_row)
    
    return stage_file


# 기존 함수명 유지 (하위 호환성)
def load_datasets(config: DataConfig, tokenizer, model_family: str, max_length: int):
    """기존 API 호환을 위한 wrapper - load_datasets_with_fallback 사용"""
    return load_datasets_with_fallback(config, tokenizer, model_family, max_length)


def create_dataloaders(dataset_orig, dataset_para, batch_size: int):
    """기존 API 호환을 위한 wrapper - create_dataloaders_flexible 사용"""
    return create_dataloaders_flexible(dataset_orig, dataset_para, batch_size)