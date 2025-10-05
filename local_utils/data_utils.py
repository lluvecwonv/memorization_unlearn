"""
데이터 처리 관련 유틸리티
"""

import csv
import os
from typing import Dict, List, Any
from torch.utils.data import DataLoader

import sys
import os

# TOFU 프로젝트 루트를 맨 앞에 추가하여 우선순위 높임
sys.path.insert(0, '/root/task_vector/tofu')
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from local_utils.data_module import TextDatasetQA, custom_data_collator_with_indices, convert_raw_data_to_model_format
from config.config import DataConfig, AnalysisConfig


def load_datasets(config: DataConfig, tokenizer, model_family: str, max_length: int):
    """데이터셋 로드 - TOFU split 타입에 따라 원본/패러프레이즈 분리"""
    
    from datasets import load_dataset

    # ✅ 1. 원본 질문 데이터셋 로드
    try:
        dataset_orig_raw = load_dataset(config.data_path, config.forget_split)
        sample = dataset_orig_raw['train'][0] if 'train' in dataset_orig_raw else dataset_orig_raw[0]
        available_fields_orig = list(sample.keys())
        base_split_name = getattr(config, 'forget_split', getattr(config, 'split', 'unknown_split'))
        print(f"� Available fields in {base_split_name}: {available_fields_orig}")
    except Exception as e:
        print(f"⚠️ Could not check original dataset structure: {e}")
        available_fields_orig = ['question', 'answer']

    # ✅ 2. 패러프레이즈 질문 데이터셋 로드 (split_perturbed)
    #    - forget 계열: forgetXX_perturbed
    #    - retain 계열: retain_perturbed (retain90_perturbed 등은 존재하지 않음)
    if str(config.forget_split).startswith("retain"):
        perturbed_split = "retain_perturbed"
    else:
        perturbed_split = config.forget_split + "_perturbed"
    print(f"🔍 Checking for perturbed split: {perturbed_split}")
    try:
        dataset_para_raw = load_dataset(config.data_path, perturbed_split)
        sample = dataset_para_raw['train'][0] if 'train' in dataset_para_raw else dataset_para_raw[0]
        available_fields_para = list(sample.keys())
        print(f"� Available fields in {perturbed_split}: {available_fields_para}")
    except Exception as e:
        # 폴백: retain 계열일 때는 retain_perturbed 재시도
        print(f"⚠️ Could not check paraphrased dataset structure: {e}")
        if not str(config.forget_split).startswith("retain"):
            available_fields_para = ['question', 'answer']
        else:
            try:
                perturbed_split = "retain_perturbed"
                print(f"🔁 Fallback to perturbed split: {perturbed_split}")
                dataset_para_raw = load_dataset(config.data_path, perturbed_split)
                sample = dataset_para_raw['train'][0] if 'train' in dataset_para_raw else dataset_para_raw[0]
                available_fields_para = list(sample.keys())
                print(f"� Available fields in {perturbed_split}: {available_fields_para}")
            except Exception as e2:
                print(f"⚠️ Fallback also failed: {e2}")
                available_fields_para = ['question', 'answer']

    # ✅ 3. 원본 질문용 데이터셋 클래스 생성
    dataset_orig = TextDatasetQA(
        config.data_path,
        tokenizer,
        model_family,
        max_length,
        config.forget_split,  # ✅ 원본 split 사용
        config.question_key,
    )

    # ✅ 4. 패러프레이즈 질문용 데이터셋 클래스 생성
    # paraphrased_question 필드 있으면 사용, 없으면 question
    if 'paraphrased_question' in available_fields_para:
        print("✅ Using paraphrased_question field for analysis")
        para_question_key = 'paraphrased_question'
    else:
        print("⚠️ paraphrased_question field not found in perturbed split, using question field instead")
        para_question_key = config.question_key

    dataset_para = TextDatasetQA(
        config.data_path,
        tokenizer,
        model_family,
        max_length,
        perturbed_split,  # ✅ perturbed split 사용
        para_question_key,
    )
    print(f"✅ Loaded original dataset: {config.forget_split} ({len(dataset_orig)} samples)")
    print(f"✅ Loaded paraphrased dataset: {perturbed_split} ({len(dataset_para)} samples)")
    
    # 원시 데이터도 함께 반환 (SI 분석용)
    return dataset_orig, dataset_para



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


def create_dataloaders(dataset_orig, dataset_para, batch_size: int):
    """데이터로더 생성"""
    
    dataloader_orig = DataLoader(
        dataset_orig,
        batch_size=batch_size,
        collate_fn=custom_data_collator_with_indices,
    )
    
    dataloader_para = DataLoader(
        dataset_para,
        batch_size=batch_size,
        collate_fn=custom_data_collator_with_indices,
    )
    
    return dataloader_orig, dataloader_para


def save_results_to_csv(results: List[Dict[str, Any]], output_path: str, tag: str = ""):
    """결과를 CSV 파일로 저장"""
    
    # 파일 경로 준비
    if tag:
        stage_file = output_path.replace(".csv", f"_{tag}.csv")
    else:
        stage_file = output_path
    
    # 디렉토리 생성
    stage_dir = os.path.dirname(stage_file)
    if stage_dir:  # 빈 문자열이 아닌 경우에만 생성
        os.makedirs(stage_dir, exist_ok=True)
    
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