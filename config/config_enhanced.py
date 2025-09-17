"""
Enhanced configuration with smart dataset type detection
"""

import argparse
from dataclasses import dataclass, field
from typing import Optional


def detect_dataset_type(split_name: str) -> str:
    """데이터셋 split 이름으로부터 타입 자동 감지"""
    if 'perturbed' in split_name.lower():
        return 'perturbed'
    elif any(x in split_name.lower() for x in ['forget', 'retain']):
        return 'standard'
    else:
        return 'unknown'


@dataclass
class ModelConfig:
    """모델 관련 설정"""
    model_name: str
    model_family: str
    max_length: int = 256
    max_new_tokens: int = 256
    device_map: str = "auto"
    torch_dtype: str = "float16"


@dataclass
class EnhancedDataConfig:
    """Enhanced 데이터 관련 설정 - 자동 데이터셋 타입 감지"""
    data_path: str
    split: str = "full"
    question_key: str = "question"
    question_key_paraphrase: Optional[str] = None  # 기본값을 None으로 설정
    forget_split: str = "forget10"
    retain_split: str = "retain90"
    
    # 자동 감지 필드
    dataset_type: str = field(init=False)
    supports_paraphrase: bool = field(init=False)
    
    def __post_init__(self):
        """초기화 후 자동으로 데이터셋 타입 설정"""
        self.dataset_type = detect_dataset_type(self.split)
        
        if self.dataset_type == 'perturbed':
            self.supports_paraphrase = True
            # perturbed 데이터셋인 경우 기본 paraphrase key 설정
            if self.question_key_paraphrase is None:
                self.question_key_paraphrase = "paraphrased_question"
        else:
            self.supports_paraphrase = False
            # 표준 데이터셋인 경우 paraphrase key를 None으로 설정
            self.question_key_paraphrase = None
        
        print(f"🔍 Auto-detected dataset type: {self.dataset_type}")
        print(f"📊 Supports paraphrase: {self.supports_paraphrase}")
        if self.question_key_paraphrase:
            print(f"🔗 Paraphrase key: {self.question_key_paraphrase}")


@dataclass
class TrainingConfig:
    """언러닝 훈련 관련 설정"""
    forget_lr: float = 3e-5
    forget_epochs: int = 1
    forget_batch_size: int = 2
    forget_grad_accum: int = 4
    forget_loss: str = "ce"
    forget_beta: float = 0.5
    forget_coeff: float = 1.0
    forget_reg_coeff: float = 0.0


@dataclass
class AnalysisConfig:
    """분석 관련 설정"""
    batch_size: int = 1
    output_file: str = "memorization_scores.csv"
    run_unlearning: bool = False
    significant_shift_threshold: float = 0.1
    save_visualizations: bool = True
    visualization_dpi: int = 300
    
    # 분석 모드 설정
    analysis_mode: str = "smart"  # "smart", "force_both", "original_only"


@dataclass
class EnhancedUnlearningAnalysisConfig:
    """Enhanced 전체 설정을 담는 클래스"""
    model: ModelConfig
    data: EnhancedDataConfig
    training: TrainingConfig
    analysis: AnalysisConfig


def parse_enhanced_args() -> EnhancedUnlearningAnalysisConfig:
    """Enhanced 명령행 인수를 파싱하여 설정 객체 반환"""
    parser = argparse.ArgumentParser(description="Enhanced Unlearning Analysis Tool")
    
    # Model arguments
    parser.add_argument("--model_name", type=str, required=True, help="Model name or path")
    parser.add_argument("--model_family", type=str, required=True, help="Model family (e.g., llama2-7b)")
    parser.add_argument("--max_length", type=int, default=256, help="Maximum sequence length")
    parser.add_argument("--max_new_tokens", type=int, default=256, help="Maximum new tokens to generate")
    
    # Data arguments
    parser.add_argument("--data_path", type=str, default="locuslab/TOFU", help="TOFU dataset name")
    parser.add_argument("--split", type=str, default="forget10", help="Data split to use")
    parser.add_argument("--question_key", type=str, default="question", help="Key for original questions")
    parser.add_argument("--question_key_paraphrase", type=str, default=None, 
                        help="Key for paraphrased questions (auto-detected if not specified)")
    parser.add_argument("--forget_split", type=str, default="forget10", help="TOFU forget split")
    parser.add_argument("--retain_split", type=str, default="retain90", help="TOFU retain split")
    
    # Training arguments
    parser.add_argument("--forget_lr", type=float, default=3e-5, help="Learning rate for forgetting")
    parser.add_argument("--forget_epochs", type=int, default=1, help="Number of forgetting epochs")
    parser.add_argument("--forget_batch_size", type=int, default=2, help="Batch size for forgetting")
    parser.add_argument("--forget_grad_accum", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--forget_loss", type=str, default="ce", help="Loss type for forgetting")
    parser.add_argument("--forget_beta", type=float, default=0.5, help="Beta parameter for forgetting")
    parser.add_argument("--forget_coeff", type=float, default=1.0, help="Forgetting coefficient")
    parser.add_argument("--forget_reg_coeff", type=float, default=0.0, help="Regularization coefficient")
    
    # Analysis arguments
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for analysis")
    parser.add_argument("--output_file", type=str, default="memorization_scores.csv", help="Output file path")
    parser.add_argument("--run_unlearning", action="store_true", help="Whether to run unlearning")
    parser.add_argument("--significant_shift_threshold", type=float, default=0.1, 
                        help="Threshold for significant embedding shifts")
    parser.add_argument("--analysis_mode", type=str, default="smart", 
                        choices=["smart", "force_both", "original_only"],
                        help="Analysis mode: smart (auto-detect), force_both (always compare), original_only")
    
    args = parser.parse_args()
    
    # 설정 객체 생성
    model_config = ModelConfig(
        model_name=args.model_name,
        model_family=args.model_family,
        max_length=args.max_length,
        max_new_tokens=args.max_new_tokens
    )
    
    data_config = EnhancedDataConfig(
        data_path=args.data_path,
        split=args.split,
        question_key=args.question_key,
        question_key_paraphrase=args.question_key_paraphrase,
        forget_split=args.forget_split,
        retain_split=args.retain_split
    )
    
    training_config = TrainingConfig(
        forget_lr=args.forget_lr,
        forget_epochs=args.forget_epochs,
        forget_batch_size=args.forget_batch_size,
        forget_grad_accum=args.forget_grad_accum,
        forget_loss=args.forget_loss,
        forget_beta=args.forget_beta,
        forget_coeff=args.forget_coeff,
        forget_reg_coeff=args.forget_reg_coeff
    )
    
    analysis_config = AnalysisConfig(
        batch_size=args.batch_size,
        output_file=args.output_file,
        run_unlearning=args.run_unlearning,
        significant_shift_threshold=args.significant_shift_threshold,
        analysis_mode=args.analysis_mode
    )
    
    return EnhancedUnlearningAnalysisConfig(
        model=model_config,
        data=data_config,
        training=training_config,
        analysis=analysis_config
    )


# 기존 호환성을 위한 alias
DataConfig = EnhancedDataConfig
UnlearningAnalysisConfig = EnhancedUnlearningAnalysisConfig
parse_args = parse_enhanced_args