"""
NPO(Negative Preference Optimization) 전용 설정
"""

import argparse
from dataclasses import dataclass
from typing import Optional


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
class DataConfig:
    """데이터 관련 설정"""
    data_path: str
    split: str = "full"
    question_key: str = "question"
    question_key_paraphrase: str = "paraphrased_question"
    forget_split: str = "forget10"
    retain_split: str = "retain90"


@dataclass
class NPOTrainingConfig:
    """NPO 전용 훈련 설정"""
    # 기본 훈련 설정
    forget_lr: float = 1e-5
    forget_epochs: int = 3
    forget_batch_size: int = 4
    forget_grad_accum: int = 2
    
    # NPO 특화 설정
    forget_loss: str = "npo"  # "npo", "kld", "uld", "ce"
    forget_beta: float = 0.5
    forget_coeff: float = 1.0
    kl_coeff: float = 0.1  # KL regularization
    clip_ratio: float = 0.2  # NPO clipping ratio
    
    # 정규화
    forget_reg_coeff: float = 0.01
    
    # Reference model 설정
    reference_model_path: Optional[str] = None  # None이면 original model 사용
    
    # 평가 설정
    eval_strategy: str = "steps"
    eval_steps: int = 100
    save_strategy: str = "steps"
    save_steps: int = 200
    
    # 데이터 균형
    balance_forget_retain: bool = True
    max_samples_per_type: Optional[int] = None


@dataclass
class AnalysisConfig:
    """분석 관련 설정"""
    batch_size: int = 1
    output_file: str = "npo_analysis.csv"
    run_unlearning: bool = True  # NPO는 기본적으로 unlearning 수행
    significant_shift_threshold: float = 0.1
    save_visualizations: bool = True
    visualization_dpi: int = 300
    
    # NPO 평가 특화
    evaluate_retain_performance: bool = True
    evaluate_forget_effectiveness: bool = True
    compute_preference_scores: bool = True


@dataclass
class NPOUnlearningAnalysisConfig:
    """NPO 전체 설정"""
    model: ModelConfig
    data: DataConfig
    training: NPOTrainingConfig
    analysis: AnalysisConfig


def parse_npo_args() -> NPOUnlearningAnalysisConfig:
    """NPO 명령행 인수 파싱"""
    parser = argparse.ArgumentParser(description="NPO Unlearning Analysis Tool")
    
    # Model arguments
    parser.add_argument("--model_name", type=str, required=True, help="Model name or path")
    parser.add_argument("--model_family", type=str, required=True, help="Model family")
    parser.add_argument("--max_length", type=int, default=256, help="Maximum sequence length")
    parser.add_argument("--max_new_tokens", type=int, default=256, help="Maximum new tokens")
    
    # Data arguments
    parser.add_argument("--data_path", type=str, default="locuslab/TOFU", help="Dataset path")
    parser.add_argument("--split", type=str, default="full", help="Data split")
    parser.add_argument("--question_key", type=str, default="question", help="Question key")
    parser.add_argument("--question_key_paraphrase", type=str, default="paraphrased_question", help="Paraphrase key")
    parser.add_argument("--forget_split", type=str, default="forget10", help="Forget split")
    parser.add_argument("--retain_split", type=str, default="retain90", help="Retain split")
    
    # NPO Training arguments
    parser.add_argument("--forget_lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--forget_epochs", type=int, default=3, help="Training epochs")
    parser.add_argument("--forget_batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--forget_grad_accum", type=int, default=2, help="Gradient accumulation")
    parser.add_argument("--forget_loss", type=str, default="npo", choices=["npo", "kld", "uld", "ce"], help="Loss type")
    parser.add_argument("--forget_beta", type=float, default=0.5, help="Beta parameter")
    parser.add_argument("--forget_coeff", type=float, default=1.0, help="Forget coefficient")
    parser.add_argument("--kl_coeff", type=float, default=0.1, help="KL regularization coefficient")
    parser.add_argument("--clip_ratio", type=float, default=0.2, help="NPO clipping ratio")
    parser.add_argument("--forget_reg_coeff", type=float, default=0.01, help="Regularization coefficient")
    parser.add_argument("--reference_model_path", type=str, default=None, help="Reference model path")
    parser.add_argument("--balance_forget_retain", action="store_true", help="Balance forget/retain data")
    parser.add_argument("--max_samples_per_type", type=int, default=None, help="Max samples per type")
    
    # Analysis arguments
    parser.add_argument("--batch_size", type=int, default=1, help="Analysis batch size")
    parser.add_argument("--output_file", type=str, default="npo_analysis.csv", help="Output file")
    parser.add_argument("--run_unlearning", action="store_true", default=True, help="Run unlearning")
    parser.add_argument("--significant_shift_threshold", type=float, default=0.1, help="Shift threshold")
    parser.add_argument("--evaluate_retain_performance", action="store_true", default=True, help="Evaluate retain performance")
    parser.add_argument("--evaluate_forget_effectiveness", action="store_true", default=True, help="Evaluate forget effectiveness")
    parser.add_argument("--compute_preference_scores", action="store_true", default=True, help="Compute preference scores")
    
    args = parser.parse_args()
    
    # 설정 객체 생성
    model_config = ModelConfig(
        model_name=args.model_name,
        model_family=args.model_family,
        max_length=args.max_length,
        max_new_tokens=args.max_new_tokens
    )
    
    data_config = DataConfig(
        data_path=args.data_path,
        split=args.split,
        question_key=args.question_key,
        question_key_paraphrase=args.question_key_paraphrase,
        forget_split=args.forget_split,
        retain_split=args.retain_split
    )
    
    training_config = NPOTrainingConfig(
        forget_lr=args.forget_lr,
        forget_epochs=args.forget_epochs,
        forget_batch_size=args.forget_batch_size,
        forget_grad_accum=args.forget_grad_accum,
        forget_loss=args.forget_loss,
        forget_beta=args.forget_beta,
        forget_coeff=args.forget_coeff,
        kl_coeff=args.kl_coeff,
        clip_ratio=args.clip_ratio,
        forget_reg_coeff=args.forget_reg_coeff,
        reference_model_path=args.reference_model_path,
        balance_forget_retain=args.balance_forget_retain,
        max_samples_per_type=args.max_samples_per_type
    )
    
    analysis_config = AnalysisConfig(
        batch_size=args.batch_size,
        output_file=args.output_file,
        run_unlearning=args.run_unlearning,
        significant_shift_threshold=args.significant_shift_threshold,
        evaluate_retain_performance=args.evaluate_retain_performance,
        evaluate_forget_effectiveness=args.evaluate_forget_effectiveness,
        compute_preference_scores=args.compute_preference_scores
    )
    
    return NPOUnlearningAnalysisConfig(
        model=model_config,
        data=data_config,
        training=training_config,
        analysis=analysis_config
    )


def validate_npo_config(config: NPOUnlearningAnalysisConfig) -> bool:
    """NPO 설정 검증"""
    
    # NPO는 reference model이 필수
    if config.training.forget_loss == "npo":
        if config.training.reference_model_path is None:
            print("⚠️  NPO mode: Will use original model as reference")
        
        # Beta 값 검증
        if not (0.0 <= config.training.forget_beta <= 1.0):
            raise ValueError("NPO beta should be between 0.0 and 1.0")
        
        # Clip ratio 검증
        if not (0.0 < config.training.clip_ratio <= 1.0):
            raise ValueError("NPO clip_ratio should be between 0.0 and 1.0")
    
    # 데이터 균형 설정 검증
    if config.training.balance_forget_retain:
        print("✅ Will balance forget/retain data for stable NPO training")
    
    # 평가 설정 검증
    if not config.analysis.evaluate_retain_performance:
        print("⚠️  Not evaluating retain performance - this is not recommended for NPO")
    
    print("✅ NPO configuration validated")
    return True


# 사용 예시
if __name__ == "__main__":
    config = parse_npo_args()
    validate_npo_config(config)
    print("NPO Configuration loaded successfully!")