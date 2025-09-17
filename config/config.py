"""
Configuration settings for unlearning analysis
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
    split: str = "full"  # TOFU 기본 split
    question_key: str = "question"
    question_key_paraphrase: str = "question"  # TOFU dataset only has 'question' field
    forget_split: str = "forget10"  # TOFU 실제 split 이름
    retain_split: str = "retain90"  # TOFU 실제 split 이름


@dataclass
class TrainingConfig:
    """언러닝 훈련 관련 설정"""
    forget_lr: float = 1e-5
    forget_epochs: int = 5
    forget_batch_size: int = 2
    forget_grad_accum: int = 4
    forget_loss: str = "ce"  # "ce", "kld", "uld", "npo"
    forget_beta: float = 0.5
    forget_coeff: float = 1.0
    forget_reg_coeff: float = 0.0
    
    # NPO 특화 파라미터
    kl_coeff: float = 0.1      # KL regularization coefficient
    clip_ratio: float = 0.2    # NPO clipping ratio
    
    # DeepSpeed 관련 설정
    use_deepspeed: bool = False
    use_deepspeed_stage3: bool = False  # Stage 3 for maximum memory efficiency


@dataclass
class BeamSearchConfig:
    """빔서치 패러프레이징 설정"""
    model_name: str = "/root/task_vector/tofu/files/models/ToFU/checkpoint-625"
    num_beams: int = 5
    num_return_sequences: int = 5
    max_length: int = 512
    length_penalty: float = 1.0
    repetition_penalty: float = 1.2
    temperature: float = 0.7
    do_sample: bool = True
    top_k: int = 50
    top_p: float = 0.95
    early_stopping: bool = True
    no_repeat_ngram_size: int = 2


@dataclass
class AnalysisConfig:
    """분석 관련 설정"""
    batch_size: int = 1  # SI 분석시 메모리 절약을 위해 1로 유지
    output_file: str = "memorization_scores.csv"
    run_unlearning: bool = False
    significant_shift_threshold: float = 0.1
    save_visualizations: bool = True
    visualization_dpi: int = 300
    
    # 패러프레이징 관련 설정 (간소화)
    num_paraphrases: int = 3  # 생성할 패러프레이즈 수
    
    # Self-Influence 분석 설정
    run_si_analysis: bool = True  # SI 분석 기본 활성화
    use_deepspeed_for_si: bool = False  # SI 계산에 DeepSpeed 사용 여부 (메모리 절약을 위해 기본값은 False)
    
    # Kronfluence 설정 (간소화)
    enable_kronfluence: bool = True  # Kronfluence 팩터 계산 활성화
    influence_output_dir: str = "influence_factors"
    train_batch_size: int = 4  # Factor 계산용 배치 크기
    
    # 기존 코드와의 호환성을 위한 기본값들
    enable_cluster_analysis: bool = True
    enable_paraphrasing: bool = False  
    enable_prompt_paraphrasing: bool = False
    beam_search: Optional[object] = None  # 삭제된 BeamSearchConfig 호환성
    prompt_paraphrase_templates_file: str = ""
    prompt_paraphrase_count: int = 5
    prompt_paraphrase_max_tokens: int = 100
    prompt_paraphrase_temperature: float = 0.8
    
    
    def get(self, key, default=None):
        """dict-like access for backward compatibility"""
        return getattr(self, key, default)


@dataclass
class UnlearningAnalysisConfig:
    """전체 설정을 담는 클래스"""
    model: ModelConfig
    data: DataConfig
    training: TrainingConfig
    analysis: AnalysisConfig


def parse_args() -> UnlearningAnalysisConfig:
    """명령행 인수를 파싱하여 설정 객체 반환"""
    parser = argparse.ArgumentParser(description="Unlearning Analysis Tool")
    
    # Model arguments
    parser.add_argument("--model_name", type=str, required=True, help="Model name or path")
    parser.add_argument("--model_family", type=str, required=True, help="Model family (e.g., llama2-7b)")
    parser.add_argument("--max_length", type=int, default=256, help="Maximum sequence length")
    parser.add_argument("--max_new_tokens", type=int, default=256, help="Maximum new tokens to generate")
    
    # Data arguments
    parser.add_argument("--data_path", type=str, default="locuslab/TOFU", help="TOFU dataset name (default: locuslab/TOFU)")
    parser.add_argument("--split", type=str, default="full", help="Data split to use (full for TOFU)")
    parser.add_argument("--question_key", type=str, default="question", help="Key for original questions")
    parser.add_argument("--question_key_paraphrase", type=str, default="question", help="Key for paraphrased questions")
    parser.add_argument("--forget_split", type=str, default="forget10", help="TOFU forget split (forget01/05/10)")
    parser.add_argument("--retain_split", type=str, default="retain90", help="TOFU retain split (retain99/95/90)")
    
    # Training arguments
    parser.add_argument("--forget_lr", type=float, default=3e-5, help="Learning rate for forgetting")
    parser.add_argument("--forget_epochs", type=int, default=1, help="Number of forgetting epochs")
    parser.add_argument("--forget_batch_size", type=int, default=2, help="Batch size for forgetting")
    parser.add_argument("--forget_grad_accum", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--forget_loss", type=str, default="ce", choices=["ce", "grad_ascent", "grad_diff", "kld", "uld", "npo", "dpo"], help="Loss type for forgetting")
    parser.add_argument("--kl_coeff", type=float, default=0.1, help="KL regularization coefficient for NPO")
    parser.add_argument("--clip_ratio", type=float, default=0.2, help="NPO clipping ratio")
    parser.add_argument("--forget_beta", type=float, default=0.5, help="Beta parameter for forgetting")
    parser.add_argument("--forget_coeff", type=float, default=1.0, help="Forgetting coefficient")
    parser.add_argument("--forget_reg_coeff", type=float, default=0.0, help="Regularization coefficient")
    
    # Analysis arguments
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for analysis")
    parser.add_argument("--output_file", type=str, default="memorization_scores.csv", help="Output file path")
    parser.add_argument("--run_unlearning", action="store_true", help="Whether to run unlearning")
    parser.add_argument("--run_si_analysis", action="store_true", help="Whether to run Self-Influence analysis")
    parser.add_argument("--significant_shift_threshold", type=float, default=0.1, help="Threshold for significant embedding shifts")
    
    # Kronfluence arguments (간소화)
    parser.add_argument("--enable_kronfluence", action="store_true", help="Enable Kronfluence-based influence calculation")
    parser.add_argument("--influence_output_dir", type=str, default="influence_factors", help="Output directory for influence factors")
    
    # 패러프레이징 arguments (간소화)  
    parser.add_argument("--num_paraphrases", type=int, default=3, help="Number of paraphrases to generate")
    
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
    
    training_config = TrainingConfig(
        forget_lr=args.forget_lr,
        forget_epochs=args.forget_epochs,
        forget_batch_size=args.forget_batch_size,
        forget_grad_accum=args.forget_grad_accum,
        forget_loss=args.forget_loss,
        forget_beta=args.forget_beta,
        forget_coeff=args.forget_coeff,
        forget_reg_coeff=args.forget_reg_coeff,
        kl_coeff=args.kl_coeff,
        clip_ratio=args.clip_ratio
    )
    
    analysis_config = AnalysisConfig(
        batch_size=args.batch_size,
        output_file=args.output_file,
        run_unlearning=args.run_unlearning,
        run_si_analysis=args.run_si_analysis,
        significant_shift_threshold=args.significant_shift_threshold,
        num_paraphrases=args.num_paraphrases,
        enable_kronfluence=args.enable_kronfluence,
        influence_output_dir=args.influence_output_dir
    )
    
    return UnlearningAnalysisConfig(
        model=model_config,
        data=data_config,
        training=training_config,
        analysis=analysis_config
    )