import torch
import torch.distributed as dist
import numpy as np
import os
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Tuple, List, Dict
import torch.nn.functional as F

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from config.config import ModelConfig


def load_model_and_tokenizer(config: ModelConfig) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Rank를 활용한 분산 모델과 토크나이저 로드"""
    
    # 분산 환경 초기화
    rank, world_size, local_rank = _initialize_distributed()
    
    # 토크나이저 로드
    if config.model_family == "llama2-7b":
        tokenizer = AutoTokenizer.from_pretrained("NousResearch/Llama-2-7b-chat-hf")
    elif config.model_family == "phi":
        tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-1_5")
    else:
        tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    
    # dtype 설정
    dtype = torch.float16 
    
    # 각 rank별로 다른 GPU에 모델 로드
    if world_size > 1:
        # 멀티 GPU 환경
        device_map = {"": local_rank}  # 각 프로세스가 자신의 local_rank GPU 사용
        
        if rank == 0:
            print(f"Loading model on {world_size} GPUs...")
        
        model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            device_map=device_map,
            torch_dtype=dtype,
            trust_remote_code=True,
            ignore_mismatched_sizes=True
        )
        
    else:
        # 단일 GPU 환경
        device_map = {"": 0} if config.device_map == "auto" else config.device_map
        
        model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            device_map=device_map,
            torch_dtype=dtype,
            trust_remote_code=True,
            ignore_mismatched_sizes=True
        )
    
    # 그래디언트 체크포인팅 활성화 (메모리 절약)
    # 주의: Kronfluence와 함께 사용시 호환성 문제가 있을 수 있음
    if hasattr(model, 'gradient_checkpointing_enable'):
        # 일시적으로 비활성화 - Kronfluence hook 등록 문제 해결
        # model.gradient_checkpointing_enable()
        if rank == 0:
            print("🔧 Gradient checkpointing temporarily disabled for Kronfluence compatibility")
    
    model.eval()
    
    # rank 0에서만 로그 출력 (중복 방지)
    if rank == 0:
        print(f"Model loaded successfully on rank {rank}/{world_size}")
    
    return model, tokenizer


def _initialize_distributed() -> Tuple[int, int, int]:
    """분산 환경 초기화 및 rank 정보 반환"""
    
    if not dist.is_initialized():
        # 환경변수에서 rank 정보 가져오기
        rank = int(os.environ.get('RANK', 0))
        world_size = int(os.environ.get('WORLD_SIZE', 1))
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        
        # 멀티 GPU 환경이면 분산 초기화
        if world_size > 1:
            dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
            torch.cuda.set_device(local_rank)
    else:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
    
    return rank, world_size, local_rank


def run_generation(
    batch: dict,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    config: ModelConfig
) -> Tuple[List[str], List[str], List[str]]:
    """텍스트 생성 실행"""
    
    input_ids = batch["input_ids"]
    input_strings = tokenizer.batch_decode(input_ids, skip_special_tokens=True)

    split_symbol = " [/INST]" if config.model_family == "llama2-7b" else "Answer: "
    ground_truth = [s.split(split_symbol)[1] if split_symbol in s else "" for s in input_strings]
    input_strings = [s.split(split_symbol)[0] for s in input_strings]

    if config.model_family == "llama2-7b":
        input_strings = [s + split_symbol for s in input_strings]

    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

    inputs = tokenizer.batch_encode_plus(
        input_strings,
        add_special_tokens=True,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=config.max_length,
    ).to(model.device)

    out = model.generate(
        inputs.input_ids,
        attention_mask=inputs.attention_mask,
        max_new_tokens=config.max_new_tokens,
        do_sample=False,
        temperature=None,
        top_p=None,
        use_cache=True,
        pad_token_id=tokenizer.eos_token_id,
    )

    generated = tokenizer.batch_decode(out[:, inputs.input_ids.shape[-1] :], skip_special_tokens=True)
    return input_strings, generated, ground_truth


def compute_generation_embedding(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    text: str
) -> torch.Tensor:
    """텍스트의 임베딩 계산 - 기본 함수"""
    with torch.no_grad():
        enc = tokenizer(text, return_tensors="pt").to(model.device)
        out = model(**enc, output_hidden_states=True)
        hidden = out.hidden_states[-1].squeeze(0)
        emb = hidden.mean(dim=0)
        return emb.float().cpu()


def compute_batch_embeddings(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    texts: List[str]
) -> List[torch.Tensor]:
    """여러 텍스트의 임베딩을 한번에 계산 - 배치 처리"""
    embeddings = []
    with torch.no_grad():
        for text in texts:
            emb = compute_generation_embedding(model, tokenizer, text)
            embeddings.append(emb)
    return embeddings


def compute_similarity_from_embeddings(
    emb1: torch.Tensor,
    emb2: torch.Tensor
) -> Tuple[float, float]:
    """미리 계산된 임베딩으로 유사도 계산"""
    # 코사인 유사도
    cos_sim = torch.cosine_similarity(
        emb1.unsqueeze(0), 
        emb2.unsqueeze(0), 
        dim=1
    ).item()
    
    # 유클리드 거리
    euc_dist = torch.norm(emb1 - emb2, p=2).item()
    
    return cos_sim, euc_dist


def compute_sentence_similarity(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    text1: str,
    text2: str
) -> Tuple[float, float]:
    """두 텍스트 간의 유사도 계산 - 하위 호환성을 위해 유지"""
    emb1 = compute_generation_embedding(model, tokenizer, text1)
    emb2 = compute_generation_embedding(model, tokenizer, text2)
    return compute_similarity_from_embeddings(emb1, emb2)


def compute_sentence_similarity_optimized(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    text1: str,
    text2: str
) -> Tuple[float, float, torch.Tensor, torch.Tensor]:
    """유사도 계산 + 임베딩 반환 (재사용을 위해)"""
    emb1 = compute_generation_embedding(model, tokenizer, text1)
    emb2 = compute_generation_embedding(model, tokenizer, text2)
    
    cos_sim, euc_dist = compute_similarity_from_embeddings(emb1, emb2)
    
    return cos_sim, euc_dist, emb1, emb2


def load_dual_models_and_tokenizer(config: ModelConfig) -> Tuple[AutoModelForCausalLM, AutoModelForCausalLM, AutoTokenizer]:
    """Load both full and retain models for dual evaluation"""
    
    # 분산 환경 초기화
    rank, world_size, local_rank = _initialize_distributed()
    
    # 토크나이저 로드
    if config.model_family == "llama2-7b":
        tokenizer = AutoTokenizer.from_pretrained("NousResearch/Llama-2-7b-chat-hf")
    elif config.model_family == "phi":
        tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-1_5")
    else:
        tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    
    # dtype 설정
    dtype = torch.float16 
    
    # Full model path
    full_model_path = "/root/tofu/files/models/ToFU_full_phi"
    # Retain model path (forget10 is the retain model)
    retain_model_path = "/root/tofu/files/models/ToFU_forget10"
    
    if world_size > 1:
        device_map = {"": local_rank}
    else:
        device_map = {"": 0} if config.device_map == "auto" else config.device_map
    
    # Load full model (IN_full)
    full_model = AutoModelForCausalLM.from_pretrained(
        full_model_path,
        device_map=device_map,
        torch_dtype=dtype
    )
    
    # Load retain model (OUT_s)
    retain_model = AutoModelForCausalLM.from_pretrained(
        retain_model_path,
        device_map=device_map,
        torch_dtype=dtype
    )
    
    full_model.eval()
    retain_model.eval()
    
    if rank == 0:
        print(f"Dual models loaded successfully on rank {rank}/{world_size}")
        print(f"Full model: {full_model_path}")
        print(f"Retain model: {retain_model_path}")
    
    return full_model, retain_model, tokenizer


def calculate_log_probability_score(
    batch: dict,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    config: ModelConfig
) -> List[float]:
    """Calculate M(f,x) = -1/|Yx| * sum(log p_f(yt | prompt, y<t)) for each example"""
    
    input_ids = batch["input_ids"]
    labels = batch["labels"]
    attention_mask = batch["attention_mask"]
    
    scores = []
    
    with torch.no_grad():
        # Get model outputs
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True
        )
        
        logits = outputs.logits
        
        # Calculate log probabilities for each example in the batch
        for i in range(input_ids.size(0)):
            # Get the sequence for this example
            input_seq = input_ids[i]
            label_seq = labels[i]
            
            # Find where the answer starts (after the prompt)
            split_symbol = " [/INST]" if config.model_family == "llama2-7b" else "Answer: "
            input_text = tokenizer.decode(input_seq, skip_special_tokens=True)
            
            if split_symbol in input_text:
                # Find answer token positions
                prompt_text = input_text.split(split_symbol)[0] + split_symbol
                prompt_tokens = tokenizer.encode(prompt_text, add_special_tokens=False)
                answer_start_idx = len(prompt_tokens)
            else:
                # Fallback: assume answer starts after half the sequence
                answer_start_idx = input_seq.size(0) // 2
            
            # Calculate log probability only for answer tokens
            answer_logits = logits[i, answer_start_idx-1:-1]  # shift by 1 for next-token prediction
            answer_tokens = label_seq[answer_start_idx:]
            
            # Remove padding tokens
            valid_mask = answer_tokens != -100
            if valid_mask.sum() == 0:
                scores.append(0.0)
                continue
                
            answer_logits = answer_logits[valid_mask]
            answer_tokens = answer_tokens[valid_mask]
            
            # Calculate log probabilities
            log_probs = F.log_softmax(answer_logits, dim=-1)
            token_log_probs = log_probs.gather(1, answer_tokens.unsqueeze(-1)).squeeze(-1)
            
            # Calculate average negative log probability: -1/|Yx| * sum(log p_f(yt))
            score = -token_log_probs.mean().item()
            scores.append(score)
    
    return scores


def calculate_dual_model_scores(
    batch: dict,
    full_model: AutoModelForCausalLM,
    retain_model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    config: ModelConfig
) -> Tuple[List[float], List[float]]:
    """Calculate Acc_IN and Acc_OUT scores for each example in the batch"""
    
    # Acc_IN: scores using the full model (IN_full)
    acc_in_scores = calculate_log_probability_score(batch, full_model, tokenizer, config)
    
    # Acc_OUT: scores using the retain model (OUT_s)
    acc_out_scores = calculate_log_probability_score(batch, retain_model, tokenizer, config)
    
    return acc_in_scores, acc_out_scores