import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist


# Debug counter for TNPO logging
TNPO_DEBUG_COUNTER = 0


def get_loss(model, oracle_model, inputs, loss_type, beta=0.1, forget_weight=1.0, tokenizer=None):
    # forget_loss
    if 'TNPO_KL' in loss_type:
        forget_loss = tnpo_kl_loss(model, oracle_model, inputs, beta=beta, forget_weight=forget_weight, return_dict=False)
    elif 'TNPO' in loss_type:
        forget_loss = tnpo_loss(model, oracle_model, inputs, beta=beta, forget_weight=forget_weight, return_dict=False)
    elif 'GA' in loss_type:
        forget_loss = ga_loss(model, inputs)
    elif 'FINETUNE' in loss_type:
        forget_loss = finetune_loss(model, inputs)
    elif 'IDK' in loss_type:
        forget_loss = idk_loss(model, inputs)
    elif 'NPO_KL' in loss_type:
        forget_loss = npo_kl_loss(model, oracle_model, inputs, beta=beta, return_dict=False)
    elif 'NPO' in loss_type:
        forget_loss = npo_loss(model, oracle_model, inputs, return_dict=False)
    elif loss_type == "simnpo":
        forget_loss = simnpo_loss(model, inputs, beta=beta, return_dict=False)
    elif loss_type == 'simnpo_grad_diff':
        forget_loss = simnpo_grad_diff_loss(model, inputs, beta=beta, return_dict=False)
    elif loss_type == "tsimnpo":
        forget_loss = tsimnpo_loss(model, inputs, beta=beta, forget_weight=forget_weight, return_dict=False)

    # 트레이너가 2개 값을 기대하므로 regularization_loss도 반환
    regularization_loss = torch.tensor(0.0, device=forget_loss.device)
    return forget_loss, regularization_loss


def _unpack_inputs(inputs):
    """Unpack inputs from dict or tuple. Returns (input_ids, labels, attention_mask, forget_mask)."""
    if isinstance(inputs, dict):
        input_ids = inputs["input_ids"]
        labels = inputs["labels"]
        attention_mask = inputs.get("attention_mask", None)
        forget_mask = inputs.get("forget_mask", inputs.get("mask", None))
        return input_ids, labels, attention_mask, forget_mask
    else:
        input_ids, labels, attention_mask = inputs[:3]
        forget_mask = inputs[3] if isinstance(inputs, (list, tuple)) and len(inputs) >= 4 else None
        return input_ids, labels, attention_mask, forget_mask


def finetune_loss(model, inputs):
    input_ids, labels, attention_mask, _ = _unpack_inputs(inputs)
    outputs = model(input_ids, labels=labels, attention_mask=attention_mask)
    loss = +1 * outputs.loss
    return loss

def ga_loss(model, inputs):
    """
    Gradient Ascent loss for unlearning
    We want to maximize the loss (minimize negative log-likelihood)
    """
    input_ids, labels, attention_mask, _ = _unpack_inputs(inputs)
    outputs = model(input_ids, labels=labels, attention_mask=attention_mask)

    # Get the original cross-entropy loss
    forget_loss = outputs.loss
    valid_tokens = (labels != -100).sum().item()
    
    # For gradient ascent, we minimize the negative loss
    # This effectively maximizes the original loss (bad performance on forget data)
    loss = -forget_loss

    
    return loss

def idk_loss(model, inputs):
    """I Don't Know loss for unlearning"""
    input_ids, labels, attention_mask, _ = _unpack_inputs(inputs)
    outputs = model(input_ids, labels=labels, attention_mask=attention_mask)
    loss = outputs.loss
    return loss

# NPO Loss - Neural Preference Optimization
def npo_loss(
    model,
    oracle_model,
    inputs,
    beta: float = 0.3,
    return_dict: bool = True,
):
    """
    NPO (Neural Preference Optimization) loss for unlearning.
    
    손실:
        L_NPO = -(2/β) * E[ log σ( β * (CE_θ - CE_ref) ) ]
        where CE = cross entropy loss for the sequence
    
    Args:
        model: 현재 훈련 중인 모델
        oracle_model: 참조 모델 (oracle/fine-tuned model) - None이면 기본 CE 손실 사용
        inputs: (input_ids, labels, attention_mask, forget_mask)
        beta: NPO 강도 조절 파라미터
        return_dict: 로깅용 통계 반환 여부
    """
    input_ids, labels, attention_mask, _ = _unpack_inputs(inputs)
    
    # 현재 모델의 출력
    outputs = model(input_ids, labels=labels, attention_mask=attention_mask)
    forget_loss_current = get_batch_loss(outputs.logits, labels)
    

    # 참조 모델의 출력 (no_grad로 고정)
    with torch.no_grad():
        oracle_model.eval()
        oracle_outputs = oracle_model(input_ids, labels=labels, attention_mask=attention_mask)
        forget_loss_oracle = get_batch_loss(oracle_outputs.logits, labels)
    
    # NPO 손실 계산
    neg_log_ratios = forget_loss_current - forget_loss_oracle
    loss = -F.logsigmoid(beta * neg_log_ratios).mean() * 2 / beta
    return loss


def npo_kl_loss(
    model,
    oracle_model, 
    inputs,
    beta: float = 0.3,
    npo_coeff: float = 1.0,
    kl_coeff: float = 0.1,
    return_dict: bool = True,
):
    """
    NPO + KL regularization loss for unlearning.
    
    Combined loss:
        L = npo_coeff * L_NPO + kl_coeff * L_KL
        L_NPO = -(2/β) * E[ log σ( β * (CE_θ - CE_oracle) ) ]  (forget data)
        L_KL = KL_div(current_model, oracle_model)  (retain data)
    """
    # inputs should be (forget_inputs, retain_inputs)
    if isinstance(inputs, tuple) and len(inputs) == 2:
        forget_inputs, retain_inputs = inputs
    else:
        # If only forget inputs provided, use them for both
        forget_inputs = inputs
        retain_inputs = inputs
    
    # NPO loss on forget data
    forget_input_ids, forget_labels, forget_attention_mask, _ = _unpack_inputs(forget_inputs)
    forget_outputs = model(forget_input_ids, labels=forget_labels, attention_mask=forget_attention_mask)
    forget_loss_current = get_batch_loss(forget_outputs.logits, forget_labels)
    
    with torch.no_grad():
        oracle_model.eval()
        forget_outputs_oracle = oracle_model(forget_input_ids, labels=forget_labels, attention_mask=forget_attention_mask)
        forget_loss_oracle = get_batch_loss(forget_outputs_oracle.logits, forget_labels)
    
    neg_log_ratios = forget_loss_current - forget_loss_oracle
    forget_loss = -F.logsigmoid(beta * neg_log_ratios).mean() * (2.0 / beta)
    
    # KL regularization on retain data  
    retain_input_ids, retain_labels, retain_attention_mask, _ = _unpack_inputs(retain_inputs)
    
    with torch.no_grad():
        retain_outputs_oracle = oracle_model(retain_input_ids, labels=retain_labels, attention_mask=retain_attention_mask)
        retain_probs_oracle = F.log_softmax(retain_outputs_oracle.logits, dim=-1)
        retain_probs_oracle = retain_probs_oracle.view(-1, retain_outputs_oracle.logits.shape[-1])
    
    retain_outputs_current = model(retain_input_ids, labels=retain_labels, attention_mask=retain_attention_mask)
    retain_probs_current = F.log_softmax(retain_outputs_current.logits, dim=-1)
    retain_probs_current = retain_probs_current.view(-1, retain_outputs_current.logits.shape[-1])
    
    # KL divergence: minimize distance from oracle on retain data
    retain_loss = F.kl_div(retain_probs_current, retain_probs_oracle, reduction='batchmean', log_target=True)
    
    # Combined loss
    total_loss = npo_coeff * forget_loss + kl_coeff * retain_loss
    
    if return_dict:
        with torch.no_grad():
            stats = {
                "loss": total_loss,
                "forget_loss": forget_loss,
                "retain_loss": retain_loss,
                "npo_coeff": npo_coeff,
                "kl_coeff": kl_coeff,
            }
        return stats
    return total_loss


def tnpo_kl_loss(
    model,
    oracle_model,
    inputs,
    beta: float = 0.3,
    forget_weight: float = 1.0,
    kl_coeff: float = 0.1,
    return_dict: bool = True,
    clamp_abs: float = 20.0,
):
    """
    TNPO + KL regularization loss for token-level unlearning.
    
    Combined loss:
        L = forget_weight * L_TNPO + kl_coeff * L_KL
        L_TNPO = -(2/β) * mean_{forget tokens} [ log σ( β * (CE_θ - CE_oracle) ) ]
        L_KL = KL_div(current_model, oracle_model)  (retain data)
    """
    # inputs should be (forget_inputs, retain_inputs) for TNPO_KL
    if isinstance(inputs, tuple) and len(inputs) == 2:
        forget_inputs, retain_inputs = inputs
    else:
        # If only forget inputs provided, use them for both
        forget_inputs = inputs
        retain_inputs = inputs
    
    # TNPO loss on forget data (token-level)
    forget_input_ids, forget_labels, forget_attention_mask, forget_mask = _unpack_inputs(forget_inputs)
    forget_out = model(forget_input_ids, labels=forget_labels, attention_mask=forget_attention_mask)
    forget_logits = forget_out.logits

    with torch.no_grad():
        oracle_model.eval()
        forget_oracle_out = oracle_model(forget_input_ids, labels=forget_labels, attention_mask=forget_attention_mask)
        forget_oracle_logits = forget_oracle_out.logits
        
    # Token-level TNPO computation
    sl   = forget_logits[...,     :-1, :].contiguous()
    srl  = forget_oracle_logits[..., :-1, :].contiguous()
    y    = forget_labels[...,     1:  ].contiguous()

    valid = (y != -100)

    logp     = F.log_softmax(sl.float(),  dim=-1)
    logp_oracle = F.log_softmax(srl.float(), dim=-1)
    
    y_safe = y.masked_fill(~valid, 0)
    tgt = y_safe.unsqueeze(-1)
    gp  = torch.gather(logp,         2, tgt).squeeze(-1)  # log π_θ(y|x)
    gop = torch.gather(logp_oracle,  2, tgt).squeeze(-1)  # log π_oracle(y|x)

    ce_theta  = -gp
    ce_oracle = -gop

    # Apply forget mask for token-level selection
    fmask = forget_mask[..., 1:].contiguous().bool()
    forget_hits = (fmask & valid)

    # TNPO loss on forget tokens only
    if forget_hits.any():
        neg_log_ratios = (ce_theta - ce_oracle)[forget_hits].clamp(-clamp_abs, clamp_abs)
        tnpo_loss = -F.logsigmoid(beta * neg_log_ratios).mean() * (2.0 / beta)
    else:
        # No forget tokens found - use zero loss
        tnpo_loss = torch.tensor(0.0, device=forget_logits.device)
    
    # KL regularization on retain data
    retain_input_ids, retain_labels, retain_attention_mask, _ = _unpack_inputs(retain_inputs)
    
    with torch.no_grad():
        retain_outputs_oracle = oracle_model(retain_input_ids, labels=retain_labels, attention_mask=retain_attention_mask)
        retain_probs_oracle = F.log_softmax(retain_outputs_oracle.logits, dim=-1)
        retain_probs_oracle = retain_probs_oracle.view(-1, retain_outputs_oracle.logits.shape[-1])
    
    retain_outputs_current = model(retain_input_ids, labels=retain_labels, attention_mask=retain_attention_mask)
    retain_probs_current = F.log_softmax(retain_outputs_current.logits, dim=-1)
    retain_probs_current = retain_probs_current.view(-1, retain_outputs_current.logits.shape[-1])
    
    # KL divergence: minimize distance from oracle on retain data
    kl_loss = F.kl_div(retain_probs_current, retain_probs_oracle, reduction='batchmean', log_target=True)
    
    # Combined loss
    total_loss = forget_weight * tnpo_loss + kl_coeff * kl_loss
    
    if return_dict:
        with torch.no_grad():
            stats = {
                "loss": total_loss,
                "tnpo_loss": tnpo_loss,
                "kl_loss": kl_loss,
                "forget_tokens": forget_hits.sum().item(),
                "forget_weight": forget_weight,
                "kl_coeff": kl_coeff,
            }
        return stats
    return total_loss


def get_batch_loss(output, labels):
    shifted_labels = labels[..., 1:].contiguous()
    output = output[..., :-1, :].contiguous()

    loss_function = nn.CrossEntropyLoss(ignore_index=-100, reduction='none')
    # get the sum loss for each sequence in a batch
    loss = loss_function(output.transpose(-1,-2), shifted_labels).sum(dim=-1)

    return loss



def tnpo_loss(
    model,
    oracle_model,
    inputs,
    beta: float = 3.0,          # 권장 0.1~0.5
    forget_weight: float = 1.0, # Forget 토큰 가중치
    return_dict: bool = True,   # 로깅용 서브 로스 반환 여부
    clamp_abs: float = 20.0,    # 수치 안정화용 클램프
):
    """
    Token-level NPO (TNPO-) — Forget 토큰만 사용 (logsigmoid 버전).

    손실:
        L_forget = -(2/β) * mean_{forget tokens} [ log σ( β * (CE_θ - CE_ref) ) ]
        where CE = -log π(target),  (CE_θ - CE_ref) = -log(π_θ/π_ref)

    구현 포인트:
      - 라벨 시프트(next-token)와 마스크 정합성 유지
      - oracle_model은 고정 (eval + no_grad)
      - 수치 안정화: fp32 log_softmax, clamp
      - forget 토큰이 없을 때 NaN 방지 (graph-safe zero)
    """
    input_ids, labels, attention_mask, forget_mask = _unpack_inputs(inputs)
    out = model(input_ids, labels=labels, attention_mask=attention_mask)
    logits = out.logits

    with torch.no_grad():
        oracle_model.eval()
        oracle_out = oracle_model(input_ids, labels=labels, attention_mask=attention_mask)
        oracle_logits = oracle_out.logits
        
    sl   = logits[...,     :-1, :].contiguous()
    srl  = oracle_logits[..., :-1, :].contiguous()
    y    = labels[...,     1:  ].contiguous()

    valid = (y != -100)

    logp     = F.log_softmax(sl.float(),  dim=-1)
    logp_oracle = F.log_softmax(srl.float(), dim=-1)
    
    y_safe = y.masked_fill(~valid, 0)
    tgt = y_safe.unsqueeze(-1)
    gp  = torch.gather(logp,         2, tgt).squeeze(-1)  # log π_θ(y|x)
    gop = torch.gather(logp_oracle,  2, tgt).squeeze(-1)  # log π_oracle(y|x)

    ce_theta  = -gp
    ce_oracle = -gop

    fmask = forget_mask[..., 1:].contiguous().bool()
    forget_hits = (fmask & valid)

    # -log ratio = (-log π_θ) - (-log π_oracle) = CE_θ - CE_oracle = -log(π_θ/π_oracle)
    neg_log_ratios = (ce_theta - ce_oracle)[forget_hits]
    forget_loss = -F.logsigmoid(beta * neg_log_ratios).mean() * (2.0 / beta)
    
    loss = forget_weight * forget_loss

    if return_dict:
        with torch.no_grad():
            stats = {
                "loss": loss,
                "forget_loss": forget_loss,
                "forget_tokens": forget_hits.sum().item(),
            }
        return stats
    return loss




def simnpo_loss(
    model,
    inputs,
    beta: float = 2.5,
    gamma: float = 0.0,
    return_dict: bool = True,
):
    """
    Simplified NPO loss for single input (forget data only)
    """
    input_ids, labels, attention_mask, _ = _unpack_inputs(inputs)
    outputs = model(input_ids, labels=labels, attention_mask=attention_mask)
    loss_mask = labels != -100
    forget_loss = get_batch_loss(outputs.logits, labels) / loss_mask.sum(-1) - gamma

    loss = -F.logsigmoid(beta * forget_loss).mean() * 2 / beta
    
    if return_dict:
        return {"loss": loss, "forget_loss": forget_loss}
    return loss


def tsimnpo_loss(
    model,
    inputs,
    beta: float = 2.5,
    gamma: float = 1.0,
    forget_weight: float = 1.0,
    return_dict: bool = True,
    clamp_abs: float = 20.0,
):
    """
    Token-level Simplified NPO loss for single input (forget data only)
    """
    input_ids, labels, attention_mask, forget_mask = _unpack_inputs(inputs)
    out = model(input_ids, labels=labels, attention_mask=attention_mask)
    logits = out.logits
        
    sl = logits[..., :-1, :].contiguous()
    y = labels[..., 1:].contiguous()

    valid = (y != -100)

    logp = F.log_softmax(sl.float(), dim=-1)
    
    y_safe = y.masked_fill(~valid, 0)
    tgt = y_safe.unsqueeze(-1)
    gp = torch.gather(logp, 2, tgt).squeeze(-1)  # log π_θ(y|x)

    ce_theta = -gp

    # Forget mask is required for token-level operation
    if forget_mask is None:
        raise ValueError("tsimnpo_loss requires forget_mask to specify which tokens to forget")
    
    fmask = forget_mask[..., 1:].contiguous().bool()
    forget_hits = (fmask & valid)

    # Debug: Print selected tokens
    if forget_hits.any() and torch.distributed.get_rank() == 0:
        # Get first batch item for debugging
        batch_idx = 0
        forget_tokens_indices = torch.where(forget_hits[batch_idx])[0]

    # Token-level simplified NPO: CE_θ - gamma
    if forget_hits.any():
        token_loss = (ce_theta - gamma)[forget_hits].clamp(-clamp_abs, clamp_abs)
        forget_loss = -F.logsigmoid(beta * token_loss).mean() * (2.0 / beta)
    else:
        forget_loss = torch.tensor(0.0, device=logits.device)
        if torch.distributed.get_rank() == 0:
            print("[TSIMNPO WARNING] No forget tokens found in this batch")
    
    loss = forget_weight * forget_loss

    if return_dict:
        with torch.no_grad():
            stats = {
                "loss": loss,
                "forget_loss": forget_loss,
                "forget_tokens": forget_hits.sum().item(),
            }
        return stats
    return loss


def simnpo_grad_diff_loss(
    model,
    inputs,
    beta: float = 0.1,
    gamma: float = 1.0,
    npo_coeff: float = 1.0,
    grad_diff_coeff: float = 0.1,
    return_dict: bool = True,
):
    """
    Simplified NPO with gradient difference regularization
    """
    forget_inputs, retain_inputs = inputs
    input_ids, labels, attention_mask = forget_inputs
    outputs = model(input_ids, labels=labels, attention_mask=attention_mask)
    loss_mask = labels != -100
    forget_loss = get_batch_loss(outputs.logits, labels) / loss_mask.sum(-1) - gamma
    forget_loss = -F.logsigmoid(beta * forget_loss).mean() * 2 / beta

    retain_input_ids, retain_labels, retain_attention_mask = retain_inputs
    retain_outputs = model(retain_input_ids, labels=retain_labels, attention_mask=retain_attention_mask)
    retain_loss = retain_outputs.loss
    loss = npo_coeff * forget_loss + grad_diff_coeff * retain_loss

    if return_dict:
        return {"loss": loss, "forget_loss": forget_loss, "retain_loss": retain_loss}
    return loss


def kl_loss(model, oracle_model, inputs):
    input_ids, labels, attention_mask = inputs[:3]

    outputs = model(input_ids, labels=labels, attention_mask=attention_mask)
    logits = outputs.logits.detach()
    probs = F.log_softmax(outputs.logits, dim=-1).view(-1, logits.shape[-1])

    with torch.no_grad():
        outputs_oracle = oracle_model(input_ids, labels=labels, attention_mask=attention_mask)
    oracle_probs = F.log_softmax(outputs_oracle.logits, dim=-1).view(-1, outputs_oracle.logits.shape[-1])

    loss = nn.functional.kl_div(
        probs, oracle_probs, reduction='batchmean', log_target=True)

    return loss
