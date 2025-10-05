"""
Required set-up for influence function computation.
Prepares models for running EK-FAC and defines the language modeling task
(in our case, cross-entropy training and query loss).
"""
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from torch import nn

from kronfluence.task import Task

BATCH_TYPE = Dict[str, torch.Tensor]

class LanguageModelingTask(Task):
    def __init__(self, config: dict) -> None:
        super().__init__()
        self.config = config

    def compute_train_loss(
        self,
        batch: BATCH_TYPE,
        model: nn.Module,
        sample: bool = False,
    ) -> torch.Tensor:
        # 안전하게 batch에서 데이터 추출
        if isinstance(batch, dict):
            input_ids = batch["input_ids"]
            attention_mask = batch.get("attention_mask", None)
            labels = batch["labels"]
        elif isinstance(batch, (list, tuple)):
            # batch가 리스트인 경우 일반적인 패턴들 시도
            if len(batch) >= 3:
                input_ids = batch[0]
                attention_mask = batch[1] if len(batch) > 1 else None
                labels = batch[2] if len(batch) > 2 else batch[0]
            else:
                input_ids = batch[0]
                attention_mask = None
                labels = batch[0]
        else:
            raise ValueError(f"Unsupported batch type: {type(batch)}")
        
        logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        ).logits.float()
        logits = logits[..., :-1, :].contiguous()
        logits = logits.view(-1, logits.size(-1))
        labels = labels[..., 1:].contiguous()
        
        if not sample:
            summed_loss = F.cross_entropy(logits, labels.view(-1), reduction="sum", ignore_index=-100)
        else:
            with torch.no_grad():
                probs = torch.nn.functional.softmax(logits.detach(), dim=-1)
                sampled_labels = torch.multinomial(probs, num_samples=1).flatten()
                masks = labels.view(-1) == -100
                sampled_labels[masks] = -100
            summed_loss = F.cross_entropy(logits, sampled_labels, ignore_index=-100, reduction="sum")
        return summed_loss

    def compute_measurement(
        self,
        batch: BATCH_TYPE,
        model: nn.Module,
    ) -> torch.Tensor:
        # 안전하게 batch에서 데이터 추출
        if isinstance(batch, dict):
            input_ids = batch["input_ids"]
            attention_mask = batch.get("attention_mask", None)
            labels = batch["labels"]
        elif isinstance(batch, (list, tuple)):
            # batch가 리스트인 경우 일반적인 패턴들 시도
            if len(batch) >= 3:
                input_ids = batch[0]
                attention_mask = batch[1] if len(batch) > 1 else None
                labels = batch[2] if len(batch) > 2 else batch[0]
            else:
                input_ids = batch[0]
                attention_mask = None
                labels = batch[0]
        else:
            raise ValueError(f"Unsupported batch type: {type(batch)}")
        
        logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        ).logits.float()
        shift_labels = labels[..., 1:].contiguous().view(-1)
        logits = logits[..., :-1, :].contiguous().view(-1, logits.size(-1))
        return F.cross_entropy(logits, shift_labels, ignore_index=-100, reduction="sum")

    def get_influence_tracked_modules(self) -> Optional[List[str]]:
        """
        추적할 모듈들을 반환. None을 반환하면 kronfluence가 자동으로 적절한 모듈들을 찾음
        (nn.Linear 및 nn.Conv2d 모듈들)
        """
        return None

    def get_attention_mask(self, batch: BATCH_TYPE) -> Optional[torch.Tensor]:
        """
        배치에서 attention mask를 안전하게 추출
        """
        try:
            # batch가 딕셔너리인 경우
            if isinstance(batch, dict):
                return batch.get("attention_mask", None)
            
            # batch가 리스트인 경우 (첫 번째 요소가 딕셔너리일 가능성)
            elif isinstance(batch, (list, tuple)) and len(batch) > 0:
                if isinstance(batch[0], dict):
                    return batch[0].get("attention_mask", None)
                else:
                    # 리스트의 두 번째 요소가 attention_mask일 수도 있음
                    if len(batch) > 1:
                        return batch[1] if isinstance(batch[1], torch.Tensor) else None
            
            # 기타 경우에는 None 반환 (attention mask 없음을 의미)
            return None
            
        except Exception as e:
            # 안전하게 None 반환
            return None
    
    def set_model(self, model):
        """임시 호환성을 위한 빈 메서드 (사용되지 않음)"""
        pass