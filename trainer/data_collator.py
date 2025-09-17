import os
import torch


_FORGET_MASK_CACHE = None
_SI_SCORES_CACHE = None


# _fit_to_shape 함수도 더 이상 사용되지 않음


def create_custom_data_collator_forget(loss_type=None):
    """
    Factory function to create a data collator with loss_type awareness.
    Returns a dict batch compatible with HF Trainer.
    """
    def custom_data_collator_forget(samples):
        """
        Collate into a dict batch: {input_ids, labels, attention_mask[, forget_mask]}.
        Uses only forget split from dataset outputs.
        """
        # samples는 [forget_data, retain_data] 형태의 리스트
        forget_data = [sample[0] for sample in samples]

        input_ids = torch.stack([s[0] for s in forget_data])
        labels = torch.stack([s[1] for s in forget_data])
        attention_mask = torch.stack([s[2] for s in forget_data])

        batch = {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }

        # TNPO 또는 tsimnpo일 때만 forget mask 추가 (있을 때만)
        if loss_type and ('TNPO' in str(loss_type).upper() or 'tsimnpo' in str(loss_type).lower()):
            if len(forget_data[0]) >= 4:
                masks = torch.stack([s[3] for s in forget_data])
                batch["forget_mask"] = masks

        return batch

    return custom_data_collator_forget


def custom_data_collator_forget(samples):
    """
    Legacy function for backward compatibility
    """
    return create_custom_data_collator_forget()(samples)


def compute_metrics(pred):
    """Compute metrics for evaluation"""
    import torch
    import torch.nn as nn
    
    logits, labels = torch.from_numpy(pred.predictions), torch.from_numpy(pred.label_ids)
    preds = torch.from_numpy(pred.predictions.argmax(-1))
    shifted_labels = labels[..., 1:].contiguous()
    acc = torch.mean((preds[..., :-1] == shifted_labels).float())
    loss = get_loss(logits, labels)
    return {"eval accuracy": acc, "eval loss": loss.item()}


def get_loss(output, labels):
    """Compute loss for evaluation"""
    shifted_labels = labels[..., 1:].contiguous()
    output = output[..., :-1, :].contiguous()
    loss_function = nn.CrossEntropyLoss(ignore_index=-100)
    loss = loss_function(output.view(-1, output.size(-1)), shifted_labels.view(-1))
    return loss


def get_batch_loss(logits, labels):
    """
    Compute per-sample loss for a batch.
    Used in NPO and DPO loss computations.
    """
    loss_fct = torch.nn.CrossEntropyLoss(reduction='none', ignore_index=-100)
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    
    # Flatten the tokens
    shift_logits = shift_logits.view(-1, shift_logits.size(-1))
    shift_labels = shift_labels.view(-1)
    
    # Compute loss
    loss = loss_fct(shift_logits, shift_labels)
    loss = loss.view(labels.size(0), -1)  # Reshape to [batch_size, seq_len-1]
    
    # Mask and average
    loss_mask = (shift_labels != -100).view(labels.size(0), -1).float()
    loss = (loss * loss_mask).sum(dim=1) / loss_mask.sum(dim=1).clamp(min=1)
    
    return loss

