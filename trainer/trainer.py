import copy
import os
import json
import torch
from pathlib import Path
from transformers import Trainer
from transformers.trainer_utils import seed_worker
from transformers.utils import is_datasets_available
import datasets
from torch.utils.data import DataLoader
import torch.nn.functional as F
import deepspeed

from .losses import get_loss


class CustomTrainerForgetting(Trainer):
    def __init__(self, *args, **kwargs):
        self.loss_type = kwargs.pop('loss_type', 'NPO+GD')
        self.oracle_model = kwargs.pop('oracle_model', None)
        self.eval_cfg = kwargs.pop('eval_cfg', None)
        self.seed = kwargs.pop('seed', 42)
        
        # NPO/DPO specific parameters
        self.beta = kwargs.pop('beta', 0.1)
        
        # Forget token masking parameter (for TNPO)
        self.forget_weight = kwargs.pop('toxic_lambda', 1.0)

        super(CustomTrainerForgetting, self).__init__(*args, **kwargs)

        if self.oracle_model is not None:
            self.oracle_model = self.e_prepare_deepspeed(self.oracle_model)
        print(f'oracle_model: {self.oracle_model}')
            
    def get_train_dataloader(self):
        """Override to support custom seeding for reproducibility"""
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")
        
        train_dataset = self.train_dataset
        data_collator = self.data_collator
        
        if is_datasets_available() and isinstance(train_dataset, datasets.Dataset):
            train_dataset = self._remove_unused_columns(train_dataset, description="training")
        else:
            data_collator = self._get_collator_with_removed_columns(data_collator, description="training")

        dataloader_params = {
            "batch_size": self._train_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }

        # Custom generator for reproducibility
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.state.global_step)
        
        if not isinstance(train_dataset, torch.utils.data.IterableDataset):
            dataloader_params["generator"] = generator
            dataloader_params["shuffle"] = True
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["worker_init_fn"] = seed_worker

        return self.accelerator.prepare(DataLoader(train_dataset, **dataloader_params))

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """Simplified loss computation - all losses now handled in losses.py"""
        
        # Get forget_weight from trainer attributes (for TNPO and tsimnpo)
        loss_key = str(self.loss_type).lower()
        forget_weight = getattr(self, 'forget_weight', 1.0) if ('tnpo' in loss_key or 'tsimnpo' in loss_key) else None
        
        # All loss computation now unified through get_loss function
        forget_loss, regularization_loss = get_loss(
            model, self.oracle_model, inputs, self.loss_type, 
            beta=self.beta, forget_weight=forget_weight, tokenizer=self.tokenizer
        )
        
        # Combine losses (regularization_loss can be 0 for some loss types)
        loss = forget_loss
        
        return loss if not return_outputs else (loss, None)

    def prediction_step(self, model, inputs, prediction_loss_only: bool, ignore_keys=None):
        """Prediction step for evaluation"""
        if isinstance(inputs, dict):
            input_ids = inputs["input_ids"]
            labels = inputs["labels"]
            attention_mask = inputs.get("attention_mask", None)
        else:
            input_ids, labels, attention_mask = inputs[:3]
        with torch.no_grad():
            outputs = model(input_ids, labels=labels, attention_mask=attention_mask)
            logits = outputs.logits
            loss = outputs.loss
            print(f"loss: {loss.item():.6f}")
        return (loss, logits, labels)


    def e_prepare_deepspeed(self, model):
        # Adapted from accelerate: https://github.com/huggingface/accelerate/blob/739b135f8367becb67ffaada12fe76e3aa60fefd/src/accelerate/accelerator.py#L1473
        deepspeed_plugin = self.accelerator.state.deepspeed_plugin
        config_kwargs = copy.deepcopy(deepspeed_plugin.deepspeed_config)

        if model is not None:
            if hasattr(model, "config"):
                hidden_size = (
                    max(model.config.hidden_sizes)
                    if getattr(model.config, "hidden_sizes", None)
                    else getattr(model.config, "hidden_size", None)
                )
                if hidden_size is not None and config_kwargs["zero_optimization"]["stage"] == 3:
                    # Note that `stage3_prefetch_bucket_size` can produce DeepSpeed messages like: `Invalidate trace cache @ step 0: expected module 1, but got module 0`
                    # This is expected and is not an error, see: https://github.com/microsoft/DeepSpeed/discussions/4081
                    config_kwargs.update(
                        {
                            "zero_optimization.reduce_bucket_size": hidden_size * hidden_size,
                            "zero_optimization.stage3_param_persistence_threshold": 10 * hidden_size,
                            "zero_optimization.stage3_prefetch_bucket_size": 0.9 * hidden_size * hidden_size,
                        }
                    )

        # If ZeRO-3 is used, we shard both the active and reference model.
        # Otherwise, we assume the reference model fits in memory and is initialized on each device with ZeRO disabled (stage 0)

        # for BLUE, we use ZeRO-3 here
        if config_kwargs["zero_optimization"]["stage"] != 3:
            config_kwargs["zero_optimization"]["stage"] = 0

        config_kwargs["optimizer"] = {"type": None}
        model, *_ = deepspeed.initialize(model=model, config=config_kwargs)
        model.eval()
        #set the gradients to false for every parameter
        for param in model.parameters():
            param.requires_grad = False

        return model
