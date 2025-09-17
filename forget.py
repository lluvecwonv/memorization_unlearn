from local_utils.data_module import TextForgetDatasetQA, TextForgetDatasetDPOQA, TNPOForgetDatasetQA
from trainer.trainer import CustomTrainerForgetting
from trainer.data_collator import create_custom_data_collator_forget
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, set_seed

import hydra 
import transformers
import os
from peft import LoraConfig, get_peft_model, PeftModel
from pathlib import Path
from utils import get_model_identifiers_from_yaml
from omegaconf import OmegaConf

def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])
    if 'lm_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

@hydra.main(version_base=None, config_path="config", config_name="forget")
def main(cfg):
    num_devices = int(os.environ.get('WORLD_SIZE', 1))
    print(f"num_devices: {num_devices}")
    print(f"WORLD_SIZE={num_devices}, LOCAL_RANK={os.environ.get('LOCAL_RANK', 'None')}")

    if os.environ.get('LOCAL_RANK') is not None:
        local_rank = int(os.environ.get('LOCAL_RANK', '0'))
        device_map = {'': local_rank}

    set_seed(cfg.seed)

    os.environ["WANDB_DISABLED"] = "true"
    model_cfg = get_model_identifiers_from_yaml(cfg.model_family)
    model_id = model_cfg["hf_key"]
    model_path = cfg.model_path
    print(f"model_path: {model_path}")

    print("######################")
    print("Saving to: ", cfg.save_dir)
    print("######################")
    # save cfg in cfg.save_dir
    if local_rank == 0:
        if os.path.exists(cfg.save_dir):
            print("Directory already exists")
            if not cfg.overwrite_dir:
                exit()

        Path(cfg.save_dir).mkdir(parents=True, exist_ok=True)

        with open(f"{cfg.save_dir}/config.yaml", "w") as file:
            OmegaConf.save(cfg, file)

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token

    max_length = 256
    if cfg.forget_loss == "dpo":
        torch_format_dataset = TextForgetDatasetDPOQA(cfg.data_path, tokenizer=tokenizer, model_family = cfg.model_family, max_length=max_length, split=cfg.split)
    elif "TNPO" in cfg.forget_loss or "tsimnpo" in cfg.forget_loss:
        torch_format_dataset = TNPOForgetDatasetQA(cfg.data_path, tokenizer=tokenizer, model_family = cfg.model_family, max_length=max_length, split=cfg.split, loss_type=cfg.forget_loss)
    else:
        torch_format_dataset = TextForgetDatasetQA(cfg.data_path, tokenizer=tokenizer, model_family = cfg.model_family, max_length=max_length, split=cfg.split, loss_type=cfg.forget_loss)
    
    # forget 토큰 마스킹 적용
    # forget_mask_path 설정 확인
    mask_path = getattr(cfg, 'forget_mask_path', None) or getattr(cfg, 'toxic_token_mask_path', None) or getattr(cfg, 'si_scores_path', None)
    if mask_path is not None and mask_path.strip() != "":
        print(f"Loading forget mask from {mask_path}")
        forget_mask = torch.load(mask_path, weights_only=True)
        
        # 데이터셋에 마스크 설정
        torch_format_dataset.set_forget_mask(forget_mask)
        print(f"Applied forget masking to {len(forget_mask)} examples")
    
    batch_size = cfg.batch_size
    gradient_accumulation_steps = cfg.gradient_accumulation_steps
    print(f"Dataset size={len(torch_format_dataset)}")
    denom = batch_size*gradient_accumulation_steps*num_devices
    print(f"batch_size={batch_size}, grad_accumulation_steps={gradient_accumulation_steps}, WORLD_SIZE={num_devices}, denom={denom}")
    steps_per_epoch = len(torch_format_dataset)//denom

    max_steps = int(cfg.num_epochs*len(torch_format_dataset))//denom
    print(f"steps_per_epoch: {steps_per_epoch}, max_steps: {max_steps}")

    # 개선된 훈련 인자 설정
    training_args = transformers.TrainingArguments(
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=max(1, steps_per_epoch),
            max_steps=max_steps,
            learning_rate=cfg.lr,
            bf16=True,
            bf16_full_eval=True,
            logging_steps=max(1,max_steps//20),
            logging_dir=f'{cfg.save_dir}/logs',
            output_dir=cfg.save_dir,
            optim="paged_adamw_32bit",
            save_strategy="steps" if cfg.save_model and (not cfg.eval_only) else "no",
            save_steps=steps_per_epoch,
            save_only_model=True,
            ddp_find_unused_parameters= False,
            deepspeed='/root/npo/config/ds_config.json',
            weight_decay = cfg.weight_decay,
            eval_steps = steps_per_epoch,
            eval_strategy = "steps" if cfg.eval_while_train else "no",
            seed=cfg.seed,
            # 추가된 개선사항 (필요한 것만)
            lr_scheduler_type=getattr(cfg, 'lr_scheduler_type', 'linear'),
            adam_beta1=getattr(cfg, 'adam_beta1', 0.9),
            adam_beta2=getattr(cfg, 'adam_beta2', 0.95),
            adam_epsilon=getattr(cfg, 'adam_epsilon', 1e-08),
            max_grad_norm=getattr(cfg, 'max_grad_norm', 1.0),
            remove_unused_columns=False,
        )
    print(f"save_steps: {training_args.save_steps}, eval_steps: {training_args.eval_steps}, warmup_steps: {training_args.warmup_steps}")
    
    #first get the base model architectur2e
    #if there is a pytorch*.bin file in the model path, then load that. use regex there can be anythign in between pytorch and .bin
    import re
    path_found = False
    for file in os.listdir(cfg.model_path):
        if re.search("pytorch.*\.bin", file):
            path_found = True
            break
        
        if re.search("model-*\.safetensors", file):
            path_found = True
            break

    oracle_model = None

    if path_found:
        config = AutoConfig.from_pretrained(model_id)

        print("Loading from checkpoint")
        model = AutoModelForCausalLM.from_pretrained(cfg.model_path, config=config, use_flash_attention_2=model_cfg["flash_attention2"]=="true", torch_dtype=torch.bfloat16, trust_remote_code = True)
        # Load oracle/reference model when needed (KL regularization, TNPO, or NPO requires ref)
        if ("KL" in cfg.forget_loss) or ("TNPO" in cfg.forget_loss) or ("NPO" in cfg.forget_loss and "simnpo" not in cfg.forget_loss):
            oracle_model = AutoModelForCausalLM.from_pretrained(
                cfg.model_path,
                config=config,
                use_flash_attention_2=model_cfg["flash_attention2"]=="true",
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
            )
            print("######################")
            print(f"Loaded oracle model: {oracle_model}")

    else:
        print("Loading after merge and unload")
        model = AutoModelForCausalLM.from_pretrained(model_id, use_flash_attention_2=model_cfg["flash_attention2"]=="true", torch_dtype=torch.bfloat16, device_map=device_map)
        #now use the checkpoint to add the LoRA modules
        model = PeftModel.from_pretrained(model, model_id = cfg.model_path)
        #save this as a standard model so that we can again do PEFT style finetuneing from scratch
        model = model.merge_and_unload()
        #save the model for next time
        model.save_pretrained(cfg.model_path)

        # Load oracle/reference model when needed (KL regularization, TNPO, or NPO requires ref)
        if ("KL" in cfg.forget_loss) or ("TNPO" in cfg.forget_loss) or ("NPO" in cfg.forget_loss and "simnpo" not in cfg.forget_loss):
            ref_config = AutoConfig.from_pretrained(model_id)
            oracle_model = AutoModelForCausalLM.from_pretrained(
                cfg.model_path,
                config=ref_config,
                use_flash_attention_2=model_cfg["flash_attention2"]=="true",
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
            )
            print("######################")
            print(f"Loaded oracle model: {oracle_model}")
    
    
    # Hot fix for https://discuss.huggingface.co/t/help-with-llama-2-finetuning-s/root/npo/results/TOFU_phi_TNPO_forget10/checkpoint-24/root/npo/results/TOFU_phi_TNPO_forget10/checkpoint-24etup/50035
    model.generation_config.do_sample = True
    
    #now we have a HuggingFace model 
    if model_cfg["gradient_checkpointing"] == "true":
        model.gradient_checkpointing_enable()
    
    config = LoraConfig(
        r=cfg.LoRA.r, 
        lora_alpha=cfg.LoRA.alpha, 
        target_modules=find_all_linear_names(model), 
        lora_dropout=cfg.LoRA.dropout,
        bias="none", 
        task_type="CAUSAL_LM"
    )
    if cfg.LoRA.r != 0:
        model = get_peft_model(model, config)
        print_trainable_parameters(model)

    
    # forget 토큰 마스킹 파라미터 설정
    forget_weight = getattr(cfg, 'forget_weight', getattr(cfg, 'toxic_lambda', 1.0)) if mask_path is not None else None
    
    trainer = CustomTrainerForgetting(
        model=model,
        tokenizer=tokenizer,
        train_dataset=torch_format_dataset,
        eval_dataset = torch_format_dataset,
        compute_metrics=None,                # the callback for computing metrics, None in this case since you're doing it in your callback
        # callbacks=[GlobalStepDeletionCallback],
        args=training_args,
        data_collator=create_custom_data_collator_forget(cfg.forget_loss),
        oracle_model = oracle_model,
        loss_type = cfg.forget_loss,
        toxic_lambda = forget_weight,  # forget 토큰 마스킹 강도
    )
    model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
    # trainer.train()
    if cfg.eval_only:
        trainer.evaluate()
    else:
        trainer.train()

    #save the tokenizer
    if cfg.save_model and (not cfg.eval_only):
        model.save_pretrained(cfg.save_dir)
        tokenizer.save_pretrained(cfg.save_dir)

    #delete all "global_step*" files in the save_dir/checkpoint-*/ directories
    if local_rank == 0:
        for file in Path(cfg.save_dir).glob("checkpoint-*"):
            for global_step_dir in file.glob("global_step*"):
                #delete the directory
                import shutil
                shutil.rmtree(global_step_dir)



if __name__ == "__main__":
    main()
