import os
import torch
import warnings
import hydra
from omegaconf import DictConfig

from data_module import TextForgetDatasetQA
from memorization.analysis.paraphrase_analyzer import DualModelAnalyzer
from memorization.analysis.paraphrase_generator import ParaphraseGenerator
from memorization.utils import load_models_and_tokenizer, create_paraphrased_dataset, save_results_for_notebook


def generate_paraphrases_for_questions(questions, generator, num_paraphrases, model=None, tokenizer=None):

    all_paraphrases = []
    failed_count = 0

    for idx, question in enumerate(questions):
        # Try beam search paraphrases first
        paraphrases = generator.generate_beam_paraphrases(question, model, tokenizer)

        # If beam search is not available or failed, use prompt-based generation
        if not paraphrases and generator.prompt_paraphrasing_enabled:
            paraphrases = generator.generate_prompt_paraphrases(question, model, tokenizer)

        # Extract just the text from paraphrase dicts
        paraphrase_texts = [p['text'] for p in paraphrases] if paraphrases else []

        # Skip if paraphrase generation failed
        if not paraphrase_texts:
            warnings.warn(f"Failed to generate paraphrases for question {idx}: {question[:50]}...")
            failed_count += 1
            continue

        # Trim to match num_paraphrases if we have more
        if len(paraphrase_texts) > num_paraphrases:
            paraphrase_texts = paraphrase_texts[:num_paraphrases]

        all_paraphrases.append(paraphrase_texts)

    if failed_count > 0:
        print(f"Warning: Failed to generate paraphrases for {failed_count}/{len(questions)} questions")

    return all_paraphrases


@hydra.main(version_base=None, config_path="config", config_name="paraphrase_analysis")
def main(cfg: DictConfig):
    # Device setup - Only run on main process to avoid OOM
    local_rank = int(os.environ.get('LOCAL_RANK', 0))

    # Only run on the first process to avoid duplicate work and OOM
    if local_rank != 0:
        return

    if torch.cuda.is_available():
        # Use GPU 0 for full_model and GPU 1 for retain_model
        device_full = torch.device('cuda:0')
        device_retain = torch.device('cuda:1')
        print(f"Using GPU 0 for full_model and GPU 1 for retain_model")
    else:
        device_full = torch.device('cpu')
        device_retain = torch.device('cpu')

    os.makedirs(cfg.analysis.output_dir, exist_ok=True)

    # Load models and tokenizer
    full_model, retain_model, tokenizer = load_models_and_tokenizer(
        cfg.model_family,
        full_model_path=cfg.get('full_model_path'),
        retain_model_path=cfg.get('retain_model_path')
    )

    # Place models on different GPUs to avoid OOM
    print("Loading full_model to GPU 0...")
    full_model = full_model.to(device_full)
    print("Loading retain_model to GPU 1...")
    retain_model = retain_model.to(device_retain)

    # Load dataset
    dataset = TextForgetDatasetQA(
        data_path=cfg.data_path,
        tokenizer=tokenizer,
        model_family=cfg.model_family,
        max_length=cfg.generation.max_length,
        split=cfg.split
    )

    # Create analyzer and generator
    analyzer = DualModelAnalyzer(
        batch_size=cfg.analysis.batch_size,
        output_dir=cfg.analysis.output_dir,
        model_config=cfg
    )
    generator = ParaphraseGenerator(cfg)

    # STEP 1: Analyze original dataset
    print("\n" + "="*60)
    print("STEP 1: Analyzing original dataset...")
    print("="*60)
    orig_results = analyzer.run_dual_model_analysis(
        full_model=full_model,
        retain_model=retain_model,
        tokenizer=tokenizer,
        dataset=dataset,
        tag="original"
    )

    # Save original results in notebook format
    print("\nSaving original results...")
    save_results_for_notebook(
        original_results=orig_results,
        paraphrase_results=[],
        output_dir=cfg.analysis.output_dir,
        data_path=cfg.data_path,
        split=cfg.split
    )

    # STEP 2: Generate paraphrases using ParaphraseGenerator
    print("\n" + "="*60)
    print("STEP 2: Generating paraphrases...")
    print("="*60)
    questions = [r['question'] for r in orig_results]
    all_paraphrases = generate_paraphrases_for_questions(
        questions=questions,
        generator=generator,
        num_paraphrases=cfg.analysis.num_paraphrases,
        model=full_model,
        tokenizer=tokenizer
    )

    # STEP 3: Create paraphrased dataset and analyze
    print("\n" + "="*60)
    print("STEP 3: Analyzing paraphrased dataset...")
    print("="*60)
    para_dataset = create_paraphrased_dataset(dataset, all_paraphrases, tokenizer, cfg.model_family)

    para_results = analyzer.run_dual_model_analysis(
        full_model=full_model,
        retain_model=retain_model,
        tokenizer=tokenizer,
        dataset=para_dataset,
        tag="paraphrase"
    )

    # STEP 4: Save paraphrase results
    print("\n" + "="*60)
    print("STEP 4: Saving paraphrase results...")
    print("="*60)
    save_results_for_notebook(
        original_results=orig_results,
        paraphrase_results=para_results,
        output_dir=cfg.analysis.output_dir,
        data_path=cfg.data_path,
        split=cfg.split
    )

    # Also save combined results for internal use
    combined_results = analyzer.combine_and_save_results(
        original_results=orig_results,
        paraphrase_results=para_results,
        tag="combined"
    )

    print("\n" + "="*60)
    print("Analysis complete!")
    print(f"Results saved to: {cfg.analysis.output_dir}")
    print("="*60)


if __name__ == "__main__":
    main()

# master_port=29500
# CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port=$master_port paraphrase_main.py --config-name=paraphrase_analysis