import os
import torch
import warnings
import hydra
from omegaconf import DictConfig

from data_module import TextForgetDatasetQA
from memorization.analysis.paraphrase_analyzer import DualModelAnalyzer
from memorization.analysis.paraphrase_generator import ParaphraseGenerator
from memorization.utils import load_models_and_tokenizer, create_paraphrased_dataset


def generate_paraphrases_for_questions(questions, generator, num_paraphrases, model=None, tokenizer=None):
    """Generate paraphrases for a list of questions

    Args:
        questions: List of questions to paraphrase
        generator: ParaphraseGenerator instance
        num_paraphrases: Number of paraphrases to generate per question
        model: Optional model for prompt-based generation
        tokenizer: Optional tokenizer for prompt-based generation

    Returns:
        List of paraphrase lists, one per question
    """
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
    # Device setup
    if torch.cuda.is_available():
        local_rank = cfg.get('local_rank', int(os.environ.get('LOCAL_RANK', 0)))
        torch.cuda.set_device(local_rank)
        device = torch.device(f'cuda:{local_rank}')
    else:
        device = torch.device('cpu')

    os.makedirs(cfg.analysis.output_dir, exist_ok=True)

    # Load models and tokenizer
    full_model, retain_model, tokenizer = load_models_and_tokenizer(
        cfg.model_family,
        full_model_path=cfg.get('full_model_path'),
        retain_model_path=cfg.get('retain_model_path')
    )
    full_model = full_model.to(device)
    retain_model = retain_model.to(device)

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
        model_config=cfg.model
    )
    generator = ParaphraseGenerator(cfg)

    # STEP 1: Analyze original dataset
    orig_results = analyzer.run_dual_model_analysis(
        full_model=full_model,
        retain_model=retain_model,
        tokenizer=tokenizer,
        dataset=dataset,
        tag="original"
    )

    # STEP 2: Generate paraphrases using ParaphraseGenerator
    questions = [r['question'] for r in orig_results]
    all_paraphrases = generate_paraphrases_for_questions(
        questions=questions,
        generator=generator,
        num_paraphrases=cfg.analysis.num_paraphrases,
        model=full_model,
        tokenizer=tokenizer
    )

    # STEP 3: Create paraphrased dataset and analyze
    para_dataset = create_paraphrased_dataset(dataset, all_paraphrases, tokenizer, cfg.model_family)

    para_results = analyzer.run_dual_model_analysis(
        full_model=full_model,
        retain_model=retain_model,
        tokenizer=tokenizer,
        dataset=para_dataset,
        tag="paraphrase"
    )

    # STEP 4: Combine and save results
    combined_results = analyzer.combine_and_save_results(
        original_results=orig_results,
        paraphrase_results=para_results,
        tag="combined"
    )


if __name__ == "__main__":
    main()
