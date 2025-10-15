import torch
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class ParaphraseGenerator:
    """Generate paraphrases of questions using various strategies"""

    def __init__(self, config):
        """Initialize ParaphraseGenerator

        Args:
            config: Configuration object with paraphrasing settings
        """
        self.config = config

        # Extract paraphrasing settings
        self.num_paraphrases = getattr(config.analysis, 'num_paraphrases', 3)
        self.num_beams = getattr(config.analysis, 'num_beams', 5)
        self.temperature = getattr(config.analysis, 'paraphrase_temperature', 0.7)

        # Check if prompt-based paraphrasing is enabled
        self.prompt_paraphrasing_enabled = getattr(config.analysis, 'use_prompt_paraphrasing', False)

        logger.info(f"ParaphraseGenerator initialized: num_paraphrases={self.num_paraphrases}, "
                   f"num_beams={self.num_beams}, prompt_enabled={self.prompt_paraphrasing_enabled}")

    def generate_beam_paraphrases(self, question: str, model=None, tokenizer=None) -> List[Dict[str, Any]]:
        """Generate paraphrases using beam search

        Args:
            question: Original question text
            model: Optional model for generation (if None, uses simple heuristics)
            tokenizer: Optional tokenizer (if None, uses simple heuristics)

        Returns:
            List of paraphrase dictionaries with 'text' and 'score' fields
        """
        if model is None or tokenizer is None:
            # Fallback to simple rule-based paraphrasing
            logger.warning("No model/tokenizer provided for beam search, using fallback heuristics")
            return self._generate_heuristic_paraphrases(question)

        try:
            paraphrases = []

            # Create paraphrase prompt
            prompt = f"Rephrase: {question}\nRephrase:"

            # Tokenize
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

            with torch.no_grad():
                # Generate with beam search
                outputs = model.generate(
                    inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    max_new_tokens=len(question.split()) + 10,
                    min_new_tokens=5,
                    num_beams=self.num_beams,
                    num_return_sequences=min(self.num_paraphrases, self.num_beams),
                    do_sample=False,
                    early_stopping=True,
                    pad_token_id=tokenizer.eos_token_id,
                    repetition_penalty=1.2
                )

            # Extract paraphrases from outputs
            input_length = inputs.input_ids.shape[-1]

            for idx, output in enumerate(outputs):
                generated_tokens = output[input_length:]
                paraphrase_text = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

                # Clean up the paraphrase
                paraphrase_text = paraphrase_text.replace("Rephrase:", "").strip()
                paraphrase_text = paraphrase_text.replace("->", "").strip()

                if paraphrase_text and len(paraphrase_text.split()) >= 3:
                    paraphrases.append({
                        'text': paraphrase_text,
                        'score': 1.0 / (idx + 1),  # Higher score for earlier beam results
                        'method': 'beam_search'
                    })

            logger.debug(f"Generated {len(paraphrases)} beam search paraphrases")
            return paraphrases[:self.num_paraphrases]

        except Exception as e:
            logger.error(f"Beam search paraphrasing failed: {e}")
            return self._generate_heuristic_paraphrases(question)

    def generate_prompt_paraphrases(self, question: str, model, tokenizer) -> List[Dict[str, Any]]:
        """Generate paraphrases using prompt-based generation

        Args:
            question: Original question text
            model: Language model for generation
            tokenizer: Tokenizer

        Returns:
            List of paraphrase dictionaries with 'text' and 'score' fields
        """
        if not self.prompt_paraphrasing_enabled:
            logger.debug("Prompt paraphrasing is disabled")
            return []

        try:
            paraphrases = []

            # Different prompt templates for diversity
            prompt_templates = [
                f"Rewrite the following question in a different way:\nQuestion: {question}\nRewritten question:",
                f"Paraphrase this question:\n{question}\nParaphrase:",
                f"Say the same thing differently:\n{question}\nDifferent version:",
            ]

            for template_idx, prompt in enumerate(prompt_templates[:self.num_paraphrases]):
                # Tokenize
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token

                inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

                with torch.no_grad():
                    outputs = model.generate(
                        inputs.input_ids,
                        attention_mask=inputs.attention_mask,
                        max_new_tokens=len(question.split()) + 15,
                        do_sample=True,
                        temperature=self.temperature,
                        top_p=0.9,
                        top_k=50,
                        pad_token_id=tokenizer.eos_token_id,
                        eos_token_id=tokenizer.eos_token_id
                    )

                # Decode
                input_length = inputs.input_ids.shape[-1]
                generated_tokens = outputs[0][input_length:]
                paraphrase_text = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

                # Clean up
                paraphrase_text = self._clean_paraphrase(paraphrase_text)

                if paraphrase_text and len(paraphrase_text.split()) >= 3:
                    paraphrases.append({
                        'text': paraphrase_text,
                        'score': 1.0,
                        'method': 'prompt_based'
                    })

            logger.debug(f"Generated {len(paraphrases)} prompt-based paraphrases")
            return paraphrases

        except Exception as e:
            logger.error(f"Prompt-based paraphrasing failed: {e}")
            return []

    def _generate_heuristic_paraphrases(self, question: str) -> List[Dict[str, Any]]:
        """Generate simple rule-based paraphrases as fallback

        Args:
            question: Original question text

        Returns:
            List of paraphrase dictionaries
        """
        paraphrases = []

        # Simple rule-based transformations
        rules = [
            (r"What is", "What's"),
            (r"Who is", "Who's"),
            (r"can you tell me", "tell me"),
            (r"Could you", "Can you"),
            (r"Would you", "Can you"),
            (r"\?", ""),  # Remove question mark
        ]

        # Apply each rule
        for i, (pattern, replacement) in enumerate(rules[:self.num_paraphrases]):
            import re
            paraphrase_text = re.sub(pattern, replacement, question, flags=re.IGNORECASE)

            if paraphrase_text != question:
                paraphrases.append({
                    'text': paraphrase_text,
                    'score': 0.5,
                    'method': 'heuristic'
                })

        # If no rules applied, just return original with slight modification
        if not paraphrases:
            paraphrases.append({
                'text': question,
                'score': 1.0,
                'method': 'original'
            })

        logger.debug(f"Generated {len(paraphrases)} heuristic paraphrases")
        return paraphrases[:self.num_paraphrases]

    def _clean_paraphrase(self, text: str) -> str:
        """Clean up generated paraphrase text

        Args:
            text: Raw paraphrase text

        Returns:
            Cleaned paraphrase text
        """
        # Remove common artifacts
        text = text.replace("Rephrase:", "").strip()
        text = text.replace("Rewritten question:", "").strip()
        text = text.replace("Paraphrase:", "").strip()
        text = text.replace("Different version:", "").strip()
        text = text.replace("->", "").strip()

        # Remove multiple spaces
        import re
        text = re.sub(r'\s+', ' ', text).strip()

        return text
