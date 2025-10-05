"""
Paraphrase Generation Debugging and Storage Module

This module provides detailed debugging, logging, and storage capabilities 
for paraphrase generation process.
"""

import os
import json
import csv
import pandas as pd
from datetime import datetime
from typing import List, Dict, Any, Tuple
import logging
import torch

logger = logging.getLogger(__name__)


class ParaphraseDebugger:
    """Debug and store paraphrase generation process"""
    
    def __init__(self, output_dir: str = "debug_paraphrases"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize debug storage
        self.debug_data = []
        self.generation_stats = {
            'total_attempts': 0,
            'successful_generations': 0,
            'failed_generations': 0,
            'quality_filtered': 0,
            'duplicate_filtered': 0
        }
        
        # Create session timestamp
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        logger.info(f"Initialized ParaphraseDebugger with session ID: {self.session_id}")
    
    def log_generation_attempt(
        self,
        sample_index: int,
        original_question: str,
        prompt_used: str,
        prompt_index: int,
        generated_text: str,
        success: bool,
        filter_reason: str = None,
        generation_params: Dict = None
    ):
        """Log each paraphrase generation attempt"""
        
        self.generation_stats['total_attempts'] += 1
        
        debug_entry = {
            'session_id': self.session_id,
            'timestamp': datetime.now().isoformat(),
            'sample_index': sample_index,
            'original_question': original_question,
            'prompt_used': prompt_used,
            'prompt_index': prompt_index,
            'generated_text': generated_text,
            'success': success,
            'filter_reason': filter_reason,
            'generation_params': generation_params or {},
            'text_length': len(generated_text) if generated_text else 0,
            'words_count': len(generated_text.split()) if generated_text else 0
        }
        
        self.debug_data.append(debug_entry)
        
        # Update stats
        if success:
            self.generation_stats['successful_generations'] += 1
        else:
            self.generation_stats['failed_generations'] += 1
            if filter_reason:
                if 'quality' in filter_reason.lower():
                    self.generation_stats['quality_filtered'] += 1
                elif 'duplicate' in filter_reason.lower():
                    self.generation_stats['duplicate_filtered'] += 1
        
        # Real-time logging for important events
        if not success and filter_reason:
            logger.debug(f"Sample {sample_index}, Prompt {prompt_index}: {filter_reason}")
            logger.debug(f"  Original: {original_question[:50]}...")
            logger.debug(f"  Generated: {generated_text[:50]}..." if generated_text else "  Generated: [EMPTY]")
    
    def log_sample_summary(
        self,
        sample_index: int,
        original_question: str,
        dataset_paraphrase: str,
        generated_paraphrases: List[str],
        final_paraphrases_used: List[str]
    ):
        """Log summary for each sample"""
        
        summary_entry = {
            'session_id': self.session_id,
            'sample_index': sample_index,
            'original_question': original_question,
            'dataset_paraphrase': dataset_paraphrase,
            'generated_paraphrases': generated_paraphrases,
            'final_paraphrases_used': final_paraphrases_used,
            'generation_success_rate': len(generated_paraphrases) / 3.0 if generated_paraphrases else 0.0,  # Assuming 3 attempts
            'timestamp': datetime.now().isoformat()
        }
        
        # Log to console for monitoring
        logger.info(f"Sample {sample_index} Summary:")
        logger.info(f"  Original: {original_question[:60]}...")
        logger.info(f"  Dataset Para: {(dataset_paraphrase[:60] + '...') if dataset_paraphrase else 'None'}")
        logger.info(f"  Generated: {len(generated_paraphrases)} paraphrases")
        logger.info(f"  Final Used: {len(final_paraphrases_used)} paraphrases")
        
        # Store detailed sample data
        sample_debug_file = os.path.join(self.output_dir, f"sample_{sample_index:03d}_{self.session_id}.json")
        with open(sample_debug_file, 'w', encoding='utf-8') as f:
            json.dump(summary_entry, f, ensure_ascii=False, indent=2)
    
    def save_debug_session(self, prefix: str = "paraphrase_debug") -> Tuple[str, str, str]:
        """Save complete debugging session data"""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 1. Save detailed CSV
        csv_path = os.path.join(self.output_dir, f"{prefix}_detailed_{timestamp}.csv")
        df = pd.DataFrame(self.debug_data)
        df.to_csv(csv_path, index=False, encoding='utf-8')
        
        # 2. Save JSON with stats
        json_path = os.path.join(self.output_dir, f"{prefix}_session_{timestamp}.json")
        session_data = {
            'session_info': {
                'session_id': self.session_id,
                'timestamp': datetime.now().isoformat(),
                'total_debug_entries': len(self.debug_data)
            },
            'generation_stats': self.generation_stats,
            'detailed_entries': self.debug_data
        }
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(session_data, f, ensure_ascii=False, indent=2)
        
        # 3. Save human-readable report
        report_path = os.path.join(self.output_dir, f"{prefix}_report_{timestamp}.txt")
        self._save_readable_report(report_path)
        
        logger.info(f"Debug session saved:")
        logger.info(f"  📊 Detailed CSV: {csv_path}")
        logger.info(f"  📋 JSON Data: {json_path}")
        logger.info(f"  📄 Report: {report_path}")
        
        return csv_path, json_path, report_path
    
    def _save_readable_report(self, report_path: str):
        """Save human-readable debug report"""
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("PARAPHRASE GENERATION DEBUG REPORT\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"Session ID: {self.session_id}\n")
            f.write(f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Statistics Summary
            f.write("📊 GENERATION STATISTICS\n")
            f.write("-" * 40 + "\n")
            stats = self.generation_stats
            f.write(f"Total Attempts: {stats['total_attempts']}\n")
            f.write(f"Successful Generations: {stats['successful_generations']}\n")
            f.write(f"Failed Generations: {stats['failed_generations']}\n")
            f.write(f"Quality Filtered: {stats['quality_filtered']}\n")
            f.write(f"Duplicate Filtered: {stats['duplicate_filtered']}\n")
            
            if stats['total_attempts'] > 0:
                success_rate = stats['successful_generations'] / stats['total_attempts']
                f.write(f"Success Rate: {success_rate:.2%}\n")
            
            f.write("\n")
            
            # Sample-by-sample analysis
            f.write("📋 SAMPLE-BY-SAMPLE ANALYSIS\n")
            f.write("-" * 40 + "\n")
            
            # Group by sample
            samples = {}
            for entry in self.debug_data:
                sample_idx = entry['sample_index']
                if sample_idx not in samples:
                    samples[sample_idx] = []
                samples[sample_idx].append(entry)
            
            for sample_idx in sorted(samples.keys()):
                sample_entries = samples[sample_idx]
                original_q = sample_entries[0]['original_question']
                
                f.write(f"\nSample {sample_idx}:\n")
                f.write(f"  Original: {original_q}\n")
                
                successful = [e for e in sample_entries if e['success']]
                failed = [e for e in sample_entries if not e['success']]
                
                f.write(f"  Attempts: {len(sample_entries)} | Success: {len(successful)} | Failed: {len(failed)}\n")
                
                # Show successful generations
                if successful:
                    f.write("  ✅ Successful Generations:\n")
                    for i, entry in enumerate(successful, 1):
                        f.write(f"    {i}. {entry['generated_text']}\n")
                
                # Show failed generations with reasons
                if failed:
                    f.write("  ❌ Failed Generations:\n")
                    for i, entry in enumerate(failed, 1):
                        reason = entry.get('filter_reason', 'Unknown')
                        text = entry.get('generated_text', '[EMPTY]')
                        f.write(f"    {i}. [{reason}] {text[:50]}...\n")
            
            f.write("\n" + "=" * 80 + "\n")
            f.write("END OF DEBUG REPORT\n")
            f.write("=" * 80 + "\n")
    
    def get_generation_stats(self) -> Dict[str, Any]:
        """Get current generation statistics"""
        return self.generation_stats.copy()
    
    def print_session_summary(self):
        """Print session summary to console"""
        
        stats = self.generation_stats
        logger.info("🔍 PARAPHRASE GENERATION SESSION SUMMARY")
        logger.info("=" * 50)
        logger.info(f"Session ID: {self.session_id}")
        logger.info(f"Total Attempts: {stats['total_attempts']}")
        logger.info(f"Successful: {stats['successful_generations']}")
        logger.info(f"Failed: {stats['failed_generations']}")
        
        if stats['total_attempts'] > 0:
            success_rate = stats['successful_generations'] / stats['total_attempts']
            logger.info(f"Success Rate: {success_rate:.2%}")
        
        logger.info(f"Quality Filtered: {stats['quality_filtered']}")
        logger.info(f"Duplicate Filtered: {stats['duplicate_filtered']}")
        logger.info("=" * 50)


class EnhancedParaphraseGenerator:
    """Enhanced paraphrase generator with debugging capabilities"""
    
    def __init__(self, model, tokenizer, device='cuda', debugger: ParaphraseDebugger = None):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.debugger = debugger or ParaphraseDebugger()
    
    def generate_paraphrases_with_debug(
        self, 
        sample_index: int,
        question: str, 
        n_paraphrases: int = 3
    ) -> List[str]:
        """Generate paraphrases with detailed debugging"""
        
        paraphrases = []
        
        # Different prompting strategies for variety
        prompts = [
            f"Paraphrase this question: {question}\nParaphrased version:",
            f"Rephrase the following question using different words: {question}\nRephrased question:",
            f"Ask the same question in a different way: {question}\nAlternative question:",
            f"Rewrite this question while keeping the same meaning: {question}\nRewritten question:"
        ]
        
        generation_params = {
            'max_new_tokens': 80,
            'do_sample': True,
            'temperature': 0.8,
            'top_p': 0.9,
            'repetition_penalty': 1.1
        }
        
        logger.debug(f"Starting paraphrase generation for sample {sample_index}")
        
        for i in range(min(n_paraphrases, len(prompts))):
            prompt = prompts[i]
            generated_text = ""
            success = False
            filter_reason = None
            
            try:
                inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        **generation_params,
                        pad_token_id=self.tokenizer.eos_token_id,
                        eos_token_id=self.tokenizer.eos_token_id
                    )
                
                # Extract only the generated part
                generated = outputs[:, inputs['input_ids'].shape[1]:]
                generated_text = self.tokenizer.decode(generated[0], skip_special_tokens=True).strip()
                
                # Clean up the generated text
                generated_text = generated_text.split('\n')[0]  # Take only first line
                generated_text = generated_text.strip('.,!?;: ')
                
                # Quality checks with detailed logging
                if not generated_text:
                    filter_reason = "Empty generation"
                elif len(generated_text) < 10:
                    filter_reason = f"Too short ({len(generated_text)} chars)"
                elif len(generated_text) > 200:
                    filter_reason = f"Too long ({len(generated_text)} chars)"
                elif generated_text.lower() == question.lower():
                    filter_reason = "Identical to original"
                elif generated_text in paraphrases:
                    filter_reason = "Duplicate paraphrase"
                else:
                    # Additional quality checks
                    words_original = set(question.lower().split())
                    words_generated = set(generated_text.lower().split())
                    overlap_ratio = len(words_original & words_generated) / len(words_original | words_generated)
                    
                    if overlap_ratio > 0.9:
                        filter_reason = f"Too similar (overlap: {overlap_ratio:.2f})"
                    elif overlap_ratio < 0.3:
                        filter_reason = f"Too different (overlap: {overlap_ratio:.2f})"
                    else:
                        success = True
                        paraphrases.append(generated_text)
                
            except Exception as e:
                filter_reason = f"Generation error: {str(e)}"
                logger.warning(f"Generation failed for sample {sample_index}, prompt {i}: {e}")
            
            # Log this attempt
            self.debugger.log_generation_attempt(
                sample_index=sample_index,
                original_question=question,
                prompt_used=prompt,
                prompt_index=i,
                generated_text=generated_text,
                success=success,
                filter_reason=filter_reason,
                generation_params=generation_params
            )
        
        # If we couldn't generate enough good paraphrases, create simple variations
        original_count = len(paraphrases)
        while len(paraphrases) < n_paraphrases:
            simple_variations = [
                f"What is the answer to: {question}",
                f"Can you tell me about {question.replace('?', '').lower()}?",
                f"Please explain {question.replace('?', '').lower()}"
            ]
            
            for variation in simple_variations:
                if len(paraphrases) < n_paraphrases and variation not in paraphrases:
                    paraphrases.append(variation)
                    
                    # Log simple variation
                    self.debugger.log_generation_attempt(
                        sample_index=sample_index,
                        original_question=question,
                        prompt_used="Simple variation",
                        prompt_index=-1,
                        generated_text=variation,
                        success=True,
                        filter_reason="Simple fallback variation"
                    )
                    break
            
            if len(paraphrases) == original_count:  # No progress
                fallback = f"Alternative: {question}"
                paraphrases.append(fallback)
                
                self.debugger.log_generation_attempt(
                    sample_index=sample_index,
                    original_question=question,
                    prompt_used="Fallback",
                    prompt_index=-1,
                    generated_text=fallback,
                    success=True,
                    filter_reason="Final fallback"
                )
                break
        
        logger.info(f"Generated {len(paraphrases)} paraphrases for sample {sample_index}")
        return paraphrases[:n_paraphrases]