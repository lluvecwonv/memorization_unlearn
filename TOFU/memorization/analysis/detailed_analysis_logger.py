"""
상세 분석 결과 로깅 및 저장 모듈
원본과 패러프레이즈 각각의 결과를 추적하고 분석
"""

import json
import pandas as pd
from typing import Dict, List, Any, Optional
from pathlib import Path
import torch
import numpy as np

from local_utils.logger import get_logger

logger = get_logger(__name__)


class DetailedAnalysisLogger:
    """상세 분석 결과를 로깅하고 저장하는 클래스"""
    
    def __init__(self, config, tag: str = "analysis"):
        self.config = config
        self.tag = tag
        
        # 로그 데이터 저장소
        self.detailed_logs = {
            'original_responses': [],
            'beam_paraphrase_responses': [],
            'prompt_paraphrase_responses': [],
            'response_comparisons': [],
            'memorization_analysis': [],
            'generation_metadata': [],
            'individual_paraphrase_results': []  # 새로운 개별 결과 저장
        }
        
        # 출력 디렉토리 설정
        self.output_dir = Path(f"detailed_analysis_{tag}")
        self.output_dir.mkdir(exist_ok=True)
        
        logger.info(f"📝 DetailedAnalysisLogger initialized for {tag}")
    
    def log_original_response(self, sample_idx: int, question: str, response: str, 
                             generation_metadata: Dict[str, Any]):
        """원본 질문-응답 쌍 로깅"""
        
        log_entry = {
            'sample_idx': sample_idx,
            'question': question,
            'response': response,
            'response_length': len(response),
            'question_length': len(question),
            'generation_time': generation_metadata.get('generation_time', 0),
            'model_config': {
                'max_new_tokens': getattr(self.config.model, 'max_new_tokens', None),
                'temperature': getattr(self.config.model, 'temperature', None),
                'do_sample': getattr(self.config.model, 'do_sample', None)
            },
            'timestamp': pd.Timestamp.now().isoformat()
        }
        
        self.detailed_logs['original_responses'].append(log_entry)
        logger.debug(f"📝 Logged original response for sample {sample_idx}")
    
    def log_beam_paraphrase_responses(self, sample_idx: int, original_question: str,
                                     paraphrases: List[Dict], responses: List[Dict]):
        """빔서치 패러프레이즈 응답들 로깅"""
        
        for i, (paraphrase, response) in enumerate(zip(paraphrases, responses)):
            log_entry = {
                'sample_idx': sample_idx,
                'original_question': original_question,
                'paraphrase_idx': i,
                'paraphrase_text': paraphrase.get('text', ''),
                'paraphrase_confidence': paraphrase.get('confidence_score', 0.0),
                'generated_response': response.get('generated_response', ''),
                'cosine_similarity': response.get('cosine_similarity', 0.0),
                'euclidean_distance': response.get('euclidean_distance', 0.0),
                'method': 'beam_search',
                'paraphrase_length': len(paraphrase.get('text', '')),
                'response_length': len(response.get('generated_response', '')),
                'timestamp': pd.Timestamp.now().isoformat()
            }
            
            self.detailed_logs['beam_paraphrase_responses'].append(log_entry)
        
        logger.debug(f"📝 Logged {len(responses)} beam paraphrase responses for sample {sample_idx}")
    
    def log_prompt_paraphrase_responses(self, sample_idx: int, original_question: str,
                                       paraphrases: List[Dict], responses: List[Dict]):
        """프롬프트 기반 패러프레이즈 응답들 로깅"""
        
        for i, (paraphrase, response) in enumerate(zip(paraphrases, responses)):
            log_entry = {
                'sample_idx': sample_idx,
                'original_question': original_question,
                'paraphrase_idx': i,
                'paraphrase_text': paraphrase.get('text', ''),
                'paraphrase_method': paraphrase.get('method', 'prompt_based'),
                'generated_response': response.get('generated_response', ''),
                'cosine_similarity': response.get('cosine_similarity', 0.0),
                'euclidean_distance': response.get('euclidean_distance', 0.0),
                'method': 'prompt_based',
                'paraphrase_length': len(paraphrase.get('text', '')),
                'response_length': len(response.get('generated_response', '')),
                'timestamp': pd.Timestamp.now().isoformat()
            }
            
            self.detailed_logs['prompt_paraphrase_responses'].append(log_entry)
        
        logger.debug(f"📝 Logged {len(responses)} prompt paraphrase responses for sample {sample_idx}")
    
    def log_response_comparison(self, sample_idx: int, original_question: str, original_response: str,
                               paraphrase_method: str, paraphrase_text: str, paraphrase_response: str,
                               similarity_metrics: Dict[str, float]):
        """응답 비교 결과 로깅"""
        
        comparison_entry = {
            'sample_idx': sample_idx,
            'original_question': original_question,
            'original_response': original_response,
            'paraphrase_method': paraphrase_method,
            'paraphrase_text': paraphrase_text,
            'paraphrase_response': paraphrase_response,
            'cosine_similarity': similarity_metrics.get('cosine_similarity', 0.0),
            'euclidean_distance': similarity_metrics.get('euclidean_distance', 0.0),
            'response_length_diff': len(paraphrase_response) - len(original_response),
            'question_length_diff': len(paraphrase_text) - len(original_question),
            'is_identical_response': original_response == paraphrase_response,
            'response_overlap_ratio': self._calculate_text_overlap(original_response, paraphrase_response),
            'timestamp': pd.Timestamp.now().isoformat()
        }
        
        self.detailed_logs['response_comparisons'].append(comparison_entry)
        logger.debug(f"📝 Logged response comparison for sample {sample_idx} ({paraphrase_method})")
    
    def log_individual_paraphrase_result(self, sample_idx: int, method: str, paraphrase_idx: int,
                                       original_question: str, paraphrase_text: str, 
                                       original_response: str, paraphrase_response: str,
                                       cosine_similarity: float, euclidean_distance: float):
        """개별 패러프레이즈 결과를 1:1 매칭으로 실시간 저장"""
        
        individual_result = {
            'sample_idx': sample_idx,
            'method': method,  # 'beam_search' or 'prompt_based'
            'paraphrase_idx': paraphrase_idx,
            'original_question': original_question,
            'paraphrase_text': paraphrase_text,
            'original_response': original_response,
            'paraphrase_response': paraphrase_response,
            'cosine_similarity': cosine_similarity,
            'euclidean_distance': euclidean_distance,
            'question_length': len(original_question),
            'paraphrase_length': len(paraphrase_text),
            'original_response_length': len(original_response),
            'paraphrase_response_length': len(paraphrase_response),
            'response_length_diff': len(paraphrase_response) - len(original_response),
            'question_length_diff': len(paraphrase_text) - len(original_question),
            'is_identical_response': original_response == paraphrase_response,
            'response_overlap_ratio': self._calculate_text_overlap(original_response, paraphrase_response),
            'question_overlap_ratio': self._calculate_text_overlap(original_question, paraphrase_text),
            'timestamp': pd.Timestamp.now().isoformat()
        }
        
        self.detailed_logs['individual_paraphrase_results'].append(individual_result)
        
        # 실시간 CSV 저장 (옵션)
        if len(self.detailed_logs['individual_paraphrase_results']) % 10 == 0:
            self._save_individual_results_incremental()
        
        logger.debug(f"💾 Saved individual result: {method} {paraphrase_idx} for sample {sample_idx}")
    
    def _save_individual_results_incremental(self):
        """개별 결과를 점진적으로 CSV 파일에 저장"""
        if self.detailed_logs['individual_paraphrase_results']:
            df = pd.DataFrame(self.detailed_logs['individual_paraphrase_results'])
            output_file = self.output_dir / f"individual_paraphrase_results_{self.tag}.csv"
            df.to_csv(output_file, index=False)
            logger.debug(f"💾 Incremental save: {len(df)} individual results to {output_file}")
    
    def _calculate_text_overlap(self, text1: str, text2: str) -> float:
        """두 텍스트 간의 겹치는 비율 계산"""
        
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 and not words2:
            return 1.0
        if not words1 or not words2:
            return 0.0
            
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0
    
    def save_detailed_logs(self):
        """상세 로그들을 파일로 저장"""
        
        logger.info("💾 Saving detailed analysis logs...")
        
        # 1. JSON 형태로 원본 데이터 저장
        json_file = self.output_dir / f"detailed_logs_{self.tag}.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(self.detailed_logs, f, ensure_ascii=False, indent=2, default=str)
        
        # 2. CSV 형태로 각 카테고리별 저장
        for category, data in self.detailed_logs.items():
            if data:  # 데이터가 있는 경우만
                df = pd.DataFrame(data)
                csv_file = self.output_dir / f"{category}_{self.tag}.csv"
                df.to_csv(csv_file, index=False, encoding='utf-8')
                logger.info(f"📊 Saved {len(data)} records to {csv_file}")
        
        # 3. 요약 통계 생성
        self._generate_summary_report()
        
        logger.info(f"✅ All detailed logs saved to {self.output_dir}")
    
    def _generate_summary_report(self):
        """요약 통계 리포트 생성"""
        
        summary = {
            'analysis_tag': self.tag,
            'total_samples': len(self.detailed_logs['original_responses']),
            'beam_paraphrases_generated': len(self.detailed_logs['beam_paraphrase_responses']),
            'prompt_paraphrases_generated': len(self.detailed_logs['prompt_paraphrase_responses']),
            'response_comparisons_made': len(self.detailed_logs['response_comparisons']),
            'memorization_analyses': len(self.detailed_logs['memorization_analysis'])
        }
        
        # 메모라이제이션 요약 통계
        if self.detailed_logs['memorization_analysis']:
            beam_avg_similarities = [entry['beam_memorization']['avg_cosine_similarity'] 
                                   for entry in self.detailed_logs['memorization_analysis']
                                   if entry['beam_memorization']['num_paraphrases'] > 0]
            
            prompt_avg_similarities = [entry['prompt_memorization']['avg_cosine_similarity'] 
                                     for entry in self.detailed_logs['memorization_analysis']
                                     if entry['prompt_memorization']['num_paraphrases'] > 0]
            
            summary['memorization_stats'] = {
                'beam_search': {
                    'samples_with_paraphrases': len(beam_avg_similarities),
                    'avg_memorization_similarity': np.mean(beam_avg_similarities) if beam_avg_similarities else 0.0,
                    'std_memorization_similarity': np.std(beam_avg_similarities) if beam_avg_similarities else 0.0
                },
                'prompt_based': {
                    'samples_with_paraphrases': len(prompt_avg_similarities),
                    'avg_memorization_similarity': np.mean(prompt_avg_similarities) if prompt_avg_similarities else 0.0,
                    'std_memorization_similarity': np.std(prompt_avg_similarities) if prompt_avg_similarities else 0.0
                }
            }
        
        # 응답 길이 통계
        if self.detailed_logs['original_responses']:
            original_lengths = [entry['response_length'] for entry in self.detailed_logs['original_responses']]
            summary['response_length_stats'] = {
                'original_avg_length': np.mean(original_lengths),
                'original_std_length': np.std(original_lengths),
                'original_min_length': np.min(original_lengths),
                'original_max_length': np.max(original_lengths)
            }
        
        # 요약 리포트 저장
        summary_file = self.output_dir / f"analysis_summary_{self.tag}.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
        
        # 텍스트 리포트도 생성
        self._generate_text_report(summary)
        
        logger.info(f"📋 Summary report saved to {summary_file}")
    
    def _generate_text_report(self, summary: Dict[str, Any]):
        """텍스트 형태의 상세 리포트 생성"""
        
        report_file = self.output_dir / f"detailed_analysis_report_{self.tag}.txt"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write(f"DETAILED PARAPHRASE ANALYSIS REPORT ({self.tag})\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"📊 OVERVIEW:\n")
            f.write(f"  - Total samples processed: {summary['total_samples']}\n")
            f.write(f"  - Beam search paraphrases generated: {summary['beam_paraphrases_generated']}\n")
            f.write(f"  - Prompt-based paraphrases generated: {summary['prompt_paraphrases_generated']}\n")
            f.write(f"  - Response comparisons made: {summary['response_comparisons_made']}\n\n")
            
            # 메모라이제이션 통계
            if 'memorization_stats' in summary:
                f.write(f"🧠 MEMORIZATION ANALYSIS:\n")
                beam_stats = summary['memorization_stats']['beam_search']
                prompt_stats = summary['memorization_stats']['prompt_based']
                
                f.write(f"  Beam Search Memorization:\n")
                f.write(f"    - Samples with paraphrases: {beam_stats['samples_with_paraphrases']}\n")
                f.write(f"    - Average similarity: {beam_stats['avg_memorization_similarity']:.4f} ± {beam_stats['std_memorization_similarity']:.4f}\n\n")
                
                f.write(f"  Prompt-based Memorization:\n")
                f.write(f"    - Samples with paraphrases: {prompt_stats['samples_with_paraphrases']}\n")
                f.write(f"    - Average similarity: {prompt_stats['avg_memorization_similarity']:.4f} ± {prompt_stats['std_memorization_similarity']:.4f}\n\n")
            
            # 응답 길이 통계
            if 'response_length_stats' in summary:
                length_stats = summary['response_length_stats']
                f.write(f"📏 RESPONSE LENGTH STATISTICS:\n")
                f.write(f"  - Average length: {length_stats['original_avg_length']:.1f} ± {length_stats['original_std_length']:.1f}\n")
                f.write(f"  - Length range: {length_stats['original_min_length']} - {length_stats['original_max_length']}\n\n")
            
            f.write("=" * 80 + "\n")
            f.write("📁 GENERATED FILES:\n")
            f.write(f"  - JSON logs: detailed_logs_{self.tag}.json\n")
            f.write(f"  - CSV files: *_{self.tag}.csv\n")
            f.write(f"  - Summary: analysis_summary_{self.tag}.json\n")
            f.write("=" * 80 + "\n")
        
        logger.info(f"📄 Text report saved to {report_file}")
    
    def get_analysis_insights(self) -> Dict[str, Any]:
        """분석 결과에서 인사이트 추출"""
        
        insights = {
            'high_memorization_cases': [],
            'low_memorization_cases': [],
            'identical_responses': [],
            'response_length_outliers': []
        }
        
        # 높은/낮은 메모라이제이션 케이스 식별
        for comparison in self.detailed_logs['response_comparisons']:
            cos_sim = comparison['cosine_similarity']
            
            if cos_sim > 0.9:
                insights['high_memorization_cases'].append({
                    'sample_idx': comparison['sample_idx'],
                    'similarity': cos_sim,
                    'method': comparison['paraphrase_method'],
                    'question': comparison['original_question'][:100] + "..."
                })
            elif cos_sim < 0.3:
                insights['low_memorization_cases'].append({
                    'sample_idx': comparison['sample_idx'],
                    'similarity': cos_sim,
                    'method': comparison['paraphrase_method'],
                    'question': comparison['original_question'][:100] + "..."
                })
            
            if comparison['is_identical_response']:
                insights['identical_responses'].append({
                    'sample_idx': comparison['sample_idx'],
                    'method': comparison['paraphrase_method'],
                    'question': comparison['original_question'][:100] + "..."
                })
        
        return insights