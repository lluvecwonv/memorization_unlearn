"""
메모라이제이션 분석 전용 모듈
"""

import torch
import numpy as np
from typing import Dict, List, Any
from local_utils.logger import get_logger
# 임베딩 제거 - compute_generation_embedding 불필요

logger = get_logger(__name__)


class MemorizationAnalyzer:
    """메모라이제이션 측정 및 분석을 담당하는 클래스"""
    
    def __init__(self, config):
        self.config = config
    
    def analyze_beam_search_memorization(self, beam_paraphrases: List[Dict], original_response: str, model, tokenizer, response_generator) -> List[Dict]:
        """빔서치 패러프레이즈들에 대한 메모라이제이션 분석 (임베딩 제거)"""
        results = []
        if not beam_paraphrases:
            return results
        
        for j, beam_para in enumerate(beam_paraphrases):
            beam_text = beam_para.get('text', '')
            if not beam_text:
                continue
                
            # 응답 생성
            beam_response = response_generator.generate_response_for_paraphrase(beam_text, model, tokenizer)
            if not beam_response:
                continue
                
            # 간단한 텍스트 유사도 측정 (임베딩 대신)
            text_similarity = self._calculate_text_similarity(original_response, beam_response)
            
            results.append({
                'paraphrase_text': beam_text,
                'generated_response': beam_response,
                'text_similarity': text_similarity,
                'confidence_score': beam_para.get('confidence_score', 0.0),
                'paraphrase_rank': j + 1,
                'method': 'beam_search'
            })
            
        return results
    
    def analyze_prompt_memorization(self, prompt_paraphrases: List[Dict], original_response: str, model, tokenizer, response_generator) -> List[Dict]:
        """프롬프트 기반 패러프레이즈들에 대한 메모라이제이션 분석 (임베딩 제거)"""
        results = []
        if not prompt_paraphrases:
            return results
        
        for j, prompt_para in enumerate(prompt_paraphrases):
            prompt_text = prompt_para.get('text', '')
            if not prompt_text:
                continue
                
            # 응답 생성
            prompt_response = response_generator.generate_response_for_paraphrase(prompt_text, model, tokenizer)
            if not prompt_response:
                continue
                
            # 간단한 텍스트 유사도 측정 (임베딩 대신)
            text_similarity = self._calculate_text_similarity(original_response, prompt_response)
            
            results.append({
                'paraphrase_text': prompt_text,
                'generated_response': prompt_response,
                'text_similarity': text_similarity,
                'method': 'prompt_based',
                'paraphrase_rank': j + 1
            })
            
        return results
    
    def analyze_fallback_memorization(self, original_response: str, paraphrase_response: str, model, tokenizer) -> Dict[str, float]:
        """Fallback: 데이터셋 기반 메모라이제이션 분석 (임베딩 제거)"""
        text_similarity = self._calculate_text_similarity(original_response, paraphrase_response)
        
        return {
            'text_similarity': text_similarity
        }
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """간단한 텍스트 유사도 계산 (임베딩 대신 사용)"""
        if not text1 or not text2:
            return 0.0
        
        # 간단한 단어 기반 Jaccard 유사도
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0
    
    def get_main_result(self, beam_results: List[Dict], prompt_results: List[Dict], fallback_result: Dict[str, float]) -> Dict[str, Any]:
        """주 분석 결과 선택 (빔서치 우선, 없으면 프롬프트, 둘 다 없으면 fallback)"""
        if beam_results:
            main_result = beam_results[0]
            return {
                'text_similarity': main_result['text_similarity'],
                'generated_response': main_result.get('response', main_result.get('generated_response', '')),
                'source': 'beam_search'
            }
        elif prompt_results:
            main_result = prompt_results[0]
            return {
                'text_similarity': main_result['text_similarity'],
                'generated_response': main_result.get('response', main_result.get('generated_response', '')),
                'source': 'prompt_based'
            }
        else:
            return {
                'text_similarity': fallback_result['text_similarity'],
                'generated_response': None,
                'source': 'fallback'
            }