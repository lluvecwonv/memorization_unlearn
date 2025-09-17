"""
선별된 샘플 데이터 내보내기 유틸리티
"""

import json
import csv
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple
import os
from datetime import datetime

from .logger import get_logger

logger = get_logger(__name__)


class SampleDataExporter:
    """선별된 샘플의 상세 정보를 내보내는 클래스"""
    
    def __init__(self, output_dir: str = "results/selected_samples"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def export_selected_samples_detailed(
        self, 
        before_results: Dict[str, Any],
        after_results: Dict[str, Any],
        top_indices: List[int],
        bottom_indices: List[int],
        save_prefix: str = "selected_samples"
    ) -> Tuple[str, str, str]:
        """
        선별된 상위/하위 샘플의 상세 정보를 여러 형식으로 저장
        
        Args:
            before_results: 언러닝 전 분석 결과
            after_results: 언러닝 후 분석 결과  
            top_indices: 상위 10개 샘플 인덱스
            bottom_indices: 하위 10개 샘플 인덱스
            save_prefix: 저장 파일명 접두사
            
        Returns:
            Tuple of (csv_path, json_path, summary_path)
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 1. 상세 데이터 수집
        detailed_data = self._collect_detailed_sample_data(
            before_results, after_results, top_indices, bottom_indices
        )
        
        # 2. CSV 형식으로 저장
        csv_path = os.path.join(self.output_dir, f"{save_prefix}_detailed_{timestamp}.csv")
        self._save_to_csv(detailed_data, csv_path)
        
        # 3. JSON 형식으로 저장 (더 구조화된 데이터)
        json_path = os.path.join(self.output_dir, f"{save_prefix}_structured_{timestamp}.json")
        self._save_to_json(detailed_data, json_path)
        
        # 4. 요약 텍스트 파일 생성
        summary_path = os.path.join(self.output_dir, f"{save_prefix}_summary_{timestamp}.txt")
        self._save_summary_report(detailed_data, summary_path)
        
        logger.info(f"📁 Exported selected samples data:")
        logger.info(f"  - CSV: {csv_path}")
        logger.info(f"  - JSON: {json_path}")
        logger.info(f"  - Summary: {summary_path}")
        
        return csv_path, json_path, summary_path
    
    def _collect_detailed_sample_data(
        self, 
        before_results: Dict[str, Any],
        after_results: Dict[str, Any], 
        top_indices: List[int],
        bottom_indices: List[int]
    ) -> List[Dict[str, Any]]:
        """선별된 샘플들의 상세 데이터 수집"""
        
        detailed_data = []
        
        # 전체 결과에서 선별된 인덱스의 데이터 추출
        before_samples = before_results.get("results", [])
        after_samples = after_results.get("results", [])
        
        before_cosine = before_results.get("cosine", [])
        before_euclidean = before_results.get("euclidean", [])
        after_cosine = after_results.get("cosine", [])
        after_euclidean = after_results.get("euclidean", [])
        
        # 상위 10개 처리
        for rank, idx in enumerate(top_indices, 1):
            sample_data = self._extract_sample_info(
                idx, rank, "TOP", 
                before_samples, after_samples,
                before_cosine, before_euclidean,
                after_cosine, after_euclidean
            )
            detailed_data.append(sample_data)
        
        # 하위 10개 처리  
        for rank, idx in enumerate(bottom_indices, 1):
            sample_data = self._extract_sample_info(
                idx, rank, "BOTTOM",
                before_samples, after_samples, 
                before_cosine, before_euclidean,
                after_cosine, after_euclidean
            )
            detailed_data.append(sample_data)
            
        return detailed_data
    
    def _extract_sample_info(
        self, 
        idx: int, 
        rank: int, 
        category: str,
        before_samples: List[Dict],
        after_samples: List[Dict],
        before_cosine: List[float],
        before_euclidean: List[float], 
        after_cosine: List[float],
        after_euclidean: List[float]
    ) -> Dict[str, Any]:
        """개별 샘플의 상세 정보 추출"""
        
        # Before 데이터
        before_sample = before_samples[idx] if idx < len(before_samples) else {}
        
        # After 데이터 (선별된 샘플 순서로 저장됨)
        after_idx = rank - 1 if category == "TOP" else len(after_samples) - rank
        after_sample = after_samples[after_idx] if after_idx < len(after_samples) else {}
        
        # 메트릭 계산
        before_cos = before_cosine[idx] if idx < len(before_cosine) else 0.0
        before_euc = before_euclidean[idx] if idx < len(before_euclidean) else 0.0
        after_cos = after_cosine[after_idx] if after_idx < len(after_cosine) else 0.0
        after_euc = after_euclidean[after_idx] if after_idx < len(after_euclidean) else 0.0
        
        return {
            "original_index": idx,
            "rank": rank,
            "category": category,
            
            # 입력 텍스트
            "question": before_sample.get("Question", ""),
            "ground_truth": before_sample.get("GroundTruth", ""),
            
            # 출력 텍스트 (Before/After)
            "response_before": before_sample.get("Predicted", ""),
            "response_after": after_sample.get("Predicted", ""),
            "paraphrased_response_before": before_sample.get("Paraphrased_Response", ""),
            "paraphrased_response_after": after_sample.get("Paraphrased_Response", ""),
            
            # 메트릭 변화
            "cosine_before": before_cos,
            "cosine_after": after_cos,
            "cosine_change": before_cos - after_cos,
            "euclidean_before": before_euc,
            "euclidean_after": after_euc,
            "euclidean_change": after_euc - before_euc
        }
    
    def _save_to_csv(self, data: List[Dict[str, Any]], filepath: str):
        """CSV 형식으로 저장"""
        df = pd.DataFrame(data)
        df.to_csv(filepath, index=False, encoding='utf-8')
        logger.info(f"💾 Saved detailed CSV: {filepath}")
    
    def _save_to_json(self, data: List[Dict[str, Any]], filepath: str):
        """JSON 형식으로 저장 (구조화된 형태)"""
        structured_data = {
            "export_timestamp": datetime.now().isoformat(),
            "total_samples": len(data),
            "top_10_samples": [item for item in data if item["category"] == "TOP"],
            "bottom_10_samples": [item for item in data if item["category"] == "BOTTOM"],
            "summary_stats": self._calculate_summary_stats(data)
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(structured_data, f, ensure_ascii=False, indent=2)
        logger.info(f"💾 Saved structured JSON: {filepath}")
    
    def _save_summary_report(self, data: List[Dict[str, Any]], filepath: str):
        """요약 텍스트 리포트 생성"""
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("선별된 샘플 상세 분석 리포트\n")
            f.write("=" * 80 + "\n")
            f.write(f"생성 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"총 샘플 수: {len(data)}\n\n")
            
            # 상위 10개 요약
            f.write("🔺 상위 10개 샘플 (높은 유사도)\n")
            f.write("-" * 40 + "\n")
            top_samples = [item for item in data if item["category"] == "TOP"]
            for sample in top_samples:
                f.write(f"순위 {sample['rank']}: 원본 인덱스 {sample['original_index']}\n")
                f.write(f"  질문: {sample['question'][:100]}...\n")
                f.write(f"  언러닝 전 응답: {sample['response_before'][:100]}...\n")
                f.write(f"  언러닝 후 응답: {sample['response_after'][:100]}...\n")
                f.write(f"  코사인 유사도 변화: {sample['cosine_change']:.4f}\n")
                f.write(f"  유클리드 거리 변화: {sample['euclidean_change']:.4f}\n\n")
            
            # 하위 10개 요약
            f.write("🔻 하위 10개 샘플 (낮은 유사도)\n")
            f.write("-" * 40 + "\n")
            bottom_samples = [item for item in data if item["category"] == "BOTTOM"]
            for sample in bottom_samples:
                f.write(f"순위 {sample['rank']}: 원본 인덱스 {sample['original_index']}\n")
                f.write(f"  질문: {sample['question'][:100]}...\n")
                f.write(f"  언러닝 전 응답: {sample['response_before'][:100]}...\n")
                f.write(f"  언러닝 후 응답: {sample['response_after'][:100]}...\n")
                f.write(f"  코사인 유사도 변화: {sample['cosine_change']:.4f}\n")
                f.write(f"  유클리드 거리 변화: {sample['euclidean_change']:.4f}\n\n")
            
            # 통계 요약
            stats = self._calculate_summary_stats(data)
            f.write("📊 통계 요약\n")
            f.write("-" * 40 + "\n")
            f.write(f"평균 코사인 유사도 변화: {stats['avg_cosine_change']:.4f}\n")
            f.write(f"평균 유클리드 거리 변화: {stats['avg_euclidean_change']:.4f}\n")
            f.write(f"최대 코사인 변화: {stats['max_cosine_change']:.4f}\n")
            f.write(f"최소 코사인 변화: {stats['min_cosine_change']:.4f}\n")
            
        logger.info(f"📄 Saved summary report: {filepath}")
    
    def _calculate_summary_stats(self, data: List[Dict[str, Any]]) -> Dict[str, float]:
        """요약 통계 계산"""
        cosine_changes = [item["cosine_change"] for item in data]
        euclidean_changes = [item["euclidean_change"] for item in data]
        
        return {
            "avg_cosine_change": np.mean(cosine_changes),
            "avg_euclidean_change": np.mean(euclidean_changes),
            "max_cosine_change": np.max(cosine_changes),
            "min_cosine_change": np.min(cosine_changes),
            "max_euclidean_change": np.max(euclidean_changes),
            "min_euclidean_change": np.min(euclidean_changes),
        }