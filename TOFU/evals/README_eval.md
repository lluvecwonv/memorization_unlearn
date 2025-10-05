# 언러닝 모델 평가 가이드

## 🚀 빠른 시작 (추천)

### Model Utility & Forget Quality 계산 (가장 간단)
```bash
cd /root/Unlearn-Simple/TOFU/evals
./run_compute_metrics.sh \
  /path/to/retain_eval_log_aggregated.json \
  /path/to/unlearned_eval_log_aggregated.json \
  ./results \
  simnpo_forget10
```

### 실제 예시
```bash
./run_compute_metrics.sh \
  /root/Unlearn-Simple/TOFU/paper_models/final_ft_noLORA_5_epochs_inst_lr1e-05_llama2-7b_retain90_seed42_1/checkpoint-562/eval_results/ds_size300/eval_log.json \
  /root/Unlearn-Simple/TOFU/paper_models/final_ft_noLORA_5_epochs_inst_lr1e-05_llama2-7b_full_seed42_1/checkpoint-625/unlearned/simnpo/checkpoint-125/eval_log_aggregated.json \
  ./results \
  simnpo_forget10
```

### 기존 평가 결과 비교 (간단한 방법)
```bash
cd /root/Unlearn-Simple/TOFU/evals
./run_simple_eval.sh \
  /path/to/retain_model_eval.json \
  /path/to/unlearned_model_eval.json \
  ./results \
  simnpo_forget10
```

### 기존 평가 결과 비교 (상세한 방법)
```bash
cd /root/Unlearn-Simple/TOFU/evals
./run_compare_eval.sh \
  /path/to/retain_model_eval.json \
  /path/to/unlearned_model_eval.json \
  ./results \
  simnpo_forget10
```

## 전체 평가 실행 (새로 평가)

### 기본 사용법
```bash
cd /root/Unlearn-Simple/TOFU/evals
./run_full_eval.sh <retain_모델_경로> <언러닝된_모델_경로> <데이터_경로> [결과_저장_경로] [방법_이름]
```

### 예시
```bash
# 기본 사용
./run_full_eval.sh \
  /path/to/retain/model \
  /path/to/unlearned/model \
  /path/to/data

# 모든 옵션 지정
./run_full_eval.sh \
  /path/to/retain/model \
  /path/to/unlearned/model \
  /path/to/data \
  ./results \
  simnpo_forget10

# 실제 경로 예시
./run_full_eval.sh \
  /root/Unlearn-Simple/TOFU/paper_models/final_ft_noLORA_5_epochs_inst_lr1e-05_llama2-7b_retain90_seed42_1/checkpoint-562 \
  /root/Unlearn-Simple/TOFU/paper_models/final_ft_noLORA_5_epochs_inst_lr1e-05_llama2-7b_full_seed42_1/checkpoint-625/unlearned/simnpo/checkpoint-125 \
  /path/to/data \
  ./results \
  simnpo_forget10
```

## 개별 평가 실행

### 1. 기본 평가 (eval.py)
```bash
# 배치 스크립트 사용 (추천)
./run_eval_basic.sh /path/to/model locuslab/TOFU retain retain_eval ./results llama2-7b
./run_eval_basic.sh /path/to/model locuslab/TOFU forget forget_eval ./results llama2-7b

# 직접 실행
python eval.py \
  --config-name=eval \
  model_path=/path/to/unlearned/model \
  data_path=/path/to/retain/data \
  split=retain \
  eval_task=retain_eval \
  save_dir=./results
```

### 2. 종합 평가 (eval_everything.py)
```bash
# 배치 스크립트 사용 (추천)
./run_eval_comprehensive.sh /path/to/model "locuslab/TOFU,locuslab/TOFU" "retain,forget" "retain_eval,forget_eval" ./results llama2-7b

# 직접 실행
python eval_everything.py \
  --config-name=eval_everything \
  model_path=/path/to/unlearned/model \
  data_path="[/path/to/retain, /path/to/forget]" \
  split_list="[retain, forget]" \
  eval_task="[retain_comp, forget_comp]" \
  save_dir=./results
```

### 3. 데이터 증강 평가 (eval_augmentation.py)
```bash
# 배치 스크립트 사용 (추천)
./run_eval_augmentation.sh /path/to/model locuslab/TOFU retain augmentation_eval answer answer_perturbed ./results llama2-7b

# 직접 실행
python eval_augmentation.py \
  --config-name=eval_augmentation \
  model_path=/path/to/model \
  data_path=locuslab/TOFU \
  split=retain \
  eval_task=augmentation_eval \
  base_answer_key=answer \
  compare_answer_key=answer_perturbed \
  save_dir=./results
```

## 평가 결과

### 생성되는 파일들
- `retain_eval.json`: Retain 데이터셋 기본 평가 결과
- `forget_eval.json`: Forget 데이터셋 기본 평가 결과
- `retain_comprehensive.json`: Retain 데이터셋 종합 평가 결과
- `forget_comprehensive.json`: Forget 데이터셋 종합 평가 결과
- `{method_name}_aggregated.json`: 집계된 최종 결과
- `evaluation_summary.txt`: 평가 요약 보고서

### 주요 평가 메트릭
- **ROUGE 점수**: 생성된 텍스트와 정답 간의 유사도
- **정확도**: 토큰 레벨 정확도
- **손실값**: 모델의 손실값
- **Perturbation 분석**: 원본 답변과 수정된 답변 간 비교

## 결과 해석

### 1. 언러닝 효과 확인
- Forget 데이터셋에서 낮은 성능 = 언러닝 성공
- Retain 데이터셋에서 높은 성능 = 일반 성능 유지

### 2. 편향성 분석
- Perturbation ratio가 높으면 모델이 특정 답변을 선호
- 낮으면 모델이 중립적

### 3. 품질 평가
- ROUGE 점수가 높으면 생성 품질이 좋음
- 정확도가 높으면 모델이 정확한 답변 생성

## 문제 해결

### 일반적인 오류
1. **모델 로딩 실패**: 모델 경로와 설정 확인
2. **데이터 로딩 실패**: 데이터 경로와 형식 확인
3. **메모리 부족**: 배치 크기 줄이기
4. **CUDA 오류**: GPU 메모리 확인

### 디버깅 팁
- `overwrite=true` 옵션으로 기존 결과 덮어쓰기
- `ds_size=100` 옵션으로 작은 데이터셋으로 테스트
- 로그 파일 확인하여 오류 원인 파악
