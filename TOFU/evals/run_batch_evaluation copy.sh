#!/bin/bash

# 간단한 배치 평가 스크립트
# 사용법: ./run_batch_evaluation.sh (경로들이 하드코딩되어 있음)

set -e

# 하드코딩된 경로들
RETAIN_MODEL="/root/Unlearn-Simple/TOFU/paper_models/final_ft_noLORA_5_epochs_inst_lr1e-05_llama2-7b_retain90_seed42_1/checkpoint-562"
FORGET_FOLDERS=("/root/Unlearn-Simple/TOFU/paper_models/final_ft_noLORA_5_epochs_inst_lr1e-05_llama2-7b_full_seed42_1/checkpoint-625/unlearned/8GPU_simnpo_1e-05_forget10_epoch10_batch4_accum4_beta4.0_gamma1.5_grad_diff_coeff1.0_reffine_tuned_evalsteps_per_epoch_seed1001_1")
DATA_PATH="locuslab/TOFU"
OUTPUT_DIR="./results"

# 기본값 설정
MODEL_FAMILY="llama2-7b"
BATCH_SIZE=16

# 작업 디렉토리 변경 (data_module.py가 있는 곳으로)
cd "/root/Unlearn-Simple/TOFU"

# Python 경로 설정
export PYTHONPATH="/root/Unlearn-Simple/TOFU:$PYTHONPATH"

# 출력 디렉토리 생성
mkdir -p "$OUTPUT_DIR"
mkdir -p "./logs"

echo "리테인 모델: $RETAIN_MODEL"
echo "폴겟 폴더들: ${FORGET_FOLDERS[*]}"
echo "데이터 경로: $DATA_PATH"
echo "출력 디렉토리: $OUTPUT_DIR"

# 1단계: 체크포인트 개수 세기
echo "1단계: 체크포인트 개수 세기"
for folder in "${FORGET_FOLDERS[@]}"; do
    echo "폴더 체크포인트 세기: $folder"
    python evals/count_checkpoints.py \
        --folder "$folder" \
        --log-dir "./logs" \
        --output "$OUTPUT_DIR/checkpoint_count_$(basename "$folder").json"
done

# 2단계: 리테인 모델 평가 (한 번만)
echo "2단계: 리테인 모델 평가"
RETAIN_OUTPUT_DIR="$RETAIN_MODEL/eval_results"
mkdir -p "$RETAIN_OUTPUT_DIR"

# 리테인 모델 평가 결과 파일들이 이미 존재하는지 확인
RETAIN_FILES_EXIST=true
for eval_file in "eval_log.json" "eval_real_author_wo_options.json" "eval_real_world_wo_options.json" "eval_log_forget.json"; do
    if [ ! -f "$RETAIN_OUTPUT_DIR/$eval_file" ]; then
        RETAIN_FILES_EXIST=false
        break
    fi
done

if [ "$RETAIN_FILES_EXIST" = true ]; then
    echo "리테인 모델 평가 결과가 이미 존재함 - 스킵"
else
    echo "리테인 모델 평가 실행 중..."
    python evals/eval_everything.py \
        --config-name=eval_everything \
        model_path="$RETAIN_MODEL" \
        save_dir="$RETAIN_OUTPUT_DIR" \
        overwrite=true
fi

# 3단계: 폴겟 폴더들 평가
echo "3단계: 폴겟 폴더들 평가"
RETAIN_LOG_PATH="$RETAIN_OUTPUT_DIR/eval_log.json"
RETAIN_FORGET_LOG_PATH="$RETAIN_OUTPUT_DIR/eval_log_forget.json"

for forget_folder in "${FORGET_FOLDERS[@]}"; do
    echo "폴겟 폴더 평가: $forget_folder"
    
    # 체크포인트들 찾기 (정상 작동하는 것만)
    CHECKPOINTS=($(find "$forget_folder" -type d -name "checkpoint-*" | sort -V))
    
    # 손상된 체크포인트 제외 (checkpoint-36, checkpoint-48, checkpoint-60 등)
    VALID_CHECKPOINTS=()
    for checkpoint in "${CHECKPOINTS[@]}"; do
        checkpoint_name=$(basename "$checkpoint")
        # checkpoint-12만 평가 (정상 작동 확인됨)
        if [[ "$checkpoint_name" == "checkpoint-12" ]]; then
            VALID_CHECKPOINTS+=("$checkpoint")
        fi
    done
    CHECKPOINTS=("${VALID_CHECKPOINTS[@]}")
    
    if [ ${#CHECKPOINTS[@]} -eq 0 ]; then
        echo "체크포인트를 찾을 수 없음: $forget_folder"
        continue
    fi
    
    echo "발견된 체크포인트 개수: ${#CHECKPOINTS[@]}"
    
    # 각 체크포인트에 대해 평가
    for checkpoint in "${CHECKPOINTS[@]}"; do
        checkpoint_name=$(basename "$checkpoint")
        echo "체크포인트 평가: $checkpoint_name"
        
        # 폴겟 모델 평가 - checkpoint 폴더 안에 eval_results 생성
        forget_output_dir="$checkpoint/eval_results"
        mkdir -p "$forget_output_dir"
        
        # 폴겟 모델 평가 결과 파일들이 이미 존재하는지 확인
        FORGET_FILES_EXIST=true
        for eval_file in "eval_log.json" "eval_real_author_wo_options.json" "eval_real_world_wo_options.json" "eval_log_forget.json"; do
            if [ ! -f "$forget_output_dir/$eval_file" ]; then
                FORGET_FILES_EXIST=false
                break
            fi
        done
        
        if [ "$FORGET_FILES_EXIST" = true ]; then
            echo "체크포인트 $checkpoint_name 평가 결과가 이미 존재함 - 평가 스킵"
        else
            echo "체크포인트 $checkpoint_name 평가 실행 중..."
            python evals/eval_everything.py \
                --config-name=eval_everything \
                model_path="$checkpoint" \
                save_dir="$forget_output_dir" \
                overwrite=true
            
            if [ $? -ne 0 ]; then
                echo "체크포인트 $checkpoint_name 평가 실패"
                continue
            fi
        fi
        
        # compute_metrics 실행 - Hydra 설정 사용
        metrics_output_path="$forget_output_dir/metrics_summary.json"
        
        # 메트릭 요약 파일이 이미 존재하는지 확인
        if [ -f "$metrics_output_path" ]; then
            echo "체크포인트 $checkpoint_name 메트릭 요약이 이미 존재함 - 메트릭 계산 스킵"
        else
            echo "체크포인트 $checkpoint_name 메트릭 계산 실행 중..."
            python evals/compute_metrics.py \
                retain_result="$RETAIN_OUTPUT_DIR/eval_log_aggregated.json" \
                ckpt_result="$forget_output_dir/eval_log_aggregated.json" \
                method_name="SimNPO" \
                submitted_by="batch_eval" \
                save_file="$metrics_output_path"
        fi
        
        if [ -f "$metrics_output_path" ]; then
            echo "체크포인트 $checkpoint_name 평가 완료"
            
            # 평가 완료 후 모델 파일들 삭제 (eval_results 폴더는 보존)
            echo "체크포인트 $checkpoint_name 모델 파일 삭제 중..."
            find "$checkpoint" -name "*.safetensors" -o -name "*.bin" -o -name "*.pth" | xargs rm -f
            find "$checkpoint" -name "model-*" | xargs rm -f
            find "$checkpoint" -name "pytorch_model*" | xargs rm -f
            find "$checkpoint" -name "rng_state_*.pth" | xargs rm -f
            find "$checkpoint" -name "scheduler.pt" | xargs rm -f
            find "$checkpoint" -name "training_args.bin" | xargs rm -f
            find "$checkpoint" -name "zero_to_fp32.py" | xargs rm -f
            find "$checkpoint" -name "global_step*" -type d | xargs rm -rf
            echo "체크포인트 $checkpoint_name 모델 파일 삭제 완료 (eval_results 폴더는 보존됨)"
        else
            echo "체크포인트 $checkpoint_name 메트릭 계산 실패"
        fi
    done
done

echo "배치 평가 완료!"
echo "결과 디렉토리: $OUTPUT_DIR"