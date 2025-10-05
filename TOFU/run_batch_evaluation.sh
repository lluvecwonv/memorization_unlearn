#!/usr/bin/env bash
set -euo pipefail

# Disable DeepSpeed compilation to avoid CPU Adam errors
export DS_BUILD_CPU_ADAM=0
export DS_BUILD_AIO=0

# 하드코딩된 경로들
RETAIN_MODEL="/root/Unlearn-Simple/TOFU/paper_models/final_ft_noLORA_5_epochs_inst_lr1e-05_llama2-7b_retain90_seed42_1/checkpoint-562"
FORGET_FOLDERS=("/root/Unlearn-Simple/TOFU/paper_models/final_ft_noLORA_5_epochs_inst_lr1e-05_llama2-7b_full_seed42_1/checkpoint-625/unlearned/8GPU_tnpo_1e-05_forget10_epoch10_batch4_accum4_beta0.05_gamma0.5_grad_diff_coeff1.0_reffine_tuned_evalsteps_per_epoch_seed1001_1")
OUTPUT_DIR="./results"

# 기본값 설정
MODEL_FAMILY="llama2-7b"
SPLIT="forget10_perturbed"

# 작업 디렉토리 변경
cd "/root/Unlearn-Simple/TOFU"

# Python 경로 설정
export PYTHONPATH="/root/Unlearn-Simple/TOFU:${PYTHONPATH:-}"

# 출력 디렉토리 생성
mkdir -p "$OUTPUT_DIR"
mkdir -p "./logs"

echo "리테인 모델: $RETAIN_MODEL"
echo "폴겟 폴더들: ${FORGET_FOLDERS[*]}"
echo "출력 디렉토리: $OUTPUT_DIR"

# 1단계: 리테인 모델 평가 (한 번만)
echo "1단계: 리테인 모델 평가"
RETAIN_OUTPUT_DIR="$RETAIN_MODEL/eval_results"
mkdir -p "$RETAIN_OUTPUT_DIR"

# 리테인 모델 평가 결과가 이미 존재하는지 확인 (필요한 JSON 파일들 체크)
if [ -f "$RETAIN_OUTPUT_DIR/eval_log.json" ] && \
   [ -f "$RETAIN_OUTPUT_DIR/eval_log_forget.json" ] && \
   [ -f "$RETAIN_OUTPUT_DIR/eval_real_world_wo_options.json" ] && \
   [ -f "$RETAIN_OUTPUT_DIR/eval_real_author_wo_options.json" ]; then
    echo "리테인 모델 평가 결과가 이미 존재함 - 스킵"
else
    echo "리테인 모델 평가 실행 중..."
    
    # evaluate_util.py 실행 (GPU 1개만 사용)
    CUDA_VISIBLE_DEVICES=0 python evaluate_util.py \
        model_family=$MODEL_FAMILY \
        split=$SPLIT \
        model_path="$RETAIN_MODEL" \
        save_dir="$RETAIN_OUTPUT_DIR"
    
    if [ $? -ne 0 ]; then
        echo "리테인 모델 평가 실패"
        exit 1
    fi
fi

# 2단계: 폴겟 폴더들 평가
echo "2단계: 폴겟 폴더들 평가"

for forget_folder in "${FORGET_FOLDERS[@]}"; do
    echo "폴겟 폴더 평가: $forget_folder"
    
    # 체크포인트들 찾기 (이미 완료된 것만)
    CHECKPOINTS=($(find "$forget_folder" -type d -name "checkpoint-*" | sort -V))
    
    # 이미 완료된 체크포인트만 필터링 (checkpoint-12만)
    VALID_CHECKPOINTS=()
    for checkpoint in "${CHECKPOINTS[@]}"; do
        checkpoint_name=$(basename "$checkpoint")
        eval_results_dir="$checkpoint/eval_results"
        
        # 모든 체크포인트 처리
        if [ -d "$checkpoint" ]; then
            VALID_CHECKPOINTS+=("$checkpoint")
            echo "유효한 체크포인트 발견: $checkpoint_name"
        else
            echo "체크포인트 $checkpoint_name 스킵 (디렉토리가 존재하지 않음)"
        fi
    done
    CHECKPOINTS=("${VALID_CHECKPOINTS[@]}")
    
    if [ ${#CHECKPOINTS[@]} -eq 0 ]; then
        echo "체크포인트를 찾을 수 없음: $forget_folder"
        continue
    fi
    
    echo "발견된 체크포인트 개수: ${#CHECKPOINTS[@]}"
    
    # 각 체크포인트에 대해 평가
    for i in "${!CHECKPOINTS[@]}"; do
        checkpoint="${CHECKPOINTS[$i]}"
        checkpoint_name=$(basename "$checkpoint")
        echo "체크포인트 평가: $checkpoint_name"
        
        # 폴겟 모델 평가 - checkpoint 폴더 안에 eval_results 생성
        forget_output_dir="$checkpoint/eval_results"
        mkdir -p "$forget_output_dir"
        
        # 폴겟 모델 평가 결과가 이미 존재하는지 확인 (필요한 JSON 파일들 체크)
        if [ -f "$forget_output_dir/eval_log.json" ] && \
           [ -f "$forget_output_dir/eval_log_forget.json" ] && \
           [ -f "$forget_output_dir/eval_real_world_wo_options.json" ] && \
           [ -f "$forget_output_dir/eval_real_author_wo_options.json" ]; then
            echo "체크포인트 $checkpoint_name 평가 결과가 이미 존재함 - 평가 스킵"
        elif [ ! -f "$checkpoint/model-00001-of-00003.safetensors" ]; then
            echo "체크포인트 $checkpoint_name 모델 파일이 없음 - 스킵"
        else
            echo "체크포인트 $checkpoint_name 평가 실행 중..."
            
            # evaluate_util.py 실행 (GPU 1개만 사용)
            CUDA_VISIBLE_DEVICES=0 python evaluate_util.py \
                model_family=$MODEL_FAMILY \
                split=$SPLIT \
                model_path="$checkpoint" \
                save_dir="$forget_output_dir"
            
            if [ $? -ne 0 ]; then
                echo "체크포인트 $checkpoint_name 평가 실패"
                continue
            fi
        fi
        
        # aggregate_eval_stat.py 실행
        metrics_output_path="$forget_output_dir/metrics_summary.json"
        
        # 메트릭 요약 파일이 이미 존재하는지 확인
        if [ -f "$metrics_output_path" ]; then
            echo "체크포인트 $checkpoint_name 메트릭 요약이 이미 존재함 - 메트릭 계산 스킵"
            # 기존 메트릭 파일에서 결과 읽어서 출력
            model_utility=$(python -c "import json; data=json.load(open('$metrics_output_path')); print(f\"{data.get('Model Utility', 'N/A'):.4f}\" if isinstance(data.get('Model Utility'), (int, float)) else 'N/A')")
            forget_quality=$(python -c "import json; data=json.load(open('$metrics_output_path')); print(f\"{data.get('Forget Quality', 'N/A'):.4f}\" if isinstance(data.get('Forget Quality'), (int, float)) else 'N/A')")
            ks_test=$(python -c "import json; data=json.load(open('$metrics_output_path')); print(f\"{data.get('KS Test Forget', 'N/A'):.4f}\" if isinstance(data.get('KS Test Forget'), (int, float)) else 'N/A')")
            echo "[$checkpoint_name] Model Utility: $model_utility, Forget Quality: $forget_quality, KS Test: $ks_test"
        else
            echo "체크포인트 $checkpoint_name 메트릭 계산 실행 중..."
            python aggregate_eval_stat.py \
                retain_result="$RETAIN_OUTPUT_DIR/eval_log_aggregated.json" \
                ckpt_result="$forget_output_dir/eval_log_aggregated.json" \
                method_name="SimNPO" \
                submitted_by="batch_eval" \
                save_file="$metrics_output_path"
            
            if [ $? -ne 0 ]; then
                echo "체크포인트 $checkpoint_name 메트릭 계산 실패"
                continue
            fi
            
            # 계산 직후 결과 출력
            model_utility=$(python -c "import json; data=json.load(open('$metrics_output_path')); print(f\"{data.get('Model Utility', 'N/A'):.4f}\" if isinstance(data.get('Model Utility'), (int, float)) else 'N/A')")
            forget_quality=$(python -c "import json; data=json.load(open('$metrics_output_path')); print(f\"{data.get('Forget Quality', 'N/A'):.4f}\" if isinstance(data.get('Forget Quality'), (int, float)) else 'N/A')")
            ks_test=$(python -c "import json; data=json.load(open('$metrics_output_path')); print(f\"{data.get('KS Test Forget', 'N/A'):.4f}\" if isinstance(data.get('KS Test Forget'), (int, float)) else 'N/A')")
            echo "[$checkpoint_name] Model Utility: $model_utility, Forget Quality: $forget_quality, KS Test: $ks_test"
        fi
        
        # 평가와 메트릭 계산이 모두 완료되면 체크포인트 삭제
        # 모든 필요한 JSON 파일들과 metrics_summary.json이 존재하는지 확인
        if [ -f "$forget_output_dir/eval_log.json" ] && \
           [ -f "$forget_output_dir/eval_log_forget.json" ] && \
           [ -f "$forget_output_dir/eval_real_world_wo_options.json" ] && \
           [ -f "$forget_output_dir/eval_real_author_wo_options.json" ] && \
           [ -f "$metrics_output_path" ]; then
            echo "체크포인트 $checkpoint_name 평가 완료 - eval_results 폴더만 남기고 나머지 삭제 중..."
            
            # eval_results 폴더를 임시로 백업
            temp_eval_dir="/tmp/eval_results_backup_$(basename $checkpoint)"
            cp -r "$forget_output_dir" "$temp_eval_dir"
            
            # 체크포인트 전체 삭제
            rm -rf "$checkpoint"
            
            # 체크포인트 디렉토리 재생성 후 eval_results만 복원
            mkdir -p "$checkpoint"
            mv "$temp_eval_dir" "$forget_output_dir"
            
            echo "체크포인트 $checkpoint_name 정리 완료 - eval_results 폴더만 유지됨"
        fi
        
    done
done

echo "배치 평가 완료!"
echo "결과 디렉토리: $OUTPUT_DIR"

# 3단계: 전체 결과 요약 생성 (선택사항)
echo "3단계: 전체 결과 요약 생성"
SUMMARY_FILE="$OUTPUT_DIR/batch_evaluation_summary.csv"

echo "checkpoint,model_utility,forget_quality,ks_test_statistic" > "$SUMMARY_FILE"

for forget_folder in "${FORGET_FOLDERS[@]}"; do
    CHECKPOINTS=($(find "$forget_folder" -type d -name "checkpoint-*" | sort -V))
    
    for checkpoint in "${CHECKPOINTS[@]}"; do
        checkpoint_name=$(basename "$checkpoint")
        metrics_file="$checkpoint/eval_results/metrics_summary.json"
        
        if [ -f "$metrics_file" ]; then
            # JSON에서 메트릭 추출
            model_utility=$(python -c "import json; data=json.load(open('$metrics_file')); print(f\"{data.get('Model Utility', 'N/A'):.4f}\" if isinstance(data.get('Model Utility'), (int, float)) else 'N/A')")
            forget_quality=$(python -c "import json; data=json.load(open('$metrics_file')); print(f\"{data.get('Forget Quality', 'N/A'):.4f}\" if isinstance(data.get('Forget Quality'), (int, float)) else 'N/A')")
            ks_test=$(python -c "import json; data=json.load(open('$metrics_file')); print(f\"{data.get('KS Test Forget', 'N/A'):.4f}\" if isinstance(data.get('KS Test Forget'), (int, float)) else 'N/A')")
            
            echo "$checkpoint_name,$model_utility,$forget_quality,$ks_test" >> "$SUMMARY_FILE"
            
            # 콘솔에도 결과 출력
            echo "[$checkpoint_name] Model Utility: $model_utility, Forget Quality: $forget_quality, KS Test: $ks_test"
        fi
    done
done

echo ""
echo "============================================"
echo "평가 결과 요약"
echo "============================================"
if [ -f "$SUMMARY_FILE" ]; then
    cat "$SUMMARY_FILE" | column -t -s ','
fi
echo "============================================"
echo ""
echo "요약 파일 생성 완료: $SUMMARY_FILE"