#!/usr/bin/env python3
"""
체크포인트 개수를 세고 로그를 저장하는 스크립트
"""
import os
import json
import argparse
from pathlib import Path
import logging
from datetime import datetime

def setup_logging(log_file):
    """로깅 설정"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def count_checkpoints(folder_path):
    """폴더 내 체크포인트 개수를 세는 함수"""
    folder_path = Path(folder_path)
    
    if not folder_path.exists():
        return 0, []
    
    checkpoints = []
    
    # checkpoint-숫자 형태의 폴더들을 찾음
    for item in folder_path.iterdir():
        if item.is_dir() and item.name.startswith('checkpoint-'):
            try:
                checkpoint_num = int(item.name.split('-')[1])
                checkpoints.append((checkpoint_num, str(item)))
            except (ValueError, IndexError):
                continue
    
    # 체크포인트 번호순으로 정렬
    checkpoints.sort(key=lambda x: x[0])
    
    return len(checkpoints), checkpoints

def main():
    parser = argparse.ArgumentParser(description="폴더 내 체크포인트 개수를 세고 로그를 저장")
    parser.add_argument("--folder", required=True, help="체크포인트를 세고 싶은 폴더 경로")
    parser.add_argument("--log-dir", default="./logs", help="로그 파일을 저장할 디렉토리")
    parser.add_argument("--output", help="결과를 저장할 JSON 파일 경로")
    
    args = parser.parse_args()
    
    # 로그 디렉토리 생성
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # 로그 파일 설정
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"checkpoint_count_{timestamp}.log"
    
    logger = setup_logging(log_file)
    
    logger.info(f"체크포인트 개수 세기 시작: {args.folder}")
    
    # 체크포인트 개수 세기
    count, checkpoints = count_checkpoints(args.folder)
    
    logger.info(f"총 체크포인트 개수: {count}")
    
    # 결과 출력
    result = {
        "folder_path": str(Path(args.folder).absolute()),
        "checkpoint_count": count,
        "checkpoints": [
            {
                "number": num,
                "path": path,
                "name": Path(path).name
            }
            for num, path in checkpoints
        ],
        "timestamp": datetime.now().isoformat(),
        "log_file": str(log_file)
    }
    
    # 콘솔에 출력
    print(f"폴더: {args.folder}")
    print(f"체크포인트 개수: {count}")
    print(f"로그 파일: {log_file}")
    
    if checkpoints:
        print("\n체크포인트 목록:")
        for num, path in checkpoints:
            print(f"  - {Path(path).name} (번호: {num})")
    
    # JSON 파일로 저장
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        logger.info(f"결과 저장: {output_path}")
    
    logger.info("체크포인트 개수 세기 완료")
    
    return result

if __name__ == "__main__":
    main()
