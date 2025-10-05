"""
로깅 유틸리티
"""
import sys
sys.path.append('/root/task_vector/tofu/add')


import logging
import sys
from typing import Optional



def setup_logging(level: str = "INFO", log_file: Optional[str] = None) -> logging.Logger:
    """로깅 설정"""
    
    # 로그 포맷 설정
    formatter = logging.Formatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # 루트 로거 설정
    logger = logging.getLogger("unlearning_analysis")
    logger.setLevel(getattr(logging, level.upper()))
    
    # 기존 핸들러 제거
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # 콘솔 핸들러 추가
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # 파일 핸들러 추가 (선택적)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str = "unlearning_analysis") -> logging.Logger:
    """로거 인스턴스 반환"""
    return logging.getLogger(name)