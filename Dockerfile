# 기본 이미지 설정 (PyTorch + CUDA 11.2 지원)
FROM nvidia/cuda:11.2.2-runtime-ubuntu20.04

# 작업 디렉토리 설정
WORKDIR /app

# 필수 패키지 설치
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/*

# Python 환경 설정
RUN pip3 install --no-cache-dir --upgrade pip

# requirements.txt 설치
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# GitHub 레포 클론 및 프로젝트 복사
RUN git clone https://github.com/merware4969/LLM_hansoldeco_repo.git /app

# 실행할 스크립트 설정
CMD ["python3", "main.py"]
