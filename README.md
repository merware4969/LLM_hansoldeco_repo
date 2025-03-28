🚧 건설 안전사고 예방 AI (LLM 기반 RAG)

이 프로젝트는 건설 현장에서 발생하는 사고 데이터를 분석하고, 관련 지침서를 참고하여 재발 방지 대책 및 향후 조치 계획을 자동으로 생성하는 AI 모델을 구축하는 것입니다. LLM (Large Language Model)과 FAISS 기반 벡터 검색을 활용하여, 사용자 질문에 대한 최적의 답변을 제공합니다.

📌 주요 기능

PDF 문서 분석: 건설 안전 지침서를 분석하여 RAG (Retrieval-Augmented Generation) 기반 답변을 생성

건설 사고 데이터 처리: CSV 데이터에서 사고 유형을 분류하고, 유사 사례를 검색하여 대응책을 생성

FAISS 기반 벡터 검색: 사고 데이터 및 지침서를 벡터화하여, 가장 유사한 사례를 검색 후 활용

Hugging Face LLM 사용: GPT-J 6B 모델을 8bit 양자화하여 사용, 재발 방지 대책 생성

Docker 컨테이너 지원: 프로젝트 환경을 쉽게 설정하고 실행할 수 있도록 Docker 기반 배포 지원

🛠️ 프로젝트 파일 구조

.
├── Dockerfile               # Docker 이미지 구성 파일
├── main.py                  # 메인 실행 스크립트
├── requirements.txt         # 프로젝트에 필요한 Python 라이브러리
├── open/                    # 데이터 및 PDF 문서 저장 폴더
│   ├── train.csv            # 훈련 데이터
│   ├── test.csv             # 테스트 데이터
│   ├── sample_submission.csv # 제출 양식
│   ├── 건설안전지침/        # PDF 지침서 폴더
└── README.md               # 프로젝트 설명

📦 필요 라이브러리

이 프로젝트에서 사용되는 라이브러리는 requirements.txt 파일에 정의되어 있으며, 다음과 같은 주요 패키지가 포함됩니다.

# PyTorch + CUDA 11.2 지원
torch==1.12.1+cu112
torchvision==0.13.1+cu112
torchaudio==0.12.1+cu112

# NLP 및 LLM 관련 패키지
transformers==4.26.1
langchain==0.1.10
langchain-community==0.1.10
faiss-cpu==1.7.4
sentence-transformers==2.2.2

# 데이터 처리 및 PDF 분석
pymupdf==1.23.6
pandas==1.5.3
numpy==1.23.5
scipy==1.10.1

🚀 Docker 환경에서 실행하는 방법

이 프로젝트는 Docker 컨테이너 환경에서 실행할 수 있습니다.

1️⃣ Docker 이미지 빌드

docker build -t llm_hansoldeco .

2️⃣ Docker 컨테이너 실행 (GPU 활성화)

docker run --gpus all --rm -it llm_hansoldeco

3️⃣ 컨테이너 내부에서 main.py 실행

python3 main.py

✅ 실행 예시

테스트 실행 시작... 총 샘플 수: 100
[샘플 1/100] 진행 중...
[샘플 50/100] 진행 중...

테스트 완료! 총 결과 수: 100
결과 저장 완료: open/main_submission.csv

📊 모델 및 데이터 처리

데이터 전처리: CSV 데이터를 읽어와 공사 종류 및 사고 유형을 대/중분류로 나눔

PDF 문서 처리: 건설 안전 지침서에서 주요 정보를 추출 후 FAISS 벡터로 변환

질문-답변 생성: 유사 사고를 검색 후, GPT-J 6B LLM을 활용해 적절한 조치 계획을 자동 생성

📜 참고 자료

Hugging Face GPT-J 6B 모델 (링크)

FAISS 벡터 검색 (공식 문서)

LangChain RAG (GitHub)

📌 주의사항

GPU 환경에서 실행을 권장하며, CUDA 11.2가 지원되는지 확인 필요

open/건설안전지침/ 폴더에 PDF 파일이 포함되어 있어야 정상 작동

이 프로젝트가 도움이 되셨다면 ⭐️ Star를 눌러주세요! 😊