import os
import pandas as pd
import fitz  # PyMuPDF
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import torch

# 경로 설정
DATA_PATH = "open/"
GUIDELINES_PATH = "open/건설안전지침/"  # PDF 파일이 있는 폴더
SUBMISSION_FILE = "open/sample_submission.csv"
OUTPUT_FILE = "open/main_submission.csv"

# 데이터 로드
train = pd.read_csv(os.path.join(DATA_PATH, "train.csv"), encoding="utf-8-sig")
test = pd.read_csv(os.path.join(DATA_PATH, "test.csv"), encoding="utf-8-sig")

# ---------------------------
# PDF 문서 로드 (Chunking 제거)
# ---------------------------
def load_pdfs(pdf_folder):
    all_texts = []
    for file in os.listdir(pdf_folder):
        if file.endswith(".pdf"):
            file_path = os.path.join(pdf_folder, file)
            doc = fitz.open(file_path)
            full_text = []
            for page in doc:
                page_text = page.get_text("text")
                full_text.append(page_text.strip())
            # 하나의 문자열로 합침
            merged_text = "\n".join(full_text)
            all_texts.append(merged_text)
    return all_texts

# PDF 데이터 로드
pdf_documents = load_pdfs(GUIDELINES_PATH)

# 데이터 전처리 (대/중분류 구분)
for df in [train, test]:
    df['공사종류(대분류)'] = df['공사종류'].str.split(' / ').str[0]
    df['공사종류(중분류)'] = df['공사종류'].str.split(' / ').str[1]
    df['공종(대분류)'] = df['공종'].str.split(' > ').str[0]
    df['공종(중분류)'] = df['공종'].str.split(' > ').str[1]
    df['사고객체(대분류)'] = df['사고객체'].str.split(' > ').str[0]
    df['사고객체(중분류)'] = df['사고객체'].str.split(' > ').str[1]

# 질문-답변 데이터 생성
train_data = train.apply(lambda row: {
    "question": (
        f"공사종류 대분류 '{row['공사종류(대분류)']}', 중분류 '{row['공사종류(중분류)']}' 공사 중 "
        f"공종 대분류 '{row['공종(대분류)']}', 중분류 '{row['공종(중분류)']}' 작업에서 "
        f"사고객체 '{row['사고객체(대분류)']}'(중분류: '{row['사고객체(중분류)']}')와 관련된 사고가 발생했습니다. "
        f"작업 프로세스는 '{row['작업프로세스']}'이며, 사고 원인은 '{row['사고원인']}'입니다. "
        f"재발 방지 대책 및 향후 조치 계획은 무엇인가요?"
    ),
    "answer": row["재발방지대책 및 향후조치계획"]
}, axis=1)
train_data = pd.DataFrame(list(train_data))

# 임베딩 모델 설정 (FAISS 벡터 검색용)
embedding_model_name = "jhgan/ko-sbert-nli"
embedding = HuggingFaceEmbeddings(model_name=embedding_model_name)

# ----------------------------
# FAISS 벡터 스토어 생성 (사고 데이터 + PDF 문서)
# ----------------------------
train_documents = [
    f"Q: {q}\nA: {a}"
    for q, a in zip(train_data['question'], train_data['answer'])
]
# PDF 문서도 각각 추가
train_documents.extend(pdf_documents)

vector_store = FAISS.from_texts(train_documents, embedding)
retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# ----------------------------
# LLM 모델 설정 (GPT-J 6B 8bit Quant)
# ----------------------------
# bitsandbytes >= 0.37.0 필요
# pip install bitsandbytes

model_name = "EleutherAI/gpt-j-6B"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

# 8비트 양자화 로드
# device_map="auto"로 GPU/CPU 자동 분산
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    load_in_8bit=True,    # 8bit quantization
    device_map="auto"     # CUDA + CPU 오프로딩
)

# text-generation 파이프라인
text_generation_pipeline = pipeline(
    task="text-generation",
    model=model,
    tokenizer=tokenizer,
    max_length=512,
    temperature=0.7,
    top_p=0.9,
    do_sample=True,
)

llm = HuggingFacePipeline(pipeline=text_generation_pipeline)

# RAG Prompt Template
prompt_template = """
### 건설 안전 전문가 지침
질문에 대한 답변을 핵심 내용만 요약하여 작성하세요.
- 서론, 배경 설명 없이 핵심만 기술하세요.
- 구체적인 조치 항목을 나열하세요.

{context}

### 질문:
{question}
"""

prompt = PromptTemplate(
    input_variables=["context", "question"], 
    template=prompt_template
)

# RAG 체인 생성
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": prompt}
)

# test DataFrame에 question 컬럼 추가
test['question'] = test.apply(
    lambda row: (
        f"공사종류 대분류 '{row['공사종류(대분류)']}', 중분류 '{row['공사종류(중분류)']}' 공사 중 "
        f"공종 대분류 '{row['공종(대분류)']}', 중분류 '{row['공종(중분류)']}' 작업에서 "
        f"사고객체 '{row['사고객체(대분류)']}'(중분류: '{row['사고객체(중분류)']}')와 관련된 사고가 발생했습니다. "
        f"작업 프로세스는 '{row['작업프로세스']}'이며, 사고 원인은 '{row['사고원인']}'입니다. "
        f"재발 방지 대책 및 향후 조치 계획은 무엇인가요?"
    ),
    axis=1
)

# 테스트 데이터 예측
test_results = []
print("테스트 실행 시작... 총 샘플 수:", len(test))

for idx, row in test.iterrows():
    if (idx + 1) % 50 == 0 or idx == 0:
        print(f"\n[샘플 {idx + 1}/{len(test)}] 진행 중...")

    prevention_result = qa_chain.invoke(row['question'])
    result_text = prevention_result['result']
    test_results.append(result_text)

print("\n테스트 완료! 총 결과 수:", len(test_results))

# 제출 파일 생성
submission = pd.read_csv(SUBMISSION_FILE, encoding="utf-8-sig")
submission.iloc[:, 1] = test_results  # 결과 저장

# 결과 CSV 저장
submission.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
print("결과 저장 완료:", OUTPUT_FILE)
