# -*- coding: utf-8 -*-
"""
Created on Tue May 21 20:42:10 2024

@author: sanghwi
"""

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from langchain_qdrant import Qdrant
import qdrant_client
import os
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from diff_match_patch import diff_match_patch

app = Flask(__name__)

# vercel.json에서 전체 허용으로 설정
CORS(app)

@app.route('/')
def home():
   return render_template('index.html')


@app.route('/rag_chat', methods=['POST','OPTIONS'])
def process():
    if request.method == 'OPTIONS':
        return '', 200  # Preflight response must be HTTP 200 OK
    data = request.json
    
    # data parsing
    status = data.get("status")
    company = data.get("company")
    occupation = data.get("occupation")
    question=data.get("question")
    content = data.get("answer")
    # model = data.get("model")
    model = 'ft:gpt-3.5-turbo-0125:personal:reditor:9TBncHsL'
    temperature = data.get("temperature")
    collection_name = data.get("collection") if data.get("collection") else "resume_detail"
    mode = data.get("mode") if data.get("mode") else "lite"
    
    # qdrant client
    client = qdrant_client.QdrantClient(
        os.environ["QDRANT_HOST"],
        api_key=os.environ["QDRANT_API_KEY"]
    )
    os.environ['OPENAI_API_KEY'] =os.environ["OPENAI_API_KEY"]
    
    #embedding
    embeddings = OpenAIEmbeddings()


    #vectorstore
    vectorstore = Qdrant(
        client=client,
        collection_name=collection_name,
        embeddings=embeddings
    )

    #OpenAI Model
    llm = ChatOpenAI(model=model, temperature=temperature)

    # Retriever 3개 까지
    kwarg = 5 if mode=="pro" else 3
    retriever = vectorstore.as_retriever(search_kwargs={"k": kwarg})

    # VectorDBQA 체인 생성
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )
    
    #prompt engineering
    # query = f"Instruction: 자기소개서를 단계별로 첨삭해줘. P1: 장점과 경험을 더 드러낼 수 있는 문장으로 수정합니다. P2: 문항에 기재된 질문사항에 맞게 답변하도록 수정합니다. P3: 부자연스러운 문장을 수정합니다. Condition: 1. 종류: {status} 2. 회사: {company} 3. 업종: {occupation} 4. 문항 : {question} \n Content: {content} \n"
    query = f"""다음 자기소개서를 단계별로 첨삭해주고 실명이 포함되어 있다면 삭제해줘. 1단계: 장점과 경험을 더 드러낼 수 있는 문장으로 수정합니다. 2단계: 문항에 기재된 질문사항에 맞게 답변하도록 수정합니다. 3단계: 부자연스러운 문장을 수정합니다. 답변은 최종 수정된 내용만 보여줘 Condition: 1. 종류: {status} 2. 회사: {company} 3. 업종: {occupation} 4. 문항 : {question}\n Content: {content}\n\n"""


    result = qa_chain.invoke({"query": query})
    
    #diff-match-patch
    dmp = diff_match_patch()
    dmp.Diff_EditCost = 4
    diff = dmp.diff_main(content, result["result"])
    dmp.diff_cleanupSemantic(diff)
    result
    #print(result["source_documents"])
    return jsonify({'diff':diff, 'result': result["result"]})
    
if __name__ == '__main__':
    app.run(port=5000)