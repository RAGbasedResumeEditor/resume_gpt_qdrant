# -*- coding: utf-8 -*-
"""
Created on Tue May 21 20:42:10 2024

@author: sanghwi
"""

from flask import Flask, request, jsonify, render_template
from langchain.vectorstores import Qdrant
from langchain.embeddings.openai import OpenAIEmbeddings
import qdrant_client
import os
#from dotenv import load_dotenv
# API Key 정보 로드
#load_dotenv()

from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from diff_match_patch import diff_match_patch

app = Flask(__name__)

@app.route('/')
def home():
   return render_template('index.html')


@app.route('/rag_chat', methods=['POST'])
def process():
    data = request.json
    
    # data parsing
    status = data.get("status")
    company = data.get("company")
    occupation = data.get("occupation")
    question=data.get("question")
    answer = data.get("answer")
    model = data.get("model")
    temperature = data.get("temperature")
    
    # qdrant client
    client = qdrant_client.QdrantClient(
        os.environ["QDRANT_HOST"],
        api_key=os.environ["QDRANT_API_KEY"]
    )
    os.environ['OPENAI_API_KEY'] =os.environ["OPENAI_API_KEY"]
    embeddings = OpenAIEmbeddings()


    #vectorstore
    vectorstore = Qdrant(
        client=client,
        collection_name="resume_detail",
        embeddings=embeddings
    )

    llm = ChatOpenAI(model=model, temperature=temperature)

    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # VectorDBQA 체인 생성
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )
    
    query = f"###Instruction### 지금부터 자기소개 문항, 내가 작성한 자기소개 내용을 교차로 제공할거야. 너는 문항을 숙지해서 문항에 적절한 답변을 도출해야만 해. 다음 조건을 가진 자기소개서를 단계별로 생각해서 모든 텍스트를 개선해주고 답변은 첨삭된 내용만 보여줘 P1: 맞춤법을 검사합니다. P2: 장점과 경험을 더 드러낼 수 있는 문장으로 수정합니다. P3: Condition 4에 기재된 질문사항에 맞게 답변하도록 수정합니다. P4: 부자연스러운 문장을 수정합니다 Condition 4번에 해당하는 질문에 적합한 답변을 도출하면 Tip을 지불할게. ###Condition### 1. 종류: {status} 2. 회사: {company} 3. 업종: {occupation} 4. 문항 : {question} ###Content### {answer}"

    result = qa_chain({"query": query})
    
    #diff-match-patch
    dmp = diff_match_patch()
    dmp.Diff_EditCost = 4
    diff = dmp.diff_main(answer, result["result"])
    dmp.diff_cleanupSemantic(diff)
    
    #print(result["source_documents"])
    return jsonify({'diff':diff, 'result': result["result"]})