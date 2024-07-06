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
from bs4 import BeautifulSoup
import requests
import random
from dotenv import load_dotenv
load_dotenv()

app = Flask(__name__)

# vercel.json에서 전체 허용으로 설정
CORS(app)
@app.route('/')
def home():
   return render_template('index.html')

@app.route('/resume_guide', methods=['POST','OPTIONS'])
def resume_guide():
    if request.method == 'OPTIONS':
        return '', 200  # Preflight response must be HTTP 200 OK
    try:
        data = request.json
        
        # data parsing
        status = data.get("status")
        company = data.get("company")
        occupation = data.get("occupation")
        question_list = [list(item.values())[0] for item in data.get("questions")]
        award_list = [list(item.values())[0] for item in data.get("awards")] if data.get("awards") else "Empty"
        experience_list = [list(item.values())[0] for item in data.get("experiences")] if data.get("experiences") else "Empty"
        questions='\n'.join(question_list)
        awards = '\n'.join(award_list)
        experiences = '\n'.join(experience_list)

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
            collection_name="resume_all_1536",
            embeddings=embeddings
        )
    
        #OpenAI Model
        llm = ChatOpenAI(model="gpt-4o", temperature=1)
    
        # Retriever 3개 까지
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    
        # VectorDBQA 체인 생성
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True
        )
        query = f""" 회사와 직무를 참고하여 자기소개서 질문 리스트에 각각 수상 내역과 직무관련 경험을 분배해줘.
        가이드는 다음과 같은 순서로 진행 돼. 순차적으로 수행하되 수행 완료된 내용만 보여줘. 관련된 수상 내역이나 직무 관련 경험이 없다면 추천할 항목이 없다고 표시해줘. 
        1. 질문 리스트마다 소재로 삼으면 좋을 수상 경력 혹은 직무 관련 경험을 겹치지 않도록 골라서 내용만 그대로 기재한다.(복수 선택 가능)
        2. 해당 경험을 어떤 식으로 적으면 좋을 지 간단한 예시를 보여준다.
        신입/경력 여부: {status}, 회사: {company}, 희망 직무: {occupation}
        질문 리스트: {questions}, 수상 내역:{awards}, 직무 관련 경험:{experiences}
        """

        result = qa_chain.invoke({"query": query})
        return jsonify({"status": "Success", "result":result["result"]}), 200
    except Exception as e:
        return jsonify({'status':'Fail', 'error':str(e)}),500

    

@app.route('/rag_chat', methods=['POST','OPTIONS'])
def process():
    if request.method == 'OPTIONS':
        return '', 200  # Preflight response must be HTTP 200 OK
    
    try:
        data = request.json
        
        # data parsing
        status = data.get("status")
        company = data.get("company")
        occupation = data.get("occupation")
        question=data.get("question")
        content = data.get("answer")
        model = data.get("model")
        temperature = data.get("temperature")
        collection_name = data.get("collection") if data.get("collection") else "resume_detail"
        mode = data.get("mode") if data.get("mode") else "lite"
        technique = data.get("technique") if data.get("technique") else "normal"
        
        if len(content)<100:
            return jsonify({'status':'Fail','diff':None, 'result': '100자 이상 작성해주세요.'})
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
        if technique == "normal":
            query = f"""다음 자기소개서를 단계별로 첨삭해주고 실명이 포함되어 있다면 삭제해줘. 1단계: 장점과 경험을 더 드러낼 수 있는 문장으로 수정합니다. 2단계: 문항에 기재된 질문사항에 맞게 답변하도록 수정합니다. 3단계: 부자연스러운 문장을 수정합니다. 답변은 최종 수정된 내용만 보여줘 Condition: 1. 종류: {status} 2. 회사: {company} 3. 업종: {occupation} 4. 문항 : {question}\n Content: {content}\n\n"""
        else:
            query = f"""다음 조건에 맞는 자기소개서를 {technique}기법을 적용하여 수정해주고 소제목 없이 자연스럽게 연결해줘. 답변은 Content와 글자수가 비슷하게 맞추되, 수정된 내용만 출력해줘. Condition: 1. 종류: {status} 2. 회사: {company} 3. 업종: {occupation} 4. 문항 : {question}\n Content: {content}\n\n """
    
    
        result = qa_chain.invoke({"query": query})
        
        #diff-match-patch
        dmp = diff_match_patch()
        dmp.Diff_EditCost = 4
        diff = dmp.diff_main(content, result["result"])
        dmp.diff_cleanupSemantic(diff)
        
        #print(result["source_documents"])
        return jsonify({'status':'Success','diff':diff, 'result': result["result"]+"\n\n\n*"+technique+" 방식으로 작성된 자소서 입니다."})
    except Exception as e:
        return jsonify({'status':'Fail', 'error':str(e)}),500
    
@app.route('/job_search', methods=['POST', 'OPTIONS'])
def job_search():
    if request.method == 'OPTIONS':
        return '', 200  # Preflight response must be HTTP 200 OK
    try:
        data = request.json
        
        # data parsing
        salTp = data.get("salTp")
        minPay = data.get("minPay")
        maxPay = data.get("maxPay")
        career=data.get("career")
        keyword = data.get("keyword")
        education = data.get("education")
        
        url = "http://openapi.work.go.kr/opi/opi/opia/wantedApi.do?authKey="
        serviceKey = os.environ['WORKNET_API_KEY']
        Calltp = "&callTp=L"
        Return = "&returnType=XML"
        StartPage="&startPage=1"
        empTpGb="&empTpGb=1"
        # 끝 1095page
        Display = "&display=100"
        option = f"&education={education}&salTp={salTp}&minPay={minPay}&maxPay={maxPay}&career={career}&keyword={keyword}"
        
        def parse():
            try:
                COMPANY = wanted.find("company").get_text()
                TITLE = wanted.find("title").get_text()
                SAL_TMNM = wanted.find("salTpNm").get_text()
                SAL = wanted.find("sal").get_text()
                REGION = wanted.find("region").get_text()
                HOLIDAY_TPNM = wanted.find("holidayTpNm").get_text()
                MIN_DEUBG = wanted.find("minEdubg").get_text()
                CAREER = wanted.find("career").get_text()
                regDt = wanted.find("regDt").get_text()
                jobsCd = wanted.find("jobsCd").get_text()
                return {
                    "회사명":COMPANY,
                    "체용제목":TITLE,
                    "임금형태":SAL_TMNM,
                    "급여":SAL,
                    "근무지역":REGION,
                    "근무형태":HOLIDAY_TPNM,
                    "최소학력":MIN_DEUBG,
                    "경력":CAREER,
                    "등록일자":regDt,
                    "직종코드":jobsCd
                }
            except:
                return {
                    "회사명":None,
                    "체용제목":None,
                    "임금형태":None,
                    "급여":None,
                    "근무지역":None,
                    "근무형태":None,
                    "최소학력":None,
                    "경력":None,
                    "직종코드":None,
                    "우대조건":None
                }
            
        #parsing 하기
        result = requests.get(url+serviceKey+Calltp+Return+StartPage+empTpGb+Display+option)
        soup = BeautifulSoup(result.text,'lxml-xml')
        wanteds = soup.find_all("wanted")
         
        row = []
        for wanted in wanteds:
            row.append(parse()) 
            
        return jsonify({"status":"Success","result":row}),200
    except Exception as e:
        return jsonify({'status':'Fail', 'error':str(e)}),500
    
    
@app.route('/social_enterprise', methods=['GET', 'OPTIONS'])
def social_enterprise():
    if request.method == 'OPTIONS':
        return '', 200  # Preflight response must be HTTP 200 OK
    try:
        page = random.randint(0,6500)
        serviceKey = os.environ["SOCIAL_ENTERPRISE_API_KEY"]
        url = f"https://api.odcloud.kr/api/socialEnterpriseList/v1/authCompanyList?page={page}&perPage=1&serviceKey={serviceKey}"
        response = requests.get(url)
    
        if response.status_code == 200:
            data = response.json()
            return jsonify(data)
        else:
            return jsonify({"status": "Error"}), response.status_code
    except Exception as e:
        return jsonify({'status':'Fail', 'error':str(e)}),500
    
if __name__ == '__main__':
    app.run(host='localhost', port=5000)