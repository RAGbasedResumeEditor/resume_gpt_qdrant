# -*- coding: utf-8 -*-
"""
Created on Tue May 21 20:42:10 2024

@author: rlatk
"""

from flask import Flask, request, jsonify
import getpass
import os


from langchain.vectorstores import Qdrant
from langchain.embeddings.openai import OpenAIEmbeddings
import qdrant_client
import os
from dotenv import load_dotenv
# API Key 정보 로드
load_dotenv()
import bs4
from langchain import hub
from langchain_community.document_loaders import WebBaseLoader
from langchain_qdrant import Qdrant
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA

app = Flask(__name__)

@app.route('/rag_chat', methods=['POST'])
def process():
    data = request.json
    query=data.get("messages")
    client = qdrant_client.QdrantClient(
        os.getenv("QDRANT_HOST"),
        api_key=os.getenv("QDRANT_API_KEY")
    )
    os.environ['OPENAI_API_KEY'] =os.getenv("OPENAI_API_KEY")

    embeddings = OpenAIEmbeddings()

    vectorstore = Qdrant(
        client=client,
        collection_name=os.getenv("QDRANT_COLLECTION"),
        embeddings=embeddings
    )

    llm = ChatOpenAI(model="gpt-4-turbo")

    retriever = vectorstore.as_retriever()

    # VectorDBQA 체인 생성
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )
    result = qa_chain({"query": query})
    print(result["source_documents"])
    return jsonify({'result': result["result"]})

if __name__ == '__main__':
    app.run(host='localhost', port=5000)