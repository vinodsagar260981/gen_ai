import os
from dotenv import load_dotenv
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI

#import for pdf projects
import urllib
import warnings
from pathlib import Path as p
from pprint import pprint

import pandas as pd
from langchain.prompts import PromptTemplate 
from langchain.chains.question_answering import load_qa_chain
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS



load_dotenv()

API_KEY = os.getenv("API_KEY")
genai.configure(api_key=API_KEY)
model = ChatGoogleGenerativeAI(model="gemini-pro",google_api_key=API_KEY,
                             temperature=0.2,convert_system_message_to_human=True)


# ask_question = input("How Can i help you !: ")
# response = model.generate_content(ask_question)
# print(response.text)

llm = ChatGoogleGenerativeAI(model="gemini-pro",api_key=API_KEY)
result = llm.invoke("what is the currency of swizerland")
# print(result.content.replace("**", '').replace("*", ""))

warnings.filterwarnings("ignore")

pdf_loader = PyPDFLoader("D:/Development/HP_Development/AI/RAG_ReadPDF/western_birds.pdf")
pages = pdf_loader.load_and_split()
# print(pages[9].page_content)
# print(len(pages))


#RAG pipeline : Embedding + gemini LLM

text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
context = "\n\n".join(str(p.page_content) for p in pages)
texts = text_splitter.split_text(context)
print("------text splitter------")

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=API_KEY)
print("-----embedding-----")

vector_index = FAISS.from_texts(texts, embeddings).as_retriever(search_kwargs={"k":15})
print("---vector chroma-----")


qa_chain = RetrievalQA.from_chain_type(
    model,
    retriever=vector_index,
    return_source_documents=True
)

template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. Keep the answer as concise as possible. Always say "thanks for asking!" at the end of the answer.
{context}
Question: {question}
Helpful Answer:"""
QA_CHAIN_PROMPT = PromptTemplate.from_template(template)# Run chain
qa_chain = RetrievalQA.from_chain_type(
    model,
    retriever=vector_index,
    return_source_documents=True,
    chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
)
    
question = input("Ask anything related to PDF(western_ghats): ")
result = qa_chain({"query": question})
print(result["result"])
# print(result["source_documents"])