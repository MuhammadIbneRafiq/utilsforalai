from langchain_groq import ChatGroq
import os
from datetime import datetime
import sentence_transformers
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_community.document_loaders.merge import MergedDataLoader
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.output_parsers import JsonOutputParser
from typing import Union, List, Tuple, Dict
from langchain_chroma import Chroma
from pdfminer.high_level import extract_text
from langchain.schema import Document

GROQ_LLM = ChatGroq(
    api_key=os.getenv('GROQ_API_KEY'),
    model="llama3-70b-8192"
) 

file_path = './invoice_from_ocr.pdf'

def rag_chain_node():
    # Try loading PDF with PyPDFLoader
    # print("Loading PDF with PyPDFLoader...")
    loader_csv = PyPDFLoader(file_path=file_path)
    
    docs_all = loader_csv.load()
    # print(f"Loaded documents: {docs_all}")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=300)
    
    texts = text_splitter.split_documents(docs_all)
    # print(f"Split texts: {texts}")

    from langchain_community.embeddings import HuggingFaceBgeEmbeddings
    model_name = "BAAI/bge-base-en"
    encode_kwargs = {'normalize_embeddings': True}  # set True to compute cosine similarity
    bge_embeddings = HuggingFaceBgeEmbeddings(
        model_name=model_name,
        encode_kwargs=encode_kwargs
    )
    # print(f"Using embeddings model: {model_name}")

    persist_directory = 'db'
    embedding = bge_embeddings
    # print("Creating vector database from documents...")

    vectordb = Chroma.from_documents(documents=texts, embedding=embedding, persist_directory=persist_directory)
    retriever = vectordb.as_retriever(search_kwargs={"k": 5})

    rag_prompt = PromptTemplate(
        template="""system
        You are an accounting assistant for looking through invoices in Dutch. 
        Use the following pieces of retrieved context from invoices in pdf format and sort them in category. 
        These texts from the pdf is read from an OCR so some of the words may not make sense, so trust the numbers.
        Use the context to derive your answer.
        If you don't know the answer, just say that you don't know but first try out some stuff.
        user
        QUESTION: {question} \n
        CONTEXT: {context} \n
        Answer:
        
        assistant
        """,
        input_variables=["question", "context"],
    )
    
    rag_prompt_chain = rag_prompt | GROQ_LLM | StrOutputParser()
    QUESTION = """What was all the orders in the invoice AND the total??"""
    CONTEXT = retriever.invoke(QUESTION)
    print(f"Retrieved context: {CONTEXT}")

    result = rag_prompt_chain.invoke({"question": QUESTION, "context": CONTEXT})
    print('ANSWER', result)

rag_chain_node()
