# ignore this python file, this will be later converted into the utils.js file
from langchain_groq import ChatGroq
import os
from datetime import datetime
import sentence_transformers
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_community.document_loaders.merge import MergedDataLoader
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.output_parsers import JsonOutputParser
from typing import Union, List, Tuple, Dict
from langchain_chroma import Chroma

GROQ_LLM = ChatGroq(
            api_key=os.getenv('GROQ_API_KEY'),
            model="llama3-70b-8192"
        ) 

def rag_chain_node():
    loader_csv = PyPDFLoader(file_path='freelancers_data.pdf')
    loader_all = MergedDataLoader(loaders=[loader_csv]) 
    docs_all = loader_all.load()
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=30)
    texts = text_splitter.split_documents(docs_all)
    from langchain_community.embeddings import HuggingFaceBgeEmbeddings
    model_name = "BAAI/bge-base-en"
    encode_kwargs = {'normalize_embeddings': True} # set True to compute cosine similarity
    bge_embeddings = HuggingFaceBgeEmbeddings(
        model_name=model_name,
        encode_kwargs=encode_kwargs
    )
    # from langchain_chroma import Chroma
    persist_directory = 'db'
    ## Heres the embeddings
    embedding = bge_embeddings
    vectordb = Chroma.from_documents(documents=texts, embedding=embedding, persist_directory=persist_directory)
    retriever = vectordb.as_retriever(search_kwargs={"k": 5})

    rag_prompt = PromptTemplate(
        template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
        You are an assistant for question-answering tasks. 
        Use the following pieces of retrieved context to answer the question. 
        If you don't know the answer, just say that you don't know. 
        Use 2 sentences maximum and keep the answer concise.\n
        <|eot_id|><|start_header_id|>user<|end_header_id|>
        QUESTION: {question} \n
        CONTEXT: {context} \n
        Answer:
        <|eot_id|>
        <|start_header_id|>assistant<|end_header_id|>
        """,
        input_variables=["question","context"],
    )
    
    rag_prompt_chain = rag_prompt | GROQ_LLM | StrOutputParser()
    QUESTION = """What can I do with video editing freelancers?"""
    CONTEXT = retriever.invoke(QUESTION)
    result = rag_prompt_chain.invoke({"question": QUESTION, "context":CONTEXT})
    
    rag_chain = (
    {"context": retriever , "question": RunnablePassthrough()}
    | rag_prompt
    | GROQ_LLM
    | StrOutputParser()
    )
    return rag_chain.invoke("I am looking for freelancers for video editing. how much budget options are available?")


print(rag_chain_node())