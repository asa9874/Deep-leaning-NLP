#https://wikidocs.net/251190

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaLLM
from langchain.embeddings import HuggingFaceEmbeddings

#문서 로드 
loader = PyMuPDFLoader("08RAG/test.pdf")
docs = loader.load()

#문서 분할 
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
split_documents = text_splitter.split_documents(docs)

#임베딩 생성
model_name = "sentence-transformers/all-mpnet-base-v2"
embeddings = HuggingFaceEmbeddings(model_name=model_name)

#DB 생성
vectorstore = FAISS.from_documents(documents=split_documents, embedding=embeddings)

#검색기 생성
retriever = vectorstore.as_retriever()

# 프롬프트 생성
prompt = PromptTemplate.from_template(
    """
You are a highly accurate assistant specializing in Korean. 
Use only the provided context to answer the question, and include detailed reasoning.
Do not make assumptions or provide unrelated information.
모든 답변은 - 형태로 나눠서 한다.

--------------
#질문: 
{question} 
#Context: 
{context} 

#답변:"""
)

#언어모델(LLM) 생성
llm = OllamaLLM(model="llama3.1")

#체인(Chain) 생성
chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

question = "메구밍에 대해 알려줘"
response = chain.invoke(question)
print(response)