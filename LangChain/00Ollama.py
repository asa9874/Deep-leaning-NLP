from langchain_core.prompts import PromptTemplate

from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import OllamaLLM

llm = OllamaLLM(model="llama3.1")

template = "{country}의 수도는 어디입니까?"

prompt = PromptTemplate.from_template(template=template)
chain = prompt | llm | StrOutputParser()

while True:
    country = input("나라를 입력하세요: ")
    result = chain.invoke({"country": country})
    print(result)