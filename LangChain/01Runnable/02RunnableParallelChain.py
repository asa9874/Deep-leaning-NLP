from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel

from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import OllamaLLM

llm = OllamaLLM(model="llama3.1")


chain1 = (PromptTemplate.from_template(template="""{country}의 수도는 어디입니까?""") 
        | llm 
        | StrOutputParser()
)

chain2 = (PromptTemplate.from_template(template="""{country}는 어느대륙에 있습니까?""") 
        | llm 
        | StrOutputParser()
)

# RunnableParallel 클래스를 사용하여 두 개의 Runnable을 병렬로 실행합니다.
combined_chain = RunnableParallel(capital=chain1, Confession=chain2)

while True:
    country = input("나라를 입력하세요: ")
    result = combined_chain.invoke({"country": country})
    print(result)