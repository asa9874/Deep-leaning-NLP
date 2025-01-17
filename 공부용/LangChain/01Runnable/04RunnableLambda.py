from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda,RunnablePassthrough

from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import OllamaLLM
from datetime import datetime
llm = OllamaLLM(model="llama3.1")

def get_today(a):
    # 오늘 날짜를 가져오기
    return datetime.today().strftime("%b-%d")

prompt=PromptTemplate.from_template("""{country}의 수도는 어디입니까? {country}의 {today}는 무슨요일인가요?""")


chain1 = (
    {"today":RunnableLambda(get_today),"country": RunnablePassthrough()}
      | prompt 
      | llm 
      | StrOutputParser()
)


while True:
    country = input("나라를 입력하세요: ")
    result = chain1.invoke({"country": country})
    print(result)