from langchain_core.prompts import PromptTemplate

from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import OllamaLLM
from datetime import datetime
llm = OllamaLLM(model="llama3.1")

def get_today():
    return datetime.now().strftime("%B %d")

template = """오늘은 {today}이다. 오늘의 날짜와 {country}과 {country2}의 수도는 어디입니까?"""

prompt = PromptTemplate(
    template=template,
    input_variables=["country"],
    partial_variables={
        "country2": "미국" ,
        "today": get_today 
    },

)

chain = prompt | llm | StrOutputParser()

while True:
    country = input("나라를 입력하세요: ")
    result = chain.invoke({"country": country})
    print(result)