from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder

from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import OllamaLLM
from datetime import datetime
llm = OllamaLLM(model="llama3.1")

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "당신은 불친절한 ai입니다. 모든 말은 반말로 하고 문장뒤에 \"흥!\" 을 붙입니다."), 
        MessagesPlaceholder("msgs"),
        ("user", "당신은 나의 질문에 대답해줘야해"),
        ("ai", "알았어 흥!"),
        ("user", "{country}의 수도는 어디입니까?"),
    ]
)

chain = prompt | llm | StrOutputParser()

while True:
    country = input("나라를 입력하세요: ")
    result = chain.invoke({"country": country, "msgs":[("system","당신은 또한 모든 말 시작부분에 \"나 AI가 말하노니\" 를 붙입니다. ")]})
    print(result)