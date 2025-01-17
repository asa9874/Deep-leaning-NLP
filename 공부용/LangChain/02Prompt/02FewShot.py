from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder,FewShotPromptTemplate,PromptTemplate

from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import OllamaLLM
llm = OllamaLLM(model="llama3.1")

#  예시문장
examples1 = [
    {"question": "일본의 수도는 어디입니까?", "answer": "나 AI가 말하노니, 멍청한놈! 도쿄이다 흥!"},
    {"question": "중국의 수도는 어디입니까?", "answer": "나 AI가 말하노니, 멍청한사람! 베이징이다 흥!"},
]
# 예시문장 템플릿
example_prompt = PromptTemplate(
    template="Question: {question}\nAnswer: {answer}",
    input_variables=["question", "answer"]
)

# fewshot 템플릿
prompt = FewShotPromptTemplate(
    examples=examples1,
    example_prompt=example_prompt,
    suffix="Question:\n{question}\nAnswer:",
    input_variables=["question"],
)

chain = prompt | llm | StrOutputParser()

while True:
    question = input("질문입력:")
    result = chain.invoke({"question": question})
    print(result)