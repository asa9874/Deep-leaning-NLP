import asyncio
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import OllamaLLM

llm = OllamaLLM(model="llama3.1")

template = """
당신은 영어를 가르치는 10년차 영어 선생님입니다. 상황에 [FORMAT]에 영어 회화를 작성해 주세요.

상황:
{question}

FORMAT:
- 영어 회화:
- 한글 해석:
"""

prompt = PromptTemplate.from_template(template=template)
chain = prompt | llm | StrOutputParser()

async def main():
    question = input("질문을 입력하세요: ")
    async for chunk in chain.astream({"question": question}):
        print(chunk, end="")
    print("\n")

# 비동기 함수 실행
asyncio.run(main())