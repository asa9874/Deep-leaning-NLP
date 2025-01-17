from langchain_core.prompts import PromptTemplate

from langchain_core.output_parsers import StrOutputParser,PydanticOutputParser
from langchain_ollama import OllamaLLM
from pydantic import BaseModel, Field
llm = OllamaLLM(
    model="llama3.1",
    format="json"
)


prompt = PromptTemplate(
    template=
    """
    다음 내용중 중요 내용을 요약하시오: {email}
    
    Format:{format}

    """
)

class EmailSummary(BaseModel):
    person: str = Field(description="메일을 보낸 사람")
    email: str = Field(description="메일을 보낸 사람의 이메일 주소")
    subject: str = Field(description="메일 제목")
    summary: str = Field(description="메일 본문을 요약한 텍스트")
    date: str = Field(description="메일 본문에 언급된 미팅 날짜와 시간")

parser = PydanticOutputParser(pydantic_object=EmailSummary)
prompt = prompt.partial(format=parser.get_format_instructions())

chain = prompt | llm | StrOutputParser()

email = """
From: 박지훈 (jihoon.park@cybertechsolutions.kr)
To: 김은지 (eunji.kim@futuretechindustries.co.kr)
Subject: "AEROLITE" 스마트폰 부품 공급 협력 제안 및 미팅 요청

안녕하세요, 김은지 대리님,

저는 사이버테크솔루션즈의 박지훈 부장입니다. 최근 귀사의 "AEROLITE" 스마트폰 모델에 대해 관련 기사 및 보도자료를 통해 접하게 되었고, 해당 모델이 뛰어난 성능과 혁신적인 디자인을 자랑한다는 점에서 큰 관심을 가지고 있습니다. 저희 사이버테크솔루션즈는 스마트폰 부품 및 액세서리 제조에 있어 오랜 경험과 전문성을 바탕으로 시장에서 신뢰받는 기업입니다.

AEROLITE 모델의 부품 공급 가능성을 논의하고자 합니다. 특히, 스마트폰의 화면 디스플레이, 배터리 성능, 그리고 카메라 모듈 등 주요 부품에 대한 세부 정보와 기술 사양을 요청드리고 싶습니다. 이를 통해 저희가 귀사의 요구에 맞는 맞춤형 부품을 공급할 수 있는 방안을 제시하고자 합니다.

또한, 협력 가능성을 보다 구체적으로 논의하고, 귀사와의 유망한 파트너십을 구축할 수 있는 기회를 마련하고자 합니다. 이를 위해 다음 주 목요일(1월 18일) 오전 10시에 미팅을 제안드립니다. 귀사 사무실에서 만나 귀사의 제품과 요구 사항에 대해 더 깊은 이야기를 나눌 수 있을까요?

회의 후에는 협력 세부 사항과 가능한 공급 계획에 대해 논의하고, 상호 윈-윈할 수 있는 방향으로 진행할 수 있도록 하겠습니다.

시간을 내어 주시면 감사하겠습니다. 빠른 회신을 부탁드리며, 바쁘시겠지만 일정을 확인해 주시면 감사하겠습니다.

감사합니다.

박지훈
부장
사이버테크솔루션즈

"""





response = chain.invoke({"email": email})

print(response)