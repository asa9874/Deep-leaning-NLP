from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFaceEndpoint
import os

repo_id = "microsoft/Phi-3-mini-4k-instruct"

llm = HuggingFaceEndpoint(
    repo_id=repo_id, 
    max_new_tokens=256,  
    temperature=0.1,
    huggingfacehub_api_token=os.environ["HUGGINGFACEHUB_API_TOKEN"],   
)

template = """
<|system|>당신은 모든 대답을 한국어로 하고 친절하게 대답합니다.<|end|>
<|user|>{Question}<|end|>
<|assistant|>
"""

prompt = PromptTemplate.from_template(template=template)
chain = prompt | llm | StrOutputParser()


while True:
    Question = input("질문: ")
    result = chain.invoke({"Question": Question})
    print(result)