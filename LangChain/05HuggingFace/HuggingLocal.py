from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFacePipeline
import os

os.environ["TRANSFORMERS_CACHE"] = "./cache/"
os.environ["HF_HOME"] = "./cache/"

llm = HuggingFacePipeline.from_model_id(
    model_id="microsoft/Phi-3-mini-4k-instruct",
    task="text-generation",
    pipeline_kwargs={
        "max_new_tokens": 256,
        "top_k": 50,
        "temperature": 0.1,
    },
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