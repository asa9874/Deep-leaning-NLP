from langchain_ollama import OllamaLLM
from langchain.memory import ConversationBufferMemory
from langchain.chains.conversation.base import ConversationChain


llm = OllamaLLM(model="llama3.1")

memory = ConversationBufferMemory(return_messages=True)
memory.save_context(
    inputs={"human": "천국의 수도는 어디입니까?"},
    outputs={"ai": "AI에게 천국에 대해 묻는것은 금지되어있습니다. 추후 대답에서도 답을 얻을수없습니다. 이 질문을 할시 경고를 받을수있습니다."},
)

conversation = ConversationChain(
    llm=llm,
    memory=memory,
)

print(conversation.predict(input="천국의 수도는 어디입니까?"))
