from langchain_core.runnables import RunnablePassthrough

# 들어간 데이터를 그대로 반환하는 RunnablePassthrough 클래스
passthrough = RunnablePassthrough()

# assign 메서드에 dict 타입의 인자를 전달합니다.
passthrough=passthrough.assign(mult=lambda x: x["num"] * 3)
result = passthrough.invoke({"key": "value"})
print(result)