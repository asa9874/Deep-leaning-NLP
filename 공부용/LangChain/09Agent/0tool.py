from langchain_teddynote.tools import GoogleNews
from langchain.tools import tool
from langchain_ollama import OllamaLLM
from langchain.agents import initialize_agent
import re

#도구들정의
@tool
def find_News_ByKeyWords(keyword :str):
    """
    A tool that returns the top five news data received as keywords.
    """
    tool = GoogleNews()
    return tool.search_by_keyword(keyword, k=5)

@tool
def extract_urls(news_data):
    """
    Extract only 'url' from each news item and return it to the list
    """
    urls = re.findall(r'https?://[^\s,\'"]+', news_data)

    return urls

tools = [extract_urls,find_News_ByKeyWords]




llm = OllamaLLM(model="llama3.1")

agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent_type="zero-shot-react-description",  # 에이전트 타입 설정
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=3
)


question = " \"애니메이션\"에 대한 기사 5개를 찾은다음 url을 찾아주세요"
response = agent.run(question)
print(response)