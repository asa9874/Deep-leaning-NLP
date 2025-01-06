from transformers import pipeline

conversation = pipeline("text-generation", model="microsoft/DialoGPT-medium")

# 대화 이력을 주고 대답을 유도
dialogue_history = "User: Hello, who are you?\nBot: My name is chatbot\nUser: What is your name?\nBot:"

response = conversation(dialogue_history, max_length=100)
print(response[0]['generated_text'])