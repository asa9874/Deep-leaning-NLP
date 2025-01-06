from transformers import pipeline
qa = pipeline("question-answering", model="bert-large-uncased-whole-word-masking-finetuned-squad")
context = "Hugging Face is creating amazing tools for NLP."
answer = qa(question="What is Hugging Face creating?", context=context)

print(answer)