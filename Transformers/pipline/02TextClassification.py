from transformers import pipeline
classifier = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")
output = classifier("I love this movie!")
print(output)