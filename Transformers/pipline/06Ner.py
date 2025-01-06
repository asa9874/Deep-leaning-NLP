from transformers import pipeline 
ner = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english")
result = ner("Hugging Face Inc. is based in New York City.")

print(result)