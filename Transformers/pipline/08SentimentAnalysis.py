from transformers import pipeline
sentiment_analyzer = pipeline("sentiment-analysis")
result = sentiment_analyzer("I am so happy with this product!")
print(result)