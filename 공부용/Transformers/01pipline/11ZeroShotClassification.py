from transformers import pipeline

classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# 문장과 후보 클래스 정의
sequence = "I had a great time at the concert last night."
candidate_labels = ["positive", "negative", "neutral"]

# 문장을 주어진 클래스에 대해 분류
result = classifier(sequence, candidate_labels)

print(result)