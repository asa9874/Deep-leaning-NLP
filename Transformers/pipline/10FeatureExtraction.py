from transformers import pipeline
feature_extractor = pipeline("feature-extraction", model="bert-base-uncased")

# 텍스트에서 특징 벡터 추출
text = "The quick brown fox jumps over the lazy dog"
features = feature_extractor(text)

print(features)