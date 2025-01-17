from transformers import pipeline

fill_mask = pipeline("fill-mask", model="bert-base-uncased")

# 문장에서 마스크된 단어 예측
result = fill_mask("The capital of France is [MASK].")

print(result)