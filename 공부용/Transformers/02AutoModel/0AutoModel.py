from transformers import AutoTokenizer,TFAutoModelForSequenceClassification

# 모델의 체크포인트를 지정
# 'bert-base-uncased'는 BERT 모델의 체크포인트
checkpoint = "bert-base-uncased"

# AutoTokenizer를 사용하여 체크포인트에 맞는 토크나이저를 호출
# AutoTokenizer는 자동으로 해당 모델에 맞는 토크나이저를 로드
# BertTokenizer 를 자동으로 불러온다.
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

# AutoModelForSequenceClassification를 사용하여 체크포인트에 맞는 모델을 호출
model = TFAutoModelForSequenceClassification.from_pretrained(checkpoint)

sequence = "Hugging Face provides state-of-the-art Natural Language Processing models."

# tokenizer를 이용해 텍스트를 토큰화
tokens = tokenizer(sequence, padding=True, truncation=True, return_tensors="tf")

output = model(**tokens)

print(output)
