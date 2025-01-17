from transformers import AutoTokenizer,TFAutoModelForSequenceClassification, DataCollatorWithPadding
import tensorflow as tf
from datasets import load_dataset

### 데이터셋 로드
raw_datasets = load_dataset("glue", "mrpc")

### 토크나이저와 모델 로드
checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = TFAutoModelForSequenceClassification.from_pretrained(checkpoint)
optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)

### 토큰화 함수
def tokenize_function(example):
    return tokenizer(example["sentence1"], example["sentence2"], truncation=True)

### 데이터셋 토큰화
tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)

### 데이터셋 패딩
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# 데이터셋 저장
tokenized_datasets.save_to_disk("./tokenized_datasets")

print(tokenized_datasets["train"][0])