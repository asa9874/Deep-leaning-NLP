from datasets import load_from_disk
from transformers import TrainingArguments,AutoModelForSequenceClassification,Trainer,DataCollatorWithPadding,AutoTokenizer

### 데이터셋 로드
tokenized_datasets = load_from_disk("./tokenized_datasets")

### 토크나이저와 모델 로드, 데이터셋 패딩
checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)

### Trainer 설정
training_args = TrainingArguments(
    output_dir="test-trainer",  # 저장 디렉토리
    num_train_epochs=3,         # 학습 에포크 수
    per_device_train_batch_size=8,  # 학습 배치 크기
    evaluation_strategy="epoch",   # 매 에포크마다 평가
    save_strategy="epoch",         # 매 에포크마다 체크포인트 저장
    logging_dir="test-logs",       # 로그 저장 디렉토리
)


### Trainer 객체 생성
trainer = Trainer(
    model,
    training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
)

### Trainer 학습
trainer.train()

