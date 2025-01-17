from datasets import load_from_disk
from transformers import TrainingArguments,AutoModelForSequenceClassification,Trainer,DataCollatorWithPadding,AutoTokenizer
import numpy as np
import evaluate

### 데이터셋 로드
tokenized_datasets = load_from_disk("./tokenized_datasets")

###  checkpoint 경로
checkpoint_path = "test-trainer/checkpoint-1377"

### 토크나이저와 모델 로드, 데이터셋 패딩
checkpoint = checkpoint_path
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)

### Trainer 설정
training_args = TrainingArguments("test-trainer", evaluation_strategy="epoch")

### 평가 함수
def compute_metrics(eval_preds):
    metric = evaluate.load("glue", "mrpc")
    logits, labels = eval_preds.predictions, eval_preds.label_ids
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)



### Trainer 객체 생성
trainer = Trainer(
    model,
    training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

### Trainer 평가
predictions = trainer.predict(tokenized_datasets["validation"])

print(predictions.predictions.shape, predictions.label_ids.shape)
print(compute_metrics(predictions))

