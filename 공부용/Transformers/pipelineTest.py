from transformers import AutoModel, AutoTokenizer

model_name = "microsoft/resnet-50"

try:
    model = AutoModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print(f"{model_name}은(는) pipeline에서 사용할 수 있는 모델입니다.")
except Exception as e:
    print(f"{model_name}은(는) pipeline에서 사용할 수 없는 모델입니다. 오류: {e}")