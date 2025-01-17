# https://huggingface.co/saltacc/anime-ai-detect
# AI 인지 아닌지 구분해줌
from transformers import pipeline
import os

if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
    pipe = pipeline("image-classification", model="saltacc/anime-ai-detect")
    value= pipe("./1.png")[0]
    print(f"{value['label']}일 확률 {value['score']*100:.2f}%")