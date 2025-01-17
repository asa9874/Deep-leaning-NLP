from nltk import FreqDist
import numpy as np
from nltk.tokenize import TreebankWordTokenizer
import re
import tensorflow as tf
import tensorflow_text as tf_text

#텍스트
text="hello my name is asa. I'm very strong and hello hello"

#소문자화
text = text.lower()

#구두점 및 특수문자 제거
text = re.sub(r'[^\w\s]', '', text)

#토큰화
tokenizer = TreebankWordTokenizer() 
textTok = tokenizer.tokenize(text) 
print(textTok)

#빈도수 계산
vocab = FreqDist(textTok)
print(vocab["hello"])
print(vocab.most_common(4))


# 텍스트 데이터를 정수로 변환 
vocab = tf_text.vocab.from_text(text)
encoded_texts = vocab.lookup(textTok)  # 단어 집합을 기반으로 정수 인코딩 
print(encoded_texts)