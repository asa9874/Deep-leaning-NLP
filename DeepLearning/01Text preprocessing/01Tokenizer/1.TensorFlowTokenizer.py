#단어 토큰라이저 (상수 텐서) 

import tensorflow as tf
from nltk.tokenize import TreebankWordTokenizer
#상수 텐서
text = tf.constant("Hello, my name is asa! How are you?")

# 특수 문자 제거
text = tf.strings.regex_replace(text, r'[^a-zA-Z0-9 ]', '')

# 단어로 분리 
tokens = tf.strings.split(text)

# 결과 출력
# NumPy 배열로 변환해 확인
print(tokens.numpy())  

