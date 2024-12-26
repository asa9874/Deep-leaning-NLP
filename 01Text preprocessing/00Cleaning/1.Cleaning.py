#Cleaning

import tensorflow as tf

# 상수 텍스트
text = tf.constant("Hello, hello, Hello!")

# 비문자 제거: 알파벳과 숫자만 남김
text = tf.strings.regex_replace(text, r'[^a-zA-Z0-9 ]', '')

# 소문자화
text = tf.strings.lower(text)

#토큰화
tokens= tf.strings.split(text)

# 결과 확인
print(tokens.numpy())