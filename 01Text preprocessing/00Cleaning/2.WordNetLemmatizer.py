#WordNetLemmatizer
import nltk
from nltk.stem import WordNetLemmatizer
import tensorflow as tf
from nltk.tokenize import word_tokenize

nltk.download('wordnet') 
lemmatizer = WordNetLemmatizer()

# 상수 텍스트
text = tf.constant("name,names")

# 소문자화
text = tf.strings.lower(text)

#토큰화
tokens = word_tokenize(text.numpy().decode('utf-8'))
 
# 알파벳과 숫자만 남김
tokens = [token for token in tokens if token.isalnum()] 

#표제어 추출
lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]  # decode('utf-8') 필요 없음

# 결과 확인
print(lemmatized_tokens)