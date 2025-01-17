import re
import urllib.request
import zipfile
from lxml import etree
from nltk.tokenize import word_tokenize, sent_tokenize
from gensim.models import Word2Vec

### 데이터 다운로드
urllib.request.urlretrieve("https://raw.githubusercontent.com/ukairia777/tensorflow-nlp-tutorial/main/09.%20Word%20Embedding/dataset/ted_en-20160408.xml", filename="ted_en-20160408.xml")



### 전처리
targetXML = open('ted_en-20160408.xml', 'r', encoding='UTF8')
target_text = etree.parse(targetXML)

# Content 부분을 가져옴.
parse_text = '\n'.join(target_text.xpath('//content/text()'))

# 정규 표현식
content_text = re.sub(r'\([^)]*\)', '', parse_text)

# 문장 토큰화
sent_text = sent_tokenize(content_text)

# 구두점을 제거,대문자를 소문자로 변환.
normalized_text = []
for string in sent_text:
    tokens = re.sub(r"[^a-z0-9]+", " ", string.lower())
    normalized_text.append(tokens)

# 각 문장에 대해서 NLTK를 이용하여 단어 토큰화를 수행.
result = [word_tokenize(sentence) for sentence in normalized_text]



### Word2Vec 훈련
model = Word2Vec(sentences=result, vector_size=100, window=5, min_count=5, workers=4, sg=0)

### 모델 저장
model.wv.save_word2vec_format('eng_w2v') # 모델 저장

print("Word2Vec 모델 저장 완료")