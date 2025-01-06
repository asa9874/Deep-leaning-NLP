from soynlp.word import WordExtractor
from soynlp import DoublespaceLineCorpus
import pickle
import os
# 현재 스크립트가 위치한 디렉토리 경로 가져오기
script_dir = os.path.dirname(os.path.realpath(__file__))

# stopwords.xlsx의 절대 경로 생성
file_path = os.path.join(script_dir, "2016-10-20.txt")

# 파일읽고 나누기기
corpus = DoublespaceLineCorpus(file_path)

#단어 점수를 추출하는 기능
word_extractor = WordExtractor()

#텍스트 데이터(corpus)로 단어 점수 학습을 시작
word_extractor.train(corpus)

# 학습된 데이터를 바탕으로 각 단어의 점수를 추출
word_score_table = word_extractor.extract()

# pickle을 사용하여 데이터 저장
with open('word_score_table.pkl', 'wb') as f:
    pickle.dump(word_score_table, f)