import pickle
import os
from soynlp.tokenizer import LTokenizer
from soynlp.tokenizer import MaxScoreTokenizer
from soynlp.normalizer import *

# 현재 스크립트가 위치한 디렉토리 경로 가져오기
script_dir = os.path.dirname(os.path.realpath(__file__))

# stopwords.xlsx의 절대 경로 생성
file_path = os.path.join(script_dir, 'word_score_table.pkl')


# pickle 파일 불러오기
with open(file_path, 'rb') as f:
    word_score_table = pickle.load(f)

# 응집확률
print(word_score_table["반포한강"].cohesion_forward)

# 브랜칭 엔트로피
print(word_score_table["디스"].right_branching_entropy)

# 응집확률기준으로 L+R 토큰분리 
#띄어쓰기 단위로 나눈 어절 토큰찾기
scores = {word:score.cohesion_forward for word, score in word_score_table.items()}
l_tokenizer = LTokenizer(scores=scores)
print(l_tokenizer.tokenize("범죄를 척결하자", flatten=False))


#  최대 점수 토크나이저
maxscore_tokenizer = MaxScoreTokenizer(scores=scores)
print(maxscore_tokenizer.tokenize("범죄를척결하자"))



# 반복제거
print(emoticon_normalize('반갑습니다다다다다다ㅋㅋㅋㅋ', num_repeats=1))