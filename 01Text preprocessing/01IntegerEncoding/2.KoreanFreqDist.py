# 한국어 정수 인코딩
from konlpy.tag import Okt 
from nltk import FreqDist

#토큰화
okt = Okt() 
text="나는 당신과 그의 소중한 사람이 될수있었나요? 그러면 좋겠네요. 당신을 사랑해요."
textTok = okt.nouns(text)
print(textTok) 

#빈도수 계산
vocab = FreqDist(textTok)
print(vocab["당신"])
print(vocab.most_common(4))