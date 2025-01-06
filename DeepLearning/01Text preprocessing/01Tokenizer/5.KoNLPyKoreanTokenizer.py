# 한국어 토큰화

from konlpy.tag import Okt
from konlpy.tag import Kkma

okt = Okt()
kkma = Kkma()
text="나는 당신과 그의 소중한 사람이 될수있었나요? 그러면 좋겠네요."

print("형태소")
print(okt.morphs(text))
print(kkma.morphs(text))

print("품사")
print(okt.pos(text))
print(kkma.pos(text))

print("명사")
print(okt.nouns(text))
print(kkma.nouns(text))

