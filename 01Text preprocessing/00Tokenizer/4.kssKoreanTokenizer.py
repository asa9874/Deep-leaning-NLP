# 한국어 토큰화

import kss

text = '나는 자랑스럽다. 나 자신이 정말 자랑스러운가? 그래그래 맞다.'
print(kss.split_sentences(text))
print(kss.split_morphemes(text))