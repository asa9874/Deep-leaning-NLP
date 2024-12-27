from konlpy.tag import Okt

okt = Okt()
def build_bag_of_words(text):
    # 토큰화
    tokens = okt.morphs(text)
    tokens = [token for token in tokens if token.isalnum()]  # 알파벳 및 숫자만 필터링
    
    word_idx = {}  # 단어와 인덱스를 매핑
    freq = []  # 단어의 빈도
    for word in tokens:
        if word not in word_idx:
            word_idx[word] = len(word_idx)
            freq.append(1)  # 새로 등장한 단어는 1로 초기화
        else:
            idx = word_idx[word]
            freq[idx] += 1  # 재등장한 단어는 빈도수 증가

    return word_idx, freq


text="최근 들어 많은 사람들이 취미로 코딩을 배우기 시작했습니다. 특히 온라인 교육 플랫폼과 유튜브 등의 자원을 활용하여 쉽고 빠르게 기초를 배우고 있습니다. 코딩을 배우면 문제 해결 능력도 향상되고, 창의적 사고를 기를 수 있는 장점이 많습니다. 또한, 프로그래밍 언어를 배우면 다양한 분야에서 활용할 수 있어 실용적입니다."

text2="최근 들어 많은 사람들이 취미로 코딩을 배우기 시작했습니다."

print(build_bag_of_words(text + text2))