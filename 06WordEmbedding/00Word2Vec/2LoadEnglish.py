from gensim.models import KeyedVectors

loaded_model = KeyedVectors.load_word2vec_format("eng_w2v") # 모델 로드
while True:
    try:
        word = input("단어를 입력하세요: ")
        if word == "exit": break
        output = loaded_model.most_similar(word)
        print(output)
    except:
        print("해당 단어는 모델에 없습니다.")