from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical


text = "한겨울 밤의 하늘은 차갑고 맑았다. 별빛은 검은 캔버스 위에 수를 놓듯 흩어져 있었고, 달은 마치 은빛으로 도금된 보석처럼 찬란하게 빛났다. 바람은 나뭇가지 사이를 스치며 낮게 울렸고, 얼음처럼 차가운 공기는 숨을 쉴 때마다 뺨을 얼렸다. 하지만 이 고요한 순간 속에는 뭔가 설명할 수 없는 따스함이 있었다. 어쩌면 그것은 혼자가 아니라는 깨달음에서 오는 안도감이었을까?"

#정수인코딩
tokenizer = Tokenizer()
tokenizer.fit_on_texts([text])
print('단어 집합 :',tokenizer.word_index)
text2 = "별빛은 검은 캔버스 차갑고 맑았다"
encoded = tokenizer.texts_to_sequences([text2])[0]
print(encoded)

#원핫 인코딩
one_hot = to_categorical(encoded)
print(one_hot)