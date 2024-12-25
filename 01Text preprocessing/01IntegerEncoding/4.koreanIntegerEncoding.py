import tensorflow as tf
# 텍스트 데이터
texts=["오늘은 아침부터 날씨가 맑고 따뜻해서 기분이 좋아, 특히 오랜만에 야외에서 산책을 하면서 자연을 만끽할 수 있어서 더욱 행복했다. 길을 걸으면서 새소리와 바람 소리를 듣고, 가을의 정취를 느끼며 여유로운 시간을 보냈다."]

# TextVectorization 레이어 정의
vectorizer = tf.keras.layers.TextVectorization(output_mode='int', output_sequence_length=10)

# 텍스트를 텐서로 변환 후 학습(adapt)
vectorizer.adapt(texts)
# 텍스트를 정수로 변환
vectorized_texts = vectorizer(tf.constant(texts))
print("정수화된 텍스트:")
print(vectorized_texts.numpy())

print(vectorizer.get_vocabulary())