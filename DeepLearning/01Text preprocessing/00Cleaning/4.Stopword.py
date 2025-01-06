#stopwords
import nltk
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import tensorflow as tf

nltk.download('stopwords')
#포터 알고리즘
stemmer = PorterStemmer()
text = tf.constant("The internet has revolutionized the way we communicate, learn, and work. It has connected people across the world, breaking down barriers of distance and time. With the rise of social media platforms, individuals can now share their thoughts, ideas, and experiences with a global audience. While the internet has brought countless benefits, it has also introduced new challenges, including privacy concerns and the spread of misinformation. As technology continues to evolve, it is crucial to navigate the digital world responsibly and with awareness.")
text = tf.strings.lower(text)
tokens = word_tokenize(text.numpy().decode('utf-8'))
tokens = [token for token in tokens if token.isalnum()] 
tokens = [stemmer.stem(word) for word in tokens]

#불용어
stop_words = set(nltk.corpus.stopwords.words('english'))
result = []
for word in tokens: 
    if word not in stop_words: 
        result.append(word)


print(tokens)
print(result)