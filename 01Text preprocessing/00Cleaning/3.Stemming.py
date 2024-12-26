#Stemming

from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import tensorflow as tf

#포터 알고리즘
stemmer = PorterStemmer()

text = tf.constant("name, naming, named, names")
text = tf.strings.lower(text)
tokens = word_tokenize(text.numpy().decode('utf-8'))
tokens = [token for token in tokens if token.isalnum()] 

print(tokens)
print([stemmer.stem(word) for word in tokens])