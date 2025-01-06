from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

corpus = [
    "hello my name is hello hello",
    "my name is John",
    "hello John my name"
]

#DTM
vector = CountVectorizer().fit(corpus)
print(vector.fit_transform(corpus).toarray()) 
print(vector.vocabulary_)


#TF-IDF
tfidfv = TfidfVectorizer().fit(corpus)
print(tfidfv.transform(corpus).toarray())
print(tfidfv.vocabulary_)