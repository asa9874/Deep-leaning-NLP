from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from numpy import dot
from numpy.linalg import norm
def cos_sim(A, B):
  return dot(A, B)/(norm(A)*norm(B))

corpus = [
    "hello my name is hello hello",
    "my name is John",
    "hello John my name"
]
#TF-IDF
tfidfv = TfidfVectorizer().fit(corpus)
tfidfArr=tfidfv.transform(corpus).toarray()
print(tfidfArr)
print(tfidfv.vocabulary_)

#cos_sim 유사도
print("유사도:",cos_sim(tfidfArr[0],tfidfArr[1]))