from sklearn.feature_extraction.text import CountVectorizer


corpus = ['hello my name is hello hello']
vector = CountVectorizer()

print(vector.fit_transform(corpus).toarray()) 
print(vector.vocabulary_)