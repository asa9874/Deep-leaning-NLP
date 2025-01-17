#단어 토큰라이저 

from nltk.tokenize import TreebankWordTokenizer
tokenizer = TreebankWordTokenizer()
text = "Hello, my name is asa! How are you?"
print(tokenizer.tokenize(text))  