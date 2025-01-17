from transformers import pipeline
summarizer = pipeline("summarization")
text = "Artificial intelligence has rapidly evolved over the past few decades, transforming numerous industries by enabling machines to perform tasks that were once considered exclusive to human beings, such as understanding natural language, recognizing images, and making decisions based on data. With advancements in machine learning algorithms, deep learning models, and the availability of massive datasets, AI has made significant strides in areas like healthcare, finance, autonomous vehicles, and entertainment, bringing both immense potential for innovation and challenges regarding ethics, privacy, and job displacement."
summary = summarizer(text)

print(summary)