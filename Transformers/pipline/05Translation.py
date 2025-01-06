from transformers import pipeline
translator = pipeline("translation_en_to_fr")
translated_text = translator("Hello, how are you?")

print(translated_text)