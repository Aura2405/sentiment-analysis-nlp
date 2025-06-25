
from transformers import pipeline

sentiment_analyzer = pipeline("sentiment-analysis")

review = "The movie was fantastic, one of it's kind!"

result = sentiment_analyzer(review)

print("Sentiment : ",result[0]['label'])
print("Score : ", f"{result[0]['score']:4f}")