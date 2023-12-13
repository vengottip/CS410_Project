import pandas as pd
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# Load data
data = pd.read_csv('tweets.csv')

# Initialize the SentimentIntensityAnalyzer
sia = SentimentIntensityAnalyzer()

# Apply the analyzer to the text data
data['sentiment_scores'] = data['text'].apply(lambda text: sia.polarity_scores(text))

# Calculate the sentiment based on the scores
data['sentiment'] = data['sentiment_scores'].apply(lambda score_dict: score_dict['compound'])

print(data.head())