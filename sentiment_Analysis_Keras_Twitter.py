# Step 1: Import necessary libraries and modules
import tweepy
from textblob import TextBlob
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, Dropout
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import re

# Step 2: Set up Twitter API credentials
consumer_key = 'your_consumer_key'
consumer_secret = 'your_consumer_secret'
access_token = 'your_access_token'
access_token_secret = 'your_access_token_secret'

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth)

# Step 3: Use Tweepy to scrape tweets about tech companies
public_tweets = api.search('Tech Companies')

# Step 4: Preprocess the tweets
def preprocess_tweet(text):
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\@\w+|\#','', text)
    text = text.lower()
    return text

# Step 5: Use TextBlob to perform sentiment analysis on the tweets
tweets = []
for tweet in public_tweets:
    text = preprocess_tweet(tweet.text)
    tweets.append((text, TextBlob(text).sentiment.polarity))

# Step 6: Prepare the data for the LSTM model
tokenizer = Tokenizer(num_words=10000)
tokenizer.fit_on_texts([tweet[0] for tweet in tweets])
sequences = tokenizer.texts_to_sequences([tweet[0] for tweet in tweets])
data = pad_sequences(sequences, maxlen=100)
labels = [tweet[1] for tweet in tweets]

X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2)

# Step 7: Define, compile, and train the LSTM model
model = Sequential()
model.add(Embedding(10000, 128))
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer=Adam(), metrics=['accuracy'])
model.fit(X_train, y_train, batch_size=32, epochs=3, validation_data=(X_test, y_test))

# Step 8: Evaluate the model
score, acc = model.evaluate(X_test, y_test, batch_size=32)

# Step 9: Make predictions
predictions = model.predict(X_test)