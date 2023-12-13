import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM
from sklearn.model_selection import train_test_split
#from keras.utils.np_utils import to_categorical
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score


# Download vader_lexicon
nltk.download('vader_lexicon')

# Load data
data = pd.read_csv('C:\\Users\\gvkri\\OneDrive\\Documents\\Tech\\UIUC\\cs 410 Text information systems\\projects\\Team_Project\\CS410_Project\\zoe_dataset\\final_dataset_mod2_1_20.csv')

print(data['text'])
# Preprocessing
max_fatures = 2000
tokenizer = Tokenizer(num_words=max_fatures, split=' ')
tokenizer.fit_on_texts(data['text'].values)
X = tokenizer.texts_to_sequences(data['text'].values)
X = pad_sequences(X, maxlen = 1000)

# Initialize the Vader SentimentIntensityAnalyzer
sia = SentimentIntensityAnalyzer()

# Apply the analyzer to the text data
data['vader_scores'] = data['text'].apply(lambda text: sia.polarity_scores(text))
data['vader_sentiment'] = data['vader_scores'].apply(lambda score_dict: score_dict['compound'])

data.to_csv('C:\\Users\\gvkri\\OneDrive\\Documents\\Tech\\UIUC\\cs 410 Text information systems\\projects\\Team_Project\\CS410_Project\\zoe_dataset\\final_dataset_mod_output.csv', index=False)

data.head(10)
# Build the LSTM model
embed_dim = 128
lstm_out = 196
model = Sequential()
model.add(Embedding(max_fatures, embed_dim,input_length = X.shape[1]))
model.add(LSTM(lstm_out, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(3,activation='softmax'))
model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])

# Split the data
Y = pd.get_dummies(data['sentiment']).values
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.33, random_state = 42)

print(X_train.shape,Y_train.shape)
print(X_test.shape,Y_test.shape)
print(X.shape,Y.shape)

# Train the LSTM model
batch_size = 32
model.fit(X_train, Y_train, epochs = 7, batch_size=batch_size, verbose = 2)

# Predict the sentiment of the testing data using the LSTM model
lstm_predictions = model.predict(X_test)

# Assuming lstm_predictions is your model's predictions
predicted_labels = np.argmax(lstm_predictions, axis=1)

# Convert the predicted labels back to 'positive', 'negative', 'neutral'
label_mapping = {0: 'positive', 1: 'negative', 2: 'neutral'}
predicted_sentiments = [label_mapping[label] for label in predicted_labels]

# Now predicted_sentiments contains the final predicted labels for each sample
print(predicted_sentiments)

# Create a DataFrame from the test data and the predicted sentiments
test_data_df = pd.DataFrame(X_test, columns=['text'])
test_data_df['predicted_sentiment'] = predicted_sentiments

# Print the text and the corresponding predicted sentiment
for index, row in test_data_df.iterrows():
    print(f"Text: {row['text']}, Predicted Sentiment: {row['predicted_sentiment']}")


# Print the confusion matrix
print(confusion_matrix(Y_test.argmax(axis=1), predicted_labels))

# Print the precision and recall, among other metrics
print(classification_report(Y_test.argmax(axis=1), predicted_labels, digits=3))

# Print the overall accuracy
print("Overall accuracy:", accuracy_score(Y_test.argmax(axis=1), predicted_labels))

# Print the precision and recall, among other metrics
print("Precision:", precision_score(Y_test.argmax(axis=1), predicted_labels, average='macro'))
print("Recall:", recall_score(Y_test.argmax(axis=1), predicted_labels, average='macro'))

# Print the confusion matrix
cm = confusion_matrix(Y_test.argmax(axis=1), predicted_labels)
print(cm)

# Plot the confusion matrix
plt.matshow(cm)
plt.title('Confusion matrix')
plt.colorbar()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

# Print the classification report
print(classification_report(Y_test.argmax(axis=1), predicted_labels, target_names=["Negative", "Neutral", "Positive"]))
print("Accuracy:", accuracy_score(Y_test.argmax(axis=1), predicted_labels))
print("Precision:", precision_score(Y_test.argmax(axis=1), predicted_labels))
print("Recall:", recall_score(Y_test.argmax(axis=1), predicted_labels))

# Print the confusion matrix
cm = confusion_matrix(Y_test.argmax(axis=1), predicted_labels)
print(cm)

# Plot the confusion matrix
cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
plt.matshow(cm)
plt.title('Confusion matrix')
plt.colorbar()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

# Predict the sentiment of the testing data using the Vader model
vader_predictions = data['vader_sentiment']




# Combine the predictions from both models
final_predictions = 0.5 * (lstm_predictions + data['vader_sentiment'])

print(final_predictions)