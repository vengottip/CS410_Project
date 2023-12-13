import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM
from sklearn.model_selection import train_test_split
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import auc
from sklearn.metrics import roc_auc_score, roc_curve
#from keras.utils.np_utils import to_categorical

nltk.download('vader_lexicon')

# Load data
data = pd.read_csv('C:\\Users\\gvkri\\OneDrive\\Documents\\Tech\\UIUC\\cs 410 Text information systems\\projects\\Team_Project\\CS410_Project\\tweets.csv')

# Preprocessing
max_fatures = 2000
tokenizer = Tokenizer(num_words=max_fatures, split=' ')
tokenizer.fit_on_texts(data['text'].values)
X = tokenizer.texts_to_sequences(data['text'].values)
X = pad_sequences(X)

# Build the model
embed_dim = 128
lstm_out = 196
model = Sequential()
model.add(Embedding(max_fatures, embed_dim,input_length = X.shape[1]))
model.add(LSTM(lstm_out, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(32,activation='softmax'))
model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])

# Initialize the SentimentIntensityAnalyzer
sia = SentimentIntensityAnalyzer()

# Apply the analyzer to the text data
data['sentiment_scores'] = data['text'].apply(lambda text: sia.polarity_scores(text))

# Calculate the sentiment based on the scores
data['sentiment'] = data['sentiment_scores'].apply(lambda score_dict: score_dict['compound'])

print(data)
print(data['sentiment'])
print(pd.get_dummies(data['sentiment']))

# Split the data
Y = pd.get_dummies(data['sentiment']).values
print(Y)
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.33, random_state = 42)
print(X_train.shape,Y_train.shape)
print(X_test.shape,Y_test.shape)

# Train the model
batch_size = 32
history = model.fit(X_train, Y_train, epochs = 7, batch_size=batch_size, validation_split=0.2, verbose = 2)


# Evaluate the model
""" validation_size = 6
X_validate = X_test[-validation_size:]
Y_validate = Y_test[-validation_size:]
X_test = X_test[:-validation_size]
Y_test = Y_test[:-validation_size]

print(X_validate.shape,Y_validate.shape)
print(X_test.shape,Y_test.shape)
 """

score,acc = model.evaluate(X_test, Y_test, verbose = 2, batch_size = batch_size)
print("score: %.2f" % (score))
print("acc: %.2f" % (acc))

""" print("Test score:", score[0])
print("Test accuracy:", score[1]) """

# Plot the accuracy and loss
plt.plot(history.history["accuracy"])
plt.plot(history.history["val_accuracy"])
plt.title("Model Accuracy")
plt.ylabel("Accuracy")
plt.xlabel("Epoch")
plt.legend(["Train", "Test"], loc="upper left")
plt.show()

plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.title("Model Loss")
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.legend(["Train", "Test"], loc="upper right")
plt.show()

# Predict the sentiment
y_pred = model.predict(X_test)
y_pred = np.round(y_pred).astype(int).reshape(y_pred.size)

Y_test = np.ravel(Y_test)
y_pred = np.round(y_pred).astype(int).reshape(len(Y_test))

# Print the classification report
print("Classification Report")
print(classification_report(Y_test, y_pred, target_names=["Negative", "Positive"]))
print("Accuracy:", accuracy_score(Y_test, y_pred))
print("Precision:", precision_score(Y_test, y_pred))
print("Recall:", recall_score(Y_test, y_pred))

# Plot the confusion matrix
cm = confusion_matrix(Y_test, y_pred)
cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
sns.set(font_scale=1.5)
sns.heatmap(
    cm,
    annot=True,
    annot_kws={"size": 10},
    cmap=plt.cm.Greens,
    linewidths=0.2,
    fmt=".2f",
    xticklabels=["Negative", "Positive"],
    yticklabels=["Negative", "Positive"],
)
plt.xlabel("Predicted label")
plt.ylabel("True label")
plt.show()

# Plot the ROC curve
fpr, tpr, thresholds = roc_curve(Y_test, y_pred)
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, color="darkorange", lw=2, label="ROC curve (area = %0.2f)" % roc_auc)
plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend(loc="lower right")
plt.show()

# Plot the precision-recall curve
""" average_precision = average_precision_score(y_test, y_pred)
disp = plot_precision_recall_curve(model, X_test, y_test)
disp.ax_.set_title("2-class Precision-Recall curve: AP={0:0.2f}".format(average_precision)) """
plt.show()

# Plot the confusion matrix
#plot_confusion_matrix(model, X_test, y_test, cmap=plt.cm.Blues)
plt.show()
y_true = ["True label"]  # Replace with your actual labels
y_pred = ["Predicted label"]  # Replace with your predicted labels
# Compute confusion matrix
cm = confusion_matrix(y_true, y_pred)

# Plotting confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# Plot the ROC curve
#plot_roc_curve(model, X_test, y_test)
#plt.show()

# Plot the precision-recall curve
""" precision_recall_curve(model, X_test, y_test)
plt.show() """
