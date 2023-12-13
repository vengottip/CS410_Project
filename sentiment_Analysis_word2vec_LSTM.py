import pandas as pd
from gensim.models import Word2Vec
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM
from sklearn.model_selection import train_test_split
#from keras.utils.np_utils import to_categorical
from keras.preprocessing.text import text_to_word_sequence

# Load data
data = pd.read_csv('C:\\Users\\gvkri\\OneDrive\\Documents\\Tech\\UIUC\\cs 410 Text information systems\\projects\\Team_Project\\CS410_Project\\zoe_dataset\\final_dataset_mod2_1_20.csv')

# Preprocessing
data['tokens'] = data['text'].apply(text_to_word_sequence)




print(data['tokens'])
# Train a Word2Vec model
w2v_model = Word2Vec(data['tokens'], vector_size=100, window=5, min_count=1, workers=4)

# Convert the text data into vectors
data['vectors'] = data['tokens'].apply(lambda tokens: [w2v_model.wv[token] for token in tokens])
X = pad_sequences(data['vectors'])
#X = data['vectors'].to_list()  # Convert to list of lists, not a Pandas Series

print(data)

print(data['sentiment'])


# Preprocessing
max_fatures = 2000
tokenizer = Tokenizer(num_words=max_fatures, split=' ')
tokenizer.fit_on_texts(data['text'].values)
X1 = tokenizer.texts_to_sequences(data['text'].values)
X1 = pad_sequences(X1)

print(X1)

# Build the LSTM model
embed_dim = 100
lstm_out = 2000

# Assuming you have already trained your Word2Vec model
vocab_size = len(w2v_model.wv)

print(vocab_size)

model = Sequential()
model.add(Embedding(input_dim=vocab_size, output_dim=embed_dim, input_length=X.shape[1], weights=[w2v_model.wv.vectors], trainable=False))
model.add(LSTM(lstm_out, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(3,activation='softmax'))
model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])

# Split the data
Y = pd.get_dummies(data['sentiment']).values
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.33, random_state = 42)

print(X_train.shape)
print(Y_train.shape)

# Train the LSTM model
batch_size = 32
model.fit(X_train, Y_train, epochs = 7, batch_size=batch_size, verbose = 2)

# Predict the sentiment of the testing data
predictions = model.predict(X_test)

print(predictions)