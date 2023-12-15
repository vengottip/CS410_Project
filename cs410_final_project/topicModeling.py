# -*- coding: utf-8 -*-

import pandas as pd

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer

import gensim.corpora as corpora
from gensim.models import LdaModel, TfidfModel

nltk.download('stopwords')

data_path ='IF_output.csv'
retrival_output = pd.read_csv(data_path)

text = retrival_output["text"].astype(str).str.replace(r'#\w+', '', regex=True)

# Tokenize the text using a regular expression to retain words with a length of at least three characters
tokenizer = RegexpTokenizer(r'\b[a-zA-Z]{3,}\b')
tokenized_text = [tokenizer.tokenize(s) for s in text]

# Remove stopwords from the tokenized text
stop_words = stopwords.words('english')
tokenized_all = [list(filter(lambda w: w not in stop_words, s)) for s in tokenized_text]

# Create a Gensim dictionary and a bag-of-words representation (corpus) from the tokenized text
document_all = corpora.Dictionary(tokenized_all)
corpus = [document_all.doc2bow(t) for t in tokenized_all]

# Create a TF-IDF model based on the corpus
tfidf_model = TfidfModel(corpus)

# Identify low-value words (words with low TF-IDF scores) based on a threshold (0.1)
words_lowValue = [id for bow in corpus for id, value in tfidf_model[bow] if value < 0.1]

# Filter out low-value words from the dictionary and the corpus
document_all.filter_tokens(bad_ids=words_lowValue)

corpus = []
for line in tokenized_all:
    bow_representation = document_all.doc2bow(line)
    corpus.append(bow_representation)
def format_lda_topics(topics_num):
    lda = LdaModel(corpus, num_topics=topics_num, id2word=document_all, passes=10, iterations=100)

    topics = lda.show_topics()

    # Convert topics to a list of dictionaries
    formatted_topics = []
    for topic in topics:
        topic_id, topic_terms = topic
        terms = [(term.split('*')[1].strip(), float(term.split('*')[0])) for term in topic_terms.split(' + ')]
        formatted_topics.append({'topic_id': topic_id, 'terms': terms})

    return formatted_topics

formatted_topics = format_lda_topics(3)
print(formatted_topics)
