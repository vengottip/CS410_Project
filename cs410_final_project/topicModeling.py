# -*- coding: utf-8 -*-

import pandas as pd

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer

import gensim.corpora as corpora
from gensim.models import LdaModel, TfidfModel

#nltk.download('punkt')
nltk.download('stopwords')

data_path ='final_dataset.csv'
retrival_output = pd.read_csv(data_path)

text = retrival_output["text"].astype(str).str.replace(r'#\w+', '', regex=True)

# RegEx for words with length > 3
tokenizer = RegexpTokenizer(r'\b[a-zA-Z]{3,}\b')
tokenized_text = [tokenizer.tokenize(s) for s in text]

# remove stop words
stop_words = stopwords.words('english')
tokenized_des = [[w for w in s if w not in stop_words] for s in tokenized_text]


# create corpus
dct_des = corpora.Dictionary(tokenized_des)
corpus_des = [dct_des.doc2bow(line) for line in tokenized_des]


tfidf_model = TfidfModel(corpus_des)  # fit tf-idf model

low_value_words = []
for bow in corpus_des:
    low_value_words += [id for id, value in tfidf_model[bow] if value < 0.1]

#[dct_des.get(low_value_words[i]) for i in range(10)]

# filter words from the dictionary and the corpus
dct_des.filter_tokens(bad_ids=low_value_words)
corpus_des = [dct_des.doc2bow(line) for line in tokenized_des]

# topics_num = 1

def format_lda_topics(topics_num):
    lda = LdaModel(corpus_des, num_topics=topics_num, id2word=dct_des, passes=10, iterations=500)
    # Extract topics
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
