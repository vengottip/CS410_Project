# -*- coding: utf-8 -*-

import pandas as pd
# TF-IDF Feature Generation
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import RegexpTokenizer
from wordcloud import WordCloud
from wordcloud import STOPWORDS
import matplotlib.pyplot as plt
import os

data_path ='final_dataset.csv'
text = pd.read_csv(data_path)

subset_top = text["text"].astype(str).str.replace(r'#\w+', '', regex=True)
subset_top = subset_top.head(100)

# Initialize regex tokenizer
tokenizer = RegexpTokenizer(r'\w+')

# # Vectorize document using TF-IDF
tf_idf_vect = TfidfVectorizer(lowercase=True,
                        stop_words='english',
                        ngram_range = (1,2),
                        max_df=0.5,
                        min_df=0.01,
                        tokenizer = tokenizer.tokenize)

# Fit and Transfrom Text Data
model_data = tf_idf_vect.fit_transform(subset_top)

# Check Shape of Count Vector
model_data.shape


"""# **LDA model**"""

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem.snowball import SnowballStemmer

import gensim.corpora as corpora
from gensim.models import LdaModel, TfidfModel

nltk.download('punkt')
nltk.download('stopwords')


# RegEx for words with length > 3
tokenizer = RegexpTokenizer(r'\b[a-zA-Z]{3,}\b')
tokenized_des = [tokenizer.tokenize(s) for s in subset_top]

# remove stop words
stop_words = stopwords.words('english')
tokenized_des = [[w for w in s if w not in stop_words] for s in tokenized_des]

'''
# stemming the words
stemmer = SnowballStemmer("english")
tokenized_des = [[stemmer.stem(w) for w in s] for s in tokenized_des]
'''


# create corpus
dct_des = corpora.Dictionary(tokenized_des)
corpus_des = [dct_des.doc2bow(line) for line in tokenized_des]


tfidf_model = TfidfModel(corpus_des)  # fit tf-idf model

low_value_words = []
for bow in corpus_des:
    low_value_words += [id for id, value in tfidf_model[bow] if value < 0.1]

[dct_des.get(low_value_words[i]) for i in range(10)]

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


# # Iterate over the topic_id values
# for i in range(topics_num):
#     # Create a WordCloud for the current topic
#     font_path = 'Country Market.ttf' 
#     tmp_cloud = WordCloud(width=800, height=600, background_color="white", font_path = font_path)
#     tmp_cloud.fit_words(dict(lda.show_topic(i, 50)))

#     # Create a figure and axes for the subplot
#     fig, ax = plt.subplots(figsize=(8, 6))

#     # Display the WordCloud
#     ax.imshow(tmp_cloud)
#     ax.axis('off')  # Turn off the axis

#     # Save the individual WordCloud image
#     output_directory = 'images'
#     os.makedirs(output_directory, exist_ok=True)
#     wordcloud_image_path = os.path.join(output_directory, f'wordcloud_topic_{i}.png')
#     plt.savefig(wordcloud_image_path, overwrite=True)
#     plt.close()  # Close the figure to release resources
