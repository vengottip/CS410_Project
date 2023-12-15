

# CS410 Project: TweetInsight Explorer for Technology

## Overview

Our system, the TechSentiment Insight Analyzer, developed for the CS410 final project, empowers users to gain insights into sentiments in the technology industry and leading tech companies. Users input specific queries, and the system displays relevant tweets, identifies and presents the topics expressed in the relevant tweets and frequent terms used in the topics, offering an overview of the public discourse related to the query. Our system also performs sentiment analysis to show distribution of positive, negative, and neutral sentiments in the relevant tweets.


## Dependencies 
Required packages are listed in requirements.txt



```
code here
```



## Dataset Creation:

To construct our dataset of tweets, we collected the most recent posts using hashtags associated with major tech companies. 
We utilized the ntscraper library for scraping purposes. Following that, we conducted data cleaning to remove URLs, emojis, symbols, and mentions of Twitter users to preserve user privacy. In total, our dataset consists of 21,600 tweets.



## Text Retrieval

We use PyTerrier which is a comprehensive and scalable toolkit for information retrieval. We index the content (text) and document number for faster retrieval later. After comparing different weighting models such as BM25 and DirichletLM, we choose BM25 for its effectiveness and robustness. Then we obtain the result from queries and list the most relevant contents to the UI.

## Text Mining and Analysis 

### Sentiment Analysis


### Topic Modeling

We use Latent Dirichlet Allocation (LDA) from the Gensim library for topic modeling. LDA is a probabilistic generative model that helps uncover hidden structures within a collection of documents, it assumes that documents are mixtures of topics, and each topic is a mixture of words.  We preprocess the data by removing hashtag words, stopwords, filtering out words with less than 3 characters, and tokenizing the text. 
The LDA model outputs a set of topics, each represented as a distribution of words.

## UI



## File list: 

xxx.py: 

