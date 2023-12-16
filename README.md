

# CS410 Project

App: TweetInsight Explorer for Technology

## 1. Overview

Our app, TweetInsight Explorer for Technology, developed for the CS410 final project, empowers users to gain insights into sentiments in the technology industry and leading tech companies. Users input specific queries, and the system displays relevant tweets, identifies and presents the topics expressed in the relevant tweets and frequent terms used in the topics, offering an overview of the public discourse related to the query. Our system also performs sentiment analysis to show distribution of positive, negative, and neutral sentiments in the relevant tweets.


## 2. Team member: 

Venkata Gottiparthi,
Qi Zhou,
Zhao Li,
Hannah Ke,
Yixin Xu


## 3. Video:
https://mediaspace.illinois.edu/media/t/1_rmfrdged

## 4. How to use the App
* Download the source folder and requirements.txt
* Install the dependencies listed in requirements.txt file. To do so, open a terminal or command prompt, and run the following command:
  
```
pip3 install -r requirements.txt
```

* Start the Flask Server:
    * Run the app.py within the source folder to start the Flask server. Run the following command:
      
 ```
python3 app.py
```
      
* Access the Web Interface:
    * Wait until the server finishes loading and the URL (http://127.0.0.1:5000/) is shown in terminal
    * Open a web browser and go to http://127.0.0.1:5000/.
    * You will see the "TweetInsight Explorer" interface with a search form.
* Perform a Search:
    * Enter a search query in the input field (e.g., “IBM AI innovation”).
    * Optionally, specify the number of results and topics.
    * Click the "Search" button.
* View Results:
    * The relevent tweets, a bar chart for topic modeling and a pie chart for sentiment distribution pie chart will be displayed.


## 5. Dependencies 
Required packages are listed in requirements.txt


## 6. Dataset Creation:

To construct our dataset of tweets, we collected the most recent posts using hashtags associated with major tech companies. 
We utilized the ntscraper library for scraping purposes. Following that, we conducted data cleaning to remove URLs, emojis, symbols, and mentions of Twitter users to preserve user privacy. In total, our dataset consists of 21,600 tweets. 

After the dataset (final_dataset.csv) was created, sentiment analysis was done on the entire dataset as described in 8.1 below to generate the sentiment label of the data, and the same was added as a column in the dataset for further analysis.


## 7. Text Retrieval

We use PyTerrier which is a comprehensive and scalable toolkit for information retrieval. We index the content (text) and document number for faster retrieval later. After comparing different weighting models such as BM25 and DirichletLM, we choose BM25 for its effectiveness and robustness. Then we obtain the result from queries and list the most relevant contents to the UI.

## 8. Text Mining and Analysis 

### 8.1. Sentiment Analysis
The sentiment analysis of the text data scraped from twitter was done using a pre-trained sentiment analysis model.( "lxyuan/distilbert-base-multilingual-cased-sentiments-student,") based on the DistilBERT architecture. DistilBERT is a smaller and faster version of BERT while retaining much of its performance.
The pipeline function from transformers is used to load this pre-trained model for sentiment analysis.  Pandas library is used  to handle File and dataframe operations.
The model calculates three sentiment scores for each record of text which are stored in a column of the dataframe. One score is for the positive sentiment, one for negative and one for neutral. Programmatically the highest score and the corresponding sentiment label are extracted and stored in two other columns in the dataframe.

### 8.2. Topic Modeling

We use Latent Dirichlet Allocation (LDA) from the Gensim library for topic modeling. LDA is a probabilistic generative model that helps uncover hidden structures within a collection of documents, it assumes that documents are mixtures of topics, and each topic is a mixture of words.  We preprocess the data by removing hashtag words, stopwords, filtering out words with less than 3 characters, and tokenizing the text. 
The LDA model outputs a set of topics, each represented as a distribution of words.

## 9. UI

- Frontend (HTML and JavaScript):
Search form structure and dynamic UI updates.
- Backend (Flask and Python Scripts):
Flask routes for home and search.
Python scripts(eg. tweet retrieval, topic formatting).
- Visualizations (Plotly):
Pie chart for overall data.
Bar chart for topic terms probabilities.
- Styling (CSS):
Enhancements and specific adjustments.

