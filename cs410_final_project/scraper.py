import pandas as pd
from ntscraper import Nitter
import demoji
import glob

scraper = Nitter()

def get_tweets(name,modes):
    tweets = scraper.get_tweets(name,mode = modes, language = "en", number = 5000 )
    final_tweets = []
    for tweet in tweets['tweets']:
        data = [tweet['link'], tweet['text'],tweet['date'],tweet['stats']['likes'],tweet['stats']['quotes'],tweet['stats']['comments'],tweet['stats']['retweets']]
        final_tweets.append(data)
    data = pd.DataFrame(final_tweets,columns= ['link','text','date','likes','quotes','comments','retweets'])

    return data

def clean_tweets(data):

    # remove http link
    data['text'] = data['text'].str.replace(
        r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', ' ', regex=True)

    # remove text that starts with "@" followed by a username.
    pattern = r'@\w+'
    data['text'] = data['text'].str.replace(pattern, '', regex=True)

    # remove emoji and symbols
    demoji.download_codes()
    def clean_emojis(text):
        return demoji.replace(text, '')

    data['text'] = data['text'].apply(clean_emojis)

    final_data = data.drop_duplicates(subset=['text'])

    return final_data

def combine_df():

    csv_files = glob.glob('./*.csv')

    print(csv_files)

    dfs = [pd.read_csv(csv_file) for csv_file in csv_files]

    combined_df = pd.concat(dfs, ignore_index=True)

    cleaned_data = clean_tweets(combined_df)

    cleaned_data.to_csv("final_dataset.csv",index=False)


if __name__ == "__main__":


    hashtag = ["Google","Amazon","Apple", "Meta", "Microsoft", "Intel", "Tesla","Uber", "Nvidia","Openai","Salesforce"]

    for i in hashtag:
        data = get_tweets(hashtag, 'hashtag')
        data.to_csv(f'{hashtag}_tweets.csv', index=False)
        

    combine_df()





