import pyterrier as pt
if not pt.started():
    pt.init()

import pandas as pd
import pyterrier as pt

# Initialize PyTerrier
if not pt.started():
    pt.init()

def retrieve_tweets_with_query(query, nums):
    # Load the CSV file
    file_path = 'final_dataset.csv'
    df = pd.read_csv(file_path)
    # Add a 'docno' column to the dataframe
    df['docno'] = df.index + 1
    # Reorder the columns to make 'docno' the first column
    df = df[['docno'] + [col for col in df.columns if col != 'docno']]
    df['docno'] = df['docno'].astype(str)
    # Create an index
    indexer = pt.DFIndexer("./tweet_index", overwrite=True)
    index_ref = indexer.index(df['text'], df['docno'])

    # Load the index
    index = pt.IndexFactory.of(index_ref)

    # Retrieving with a query
    retriever = pt.BatchRetrieve(index, wmodel="BM25")

    # retriever = pt.BatchRetrieve(index, wmodel="DirichletLM")
    results = retriever.transform([query])

    # Get top 10 results
    top_10 = results.head(nums)

    # retrieve the full text for each docno in the top_10 results
    # We will add a new 'full_text' column to the top_10 DataFrame
    top_10_df = pd.DataFrame(top_10)

    # Assuming 'top_10' is a list or some iterable of search results
    # print(top_10)

    top_10_df = pd.DataFrame(top_10)
    top_10_df['text'] = top_10_df['docno'].apply(lambda x: df[df['docno'] == str(x)]['text'].iloc[0])
    top_10_df['extracted_Sentiment_label'] = top_10_df['docno'].apply(lambda x: df[df['docno'] == str(x)]['extracted_Sentiment_label'].iloc[0])
    
    # Now top_10_df contains the full text of each tweet alongside the original search result data
    # return top_10_df[['docno', 'full_text', 'rank', 'query']]
    # print(top_10_df[['docno', 'full_text', 'rank', 'query']])
    result_list = top_10_df[['docno', 'text', 'extracted_Sentiment_label' ,'rank', 'query']].astype(str).values.tolist()

    # Export the DataFrame to a CSV file
    output_csv_file = 'IF_output.csv'
    top_10_df.to_csv(output_csv_file, index=False)
    result = []
    for data_array in result_list:
        tmp = []
        tmp.append(data_array[1])

        # Join the tmp list into a single string
        result.append(', '.join(tmp))

    return result;

