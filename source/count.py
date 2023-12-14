import pandas as pd

def get_sentiment_label_counts(data_path):

    # Read the CSV file into a Pandas DataFrame
    retrieval_output = pd.read_csv(data_path)

    # Check if the "extracted_Sentiment_label" column exists in the DataFrame
    if 'extracted_Sentiment_label' in retrieval_output.columns:
        # Count the occurrences of each unique value in the "extracted_Sentiment_label" column
        sentiment_label_counts = retrieval_output['extracted_Sentiment_label'].value_counts()


        # Display the counts
        # print("\nSentiment Label Counts:")
        result = {}
        for label, count in sentiment_label_counts.items():
            # print(f"{label}: {count}")
            result[label] = int(count)
        print(result)
        return result



