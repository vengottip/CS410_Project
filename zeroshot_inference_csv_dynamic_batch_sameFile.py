import sys
import os
import pandas as pd
from transformers import pipeline

# Load sentiment analysis model
distilled_student_sentiment_classifier = pipeline(
    model="lxyuan/distilbert-base-multilingual-cased-sentiments-student", 
    return_all_scores=True
)

def process_sentiment_analysis(input_csv_path):
    # Specify batch size
    batch_size = 1

    # Create a CSV file iterator to read the file in chunks
    csv_iterator = pd.read_csv(input_csv_path, chunksize=batch_size)

    # Initialize an empty DataFrame to store summary results
    summary_df = pd.DataFrame()

    for i, df_chunk in enumerate(csv_iterator):
        try:

            # Perform sentiment analysis and store results in a new column 'sentiment_scores'
            df_chunk['sentiment_scores'] = df_chunk['text'].apply(lambda text: distilled_student_sentiment_classifier(text))

            # Extract maximum score and label for each row
            df_chunk['extracted_Sentiment_score'] = df_chunk['sentiment_scores'].apply(lambda scores: max(scores[0], key=lambda x: x['score'])['score'])
            df_chunk['extracted_Sentiment_label'] = df_chunk['sentiment_scores'].apply(lambda scores: max(scores[0], key=lambda x: x['score'])['label'])

            # Generate output file paths based on input file name
            base_path, file_extension = os.path.splitext(input_csv_path)
            output_csv_path = f"{base_path}_out_batch.csv"  # Unique output path for each batch

            # Write the DataFrame with sentiment scores back to a new CSV file
            df_chunk.to_csv(output_csv_path, mode='a', index=False, header=not os.path.exists(output_csv_path))

            # Aggregate data by 'extracted_Sentiment_label' and calculate the percentage for each batch
            summary_batch_df = df_chunk['extracted_Sentiment_label'].value_counts(normalize=True).mul(100).reset_index()
            summary_batch_df.columns = ['label', 'percentage']

            # Append the summary for the current batch to the overall summary DataFrame
            summary_df = pd.concat([summary_df, summary_batch_df], ignore_index=True)

            # Print progress
            print(f"Processed batch {i + 1}")
        
        except Exception as e:
            # If an error occurs, catch it, record the 'text' column in 'error.csv', and continue to the next record
            print(f"Error in batch {i + 1}: {str(e)}")
            
            # Record the 'text' column of the problematic batch in 'error.csv'
            error_df = df_chunk[['text']].copy()
            error_df['error_message'] = str(e)
            error_csv_path = f"{base_path}_error_batch.csv"
            error_df.to_csv(error_csv_path, mode='a', index=False, header=not os.path.exists(error_csv_path))

            # Continue to the next batch
            continue

    # Aggregate data by 'extracted_Sentiment_label' and calculate the percentage for the entire dataset
    summary_df = summary_df.groupby('label')['percentage'].sum().reset_index()

    # Write the summary DataFrame to a new CSV file
    summary_csv_path = f"{base_path}_summary.csv"
    summary_df.to_csv(summary_csv_path, index=False)

if __name__ == "__main__":
    # Check if the correct number of command-line arguments is provided
    if len(sys.argv) != 2:
        print("Usage: python script.py <input_csv_path>")
        sys.exit(1)

    # Get input CSV file path from command-line argument
    input_csv_path = sys.argv[1]

    # Call the function to process sentiment analysis and generate output files
    process_sentiment_analysis(input_csv_path)