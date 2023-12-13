import pandas as pd

# FILEPATH: Untitled-1

# Read data from CSV file into a data frame
df = pd.read_csv('C:\\Users\\gvkri\\OneDrive\\Documents\\Tech\\UIUC\\cs 410 Text information systems\\projects\\Team_Project\\CS410_Project\\zoe_dataset\\final_dataset_mod2_1_20.csv')

# Print the data
print(df)

# Print the length of the 'text' column
df['text_length'] = df['text'].apply(len)
print(df['text_length'])


