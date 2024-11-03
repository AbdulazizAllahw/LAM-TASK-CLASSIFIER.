import pandas as pd
import json

# Load the merged data from the JSON file
merged_file_path = 'data/data_processing/ALL_DATA_MERGED.json'

with open(merged_file_path, 'r') as outfile:
    merged_data = json.load(outfile)

# Convert the merged data to a pandas DataFrame
df = pd.DataFrame(merged_data)

# Display basic information about the DataFrame
print("DataFrame Info:")
print(df.info())

# Display the first few rows of the DataFrame
print("\nFirst few rows of the DataFrame:")
print(df.head())

# Example analyses
# 1. Describe the data to get basic statistics (for numerical columns)
print("\nStatistical Summary:")
print(df.describe())

# 2. Check for missing values
print("\nMissing Values:")
print(df.isnull().sum())

# 3. Group by a specific column (example: 'category' if it exists) and count occurrences
if 'category' in df.columns:
    print("\nCounts by Category:")
    print(df['task'].value_counts())

# 4. You can also filter the data based on a condition (example: 'status' if it exists)
if 'status' in df.columns:
    filtered_data = df[df['status'] == 'active']
    print("\nFiltered Data (Active Status):")
    print(filtered_data)
