import json
import os

# Define the path to your raw_data directory
data_dir = 'data/raw_data/merge_temp'  # Adjust if needed

# Function to check the structure of a JSON file
def check_json_structure(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
        print(f"Structure of {file_path}:")
        print(f"Type: {type(data)}")
        print(f"Sample content: {data[:5] if isinstance(data, list) else list(data.items())[:5]}")  # Print first 5 elements for inspection
        print("\n")
        return data  # Return the data for merging

# Function to recursively find all JSON files
def find_json_files(directory):
    json_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.json'):
                json_files.append(os.path.join(root, file))
    return json_files

# Find all JSON files
json_files = find_json_files(data_dir)

# Load and check the structure of each JSON file, then merge if they are lists
merged_data = []

for file in json_files:
    data = check_json_structure(file)
    if isinstance(data, list):
        merged_data.extend(data)  # Merge the list data
    else:
        print(f"{file} does not contain a list. Skipping...")

# Save the merged data into a new JSON file
output_file = 'data/data_processing/ALL_DATA_MERGED.json'
with open(output_file, 'w', encoding='utf-8') as outfile:
    json.dump(merged_data, outfile, indent=2, ensure_ascii=False)

print(f"Files merged successfully into {output_file}")
print(f"Total number of items in merged data: {len(merged_data)}")
