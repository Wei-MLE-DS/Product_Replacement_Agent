import json
import pandas as pd
import openai
import os

# Function to extract the first n records from a JSONL file and save as CSV
def extract_sample_from_jsonl(jsonl_path, csv_path, n=100):
    """
    Extract the first n records from a JSONL file and save as a CSV.
    """
    records = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            records.append(json.loads(line))
    df = pd.DataFrame(records)
    df.to_csv(csv_path, index=False)
    print(f"Extracted {len(df)} records to {csv_path}")


if __name__ == "__main__":
    # Change 'your_pet_review.jsonl' to your actual JSONL file name
    extract_sample_from_jsonl('Pet_Supplies.jsonl', 'amazon_review_pets.csv', n=100) 
    extract_sample_from_jsonl('meta_Pet_Supplies.jsonl', 'meta_amazon_review_pets.csv', n=100) 