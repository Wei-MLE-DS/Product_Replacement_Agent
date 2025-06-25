#!/usr/bin/env python3
"""
Script to download images from SEED dataset to local folders
Handles file paths instead of URLs
"""

import os
import json
from datasets import load_dataset
from tqdm import tqdm
import pandas as pd

def save_metadata_to_csv(dataset, output_file):
    """Save dataset metadata to CSV file"""
    data_list = []
    
    for i, example in enumerate(dataset):
        row = {
            'example_id': i,
            'source_image': example.get('source_image', ''),
            'instruction_1': example.get('instruction_1', ''),
            'instruction_2': example.get('instruction_2', ''),
            'instruction_3': example.get('instruction_3', ''),
            'instruction_4': example.get('instruction_4', ''),
            'instruction_5': example.get('instruction_5', ''),
            'instruction_6': example.get('instruction_6', ''),
            'edit_image_1': example.get('edit_image_1', ''),
            'edit_image_2': example.get('edit_image_2', ''),
            'edit_image_3': example.get('edit_image_3', ''),
            'edit_image_4': example.get('edit_image_4', ''),
            'edit_image_5': example.get('edit_image_5', ''),
            'edit_image_6': example.get('edit_image_6', ''),
        }
        data_list.append(row)
    
    df = pd.DataFrame(data_list)
    df.to_csv(output_file, index=False)
    print(f"Metadata saved to {output_file}")
    return df

def main():
    # Create directories if they don't exist
    base_dir = "photoshopped_images"
    original_dir = os.path.join(base_dir, "original")
    edited_dir = os.path.join(base_dir, "edited")
    metadata_dir = os.path.join(base_dir, "metadata")
    
    os.makedirs(original_dir, exist_ok=True)
    os.makedirs(edited_dir, exist_ok=True)
    os.makedirs(metadata_dir, exist_ok=True)
    
    print("Loading SEED dataset...")
    
    try:
        # Load a small subset for testing
        dataset = load_dataset("AILab-CVC/SEED-DATA-Edit-Part2-3", split="train[:100]")
        
        print(f"Dataset loaded successfully! Found {len(dataset)} examples")
        
        # Save metadata to CSV
        metadata_file = os.path.join(metadata_dir, "seed_dataset_metadata.csv")
        df = save_metadata_to_csv(dataset, metadata_file)
        
        # Print sample data
        print("\nSample data structure:")
        print(f"Columns: {list(dataset[0].keys())}")
        print(f"First example source_image: {dataset[0]['source_image']}")
        print(f"First example instruction_1: {dataset[0]['instruction_1']}")
        
        # Save instructions to a separate file
        instructions_file = os.path.join(metadata_dir, "instructions.txt")
        with open(instructions_file, 'w', encoding='utf-8') as f:
            for i, example in enumerate(dataset):
                f.write(f"Example {i}:\n")
                f.write(f"Source: {example.get('source_image', 'N/A')}\n")
                for j in range(1, 7):
                    instruction = example.get(f'instruction_{j}', '')
                    edit_image = example.get(f'edit_image_{j}', '')
                    if instruction:
                        f.write(f"Edit {j}: {instruction}\n")
                        f.write(f"Edited Image: {edit_image}\n")
                f.write("-" * 50 + "\n")
        
        print(f"Instructions saved to {instructions_file}")
        
        # Create a summary file
        summary_file = os.path.join(metadata_dir, "dataset_summary.txt")
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write(f"SEED Dataset Summary\n")
            f.write(f"===================\n")
            f.write(f"Total examples: {len(dataset)}\n")
            f.write(f"Columns: {list(dataset[0].keys())}\n\n")
            
            # Count non-empty instructions
            instruction_counts = {}
            for j in range(1, 7):
                count = sum(1 for ex in dataset if ex.get(f'instruction_{j}', '').strip())
                instruction_counts[f'instruction_{j}'] = count
            
            f.write("Instruction counts:\n")
            for key, count in instruction_counts.items():
                f.write(f"  {key}: {count} examples\n")
        
        print(f"Summary saved to {summary_file}")
        
        print(f"\nDataset analysis completed!")
        print(f"Check the {base_dir} folder for:")
        print(f"  - {metadata_dir}/seed_dataset_metadata.csv (metadata)")
        print(f"  - {metadata_dir}/instructions.txt (instructions)")
        print(f"  - {metadata_dir}/dataset_summary.txt (summary)")
        
        # Note about images
        print(f"\nNote: The dataset contains file paths, not URLs.")
        print(f"To access the actual images, you would need to:")
        print(f"1. Download the full dataset from Hugging Face")
        print(f"2. Extract the image files from the dataset")
        print(f"3. Use the file paths to locate the images")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main() 