#!/usr/bin/env python3
"""
Script to download images from SEED dataset to local folders
"""

import os
import requests
from datasets import load_dataset
from tqdm import tqdm
from PIL import Image
import io

def download_image(url, save_path):
    """Download an image from URL and save it locally"""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        # Open and save the image
        img = Image.open(io.BytesIO(response.content))
        img.save(save_path)
        return True
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        return False

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
        # Try to load the dataset with streaming to avoid memory issues
        dataset = load_dataset("AILab-CVC/SEED-DATA-Edit-Part2-3", streaming=True)
        
        print("Dataset loaded successfully!")
        print(f"Available splits: {list(dataset.keys())}")
        
        # Process the first few examples to test
        max_examples = 10  # Limit for testing
        
        for split_name, split_data in dataset.items():
            print(f"\nProcessing split: {split_name}")
            
            for i, example in enumerate(split_data):
                if i >= max_examples:
                    break
                    
                print(f"Processing example {i+1}/{max_examples}")
                
                # Print the structure of the example
                print(f"Example keys: {list(example.keys())}")
                
                # Try to find image URLs in the example
                for key, value in example.items():
                    if isinstance(value, str) and ('http' in value or '.jpg' in value or '.png' in value):
                        print(f"Found potential image URL in {key}: {value[:100]}...")
                        
                        # Determine if this is original or edited
                        if 'source' in key.lower() or 'original' in key.lower():
                            save_dir = original_dir
                            prefix = "original"
                        elif 'edit' in key.lower() or 'target' in key.lower():
                            save_dir = edited_dir
                            prefix = "edited"
                        else:
                            save_dir = edited_dir
                            prefix = "unknown"
                        
                        # Download the image
                        filename = f"{prefix}_{split_name}_{i}_{key}.jpg"
                        save_path = os.path.join(save_dir, filename)
                        
                        if download_image(value, save_path):
                            print(f"Downloaded: {filename}")
                        else:
                            print(f"Failed to download: {value}")
                
                print("-" * 50)
        
        print(f"\nDownload completed! Check the {base_dir} folder for images.")
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Trying alternative approach...")
        
        # Alternative: Try to load without streaming
        try:
            dataset = load_dataset("AILab-CVC/SEED-DATA-Edit-Part2-3", split="train[:10]")
            print("Loaded dataset without streaming")
            print(f"Dataset structure: {dataset}")
            print(f"First example: {dataset[0]}")
        except Exception as e2:
            print(f"Alternative approach also failed: {e2}")

if __name__ == "__main__":
    main() 