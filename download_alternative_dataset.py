#!/usr/bin/env python3
"""
Script to download images from alternative datasets
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

def try_dataset(dataset_name, max_examples=5):
    """Try to load and analyze a dataset"""
    print(f"\n{'='*50}")
    print(f"Trying dataset: {dataset_name}")
    print(f"{'='*50}")
    
    try:
        # Try to load the dataset
        dataset = load_dataset(dataset_name, split="train")
        print(f"✅ Successfully loaded {dataset_name}")
        print(f"Dataset size: {len(dataset)} examples")
        print(f"Columns: {list(dataset[0].keys())}")
        
        # Show first example
        print(f"First example: {dataset[0]}")
        
        return dataset
        
    except Exception as e:
        print(f"❌ Failed to load {dataset_name}: {e}")
        return None

def main():
    # Create directories
    base_dir = "photoshopped_images"
    original_dir = os.path.join(base_dir, "original")
    edited_dir = os.path.join(base_dir, "edited")
    metadata_dir = os.path.join(base_dir, "metadata")
    
    os.makedirs(original_dir, exist_ok=True)
    os.makedirs(edited_dir, exist_ok=True)
    os.makedirs(metadata_dir, exist_ok=True)
    
    # List of potential datasets to try
    datasets_to_try = [
        "cifar10",  # Simple image classification dataset
        "mnist",    # Handwritten digits
        "fashion_mnist",  # Fashion items
        "imagenet-1k",  # Large image dataset (if available)
    ]
    
    successful_datasets = []
    
    for dataset_name in datasets_to_try:
        dataset = try_dataset(dataset_name)
        if dataset:
            successful_datasets.append((dataset_name, dataset))
    
    if not successful_datasets:
        print("\n❌ No datasets could be loaded successfully.")
        print("Let's try a different approach...")
        
        # Try to create a simple test dataset
        print("\nCreating a simple test dataset with sample images...")
        create_test_dataset(base_dir)
        return
    
    # Use the first successful dataset
    dataset_name, dataset = successful_datasets[0]
    print(f"\n🎉 Using dataset: {dataset_name}")
    
    # Download a few sample images
    max_downloads = 10
    downloaded_count = 0
    
    for i, example in enumerate(dataset):
        if downloaded_count >= max_downloads:
            break
            
        # Try to find image data in the example
        if 'image' in example:
            # If it's a PIL Image object
            img = example['image']
            filename = f"{dataset_name}_sample_{i}.png"
            save_path = os.path.join(original_dir, filename)
            
            try:
                img.save(save_path)
                print(f"✅ Saved: {filename}")
                downloaded_count += 1
            except Exception as e:
                print(f"❌ Failed to save {filename}: {e}")
        
        elif 'img' in example:
            # Alternative image field
            img = example['img']
            filename = f"{dataset_name}_sample_{i}.png"
            save_path = os.path.join(original_dir, filename)
            
            try:
                img.save(save_path)
                print(f"✅ Saved: {filename}")
                downloaded_count += 1
            except Exception as e:
                print(f"❌ Failed to save {filename}: {e}")
    
    print(f"\n📁 Downloaded {downloaded_count} images to {original_dir}")
    print(f"📊 Dataset analysis completed!")

def create_test_dataset(base_dir):
    """Create a simple test dataset with sample images"""
    import numpy as np
    
    # Create some simple test images
    for i in range(5):
        # Create a simple colored image
        img_array = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        img = Image.fromarray(img_array)
        
        filename = f"test_image_{i}.png"
        save_path = os.path.join(base_dir, "original", filename)
        img.save(save_path)
        print(f"✅ Created test image: {filename}")
    
    print(f"\n📁 Created 5 test images in {base_dir}/original/")

if __name__ == "__main__":
    main() 