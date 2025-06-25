# Edited Images Dataset

This folder contains all edited/photoshopped images collected from various sources.

## 📊 Summary
- **Total images**: 1,080
- **Image format**: PNG
- **Image size**: 256x256 pixels (standardized)

## 📁 Sources

### 1. Main Photoshopped Dataset (1,000 images)
- **Source**: `photoshopped_dataset/photoshopped/`
- **Files**: `photoshopped_00000_*.png` to `photoshopped_00999_*.png`
- **Description**: High-quality synthetic photoshopped images with various effects
- **Effects applied**: brightness, contrast, color, saturation, blur, sharpen, edge_enhance, emboss, find_edges, smooth, noise, vintage, sepia, posterize, solarize, invert, mirror, rotate, scale

### 2. Photoshopped Images (40 images)
- **Source**: `photoshopped_images/edited/`
- **Files**: `photoshopped_blur_*.png`, `photoshopped_brightness_*.png`, etc.
- **Description**: Various photoshop effects applied to base images
- **Effects**: blur, brightness, color_boost, contrast, noise, saturation, sharpen, vintage

### 3. AI Generated Images (20 images)
- **Source**: `image_authenticity_dataset/ai_generated/`
- **Files**: `ai_generated_*.png`
- **Description**: Synthetic AI-generated images with various patterns and effects

### 4. Image Authenticity Photoshopped (20 images)
- **Source**: `image_authenticity_dataset/photoshopped/`
- **Files**: `photoshopped_*.png`
- **Description**: Photoshopped images from the authenticity dataset

## 🎯 Usage
This dataset is ready for training image authenticity detection models. The images represent various types of edited/manipulated content that can be used to train models to distinguish between real and edited images.

## 📋 File Naming Convention
- `photoshopped_XXXXX_effect1_effect2_effect3.png` - Main photoshopped dataset
- `photoshopped_effect_*.png` - Photoshopped images with specific effects
- `ai_generated_*.png` - AI-generated images
- `photoshopped_*.png` - Authenticity dataset photoshopped images

## 🔧 Effects Applied
- **Basic adjustments**: Brightness, contrast, color, saturation
- **Filters**: Blur, sharpen, edge enhance, emboss, find edges, smooth
- **Special effects**: Noise, vintage, sepia, posterize, solarize, invert
- **Transformations**: Mirror, rotate, scale
- **AI patterns**: Various synthetic patterns and textures

## 🐍 Python Scripts for Downloading Images

### Available Scripts

#### 1. `create_photoshopped_only.py` - Generate Synthetic Photoshopped Images
```bash
# Generate 1,000 photoshopped images from scratch
python create_photoshopped_only.py
```
**What it does:**
- Creates 200 diverse base images (rectangles, gradients, patterns, text, noise, complex shapes)
- Applies 1-4 random photoshop effects to each base image
- Generates 1,000 photoshopped images with various effects
- Output: `photoshopped_dataset/` folder

**Effects included:**
- Brightness, contrast, color, saturation adjustments
- Blur, sharpen, edge enhance, emboss, find edges, smooth filters
- Noise, vintage, sepia, posterize, solarize effects
- Invert, mirror, rotate, scale transformations

#### 2. `download_photoshopped_hf_final.py` - Download from Hugging Face Datasets
```bash
# List available datasets
python download_photoshopped_hf_final.py --list-datasets

# Download from CIFAR-10 and create photoshopped versions
python download_photoshopped_hf_final.py --dataset cifar10 --max-images 500

# Download from MNIST and create photoshopped versions
python download_photoshopped_hf_final.py --dataset mnist --max-images 300

# Download from Food101 and create photoshopped versions
python download_photoshopped_hf_final.py --dataset food101 --max-images 200
```
**What it does:**
- Downloads images from Hugging Face datasets
- Creates photoshopped versions with random effects
- Output: `hf_photoshopped/` folder with `original/` and `photoshopped/` subfolders

**Available datasets:**
- `cifar10` (50,000 samples)
- `mnist` (60,000 samples) 
- `food101` (75,750 samples)

#### 3. `create_large_dataset.py` - Generate Large Mixed Dataset
```bash
# Generate large dataset with photoshopped and AI-generated images
python create_large_dataset.py
```
**What it does:**
- Creates 100 base images
- Generates 500 photoshopped images with various effects
- Generates 500 AI-generated images with patterns
- Output: `large_dataset/` folder

#### 4. `download_photoshopped_hf_v2.py` - Advanced Hugging Face Downloader
```bash
# Search for available datasets
python download_photoshopped_hf_v2.py --search

# Try all known photoshopped datasets
python download_photoshopped_hf_v2.py --all

# Download from specific dataset
python download_photoshopped_hf_v2.py --dataset cifar10 --max-images 1000
```

### Prerequisites
```bash
# Install required packages
conda activate nlp_env
pip install datasets pillow tqdm requests scipy numpy
```

### Environment Setup
```bash
# Activate conda environment
conda activate nlp_env

# Verify installation
python -c "import datasets, PIL, tqdm, requests, scipy, numpy; print('All packages installed!')"
```

### Usage Examples

#### Generate New Photoshopped Images
```bash
# Quick generation (100 images)
python create_photoshopped_only.py

# The script will create:
# - photoshopped_dataset/base_images/ (200 base images)
# - photoshopped_dataset/photoshopped/ (1,000 photoshopped images)
# - photoshopped_dataset/dataset_info.txt (documentation)
```

#### Download from Hugging Face
```bash
# Download and create photoshopped versions
python download_photoshopped_hf_final.py --dataset cifar10 --max-images 1000

# This will create:
# - hf_photoshopped/original/ (original images)
# - hf_photoshopped/photoshopped/ (photoshopped versions)
```

#### Combine All Sources
```bash
# Run multiple scripts to get diverse dataset
python create_photoshopped_only.py
python download_photoshopped_hf_final.py --dataset cifar10 --max-images 500
python download_photoshopped_hf_final.py --dataset mnist --max-images 300

# Then copy all to edited folder
mkdir -p edited
cp photoshopped_dataset/photoshopped/*.png edited/
cp hf_photoshopped/photoshopped/*.png edited/
```

### Output Structure
```
edited/
├── README.md                    # This file
├── photoshopped_00000_*.png     # Main synthetic dataset
├── photoshopped_effect_*.png    # Effect-specific images
├── ai_generated_*.png           # AI-generated images
└── cifar10_photoshopped_*.png   # Hugging Face dataset images
```

### Tips
- Use `--max-images` parameter to control dataset size
- Different datasets provide different types of base images
- Combine multiple sources for more diverse training data
- All images are automatically resized to 256x256 pixels
- Check the generated documentation files for detailed information
