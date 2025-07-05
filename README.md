# Image Metadata Extraction and Hybrid Model Building

This project provides a workflow for extracting rich metadata from images and building a hybrid deep learning model to classify images as **real**, **photoshopped (edited)**, or **AI-generated**.

## Overview

The core steps are implemented in the notebook [`image_metadata_extraction.ipynb`](image_metadata_extraction.ipynb):

1. **Metadata Extraction:**  
   - Extracts features such as EXIF data, DCT statistics, entropy, blur/sharpness, Hu moments, noise, edge density, and quantization tables from images.
   - Processes images from three folders: `human/`, `edited/`, and `AI/`.
   - Saves the extracted features and labels into a CSV file for further processing.

2. **Data Preparation:**  
   - Splits the extracted metadata into training, validation, and test sets.
   - Normalizes metadata features using `StandardScaler` from scikit-learn.

3. **Hybrid Model Building:**  
   - Defines a PyTorch model (`HybridNet`) that combines:
     - Image features from EfficientNet-B0 (pretrained on ImageNet).
     - Tabular metadata features from the extraction step.
   - Trains the model to classify images into three categories: real, photoshopped, or AI-generated.
   - Evaluates the model on validation and test sets.

## Extracted Image Features

The following metadata features are extracted from each image:

- **EXIF Data** (for JPEG/TIFF):
  - `Make`: Camera manufacturer.
  - `Model`: Camera model.
  - `Software`: Software used to process the image.
  - `DateTimeOriginal`: Original date and time when the image was created.
- **Quantization Table (JPEG only):**
  - `qtable_mean`, `qtable_std`, `qtable_max`, `qtable_min`: Statistics of the JPEG quantization table, which can indicate compression artifacts.
- **Entropy:**
  - `r_entropy`, `g_entropy`, `b_entropy`: Entropy (information content) of each color channel, reflecting image complexity and randomness.
- **Blur/Sharpness:**
  - `blur_metric`: Variance of the Laplacian, a measure of image sharpness (lower values indicate blur).
- **Hu Moments:**
  - `hu_1` to `hu_7`: Invariant moments capturing shape characteristics of the image, robust to scale, rotation, and translation.
- **DCT Features:**
  - `dct_mean`, `dct_std`, `dct_max`, `dct_energy`: Statistics of the Discrete Cosine Transform coefficients, capturing frequency information and compression artifacts.
- **Noise Features:**
  - `noise_mean`, `noise_std`: Mean and standard deviation of the difference between the image and its Gaussian-blurred version, indicating noise level.
- **Edge Density:**
  - `edge_density`: Proportion of edge pixels detected by the Canny edge detector, reflecting image detail and structure.
- **Image Statistics:**
  - `width`, `height`: Image dimensions in pixels.

## How to Use

1. **Prepare your image folders:**
   - Place real images in `image_detection/human/`
   - Place photoshopped images in `image_detection/edited/`
   - Place AI-generated images in `image_detection/AI/`

2. **Run the notebook:**
   - Open `image_metadata_extraction.ipynb` in Jupyter or VSCode.
   - Execute all cells to:
     - Extract metadata and save to CSV.
     - Split and normalize the data.
     - Train and evaluate the hybrid model.

3. **Outputs:**
   - Metadata CSVs: `image_detection/train_meta.csv`, `val_meta.csv`, `test_meta.csv`
   - Trained model: `image_detection/hybrid_efficientnetb0_metadata.pth`
   - Scaler: `image_detection/metadata_scaler.pkl`

## Requirements

- Python 3.8+
- PyTorch
- torchvision
- pandas
- numpy
- scikit-learn
- Pillow
- opencv-python
- piexif
- tqdm

Install dependencies with:
```sh
pip install torch torchvision pandas numpy scikit-learn pillow opencv-python piexif tqdm
```

## Notes

- The notebook is modular: you can use the extraction functions and model code in your own scripts.
- The workflow is designed for easy extension to new datasets or additional metadata features. 
