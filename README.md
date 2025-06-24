🛡️ Smart Agent for Fraud Detection and Product Replacement

Overview

This repository contains the codebase and models for an intelligent agent system designed to support fraud detection and product replacement recommendation. The agent leverages computer vision, image metadata, customer behavior data, and existing recommender systems to automate the detection of fraudulent returns and recommend appropriate replacement products.

🔍 Components

1. Fraud Detection Module
This module identifies potentially fraudulent return cases using both image analysis and customer history.

a. Image-Based Detection

Goal: Classify images into three categories:
📷 Camera-captured (Real)
🧠 AI-generated
🖼️ Photoshopped (PS)
Approach:
Fine-tuned CNN models on real, AI, and edited images.
Metadata feature extraction: EXIF data, DCT patterns, image quality, blur/sharpness, entropy, and statistical moments (e.g., Hu moments).
Hybrid model combining image and metadata features.
b. Behavioral Analysis

Input Data:
Purchase history
Return patterns
Product reviews
Output from the image detection model
Goal: Identify suspicious customer behavior using ML classifiers or rule-based models.
2. Product Replacement Recommendation Module
Leverages existing recommender systems and sample product datasets.
Filters and ranks replacement suggestions based on:
Similarity to original purchase
Inventory status
Review scores
Return frequency


smart-agent/
│
├── data/                     # Raw and processed image & metadata
├── notebooks/                # EDA, experiments, and training logs
├── models/                   # Saved trained models
├── src/
│   ├── fraud_detection/
│   │   ├── image_model.py    # CNN model for image classification
│   │   ├── metadata_model.py # Metadata-based classifier
│   │   └── fusion_model.py   # Ensemble or hybrid model logic
│   └── product_replacement/
│       └── recommender.py    # Product replacement logic
├── utils/                    # Feature extraction, image processing, helpers
├── config/                   # Config files, model parameters
├── requirements.txt
└── README.md
