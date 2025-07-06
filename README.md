# Smart Agent for Fraud Detection and Product Replacement

## Overview

This repository contains the codebase and models for an intelligent agent system designed to support fraud detection and product replacement recommendation. The agent leverages computer vision, image metadata, customer behavior data, and existing recommender systems to automate the detection of fraudulent returns and recommend appropriate replacement products.


---

## Architecture

### **Router-Based 3-Agent System**
The system is built with a clean, modular architecture using LangGraph:

1. **Image Agent** - Classifies uploaded images as "real", "ai_generated", or "photoshopped"
2. **Reason Agent** - Uses LLM to classify return reasons and create summaries
3. **Recommendation Agent** - Finds alternative products using fuzzy matching and similarity scoring

---

## Features
- **Fraud Image Detection Classification**: Detects real, AI-generated, or photoshopped images
- **LLM-Powered Reason Analysis**: Classifies return reasons using OpenAI GPT-3.5-turbo
- **Smart Product Recommendations**: Uses fuzzy matching + Jaccard similarity for accurate recommendations
- **Router-Based Workflow**: Clean conditional routing between agents
- **Reason-Based Filtering**: Different recommendation strategies based on return reason
- **Streamlit UI**: User-friendly chatbot interface with test mode

---

## Project Structure

```
product_return_agent/
├── data/
│   ├── amazon_review_pets.csv
│   ├── meta_amazon_review_pets.csv
├── src/
│   ├── main.py
│   ├── image_agent.py 
│   ├── reason_agent.py 
│   ├── recommendation_agent.py 
│   ├── image_agent.py 
│   ├── product_return_agent_ui.app.py
├── test/
│   ├── test_reasonAgent.py
│   ├── test_recommendationAgent.py
├── requirements.txt
├── README.md
├── workflow.mmd
```

---

## Setup

1. **Clone the repository**
2. **Create and activate a Python 3.9+ virtual environment**
   ```bash
   python3.9 -m venv venv
   source venv/bin/activate
   ```
3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
4. **Set up API keys**
   - Create a `.env` file with your OpenAI and Tavily API keys:
     ```env
     OPENAI_API_KEY=your_openai_key
     ```
5. **Place all data files in the `data/` directory**
   - Example: `data/meta_amazon_review_pets.csv`, `data/amazon_review_pets.csv`, etc.

---

## Usage

### **Run the Streamlit UI**
```bash
streamlit run src/product_return_agent_ui.py
```
- Upload a product image or use test mode.
- Enter product title and reason for return.
- View recommendations in a chat-like interface.
- **Note:** All data files must be in the `data/` directory.


---

## Agent Details

### **Image Agent**
- **Purpose**: Classify images into three categories:
📷 Camera-captured (Real)
🧠 AI-generated
🖼️ Photoshopped (PS)
Approach:
Fine-tuned CNN models on real, AI, and edited images.
Metadata feature extraction: EXIF data, DCT patterns, image quality, blur/sharpness, entropy, and statistical moments (e.g., Hu moments).
- **Output**: "real", "ai_generated", "photoshopped"
- **File**: `src/image_agent.py`

### **Reason Agent**
- **Purpose**: Analyze return reasons using LLM
- **Features**: 
  - Classifies reasons into categories (too_expensive, bad_quality, etc.)
  - Creates concise summaries
  - Provides confidence scoring
  - Suggests filtering strategies
- **File**: `src/reason_agent.py`

### **Recommendation Agent**
- **Purpose**: Find alternative products
- **Features**:
  - Fuzzy title matching using `fuzzywuzzy`
  - Rating filtering (≥4.0 stars)
  - Reason-based filtering strategies
  - Jaccard similarity for metadata matching
  - Formatted recommendation output
- **File**: `src/recommendation_agent.py`

---



## Workflow Diagram

The workflow is visualized in `workflow.png`:

![Workflow Diagram](workflow.png)


---

## Notes
- The agent uses fuzzy string matching for robust product title clustering.
- Test mode allows you to skip image upload and simulate different validation results.
- For best results, ensure your virtual environment is activated before running scripts.
- All scripts and the UI expect data files to be in the `data/` directory.

---

