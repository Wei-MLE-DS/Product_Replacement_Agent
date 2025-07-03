# Product Return Agent

A conversational agent for product returns and recommendations, built with Python, LangGraph, and Streamlit. The agent helps users upload product images, validates them, collects return reasons, and recommends better products from a metadata dataset or via web search.

---

## Features
- **Image Upload & Validation**: Detects valid, AI-generated, or photoshopped images.
- **Conversational UI**: Collects product title and return reason from the user.
- **Metadata-Based Recommendation**: Finds similar or better products using fuzzy matching and metadata similarity.
- **Web Search Fallback**: If no suitable product is found locally, the agent will automatically perform a web search and return a clearly formatted web result as a fallback.
- **Streamlit UI**: User-friendly chatbot interface with test mode for easy testing.
- **Automated and Interactive CLI Testing**: Use test scripts for both automated and manual testing.

---

## Project Structure (Current)

```
product_return_agent/
├── data/
│   ├── amazon_review_pets.csv
│   ├── meta_amazon_review_pets.csv
│   └── ...
├── src/
│   ├── main.py
│   ├── extract_pet_review_sample.py
│   ├── product_return_agent_ui.app.py
│   └── ...
├── test/
│   ├── test_agent.py
│   ├── cli_agent_test.py
│   ├── test_web_search.py
│   └── ...
├── requirements.txt
├── README.md
├── .gitignore
├── workflow.mmd
├── workflow.png
├── venv/
│   └── ... (virtual environment files)
└── ...
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
     TAVILY_API_KEY=your_tavily_key
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
- If no similar product is found locally, the agent will automatically perform a web search and return a clearly formatted web result as a fallback.
- **Note:** All data files must be in the `data/` directory.

### **Automated Test of the Agent**
```bash
python test/test_agent.py
```
- Runs a set of automated test cases for the agent logic with sample input and prints the results for each scenario.
- **Note:** All data files must be in the `data/` directory.

### **Interactive CLI Test of the Agent**
```bash
python test/cli_agent_test.py
```
- Prompts you for image path, product title, and reason for return, then runs the agent and prints the recommendation.
- Useful for manual, human-in-the-loop testing.
- **Note:** All data files must be in the `data/` directory.

### **Test Web Search Fallback**
```bash
python test/test_web_search.py
```
- Runs a test of the web search fallback logic with a sample product title and reason that does not exist in the local data, ensuring the web search fallback is triggered and the result is clearly formatted.
- **Note:** All data files must be in the `data/` directory.

---

## Workflow Diagram

The workflow is visualized in `workflow.png`:

![Workflow Diagram](workflow.png)

---

## Dependencies
- Python 3.9+
- streamlit
- pandas
- fuzzywuzzy
- python-Levenshtein (optional, for speed)
- openai
- tavily
- langgraph
- langchain-core
- langchain-ollama (optional)
- scikit-learn
- python-dotenv

---

## Notes
- The agent uses fuzzy string matching for robust product title clustering.
- Test mode allows you to skip image upload and simulate different validation results.
- For best results, ensure your virtual environment is activated before running scripts.
- Both automated and interactive CLI testing are supported via `test_agent.py` and `cli_agent_test.py`.
- All scripts and the UI expect data files to be in the `data/` directory.
- If no suitable product is found locally, the agent will automatically perform a web search and return a clearly formatted web result as a fallback.

---

## License
MIT 