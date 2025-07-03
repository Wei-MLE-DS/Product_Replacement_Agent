import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

from main import web_search_node, ReturnAgentState


if __name__ == "__main__":
    # Example test input that will NOT match any product title, to trigger web search fallback
    state = {
        "user_id": "test_user",
        "image_path": "dummy.jpg",
        "image_validation": "valid",
        "product_title": "Completely Nonexistent Product Title",
        "return_reason": "This is a test to trigger web search fallback.",
        "recommendation": "",
        "messages": [],
    }
    result_state = web_search_node(state)
    print("Web Search Recommendation Result:")
    print(result_state["recommendation"]) 