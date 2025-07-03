import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

from main import graph

def run_full_interactive():
    print("=== Interactive CLI Product Return Agent ===")
    image_path = input("Please upload a picture of the product (enter file path): ")
    product_title = input("Enter the product title: ")
    return_reason = input("Enter the reason for returning the product: ")
    initial_state = {
        "user_id": "test_user",
        "image_path": image_path,
        "image_validation": "valid",  # Force valid for CLI
        "product_title": product_title,
        "return_reason": return_reason,
        "recommendation": "",
        "messages": [],
        "interactive": False,  # No need for node-level input
    }
    config = {"configurable": {"thread_id": "cli-thread"}}
    result_state = graph.invoke(initial_state, config)
    print("\n=== Recommendation ===")
    print(result_state.get("recommendation", "No recommendation found."))

if __name__ == "__main__":
    run_full_interactive() 