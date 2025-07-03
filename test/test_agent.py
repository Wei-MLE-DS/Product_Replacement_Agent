import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

from main import run_agent_streamlit, graph

# Simulate test cases for the agent
# 1 = valid, 2 = ai-generated, 3 = photoshopped, 4 = invalid
image_labels = {
    1: 'valid',
    2: 'ai-generated',
    3: 'photoshopped',
    4: 'invalid',
}

def run_interactive():
    # This will trigger CLI prompts for all fields
    initial_state = {
        "user_id": "test_user",
        "image_path": "",
        "image_validation": "",
        "product_title": "",
        "return_reason": "",
        "recommendation": "",
        "messages": [],
        "interactive": True,
    }
    config = {"configurable": {"thread_id": "test-thread"}}
    result_state = graph.invoke(initial_state, config)
    print(result_state.get("recommendation", "No recommendation found."))

def run_noninteractive(label, label_num):
    initial_state = {
        "user_id": "test_user",
        "image_path": f"dummy_path_{label_num}.jpg",
        "image_validation": label,
        "product_title": "Dog Toy",
        "return_reason": "I want a safer and more durable product.",
        "recommendation": "",
        "messages": [],
        "interactive": True,
    }
    config = {"configurable": {"thread_id": "test-thread"}}
    result_state = graph.invoke(initial_state, config)
    print(result_state.get("recommendation", "No recommendation found."))

if __name__ == "__main__":
    print("\n=== Test Case: Image label 1 (valid) ===")
    run_interactive()
    for label_num, label in {2: 'ai-generated', 3: 'photoshopped', 4: 'invalid'}.items():
        print(f"\n=== Test Case: Image label {label_num} ({label}) ===")
        run_noninteractive(label, label_num)

# Note: You may need to update run_agent_streamlit in main.py to accept an optional image_validation_override param
# and use it to set the image_validation in the state for testing purposes. 