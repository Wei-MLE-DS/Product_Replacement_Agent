from main import run_agent_streamlit

# Simulate test cases for the agent
# 1 = valid, 2 = ai-generated, 3 = photoshopped, 4 = invalid
image_labels = {
    1: 'valid',
    2: 'ai-generated',
    3: 'photoshopped',
    4: 'invalid',
}

# Example user inputs
product_title = "Dog Toy"
return_reason = "I want a safer and more durable product."

for label_num, label in image_labels.items():
    print(f"\n=== Test Case: Image label {label_num} ({label}) ===")
    # Instead of running image validation, directly set the label in the state
    initial_state = {
        "user_id": "test_user",
        "image_path": f"dummy_path_{label_num}.jpg",
        "image_validation": label,
        "product_title": product_title,
        "return_reason": return_reason,
        "recommendation": "",
        "messages": [],
    }
    # Call the agent logic (simulate the workflow)
    result = run_agent_streamlit(
        initial_state["image_path"],
        initial_state["product_title"],
        initial_state["return_reason"],
        image_validation_override=label  # You may need to add this param to your backend
    )
    print(result)

# Note: You may need to update run_agent_streamlit in main.py to accept an optional image_validation_override param
# and use it to set the image_validation in the state for testing purposes. 