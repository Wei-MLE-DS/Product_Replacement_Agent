from main import web_search_node, ReturnAgentState


if __name__ == "__main__":
    # Example test input
    state = {
        "user_id": "test_user",
        "image_path": "dummy.jpg",
        "image_validation": "valid",
        "product_title": "Bone Broth for Dogs &Cats",
        "return_reason": "my cat do not like the taste of it",
        "recommendation": "",
        "messages": [],
    }
    result_state = web_search_node(state)
    print("Web Search Recommendation Result:")
    print(result_state["recommendation"]) 