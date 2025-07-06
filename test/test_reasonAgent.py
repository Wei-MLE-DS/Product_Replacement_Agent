
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, '..', 'src')
sys.path.insert(0, src_dir)

from reason_agent import ReasonAgent, classify_return_reason, summarize_return_reason, analyze_reason

def test_reason_agent():
    """Test the ReasonAgent class"""
    print("=== Testing ReasonAgent Class ===")
    
    # Initialize the agent
    agent = ReasonAgent()
    
    test_cases = [
        {
            "return_reason": "This product is too expensive for what it offers",
            "product_title": "Premium Dog Toy"
        },
        {
            "return_reason": "Poor quality, my cat won't eat it",
            "product_title": "Cat Food"
        },
        {
            "return_reason": "Wrong item was delivered",
            "product_title": "Dog Collar"
        },
        {
            "return_reason": "Defective product, doesn't work properly",
            "product_title": "Electronic Pet Feeder"
        },
        {
            "return_reason": "Changed my mind, don't need it anymore",
            "product_title": "Pet Carrier"
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n--- Test Case {i} ---")
        print(f"Product: {test_case['product_title']}")
        print(f"Reason: {test_case['return_reason']}")
        
        # Test comprehensive insights
        insights = agent.get_reason_insights(**test_case)
        print(f"Classification: {insights['classification']}")
        print(f"Summary: {insights['summary']}")
        print(f"Confidence: {insights['confidence']}")
        print(f"Suggested Filters: {insights['suggested_filters']}")

def test_standalone_functions():
    """Test the standalone functions for backward compatibility"""
    print("\n=== Testing Standalone Functions ===")
    
    test_reason = "This product is too expensive for what it offers"
    test_title = "Premium Dog Toy"
    
    # Test classification
    classification = classify_return_reason(test_reason)
    print(f"Classification: {classification}")
    
    # Test summarization
    summary = summarize_return_reason(test_reason, test_title)
    print(f"Summary: {summary}")
    
    # Test analysis
    analysis = analyze_reason(test_reason, test_title)
    print(f"Analysis: {analysis}")

def test_edge_cases():
    """Test edge cases and error handling"""
    print("\n=== Testing Edge Cases ===")
    
    agent = ReasonAgent()
    
    edge_cases = [
        {
            "return_reason": "",  # Empty reason
            "product_title": "Test Product"
        },
        {
            "return_reason": "Very long reason " * 50,  # Very long reason
            "product_title": "Test Product"
        },
        {
            "return_reason": "Special chars: !@#$%^&*()",  # Special characters
            "product_title": "Test Product"
        }
    ]
    
    for i, test_case in enumerate(edge_cases, 1):
        print(f"\n--- Edge Case {i} ---")
        try:
            insights = agent.get_reason_insights(**test_case)
            print(f"Result: {insights}")
        except Exception as e:
            print(f"Error: {e}")

def test_classification_categories():
    """Test all classification categories"""
    print("\n=== Testing Classification Categories ===")
    
    agent = ReasonAgent()
    
    category_tests = [
        ("too_expensive", "This is way too expensive for what it is"),
        ("bad_quality", "Poor quality, broke after first use"),
        ("wrong_item", "Wrong item was delivered"),
        ("defective", "Defective product, doesn't work"),
        ("changed_mind", "Changed my mind, don't need it"),
        ("late_delivery", "Delivery was very late"),
        ("missing_parts", "Missing parts in the package"),
        ("better_alternative", "Found a better alternative")
    ]
    
    for expected_category, reason in category_tests:
        classification = agent._classify_return_reason(reason)
        print(f"Expected: {expected_category}, Got: {classification}")

if __name__ == "__main__":
    # Run all tests
    test_reason_agent()
    test_standalone_functions()
    test_edge_cases()
    test_classification_categories()
    
    print("\n=== All Tests Completed ===")