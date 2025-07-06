
import sys
import os
import unittest
import tempfile
import pandas as pd

# Add src directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, '..', 'src')
sys.path.insert(0, src_dir)

from recommendation_agent import RecommendationAgent, find_alternatives, format_recommendations

class TestRecommendationAgent(unittest.TestCase):
    """Test the RecommendationAgent functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Create temporary CSV file with test data
        self.temp_csv = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        
        # Sample test data
        self.test_data = [
            {
                'title': 'Premium Dog Toy Ball',
                'average_rating': 4.8,
                'price': 25.99,
                'features': '["Durable", "Safe for dogs", "Interactive"]',
                'description': 'High-quality dog toy for active pets',
                'categories': 'Pet Toys',
                'parent_asin': 'B123456789'
            },
            {
                'title': 'Budget Dog Toy',
                'average_rating': 4.2,
                'price': 8.99,
                'features': '["Affordable", "Basic"]',
                'description': 'Simple dog toy at low price',
                'categories': 'Pet Toys',
                'parent_asin': 'B987654321'
            },
            {
                'title': 'Cat Food Premium',
                'average_rating': 4.9,
                'price': 35.99,
                'features': '["High quality", "Natural ingredients"]',
                'description': 'Premium cat food with natural ingredients',
                'categories': 'Pet Food',
                'parent_asin': 'B555666777'
            }
        ]
        
        # Write test data to CSV
        df = pd.DataFrame(self.test_data)
        df.to_csv(self.temp_csv.name, index=False)
        self.temp_csv.close()
        
        # Initialize agent with test data
        self.agent = RecommendationAgent(self.temp_csv.name)
    
    def tearDown(self):
        """Clean up test fixtures"""
        os.unlink(self.temp_csv.name)
    
    def test_agent_initialization(self):
        """Test agent initialization"""
        print("Testing agent initialization...")
        self.assertIsNotNone(self.agent.df)
        self.assertEqual(len(self.agent.df), 3)
        self.assertEqual(self.agent.csv_path, self.temp_csv.name)
        print("  Agent initialized with 3 test products")
    
    def test_initialization_missing_file(self):
        """Test initialization with missing CSV file"""
        print("Testing initialization with missing file...")
        agent = RecommendationAgent("nonexistent_file.csv")
        self.assertTrue(agent.df.empty)
        print("  Agent handles missing file gracefully")
    
    def test_reason_based_filtering(self):
        """Test filtering based on return reasons"""
        print("Testing reason-based filtering...")
        
        # Test too_expensive filtering
        filtered_expensive = self.agent._apply_reason_based_filtering("too_expensive")
        prices = filtered_expensive['price'].tolist()
        self.assertEqual(prices, sorted(prices))  # Should be sorted by price ascending
        print(f"  'too_expensive' filtering: prices sorted {prices}")
        
        # Test bad_quality filtering
        filtered_quality = self.agent._apply_reason_based_filtering("bad_quality")
        ratings = filtered_quality['average_rating'].tolist()
        self.assertEqual(ratings, sorted(ratings, reverse=True))  # Should be sorted by rating descending
        print(f"  'bad_quality' filtering: ratings sorted {ratings}")
    
    def test_find_similar_products(self):
        """Test finding similar products"""
        print("Testing similar product search...")
        df = self.agent.df
        
        # Test with dog toy search
        similar_products = self.agent._find_similar_products(
            df, "Dog Toy", "Looking for alternatives"
        )
        
        self.assertIsInstance(similar_products, list)
        if similar_products:
            print(f"  Found {len(similar_products)} similar products for 'Dog Toy'")
        else:
            print("  No similar products found (threshold not met)")
    
    def test_find_similar_products_no_matches(self):
        """Test finding similar products with no matches"""
        print("Testing similar product search with no matches...")
        df = self.agent.df
        
        similar_products = self.agent._find_similar_products(
            df, "Completely Different Product", "No matches expected"
        )
        
        self.assertEqual(similar_products, [])
        print("  Correctly returned empty list for no matches")
    
    def test_content_similarity(self):
        """Test content similarity computation"""
        print("Testing content similarity...")
        user_text = "Dog toy for active pets"
        product_row = self.agent.df.iloc[0]  # Premium Dog Toy Ball
        
        similarity = self.agent._compute_content_similarity(user_text, product_row)
        
        self.assertIsInstance(similarity, float)
        self.assertGreaterEqual(similarity, 0.0)
        self.assertLessEqual(similarity, 1.0)
        print(f"  Similarity score: {similarity:.3f}")
    
    def test_jaccard_similarity(self):
        """Test Jaccard similarity computation"""
        print("Testing Jaccard similarity...")
        similarity = self.agent._jaccard_similarity("dog toy", "dog toy ball")
        
        self.assertIsInstance(similarity, float)
        self.assertGreaterEqual(similarity, 0.0)
        self.assertLessEqual(similarity, 1.0)
        print(f"  Jaccard similarity: {similarity:.3f}")
    
    def test_feature_extraction(self):
        """Test feature extraction from JSON"""
        print("Testing feature extraction...")
        features_json = '["Feature 1", "Feature 2", "Feature 3", "Feature 4"]'
        extracted = self.agent._extract_features(features_json)
        
        self.assertIn("Feature 1", extracted)
        self.assertIn("Feature 2", extracted)
        self.assertIn("Feature 3", extracted)
        # Should only include first 3 features
        self.assertNotIn("Feature 4", extracted)
        print(f"  Extracted features: {extracted}")
    
    def test_feature_extraction_invalid(self):
        """Test feature extraction with invalid JSON"""
        print("Testing feature extraction with invalid JSON...")
        features_invalid = "Invalid JSON string"
        extracted = self.agent._extract_features(features_invalid)
        
        self.assertEqual(extracted, features_invalid[:200])
        print(f"  Handled invalid JSON: {extracted[:30]}...")
    
    def test_format_recommendations(self):
        """Test recommendation formatting"""
        print("Testing recommendation formatting...")
        recommendations = [
            {
                'title': 'Test Product',
                'average_rating': 4.5,
                'price': 19.99,
                'features': '["Test feature 1", "Test feature 2"]',
                'parent_asin': 'B123456789'
            }
        ]
        
        formatted = self.agent.format_recommendations(recommendations)
        
        self.assertIn("Test Product", formatted)
        self.assertIn("4.5", formatted)
        self.assertIn("$19.99", formatted)
        self.assertIn("Test feature 1", formatted)
        self.assertIn("B123456789", formatted)
        print("  Recommendation formatted correctly")
    
    def test_format_recommendations_empty(self):
        """Test formatting empty recommendations"""
        print("Testing empty recommendation formatting...")
        formatted = self.agent.format_recommendations([])
        
        self.assertEqual(formatted, "No suitable alternatives found.")
        print("  Empty recommendations handled correctly")
    
    def test_find_alternatives_integration(self):
        """Test the main find_alternatives method"""
        print("Testing find_alternatives integration...")
        
        alternatives = self.agent.find_alternatives(
            "Dog Toy", "too_expensive", "Looking for cheaper options"
        )
        
        self.assertIsInstance(alternatives, list)
        print(f"  Found {len(alternatives)} alternatives for 'Dog Toy' with 'too_expensive' reason")
    
    def test_find_alternatives_different_reason(self):
        """Test find_alternatives with different reason"""
        print("Testing find_alternatives with 'bad_quality' reason...")
        
        alternatives = self.agent.find_alternatives(
            "Cat Food", "bad_quality", "Looking for better quality"
        )
        
        self.assertIsInstance(alternatives, list)
        print(f"  Found {len(alternatives)} alternatives for 'Cat Food' with 'bad_quality' reason")
    
    def test_standalone_functions(self):
        """Test standalone functions for backward compatibility"""
        print("Testing standalone functions...")
        
        # Test find_alternatives function
        alternatives = find_alternatives("Dog Toy", "too_expensive", "Test")
        self.assertIsInstance(alternatives, list)
        
        # Test format_recommendations function
        formatted = format_recommendations([])
        self.assertEqual(formatted, "No suitable alternatives found.")
        
        print("  Standalone functions work correctly")

def run_recommendation_tests():
    """Run recommendation agent tests with clear reporting"""
    print("Starting Recommendation Agent Tests")
    print("=" * 50)
    print("Testing recommendation functionality only...")
    print()
    
    # Create test suite
    test_suite = unittest.TestSuite()
    tests = unittest.TestLoader().loadTestsFromTestCase(TestRecommendationAgent)
    test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
    result = runner.run(test_suite)
    
    # Print summary
    print("\n" + "=" * 50)
    print("RECOMMENDATION TEST SUMMARY")
    print("=" * 50)
    print(f"Total Tests: {result.testsRun}")
    print(f"Passed: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failed: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    success_rate = ((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun) * 100
    print(f"Success Rate: {success_rate:.1f}%")
    
    if result.failures:
        print("\nFailures:")
        for test, traceback in result.failures:
            print(f"  - {test}")
    
    if result.errors:
        print("\nErrors:")
        for test, traceback in result.errors:
            print(f"  - {test}")
    
    print("\n" + "=" * 50)
    
    if success_rate == 100:
        print("All recommendation tests passed!")
        print("RecommendationAgent class works correctly")
        print("All methods function as expected")
        print("Standalone functions are compatible")
    else:
        print("Some recommendation tests failed")
        print("Please fix the issues before proceeding")
    
    return result.wasSuccessful()

def run_agent_test_cases():
    """Run the test cases from the recommendation agent main section"""
    print("\n" + "=" * 50)
    print("RUNNING AGENT TEST CASES")
    print("=" * 50)
    
    # Test the recommendation agent
    agent = RecommendationAgent()
    
    test_cases = [
        {
            "product_title": "Dog harness",
            "reason_classification": "too_expensive",
            "reason_summary": "Looking for cheaper alternatives"
        },
        {
            "product_title": "Cat Food",
            "reason_classification": "bad_quality",
            "reason_summary": "Poor quality product"
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n=== Test Case {i} ===")
        print(f"Product: {test_case['product_title']}")
        print(f"Reason: {test_case['reason_classification']}")
        print(f"Summary: {test_case['reason_summary']}")
        
        try:
            recommendations = agent.find_alternatives(**test_case)
            formatted_result = agent.format_recommendations(recommendations)
            print("Result:")
            print(formatted_result)
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    # Run unit tests first
    success = run_recommendation_tests()
    
    # Then run the agent test cases
    run_agent_test_cases()
    
    sys.exit(0 if success else 1)