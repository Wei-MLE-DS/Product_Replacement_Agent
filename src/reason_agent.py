
import os
import openai
from typing import Dict, Any
from dotenv import load_dotenv

load_dotenv()

# --- Constants ---
REASON_CLASSIFICATIONS = [
    "too_expensive", "bad_quality", "wrong_item", "defective", 
    "changed_mind", "late_delivery", "missing_parts", "better_alternative"
]

class ReasonAgent:
    """
    Agent responsible for analyzing return reasons
    Goal: Classify reason and create summary using LLM
    """
    
    def __init__(self, openai_api_key: str = None):
        """
        Initialize the reason agent
        
        Args:
            openai_api_key: OpenAI API key (optional, will use env var if not provided)
        """
        self.api_key = self._get_api_key(openai_api_key)
        
        if not self.api_key:
            raise ValueError(
                "OpenAI API key is required. Set OPENAI_API_KEY in:\n"
                "1. Direct parameter to ReasonAgent()\n"
                "2. .env file in project root\n"
                "3. System environment variable"
            )
        
        self.client = openai.OpenAI(api_key=self.api_key)
    
    def _get_api_key(self, direct_key: str = None) -> str:
        """
        Get API key from multiple sources with priority order
        
        Args:
            direct_key: API key passed directly to constructor
            
        Returns:
            API key string or None if not found
        """
        #  Direct parameter
        if direct_key:
            return direct_key
        
        # Try to load from .env file first
        load_dotenv()
        
        #  Check system environment variables (including those from .env)
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            return api_key
        
        # No API key found from any source
        print("No API key found in any source")
        return None
    
    def analyze_reason(self, return_reason: str, product_title: str = "") -> Dict[str, str]:
        """
        Analyze return reason and return classification and summary
        
        Args:
            return_reason: The reason for returning the product
            product_title: Title of the product (optional, for context)
            
        Returns:
            Dictionary with 'classification' and 'summary' keys
        """
        if not return_reason:
            return {
                "classification": "",
                "summary": ""
            }
        
        # Classify the reason
        classification = self._classify_return_reason(return_reason)
        
        # Create summary
        summary = self._summarize_return_reason(return_reason, product_title)
        
        return {
            "classification": classification,
            "summary": summary
        }
    
    def _classify_return_reason(self, reason: str) -> str:
        """
        Use LLM to classify return reason into predefined categories
        
        Args:
            reason: The return reason text
            
        Returns:
            Classified reason category
        """
        prompt = f"""
        Classify the following product return reason into one of these categories: {', '.join(REASON_CLASSIFICATIONS)}.
        Return only the category name.
        
        Reason: {reason}
        Category:"""
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=10,
                temperature=0,
            )
            return response.choices[0].message.content.strip().lower()
        except Exception as e:
            print(f"Error classifying reason: {e}")
            return "other"
    
    def _summarize_return_reason(self, reason: str, product_title: str) -> str:
        """
        Use LLM to create a concise summary of the return reason
        
        Args:
            reason: The return reason text
            product_title: Title of the product
            
        Returns:
            Summarized reason
        """
        prompt = f"""
        Summarize the following return reason for a product in 1-2 sentences:
        Product: {product_title}
        Reason: {reason}
        
        Summary:"""
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=50,
                temperature=0.3,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error summarizing reason: {e}")
            return reason[:100] + "..." if len(reason) > 100 else reason
    
    def get_reason_insights(self, return_reason: str, product_title: str = "") -> Dict[str, Any]:
        """
        Get comprehensive insights about the return reason
        
        Args:
            return_reason: The return reason text
            product_title: Title of the product
            
        Returns:
            Dictionary with classification, summary, and additional insights
        """
        analysis = self.analyze_reason(return_reason, product_title)
        
        # Add additional insights based on classification
        insights = {
            "classification": analysis["classification"],
            "summary": analysis["summary"],
            "original_reason": return_reason,
            "product_title": product_title,
            "confidence": self._get_classification_confidence(return_reason, analysis["classification"]),
            "suggested_filters": self._get_suggested_filters(analysis["classification"])
        }
        
        return insights
    
    def _get_classification_confidence(self, reason: str, classification: str) -> str:
        """
        Get confidence level for the classification
        
        Args:
            reason: Original reason text
            classification: Classified category
            
        Returns:
            Confidence level (high/medium/low)
        """
        # Simple heuristic based on keyword matching
        keywords = {
            "too_expensive": ["expensive", "price", "cost", "cheap", "overpriced"],
            "bad_quality": ["quality", "poor", "bad", "broken", "defective"],
            "wrong_item": ["wrong", "different", "not what", "ordered"],
            "defective": ["defective", "broken", "not working", "faulty"]
        }
        
        if classification in keywords:
            matching_keywords = [kw for kw in keywords[classification] if kw.lower() in reason.lower()]
            if len(matching_keywords) >= 2:
                return "high"
            elif len(matching_keywords) >= 1:
                return "medium"
        
        return "low"
    
    def _get_suggested_filters(self, classification: str) -> Dict[str, Any]:
        """
        Get suggested filtering criteria based on classification
        
        Args:
            classification: The classified reason
            
        Returns:
            Dictionary with suggested filters
        """
        filter_suggestions = {
            "too_expensive": {
                "sort_by": "price",
                "sort_order": "ascending",
                "max_price_ratio": 0.8
            },
            "bad_quality": {
                "sort_by": "rating",
                "sort_order": "descending",
                "min_rating": 4.5
            },
            "wrong_item": {
                "sort_by": "similarity",
                "sort_order": "descending"
            },
            "defective": {
                "sort_by": "rating",
                "sort_order": "descending",
                "min_rating": 4.3
            }
        }
        
        return filter_suggestions.get(classification, {
            "sort_by": "rating",
            "sort_order": "descending"
        })

# --- Standalone Functions for Backward Compatibility ---
def classify_return_reason(reason: str) -> str:
    """Standalone function to classify return reason"""
    agent = ReasonAgent()
    return agent._classify_return_reason(reason)

def summarize_return_reason(reason: str, product_title: str) -> str:
    """Standalone function to summarize return reason"""
    agent = ReasonAgent()
    return agent._summarize_return_reason(reason, product_title)

def analyze_reason(return_reason: str, product_title: str = "") -> Dict[str, str]:
    """Standalone function to analyze return reason"""
    agent = ReasonAgent()
    return agent.analyze_reason(return_reason, product_title)

# --- Testing ---
if __name__ == "__main__":
    # Test the reason agent
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
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n=== Test Case {i} ===")
        insights = agent.get_reason_insights(**test_case)
        print(f"Classification: {insights['classification']}")
        print(f"Summary: {insights['summary']}")
        print(f"Confidence: {insights['confidence']}")
        print(f"Suggested Filters: {insights['suggested_filters']}")