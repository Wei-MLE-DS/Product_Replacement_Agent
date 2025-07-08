
import os
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import jaccard_score
from fuzzywuzzy import fuzz
from typing import List, Dict, Any


# Comprehensive list of return reason categories
RETURN_REASON_CATEGORIES = [
    "too_expensive", "bad_quality", "not_as_described", "defective", "changed_mind",
    "wrong_item", "missing_parts", "better_alternative", "late_delivery", "other"
]

class RecommendationAgent:
    """
    Agent responsible for finding alternative products
    Goal: Use reason classification + product title to find replacements
    """
    
    def __init__(self, csv_path: str = None):
        """
        Initialize the recommendation agent
        
        Args:
            csv_path: Path to the product database CSV file
        """
        self.csv_path = csv_path or os.path.join("data", "meta_amazon_review_pets.csv")
        self.df = None
        self._load_data()
    
    def _load_data(self):
        """Load and preprocess the product database"""
        if not os.path.exists(self.csv_path):
            self.df = pd.DataFrame()
            return
        
        self.df = pd.read_csv(self.csv_path)
        
        # Filter by rating >= 4.0
        self.df = self.df[pd.to_numeric(self.df['average_rating'], errors='coerce') >= 4.0]

    def find_alternatives(self, product_title: str, reason_classification: str, reason_summary: str) -> List[Dict[str, Any]]:
        """
        Find alternative products based on reason classification
        
        Args:
            product_title: Title of the product being returned
            reason_classification: Classified reason for return
            reason_summary: Summary of the return reason
            
        Returns:
            List of recommended products as dictionaries
        """
        if self.df.empty:
            return []
        
        # Apply classification-specific filtering
        filtered_df = self._apply_reason_based_filtering(reason_classification)
        
        # Find similar products
        similar_products = self._find_similar_products(filtered_df, product_title, reason_summary)
        
        return similar_products
    
    def _apply_reason_based_filtering(self, reason_classification: str) -> pd.DataFrame:
        """
        Apply specific filtering based on reason classification
        
        Args:
            reason_classification: The classified reason for return
            
        Returns:
            Filtered DataFrame
        """
        df = self.df.copy()
        
        if reason_classification == "too_expensive":
            # Sort by price (ascending) to find cheaper alternatives
            df = df.sort_values(by='price', ascending=True)
        elif reason_classification == "bad_quality":
            # Sort by rating (descending) to find better quality
            df = df.sort_values(by='average_rating', ascending=False)
        elif reason_classification == "wrong_item":
            # Keep all products for category-based matching
            pass
        elif reason_classification == "defective":
            # Look for more reliable/durable products
            df = df.sort_values(by='average_rating', ascending=False)
        
        return df
    
    def _find_similar_products(self, df: pd.DataFrame, product_title: str, reason_summary: str) -> List[Dict[str, Any]]:
        """
        Find similar products using fuzzy matching and similarity scoring
        
        Args:
            df: DataFrame to search in
            product_title: Title of the product being returned
            reason_summary: Summary of the return reason
            
        Returns:
            List of similar products as dictionaries
        """
        # Fuzzy title matching
        df["title_fuzz"] = df["title"].apply(lambda t: fuzz.token_set_ratio(product_title, str(t)))
        
        # Filter by title similarity threshold
        title_threshold = 50
        matched_df = df[df["title_fuzz"] >= title_threshold]
        
        if matched_df.empty:
            return []
        
        # Compute content similarity
        user_text = f"{product_title} {reason_summary}"
        matched_df["similarity"] = matched_df.apply(
            lambda row: self._compute_content_similarity(user_text, row), axis=1
        )
        
        # Get top recommendations
        top_recommendations = matched_df.nlargest(3, "similarity")
        
        return [row.to_dict() for _, row in top_recommendations.iterrows()]
    
    def _compute_content_similarity(self, user_text: str, product_row: pd.Series) -> float:
        """
        Compute similarity between user text and product metadata
        
        Args:
            user_text: User's input text
            product_row: Product row from DataFrame
            
        Returns:
            Similarity score between 0 and 1
        """
        product_text = ' '.join([
            str(product_row.get('title', '')),
            str(product_row.get('features', '')),
            str(product_row.get('description', '')),
            str(product_row.get('categories', ''))
        ])
        
        return self._jaccard_similarity(user_text, product_text)
    
    def _jaccard_similarity(self, str1: str, str2: str) -> float:
        """
        Compute Jaccard similarity between two strings
        
        Args:
            str1: First string
            str2: Second string
            
        Returns:
            Jaccard similarity score
        """
        vectorizer = CountVectorizer(binary=True)
        X = vectorizer.fit_transform([str1, str2])
        if X.shape[1] == 0:
            return 0.0
        return jaccard_score(X.toarray()[0], X.toarray()[1])
    
    def format_recommendations(self, recommendations: List[Dict[str, Any]]) -> str:
        """
        Format recommendations as readable text blocks
        
        Args:
            recommendations: List of recommended products
            
        Returns:
            Formatted recommendation string
        """
        if not recommendations:
            return "No suitable alternatives found."
        
        formatted_blocks = []
        for product in recommendations:
            features = self._extract_features(product.get('features', ''))
            
            block = (
                f"**{product.get('title', '')}**\n"
                f"Rating: {product.get('average_rating', '')} ⭐\n"
                f"Price: ${product.get('price', '')}\n"
                f"Features: {features}\n"
                f"ASIN: {product.get('parent_asin', '')}\n"
            )
            formatted_blocks.append(block)
        
        return '\n\n'.join(formatted_blocks)
    
    def _extract_features(self, features: str) -> str:
        """
        Extract and format product features
        
        Args:
            features: Raw features string
            
        Returns:
            Formatted features string
        """
        if isinstance(features, str) and features.startswith("["):
            try:
                import ast
                features_list = ast.literal_eval(features)
                return ', '.join(features_list[:3])  # First 3 features
            except Exception:
                return features[:200]
        return features[:200]
    

# --- Standalone Functions for Backward Compatibility ---
def find_alternatives(product_title: str, reason_classification: str, reason_summary: str) -> List[Dict[str, Any]]:
    """Standalone function to find alternatives"""
    agent = RecommendationAgent()
    return agent.find_alternatives(product_title, reason_classification, reason_summary)

def format_recommendations(recommendations: List[Dict[str, Any]]) -> str:
    """Standalone function to format recommendations"""
    agent = RecommendationAgent()
    return agent.format_recommendations(recommendations)
