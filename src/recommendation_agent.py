
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics.pairwise import cosine_similarity
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from transformers import PreTrainedModel, PreTrainedTokenizer, AutoModelForSequenceClassification, AutoTokenizer

import faiss
from typing import List, Dict, Any
import ast

import json
from itertools import islice
import os
import pickle
from dotenv import load_dotenv


class RecommendationAgent:
    """
    Agent responsible for finding alternative products
    Goal: Use reason + product title to find replacements
    """
    
    def __init__(self, 
                 csv_path: str = None,
                 mapping_path: str = None,
                 indexdb_path: str = None,
                 rank_model_path: str = None):
        """
        Initialize the recommendation agent
        
        Args:
            csv_path: Path to the product database CSV file
            mapping_path: Path to the index to product id mapping list
            indexdb_path: Path to the indexed vector database

        """
        self.csv_path = csv_path
        self.mapping_path = mapping_path
        self.indexdb_path = indexdb_path
        self.rank_model_path = rank_model_path
        self.df = None
        self.indexdb = None
        self.mapping_id = None
        self.retrieve_model = None
        self._load_data()
    
    def _load_data(self):
        """Load and preprocess the product database"""
        if not os.path.exists(self.csv_path):
            self.df = pd.DataFrame()
        self.df = pd.read_csv(self.csv_path)
        self.indexdb = faiss.read_index(self.indexdb_path)
        with open(self.mapping_path, "rb") as f:
            self.mapping_id = pickle.load(f)
        model_name = "sentence-transformers/all-MiniLM-L6-v2"
        self.retrieve_model = SentenceTransformer(model_name, device="cuda" if torch.cuda.is_available() else "cpu")
        self.model_dim = self.retrieve_model.get_sentence_embedding_dimension()         
        self.rank_model = AutoModelForSequenceClassification.from_pretrained(self.rank_model_path)
        self.rank_tokenizer = AutoTokenizer.from_pretrained(self.rank_model_path)
        
        # Filter by rating >= 4.0
        #self.df = self.df[pd.to_numeric(self.df['average_rating'], errors='coerce') >= 4.0]

    def find_alternatives(self, product_title: str, reason_classification: str, k = 3) -> List[Dict[str, Any]]:
        """
        Find alternative products based on reason classification
        
        Args:
            product_title: Title of the product being returned
            reason_classification: Summarized reason for return
            
        Returns:
            List of recommended products as dictionaries
        """
        if self.df.empty:
            return []
        
        # Apply classification-specific filtering
        df_retrieve_info = self._retrieve_alternatives(product_title, k = 10)
        
        # Find similar products
        similar_products = self._rank_alternatives(df_retrieve_info, product_title, reason_classification, k)

        # Format
        #output_products = self.format_recommendations(similar_products)
        return similar_products
    
    
    def _retrieve_alternatives(self, product_title: str, k = 10) -> pd.DataFrame:
        """
        Find retrieved top 10 alternative products based on embedding of product title
        
        Args:
            product_title: Title of the product being returned
            
        Returns:
            Filtered DataFrame
        """
        q_vec = self.retrieve_model.encode([product_title], normalize_embeddings=True).astype("float32")
        D, I = self.indexdb.search(q_vec, k)      # D = distances (cosine sims), I = indices
        # Get parent_asins for top-k
        top_k_asins = [self.mapping_id[i] for i in I[0]]
        retrieve_list = list(zip(I[0], D[0], top_k_asins))

        score_list = []
        asin_list = []
        for idx, score, asin in retrieve_list:
            score_list.append(score)
            asin_list.append(asin)
        df_retrieve = pd.DataFrame({"score": score_list, "parent_asin": asin_list})
        df_retrieve_info = df_retrieve.merge(self.df, on="parent_asin", how="left")
        return df_retrieve_info[1:]
    
    def _rank_alternatives(self, df_retrieve_info: pd.DataFrame, product_title: str, reason_classification: str, k = 3) -> List[Dict[str, Any]]:
        """
        Find similar products using bert model
        
        Args:
            df_retrieve_info: retrieved DataFrame
            product_title: Title of the product being returned
            reason_classification: Summary of the return reason
            k: number of returned products
            
        Returns:
            List of similar products as dictionaries
        """
        attr_list = []
        product_value_list = []
        parent_asin_list = []
        for i in range(len(df_retrieve_info)):
            details = ast.literal_eval(df_retrieve_info['details'].iloc[i])
            #details = json.loads(df_retrieve_info['details'].iloc[i].replace("'", '"'))
            if isinstance(details, dict):
                for key in details.keys():
                    parent_asin_list.append(df_retrieve_info['parent_asin'].iloc[i])
                    attr_list.append(key)
                    product_value_list.append(details[key])

        label_list, confidence_list = self._nli_compare_batch(attr_list, product_value_list, reason_classification, self.rank_model, self.rank_tokenizer)
        df_retrieve_label = pd.DataFrame({"label": label_list, "confidence": confidence_list, 'parent_asin': parent_asin_list})
        df_retrieve_label_agg = df_retrieve_label.groupby(['parent_asin', 'label']).size().reset_index(name='count')
        df_retrieve_label_agg['contradiction'] = df_retrieve_label_agg['label'].apply(lambda x: 1 if x == 'contradiction' else 0)
        df_rank = df_retrieve_label_agg.merge(df_retrieve_info, on="parent_asin", how="left")
        df_rank.sort_values(by=['contradiction'], ascending=[True], inplace=True)
        df_rank.reset_index(inplace = True, drop = True)
        df_return = df_rank[(df_rank['average_rating'] >= 4) & (df_rank['price'].notna())][:3].copy()
        
        
        return [row.to_dict() for _, row in df_return.iterrows()]
    
    def _nli_compare_batch(self, attr_names, product_values, user_values, model: PreTrainedModel, tokenizer: PreTrainedTokenizer):
        """
        Batch version of NLI inference.
        Each example is: premise = "The product's <attr_name> is <product_value>."
                        hypothesis = <user_value>
        Returns a list of (label, probability) tuples.
        """
        assert len(attr_names) == len(product_values)

        model.eval()
        device = next(model.parameters()).device

        # Construct premises and hypotheses
        premises = [f"The product's {a} is {p}." for a, p in zip(attr_names, product_values)]
        hypotheses = [user_values] * len(premises)

        # Tokenize in batch
        inputs = tokenizer(premises, hypotheses, return_tensors="pt", truncation=True, padding=True).to(device)

        with torch.no_grad():
            logits = model(**inputs).logits
            probs = F.softmax(logits, dim=1)
        # print(probs)

        labels = ["contradiction", "entailment", "neutral"]
        label_list = []
        confidence_list = []
        for prob in probs:
            pred_idx = torch.argmax(prob).item()
            label = labels[pred_idx]
            confidence = prob[pred_idx].item()
            label_list.append(label)
            confidence_list.append(confidence)

        return label_list, confidence_list



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
        for i, product in enumerate(recommendations, 1):
            features = self._extract_features(product.get('features', ''))
            
            block = (
                f"### Recommendation {i}\n\n"
                f"**{product.get('title', '')}**\n\n"
                f"**Rating:** {product.get('average_rating', '')} ⭐\n\n"
                f"**Price:** ${product.get('price', '')}\n\n"
                f"**Features:** {features}\n\n"
                f"**ASIN:** {product.get('parent_asin', '')}\n\n"
                f"---\n\n" 
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
def find_alternatives(product_title: str, reason_classification: str) -> List[Dict[str, Any]]:
    """Standalone function to find alternatives"""
    agent = RecommendationAgent()
    return agent.find_alternatives(product_title, reason_classification)

def format_recommendations(recommendations: List[Dict[str, Any]]) -> str:
    """Standalone function to format recommendations"""
    agent = RecommendationAgent()
    return agent.format_recommendations(recommendations)
