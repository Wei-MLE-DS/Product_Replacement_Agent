import os
import torch
import numpy as np
import pandas as pd
import json
import joblib
from PIL import Image
from typing import Literal, Tuple, Optional
from image_metadata_utils import (
    extract_metadata_from_pil_image, 
    HybridNet, 
    transform, 
    load_scalers, 
    extract_and_scale_metadata
)

class ImageAgent:
    """
    Two-Stage Image Classification Agent
    
    Stage 1: Real vs Not Real
    Stage 2: Edited vs AI (only for images classified as 'not real')
    
    Returns: "real", "ai_generated", "photoshopped"
    """
    
    def __init__(self, model_dir: str = "."):
        """
        Initialize the two-stage image classification agent
        
        Args:
            model_dir: Directory containing model files and scalers
        """
        self.model_dir = model_dir
        self.device = torch.device("cpu")
        
        # Load meta features for both stages
        self.meta_features1 = self._load_meta_features('train_meta_stage1_train.csv')
        self.meta_features2 = self._load_meta_features('train_meta_stage2_train.csv')
        
        # Load scalers
        self.scaler1, self.scaler2 = load_scalers(
            f'{model_dir}/metadata_scaler_stage1.pkl',
            f'{model_dir}/metadata_scaler_stage2.pkl'
        )
        
        # Load models (using fixed versions)
        self.model1 = self._load_model('model1_real_vs_not_real_fixed.pth', len(self.meta_features1), 2)
        self.model2 = self._load_model('model2_edited_vs_ai_fixed.pth', len(self.meta_features2), 2)
        
        # Load thresholds
        self.threshold_list1, self.threshold_list2 = self._load_thresholds()
        
        print("ImageAgent initialized successfully!")
        print(f"Stage 1 meta features: {len(self.meta_features1)}")
        print(f"Stage 2 meta features: {len(self.meta_features2)}")
    
    def _load_meta_features(self, csv_file: str) -> list:
        """Load meta features from CSV file"""
        try:
            df = pd.read_csv(f'{self.model_dir}/{csv_file}')
            meta_features = [col for col in df.columns 
                           if col not in ['filename', 'label', 'format', 'label_stage1', 'label_stage2', 'folder']]
            return meta_features
        except Exception as e:
            print(f"Error loading meta features from {csv_file}: {e}")
            return []
    
    def _load_model(self, model_file: str, num_metadata_features: int, num_classes: int) -> HybridNet:
        """Load a trained model"""
        try:
            model = HybridNet(num_metadata_features=num_metadata_features, num_classes=num_classes)
            model.load_state_dict(torch.load(f'{self.model_dir}/{model_file}', map_location=self.device))
            model.eval()
            model.to(self.device)
            return model
        except Exception as e:
            print(f"Error loading model {model_file}: {e}")
            return None
    
    def _load_thresholds(self) -> Tuple[list, list]:
        """Load optimized thresholds for both stages"""
        try:
            with open(f'{self.model_dir}/best_thresholds_model1.json', 'r') as f:
                thresholds1 = json.load(f)
            with open(f'{self.model_dir}/best_thresholds_model2.json', 'r') as f:
                thresholds2 = json.load(f)
            
            threshold_list1 = [thresholds1[f"class_{i}"] for i in range(2)]  # [real, not_real]
            threshold_list2 = [thresholds2[f"class_{i}"] for i in range(2)]  # [edited, ai]
            
            return threshold_list1, threshold_list2
        except Exception as e:
            print(f"Error loading thresholds: {e}")
            return [0.5, 0.5], [0.5, 0.5]  # Default thresholds
    
    def _predict_with_thresholds(self, probs: np.ndarray, threshold_list: list) -> int:
        """
        Apply optimized thresholds to get prediction with improved handling for edge cases
        
        Args:
            probs: Probability array [class0_prob, class1_prob]
            threshold_list: Threshold values [threshold0, threshold1]
            
        Returns:
            Predicted class index (0 or 1)
        """
        # Check if probabilities are valid
        if np.any(np.isnan(probs)) or np.any(np.isinf(probs)):
            return np.argmax(probs)  # Fallback to argmax if probabilities are invalid
        
        # Get the class with highest probability
        max_prob_class = np.argmax(probs)
        max_prob = probs[max_prob_class]
        
        # Check if the highest probability exceeds its threshold
        if threshold_list[max_prob_class] is not None and max_prob > threshold_list[max_prob_class]:
            return max_prob_class
        
        # If highest probability doesn't exceed its threshold, check the other class
        other_class = 1 - max_prob_class
        other_prob = probs[other_class]
        
        if threshold_list[other_class] is not None and other_prob > threshold_list[other_class]:
            return other_class
        
        # If neither class exceeds its threshold, use a more sophisticated decision
        # Calculate confidence margin
        prob_diff = abs(probs[0] - probs[1])
        
        # If probabilities are very close (within 0.1), use threshold-based decision
        if prob_diff < 0.1:
            # Use the class that's closer to its threshold
            threshold_distances = []
            for i, (prob, threshold) in enumerate(zip(probs, threshold_list)):
                if threshold is not None:
                    distance = abs(prob - threshold)
                else:
                    distance = float('inf')
                threshold_distances.append(distance)
            
            return np.argmin(threshold_distances)
        
        # Otherwise, use the class with higher probability
        return max_prob_class
    
    def _analyze_threshold_decision(self, probs: np.ndarray, threshold_list: list) -> dict:
        """
        Analyze the threshold decision process for debugging and transparency
        
        Args:
            probs: Probability array [class0_prob, class1_prob]
            threshold_list: Threshold values [threshold0, threshold1]
            
        Returns:
            Dictionary with decision analysis
        """
        analysis = {
            'probabilities': probs.tolist(),
            'thresholds': threshold_list,
            'max_prob_class': int(np.argmax(probs)),
            'max_prob': float(np.max(probs)),
            'prob_diff': float(abs(probs[0] - probs[1])),
            'decision_reason': '',
            'threshold_exceeded': []
        }
        
        # Check if probabilities are valid
        if np.any(np.isnan(probs)) or np.any(np.isinf(probs)):
            analysis['decision_reason'] = 'fallback_to_argmax_due_to_invalid_probs'
            return analysis
        
        # Check which thresholds are exceeded
        for i, (prob, threshold) in enumerate(zip(probs, threshold_list)):
            if threshold is not None and prob > threshold:
                analysis['threshold_exceeded'].append(i)
        
        # Determine decision reason
        max_prob_class = np.argmax(probs)
        max_prob = probs[max_prob_class]
        
        if threshold_list[max_prob_class] is not None and max_prob > threshold_list[max_prob_class]:
            analysis['decision_reason'] = 'highest_prob_exceeds_threshold'
        else:
            other_class = 1 - max_prob_class
            other_prob = probs[other_class]
            
            if threshold_list[other_class] is not None and other_prob > threshold_list[other_class]:
                analysis['decision_reason'] = 'other_class_exceeds_threshold'
            elif abs(probs[0] - probs[1]) < 0.1:
                analysis['decision_reason'] = 'close_probabilities_use_threshold_distance'
            else:
                analysis['decision_reason'] = 'fallback_to_highest_probability'
        
        return analysis
    
    def classify_image(self, image_path: str) -> Literal["real", "ai_generated", "photoshopped"]:
        """
        Classify an image using the two-stage pipeline
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Classification: "real", "ai_generated", or "photoshopped"
        """
        if not image_path or not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            
            # Stage 1: Real vs Not Real
            meta1_scaled = extract_and_scale_metadata(image, self.meta_features1, self.scaler1)
            meta1 = torch.tensor(meta1_scaled, dtype=torch.float32).to(self.device)
            img_tensor = transform(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                output1 = self.model1(img_tensor, meta1)
                probs1 = torch.softmax(output1, dim=1).cpu().numpy()[0]
                stage1_pred = self._predict_with_thresholds(probs1, self.threshold_list1)
                
                if stage1_pred == 0:  # Real (human)
                    return "real"
                else:  # Not Real - need Stage 2
                    # Stage 2: Edited vs AI
                    meta2_scaled = extract_and_scale_metadata(image, self.meta_features2, self.scaler2)
                    meta2 = torch.tensor(meta2_scaled, dtype=torch.float32).to(self.device)
                    
                    output2 = self.model2(img_tensor, meta2)
                    probs2 = torch.softmax(output2, dim=1).cpu().numpy()[0]
                    stage2_pred = self._predict_with_thresholds(probs2, self.threshold_list2)
                    
                    if stage2_pred == 0:  # Edited
                        return "photoshopped"
                    else:  # AI
                        return "ai_generated"
                        
        except Exception as e:
            print(f"Error classifying image {image_path}: {e}")
            raise
    
    def classify_image_with_confidence(self, image_path: str) -> Tuple[str, float, dict]:
        """
        Classify an image and return confidence scores
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Tuple of (classification, confidence, probabilities_dict)
        """
        if not image_path or not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            
            # Stage 1: Real vs Not Real
            meta1_scaled = extract_and_scale_metadata(image, self.meta_features1, self.scaler1)
            meta1 = torch.tensor(meta1_scaled, dtype=torch.float32).to(self.device)
            img_tensor = transform(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                output1 = self.model1(img_tensor, meta1)
                probs1 = torch.softmax(output1, dim=1).cpu().numpy()[0]
                stage1_pred = self._predict_with_thresholds(probs1, self.threshold_list1)
                
                if stage1_pred == 0:  # Real (human)
                    return "real", probs1[0], {"stage1": {"real": probs1[0], "not_real": probs1[1]}}
                else:  # Not Real - need Stage 2
                    # Stage 2: Edited vs AI
                    meta2_scaled = extract_and_scale_metadata(image, self.meta_features2, self.scaler2)
                    meta2 = torch.tensor(meta2_scaled, dtype=torch.float32).to(self.device)
                    
                    output2 = self.model2(img_tensor, meta2)
                    probs2 = torch.softmax(output2, dim=1).cpu().numpy()[0]
                    stage2_pred = self._predict_with_thresholds(probs2, self.threshold_list2)
                    
                    if stage2_pred == 0:  # Edited
                        return "photoshopped", probs2[0], {
                            "stage1": {"real": probs1[0], "not_real": probs1[1]},
                            "stage2": {"edited": probs2[0], "ai": probs2[1]}
                        }
                    else:  # AI
                        return "ai_generated", probs2[1], {
                            "stage1": {"real": probs1[0], "not_real": probs1[1]},
                            "stage2": {"edited": probs2[0], "ai": probs2[1]}
                        }
                        
        except Exception as e:
            print(f"Error classifying image {image_path}: {e}")
            raise
    
    def classify_image_with_analysis(self, image_path: str) -> Tuple[str, float, dict, dict]:
        """
        Classify an image and return detailed analysis including threshold decisions
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Tuple of (classification, confidence, probabilities_dict, analysis_dict)
        """
        if not image_path or not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            
            # Stage 1: Real vs Not Real
            meta1_scaled = extract_and_scale_metadata(image, self.meta_features1, self.scaler1)
            meta1 = torch.tensor(meta1_scaled, dtype=torch.float32).to(self.device)
            img_tensor = transform(image).unsqueeze(0).to(self.device)
            
            analysis = {}
            
            with torch.no_grad():
                output1 = self.model1(img_tensor, meta1)
                probs1 = torch.softmax(output1, dim=1).cpu().numpy()[0]
                stage1_analysis = self._analyze_threshold_decision(probs1, self.threshold_list1)
                stage1_pred = self._predict_with_thresholds(probs1, self.threshold_list1)
                
                analysis['stage1'] = {
                    'analysis': stage1_analysis,
                    'prediction': stage1_pred,
                    'class_names': ['real', 'not_real']
                }
                
                if stage1_pred == 0:  # Real (human)
                    return "real", probs1[0], {"stage1": {"real": probs1[0], "not_real": probs1[1]}}, analysis
                else:  # Not Real - need Stage 2
                    # Stage 2: Edited vs AI
                    meta2_scaled = extract_and_scale_metadata(image, self.meta_features2, self.scaler2)
                    meta2 = torch.tensor(meta2_scaled, dtype=torch.float32).to(self.device)
                    
                    output2 = self.model2(img_tensor, meta2)
                    probs2 = torch.softmax(output2, dim=1).cpu().numpy()[0]
                    stage2_analysis = self._analyze_threshold_decision(probs2, self.threshold_list2)
                    stage2_pred = self._predict_with_thresholds(probs2, self.threshold_list2)
                    
                    analysis['stage2'] = {
                        'analysis': stage2_analysis,
                        'prediction': stage2_pred,
                        'class_names': ['edited', 'ai']
                    }
                    
                    if stage2_pred == 0:  # Edited
                        return "photoshopped", probs2[0], {
                            "stage1": {"real": probs1[0], "not_real": probs1[1]},
                            "stage2": {"edited": probs2[0], "ai": probs2[1]}
                        }, analysis
                    else:  # AI
                        return "ai_generated", probs2[1], {
                            "stage1": {"real": probs1[0], "not_real": probs1[1]},
                            "stage2": {"edited": probs2[0], "ai": probs2[1]}
                        }, analysis
                        
        except Exception as e:
            print(f"Error classifying image {image_path}: {e}")
            raise

# --- Standalone Functions for Backward Compatibility ---
def classify_image(image_path: str) -> str:
    """Standalone function to classify image"""
    agent = ImageAgent()
    return agent.classify_image(image_path)

def classify_image_with_confidence(image_path: str) -> Tuple[str, float, dict]:
    """Standalone function to classify image with confidence scores"""
    agent = ImageAgent()
    return agent.classify_image_with_confidence(image_path)

# --- Testing ---
if __name__ == "__main__":
    # Test the two-stage image classification agent
    print("=== Two-Stage Image Classification Agent Test ===")
    
    try:
        agent = ImageAgent()
        print("Agent initialized successfully!")
        
        # Test cases - replace with actual image paths
        test_cases = [
            "../human/000295da5dca4af09d5593174e15bb09.jpg",  # Should be "real"
            "../AI/0002f7db7beb4bf5879a0cdb7f17209d.jpg",     # Should be "ai_generated" 
            "../edited/Tp_D_CND_M_N_ani00018_sec00096_00138.tif" # Should be "photoshopped"
        ]
        
        for i, image_path in enumerate(test_cases, 1):
            print(f"\n--- Test Case {i} ---")
            print(f"Image Path: {image_path}")
            
            try:
                # Test basic classification
                classification = agent.classify_image(image_path)
                print(f"Classification: {classification}")
                
                # Test classification with detailed analysis
                classification, confidence, probs, analysis = agent.classify_image_with_analysis(image_path)
                print(f"Classification: {classification}")
                print(f"Confidence: {confidence:.3f}")
                print(f"Probabilities: {probs}")
                print(f"Threshold Analysis:")
                for stage, stage_data in analysis.items():
                    print(f"  {stage.upper()}:")
                    stage_analysis = stage_data['analysis']
                    print(f"    Decision Reason: {stage_analysis['decision_reason']}")
                    print(f"    Max Prob Class: {stage_data['class_names'][stage_analysis['max_prob_class']]}")
                    print(f"    Max Probability: {stage_analysis['max_prob']:.3f}")
                    print(f"    Probability Difference: {stage_analysis['prob_diff']:.3f}")
                    print(f"    Thresholds Exceeded: {stage_analysis['threshold_exceeded']}")
                
            except FileNotFoundError:
                print(f"File not found: {image_path}")
            except Exception as e:
                print(f"Error: {e}")
                
    except Exception as e:
        print(f"Error initializing agent: {e}")
        print("Make sure all model files are in the image_detection directory:")
        print("- model1_real_vs_not_real_fixed.pth")
        print("- model2_edited_vs_ai_fixed.pth")
        print("- metadata_scaler_stage1.pkl")
        print("- metadata_scaler_stage2.pkl")
        print("- best_thresholds_model1.json")
        print("- best_thresholds_model2.json")
        print("- train_meta_stage1_train.csv")
        print("- train_meta_stage2_train.csv")