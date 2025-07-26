import os
import piexif
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
from scipy.stats import entropy
import cv2
import tempfile
from torch.utils.data import Dataset
import joblib

valid_exts = (".jpg", ".jpeg", ".png", ".tif", ".tiff")

def extract_exif_info(image_path, img_format):
   if img_format not in ["JPEG", "TIFF"]:
       return {"Make": None, "Model": None, "Software": None, "DateTimeOriginal": None}
   try:
       exif_dict = piexif.load(image_path)
       exif_data = {}
       for ifd in exif_dict:
           for tag in exif_dict[ifd]:
               key = piexif.TAGS[ifd][tag]["name"]
               value = exif_dict[ifd][tag]
               if isinstance(value, bytes):
                   value = value.decode(errors="ignore")
               exif_data[key] = value
       return {
           "Make": exif_data.get("Make", None),
           "Model": exif_data.get("Model", None),
           "Software": exif_data.get("Software", None),
           "DateTimeOriginal": exif_data.get("DateTimeOriginal", None)
       }
   except:
       return {"Make": None, "Model": None, "Software": None, "DateTimeOriginal": None}

def extract_qtable(image_path, img_format):
   if img_format != "JPEG":
       return {"qtable_mean": None, "qtable_std": None, "qtable_max": None, "qtable_min": None}
   try:
       img = Image.open(image_path)
       qt = img.quantization
       if not qt:
           return {"qtable_mean": None, "qtable_std": None}
       q = list(qt.values())[0]
       q = np.array(q)
       return {
           "qtable_mean": np.mean(q),
           "qtable_std": np.std(q),
           "qtable_max": np.max(q),
           "qtable_min": np.min(q)
       }
   except:
       return {"qtable_mean": None, "qtable_std": None, "qtable_max": None, "qtable_min": None}

def extract_entropy(img_array):
   r, g, b = img_array[:, :, 0], img_array[:, :, 1], img_array[:, :, 2]
   return {
       "r_entropy": entropy(np.histogram(r, bins=256)[0] + 1e-7),
       "g_entropy": entropy(np.histogram(g, bins=256)[0] + 1e-7),
       "b_entropy": entropy(np.histogram(b, bins=256)[0] + 1e-7)
   }

def extract_blur_sharpness(img_array):
   gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
   lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
   return {"blur_metric": lap_var}

def extract_hu_moments(img_array):
   gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
   moments = cv2.moments(gray)
   hu_moments = cv2.HuMoments(moments).flatten()
   return {f"hu_{i+1}": float(val) for i, val in enumerate(hu_moments)}

def extract_dct_features(img_array):
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape
    # Crop to even dimensions if necessary
    if h % 2 != 0:
        gray = gray[:-1, :]
    if w % 2 != 0:
        gray = gray[:, :-1]
    dct = cv2.dct(np.float32(gray) / 255.0)
    dct_abs = np.abs(dct)
    return {
        "dct_mean": np.mean(dct_abs),
        "dct_std": np.std(dct_abs),
        "dct_max": np.max(dct_abs),
        "dct_energy": np.sum(dct_abs ** 2)
    }

def extract_noise_features(img_array):
   gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
   blurred = cv2.GaussianBlur(gray, (3, 3), 0)
   noise = gray.astype(np.float32) - blurred.astype(np.float32)
   return {
       "noise_mean": np.mean(noise),
       "noise_std": np.std(noise)
   }

def extract_edge_density(img_array):
   gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
   edges = cv2.Canny(gray, 100, 200)
   return {
       "edge_density": np.sum(edges > 0) / edges.size
   }

def extract_image_stats(img):
   return {
       "width": img.width,
       "height": img.height
   }

def process_images(folder, label):
   data = []
   for fname in tqdm(os.listdir(folder), desc=f"Processing {folder}"):
       if not fname.lower().endswith(valid_exts):
           continue
       full_path = os.path.join(folder, fname)
       try:
           img = Image.open(full_path).convert("RGB")
           img_array = np.array(img)
           img_format = img.format or os.path.splitext(fname)[-1].replace(".", "").upper()

           row = {
               "filename": fname,
               "label": label,
               "format": img_format,
               "has_exif": img_format in ['JPEG', 'TIFF'],
               "has_qtable": img_format == 'JPEG'
           }

           row.update(extract_exif_info(full_path, img_format))
           row.update(extract_image_stats(img))
           row.update(extract_entropy(img_array))
           row.update(extract_blur_sharpness(img_array))
           row.update(extract_hu_moments(img_array))
           row.update(extract_dct_features(img_array))
           row.update(extract_noise_features(img_array))
           row.update(extract_edge_density(img_array))
           row.update(extract_qtable(full_path, img_format))

           data.append(row)
       except Exception as e:
           print(f"[ERROR] {fname}: {e}")
   return data


class HybridImageDataset(Dataset):
    def __init__(self, img_root, metadata_csv, transform=None, meta_features=None, label_col='label'):
        self.img_root = img_root
        self.df = pd.read_csv(metadata_csv)
        self.transform = transform
        self.meta_features = meta_features or [col for col in self.df.columns if col not in ['filename', 'label', 'format']]
        self.label_col = label_col
        # Fill missing values in metadata
        self.df[self.meta_features] = self.df[self.meta_features].fillna(self.df[self.meta_features].mean())

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        # Use folder column if it exists, otherwise use original label
        if 'folder' in row:
            folder = row['folder']
        else:
            orig_label = row['label']  # Always 0, 1, or 2
            if orig_label == 0:
                folder = 'human'
            elif orig_label == 1:
                folder = 'edited'
            else:
                folder = 'AI'
        img_path = os.path.join(self.img_root, folder, row['filename'])
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        meta = torch.tensor(row[self.meta_features].values.astype(np.float32))
        label = torch.tensor(row[self.label_col]).long()
        return image, meta, label
    

import torchvision.models as models

class HybridNet(nn.Module):
    def __init__(self, num_metadata_features, num_classes=3):
        super().__init__()
        # Image branch: EfficientNet-B0
        self.cnn = models.efficientnet_b0(pretrained=True)
        self.cnn.classifier = nn.Identity()  # Remove final classification layer
        cnn_out_dim = 1280  # EfficientNet-B0 output features

        # Metadata branch
        self.meta_mlp = nn.Sequential(
            nn.Linear(num_metadata_features, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )

        # Fusion
        self.classifier = nn.Sequential(
            nn.Linear(cnn_out_dim + 32, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, image, metadata):
        img_feat = self.cnn(image)
        meta_feat = self.meta_mlp(metadata)
        x = torch.cat([img_feat, meta_feat], dim=1)
        return self.classifier(x)
    

from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize(256),                # Resize the shorter side to 256
    transforms.CenterCrop(224),            # Then crop the center 224x224
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def extract_metadata_from_pil_image(image, meta_features):
    """
    Extracts metadata features from a PIL image for agent inference.
    Returns a dict with keys matching meta_features.
    """
    img_array = np.array(image.convert("RGB"))
    # Guess format (PIL loses format on upload, so default to JPEG)
    img_format = image.format or "JPEG"
    # Save to temp file if you need to use piexif or quantization
    with tempfile.NamedTemporaryFile(suffix=".jpg") as tmp:
        image.save(tmp.name)
        row = {}
        row.update(extract_exif_info(tmp.name, img_format))
        row.update(extract_image_stats(image))
        row.update(extract_entropy(img_array))
        row.update(extract_blur_sharpness(img_array))
        row.update(extract_hu_moments(img_array))
        row.update(extract_dct_features(img_array))
        row.update(extract_noise_features(img_array))
        row.update(extract_edge_density(img_array))
        row.update(extract_qtable(tmp.name, img_format))
    # Fill missing features with 0 and handle None values
    result = {}
    for k in meta_features:
        value = row.get(k, 0)
        if value is None:
            result[k] = 0
        else:
            result[k] = value
    return result

def load_scalers(stage1_scaler_path='image_detection/metadata_scaler_stage1.pkl', 
                 stage2_scaler_path='image_detection/metadata_scaler_stage2.pkl'):
    """
    Load the trained scalers for both stages.
    Returns tuple of (scaler1, scaler2)
    """
    try:
        scaler1 = joblib.load(stage1_scaler_path)
        scaler2 = joblib.load(stage2_scaler_path)
        return scaler1, scaler2
    except Exception as e:
        print(f"Error loading scalers: {e}")
        return None, None

def extract_and_scale_metadata(image, meta_features, scaler):
    """
    Extract metadata from PIL image and apply scaling.
    Returns scaled metadata as numpy array.
    """
    # Extract raw metadata
    meta_dict = extract_metadata_from_pil_image(image, meta_features)
    
    # Convert to numpy array with proper handling of None values
    meta_values = []
    for f in meta_features:
        value = meta_dict[f]
        if value is None:
            meta_values.append(0.0)
        else:
            meta_values.append(float(value))
    
    meta_array = np.array(meta_values, dtype=np.float64).reshape(1, -1)
    
    # Apply scaling with NaN handling
    if scaler is not None:
        # Check if scaler has NaN values and handle them
        if np.any(np.isnan(scaler.mean_)) or np.any(np.isnan(scaler.scale_)):
            # Create a safe version of the scaler
            safe_mean = np.where(np.isnan(scaler.mean_), 0.0, scaler.mean_)
            safe_scale = np.where(np.isnan(scaler.scale_), 1.0, scaler.scale_)
            
            # Apply manual scaling
            meta_scaled = (meta_array - safe_mean) / safe_scale
        else:
            meta_scaled = scaler.transform(meta_array)
    else:
        meta_scaled = meta_array
    
    return meta_scaled